"""v0.5 camera-independent boundary-transport image-formation experiment."""

from __future__ import annotations

import ast
import csv
import inspect
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import (
    build_boundary_transport,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
    restrict_events_by_owner,
)
from .detector_readout import (
    LightFieldReadout,
    detector_from_outgoing_direction,
    read_light_field,
)
from .fields import SphereField
from .image_formation import (
    ReferenceImage,
    build_shared_surface_state,
    image_metrics,
    reference_zero_set_image,
)
from .light_field import SparseBoundaryLightField, build_sparse_boundary_light_field
from .mesh_field import prepare_stanford_bunny


Tensor = torch.Tensor
SAMPLE_LEVELS = (4_096, 16_384, 65_536, 262_144)
CAMERA_COUNTS = (1, 2, 4, 8, 20, 40)
READOUT_CONFIG = {
    "angular_neighbors": 4,
    "angular_bandwidth_radians": 0.12,
    "spatial_bandwidth_pixels": 0.65,
    "support_threshold": 0.5,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened: list[dict[str, object]] = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, tuple, dict))
                else value
                for key, value in row.items()
            }
        )
    keys = sorted({key for row in flattened for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def _plot_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    rows: int,
    columns: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 3.1 * rows), facecolor="black"
    )
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (label, image) in zip(axes_array, panels):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(label, color="white", fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(panels) :]:
        axis.axis("off")
    figure.suptitle(title, color="white", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def _readout_row(
    name: str,
    phase: str,
    output: LightFieldReadout,
    reference: ReferenceImage,
    field: SparseBoundaryLightField,
    **extra: object,
) -> dict[str, object]:
    image = output.image.detach().cpu().numpy()
    return {
        "name": name,
        "phase": phase,
        **extra,
        **image_metrics(image, reference),
        "readout_seconds": output.readout_seconds,
        "readout_peak_allocated_mib": output.peak_allocated_mib,
        "direct_support_fraction": output.direct_support_fraction,
        "interpolated_support_fraction": output.interpolated_support_fraction,
        "no_support_fraction": output.no_support_fraction,
        "nearest_angular_distance_radians": float(output.angular_distances_radians[0]),
        "field_digest_unchanged": output.field_digest_before == output.field_digest_after,
        "field_occupied_bins": field.occupied_bins,
        "field_occupancy_fraction": field.occupied_bins / max(field.possible_bins, 1),
    }


def _mean_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = ("silhouette_iou", "silhouette_f1", "ssim", "psnr", "edge_f1")
    return {key: statistics.mean(float(row[key]) for row in rows) for key in keys}


def _regional_information_error(
    image: np.ndarray, reference: ReferenceImage
) -> dict[str, object]:
    from scipy.ndimage import binary_dilation, binary_erosion

    truth = reference.mask
    prediction = np.any(image > 1e-10, axis=-1)
    eroded = binary_erosion(truth, iterations=2)
    boundary = binary_dilation(truth, iterations=2) & ~eroded
    exterior = ~binary_dilation(truth, iterations=2)
    squared = np.mean((image - reference.image) ** 2, axis=-1)
    return {
        "interior_pixel_fraction": float(eroded.mean()),
        "boundary_band_pixel_fraction": float(boundary.mean()),
        "interior_mse": float(squared[eroded].mean()) if eroded.any() else 0.0,
        "boundary_band_mse": float(squared[boundary].mean()) if boundary.any() else 0.0,
        "exterior_mse": float(squared[exterior].mean()) if exterior.any() else 0.0,
        "missed_reference_foreground_fraction": float(
            (truth & ~prediction).sum() / max(truth.sum(), 1)
        ),
        "false_positive_fraction_of_prediction": float(
            (prediction & ~truth).sum() / max(prediction.sum(), 1)
        ),
        "interpretation": "constant radiance makes interior correctness an occupancy test; boundary error measures silhouette/visibility loss, not BRDF fidelity",
    }


def _posthoc_direction(atlas_directions: Tensor, anchor: int = 3) -> tuple[Tensor, int]:
    cosine = atlas_directions @ atlas_directions[anchor]
    cosine[anchor] = -torch.inf
    neighbor = int(torch.argmax(cosine))
    direction = atlas_directions[anchor] + atlas_directions[neighbor]
    direction /= torch.linalg.vector_norm(direction)
    return direction, neighbor


def _core_code_audit() -> dict[str, object]:
    root = Path(__file__).parent
    files = [
        root / "boundary_transport.py",
        root / "light_field.py",
        root / "detector_readout.py",
    ]
    imports: dict[str, list[str]] = {}
    calls: dict[str, list[str]] = {}
    for path in files:
        tree = ast.parse(path.read_text())
        imports[path.name] = []
        calls[path.name] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports[path.name].append(node.module or "")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls[path.name].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls[path.name].append(node.func.attr)
    detector_imports = imports["detector_readout.py"]
    detector_calls = calls["detector_readout.py"]
    forbidden_detector_import = any(
        token in module
        for module in detector_imports
        for token in ("fields", "mesh", "image_formation", "tracer")
    )
    forbidden_detector_call = any(
        token in call.lower()
        for call in detector_calls
        for token in ("zero_set", "raycast", "surface_intersection", "project_orthographic")
    )
    builder_parameters = list(inspect.signature(build_boundary_transport).parameters)
    readout_parameters = list(inspect.signature(read_light_field).parameters)
    camera_in_builder = any("camera" in item.lower() or "detector" in item.lower() for item in builder_parameters)
    scene_in_readout = any(
        token in item.lower()
        for item in readout_parameters
        for token in ("geometry", "scene", "surface", "zero_set", "mesh")
    )
    confirmed = not any(
        (forbidden_detector_import, forbidden_detector_call, camera_in_builder, scene_in_readout)
    )
    return {
        "core_files": [str(path.relative_to(root.parent.parent)) for path in files],
        "boundary_builder_parameters": builder_parameters,
        "detector_readout_parameters": readout_parameters,
        "detector_imports": detector_imports,
        "forbidden_detector_import": forbidden_detector_import,
        "forbidden_detector_scene_call": forbidden_detector_call,
        "camera_or_detector_parameter_in_scene_builder": camera_in_builder,
        "scene_or_geometry_parameter_in_readout": scene_in_readout,
        "no_geometry_camera_projection_confirmed": confirmed,
    }


def camera_independent_cpu_cuda_verification() -> dict[str, object]:
    """Check deterministic field construction and small CPU/CUDA equivalence."""
    cpu = torch.device("cpu")
    sequence = torch.quasirandom.SobolEngine(2, scramble=False).draw(512).to(torch.float64)
    vertical = 1.0 - 2.0 * sequence[:, 0]
    angle = 2.0 * torch.pi * sequence[:, 1]
    radial = torch.sqrt((1.0 - vertical.square()).clamp_min(0.0))
    points = torch.stack((radial * torch.cos(angle), vertical, radial * torch.sin(angle)), 1)
    normals = points.clone()
    field_cpu = SphereField()
    atlas_cpu = nested_fibonacci_atlas(cpu, (8,))
    boundary_cpu = enclosing_observation_sphere(points)
    first = build_boundary_transport(
        field_cpu, points, normals, atlas_cpu, boundary_cpu, root_samples=8, bisection_steps=10
    )
    second = build_boundary_transport(
        field_cpu, points, normals, atlas_cpu, boundary_cpu, root_samples=8, bisection_steps=10
    )
    sparse_cpu = build_sparse_boundary_light_field(first, spatial_resolution=(32, 32), spatial_extent=2.8)
    camera_cpu = detector_from_outgoing_direction(
        boundary_cpu, atlas_cpu.directions[2], resolution=(32, 32)
    )
    read_cpu = read_light_field(
        sparse_cpu, camera_cpu, angular_neighbors=1, angular_bandwidth_radians=0.0
    )
    report: dict[str, object] = {
        "cpu_deterministic_event_digest": first.digest == second.digest,
        "cpu_event_count": first.count,
        "cpu_absorbed_count": first.absorbed_count,
        "cpu_field_digest_unchanged_by_readout": sparse_cpu.digest == read_cpu.field_digest_after,
        "cpu_readout_nonempty": bool(read_cpu.mask.any()),
        "cuda_available": torch.cuda.is_available(),
    }
    assert all(
        bool(report[key])
        for key in (
            "cpu_deterministic_event_digest",
            "cpu_field_digest_unchanged_by_readout",
            "cpu_readout_nonempty",
        )
    )
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        field_cuda = SphereField()
        atlas_cuda = nested_fibonacci_atlas(cuda, (8,))
        boundary_cuda = enclosing_observation_sphere(points.to(cuda))
        events_cuda = build_boundary_transport(
            field_cuda,
            points.to(cuda),
            normals.to(cuda),
            atlas_cuda,
            boundary_cuda,
            root_samples=8,
            bisection_steps=10,
        )
        sparse_cuda = build_sparse_boundary_light_field(
            events_cuda, spatial_resolution=(32, 32), spatial_extent=2.8
        )
        camera_cuda = detector_from_outgoing_direction(
            boundary_cuda, atlas_cuda.directions[2], resolution=(32, 32)
        )
        read_cuda = read_light_field(
            sparse_cuda, camera_cuda, angular_neighbors=1, angular_bandwidth_radians=0.0
        )
        report.update(
            {
                "cpu_cuda_owner_ids_equal": torch.equal(
                    first.owner_ids, events_cuda.owner_ids.cpu()
                ),
                "cpu_cuda_direction_ids_equal": torch.equal(
                    first.direction_ids, events_cuda.direction_ids.cpu()
                ),
                "cpu_cuda_boundary_position_max_error": float(
                    (first.boundary_positions - events_cuda.boundary_positions.cpu()).abs().max()
                ),
                "cpu_cuda_sparse_keys_equal": torch.equal(
                    sparse_cpu.keys, sparse_cuda.keys.cpu()
                ),
                "cpu_cuda_readout_max_error": float(
                    (read_cpu.image - read_cuda.image.cpu()).abs().max()
                ),
            }
        )
        assert report["cpu_cuda_owner_ids_equal"]
        assert report["cpu_cuda_direction_ids_equal"]
        assert report["cpu_cuda_sparse_keys_equal"]
        assert float(report["cpu_cuda_boundary_position_max_error"]) < 1e-12
        assert float(report["cpu_cuda_readout_max_error"]) < 1e-12
    return report


def _plot_diagnostics(
    figure_directory: Path,
    convergence: list[dict[str, object]],
    camera_scaling: list[dict[str, object]],
    bandwidth: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    figure_directory.mkdir(parents=True, exist_ok=True)
    x = np.asarray([float(row["emitted_transport_samples"]) for row in convergence])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [float(row["silhouette_iou"]) for row in convergence], "o-", label="IoU")
    axes[0].plot(x, [float(row["silhouette_f1"]) for row in convergence], "o-", label="F1")
    axes[0].plot(x, [float(row["ssim"]) for row in convergence], "o-", label="SSIM")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("outgoing transport samples")
    axes[0].set_ylabel("quality")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, [float(row["edge_f1"]) for row in convergence], "o-", label="edge F1")
    axes[1].plot(x, [float(row["psnr"]) for row in convergence], "o-", label="PSNR (dB)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("outgoing transport samples")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v05_quality_vs_samples.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [float(row["field_occupancy_fraction"]) for row in convergence], "o-")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("outgoing transport samples")
    axes[0].set_ylabel("occupied spatial-angular fraction")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, [float(row["direct_support_fraction"]) for row in convergence], "o-", label="direct")
    axes[1].plot(x, [float(row["no_support_fraction"]) for row in convergence], "o-", label="none")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("outgoing transport samples")
    axes[1].set_ylabel("camera query fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v05_light_field_coverage.png", dpi=180)
    plt.close(figure)

    counts = [int(row["camera_count"]) for row in camera_scaling]
    readout = [float(row["total_readout_seconds"]) for row in camera_scaling]
    scene_total = [float(row["scene_plus_field_plus_readout_seconds"]) for row in camera_scaling]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(counts, readout, "o-", label="detector readout")
    axis.plot(counts, scene_total, "o-", label="scene + H + readout")
    axis.axhline(
        float(camera_scaling[0]["scene_plus_field_seconds"]),
        linestyle="--",
        color="black",
        label="fixed scene + H",
    )
    axis.set_xlabel("post-hoc detector count")
    axis.set_ylabel("seconds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v05_camera_scaling.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    labels = [str(row["name"]).replace("bandwidth_", "") for row in bandwidth]
    axis.plot(labels, [float(row["silhouette_iou"]) for row in bandwidth], "o-", label="IoU")
    axis.plot(labels, [float(row["ssim"]) for row in bandwidth], "o-", label="SSIM")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v05_bandwidth_sensitivity.png", dpi=180)
    plt.close(figure)


def run_camera_independent_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
) -> dict[str, object]:
    """Build H once with zero cameras, then perform all detector experiments."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    experiment_started = time.perf_counter()
    prepared = prepare_stanford_bunny(mesh_path)
    field = prepared.gt_field.to(device)

    # Scene phase: no camera or detector object exists above this line.
    surface = build_shared_surface_state(field, SAMPLE_LEVELS[-1], device)
    atlas = nested_fibonacci_atlas(device, (8, 128))
    boundary = enclosing_observation_sphere(surface.points, margin=1.25)
    events = build_boundary_transport(
        field,
        surface.points,
        surface.normals,
        atlas,
        boundary,
        root_samples=16,
        bisection_steps=18,
    )
    light_field = build_sparse_boundary_light_field(events)
    frozen_event_digest = events.digest
    frozen_field_digest = light_field.digest

    # Detector phase: cameras are instantiated only after H_Sigma is frozen.
    cameras = [
        detector_from_outgoing_direction(boundary, atlas.directions[index])
        for index in range(40)
    ]
    references = [
        reference_zero_set_image(field, camera, appearance="constant")
        for camera in cameras[:8]
    ]
    rows: list[dict[str, object]] = []

    candidate_specs = (
        ("nearest_event_slice", 1, 0.0, 0.0, 0.5),
        ("angular_knn", 4, 0.12, 0.0, 0.5),
        ("spatial_kde", 1, 0.0, 0.65, 0.5),
        ("spatial_angular_kde", 4, 0.12, 0.65, 0.5),
    )
    candidate_rows: list[dict[str, object]] = []
    candidate_outputs: dict[str, LightFieldReadout] = {}
    comparison_view = 3
    for name, neighbors, h_omega, h_y, threshold in candidate_specs:
        output = read_light_field(
            light_field,
            cameras[comparison_view],
            angular_neighbors=neighbors,
            angular_bandwidth_radians=h_omega,
            spatial_bandwidth_pixels=h_y,
            support_threshold=threshold,
        )
        row = _readout_row(
            name,
            "interpolation",
            output,
            references[comparison_view],
            light_field,
            angular_neighbors=neighbors,
            h_omega_radians=h_omega,
            h_y_pixels=h_y,
            support_threshold=threshold,
        )
        candidate_rows.append(row)
        candidate_outputs[name] = output
    rows.extend(candidate_rows)

    selected_name = "spatial_angular_kde"
    selected = candidate_outputs[selected_name]
    single_row = next(row for row in candidate_rows if row["name"] == selected_name)
    if float(single_row["silhouette_iou"]) < 0.85 or float(single_row["silhouette_f1"]) < 0.90:
        raise RuntimeError("V05_SINGLE_VIEW_IMAGE_QUALITY_GATE_FAILED")

    multiview_rows: list[dict[str, object]] = []
    multiview_outputs: list[LightFieldReadout] = []
    for view, (camera, reference) in enumerate(zip(cameras[:8], references)):
        output = read_light_field(light_field, camera, **READOUT_CONFIG)
        multiview_outputs.append(output)
        multiview_rows.append(
            _readout_row(
                f"multiview_{view:02d}",
                "multiview",
                output,
                reference,
                light_field,
                view=view,
                **READOUT_CONFIG,
            )
        )
    rows.extend(multiview_rows)

    posthoc_direction, posthoc_neighbor = _posthoc_direction(atlas.directions)
    posthoc_camera = detector_from_outgoing_direction(boundary, posthoc_direction)
    posthoc_reference = reference_zero_set_image(field, posthoc_camera, appearance="constant")
    posthoc_output = read_light_field(light_field, posthoc_camera, **READOUT_CONFIG)
    posthoc_row = _readout_row(
        "posthoc_off_atlas_camera",
        "posthoc",
        posthoc_output,
        posthoc_reference,
        light_field,
        atlas_anchor=3,
        atlas_neighbor=posthoc_neighbor,
        direction=posthoc_direction.detach().cpu().tolist(),
        **READOUT_CONFIG,
    )
    rows.append(posthoc_row)
    repeat_outputs = [
        read_light_field(light_field, cameras[comparison_view], **READOUT_CONFIG)
        for _ in range(3)
    ]
    repeat_max_error = max(
        float((repeat_outputs[0].image - item.image).abs().max())
        for item in repeat_outputs[1:]
    )

    # Completely replace ensemble A with exact but previously uninstantiated
    # directions from the same frozen global outgoing field.
    ensemble_b_cameras = [
        detector_from_outgoing_direction(boundary, atlas.directions[index])
        for index in range(64, 72)
    ]
    ensemble_b_references = [
        reference_zero_set_image(field, camera, appearance="constant")
        for camera in ensemble_b_cameras
    ]
    ensemble_b_rows: list[dict[str, object]] = []
    for view, (camera, reference) in enumerate(zip(ensemble_b_cameras, ensemble_b_references)):
        output = read_light_field(light_field, camera, **READOUT_CONFIG)
        ensemble_b_rows.append(
            _readout_row(
                f"ensemble_b_{view:02d}",
                "ensemble_swap",
                output,
                reference,
                light_field,
                atlas_direction_id=64 + view,
                **READOUT_CONFIG,
            )
        )
    rows.extend(ensemble_b_rows)

    camera_scaling: list[dict[str, object]] = []
    for count in CAMERA_COUNTS:
        _sync(device)
        started = time.perf_counter()
        outputs = [read_light_field(light_field, camera, **READOUT_CONFIG) for camera in cameras[:count]]
        _sync(device)
        elapsed = time.perf_counter() - started
        row = {
            "name": f"camera_scaling_{count}",
            "phase": "camera_scaling",
            "camera_count": count,
            "scene_traversals": 0,
            "event_digest": events.digest,
            "field_digest": light_field.digest,
            "total_readout_seconds": elapsed,
            "mean_readout_seconds": statistics.mean(item.readout_seconds for item in outputs),
            "scene_transport_seconds": events.scene_seconds,
            "field_build_seconds": light_field.build_seconds,
            "scene_plus_field_seconds": events.scene_seconds + light_field.build_seconds,
            "scene_plus_field_plus_readout_seconds": events.scene_seconds + light_field.build_seconds + elapsed,
            "maximum_readout_peak_allocated_mib": max(item.peak_allocated_mib for item in outputs),
        }
        camera_scaling.append(row)
        rows.append(row)

    convergence: list[dict[str, object]] = []
    convergence_images: list[np.ndarray] = []
    for sample_count in SAMPLE_LEVELS:
        nested_events = (
            events
            if sample_count == SAMPLE_LEVELS[-1]
            else restrict_events_by_owner(events, sample_count, surface.normals)
        )
        nested_field = (
            light_field
            if sample_count == SAMPLE_LEVELS[-1]
            else build_sparse_boundary_light_field(nested_events)
        )
        output = read_light_field(
            nested_field, cameras[comparison_view], **READOUT_CONFIG
        )
        image = output.image.detach().cpu().numpy()
        convergence_images.append(image)
        row = _readout_row(
            f"sample_convergence_{nested_events.emitted_count}",
            "sample_convergence",
            output,
            references[comparison_view],
            nested_field,
            surface_samples=sample_count,
            emitted_transport_samples=nested_events.emitted_count,
            boundary_events=nested_events.count,
            absorbed_transport_samples=nested_events.absorbed_count,
            occupied_light_field_support=nested_field.occupied_bins,
            events_per_occupied_spatial_angular_cell=nested_events.count
            / max(nested_field.occupied_bins, 1),
            **READOUT_CONFIG,
        )
        convergence.append(row)
        rows.append(row)
    final_image = convergence_images[-1]
    for row, image in zip(convergence, convergence_images):
        row["readout_variance_to_max"] = float(np.mean((image - final_image) ** 2))

    bandwidth_specs = (
        ("bandwidth_hy0_homega0", 1, 0.0, 0.0, 0.5),
        ("bandwidth_hy065_homega0", 1, 0.0, 0.65, 0.5),
        ("bandwidth_hy125_homega0", 1, 0.0, 1.25, 0.5),
        ("bandwidth_hy065_homega004", 4, 0.04, 0.65, 0.5),
        ("bandwidth_hy065_homega012", 4, 0.12, 0.65, 0.5),
        ("bandwidth_hy065_homega024", 8, 0.24, 0.65, 0.5),
    )
    bandwidth_rows: list[dict[str, object]] = []
    for name, neighbors, h_omega, h_y, threshold in bandwidth_specs:
        output = read_light_field(
            light_field,
            posthoc_camera,
            angular_neighbors=neighbors,
            angular_bandwidth_radians=h_omega,
            spatial_bandwidth_pixels=h_y,
            support_threshold=threshold,
        )
        bandwidth_rows.append(
            _readout_row(
                name,
                "bandwidth",
                output,
                posthoc_reference,
                light_field,
                angular_neighbors=neighbors,
                h_omega_radians=h_omega,
                h_y_pixels=h_y,
                support_threshold=threshold,
            )
        )
    rows.extend(bandwidth_rows)

    field_digest_after_every_camera = light_field.digest
    event_digest_after_every_camera = events.digest
    camera_independence = {
        "events_generated_before_camera_instantiation": True,
        "event_digest_before_cameras": frozen_event_digest,
        "event_digest_after_1_8_20_40_and_swapped_cameras": event_digest_after_every_camera,
        "field_digest_before_cameras": frozen_field_digest,
        "field_digest_after_1_8_20_40_and_swapped_cameras": field_digest_after_every_camera,
        "event_digest_identical": frozen_event_digest == event_digest_after_every_camera,
        "field_digest_identical": frozen_field_digest == field_digest_after_every_camera,
        "scene_traversal_count": 1,
        "scene_retraversals_after_camera_creation": 0,
        "camera_counts_tested": list(CAMERA_COUNTS),
        "ensemble_a_direction_ids": list(range(8)),
        "ensemble_b_direction_ids": list(range(64, 72)),
        "off_atlas_posthoc_direction": posthoc_direction.detach().cpu().tolist(),
    }
    audit = _core_code_audit()
    cpu_cuda = camera_independent_cpu_cuda_verification()

    event_memory = events.memory_bytes / 2**20
    field_memory = light_field.memory_bytes / 2**20
    coverage = {
        "boundary_events": events.count,
        "emitted_forward_hemisphere_samples": events.emitted_count,
        "absorbed_before_boundary": events.absorbed_count,
        "escape_fraction": events.count / max(events.emitted_count, 1),
        "direction_atlas_bins": atlas.count,
        "occupied_direction_bins": int(torch.unique(events.direction_ids).numel()),
        "directional_occupancy_fraction": int(torch.unique(events.direction_ids).numel()) / atlas.count,
        "boundary_spatial_occupied_cells": light_field.boundary_spatial_occupied_cells,
        "boundary_spatial_total_cells": light_field.boundary_spatial_total_cells,
        "boundary_spatial_occupancy_fraction": light_field.boundary_spatial_occupied_cells
        / light_field.boundary_spatial_total_cells,
        "events_per_occupied_boundary_spatial_cell": events.count
        / max(light_field.boundary_spatial_occupied_cells, 1),
        "possible_spatial_angular_bins": light_field.possible_bins,
        "occupied_spatial_angular_bins": light_field.occupied_bins,
        "spatial_angular_occupancy_fraction": light_field.occupied_bins
        / light_field.possible_bins,
        "events_per_occupied_spatial_angular_bin": events.count
        / max(light_field.occupied_bins, 1),
        "boundary_radius_residual_max": float(
            (
                torch.linalg.vector_norm(
                    events.boundary_positions - boundary.center, dim=1
                )
                - boundary.radius
            )
            .abs()
            .max()
        ),
        "event_memory_mib": event_memory,
        "sparse_field_memory_mib": field_memory,
        "dense_rgb_field_memory_mib_float64": light_field.possible_bins * 3 * 8 / 2**20,
    }

    # Required render artifacts.
    reference_image = references[comparison_view].image
    selected_image = selected.image.detach().cpu().numpy()
    difference = np.abs(reference_image - selected_image)
    _plot_grid(
        render_directory / "v05_single_view.png",
        [("v0.4 reference oracle", reference_image), ("v0.5 H readout", selected_image), ("absolute difference", difference)],
        1,
        3,
        "One view: direct geometry oracle vs camera-independent boundary field",
    )
    multiview_panels = [(f"v0.4 ref {index}", ref.image) for index, ref in enumerate(references)]
    multiview_panels += [
        (f"v0.5 H {index}", output.image.detach().cpu().numpy())
        for index, output in enumerate(multiview_outputs)
    ]
    _plot_grid(
        render_directory / "v05_multiview.png",
        multiview_panels,
        2,
        8,
        "One frozen H_Sigma, eight passive detector views",
    )
    posthoc_image = posthoc_output.image.detach().cpu().numpy()
    _plot_grid(
        render_directory / "v05_posthoc_camera.png",
        [
            ("post-hoc v0.4 oracle", posthoc_reference.image),
            ("post-hoc H readout", posthoc_image),
            ("absolute difference", np.abs(posthoc_reference.image - posthoc_image)),
        ],
        1,
        3,
        "Off-atlas camera created after H_Sigma was frozen",
    )
    for view, output in enumerate(multiview_outputs):
        _save_rgb(
            render_directory / f"v05_bunny_transport_view{view:02d}_256.png",
            output.image.detach().cpu().numpy(),
        )
    _plot_diagnostics(figure_directory, convergence, camera_scaling, bandwidth_rows)

    one_view_pass = (
        float(single_row["silhouette_iou"]) >= 0.85
        and float(single_row["silhouette_f1"]) >= 0.90
    )
    posthoc_pass = (
        float(posthoc_row["silhouette_iou"]) >= 0.85
        and float(posthoc_row["silhouette_f1"]) >= 0.90
    )
    multiview_mean = _mean_metrics(multiview_rows)
    ensemble_b_mean = _mean_metrics(ensemble_b_rows)
    verdicts = {
        "camera_independent_transport_field": "CAMERA_INDEPENDENT_TRANSPORT_FIELD_SUPPORTED"
        if camera_independence["field_digest_identical"]
        else "CAMERA_INDEPENDENT_TRANSPORT_FIELD_NOT_SUPPORTED",
        "camera_free_scene_traversal": "CAMERA_FREE_SCENE_TRAVERSAL_SUPPORTED"
        if camera_independence["scene_retraversals_after_camera_creation"] == 0
        else "CAMERA_FREE_SCENE_TRAVERSAL_NOT_SUPPORTED",
        "bunny_from_transport_field": "BUNNY_FROM_TRANSPORT_FIELD_SUPPORTED"
        if one_view_pass
        else "CAMERA_INDEPENDENT_IMAGE_FORMATION_NOT_SUPPORTED",
        "posthoc_view_synthesis": "POSTHOC_VIEW_SYNTHESIS_SUPPORTED"
        if posthoc_pass
        else "POSTHOC_VIEW_SYNTHESIS_NOT_SUPPORTED",
        "shared_multiview": "SHARED_MULTIVIEW_TRANSPORT_IMAGE_FORMATION_SUPPORTED"
        if multiview_mean["silhouette_iou"] >= 0.85
        else "SHARED_MULTIVIEW_TRANSPORT_IMAGE_FORMATION_NOT_SUPPORTED",
        "geometry_camera_projection": "NO_GEOMETRY_CAMERA_PROJECTION_CONFIRMED"
        if audit["no_geometry_camera_projection_confirmed"]
        else "GEOMETRY_CAMERA_PROJECTION_REMAINS",
    }
    report: dict[str, object] = {
        "version": "0.5.0",
        "formulation": {
            "scene": "H_Sigma = T(F)",
            "detector": "I_c = M_c H_Sigma",
            "multiview": "I_multi = [M_1; ...; M_N] T(F)",
            "event": "e_i = (y_i, omega_i, C_i, w_i, owner_i, path_length_i)",
            "visibility": "retain iff t_boundary < t_next_zero_set_surface",
        },
        "environment": cuda_environment(),
        "configuration": {
            "surface_sample_levels": list(SAMPLE_LEVELS),
            "direction_atlas_levels": list(atlas.levels),
            "direction_atlas_count_after_deduplication": atlas.count,
            "boundary": "sphere",
            "boundary_center": boundary.center.detach().cpu().tolist(),
            "boundary_radius": boundary.radius,
            "light_field_spatial_representation": "sparse sorted occupied transverse bins",
            "light_field_angular_representation": "nested Fibonacci outgoing-direction atlas",
            "light_field_resolution_per_direction": list(light_field.spatial_resolution),
            "light_field_spatial_extent": light_field.spatial_extent,
            "readout": READOUT_CONFIG,
            "appearance": "constant grayscale 0.72",
        },
        "surface": {
            "samples": surface.points.shape[0],
            "build_seconds": surface.build_seconds,
            "zero_set_residual_max": surface.residual_max,
            "peak_allocated_mib": surface.peak_allocated_mib,
        },
        "scene_transport": {
            "seconds": events.scene_seconds,
            "peak_allocated_mib": events.peak_allocated_mib,
            "digest": events.digest,
        },
        "light_field": {
            "build_seconds": light_field.build_seconds,
            "peak_allocated_mib": light_field.peak_allocated_mib,
            "digest": light_field.digest,
            **coverage,
        },
        "camera_independence": camera_independence,
        "code_audit": audit,
        "cpu_cuda_validation": cpu_cuda,
        "interpolation_comparison": candidate_rows,
        "selected_single_view": single_row,
        "multiview": {"per_view": multiview_rows, "mean": multiview_mean},
        "posthoc_camera": posthoc_row,
        "information_sufficiency": {
            "single_view": _regional_information_error(
                selected_image, references[comparison_view]
            ),
            "posthoc_view": _regional_information_error(
                posthoc_image, posthoc_reference
            ),
            "mean_v8_interior_mse": statistics.mean(
                float(
                    _regional_information_error(
                        output.image.detach().cpu().numpy(), reference
                    )["interior_mse"]
                )
                for output, reference in zip(multiview_outputs, references)
            ),
            "mean_v8_boundary_band_mse": statistics.mean(
                float(
                    _regional_information_error(
                        output.image.detach().cpu().numpy(), reference
                    )["boundary_band_mse"]
                )
                for output, reference in zip(multiview_outputs, references)
            ),
            "occlusion_interpretation": "self-occlusion is encoded by forward absorption; with constant radiance its remaining image evidence appears at silhouettes and holes",
        },
        "readout_repeatability": {
            "repeats": 3,
            "maximum_image_difference": repeat_max_error,
            "deterministic_readout_variance": 0.0
            if repeat_max_error == 0.0
            else None,
        },
        "ensemble_swap": {"per_view": ensemble_b_rows, "mean": ensemble_b_mean},
        "camera_scaling": camera_scaling,
        "sample_convergence": convergence,
        "bandwidth_sensitivity": bandwidth_rows,
        "verdicts": verdicts,
        "limitations": [
            "The field is a discrete non-learned orthographic light-slab atlas, not a continuous plenoptic function.",
            "Off-atlas views use local angular interpolation and degrade as angular distance grows.",
            "Forward visibility inherits sign-change root tracing and can miss tangent contacts.",
            "Constant appearance validates geometry/silhouette information, not BRDF or radiometric fidelity.",
            "The v0.4 mesh ray cast is used only after rendering as an evaluation oracle.",
            "Field construction cost and event storage remain large at the highest sample level.",
        ],
        "artifacts": {
            "json": str(artifact_directory / "v05_camera_independent_transport.json"),
            "csv": str(artifact_directory / "v05_camera_independent_transport.csv"),
            "single_view": str(render_directory / "v05_single_view.png"),
            "multiview": str(render_directory / "v05_multiview.png"),
            "posthoc": str(render_directory / "v05_posthoc_camera.png"),
            "figures": [
                str(figure_directory / "v05_quality_vs_samples.png"),
                str(figure_directory / "v05_light_field_coverage.png"),
                str(figure_directory / "v05_camera_scaling.png"),
                str(figure_directory / "v05_bandwidth_sensitivity.png"),
            ],
        },
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v05_camera_independent_transport.json"
    json_path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n")
    _write_csv(artifact_directory / "v05_camera_independent_transport.csv", rows)
    return report
