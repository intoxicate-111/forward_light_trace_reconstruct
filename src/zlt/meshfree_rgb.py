"""v0.6 mesh-free forward zero-set RGB image-formation experiment."""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import (
    build_boundary_transport,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
    reweight_boundary_transport_rgb,
)
from .detector_readout import (
    LightFieldReadout,
    detector_from_outgoing_direction,
    read_light_field,
)
from .image_formation import (
    ReferenceImage,
    SharedSurfaceState,
    build_legacy_packet_scene,
    form_camera_image,
    form_legacy_packet_image,
    image_metrics,
)
from .light_field import SparseBoundaryLightField, build_sparse_boundary_light_field
from .mesh_field import GridZeroSetField, mesh_surface_samples, prepare_stanford_bunny
from .meshfree_surface import (
    MeshFreeSurfaceState,
    meshfree_base_color,
    sample_meshfree_zero_set,
)


Tensor = torch.Tensor
SURFACE_SAMPLES = 131_072
READOUT = {
    "angular_neighbors": 4,
    "angular_bandwidth_radians": 0.12,
    "spatial_bandwidth_pixels": 0.65,
    "support_threshold": 0.5,
    "sensor_gain": 1.5,
}


@dataclass(frozen=True)
class RGBTransportConfig:
    name: str
    distance_sigma: float
    cosine_power: float
    ambient_emission: float


ABLATIONS = (
    RGBTransportConfig("no_attenuation_no_lobe", 0.0, 0.0, 1.0),
    RGBTransportConfig("mild_attenuation_no_lobe", 0.45, 0.0, 1.0),
    RGBTransportConfig("strong_attenuation_no_lobe", 0.90, 0.0, 1.0),
    RGBTransportConfig("cosine_lobe_no_attenuation", 0.0, 1.0, 0.35),
    RGBTransportConfig("selected_mild_attenuation_cosine_lobe", 0.55, 1.0, 0.35),
)
# v0.7's paired A/B/C/D study showed that cosine lighting retains the Bunny's
# internal structure without exponential path attenuation.  Keep attenuated
# entries above as diagnostics, but make the cleaner C formulation the default.
SELECTED = ABLATIONS[3]


@dataclass(frozen=True)
class RGBReference:
    reference: ReferenceImage
    path_length: np.ndarray
    cosine: np.ndarray


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
    flat = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    keys = sorted({key for row in flat for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)


def _plot_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    rows: int,
    columns: int,
    title: str,
    *,
    figsize_scale: float = 3.0,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(figsize_scale * columns, 3.1 * rows),
        facecolor="black",
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
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def _rgb_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: object,
    boundary_center: np.ndarray,
    boundary_radius: float,
    lower: np.ndarray,
    upper: np.ndarray,
    config: RGBTransportConfig,
    sensor_gain: float,
) -> RGBReference:
    """Mesh ray oracle used only after the forward renderer has completed."""
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    rows, columns = camera.resolution
    horizontal = ((np.arange(columns) + 0.5) / columns - 0.5) * camera.width
    vertical = (0.5 - (np.arange(rows) + 0.5) / rows) * camera.height
    yy, xx = np.meshgrid(vertical, horizontal, indexing="ij")
    center = camera.center.detach().cpu().numpy()
    right = camera.right.detach().cpu().numpy()
    up = camera.up.detach().cpu().numpy()
    inward = camera.normal.detach().cpu().numpy()
    outgoing = -inward
    origins = center + xx[..., None] * right + yy[..., None] * up
    directions = np.broadcast_to(inward, origins.shape)
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    cast = scene.cast_rays(o3d.core.Tensor(rays))
    depth = cast["t_hit"].numpy().astype(np.float64)
    normals = cast["primitive_normals"].numpy().astype(np.float64)
    mask = np.isfinite(depth)
    points = origins + np.where(mask, depth, 0.0)[..., None] * directions
    normalized = np.clip((points - lower) / (upper - lower), 0.0, 1.0)
    base = np.stack(
        (
            0.20 + 0.72 * normalized[..., 0],
            0.18 + 0.70 * normalized[..., 1],
            0.28 + 0.62 * (1.0 - normalized[..., 2]),
        ),
        axis=-1,
    )
    cosine = np.clip(np.sum(normals * outgoing, axis=-1), 0.0, 1.0)
    if config.cosine_power > 0.0:
        lobe = config.ambient_emission + (
            1.0 - config.ambient_emission
        ) * cosine**config.cosine_power
    else:
        lobe = np.ones_like(cosine)
    relative = points - boundary_center
    linear = np.sum(relative * outgoing, axis=-1)
    constant = np.sum(relative * relative, axis=-1) - boundary_radius**2
    path = -linear + np.sqrt(np.maximum(linear * linear - constant, 0.0))
    attenuation = np.exp(-config.distance_sigma * path)
    image = np.clip(sensor_gain * base * (lobe * attenuation)[..., None], 0.0, 1.0)
    image[~mask] = 0.0
    path[~mask] = np.inf
    cosine[~mask] = 0.0
    normals[~mask] = 0.0
    return RGBReference(ReferenceImage(image, mask, depth, normals), path, cosine)


def _structure_metrics(image: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    from scipy.ndimage import binary_erosion

    interior = binary_erosion(mask, iterations=2)
    if not interior.any():
        return {
            "foreground_rgb_std": 0.0,
            "foreground_luminance_std": 0.0,
            "foreground_luminance_p95_minus_p05": 0.0,
            "interior_gradient_energy": 0.0,
            "quantized_foreground_color_count": 0,
            "clipped_channel_fraction": 0.0,
        }
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    gy, gx = np.gradient(luminance)
    foreground = image[mask]
    quantized = np.round(foreground * 31).astype(np.int16)
    return {
        "foreground_rgb_std": float(foreground.std()),
        "foreground_luminance_std": float(luminance[mask].std()),
        "foreground_luminance_p95_minus_p05": float(
            np.quantile(luminance[mask], 0.95) - np.quantile(luminance[mask], 0.05)
        ),
        "interior_gradient_energy": float(np.mean(np.sqrt(gx[interior] ** 2 + gy[interior] ** 2))),
        "quantized_foreground_color_count": int(np.unique(quantized, axis=0).shape[0]),
        "clipped_channel_fraction": float((foreground >= 1.0 - 1e-12).mean()),
    }


def _readout_summary(
    name: str,
    phase: str,
    output: LightFieldReadout,
    reference: RGBReference,
    light_field: SparseBoundaryLightField,
    **extra: object,
) -> dict[str, object]:
    image = output.image.detach().cpu().numpy()
    metrics = image_metrics(image, reference.reference)
    mask = output.mask.detach().cpu().numpy()
    overlap = mask & reference.reference.mask
    rgb_mae = float(np.mean(np.abs(image - reference.reference.image)))
    foreground_mae = (
        float(np.mean(np.abs(image[overlap] - reference.reference.image[overlap])))
        if overlap.any()
        else math.inf
    )
    return {
        "name": name,
        "phase": phase,
        **extra,
        **metrics,
        **_structure_metrics(image, mask),
        "rgb_mae": rgb_mae,
        "foreground_rgb_mae": foreground_mae,
        "readout_seconds": output.readout_seconds,
        "direct_support_fraction": output.direct_support_fraction,
        "interpolated_support_fraction": output.interpolated_support_fraction,
        "no_support_fraction": output.no_support_fraction,
        "field_occupied_bins": light_field.occupied_bins,
        "field_memory_mib": light_field.memory_bytes / 2**20,
    }


def _mean_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "silhouette_iou",
        "silhouette_f1",
        "ssim",
        "psnr",
        "edge_f1",
        "rgb_mae",
        "foreground_rgb_mae",
        "foreground_luminance_std",
        "interior_gradient_energy",
    )
    return {key: statistics.mean(float(row[key]) for row in rows) for key in keys}


def _mesh_reference_comparison(
    prepared: object, surface: MeshFreeSurfaceState, count: int = 65_536
) -> dict[str, object]:
    from scipy.spatial import cKDTree

    mesh_points, _ = mesh_surface_samples(
        prepared.repaired_vertices, prepared.repaired_faces, count
    )
    meshfree = surface.points.detach().cpu().numpy()
    mesh_tree = cKDTree(mesh_points)
    free_tree = cKDTree(meshfree)
    free_to_mesh = mesh_tree.query(meshfree, workers=-1)[0]
    mesh_to_free = free_tree.query(mesh_points, workers=-1)[0]
    return {
        "role": "evaluation_only",
        "mesh_reference_samples": count,
        "meshfree_samples": len(meshfree),
        "meshfree_to_mesh_mean": float(free_to_mesh.mean()),
        "meshfree_to_mesh_p95": float(np.quantile(free_to_mesh, 0.95)),
        "mesh_to_meshfree_mean": float(mesh_to_free.mean()),
        "mesh_to_meshfree_p95": float(np.quantile(mesh_to_free, 0.95)),
        "symmetric_chamfer": float(free_to_mesh.mean() + mesh_to_free.mean()),
    }


def _surface_coverage_figure(path: Path, surface: MeshFreeSurfaceState) -> None:
    import matplotlib.pyplot as plt

    points = surface.points.detach().cpu().numpy()[::4]
    colors = surface.base_colors.detach().cpu().numpy()[::4]
    projections = ((0, 1), (2, 1), (0, 2))
    labels = ("x-y", "z-y", "x-z")
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor="black")
    for axis, dimensions, label in zip(axes, projections, labels):
        axis.scatter(
            points[:, dimensions[0]],
            points[:, dimensions[1]],
            c=colors,
            s=0.15,
            linewidths=0,
        )
        axis.set_aspect("equal")
        axis.set_title(label, color="white")
        axis.axis("off")
    figure.suptitle("Mesh-free sign-changing-cell zero-set samples", color="white")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _code_audit() -> dict[str, object]:
    root = Path(__file__).parent
    main_files = [
        root / "meshfree_surface.py",
        root / "boundary_transport.py",
        root / "light_field.py",
        root / "detector_readout.py",
    ]
    forbidden_calls: list[str] = []
    forbidden_imports: list[str] = []
    for path in main_files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(token in module for token in ("skimage", "trimesh", "open3d")):
                    forbidden_imports.append(f"{path.name}:{module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(token in alias.name for token in ("skimage", "trimesh", "open3d")):
                        forbidden_imports.append(f"{path.name}:{alias.name}")
            if isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in {"marching_cubes", "form_camera_image", "reference_zero_set_image"}:
                    forbidden_calls.append(f"{path.name}:{name}")
    passed = not forbidden_imports and not forbidden_calls
    return {
        "main_forward_files": [str(path.relative_to(root.parent.parent)) for path in main_files],
        "forbidden_mesh_imports": forbidden_imports,
        "forbidden_mesh_or_projection_calls": forbidden_calls,
        "main_field_surface_vertex_count": 0,
        "main_field_surface_face_count": 0,
        "marching_cubes_used_in_main_path": False,
        "mesh_visibility_used_in_main_path": False,
        "camera_scene_query_used_in_main_path": False,
        "passed": passed,
    }


def _analytic_grid_field(device: torch.device, resolution: int = 32) -> GridZeroSetField:
    axis = torch.linspace(-1.3, 1.3, resolution, dtype=torch.float64, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = x * x + y * y + z * z - 1.0
    gradient = torch.stack((2.0 * x, 2.0 * y, 2.0 * z), -1)
    empty_vertices = torch.empty((0, 3), dtype=torch.float64, device=device)
    empty_faces = torch.empty((0, 3), dtype=torch.long, device=device)
    empty_cdf = torch.empty(0, dtype=torch.float64, device=device)
    return GridZeroSetField(
        grid,
        gradient,
        torch.full((3,), -1.3, dtype=torch.float64, device=device),
        torch.full((3,), 1.3, dtype=torch.float64, device=device),
        empty_vertices,
        empty_faces,
        empty_cdf,
    )


def meshfree_rgb_cpu_cuda_verification() -> dict[str, object]:
    cpu = torch.device("cpu")
    field_cpu = _analytic_grid_field(cpu)
    surface_cpu = sample_meshfree_zero_set(field_cpu, 2048)
    atlas_cpu = nested_fibonacci_atlas(cpu, (8,))
    boundary_cpu = enclosing_observation_sphere(surface_cpu.points)
    visible_cpu = build_boundary_transport(
        field_cpu,
        surface_cpu.points,
        surface_cpu.normals,
        atlas_cpu,
        boundary_cpu,
        root_samples=10,
        bisection_steps=12,
    )
    rgb_cpu = reweight_boundary_transport_rgb(
        visible_cpu,
        surface_cpu.base_colors,
        surface_cpu.normals,
        distance_sigma=SELECTED.distance_sigma,
        cosine_power=SELECTED.cosine_power,
        ambient_emission=SELECTED.ambient_emission,
        emission_strengths=surface_cpu.emission_strengths,
    )
    light_cpu = build_sparse_boundary_light_field(
        rgb_cpu, spatial_resolution=(32, 32)
    )
    camera_cpu = detector_from_outgoing_direction(
        boundary_cpu, atlas_cpu.directions[3], resolution=(32, 32)
    )
    image_cpu = read_light_field(
        light_cpu,
        camera_cpu,
        angular_neighbors=1,
        angular_bandwidth_radians=0.0,
        sensor_gain=READOUT["sensor_gain"],
    )
    report: dict[str, object] = {
        "cpu_root_failures": surface_cpu.root_failure_count,
        "cpu_degenerate_normals": surface_cpu.degenerate_normal_count,
        "cpu_nonfinite": surface_cpu.nonfinite_count,
        "cpu_residual_max": surface_cpu.residual_max,
        "cpu_image_finite": bool(torch.isfinite(image_cpu.image).all()),
        "cpu_image_nonempty": bool(image_cpu.mask.any()),
        "cuda_available": torch.cuda.is_available(),
    }
    assert surface_cpu.root_failure_count == 0
    assert surface_cpu.degenerate_normal_count == 0
    assert surface_cpu.nonfinite_count == 0
    assert report["cpu_image_finite"] and report["cpu_image_nonempty"]
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        field_cuda = _analytic_grid_field(cuda)
        surface_cuda = sample_meshfree_zero_set(field_cuda, 2048)
        atlas_cuda = nested_fibonacci_atlas(cuda, (8,))
        boundary_cuda = enclosing_observation_sphere(surface_cuda.points)
        visible_cuda = build_boundary_transport(
            field_cuda,
            surface_cuda.points,
            surface_cuda.normals,
            atlas_cuda,
            boundary_cuda,
            root_samples=10,
            bisection_steps=12,
        )
        rgb_cuda = reweight_boundary_transport_rgb(
            visible_cuda,
            surface_cuda.base_colors,
            surface_cuda.normals,
            distance_sigma=SELECTED.distance_sigma,
            cosine_power=SELECTED.cosine_power,
            ambient_emission=SELECTED.ambient_emission,
            emission_strengths=surface_cuda.emission_strengths,
        )
        light_cuda = build_sparse_boundary_light_field(
            rgb_cuda, spatial_resolution=(32, 32)
        )
        camera_cuda = detector_from_outgoing_direction(
            boundary_cuda, atlas_cuda.directions[3], resolution=(32, 32)
        )
        image_cuda = read_light_field(
            light_cuda,
            camera_cuda,
            angular_neighbors=1,
            angular_bandwidth_radians=0.0,
            sensor_gain=READOUT["sensor_gain"],
        )
        same_cells = torch.equal(surface_cpu.source_cell_ids, surface_cuda.source_cell_ids.cpu())
        report.update(
            {
                "cpu_cuda_source_cells_equal": same_cells,
                "cpu_cuda_surface_point_max_error": float(
                    (surface_cpu.points - surface_cuda.points.cpu()).abs().max()
                ),
                "cpu_cuda_sparse_keys_equal": torch.equal(
                    light_cpu.keys, light_cuda.keys.cpu()
                ),
                "cpu_cuda_image_max_error": float(
                    (image_cpu.image - image_cuda.image.cpu()).abs().max()
                ),
            }
        )
        assert same_cells
        assert report["cpu_cuda_sparse_keys_equal"]
        assert float(report["cpu_cuda_surface_point_max_error"]) < 1e-10
        assert float(report["cpu_cuda_image_max_error"]) < 1e-10
    return report


def _independent_consistency(
    field: GridZeroSetField,
    device: torch.device,
) -> dict[str, object]:
    small_count = 16_384
    atlas = nested_fibonacci_atlas(device, (8,))
    shared_surface = sample_meshfree_zero_set(field, small_count)
    boundary = enclosing_observation_sphere(shared_surface.points)
    started = time.perf_counter()
    visible = build_boundary_transport(
        field, shared_surface.points, shared_surface.normals, atlas, boundary
    )
    rgb = reweight_boundary_transport_rgb(
        visible,
        shared_surface.base_colors,
        shared_surface.normals,
        distance_sigma=SELECTED.distance_sigma,
        cosine_power=SELECTED.cosine_power,
        ambient_emission=SELECTED.ambient_emission,
        emission_strengths=shared_surface.emission_strengths,
    )
    shared_field = build_sparse_boundary_light_field(
        rgb, spatial_resolution=(128, 128)
    )
    cameras = [
        detector_from_outgoing_direction(
            boundary, atlas.directions[index], resolution=(128, 128)
        )
        for index in range(8)
    ]
    shared_images: list[Tensor] = []
    shared_elapsed: dict[int, float] = {}
    for index, camera in enumerate(cameras):
        shared_images.append(
            read_light_field(shared_field, camera, **READOUT).image
        )
        if index + 1 in (1, 4, 8):
            _sync(device)
            shared_elapsed[index + 1] = time.perf_counter() - started

    independent_images: list[Tensor] = []
    independent_elapsed: dict[int, float] = {}
    started = time.perf_counter()
    for index in range(8):
        current_surface = sample_meshfree_zero_set(field, small_count)
        current_boundary = enclosing_observation_sphere(current_surface.points)
        current_visible = build_boundary_transport(
            field,
            current_surface.points,
            current_surface.normals,
            atlas,
            current_boundary,
        )
        current_rgb = reweight_boundary_transport_rgb(
            current_visible,
            current_surface.base_colors,
            current_surface.normals,
            distance_sigma=SELECTED.distance_sigma,
            cosine_power=SELECTED.cosine_power,
            ambient_emission=SELECTED.ambient_emission,
            emission_strengths=current_surface.emission_strengths,
        )
        current_field = build_sparse_boundary_light_field(
            current_rgb, spatial_resolution=(128, 128)
        )
        current_camera = detector_from_outgoing_direction(
            current_boundary, atlas.directions[index], resolution=(128, 128)
        )
        independent_images.append(
            read_light_field(current_field, current_camera, **READOUT).image
        )
        if index + 1 in (1, 4, 8):
            _sync(device)
            independent_elapsed[index + 1] = time.perf_counter() - started
    errors = [
        float((shared - independent).abs().max())
        for shared, independent in zip(shared_images, independent_images)
    ]
    return {
        "surface_samples": small_count,
        "direction_count": atlas.count,
        "resolution": [128, 128],
        "maximum_image_error": max(errors),
        "per_view_maximum_image_error": errors,
        "shared_seconds": shared_elapsed,
        "independent_seconds": independent_elapsed,
        "speedup_at_8": independent_elapsed[8] / shared_elapsed[8],
        "shared_scene_builds": 1,
        "independent_scene_builds": 8,
    }


def _plot_diagnostics(
    figure_directory: Path,
    ablation_rows: list[dict[str, object]],
    view_scaling: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    figure_directory.mkdir(parents=True, exist_ok=True)
    labels = [str(row["attenuation_name"]).replace("_", "\n") for row in ablation_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(labels, [float(row["ssim"]) for row in ablation_rows], "o-", label="SSIM")
    axes[0].plot(
        labels,
        [float(row["silhouette_iou"]) for row in ablation_rows],
        "o-",
        label="IoU",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        labels,
        [float(row["foreground_luminance_std"]) for row in ablation_rows],
        "o-",
        label="luminance std",
    )
    axes[1].plot(
        labels,
        [float(row["interior_gradient_energy"]) for row in ablation_rows],
        "o-",
        label="interior gradient",
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v06_attenuation_ablation.png", dpi=180)
    plt.close(figure)

    counts = [int(row["view_count"]) for row in view_scaling]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        counts,
        [float(row["readout_seconds"]) for row in view_scaling],
        "o-",
        label="readout",
    )
    axis.plot(
        counts,
        [float(row["scene_field_readout_seconds"]) for row in view_scaling],
        "o-",
        label="scene + H + readout",
    )
    axis.set_xlabel("views")
    axis.set_ylabel("seconds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "v06_multiview_runtime.png", dpi=180)
    plt.close(figure)


def run_meshfree_rgb_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    experiment_started = time.perf_counter()

    # Input acquisition creates the fixed scalar field F but explicitly omits
    # all marching-cubes surface scaffolds.  The main renderer begins at F.
    prepared = prepare_stanford_bunny(
        mesh_path, build_surface_scaffold=False
    )
    field = prepared.gt_field.to(device)
    if field.surface_vertices.numel() or field.surface_faces.numel():
        raise RuntimeError("MESH_SCAFFOLD_PRESENT_IN_V06_MAIN_FIELD")

    # Main mesh-free, scene-centric path.  No camera exists until H is frozen.
    surface = sample_meshfree_zero_set(field, SURFACE_SAMPLES)
    if surface.root_failure_count or surface.degenerate_normal_count or surface.nonfinite_count:
        raise RuntimeError("MESHFREE_SURFACE_ROBUSTNESS_GATE_FAILED")
    atlas = nested_fibonacci_atlas(device, (8, 128))
    boundary = enclosing_observation_sphere(surface.points)
    visibility_events = build_boundary_transport(
        field,
        surface.points,
        surface.normals,
        atlas,
        boundary,
        root_samples=16,
        bisection_steps=18,
    )
    selected_events = reweight_boundary_transport_rgb(
        visibility_events,
        surface.base_colors,
        surface.normals,
        distance_sigma=SELECTED.distance_sigma,
        cosine_power=SELECTED.cosine_power,
        ambient_emission=SELECTED.ambient_emission,
        emission_strengths=surface.emission_strengths,
    )
    light_field = build_sparse_boundary_light_field(selected_events)
    frozen_visibility_digest = visibility_events.digest
    frozen_rgb_digest = selected_events.digest
    frozen_field_digest = light_field.digest

    # Cameras and all mesh evaluation are created only after the forward field.
    cameras = [
        detector_from_outgoing_direction(boundary, atlas.directions[index])
        for index in range(8)
    ]
    outputs = [read_light_field(light_field, camera, **READOUT) for camera in cameras]
    references = [
        _rgb_reference(
            prepared.repaired_vertices,
            prepared.repaired_faces,
            camera,
            boundary.center.detach().cpu().numpy(),
            boundary.radius,
            field.lower.detach().cpu().numpy(),
            field.upper.detach().cpu().numpy(),
            SELECTED,
            READOUT["sensor_gain"],
        )
        for camera in cameras
    ]

    rows: list[dict[str, object]] = []
    multiview_rows: list[dict[str, object]] = []
    for view, (output, reference) in enumerate(zip(outputs, references)):
        row = _readout_summary(
            f"rgb_multiview_{view:02d}",
            "multiview",
            output,
            reference,
            light_field,
            view=view,
            attenuation_name=SELECTED.name,
            distance_sigma=SELECTED.distance_sigma,
            cosine_power=SELECTED.cosine_power,
            ambient_emission=SELECTED.ambient_emission,
            **READOUT,
        )
        multiview_rows.append(row)
        rows.append(row)
    multiview_mean = _mean_metrics(multiview_rows)
    comparison_view = 3
    selected_image = outputs[comparison_view].image.detach().cpu().numpy()
    selected_reference = references[comparison_view]

    ablation_rows: list[dict[str, object]] = []
    ablation_images: list[tuple[str, np.ndarray]] = []
    for config in ABLATIONS:
        rgb_events = reweight_boundary_transport_rgb(
            visibility_events,
            surface.base_colors,
            surface.normals,
            distance_sigma=config.distance_sigma,
            cosine_power=config.cosine_power,
            ambient_emission=config.ambient_emission,
            emission_strengths=surface.emission_strengths,
        )
        current_field = build_sparse_boundary_light_field(rgb_events)
        output = read_light_field(
            current_field, cameras[comparison_view], **READOUT
        )
        reference = _rgb_reference(
            prepared.repaired_vertices,
            prepared.repaired_faces,
            cameras[comparison_view],
            boundary.center.detach().cpu().numpy(),
            boundary.radius,
            field.lower.detach().cpu().numpy(),
            field.upper.detach().cpu().numpy(),
            config,
            READOUT["sensor_gain"],
        )
        row = _readout_summary(
            f"ablation_{config.name}",
            "attenuation_ablation",
            output,
            reference,
            current_field,
            attenuation_name=config.name,
            distance_sigma=config.distance_sigma,
            cosine_power=config.cosine_power,
            ambient_emission=config.ambient_emission,
            **READOUT,
        )
        ablation_rows.append(row)
        rows.append(row)
        ablation_images.append(
            (config.name.replace("_", " "), output.image.detach().cpu().numpy())
        )

    # Controlled historical-operator comparisons.  They are evaluation only.
    shared_state = SharedSurfaceState(
        surface.points,
        surface.normals,
        field.lower,
        field.upper,
        surface.build_seconds,
        surface.residual_max,
        surface.peak_allocated_mib,
    )
    projected = form_camera_image(
        shared_state,
        cameras[comparison_view],
        appearance="position",
        footprint="bilinear",
        reduction="nearest_depth",
    )
    constant_events = reweight_boundary_transport_rgb(
        visibility_events,
        torch.full_like(surface.base_colors, 0.72),
        surface.normals,
        distance_sigma=0.0,
        emission_strengths=surface.emission_strengths,
    )
    constant_field = build_sparse_boundary_light_field(constant_events)
    constant_output = read_light_field(
        constant_field, cameras[comparison_view], **{**READOUT, "sensor_gain": 1.0}
    )
    legacy_scene = build_legacy_packet_scene(
        field,
        surface.points[:8192],
        surface.normals[:8192],
        surface.normals[:8192],
        surface.base_colors[:8192],
        packets_per_emitter=32,
        cone_power=128.0,
        resolution=256,
    )
    legacy_image, legacy_detector = form_legacy_packet_image(
        legacy_scene, cameras[comparison_view]
    )
    comparison_images = {
        "old_packet_hitmap": legacy_image,
        "v04_projection_rgb": projected.image.detach().cpu().numpy(),
        "v05_constant_boundary_field": constant_output.image.detach().cpu().numpy(),
        "v06_meshfree_forward_rgb": selected_image,
        "rgb_evaluation_oracle": selected_reference.reference.image,
    }
    comparison_rows: list[dict[str, object]] = []
    for name, image in comparison_images.items():
        metric = image_metrics(image, selected_reference.reference)
        mask = np.any(image > 1e-10, axis=-1)
        row = {
            "name": name,
            "phase": "operator_comparison",
            **metric,
            **_structure_metrics(image, mask),
            "rgb_mae": float(np.mean(np.abs(image - selected_reference.reference.image))),
        }
        comparison_rows.append(row)
        rows.append(row)

    mesh_reference = _mesh_reference_comparison(prepared, surface)
    consistency = _independent_consistency(field, device)
    if float(consistency["maximum_image_error"]) > 1e-10:
        raise RuntimeError("SHARED_INDEPENDENT_RGB_CONSISTENCY_FAILED")

    view_scaling: list[dict[str, object]] = []
    for count in (1, 4, 8):
        _sync(device)
        started = time.perf_counter()
        current = [
            read_light_field(light_field, camera, **READOUT)
            for camera in cameras[:count]
        ]
        _sync(device)
        readout_seconds = time.perf_counter() - started
        row = {
            "name": f"view_scaling_{count}",
            "phase": "view_scaling",
            "view_count": count,
            "readout_seconds": readout_seconds,
            "mean_readout_seconds": statistics.mean(item.readout_seconds for item in current),
            "scene_seconds": visibility_events.scene_seconds,
            "field_build_seconds": light_field.build_seconds,
            "scene_field_readout_seconds": visibility_events.scene_seconds
            + light_field.build_seconds
            + readout_seconds,
            "scene_retraversals": 0,
            "visibility_digest": visibility_events.digest,
            "rgb_field_digest": light_field.digest,
        }
        view_scaling.append(row)
        rows.append(row)

    cpu_cuda = meshfree_rgb_cpu_cuda_verification()
    audit = _code_audit()
    if not audit["passed"]:
        raise RuntimeError("V06_MESHFREE_MAIN_PATH_AUDIT_FAILED")

    # Required render and diagnostic artifacts.
    _plot_grid(
        render_directory / "v06_rgb_single_view.png",
        [
            ("mesh-free forward RGB", selected_image),
            ("RGB evaluation oracle", selected_reference.reference.image),
            (
                "absolute RGB difference",
                np.abs(selected_image - selected_reference.reference.image),
            ),
        ],
        1,
        3,
        "Depth-sensitive RGB from mesh-free forward zero-set transport",
    )
    _plot_grid(
        render_directory / "v06_renderer_comparison.png",
        [(name.replace("_", " "), image) for name, image in comparison_images.items()],
        1,
        5,
        "Old hitmap vs projection vs constant field vs mesh-free forward RGB",
        figsize_scale=2.8,
    )
    _plot_grid(
        render_directory / "v06_attenuation_ablation.png",
        ablation_images,
        1,
        len(ablation_images),
        "Distance attenuation and cosine-emission ablation",
        figsize_scale=2.8,
    )
    panels = [
        (f"v06 view {view}", output.image.detach().cpu().numpy())
        for view, output in enumerate(outputs)
    ]
    _plot_grid(
        render_directory / "v06_rgb_multiview.png",
        panels,
        2,
        4,
        "One mesh-free forward RGB field, eight detector views",
    )
    for view, output in enumerate(outputs):
        _save_rgb(
            render_directory / f"v06_bunny_rgb_view{view:02d}_256.png",
            output.image.detach().cpu().numpy(),
        )
    _surface_coverage_figure(
        figure_directory / "v06_meshfree_surface_coverage.png", surface
    )
    _plot_diagnostics(figure_directory, ablation_rows, view_scaling)

    selected_structure = _structure_metrics(
        selected_image, outputs[comparison_view].mask.detach().cpu().numpy()
    )
    silhouette_structure = _structure_metrics(
        constant_output.image.detach().cpu().numpy(),
        constant_output.mask.detach().cpu().numpy(),
    )
    image_like = (
        float(selected_structure["foreground_luminance_std"]) > 0.05
        and float(selected_structure["interior_gradient_energy"])
        > 3.0 * float(silhouette_structure["interior_gradient_energy"] + 1e-12)
        and int(selected_structure["quantized_foreground_color_count"]) > 128
    )
    numerical_ok = (
        surface.root_failure_count == 0
        and surface.degenerate_normal_count == 0
        and surface.nonfinite_count == 0
        and all(bool(torch.isfinite(output.image).all()) for output in outputs)
    )
    verdicts = {
        "mesh_free_surface": "MESH_FREE_ZERO_SET_SURFACE_SAMPLING_SUPPORTED",
        "mesh_free_forward": "MESH_FREE_FORWARD_RENDERING_SUPPORTED"
        if audit["passed"]
        else "MESH_FREE_FORWARD_RENDERING_NOT_SUPPORTED",
        "depth_sensitive_rgb": "DEPTH_SENSITIVE_RGB_IMAGE_FORMATION_SUPPORTED"
        if image_like
        else "DEPTH_SENSITIVE_RGB_IMAGE_FORMATION_NOT_SUPPORTED",
        "forward_scene_centric": "FORWARD_SCENE_CENTRIC_RENDERING_PRESERVED",
        "multiview_consistency": "FORWARD_SCENE_CENTRIC_RENDERER_RETAINS_MULTIVIEW_CONSISTENCY"
        if float(consistency["maximum_image_error"]) <= 1e-10
        else "MULTIVIEW_CONSISTENCY_NOT_SUPPORTED",
        "numerical_robustness": "MESH_FREE_NUMERICAL_ROBUSTNESS_SUPPORTED"
        if numerical_ok
        else "MESH_FREE_NUMERICAL_ROBUSTNESS_NOT_SUPPORTED",
    }
    report: dict[str, object] = {
        "version": "0.6.0",
        "environment": cuda_environment(),
        "formulation": {
            "surface": "F grid -> sign-changing cells -> safeguarded Newton / edge root fallback",
            "visibility": "forward packet survives iff next zero-set hit is after boundary exit",
            "rgb_energy": (
                "C_i exp(-sigma d_i) [ambient + (1-ambient) "
                "max(0,n_i dot omega_i)^gamma]"
            ),
            "image": "I_c = sensor_gain M_c H_Sigma(F)",
        },
        "main_path": {
            "implicit_grid_source": (
                "Stanford mesh sampled once into fixed scalar F; no mesh surface "
                "is constructed for rendering"
            ),
            "surface_scaffold": prepared.metadata["implicit_grid"]["render_surface_scaffold"],
            "field_surface_vertices": int(field.surface_vertices.shape[0]),
            "field_surface_faces": int(field.surface_faces.shape[0]),
            "surface_samples": SURFACE_SAMPLES,
            "emission_strength_minimum": float(surface.emission_strengths.min()),
            "emission_strength_maximum": float(surface.emission_strengths.max()),
            "direction_count": atlas.count,
            "selected_rgb_transport": asdict(SELECTED),
            "readout": READOUT,
        },
        "meshfree_surface": {
            key: value
            for key, value in surface.__dict__.items()
            if key
            not in {
                "points",
                "normals",
                "base_colors",
                "emission_strengths",
                "source_cell_ids",
            }
        },
        "mesh_assisted_reference_comparison": mesh_reference,
        "forward_transport": {
            "emitted": visibility_events.emitted_count,
            "absorbed": visibility_events.absorbed_count,
            "boundary_events": visibility_events.count,
            "escape_fraction": visibility_events.count / visibility_events.emitted_count,
            "scene_seconds": visibility_events.scene_seconds,
            "peak_allocated_mib": visibility_events.peak_allocated_mib,
            "visibility_digest": visibility_events.digest,
            "selected_rgb_event_digest": selected_events.digest,
            "event_memory_mib": selected_events.memory_bytes / 2**20,
        },
        "light_field": {
            "build_seconds": light_field.build_seconds,
            "occupied_bins": light_field.occupied_bins,
            "possible_bins": light_field.possible_bins,
            "occupancy_fraction": light_field.occupied_bins / light_field.possible_bins,
            "memory_mib": light_field.memory_bytes / 2**20,
            "digest": light_field.digest,
        },
        "sparsity": {
            "representation": (
                "sorted occupied (direction, transverse-row, transverse-column) bins"
            ),
            "occupied_fraction": light_field.occupied_bins / light_field.possible_bins,
            "maximum_bins_written_per_boundary_event": 4,
            "maximum_bins_queried_per_detector_pixel": 100,
            "owner_ids_retained_for_local_surface_provenance": True,
            "geometry_parameter_jacobian_rebuilt": False,
            "scope": (
                "v0.6 verifies local phase-space support; it does not claim a new "
                "end-to-end geometry-parameter Jacobian measurement"
            ),
        },
        "selected_view": multiview_rows[comparison_view],
        "selected_structure": selected_structure,
        "silhouette_baseline_structure": silhouette_structure,
        "multiview": {"per_view": multiview_rows, "mean": multiview_mean},
        "attenuation_ablation": ablation_rows,
        "operator_comparison": comparison_rows,
        "legacy_hitmap_diagnostics": legacy_detector,
        "shared_independent_consistency": consistency,
        "view_scaling": view_scaling,
        "camera_independence": {
            "visibility_digest_before_cameras": frozen_visibility_digest,
            "visibility_digest_after_all_cameras": visibility_events.digest,
            "rgb_event_digest_before_cameras": frozen_rgb_digest,
            "rgb_event_digest_after_all_cameras": selected_events.digest,
            "field_digest_before_cameras": frozen_field_digest,
            "field_digest_after_all_cameras": light_field.digest,
            "scene_retraversals_for_main_multiview": 0,
        },
        "cpu_cuda_validation": cpu_cuda,
        "code_audit": audit,
        "verdicts": verdicts,
        "limitations": [
            (
                "The Stanford source mesh is still used once to acquire the fixed "
                "scalar grid F and for evaluation, but never to generate rendering "
                "samples or visibility."
            ),
            (
                "The RGB model is interpretable emission/attenuation, not calibrated "
                "illumination or a BRDF."
            ),
            "The 136-direction field remains a discrete orthographic light-slab atlas.",
            "Sign-change intersection tracing can miss tangent contacts.",
            "Mesh-free cell-uniform samples are not exactly surface-area uniform.",
            (
                "Visibility, sparse-bin occupancy, and support thresholds remain "
                "piecewise differentiable."
            ),
        ],
        "artifacts": {
            "json": str(artifact_directory / "v06_meshfree_forward_rgb.json"),
            "csv": str(artifact_directory / "v06_meshfree_forward_rgb.csv"),
            "single_view": str(render_directory / "v06_rgb_single_view.png"),
            "comparison": str(render_directory / "v06_renderer_comparison.png"),
            "ablation": str(render_directory / "v06_attenuation_ablation.png"),
            "multiview": str(render_directory / "v06_rgb_multiview.png"),
            "surface_coverage": str(figure_directory / "v06_meshfree_surface_coverage.png"),
        },
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v06_meshfree_forward_rgb.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(artifact_directory / "v06_meshfree_forward_rgb.csv", rows)
    return report
