"""Shared scene-centric transport and strict multiview CUDA benchmarks."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch

from .benchmark import BenchmarkGeometry, cuda_environment, prepare_benchmark_geometry
from .camera import PlanarCamera
from .fields import deform_reference_surface, unit_normals
from .jacobian import (
    FixedTransportCell,
    _continuous_sensor,
    _footprint,
    _owners,
    _photon_batch,
    deterministic_directions,
    sparse_bilinear_transport,
)
from .tracer import PhotonBatch, first_zero_set_intersections


Tensor = torch.Tensor
T = TypeVar("T")
CAMERA_COUNTS = (1, 2, 4, 8, 16, 20)


@dataclass(frozen=True)
class MultiviewConfig:
    resolution: tuple[int, int] = (1080, 1920)
    emitters: int = 256
    packets_per_emitter: int = 8
    parameter_count: int = 32
    cone_power: float = 128.0
    root_samples: int = 64
    bisection_steps: int = 28
    collision_limit: float = 4.0
    collision_batch_size: int = 4096
    warm_runs: int = 5
    measured_runs: int = 10


@dataclass
class SceneTransportState:
    """Camera-independent output of one zero-set transport traversal."""

    points: Tensor
    current_normals: Tensor
    photons: PhotonBatch
    directions: Tensor
    surface_hits: Tensor
    surface_times: Tensor
    emitter_root_success: Tensor


@dataclass
class CameraIntersection:
    valid: Tensor
    times: Tensor
    pixels: Tensor


@dataclass
class CameraTransport:
    cell: FixedTransportCell
    transport: Tensor
    image: Tensor
    intersection: CameraIntersection
    survivor_ids: Tensor
    absorbed_detector_mask: Tensor


def multiview_cameras(
    scene: str,
    resolution: tuple[int, int],
    device: torch.device,
    *,
    count: int = 20,
) -> list[PlanarCamera]:
    """Create a deterministic Fibonacci-sphere camera set with no duplicates."""
    if scene not in {"sphere", "torus", "bunny"}:
        raise ValueError("scene must be 'sphere', 'torus', or 'bunny'")
    if count < 1:
        raise ValueError("camera count must be positive")
    distance = {"sphere": 3.0, "torus": 3.5, "bunny": 3.0}[scene]
    extent = {"sphere": 4.0, "torus": 4.5, "bunny": 2.8}[scene]
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cameras: list[PlanarCamera] = []
    for index in range(count):
        vertical = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - vertical * vertical))
        angle = index * golden_angle
        direction = torch.tensor(
            [radial * math.cos(angle), vertical, radial * math.sin(angle)],
            dtype=torch.float64,
            device=device,
        )
        center = distance * direction
        normal = -direction
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
        if abs(float(normal @ helper)) > 0.9:
            helper = torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float64, device=device
            )
        right = torch.linalg.cross(normal, helper)
        right = right / torch.linalg.vector_norm(right)
        up = torch.linalg.cross(right, normal)
        cameras.append(
            PlanarCamera(center, normal, right, up, extent, extent, resolution)
        )
    centers = torch.stack([camera.center for camera in cameras])
    if torch.unique(centers, dim=0).shape[0] != count:
        raise RuntimeError("multiview camera placement contains duplicates")
    return cameras


def camera_pose(camera: PlanarCamera) -> dict[str, object]:
    return {
        "center": camera.center.detach().cpu().tolist(),
        "normal": camera.normal.detach().cpu().tolist(),
        "right": camera.right.detach().cpu().tolist(),
        "up": camera.up.detach().cpu().tolist(),
        "width": camera.width,
        "height": camera.height,
        "resolution": list(camera.resolution),
    }


def _update_emitters(geometry: BenchmarkGeometry) -> tuple[Tensor, Tensor, Tensor]:
    points, success, _ = deform_reference_surface(
        geometry.field, geometry.reference_points, geometry.reference_normals
    )
    return points, unit_normals(geometry.field, points), success


def _generate_photons(
    geometry: BenchmarkGeometry, points: Tensor, config: MultiviewConfig
) -> tuple[Tensor, PhotonBatch]:
    directions = deterministic_directions(
        geometry.reference_normals, config.packets_per_emitter, config.cone_power
    )
    photons = _photon_batch(
        points, directions, geometry.colors, config.packets_per_emitter
    )
    return directions, photons


def _collide_scene(
    geometry: BenchmarkGeometry, photons: PhotonBatch, config: MultiviewConfig
) -> tuple[Tensor, Tensor]:
    # All supported zero sets lie inside this fixed scene-space distance from
    # their emitters.  It is independent of detector placement and exceeds the
    # 3.3-unit diameter of the larger validated torus scene.
    maximum_times = torch.full(
        (photons.count,),
        config.collision_limit,
        dtype=photons.origins.dtype,
        device=photons.origins.device,
    )
    return first_zero_set_intersections(
        geometry.field,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.bisection_steps,
        chunk_size=config.collision_batch_size,
    )


def build_scene_transport(
    geometry: BenchmarkGeometry, config: MultiviewConfig
) -> SceneTransportState:
    points, normals, success = _update_emitters(geometry)
    directions, photons = _generate_photons(geometry, points, config)
    surface_hits, surface_times = _collide_scene(geometry, photons, config)
    return SceneTransportState(
        points,
        normals,
        photons,
        directions,
        surface_hits,
        surface_times,
        success,
    )


def intersect_camera(
    camera: PlanarCamera, state: SceneTransportState
) -> CameraIntersection:
    valid, times, pixels, _ = camera.intersect(
        state.photons.origins, state.directions
    )
    return CameraIntersection(valid, times, pixels)


def _assemble_camera_cell(
    geometry: BenchmarkGeometry,
    camera: PlanarCamera,
    state: SceneTransportState,
    intersection: CameraIntersection,
    config: MultiviewConfig,
) -> tuple[FixedTransportCell, Tensor, Tensor]:
    absorbed = (
        intersection.valid
        & state.surface_hits
        & (state.surface_times < intersection.times)
    )
    survived = intersection.valid & ~absorbed
    survivor_ids = torch.nonzero(survived, as_tuple=False).flatten()
    owners = _owners(
        intersection.pixels[survivor_ids],
        intersection.times[survivor_ids],
        survivor_ids,
        camera.pixel_count,
    )
    photon_ids = owners[owners >= 0]
    emitter_ids = torch.div(
        photon_ids, config.packets_per_emitter, rounding_mode="floor"
    )
    selected_directions = state.directions[photon_ids]
    sensor_xy, _ = _continuous_sensor(
        camera, state.points[emitter_ids], selected_directions
    )
    footprint_pixels, footprint_valid, footprint_base = _footprint(
        sensor_xy, camera.resolution
    )
    photon_state = torch.full(
        (state.photons.count,), -1, dtype=torch.long, device=state.points.device
    )
    photon_state[intersection.valid] = intersection.pixels[intersection.valid]
    photon_state[absorbed] = -2
    cell = FixedTransportCell(
        camera,
        config.packets_per_emitter,
        config.cone_power,
        "reference",
        geometry.colors,
        photon_ids,
        emitter_ids,
        selected_directions,
        footprint_pixels,
        footprint_valid,
        footprint_base,
        photon_state,
        owners,
    )
    return cell, survivor_ids, absorbed


def project_camera(
    geometry: BenchmarkGeometry,
    camera: PlanarCamera,
    state: SceneTransportState,
    config: MultiviewConfig,
) -> CameraTransport:
    intersection = intersect_camera(camera, state)
    cell, survivor_ids, absorbed = _assemble_camera_cell(
        geometry, camera, state, intersection, config
    )
    transport, image = sparse_bilinear_transport(cell, state.points)
    return CameraTransport(
        cell, transport, image, intersection, survivor_ids, absorbed
    )


def _float_difference(left: Tensor, right: Tensor) -> tuple[float, float]:
    difference = (left - right).abs()
    if difference.numel() == 0:
        return 0.0, 0.0
    return float(difference.max()), float(difference.mean())


def _compare_camera(
    shared: CameraTransport, independent: CameraTransport
) -> dict[str, object]:
    shared_sparse = shared.transport.coalesce()
    independent_sparse = independent.transport.coalesce()
    image_max, image_mean = _float_difference(shared.image, independent.image)
    time_max, _ = _float_difference(
        shared.intersection.times, independent.intersection.times
    )
    same_sparse_shape = shared_sparse.shape == independent_sparse.shape
    same_sparse_nnz = shared_sparse._nnz() == independent_sparse._nnz()
    same_sparse_indices = same_sparse_nnz and torch.equal(
        shared_sparse.indices(), independent_sparse.indices()
    )
    if same_sparse_indices:
        sparse_value_max, _ = _float_difference(
            shared_sparse.values(), independent_sparse.values()
        )
    else:
        sparse_value_max = math.inf
    exact = {
        "detector_valid": torch.equal(
            shared.intersection.valid, independent.intersection.valid
        ),
        "detector_pixels": torch.equal(
            shared.intersection.pixels, independent.intersection.pixels
        ),
        "survivor_ids": torch.equal(
            shared.survivor_ids, independent.survivor_ids
        ),
        "photon_state": torch.equal(
            shared.cell.photon_state, independent.cell.photon_state
        ),
        "owner_map": torch.equal(
            shared.cell.owner_map, independent.cell.owner_map
        ),
        "transport_shape": same_sparse_shape,
        "transport_nnz": same_sparse_nnz,
        "transport_indices": same_sparse_indices,
    }
    passed = (
        all(exact.values())
        and image_max < 1e-12
        and time_max < 1e-12
        and sparse_value_max < 1e-12
    )
    return {
        "passed": passed,
        "image_max_abs_error": image_max,
        "image_mean_abs_error": image_mean,
        "camera_times_max_abs_error": time_max,
        "transport_values_max_abs_error": sparse_value_max,
        "exact": exact,
    }


def validate_multiview_equivalence(
    geometry: BenchmarkGeometry,
    cameras: list[PlanarCamera],
    config: MultiviewConfig,
) -> dict[str, object]:
    """Compare shared and repeated state construction camera by camera."""
    shared_state = build_scene_transport(geometry, config)
    if not bool(shared_state.emitter_root_success.all()):
        raise RuntimeError("shared emitter update left the normal-line root bracket")
    camera_reports: list[dict[str, object]] = []
    total_hits = 0
    total_absorbed = 0
    transport_nnz = 0
    for camera in cameras:
        shared = project_camera(geometry, camera, shared_state, config)
        independent_state = build_scene_transport(geometry, config)
        independent = project_camera(
            geometry, camera, independent_state, config
        )
        comparison = _compare_camera(shared, independent)
        camera_reports.append(comparison)
        total_hits += int(shared.survivor_ids.numel())
        total_absorbed += int(shared.absorbed_detector_mask.sum())
        transport_nnz += shared.transport._nnz()
    maximum_image_error = max(
        report["image_max_abs_error"] for report in camera_reports
    )
    mean_image_error = statistics.mean(
        report["image_mean_abs_error"] for report in camera_reports
    )
    return {
        "passed": all(report["passed"] for report in camera_reports),
        "cameras_checked": len(cameras),
        "max_image_absolute_error": maximum_image_error,
        "mean_image_absolute_error": mean_image_error,
        "max_camera_time_absolute_error": max(
            report["camera_times_max_abs_error"] for report in camera_reports
        ),
        "max_transport_value_absolute_error": max(
            report["transport_values_max_abs_error"] for report in camera_reports
        ),
        "all_integer_state_exact": all(
            all(report["exact"].values()) for report in camera_reports
        ),
        "camera_hits": total_hits,
        "absorbed_detector_bound_photons": total_absorbed,
        "transport_nnz": transport_nnz,
        "transport_density": transport_nnz
        / (len(cameras) * cameras[0].pixel_count * config.emitters),
        "scene_surface_hits": int(shared_state.surface_hits.sum()),
    }


def multiview_cpu_verification() -> dict[str, object]:
    config = MultiviewConfig(
        resolution=(12, 20),
        emitters=64,
        packets_per_emitter=4,
        parameter_count=8,
        root_samples=32,
        bisection_steps=20,
        warm_runs=0,
        measured_runs=0,
    )
    results: dict[str, object] = {}
    for scene in ("sphere", "torus"):
        geometry = prepare_benchmark_geometry(
            scene,
            torch.device("cpu"),
            emitters=config.emitters,
            parameter_count=config.parameter_count,
        )
        cameras = multiview_cameras(
            scene, config.resolution, torch.device("cpu"), count=2
        )
        results[scene] = validate_multiview_equivalence(
            geometry, cameras, config
        )
        if not results[scene]["passed"]:
            raise AssertionError(f"{scene} shared/independent mismatch")
    return results


def _record_stage(
    records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
    name: str,
    operation: Callable[[], T],
) -> T:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    records.setdefault(name, []).append((start, end))
    return result


def _timed_pipeline(
    geometry: BenchmarkGeometry,
    cameras: list[PlanarCamera],
    config: MultiviewConfig,
    *,
    shared: bool,
) -> dict[str, float]:
    records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    def make_state() -> SceneTransportState:
        updated = _record_stage(
            records, "emitter_update", lambda: _update_emitters(geometry)
        )
        points, normals, success = updated
        directions, photons = _record_stage(
            records,
            "photon_generation",
            lambda: _generate_photons(geometry, points, config),
        )
        surface_hits, surface_times = _record_stage(
            records,
            "zero_set_collision",
            lambda: _collide_scene(geometry, photons, config),
        )
        return SceneTransportState(
            points,
            normals,
            photons,
            directions,
            surface_hits,
            surface_times,
            success,
        )

    shared_state = make_state() if shared else None
    for camera in cameras:
        state = shared_state if shared_state is not None else make_state()
        intersection = _record_stage(
            records,
            "camera_intersection",
            lambda camera=camera, state=state: intersect_camera(camera, state),
        )
        cell, _, _ = _record_stage(
            records,
            "arrival_footprint",
            lambda camera=camera, state=state, intersection=intersection: (
                _assemble_camera_cell(
                    geometry, camera, state, intersection, config
                )
            ),
        )
        _record_stage(
            records,
            "sparse_transport",
            lambda cell=cell, state=state: sparse_bilinear_transport(
                cell, state.points
            ),
        )
    total_end.record()
    torch.cuda.synchronize()
    result = {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        for name, pairs in records.items()
    }
    result["geometry_side"] = sum(
        result[name]
        for name in ("emitter_update", "photon_generation", "zero_set_collision")
    )
    result["detector_side"] = sum(
        result[name]
        for name in ("camera_intersection", "arrival_footprint", "sparse_transport")
    )
    result["total"] = total_start.elapsed_time(total_end)
    return result


def _benchmark_mode(
    geometry: BenchmarkGeometry,
    cameras: list[PlanarCamera],
    config: MultiviewConfig,
    *,
    shared: bool,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    for _ in range(config.warm_runs):
        _timed_pipeline(geometry, cameras, config, shared=shared)
    torch.cuda.reset_peak_memory_stats()
    measured = [
        _timed_pipeline(geometry, cameras, config, shared=shared)
        for _ in range(config.measured_runs)
    ]
    stage_names = measured[0].keys()
    medians = {
        name: statistics.median(run[name] for run in measured)
        for name in stage_names
    }
    totals = [run["total"] for run in measured]
    return {
        "median_ms": statistics.median(totals),
        "min_ms": min(totals),
        "max_ms": max(totals),
        "stage_median_ms": medians,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def _linear_fit(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    x = [float(row["views"]) for row in rows]
    y = [float(row[key]) for row in rows]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum(
        (a - mean_x) ** 2 for a in x
    )
    intercept = mean_y - slope * mean_x
    residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
    total = sum((b - mean_y) ** 2 for b in y)
    return {
        "intercept_ms": intercept,
        "slope_ms_per_camera": slope,
        "r_squared": 1.0 - sum(value * value for value in residuals) / total,
        "rmse_ms": math.sqrt(statistics.mean(value * value for value in residuals)),
    }


def _benchmark_scene(
    scene: str, config: MultiviewConfig, device: torch.device
) -> dict[str, object]:
    geometry = prepare_benchmark_geometry(
        scene,
        device,
        emitters=config.emitters,
        parameter_count=config.parameter_count,
    )
    cameras = multiview_cameras(
        scene, config.resolution, device, count=max(CAMERA_COUNTS)
    )
    rows: list[dict[str, object]] = []
    previous_views = 0
    previous_shared = 0.0
    for views in CAMERA_COUNTS:
        selected = cameras[:views]
        correctness = validate_multiview_equivalence(
            geometry, selected, config
        )
        if not correctness["passed"]:
            raise RuntimeError(
                f"{scene} shared/independent correctness failed at {views} views"
            )
        shared_result = _benchmark_mode(
            geometry, selected, config, shared=True
        )
        independent_result = _benchmark_mode(
            geometry, selected, config, shared=False
        )
        shared_ms = float(shared_result["median_ms"])
        independent_ms = float(independent_result["median_ms"])
        rows.append(
            {
                "views": views,
                "independent_total_ms": independent_ms,
                "independent_min_ms": independent_result["min_ms"],
                "independent_max_ms": independent_result["max_ms"],
                "shared_total_ms": shared_ms,
                "shared_min_ms": shared_result["min_ms"],
                "shared_max_ms": shared_result["max_ms"],
                "speedup": independent_ms / shared_ms,
                "shared_geometry_side_ms": shared_result["stage_median_ms"][
                    "geometry_side"
                ],
                "shared_detector_side_ms": shared_result["stage_median_ms"][
                    "detector_side"
                ],
                "shared_stage_median_ms": shared_result["stage_median_ms"],
                "incremental_ms_per_view": (
                    None
                    if previous_views == 0
                    else (shared_ms - previous_shared) / (views - previous_views)
                ),
                "transport_nnz": correctness["transport_nnz"],
                "transport_density": correctness["transport_density"],
                "jacobian_nnz": None,
                "peak_allocated_mib": max(
                    shared_result["peak_allocated_mib"],
                    independent_result["peak_allocated_mib"],
                ),
                "peak_reserved_mib": max(
                    shared_result["peak_reserved_mib"],
                    independent_result["peak_reserved_mib"],
                ),
                "shared_peak_allocated_mib": shared_result[
                    "peak_allocated_mib"
                ],
                "shared_peak_reserved_mib": shared_result[
                    "peak_reserved_mib"
                ],
                "camera_hits": correctness["camera_hits"],
                "absorbed_detector_bound_photons": correctness[
                    "absorbed_detector_bound_photons"
                ],
                "scene_surface_hits": correctness["scene_surface_hits"],
                "max_output_error": correctness[
                    "max_image_absolute_error"
                ],
                "mean_output_error": correctness[
                    "mean_image_absolute_error"
                ],
                "max_camera_time_error": correctness[
                    "max_camera_time_absolute_error"
                ],
                "max_transport_value_error": correctness[
                    "max_transport_value_absolute_error"
                ],
                "integer_state_exact": correctness[
                    "all_integer_state_exact"
                ],
                "shared_scene_traversals": 1,
                "independent_scene_traversals": views,
            }
        )
        previous_views, previous_shared = views, shared_ms
    shared_fit = _linear_fit(rows, "shared_total_ms")
    independent_one = float(rows[0]["independent_total_ms"])
    for row in rows:
        ideal = int(row["views"]) * independent_one
        row["independent_deviation_from_n_times_one_percent"] = (
            100.0 * (float(row["independent_total_ms"]) - ideal) / ideal
        )
    last = rows[-1]
    allocated = [float(row["peak_allocated_mib"]) for row in rows]
    reserved = [float(row["peak_reserved_mib"]) for row in rows]
    supported = (
        all(bool(row["integer_state_exact"]) and row["max_output_error"] < 1e-12 for row in rows)
        and float(last["speedup"]) > 1.0
        and float(last["shared_total_ms"]) / float(rows[0]["shared_total_ms"])
        < float(last["independent_total_ms"])
        / float(rows[0]["independent_total_ms"])
    )
    return {
        "camera_poses": [camera_pose(camera) for camera in cameras],
        "rows": rows,
        "shared_linear_fit": shared_fit,
        "shared_1_to_20_factor": float(last["shared_total_ms"])
        / float(rows[0]["shared_total_ms"]),
        "independent_1_to_20_factor": float(last["independent_total_ms"])
        / float(rows[0]["independent_total_ms"]),
        "n20_milliseconds_saved": float(last["independent_total_ms"])
        - float(last["shared_total_ms"]),
        "transport_nnz_1_to_20_factor": float(last["transport_nnz"])
        / float(rows[0]["transport_nnz"]),
        "memory_scaling": {
            "classification": "approximately constant (camera outputs streamed)",
            "allocated_range_mib": [min(allocated), max(allocated)],
            "reserved_range_mib": [min(reserved), max(reserved)],
        },
        "supported": supported,
    }


def _write_artifacts(
    report: dict[str, object], json_path: Path, figure_directory: Path
) -> dict[str, object]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = json_path.with_suffix(".csv")
    columns = (
        "scene",
        "views",
        "independent_total_ms",
        "shared_total_ms",
        "speedup",
        "shared_geometry_side_ms",
        "shared_detector_side_ms",
        "incremental_ms_per_view",
        "transport_nnz",
        "transport_density",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "camera_hits",
        "absorbed_detector_bound_photons",
        "max_output_error",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for scene, scene_report in report["scenes"].items():
            for row in scene_report["rows"]:
                writer.writerow(
                    {name: scene if name == "scene" else row[name] for name in columns}
                )

    import matplotlib.pyplot as plt

    figure_directory.mkdir(parents=True, exist_ok=True)
    runtime_path = figure_directory / "v022_multiview_runtime.png"
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for axis, (scene, scene_report) in zip(axes, report["scenes"].items()):
        rows = scene_report["rows"]
        views = [row["views"] for row in rows]
        axis.plot(views, [row["independent_total_ms"] for row in rows], "o-", label="independent")
        axis.plot(views, [row["shared_total_ms"] for row in rows], "o-", label="shared")
        axis.set_title(scene.capitalize())
        axis.set_xlabel("cameras")
        axis.set_ylabel("median runtime (ms)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(runtime_path, dpi=160)
    plt.close(figure)

    speedup_path = figure_directory / "v022_multiview_speedup.png"
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    for scene, scene_report in report["scenes"].items():
        rows = scene_report["rows"]
        axis.plot(
            [row["views"] for row in rows],
            [row["speedup"] for row in rows],
            "o-",
            label=scene.capitalize(),
        )
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("cameras")
    axis.set_ylabel("independent / shared")
    axis.set_title("Shared multiview speedup")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(speedup_path, dpi=160)
    plt.close(figure)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "plots": [str(runtime_path), str(speedup_path)],
    }


def run_multiview_benchmark(
    *,
    config: MultiviewConfig | None = None,
    output_path: Path | None = None,
    figure_directory: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    config = config or MultiviewConfig()
    if config.warm_runs < 5 or config.measured_runs < 10:
        raise ValueError("final multiview benchmark requires 5 warm and 10 measured runs")
    if config.collision_limit < 3.3:
        raise ValueError("collision limit must cover the validated torus diameter")
    device = torch.device("cuda")
    scenes = {
        scene: _benchmark_scene(scene, config, device)
        for scene in ("sphere", "torus")
    }
    supported = all(scene["supported"] for scene in scenes.values())
    report: dict[str, object] = {
        "experiment": "v0.2.2 shared multiview forward-transport scaling",
        "environment": cuda_environment(),
        "configuration": {
            "resolution": [config.resolution[1], config.resolution[0]],
            "emitters": config.emitters,
            "packets_per_emitter": config.packets_per_emitter,
            "photons": config.emitters * config.packets_per_emitter,
            "parameter_count": config.parameter_count,
            "cone_power": config.cone_power,
            "root_samples": config.root_samples,
            "bisection_steps": config.bisection_steps,
            "collision_limit": config.collision_limit,
            "camera_counts": list(CAMERA_COUNTS),
            "warm_runs": config.warm_runs,
            "measured_runs": config.measured_runs,
            "camera_placement": "deterministic 20-point Fibonacci sphere",
            "output_memory": "dense images streamed camera-by-camera; sparse COO T per camera",
            "jacobian": "not measured; primary transport experiment only",
        },
        "scenes": scenes,
        "gates": {
            "shared_independent_equivalence": all(
                all(
                    row["max_output_error"] < 1e-12
                    and row["integer_state_exact"]
                    for row in scene["rows"]
                )
                for scene in scenes.values()
            ),
            "one_shared_scene_traversal": True,
            "full_hd_both_scenes": config.resolution == (1080, 1920),
            "twenty_distinct_cameras": True,
            "shared_scales_better_than_independent": supported,
        },
        "verdict": (
            "SHARED_MULTIVIEW_TRANSPORT_SCALING_SUPPORTED"
            if supported
            else "SHARED_MULTIVIEW_TRANSPORT_SCALING_NOT_SUPPORTED"
        ),
    }
    if output_path is not None:
        if figure_directory is None:
            raise ValueError("figure_directory is required when output_path is set")
        report["artifacts"] = _write_artifacts(
            report, output_path, figure_directory
        )
        # Rewrite once so the JSON records its own companion artifact paths.
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
