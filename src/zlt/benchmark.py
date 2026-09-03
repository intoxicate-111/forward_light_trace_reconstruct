"""Local CUDA equivalence and sparse Full-HD scaling benchmarks."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from .camera import PlanarCamera
from .fields import (
    LocalBasisField,
    SphereField,
    TorusField,
    deform_reference_surface,
    unit_normals,
)
from .jacobian import (
    FixedTransportCell,
    _continuous_sensor,
    _footprint,
    _owners,
    _photon_batch,
    _topology,
    build_fixed_transport_cell,
    deterministic_directions,
    geometry_image_jacobian,
    make_local_basis_field,
    render_fixed_transport_cell,
    sparse_bilinear_transport,
    sparse_geometry_image_jacobian,
)
from .tracer import first_zero_set_intersections, trace_photons


Tensor = torch.Tensor
RESOLUTION_SWEEP = (
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (540, 960),
    (1080, 1920),
)


@dataclass
class BenchmarkGeometry:
    field: LocalBasisField
    reference_points: Tensor
    reference_normals: Tensor
    colors: Tensor


def _base_field(scene: str) -> SphereField | TorusField:
    if scene == "sphere":
        return SphereField()
    if scene == "torus":
        return TorusField()
    raise ValueError("scene must be 'sphere' or 'torus'")


def _field_to(field: LocalBasisField, device: torch.device) -> LocalBasisField:
    return LocalBasisField(
        field.base,
        field.centers.to(device),
        field.radii.to(device),
        field.coefficients.to(device),
    )


def _camera_to(camera: PlanarCamera, device: torch.device) -> PlanarCamera:
    return PlanarCamera(
        camera.center.to(device),
        camera.normal.to(device),
        camera.right.to(device),
        camera.up.to(device),
        camera.width,
        camera.height,
        camera.resolution,
    )


def benchmark_camera(
    scene: str, resolution: tuple[int, int], device: torch.device
) -> PlanarCamera:
    distance = 3.0 if scene == "sphere" else 3.5
    position = (0.0, 0.0, distance) if scene == "sphere" else (distance, 0.0, 0.0)
    center = torch.tensor(position, dtype=torch.float64, device=device)
    normal = -center / torch.linalg.vector_norm(center)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    right = torch.linalg.cross(normal, world_up)
    right = right / torch.linalg.vector_norm(right)
    up = torch.linalg.cross(right, normal)
    extent = 4.0 if scene == "sphere" else 4.5
    # The physical detector remains square for the entire sweep; widescreen
    # resolutions therefore test only detector discretization, with rectangular pixels.
    return PlanarCamera(center, normal, right, up, extent, extent, resolution)


def prepare_benchmark_geometry(
    scene: str,
    device: torch.device,
    *,
    emitters: int = 256,
    parameter_count: int = 32,
) -> BenchmarkGeometry:
    base = _base_field(scene)
    support_radius = 0.60 if scene == "sphere" else 0.50
    cpu_field = make_local_basis_field(base, parameter_count, support_radius)
    generator = torch.Generator().manual_seed(211)
    reference_points = base.sample_surface(emitters, generator)
    reference_normals = unit_normals(base, reference_points)
    colors = base.color(reference_points)
    return BenchmarkGeometry(
        _field_to(cpu_field, device),
        reference_points.to(device),
        reference_normals.to(device),
        colors.to(device),
    )


def _compare_sparse(cpu: Tensor, cuda: Tensor) -> tuple[bool, float]:
    cpu = cpu.coalesce().cpu()
    cuda = cuda.coalesce().cpu()
    same_indices = torch.equal(cpu.indices(), cuda.indices())
    maximum_difference = (
        float((cpu.values() - cuda.values()).abs().max())
        if same_indices and cpu._nnz()
        else torch.inf
    )
    return same_indices, maximum_difference


def cuda_equivalence_report() -> dict[str, object]:
    """Compare every small deterministic transport object against CPU float64."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    geometry_cpu = prepare_benchmark_geometry("torus", cpu, emitters=128)
    coefficients = torch.where(
        torch.arange(32) % 2 == 0,
        torch.tensor(0.003, dtype=torch.float64),
        torch.tensor(-0.003, dtype=torch.float64),
    )
    geometry_cpu.field = geometry_cpu.field.with_coefficients(coefficients)
    geometry_cuda = BenchmarkGeometry(
        _field_to(geometry_cpu.field, cuda),
        geometry_cpu.reference_points.to(cuda),
        geometry_cpu.reference_normals.to(cuda),
        geometry_cpu.colors.to(cuda),
    )
    camera_cpu = benchmark_camera("torus", (16, 16), cpu)
    camera_cuda = _camera_to(camera_cpu, cuda)

    points_cpu, success_cpu, _ = deform_reference_surface(
        geometry_cpu.field,
        geometry_cpu.reference_points,
        geometry_cpu.reference_normals,
    )
    points_cuda, success_cuda, _ = deform_reference_surface(
        geometry_cuda.field,
        geometry_cuda.reference_points,
        geometry_cuda.reference_normals,
    )
    normals_cpu = unit_normals(geometry_cpu.field, points_cpu)
    normals_cuda = unit_normals(geometry_cuda.field, points_cuda)
    directions_cpu = deterministic_directions(geometry_cpu.reference_normals, 4, 128.0)
    directions_cuda = deterministic_directions(geometry_cuda.reference_normals, 4, 128.0)
    photons_cpu = _photon_batch(points_cpu, directions_cpu, geometry_cpu.colors, 4)
    photons_cuda = _photon_batch(points_cuda, directions_cuda, geometry_cuda.colors, 4)
    camera_result_cpu = camera_cpu.intersect(points_cpu.repeat_interleave(4, 0), directions_cpu)
    camera_result_cuda = camera_cuda.intersect(
        points_cuda.repeat_interleave(4, 0), directions_cuda
    )
    trace_cpu = trace_photons(geometry_cpu.field, camera_cpu, photons_cpu)
    trace_cuda = trace_photons(geometry_cuda.field, camera_cuda, photons_cuda)
    state_cpu, owners_cpu, _ = _topology(
        geometry_cpu.field, camera_cpu, points_cpu, directions_cpu, geometry_cpu.colors, 4, 64
    )
    state_cuda, owners_cuda, _ = _topology(
        geometry_cuda.field,
        camera_cuda,
        points_cuda,
        directions_cuda,
        geometry_cuda.colors,
        4,
        64,
    )
    cell_cpu = build_fixed_transport_cell(
        geometry_cpu.field,
        camera_cpu,
        points_cpu,
        geometry_cpu.reference_normals,
        geometry_cpu.colors,
        packets_per_emitter=4,
        cone_power=128.0,
    )
    cell_cuda = build_fixed_transport_cell(
        geometry_cuda.field,
        camera_cuda,
        points_cuda,
        geometry_cuda.reference_normals,
        geometry_cuda.colors,
        packets_per_emitter=4,
        cone_power=128.0,
    )
    transport_cpu, sparse_image_cpu = sparse_bilinear_transport(cell_cpu, points_cpu)
    transport_cuda, sparse_image_cuda = sparse_bilinear_transport(cell_cuda, points_cuda)
    image_cpu = render_fixed_transport_cell(cell_cpu, points_cpu)
    image_cuda = render_fixed_transport_cell(cell_cuda, points_cuda)
    dense_jacobian_cpu = geometry_image_jacobian(
        geometry_cpu.field, points_cpu, geometry_cpu.reference_normals, cell_cpu
    )
    dense_jacobian_cuda = geometry_image_jacobian(
        geometry_cuda.field, points_cuda, geometry_cuda.reference_normals, cell_cuda
    )
    sparse_jacobian_cpu = sparse_geometry_image_jacobian(
        geometry_cpu.field, points_cpu, geometry_cpu.reference_normals, cell_cpu
    )
    sparse_jacobian_cuda = sparse_geometry_image_jacobian(
        geometry_cuda.field, points_cuda, geometry_cuda.reference_normals, cell_cuda
    )
    transport_indices_equal, transport_difference = _compare_sparse(
        transport_cpu, transport_cuda
    )
    jacobian_indices_equal, jacobian_sparse_difference = _compare_sparse(
        sparse_jacobian_cpu, sparse_jacobian_cuda
    )
    stable = dense_jacobian_cpu.matrix.abs() > 1e-12
    relative_jacobian_error = (
        (dense_jacobian_cpu.matrix - dense_jacobian_cuda.matrix.cpu()).abs()[stable]
        / dense_jacobian_cpu.matrix.abs()[stable]
    )
    sparse_construction_difference = float(
        (dense_jacobian_cpu.matrix - sparse_jacobian_cpu.to_dense()).abs().max()
    )
    report = {
        "positions_max_difference": float((points_cpu - points_cuda.cpu()).abs().max()),
        "root_success_equal": torch.equal(success_cpu, success_cuda.cpu()),
        "normals_max_difference": float((normals_cpu - normals_cuda.cpu()).abs().max()),
        "directions_max_difference": float(
            (directions_cpu - directions_cuda.cpu()).abs().max()
        ),
        "camera_valid_equal": torch.equal(camera_result_cpu[0], camera_result_cuda[0].cpu()),
        "camera_times_max_difference": float(
            (camera_result_cpu[1] - camera_result_cuda[1].cpu()).abs().max()
        ),
        "detector_positions_max_difference": float(
            (camera_result_cpu[3] - camera_result_cuda[3].cpu()).abs().max()
        ),
        "absorption_state_equal": torch.equal(state_cpu, state_cuda.cpu()),
        "owner_map_equal": torch.equal(owners_cpu, owners_cuda.cpu()),
        "arrival_times_max_difference": float(
            (trace_cpu.arrival_times - trace_cuda.arrival_times.cpu()).abs().max()
        ),
        "footprint_indices_equal": torch.equal(
            cell_cpu.footprint_pixels, cell_cuda.footprint_pixels.cpu()
        ),
        "render_max_difference": float((image_cpu - image_cuda.cpu()).abs().max()),
        "cpu_direct_sparse_max_difference": float(
            (image_cpu - sparse_image_cpu).abs().max()
        ),
        "cuda_direct_sparse_max_difference": float(
            (image_cuda - sparse_image_cuda).abs().max()
        ),
        "sparse_render_max_difference": float(
            (sparse_image_cpu - sparse_image_cuda.cpu()).abs().max()
        ),
        "transport_indices_equal": transport_indices_equal,
        "transport_values_max_difference": transport_difference,
        "jacobian_indices_equal": jacobian_indices_equal,
        "jacobian_values_max_difference": jacobian_sparse_difference,
        "jacobian_median_relative_error": float(relative_jacobian_error.median()),
        "jacobian_max_relative_error": float(relative_jacobian_error.max()),
        "sparse_by_construction_max_difference": sparse_construction_difference,
    }
    scalar_targets = (
        report["positions_max_difference"],
        report["normals_max_difference"],
        report["directions_max_difference"],
        report["camera_times_max_difference"],
        report["detector_positions_max_difference"],
        report["arrival_times_max_difference"],
        report["render_max_difference"],
        report["cpu_direct_sparse_max_difference"],
        report["cuda_direct_sparse_max_difference"],
        report["sparse_render_max_difference"],
        report["transport_values_max_difference"],
    )
    booleans = (
        report["root_success_equal"],
        report["camera_valid_equal"],
        report["absorption_state_equal"],
        report["owner_map_equal"],
        report["footprint_indices_equal"],
        report["transport_indices_equal"],
        report["jacobian_indices_equal"],
    )
    report["passed"] = bool(
        all(booleans)
        and max(scalar_targets) < 1e-8
        and report["jacobian_median_relative_error"] < 1e-6
        and sparse_construction_difference < 1e-8
    )
    return report


def _stage_events() -> tuple[list[str], list[torch.cuda.Event]]:
    names = [
        "zero_set_emitter_update",
        "photon_generation",
        "camera_plane_intersection",
        "zero_set_absorption_collision",
        "local_splatting_arrival_reduction",
        "sparse_transport_construction",
        "sparse_jacobian_construction",
    ]
    return names, [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]


def _cuda_pipeline(
    geometry: BenchmarkGeometry,
    camera: PlanarCamera,
    *,
    packets_per_emitter: int,
    cone_power: float,
    root_samples: int,
    bisection_steps: int,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, object], Tensor]:
    names, events = _stage_events()
    events[0].record()
    points, success, _ = deform_reference_surface(
        geometry.field, geometry.reference_points, geometry.reference_normals
    )
    current_normals = unit_normals(geometry.field, points)
    events[1].record()

    directions = deterministic_directions(
        geometry.reference_normals, packets_per_emitter, cone_power
    )
    photons = _photon_batch(points, directions, geometry.colors, packets_per_emitter)
    events[2].record()

    valid, camera_times, camera_pixels, _ = camera.intersect(
        photons.origins, directions
    )
    candidate_ids = torch.nonzero(valid, as_tuple=False).flatten()
    events[3].record()

    absorbed, _ = first_zero_set_intersections(
        geometry.field,
        photons.origins[candidate_ids],
        directions[candidate_ids],
        camera_times[candidate_ids],
        samples=root_samples,
        bisection_steps=bisection_steps,
        chunk_size=batch_size,
    )
    survivor_ids = candidate_ids[~absorbed]
    events[4].record()

    arrivals = camera_times[survivor_ids]
    owners = _owners(
        camera_pixels[survivor_ids], arrivals, survivor_ids, camera.pixel_count
    )
    photon_ids = owners[owners >= 0]
    emitter_ids = torch.div(
        photon_ids, packets_per_emitter, rounding_mode="floor"
    )
    selected_directions = directions[photon_ids]
    sensor_xy, _ = _continuous_sensor(
        camera, points[emitter_ids], selected_directions
    )
    footprint_pixels, footprint_valid, footprint_base = _footprint(
        sensor_xy, camera.resolution
    )
    state = torch.full(
        (photons.count,), -1, dtype=torch.long, device=points.device
    )
    state[valid] = camera_pixels[valid]
    state[candidate_ids[absorbed]] = -2
    cell = FixedTransportCell(
        camera,
        packets_per_emitter,
        cone_power,
        "reference",
        geometry.colors,
        photon_ids,
        emitter_ids,
        selected_directions,
        footprint_pixels,
        footprint_valid,
        footprint_base,
        state,
        owners,
    )
    events[5].record()

    transport, image = sparse_bilinear_transport(cell, points)
    events[6].record()

    jacobian = sparse_geometry_image_jacobian(
        geometry.field,
        points,
        geometry.reference_normals,
        cell,
    )
    events[7].record()
    torch.cuda.synchronize()
    stage_times = {
        name: events[index].elapsed_time(events[index + 1])
        for index, name in enumerate(names)
    }
    stage_times["total"] = events[0].elapsed_time(events[-1])

    jacobian_indices = jacobian.indices()
    pixel_parameter = torch.unique(
        torch.div(jacobian_indices[0], 3, rounding_mode="floor")
        * geometry.field.parameter_count
        + jacobian_indices[1]
    )
    affected_parameters = pixel_parameter % geometry.field.parameter_count
    affected_counts = torch.bincount(
        affected_parameters, minlength=geometry.field.parameter_count
    )
    responsive = affected_counts[affected_counts > 0]
    metrics: dict[str, object] = {
        "root_success": bool(success.all()),
        "max_zero_set_residual": float(geometry.field.value(points).abs().max()),
        "max_normal_error": float(
            (torch.linalg.vector_norm(current_normals, dim=-1) - 1.0).abs().max()
        ),
        "emitters": points.shape[0],
        "photons": photons.count,
        "camera_hits": survivor_ids.numel(),
        "detector_bound_photons": candidate_ids.numel(),
        "absorbed": int(absorbed.sum()),
        "absorption_fraction": float(absorbed.double().mean()) if absorbed.numel() else 0.0,
        "root_query_count": candidate_ids.numel() * (root_samples + bisection_steps),
        "average_bisection_iterations": bisection_steps,
        "transport_shape": list(transport.shape),
        "transport_layout": str(transport.layout),
        "transport_nnz": transport._nnz(),
        "transport_density": transport._nnz() / transport.numel(),
        "mean_emitters_per_pixel": transport._nnz() / camera.pixel_count,
        "mean_affected_pixels_per_emitter": transport._nnz() / points.shape[0],
        "jacobian_shape": list(jacobian.shape),
        "jacobian_layout": str(jacobian.layout),
        "jacobian_nnz": jacobian._nnz(),
        "jacobian_density": jacobian._nnz() / jacobian.numel(),
        "mean_affected_pixels_per_parameter": float(affected_counts.double().mean()),
        "median_affected_pixels_per_parameter": float(affected_counts.double().median()),
        "mean_affected_pixels_per_responsive_parameter": float(responsive.double().mean()),
        "mean_affected_pixel_fraction_per_parameter": float(
            affected_counts.double().mean() / camera.pixel_count
        ),
        "mean_active_parameters_per_pixel": pixel_parameter.numel() / camera.pixel_count,
        "image_checksum": float(image.sum()),
    }
    return stage_times, metrics, image


def benchmark_cuda_resolution(
    scene: str,
    resolution: tuple[int, int],
    *,
    emitters: int = 256,
    packets_per_emitter: int = 8,
    warm_runs: int = 5,
    batch_size: int = 4096,
    output_path: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    geometry = prepare_benchmark_geometry(scene, device, emitters=emitters)
    camera = benchmark_camera(scene, resolution, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cold_stages, metrics, cold_image = _cuda_pipeline(
        geometry,
        camera,
        packets_per_emitter=packets_per_emitter,
        cone_power=128.0,
        root_samples=64,
        bisection_steps=28,
        batch_size=batch_size,
    )
    if output_path is not None:
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            output_path,
            cold_image.reshape(*camera.resolution, 3).clamp(0.0, 1.0).cpu().numpy(),
        )
    del cold_image
    warm_results = [
        _cuda_pipeline(
            geometry,
            camera,
            packets_per_emitter=packets_per_emitter,
            cone_power=128.0,
            root_samples=64,
            bisection_steps=28,
            batch_size=batch_size,
        )[0]
        for _ in range(warm_runs)
    ]
    warm_totals = [result["total"] for result in warm_results]
    stage_medians = {
        stage: statistics.median(result[stage] for result in warm_results)
        for stage in warm_results[0]
    }
    width, height = resolution[1], resolution[0]
    return {
        "scene": scene,
        "resolution": [width, height],
        "pixels": width * height,
        **metrics,
        "cold_runtime_ms": cold_stages["total"],
        "warm_runtime_median_ms": statistics.median(warm_totals),
        "warm_runtime_min_ms": min(warm_totals),
        "warm_runtime_max_ms": max(warm_totals),
        "warm_stage_median_ms": stage_medians,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "output_path": str(output_path) if output_path is not None else None,
    }


def benchmark_cpu_reference(
    scene: str = "sphere", resolution: tuple[int, int] = (256, 256)
) -> float:
    geometry = prepare_benchmark_geometry(scene, torch.device("cpu"))
    camera = benchmark_camera(scene, resolution, torch.device("cpu"))
    import time

    started = time.perf_counter()
    _cuda_pipeline_cpu(geometry, camera)
    return 1000.0 * (time.perf_counter() - started)


def _cuda_pipeline_cpu(geometry: BenchmarkGeometry, camera: PlanarCamera) -> None:
    """Equivalent untimed CPU oracle used only for the moderate comparison."""
    points, success, _ = deform_reference_surface(
        geometry.field, geometry.reference_points, geometry.reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError("CPU root correspondence failed")
    cell = build_fixed_transport_cell(
        geometry.field,
        camera,
        points,
        geometry.reference_normals,
        geometry.colors,
        packets_per_emitter=8,
        cone_power=128.0,
    )
    sparse_bilinear_transport(cell, points)
    sparse_geometry_image_jacobian(
        geometry.field, points, geometry.reference_normals, cell
    )


def cuda_environment() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False, "status": "LOCAL_CUDA_UNAVAILABLE"}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "gpu": properties.name,
        "vram_mib": properties.total_memory / 2**20,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
    }
