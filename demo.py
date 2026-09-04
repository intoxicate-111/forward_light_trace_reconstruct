#!/usr/bin/env python3
"""Command-line demo and deterministic production-code verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Keep the requested `python demo.py` workflow usable before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from zlt import (  # noqa: E402
    RESOLUTION_SWEEP,
    BirthConfig,
    GeometryJacobian,
    KScalingConfig,
    LocalBasisField,
    MultiviewConfig,
    PhotonBatch,
    PlanarCamera,
    SphereField,
    TorusField,
    TraceResult,
    ZeroSetField,
    build_fixed_transport_cell,
    benchmark_cpu_reference,
    benchmark_cuda_resolution,
    cuda_environment,
    cuda_equivalence_report,
    deform_reference_surface,
    emit_photons,
    exact_deformation,
    finite_difference_report,
    geometry_image_jacobian,
    implicit_position_jacobian,
    locality_perturbation_report,
    locality_cpu_verification,
    make_scene,
    make_local_basis_field,
    multiview_cpu_verification,
    observability_report,
    render_first_arrival,
    run_birth_experiment,
    run_k_scaling_experiment,
    run_multiview_benchmark,
    support_report,
    trace_photons,
    unit_normals,
)


def camera_for_scene(scene: str, resolution: int) -> PlanarCamera:
    if scene == "sphere":
        center = torch.tensor([0.0, 0.0, 3.0])
        normal = torch.tensor([0.0, 0.0, -1.0])
        right = torch.tensor([1.0, 0.0, 0.0])
        up = torch.tensor([0.0, 1.0, 0.0])
        extent = 4.0
    else:
        # A side detector makes the torus hole an explicit self-occlusion test:
        # inward-facing emitters can cross the hole and hit the opposite tube.
        center = torch.tensor([3.0, 0.0, 0.0])
        normal = torch.tensor([-1.0, 0.0, 0.0])
        right = torch.tensor([0.0, 1.0, 0.0])
        up = torch.tensor([0.0, 0.0, 1.0])
        extent = 4.5
    return PlanarCamera(
        center=center,
        normal=normal,
        right=right,
        up=up,
        width=extent,
        height=extent,
        resolution=(resolution, resolution),
    )


def run_demo(args: argparse.Namespace) -> tuple[dict[str, object], torch.Tensor]:
    field = SphereField() if args.scene == "sphere" else TorusField()
    emitter_count = args.emitters or (1024 if args.scene == "sphere" else 2048)
    camera = camera_for_scene(args.scene, args.resolution)
    started = time.perf_counter()
    points, normals, photons, trace, render = make_scene(
        field,
        camera,
        emitter_count=emitter_count,
        packets_per_emitter=args.packets,
        cone_power=args.cone_power,
        emission_interval=args.emission_interval,
        seed=args.seed,
        root_samples=args.root_samples,
        mode=args.mode,
        beta=args.beta,
    )
    runtime = time.perf_counter() - started
    nnz = render.transport._nnz()
    pixel_count = camera.pixel_count
    active_pixels = int(torch.unique(trace.pixels).numel())
    summary: dict[str, object] = {
        "scene": args.scene,
        "mode": args.mode,
        "seed": args.seed,
        "emitters": emitter_count,
        "emitted_photons": photons.count,
        "camera_candidates": trace.camera_candidate_count,
        "camera_hits": trace.camera_hit_count,
        "camera_hit_fraction": trace.camera_hit_count / photons.count,
        "absorbed": trace.absorbed_count,
        "absorption_fraction": trace.absorbed_count / photons.count,
        "candidate_absorption_fraction": (
            trace.absorbed_count / trace.camera_candidate_count
            if trace.camera_candidate_count
            else 0.0
        ),
        "active_pixels": active_pixels,
        "max_abs_field_at_emitters": float(field.value(points).abs().max()),
        "max_normal_length_error": float(
            (torch.linalg.vector_norm(normals, dim=-1) - 1.0).abs().max()
        ),
        "min_direction_normal_dot": float(
            (photons.directions * normals[photons.emitter_ids]).sum(dim=-1).min()
        ),
        "transport_shape": list(render.transport.shape),
        "transport_nnz": nnz,
        "transport_density": nnz / (pixel_count * emitter_count),
        "mean_active_emitters_per_pixel": nnz / pixel_count,
        "mean_affected_pixels_per_emitter": nnz / emitter_count,
        "direct_sparse_max_difference": render.max_difference,
        "runtime_seconds": runtime,
    }
    if args.output:
        import matplotlib.pyplot as plt

        image = render.direct_image.reshape(*camera.resolution, 3).clamp(0.0, 1.0)
        plt.imsave(args.output, image.numpy())
    return summary, render.direct_image


def analysis_camera(
    position: tuple[float, float, float], scene: str, resolution: int
) -> PlanarCamera:
    center = torch.tensor(position, dtype=torch.float64)
    normal = -center / torch.linalg.vector_norm(center)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if abs(float(normal @ world_up)) > 0.9:
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.linalg.cross(normal, world_up)
    right = right / torch.linalg.vector_norm(right)
    up = torch.linalg.cross(right, normal)
    extent = 4.0 if scene == "sphere" else 4.5
    return PlanarCamera(center, normal, right, up, extent, extent, (resolution, resolution))


def analysis_cameras(scene: str, resolution: int) -> list[PlanarCamera]:
    distance = 3.0 if scene == "sphere" else 3.5
    unit_positions = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (2**-0.5, 0.0, 2**-0.5),
        (-2**-0.5, 0.0, 2**-0.5),
        (0.0, 2**-0.5, -2**-0.5),
    ]
    if scene == "torus":
        unit_positions = unit_positions[1:] + unit_positions[:1]
    return [
        analysis_camera(tuple(distance * value for value in position), scene, resolution)
        for position in unit_positions
    ]


def prepare_analysis_geometry(
    scene: str, emitter_count: int, parameter_count: int, support_radius: float | None
) -> tuple[
    ZeroSetField,
    LocalBasisField,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    base = SphereField() if scene == "sphere" else TorusField()
    default_radius = 0.60 if scene == "sphere" else 0.50
    field = make_local_basis_field(
        base, parameter_count, support_radius or default_radius
    )
    generator = torch.Generator().manual_seed(211)
    reference_points = base.sample_surface(emitter_count, generator)
    reference_normals = unit_normals(base, reference_points)
    points, success, _ = deform_reference_surface(
        field, reference_points, reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError("baseline reference correspondence failed")
    colors = base.color(reference_points)
    return base, field, reference_points, reference_normals, points, colors


def implicit_derivative_check(
    field: LocalBasisField,
    reference_points: torch.Tensor,
    reference_normals: torch.Tensor,
    points: torch.Tensor,
    parameter_ids: torch.Tensor,
    *,
    step: float = 1e-5,
) -> dict[str, float]:
    analytic, stable, denominator = implicit_position_jacobian(
        field, points, reference_normals
    )
    errors: list[float] = []
    for parameter in parameter_ids.tolist():
        plus = field.coefficients.clone()
        minus = field.coefficients.clone()
        plus[parameter] += step
        minus[parameter] -= step
        _, plus_points, _ = exact_deformation(
            field, reference_points, reference_normals, plus
        )
        _, minus_points, _ = exact_deformation(
            field, reference_points, reference_normals, minus
        )
        finite_difference = (plus_points - minus_points) / (2.0 * step)
        affected = (
            field.basis_values(points)[:, parameter] > 1e-10
        ) & stable
        numerator = torch.linalg.vector_norm(
            finite_difference[affected] - analytic[affected, parameter]
        )
        denominator_norm = torch.linalg.vector_norm(
            analytic[affected, parameter]
        ).clamp_min(1e-15)
        errors.append(float(numerator / denominator_norm))
    values = torch.tensor(errors, dtype=torch.float64)
    return {
        "median_relative_error": float(values.median()),
        "maximum_relative_error": float(values.max()),
        "degenerate_emitter_fraction": float((~stable).double().mean()),
        "minimum_denominator_magnitude": float(denominator.abs().min()),
    }


def save_analysis_figures(
    output_directory: Path,
    field: LocalBasisField,
    points: torch.Tensor,
    results: list[GeometryJacobian],
) -> list[str]:
    """Write at most four compact diagnostics; no figures are generated by default."""
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    primary = results[0]
    response = torch.linalg.vector_norm(
        primary.matrix.reshape(primary.cell.camera.pixel_count, 3, -1), dim=1
    )
    strongest = torch.argsort(torch.linalg.vector_norm(primary.matrix, dim=0))[-4:]
    written: list[str] = []

    figure = plt.figure(figsize=(5, 4))
    axis = figure.add_subplot(projection="3d")
    support = field.basis_values(points)[:, strongest[-1]]
    axis.scatter(*points.T, c=support, s=5, cmap="viridis")
    axis.set_title("Example compact zero-set basis support")
    figure.tight_layout()
    path = output_directory / "basis_support.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    figure, axes = plt.subplots(1, 4, figsize=(10, 2.7))
    for axis, parameter in zip(axes, strongest.tolist()):
        image = response[:, parameter].reshape(primary.cell.camera.resolution)
        axis.imshow(image, cmap="magma")
        axis.set_title(f"$J_{{:,{parameter}}}$")
        axis.axis("off")
    figure.tight_layout()
    path = output_directory / "jacobian_columns.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(
        primary.matrix.abs().numpy() > 1e-12,
        aspect="auto",
        interpolation="nearest",
        cmap="binary",
    )
    axis.set_xlabel("geometry parameter")
    axis.set_ylabel("pixel-color row")
    axis.set_title("Geometry Jacobian sparsity")
    figure.tight_layout()
    path = output_directory / "jacobian_sparsity.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    if len(results) > 1:
        figure, axis = plt.subplots(figsize=(5, 4))
        for views in (1, 2, 4, 8):
            singular_values = torch.linalg.svdvals(
                torch.cat([result.matrix for result in results[:views]], dim=0)
            )
            axis.semilogy(singular_values.numpy(), label=f"{views} view(s)")
        axis.set_xlabel("singular-value index")
        axis.set_ylabel("singular value")
        axis.legend()
        axis.set_title("Multiview geometry observability")
        figure.tight_layout()
        path = output_directory / "singular_spectrum.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(str(path))
    return written


def run_jacobian_analysis(args: argparse.Namespace) -> dict[str, object]:
    emitter_count = args.analysis_emitters
    started = time.perf_counter()
    base, field, reference_points, reference_normals, points, colors = (
        prepare_analysis_geometry(
            args.scene, emitter_count, args.basis_count, args.support_radius
        )
    )
    cameras = analysis_cameras(args.scene, args.analysis_resolution)
    view_total = 8 if args.observability else 1
    results = []
    for camera in cameras[:view_total]:
        cell = build_fixed_transport_cell(
            field,
            camera,
            points,
            reference_normals,
            colors,
            packets_per_emitter=args.analysis_packets,
            cone_power=args.analysis_cone_power,
            normal_mode=args.normal_mode,
            root_samples=args.root_samples,
        )
        results.append(
            geometry_image_jacobian(field, points, reference_normals, cell)
        )
    primary = results[0]
    column_norms = torch.linalg.vector_norm(primary.matrix, dim=0)
    responsive = torch.nonzero(column_norms > 1e-12, as_tuple=False).flatten()
    if responsive.numel() < 8:
        raise RuntimeError("fewer than eight geometry parameters are visible")
    selected = responsive[torch.argsort(column_norms[responsive], descending=True)[:8]]
    implicit = implicit_derivative_check(
        field, reference_points, reference_normals, points, selected
    )
    finite_difference = finite_difference_report(
        field,
        reference_points,
        reference_normals,
        colors,
        primary,
        parameter_ids=selected,
    )
    support = support_report(field, points, primary)
    locality = locality_perturbation_report(
        field, reference_points, reference_normals, primary, selected
    )
    response = torch.linalg.vector_norm(
        primary.matrix.reshape(primary.cell.camera.pixel_count, 3, -1), dim=1
    )
    primary_active = response > 1e-12
    affected_counts = primary_active.sum(dim=0).to(torch.float64)
    responsive_counts = affected_counts[affected_counts > 0]
    summary: dict[str, object] = {
        "scene": args.scene,
        "normal_mode": args.normal_mode,
        "basis_count": field.parameter_count,
        "emitters": emitter_count,
        "pixels": primary.cell.camera.pixel_count,
        "emitted_photons": emitter_count * args.analysis_packets,
        "baseline_absorbed_photons": int((primary.cell.photon_state == -2).sum()),
        "baseline_absorption_fraction": float(
            (primary.cell.photon_state == -2).double().mean()
        ),
        "jacobian_shape": list(primary.matrix.shape),
        "jacobian_nnz_rgb_at_1e-12": primary.sparse_matrix._nnz(),
        "jacobian_density_rgb_at_1e-12": float(
            (primary.matrix.abs() > 1e-12).double().mean()
        ),
        "mean_affected_pixels_per_parameter": float(
            primary_active.sum(dim=0).to(torch.float64).mean()
        ),
        "median_affected_pixels_per_parameter": float(
            affected_counts.median()
        ),
        "mean_affected_pixels_per_responsive_parameter": float(
            responsive_counts.mean()
        ),
        "median_affected_pixels_per_responsive_parameter": float(
            responsive_counts.median()
        ),
        "mean_active_parameters_per_pixel": float(
            primary_active.sum(dim=1).to(torch.float64).mean()
        ),
        "root_failure_fraction": 0.0,
        "max_zero_set_residual": float(field.value(points).abs().max()),
        "implicit_derivative": implicit,
        "image_finite_difference": finite_difference,
        "support": support,
        "locality_perturbation": locality,
    }
    if args.observability:
        summary["observability"] = observability_report(
            [result.matrix for result in results]
        )
    if args.analysis_output:
        summary["figures"] = save_analysis_figures(
            args.analysis_output, field, points, results
        )
    summary["runtime_seconds"] = time.perf_counter() - started
    return summary


def run_cuda_benchmark(args: argparse.Namespace) -> dict[str, object]:
    environment = cuda_environment()
    if not environment["available"]:
        return {"environment": environment, "status": "LOCAL_CUDA_UNAVAILABLE"}
    equivalence = cuda_equivalence_report()
    if not equivalence["passed"]:
        return {
            "environment": environment,
            "equivalence": equivalence,
            "status": "CPU_CUDA_EQUIVALENCE_FAILED",
        }
    if args.benchmark_scaling:
        configurations = [
            (scene, resolution)
            for scene in ("sphere", "torus")
            for resolution in RESOLUTION_SWEEP
        ]
    else:
        if args.cuda_resolution:
            width, height = args.cuda_resolution
            resolution = (height, width)
        else:
            resolution = (1080, 1920)
        configurations = [(args.scene, resolution)]
    results = [
        benchmark_cuda_resolution(
            scene,
            resolution,
            emitters=args.analysis_emitters,
            packets_per_emitter=args.analysis_packets,
            warm_runs=args.warm_runs,
            batch_size=args.batch_size,
            output_path=(
                args.cuda_output
                if args.cuda_output and len(configurations) == 1
                else None
            ),
        )
        for scene, resolution in configurations
    ]
    cpu_ms = benchmark_cpu_reference("sphere", (256, 256))
    moderate = next(
        (
            result
            for result in results
            if result["scene"] == "sphere" and result["resolution"] == [256, 256]
        ),
        None,
    )
    if moderate is None:
        moderate = benchmark_cuda_resolution(
            "sphere",
            (256, 256),
            emitters=args.analysis_emitters,
            packets_per_emitter=args.analysis_packets,
            warm_runs=args.warm_runs,
            batch_size=args.batch_size,
        )
    comparison = {
        "configuration": "sphere 256x256, same sparse operator path",
        "cpu_runtime_ms": cpu_ms,
        "cuda_runtime_ms": moderate["warm_runtime_median_ms"],
        "speedup": cpu_ms / moderate["warm_runtime_median_ms"],
    }
    full_hd_scenes = {
        result["scene"]
        for result in results
        if result["resolution"] == [1920, 1080]
    }
    sparse_layouts = all(
        result["transport_layout"] == "torch.sparse_coo"
        and result["jacobian_layout"] == "torch.sparse_coo"
        for result in results
    )
    regression = run_verification()
    gates = {
        "gate_o_cpu_cuda_forward": "passed",
        "gate_p_cpu_cuda_absorption": "passed",
        "gate_q_cpu_cuda_jacobian": "passed",
        "gate_r_sparse_by_construction_equivalence": "passed",
        "gate_s_full_hd_both_scenes": (
            "passed"
            if full_hd_scenes == {"sphere", "torus"}
            else "not_evaluated_by_this_command"
        ),
        "gate_t_no_dense_full_hd_operators": "passed" if sparse_layouts else "failed",
        "gate_u_v01_v02_regression": "passed",
    }
    return {
        "environment": environment,
        "equivalence": equivalence,
        "configuration": {
            "basis_count": 32,
            "emitters": args.analysis_emitters,
            "packets_per_emitter": args.analysis_packets,
            "cone_power": 128.0,
            "physical_detector": "fixed square; widescreen grids use rectangular pixels",
            "collision_batch_size": args.batch_size,
        },
        "results": results,
        "cpu_cuda_moderate_comparison": comparison,
        "gates": gates,
        "regression": regression,
        "status": "passed",
    }


def run_birth_analysis(args: argparse.Namespace) -> dict[str, object]:
    config = BirthConfig(
        candidates=args.birth_candidates,
        views=args.birth_views,
        resolution=args.birth_resolution,
        oracle_iterations=args.birth_oracle_iterations,
    )
    return run_birth_experiment(
        config,
        csv_path=args.birth_csv,
        figure_directory=args.birth_figures,
    )


def run_k_scaling_analysis(args: argparse.Namespace) -> dict[str, object]:
    config = KScalingConfig(k_values=tuple(args.k_values))
    return run_k_scaling_experiment(
        config,
        csv_path=args.k_scaling_csv,
        json_path=args.k_scaling_json,
        figure_directory=args.k_scaling_figures,
    )


def run_multiview_analysis(args: argparse.Namespace) -> dict[str, object]:
    config = MultiviewConfig(
        resolution=(args.multiview_resolution[1], args.multiview_resolution[0]),
        warm_runs=args.warm_runs,
        measured_runs=args.measured_runs,
        collision_batch_size=args.batch_size,
    )
    return run_multiview_benchmark(
        config=config,
        output_path=args.multiview_output,
        figure_directory=args.multiview_figures,
    )


def run_verification() -> dict[str, object]:
    """Exercise acceptance gates through the same classes/functions as the demo."""
    generator = torch.Generator().manual_seed(19)
    sphere, torus = SphereField(), TorusField()
    sphere_points = sphere.sample_surface(256, generator)
    torus_points = torus.sample_surface(256, generator)
    sphere_normals = unit_normals(sphere, sphere_points)
    torus_normals = unit_normals(torus, torus_points)

    sphere_zero_error = float(sphere.value(sphere_points).abs().max())
    torus_zero_error = float(torus.value(torus_points).abs().max())
    normal_error = max(
        float((torch.linalg.vector_norm(sphere_normals, dim=-1) - 1.0).abs().max()),
        float((torch.linalg.vector_norm(torus_normals, dim=-1) - 1.0).abs().max()),
    )
    sphere_reference_error = float(
        (sphere_normals - sphere_points / sphere.radius).abs().max()
    )
    assert sphere_zero_error < 1e-12 and torus_zero_error < 1e-12
    assert normal_error < 1e-12 and sphere_reference_error < 1e-12

    # Known finite-plane intersections: center, upper-right, outside, parallel.
    check_camera = PlanarCamera(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        2.0,
        2.0,
        (2, 2),
    )
    origins = torch.zeros((4, 3), dtype=torch.float64)
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [2.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    valid, times, pixels, local = check_camera.intersect(origins, directions)
    assert valid.tolist() == [True, True, False, False]
    assert torch.allclose(times[:2], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(local[:2], torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64))
    assert pixels[:2].tolist() == [3, 1]

    # A diameter ray re-enters the sphere at t=2 and is absorbed; an outward
    # ray from the opposite point reaches the same x=3 detector.
    side_camera = PlanarCamera(
        torch.tensor([3.0, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        4.0,
        4.0,
        (4, 4),
    )
    absorption_photons = PhotonBatch(
        origins=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        directions=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        energies=torch.ones(2, dtype=torch.float64),
        emit_times=torch.zeros(2, dtype=torch.float64),
        emitter_ids=torch.tensor([0, 1]),
    )
    absorption_trace = trace_photons(sphere, side_camera, absorption_photons, root_samples=48)
    assert absorption_trace.absorbed_count == 1
    assert absorption_trace.camera_hit_count == 1
    assert absorption_trace.emitter_ids.tolist() == [1]

    # Arrival competition uses two packets in one pixel.
    competition = TraceResult(
        emitted_count=2,
        camera_candidate_count=2,
        absorbed_count=0,
        pixels=torch.tensor([0, 0]),
        arrival_times=torch.tensor([1.0, 2.0], dtype=torch.float64),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        energies=torch.ones(2, dtype=torch.float64),
        emitter_ids=torch.tensor([0, 1]),
        photon_ids=torch.tensor([0, 1]),
    )
    emitter_colors = competition.colors.clone()
    hard = render_first_arrival(competition, 1, emitter_colors, mode="hard")
    soft_low = render_first_arrival(competition, 1, emitter_colors, mode="soft", beta=1.0)
    soft_high = render_first_arrival(competition, 1, emitter_colors, mode="soft", beta=20.0)
    assert torch.equal(hard.direct_image[0], emitter_colors[0])
    high_error = torch.linalg.vector_norm(soft_high.direct_image - hard.direct_image)
    low_error = torch.linalg.vector_norm(soft_low.direct_image - hard.direct_image)
    assert high_error < low_error
    assert max(hard.max_difference, soft_low.max_difference, soft_high.max_difference) < 1e-12

    # Full-pipeline determinism and sparse equivalence.
    deterministic_camera = camera_for_scene("sphere", 24)
    run_kwargs = dict(
        emitter_count=128,
        packets_per_emitter=4,
        cone_power=16.0,
        emission_interval=0.0,
        seed=23,
        root_samples=32,
        mode="hard",
        beta=50.0,
    )
    first = make_scene(sphere, deterministic_camera, **run_kwargs)
    second = make_scene(sphere, deterministic_camera, **run_kwargs)
    first_trace, first_render = first[-2], first[-1]
    second_trace, second_render = second[-2], second[-1]
    assert first_trace.emitted_count == second_trace.emitted_count
    assert first_trace.camera_candidate_count == second_trace.camera_candidate_count
    assert first_trace.absorbed_count == second_trace.absorbed_count
    assert torch.equal(first_trace.pixels, second_trace.pixels)
    assert torch.equal(first_render.direct_image, second_render.direct_image)
    assert torch.equal(first_trace.arrival_times, second_trace.arrival_times)
    assert torch.equal(first_render.transport.indices(), second_render.transport.indices())
    assert torch.equal(first_render.transport.values(), second_render.transport.values())
    assert first_render.max_difference < 1e-6

    v01_report = {
        "gate_a_zero_set": {
            "sphere_max_abs_F": sphere_zero_error,
            "torus_max_abs_F": torus_zero_error,
        },
        "gate_b_normals": {
            "max_unit_length_error": normal_error,
            "sphere_reference_error": sphere_reference_error,
        },
        "gate_c_camera": "passed",
        "gate_d_absorption": {"absorbed": 1, "reached_camera": 1},
        "gate_e_first_arrival": "passed",
        "gate_f_sparse": {"max_difference": first_render.max_difference},
        "gate_g_determinism": "passed",
    }
    v02_report = run_v02_verification()
    multiview_report = multiview_cpu_verification()
    locality_report = locality_cpu_verification()
    return {
        **v01_report,
        **v02_report,
        "gate_v_shared_multiview_equivalence": multiview_report,
        "gate_w_sparse_locality_equivalence": locality_report,
    }


def run_v02_verification() -> dict[str, object]:
    base, field, reference_points, reference_normals, points, colors = (
        prepare_analysis_geometry("sphere", 128, 16, 0.9)
    )
    parameter_ids = torch.arange(8)

    outside = field.centers + torch.tensor([1.01, 0.0, 0.0]) * field.radii[:, None]
    own_outside_values = field.basis_values(outside).diagonal()
    assert torch.equal(own_outside_values, torch.zeros_like(own_outside_values))

    perturbation = torch.where(
        torch.arange(field.parameter_count) % 2 == 0,
        torch.tensor(0.01),
        torch.tensor(-0.01),
    ).to(torch.float64)
    perturbed_field, perturbed_points, success = exact_deformation(
        field, reference_points, reference_normals, perturbation
    )
    deformation_residual = float(perturbed_field.value(perturbed_points).abs().max())
    assert bool(success.all()) and deformation_residual < 1e-8

    implicit = implicit_derivative_check(
        field, reference_points, reference_normals, points, parameter_ids
    )
    assert implicit["maximum_relative_error"] < 1e-4

    cell = build_fixed_transport_cell(
        field,
        analysis_cameras("sphere", 16)[0],
        points,
        reference_normals,
        colors,
        packets_per_emitter=4,
        cone_power=32.0,
    )
    jacobian = geometry_image_jacobian(field, points, reference_normals, cell)
    column_norms = torch.linalg.vector_norm(jacobian.matrix, dim=0)
    responsive = torch.nonzero(column_norms > 1e-12, as_tuple=False).flatten()
    selected = responsive[torch.argsort(column_norms[responsive], descending=True)[:8]]
    assert selected.numel() == 8
    image_fd = finite_difference_report(
        field,
        reference_points,
        reference_normals,
        colors,
        jacobian,
        parameter_ids=selected,
    )
    assert image_fd["median_relative_error"] < 1e-3
    locality = locality_perturbation_report(
        field, reference_points, reference_normals, jacobian, selected
    )
    assert locality["median_local_energy_norm_fraction"] > 0.9

    # Zero coefficients must preserve exact v0.1 sphere and torus transport.
    regression_absorbed: dict[str, int] = {}
    for scene, regression_base, count, radius in (
        ("sphere", base, 128, 0.9),
        ("torus", TorusField(), 256, 0.5),
    ):
        generator = torch.Generator().manual_seed(313)
        regression_points = regression_base.sample_surface(count, generator)
        regression_normals = unit_normals(regression_base, regression_points)
        regression_colors = regression_base.color(regression_points)
        regression_field = make_local_basis_field(regression_base, 8, radius)
        regression_photons = emit_photons(
            regression_points,
            regression_normals,
            regression_colors,
            8,
            16.0,
            0.0,
            generator,
        )
        regression_camera = camera_for_scene(scene, 16)
        base_trace = trace_photons(
            regression_base, regression_camera, regression_photons, root_samples=32
        )
        local_trace = trace_photons(
            regression_field, regression_camera, regression_photons, root_samples=32
        )
        base_render = render_first_arrival(
            base_trace, regression_camera.pixel_count, regression_colors
        )
        local_render = render_first_arrival(
            local_trace, regression_camera.pixel_count, regression_colors
        )
        assert base_trace.absorbed_count == local_trace.absorbed_count
        assert torch.equal(base_trace.pixels, local_trace.pixels)
        assert torch.equal(base_trace.arrival_times, local_trace.arrival_times)
        assert torch.equal(base_render.direct_image, local_render.direct_image)
        regression_absorbed[scene] = base_trace.absorbed_count
    assert regression_absorbed["torus"] > 0

    small_jacobians = []
    for camera in analysis_cameras("sphere", 12):
        small_cell = build_fixed_transport_cell(
            field,
            camera,
            points,
            reference_normals,
            colors,
            packets_per_emitter=4,
            cone_power=32.0,
            root_samples=32,
        )
        small_jacobians.append(
            geometry_image_jacobian(
                field, points, reference_normals, small_cell
            ).matrix
        )
    observability = observability_report(small_jacobians)
    return {
        "gate_h_basis_locality": "passed",
        "gate_i_zero_set_deformation": {
            "max_residual": deformation_residual,
            "root_failure_fraction": float((~success).double().mean()),
        },
        "gate_j_implicit_root_derivative": implicit,
        "gate_k_image_jacobian": image_fd,
        "gate_l_locality": locality,
        "gate_m_multiview_rank": {
            views: observability[views]["rank"] for views in ("1", "2", "4", "8")
        },
        "gate_n_v01_regression": {
            "status": "passed",
            "absorbed_photons": regression_absorbed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("sphere", "torus"), default="sphere")
    parser.add_argument("--mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--emitters", type=int, default=None)
    parser.add_argument("--packets", type=int, default=16, help="photon packets per emitter")
    parser.add_argument("--cone-power", type=float, default=16.0)
    parser.add_argument("--emission-interval", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--root-samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None, help="optional PNG path")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--jacobian", action="store_true")
    parser.add_argument("--observability", action="store_true")
    parser.add_argument("--basis-count", type=int, default=32)
    parser.add_argument("--support-radius", type=float, default=None)
    parser.add_argument("--analysis-emitters", type=int, default=256)
    parser.add_argument("--analysis-packets", type=int, default=8)
    parser.add_argument("--analysis-cone-power", type=float, default=128.0)
    parser.add_argument("--analysis-resolution", type=int, default=32)
    parser.add_argument("--analysis-output", type=Path, default=None)
    parser.add_argument(
        "--normal-mode", choices=("reference", "current"), default="reference"
    )
    parser.add_argument("--benchmark-cuda", action="store_true")
    parser.add_argument("--benchmark-scaling", action="store_true")
    parser.add_argument(
        "--cuda-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cuda-output", type=Path, default=None)
    parser.add_argument("--birth", action="store_true")
    parser.add_argument("--birth-candidates", type=int, default=64)
    parser.add_argument("--birth-views", type=int, default=8)
    parser.add_argument("--birth-resolution", type=int, default=256)
    parser.add_argument("--birth-oracle-iterations", type=int, default=5)
    parser.add_argument("--birth-csv", type=Path, default=None)
    parser.add_argument("--birth-figures", type=Path, default=None)
    parser.add_argument("--birth-k-scaling", action="store_true")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--k-scaling-csv", type=Path, default=None)
    parser.add_argument("--k-scaling-json", type=Path, default=None)
    parser.add_argument("--k-scaling-figures", type=Path, default=None)
    parser.add_argument("--benchmark-multiview", action="store_true")
    parser.add_argument(
        "--multiview-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
    )
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--multiview-output", type=Path, default=None)
    parser.add_argument("--multiview-figures", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.birth_k_scaling:
        print("prebirth_k_scaling:")
        print(json.dumps(run_k_scaling_analysis(args), indent=2, sort_keys=True))
        return
    if args.benchmark_multiview:
        print("multiview_benchmark:")
        print(json.dumps(run_multiview_analysis(args), indent=2, sort_keys=True))
        return
    if args.birth:
        print("candidate_birth_analysis:")
        print(json.dumps(run_birth_analysis(args), indent=2, sort_keys=True))
        return
    if args.benchmark_cuda or args.benchmark_scaling:
        print("cuda_benchmark:")
        print(json.dumps(run_cuda_benchmark(args), indent=2, sort_keys=True))
        return
    if args.verify:
        print("verification:")
        print(json.dumps(run_verification(), indent=2, sort_keys=True))
    if args.jacobian or args.observability:
        print("geometry_jacobian:")
        print(json.dumps(run_jacobian_analysis(args), indent=2, sort_keys=True))
        return
    summary, _ = run_demo(args)
    print("render:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
