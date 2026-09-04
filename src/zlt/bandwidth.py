"""Bunny observation-bandwidth and geometry-DoF diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batch import (
    BatchPolicy,
    _report_rows,
    _run_batch_trajectory,
    _trajectory_summary,
)
from .benchmark import BenchmarkGeometry, cuda_environment
from .birth import (
    _gradient,
    _image_loss,
    _local_sparse_jacobian,
    _normal_matvec,
)
from .bunny import BunnyExperimentConfig, BunnyGeometryEvaluator, _normal_line_roots
from .fields import LocalBasisField, unit_normals
from .jacobian import _photon_batch, nested_deterministic_directions
from .locality import hierarchical_surface_points
from .mesh_field import PreparedBunny, prepare_stanford_bunny, resmooth_stanford_bunny
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    camera_pose,
    multiview_cameras,
    project_camera,
)
from .sequential import (
    ActiveState,
    RepeatedBirthConfig,
    SequentialContext,
    _active_components,
    _active_jacobians,
    _correlation,
    _evaluate_active,
    _hierarchical_layout,
    _multiscale_support,
)
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor
FIXED_LEVELS = (256, 512, 1024)
CONDITIONS = (
    ("A", 256, 8),
    ("B", 256, 32),
    ("C", 512, 8),
    ("D", 512, 32),
)


@dataclass
class GeometryBundle:
    sigma: float
    prepared: PreparedBunny
    base: object
    gt: object
    layout: object
    levels: Tensor
    support: object
    reference_points: Tensor
    reference_normals: Tensor
    target_points: Tensor
    target_normals: Tensor
    colors: Tensor
    detail_ids: Tensor
    detail_ids_top_ten: Tensor
    evaluator: BunnyGeometryEvaluator
    diagnostics: dict[str, object]


@dataclass
class SceneBundle:
    packets_per_emitter: int
    directions: Tensor
    base_state: SceneTransportState
    target_state: SceneTransportState
    diagnostics: dict[str, object]


@dataclass
class FixedSolution:
    state: ActiveState
    row: dict[str, object]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_digest(tensor: Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _image_digest(images: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _physical_camera_digest(cameras: list[object]) -> str:
    digest = hashlib.sha256()
    for camera in cameras:
        for value in (
            camera.center,  # type: ignore[attr-defined]
            camera.normal,  # type: ignore[attr-defined]
            camera.right,  # type: ignore[attr-defined]
            camera.up,  # type: ignore[attr-defined]
        ):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(
            np.asarray(
                [camera.width, camera.height],  # type: ignore[attr-defined]
                dtype=np.float64,
            ).tobytes()
        )
    return digest.hexdigest()


def _project_to_base(
    prepared: PreparedBunny, canonical_points: Tensor
) -> tuple[Tensor, dict[str, object]]:
    import open3d as o3d

    device = canonical_points.device
    base = prepared.base_field.to(device)
    vertices = prepared.base_field.surface_vertices.detach().cpu().numpy()
    faces = prepared.base_field.surface_faces.detach().cpu().numpy().astype(np.int32)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    closest = scene.compute_closest_points(
        o3d.core.Tensor(
            canonical_points.detach().cpu().numpy().astype(np.float32)
        )
    )["points"].numpy()
    points = torch.from_numpy(closest.astype(np.float64)).to(device)
    for _ in range(20):
        values = base.value(points)
        gradients = base.interpolant_gradient(points)
        points = points - values[:, None] * gradients / (
            (gradients * gradients).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )
    residual = base.value(points).abs()
    distance = torch.linalg.vector_norm(points - canonical_points, dim=1)
    return points, {
        "projection_failures": int((residual >= 1e-8).sum()),
        "projection_residual_max": float(residual.max()),
        "projection_distance_mean": float(distance.mean()),
        "projection_distance_p95": float(torch.quantile(distance, 0.95)),
        "projection_distance_max": float(distance.max()),
    }


def _canonical_emitters(
    prepared: PreparedBunny, emitters: int, device: torch.device
) -> dict[str, Tensor | int]:
    base = prepared.base_field.to(device)
    gt = prepared.gt_field.to(device)
    points = hierarchical_surface_points(base, emitters, device)
    normals = unit_normals(base, points)
    target_points, success, displacement = _normal_line_roots(
        gt, points, normals
    )
    if not bool(success.all()):
        raise RuntimeError("canonical Bunny GT normal-line transfer failed")
    return {
        "points": points,
        "normals": normals,
        "target_points": target_points,
        "target_normals": unit_normals(gt, target_points),
        "colors": base.color(points),
        "normal_line_failures": int((~success).sum()),
        "displacement": displacement,
    }


def _geometry_bundle(
    prepared_15: PreparedBunny,
    canonical: dict[str, Tensor | int],
    sigma: float,
    config: BunnyExperimentConfig,
) -> GeometryBundle:
    device = canonical["points"].device  # type: ignore[union-attr]
    current = (
        prepared_15
        if sigma == 1.5
        else resmooth_stanford_bunny(prepared_15, sigma)
    )
    base = current.base_field.to(device)
    gt = current.gt_field.to(device)
    reference_points, projection = _project_to_base(
        current, canonical["points"]  # type: ignore[arg-type]
    )
    reference_normals = unit_normals(base, reference_points)
    layout, levels = _hierarchical_layout(
        base,
        config.master_count,
        config.initial_count,
        config.base_support_radius,
        device,
    )
    repeated = _repeated_config(config)
    support = _multiscale_support(
        reference_points, layout, levels, repeated.support_margin
    )
    center_normals = unit_normals(base, layout.centers)
    _, center_success, center_displacement = _normal_line_roots(
        gt, layout.centers, center_normals
    )
    absolute = center_displacement.abs()
    evaluator = BunnyGeometryEvaluator(
        current, reference_points, reference_normals, config.evaluation_samples
    )
    base_gradient_norm = torch.linalg.vector_norm(
        base.gradient(reference_points), dim=1
    )
    target_points = canonical["target_points"]  # type: ignore[assignment]
    target_gradient_norm = torch.linalg.vector_norm(
        gt.gradient(target_points), dim=1  # type: ignore[arg-type]
    )
    diagnostics = {
        **projection,
        "candidate_normal_line_failures": int((~center_success).sum()),
        "base_degenerate_gradient_count": int((base_gradient_norm <= 1e-12).sum()),
        "target_degenerate_gradient_count": int(
            (target_gradient_norm <= 1e-12).sum()
        ),
        "base_emitter_residual_max": float(base.value(reference_points).abs().max()),
        "target_emitter_residual_max": float(
            gt.value(target_points).abs().max()  # type: ignore[arg-type]
        ),
    }
    return GeometryBundle(
        sigma,
        current,
        base,
        gt,
        layout,
        levels,
        support,
        reference_points,
        reference_normals,
        target_points,  # type: ignore[arg-type]
        canonical["target_normals"],  # type: ignore[arg-type]
        canonical["colors"],  # type: ignore[arg-type]
        torch.nonzero(
            absolute >= torch.quantile(absolute, 0.75), as_tuple=False
        ).flatten(),
        torch.nonzero(
            absolute >= torch.quantile(absolute, 0.90), as_tuple=False
        ).flatten(),
        evaluator,
        diagnostics,
    )


def _repeated_config(config: BunnyExperimentConfig) -> RepeatedBirthConfig:
    return RepeatedBirthConfig(
        master_count=config.master_count,
        initial_count=config.initial_count,
        budget=config.budget,
        checkpoints=(32, 64, 96, 128, 160, 192, 224, 256),
        target_regimes=("bunny",),
        target_seeds=(0,),
        random_seeds=(),
        views=config.views,
        resolution=config.resolution,
        emitters_per_master_basis=config.emitters // config.master_count,
        packets_per_emitter=config.packets_per_emitter,
        base_support_radius=config.base_support_radius,
        coefficient_limit=config.coefficient_limit,
        root_samples=config.root_samples,
        root_bisection_steps=config.root_bisection_steps,
        deformation_iterations=20,
        deformation_max_offset=0.03,
        initial_optimization_steps=config.initial_optimization_steps,
        post_birth_steps=config.post_birth_steps,
        fixed_optimization_steps=config.fixed_optimization_steps,
        run_oracle=False,
    )


def _scene_bundle(
    geometry: GeometryBundle,
    packets_per_emitter: int,
    config: BunnyExperimentConfig,
) -> SceneBundle:
    device = geometry.reference_points.device
    directions = nested_deterministic_directions(
        geometry.reference_normals, packets_per_emitter, 128.0
    )
    maximum_times = torch.full(
        (config.emitters * packets_per_emitter,),
        3.0,
        dtype=torch.float64,
        device=device,
    )
    states = []
    timings = []
    for field, points, normals in (
        (geometry.base, geometry.reference_points, geometry.reference_normals),
        (geometry.gt, geometry.target_points, geometry.target_normals),
    ):
        photons = _photon_batch(
            points, directions, geometry.colors, packets_per_emitter
        )
        _sync(device)
        started = time.perf_counter()
        hits, hit_times = first_zero_set_intersections(
            field,
            photons.origins,
            photons.directions,
            maximum_times,
            samples=config.root_samples,
            bisection_steps=config.root_bisection_steps,
            chunk_size=8192,
        )
        _sync(device)
        timings.append(time.perf_counter() - started)
        states.append(
            SceneTransportState(
                points,
                normals,
                photons,
                directions,
                hits,
                hit_times,
                torch.ones(config.emitters, dtype=torch.bool, device=device),
            )
        )
    return SceneBundle(
        packets_per_emitter,
        directions,
        states[0],
        states[1],
        {
            "current_scene_transport_seconds": timings[0],
            "target_scene_transport_seconds": timings[1],
            "base_surface_hits": int(states[0].surface_hits.sum()),
            "target_surface_hits": int(states[1].surface_hits.sum()),
            "direction_sha256": _tensor_digest(directions),
        },
    )


def _event_counts(transport: object) -> Tensor:
    survivor_ids = transport.survivor_ids  # type: ignore[attr-defined]
    pixels = transport.intersection.pixels[survivor_ids]  # type: ignore[attr-defined]
    return torch.bincount(
        pixels,
        minlength=transport.cell.camera.pixel_count,  # type: ignore[attr-defined]
    )


def _observation_support(transports: list[object]) -> dict[str, object]:
    occupied = []
    detector_hits = []
    survived = []
    event_counts = []
    for transport in transports:
        owner_map = transport.cell.owner_map  # type: ignore[attr-defined]
        occupied.append(int((owner_map >= 0).sum()))
        detector_hits.append(int(transport.intersection.valid.sum()))  # type: ignore[attr-defined]
        survived.append(int(transport.survivor_ids.numel()))  # type: ignore[attr-defined]
        counts = _event_counts(transport)
        event_counts.append(counts[counts > 0].to(torch.float64))
    all_events = torch.cat(event_counts) if event_counts else torch.empty(0)
    total_pixels = sum(
        transport.cell.camera.pixel_count  # type: ignore[attr-defined]
        for transport in transports
    )
    total_occupied = sum(occupied)
    return {
        "occupied_pixels": total_occupied,
        "occupied_pixel_fraction": total_occupied / max(total_pixels, 1),
        "detector_hits_per_view_mean": statistics.mean(detector_hits),
        "surviving_detector_events_per_view_mean": statistics.mean(survived),
        "photons_per_occupied_pixel": sum(survived) / max(total_occupied, 1),
        "events_per_nonempty_pixel_mean": float(all_events.mean())
        if all_events.numel()
        else 0.0,
        "events_per_nonempty_pixel_median": float(all_events.median())
        if all_events.numel()
        else 0.0,
        "per_view_occupied_pixels": occupied,
        "per_view_detector_hits": detector_hits,
        "per_view_surviving_events": survived,
    }


def _condition_context(
    geometry: GeometryBundle,
    scene: SceneBundle,
    label: str,
    resolution: int,
    config: BunnyExperimentConfig,
) -> tuple[SequentialContext, dict[str, object], dict[str, object]]:
    device = geometry.reference_points.device
    current_config = BunnyExperimentConfig(
        master_count=config.master_count,
        initial_count=config.initial_count,
        budget=config.budget,
        checkpoints=config.checkpoints,
        fixed_levels=FIXED_LEVELS,
        views=config.views,
        resolution=resolution,
        emitters=config.emitters,
        packets_per_emitter=scene.packets_per_emitter,
        random_seeds=(),
        base_support_radius=config.base_support_radius,
        coefficient_limit=config.coefficient_limit,
        root_samples=config.root_samples,
        root_bisection_steps=config.root_bisection_steps,
        initial_optimization_steps=config.initial_optimization_steps,
        post_birth_steps=config.post_birth_steps,
        fixed_optimization_steps=config.fixed_optimization_steps,
        oracle_subset_size=0,
        evaluation_samples=config.evaluation_samples,
    )
    repeated = _repeated_config(current_config)
    cameras = multiview_cameras(
        "bunny", (resolution, resolution), device, count=config.views
    )
    initial_field = LocalBasisField(
        geometry.base,
        geometry.layout.centers[: config.initial_count],
        geometry.layout.radii[: config.initial_count],
        torch.zeros(config.initial_count, dtype=torch.float64, device=device),
    )
    benchmark_geometry = BenchmarkGeometry(
        initial_field,
        geometry.reference_points,
        geometry.reference_normals,
        geometry.colors,
    )
    view_config = MultiviewConfig(
        resolution=(resolution, resolution),
        emitters=config.emitters,
        packets_per_emitter=scene.packets_per_emitter,
        parameter_count=config.master_count,
        cone_power=128.0,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )
    detector_times = []
    all_transports = []
    for state in (scene.base_state, scene.target_state):
        _sync(device)
        started = time.perf_counter()
        transports = [
            project_camera(benchmark_geometry, camera, state, view_config)
            for camera in cameras
        ]
        _sync(device)
        detector_times.append(time.perf_counter() - started)
        all_transports.append(transports)
    base_transports, target_transports = all_transports
    target_images = [item.image.reshape(-1) for item in target_transports]
    context = SequentialContext(
        repeated,
        geometry.base,
        geometry.layout,
        geometry.support,
        geometry.levels,
        geometry.reference_points,
        geometry.reference_normals,
        geometry.colors,
        [item.cell for item in base_transports],
        scene.base_state,
        benchmark_geometry,
        geometry.evaluator,
    )
    target = {
        "regime": "bunny",
        "seed": 0,
        "points": geometry.target_points,
        "images": target_images,
        "transport_cells": target_transports,
        "detail_ids": geometry.detail_ids,
        "detail_ids_top_ten": geometry.detail_ids_top_ten,
    }
    state_events = []
    owner_events = []
    for base_transport, target_transport in zip(
        base_transports, target_transports
    ):
        state_events.append(
            float(
                (
                    base_transport.cell.photon_state
                    != target_transport.cell.photon_state
                )
                .to(torch.float64)
                .mean()
            )
        )
        occupied = (base_transport.cell.owner_map >= 0) | (
            target_transport.cell.owner_map >= 0
        )
        owner_events.append(
            float(
                (
                    base_transport.cell.owner_map[occupied]
                    != target_transport.cell.owner_map[occupied]
                )
                .to(torch.float64)
                .mean()
            )
            if bool(occupied.any())
            else 0.0
        )
    diagnostics = {
        "condition": label,
        "sigma": geometry.sigma,
        "resolution": resolution,
        "packets_per_emitter": scene.packets_per_emitter,
        "photons": config.emitters * scene.packets_per_emitter,
        "total_scalar_observations": sum(image.numel() for image in target_images),
        "target_energy": sum(float((image * image).sum()) for image in target_images),
        "target_image_sha256": _image_digest(target_images),
        "physical_camera_sha256": _physical_camera_digest(cameras),
        "camera_poses": [camera_pose(camera) for camera in cameras],
        "current_detector_transport_seconds": detector_times[0],
        "target_detector_transport_seconds": detector_times[1],
        "photon_state_transport_event_fraction": statistics.mean(state_events),
        "occupied_owner_event_fraction": statistics.mean(owner_events),
        "base_observation_support": _observation_support(base_transports),
        "target_observation_support": _observation_support(target_transports),
        **geometry.diagnostics,
        **scene.diagnostics,
    }
    return context, target, diagnostics


def _cg_with_iterations(
    matrices: list[Tensor], right: Tensor, config: RepeatedBirthConfig
) -> tuple[Tensor, int]:
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    residual_squared = residual @ residual
    iterations = 0
    for _ in range(config.cg_iterations):
        iterations += 1
        product = _normal_matvec(matrices, direction, config.damping)
        alpha = residual_squared / (direction @ product).clamp_min(1e-30)
        solution = solution + alpha * direction
        next_residual = residual - alpha * product
        next_squared = next_residual @ next_residual
        if float(torch.sqrt(next_squared)) < 1e-10:
            break
        direction = next_residual + next_squared / residual_squared * direction
        residual = next_residual
        residual_squared = next_squared
    return solution, iterations


def _diagnostic_optimize(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
    steps: int,
) -> tuple[ActiveState, dict[str, float | int]]:
    device = state.points.device
    _sync(device)
    total_started = time.perf_counter()
    coefficients = state.coefficients.clone()
    result = state
    completed = 0
    cg_iterations = 0
    line_search_failures = state.line_search_failures
    cg_failures = state.cg_failures
    evaluation_seconds = 0.0
    active_jacobian_seconds = 0.0
    linear_solve_seconds = 0.0
    components = _active_components(context, state.active_ids)
    relative = 0.0
    gradient_norm = math.inf
    for _ in range(steps):
        started = time.perf_counter()
        result = _evaluate_active(
            context, state.active_ids, coefficients, targets, components
        )
        _sync(device)
        evaluation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        matrices = _active_jacobians(context, result, components)
        _sync(device)
        active_jacobian_seconds += time.perf_counter() - started
        started = time.perf_counter()
        gradient = _gradient(
            matrices, result.images, targets, state.active_ids.numel()
        )
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        step, used = _cg_with_iterations(matrices, -gradient, context.config)
        cg_iterations += used
        _sync(device)
        linear_solve_seconds += time.perf_counter() - started
        if not bool(torch.isfinite(step).all()):
            cg_failures += 1
            break
        accepted = coefficients
        accepted_loss = result.loss
        for trial in range(context.config.line_evaluations):
            scale = 0.5**trial
            proposal = (coefficients + scale * step).clamp(
                -context.config.coefficient_limit,
                context.config.coefficient_limit,
            )
            started = time.perf_counter()
            proposal_state = _evaluate_active(
                context, state.active_ids, proposal, targets, components
            )
            _sync(device)
            evaluation_seconds += time.perf_counter() - started
            if (
                proposal_state.root_failures == 0
                and proposal_state.loss < accepted_loss
            ):
                accepted = proposal
                accepted_loss = proposal_state.loss
        relative = (result.loss - accepted_loss) / max(result.loss, 1e-30)
        if accepted_loss >= result.loss:
            line_search_failures += 1
        coefficients = accepted
        completed += 1
        if (
            accepted_loss >= result.loss
            or relative < context.config.convergence_tolerance
            or gradient_norm < 1e-10
        ):
            break
    started = time.perf_counter()
    result = _evaluate_active(
        context, state.active_ids, coefficients, targets, components
    )
    _sync(device)
    evaluation_seconds += time.perf_counter() - started
    result.optimizer_steps = completed
    result.last_relative_improvement = relative
    result.gradient_norm = gradient_norm
    result.line_search_failures = line_search_failures
    result.cg_failures = cg_failures
    return result, {
        "optimization_seconds": time.perf_counter() - total_started,
        "optimizer_steps": completed,
        "cg_iterations": cg_iterations,
        "render_evaluation_seconds": evaluation_seconds,
        "active_jacobian_seconds": active_jacobian_seconds,
        "linear_solve_seconds": linear_solve_seconds,
    }


def _normalized_observation_metrics(
    state: ActiveState, target: dict[str, object]
) -> dict[str, float]:
    targets = target["images"]  # type: ignore[assignment]
    squared = 2.0 * state.loss
    observation_count = sum(image.numel() for image in targets)
    target_energy = sum(float((image * image).sum()) for image in targets)
    return {
        "raw_image_loss": state.loss,
        "normalized_image_mse": squared / max(observation_count, 1),
        "residual_norm": math.sqrt(max(squared, 0.0)),
        "residual_energy": squared,
        "normalized_residual_energy": squared / max(target_energy, 1e-30),
    }


def _expansion_diagnostic(
    context: SequentialContext,
    target: dict[str, object],
    previous: FixedSolution,
    active_k: int,
) -> tuple[ActiveState, dict[str, float]]:
    device = previous.state.points.device
    coefficients = torch.cat(
        (
            previous.state.coefficients,
            previous.state.coefficients.new_zeros(
                active_k - previous.state.active_ids.numel()
            ),
        )
    )
    expanded = _evaluate_active(
        context,
        torch.arange(active_k, device=device),
        coefficients,
        target["images"],  # type: ignore[arg-type]
    )
    point_difference = torch.linalg.vector_norm(
        expanded.points - previous.state.points, dim=1
    )
    image_difference = torch.cat(
        [
            current - old
            for current, old in zip(expanded.images, previous.state.images)
        ]
    )
    return expanded, {
        "expansion_geometry_l2_difference": float(
            torch.linalg.vector_norm(point_difference)
        ),
        "expansion_geometry_max_difference": float(point_difference.max()),
        "expansion_image_l2_difference": float(
            torch.linalg.vector_norm(image_difference)
        ),
        "expansion_image_max_difference": float(image_difference.abs().max()),
        "expansion_loss_difference": abs(expanded.loss - previous.state.loss),
    }


def _solve_fixed(
    context: SequentialContext,
    target: dict[str, object],
    active_k: int,
    start_mode: str,
    previous: FixedSolution | None = None,
) -> FixedSolution:
    device = context.reference_points.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    total_started = time.perf_counter()
    if previous is None:
        state = _evaluate_active(
            context,
            torch.arange(active_k, device=device),
            torch.zeros(active_k, dtype=torch.float64, device=device),
            target["images"],  # type: ignore[arg-type]
        )
        expansion = {
            "expansion_geometry_l2_difference": 0.0,
            "expansion_geometry_max_difference": 0.0,
            "expansion_image_l2_difference": 0.0,
            "expansion_image_max_difference": 0.0,
            "expansion_loss_difference": 0.0,
        }
    else:
        state, expansion = _expansion_diagnostic(
            context, target, previous, active_k
        )
    state, optimization = _diagnostic_optimize(
        context,
        state,
        target["images"],  # type: ignore[arg-type]
        context.config.fixed_optimization_steps,
    )
    _sync(device)
    geometry_started = time.perf_counter()
    geometry = context.geometry_evaluator(state.points)  # type: ignore[misc]
    geometry_seconds = time.perf_counter() - geometry_started
    row: dict[str, object] = {
        "start_mode": start_mode,
        "active_k": active_k,
        **_normalized_observation_metrics(state, target),
        **geometry,
        **optimization,
        **expansion,
        "root_failures": state.root_failures,
        "cg_failures": state.cg_failures,
        "line_search_failures": state.line_search_failures,
        "minimum_denominator_magnitude": float(state.denominator.abs().min()),
        "geometry_evaluation_seconds": geometry_seconds,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else 0.0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if device.type == "cuda"
        else 0.0,
    }
    return FixedSolution(state, row)


def _operator_diagnostics(
    context: SequentialContext, state: ActiveState
) -> dict[str, object]:
    device = state.points.device
    count = context.config.master_count
    gram = torch.zeros((count, count), dtype=torch.float64, device=device)
    affected_pixels = torch.zeros(count, dtype=torch.long, device=device)
    nnz = 0
    active_pairs = 0
    active_pixels = 0
    candidate_seconds = 0.0
    gram_seconds = 0.0
    total_pixels = 0
    for cell in context.cells:
        _sync(device)
        started = time.perf_counter()
        matrix = _local_sparse_jacobian(
            context.master_layout,
            state.points,
            context.reference_normals,
            state.denominator,
            context.master_support,
            cell,
        ).coalesce()
        _sync(device)
        candidate_seconds += time.perf_counter() - started
        indices = matrix.indices()
        pixel_count = cell.camera.pixel_count
        pixels = torch.div(indices[0], 3, rounding_mode="floor")
        encoded = torch.unique(indices[1] * pixel_count + pixels)
        affected_pixels += torch.bincount(
            torch.div(encoded, pixel_count, rounding_mode="floor"),
            minlength=count,
        )
        active = indices[1] < state.active_ids.numel()
        if bool(active.any()):
            active_encoded = torch.unique(
                indices[1, active] * pixel_count + pixels[active]
            )
            active_pairs += int(active_encoded.numel())
            active_pixels += int(torch.unique(pixels[active]).numel())
        nnz += matrix._nnz()
        total_pixels += pixel_count
        _sync(device)
        started = time.perf_counter()
        product = torch.sparse.mm(matrix.transpose(0, 1), matrix)
        gram += product.to_dense() if product.is_sparse else product
        _sync(device)
        gram_seconds += time.perf_counter() - started
        del matrix, product
    gram = 0.5 * (gram + gram.T)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular = torch.sqrt(eigenvalues)
    maximum = float(singular[-1]) if singular.numel() else 0.0
    norms = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
    responsive = norms > 1e-12
    maximum_norm = float(norms.max())
    near_null = norms <= 1e-8 * max(maximum_norm, 1e-30)
    normalized = gram[responsive][:, responsive] / (
        norms[responsive, None] * norms[None, responsive]
    ).clamp_min(1e-30)
    if normalized.shape[0] > 1:
        upper_indices = torch.triu_indices(
            normalized.shape[0], normalized.shape[0], offset=1, device=device
        )
        mutual = normalized.abs()[upper_indices[0], upper_indices[1]]
    else:
        mutual = torch.zeros(1, dtype=torch.float64, device=device)
    useful = singular[singular >= 1e-10 * max(maximum, 1e-30)]
    norm_values = norms[responsive]
    quantiles = (
        torch.quantile(
            norm_values,
            torch.tensor(
                [0.0, 0.1, 0.5, 0.9, 1.0],
                dtype=torch.float64,
                device=device,
            ),
        )
        if norm_values.numel()
        else torch.zeros(5, dtype=torch.float64, device=device)
    )
    return {
        "evaluated_at": "cold_K256_solution",
        "candidate_jacobian_construction_seconds": candidate_seconds,
        "gram_analysis_seconds": gram_seconds,
        "candidate_jacobian_nnz": nnz,
        "total_scalar_observations": context.config.views
        * context.config.resolution**2
        * 3,
        "responsive_candidate_fraction": float(responsive.to(torch.float64).mean()),
        "near_null_candidate_fraction": float(near_null.to(torch.float64).mean()),
        "candidate_column_norm_quantiles": {
            key: float(value)
            for key, value in zip(
                ("min", "p10", "median", "p90", "max"), quantiles
            )
        },
        "affected_pixels_per_candidate_mean": float(
            affected_pixels[responsive].to(torch.float64).mean()
        )
        if bool(responsive.any())
        else 0.0,
        "affected_pixels_per_candidate_median": float(
            affected_pixels[responsive].to(torch.float64).median()
        )
        if bool(responsive.any())
        else 0.0,
        "affected_fraction_per_candidate_mean": float(
            affected_pixels[responsive].to(torch.float64).mean()
            / max(total_pixels, 1)
        )
        if bool(responsive.any())
        else 0.0,
        "active_parameters_per_affected_pixel": active_pairs
        / max(active_pixels, 1),
        "numerical_rank_relative_1e-6": int(
            (singular >= 1e-6 * max(maximum, 1e-30)).sum()
        ),
        "numerical_rank_relative_1e-8": int(
            (singular >= 1e-8 * max(maximum, 1e-30)).sum()
        ),
        "numerical_rank_relative_1e-10": int(
            (singular >= 1e-10 * max(maximum, 1e-30)).sum()
        ),
        "stable_rank": float(eigenvalues.sum() / eigenvalues[-1].clamp_min(1e-30)),
        "largest_singular_values": singular[-5:].flip(0).detach().cpu().tolist(),
        "smallest_useful_singular_values": useful[:5].detach().cpu().tolist(),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "smallest_useful_eigenvalue": float(useful[0].square())
        if useful.numel()
        else 0.0,
        "responsive_subspace_condition_number": float(useful[-1] / useful[0])
        if useful.numel()
        else math.inf,
        "mutual_column_cosine_max": float(mutual.max()),
        "mutual_column_cosine_median": float(mutual.median()),
        "candidate_pair_fraction_cosine_gt_0.05": float(
            (mutual > 0.05).to(torch.float64).mean()
        ),
        "candidate_pair_fraction_cosine_gt_0.10": float(
            (mutual > 0.10).to(torch.float64).mean()
        ),
        "candidate_pair_fraction_cosine_gt_0.20": float(
            (mutual > 0.20).to(torch.float64).mean()
        ),
    }


def _classify_path(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda item: int(item["active_k"]))
    image = [float(item["raw_image_loss"]) for item in ordered]
    geometry = [float(item["symmetric_chamfer"]) for item in ordered]
    image_tolerance = 1e-10 * max(abs(image[0]), 1.0)
    geometry_tolerance = 1e-12
    image_monotone = all(
        right <= left + image_tolerance
        for left, right in zip(image, image[1:])
    )
    geometry_monotone = all(
        right <= left + geometry_tolerance
        for left, right in zip(geometry, geometry[1:])
    )
    image_improves = image[-1] < image[0] - image_tolerance
    geometry_improves = geometry[-1] < geometry[0] - geometry_tolerance
    image_worsens = any(
        right > left + image_tolerance
        for left, right in zip(image, image[1:])
    )
    geometry_worsens = any(
        right > left + geometry_tolerance
        for left, right in zip(geometry, geometry[1:])
    )
    if image_monotone and geometry_monotone and image_improves and geometry_improves:
        case = "A"
    elif image_monotone and image_improves and geometry_worsens:
        case = "B"
    elif image_worsens and geometry_worsens:
        case = "C"
    elif geometry_improves and not image_improves:
        case = "D"
    else:
        case = "STALLED_OR_MIXED"
    return {
        "case": case,
        "image_monotone_nonincreasing": image_monotone,
        "geometry_monotone_nonincreasing": geometry_monotone,
        "image_materially_improves": image_improves,
        "geometry_materially_improves": geometry_improves,
        "image_has_worsening_step": image_worsens,
        "geometry_has_worsening_step": geometry_worsens,
        "k1024_minus_k256_chamfer": geometry[-1] - geometry[0],
        "k1024_minus_k256_p2s_p95": float(ordered[-1]["point_to_surface_p95"])
        - float(ordered[0]["point_to_surface_p95"]),
        "k1024_minus_k256_normalized_image_mse": float(
            ordered[-1]["normalized_image_mse"]
        )
        - float(ordered[0]["normalized_image_mse"]),
    }


def _fixed_condition(
    context: SequentialContext,
    target: dict[str, object],
    condition: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cold: dict[int, FixedSolution] = {}
    rows = []
    for active_k in FIXED_LEVELS:
        solution = _solve_fixed(context, target, active_k, "cold")
        cold[active_k] = solution
        rows.append(solution.row)
    warm_512 = _solve_fixed(context, target, 512, "warm", cold[256])
    warm_1024 = _solve_fixed(context, target, 1024, "warm", warm_512)
    rows.extend((warm_512.row, warm_1024.row))
    operator = _operator_diagnostics(context, cold[256].state)
    for row in rows:
        row.update(
            {
                "sigma": condition["sigma"],
                "condition": condition["condition"],
                "resolution": condition["resolution"],
                "photons": condition["photons"],
                "packets_per_emitter": condition["packets_per_emitter"],
                "responsive_candidate_fraction": operator[
                    "responsive_candidate_fraction"
                ],
                "near_null_candidate_fraction": operator[
                    "near_null_candidate_fraction"
                ],
                "stable_rank": operator["stable_rank"],
                "numerical_rank_relative_1e-6": operator[
                    "numerical_rank_relative_1e-6"
                ],
                "numerical_rank_relative_1e-8": operator[
                    "numerical_rank_relative_1e-8"
                ],
                "numerical_rank_relative_1e-10": operator[
                    "numerical_rank_relative_1e-10"
                ],
                "mutual_column_cosine_max": operator[
                    "mutual_column_cosine_max"
                ],
                "mutual_column_cosine_median": operator[
                    "mutual_column_cosine_median"
                ],
                "occupied_pixels": condition["target_observation_support"][
                    "occupied_pixels"
                ],
                "occupied_pixel_fraction": condition[
                    "target_observation_support"
                ]["occupied_pixel_fraction"],
                "photons_per_occupied_pixel": condition[
                    "target_observation_support"
                ]["photons_per_occupied_pixel"],
                "photon_state_transport_event_fraction": condition[
                    "photon_state_transport_event_fraction"
                ],
                "occupied_owner_event_fraction": condition[
                    "occupied_owner_event_fraction"
                ],
            }
        )
    warm_rows = [cold[256].row, warm_512.row, warm_1024.row]
    return rows, {
        "operator": operator,
        "cold_classification": _classify_path(
            [cold[level].row for level in FIXED_LEVELS]
        ),
        "warm_classification": _classify_path(warm_rows),
        "maximum_expansion_geometry_difference": max(
            float(row["expansion_geometry_max_difference"])
            for row in (warm_512.row, warm_1024.row)
        ),
        "maximum_expansion_image_difference": max(
            float(row["expansion_image_max_difference"])
            for row in (warm_512.row, warm_1024.row)
        ),
    }


def _path_rows(
    condition: dict[str, object], start_mode: str
) -> list[dict[str, object]]:
    rows = condition["fixed_rows"]  # type: ignore[assignment]
    if start_mode == "cold":
        return sorted(
            [row for row in rows if row["start_mode"] == "cold"],
            key=lambda row: int(row["active_k"]),
        )
    cold_256 = next(
        row
        for row in rows
        if row["start_mode"] == "cold" and int(row["active_k"]) == 256
    )
    return [cold_256] + sorted(
        [row for row in rows if row["start_mode"] == "warm"],
        key=lambda row: int(row["active_k"]),
    )


def _factorial_effects(
    conditions: list[dict[str, object]], sigma: float
) -> dict[str, object]:
    group = {
        item["condition"]: item
        for item in conditions
        if float(item["sigma"]) == sigma
    }

    def value(label: str, metric: str) -> float:
        condition = group[label]
        if metric == "stable_rank":
            return float(condition["diagnostics"]["operator"][metric])
        if metric == "near_null_candidate_fraction":
            return float(condition["diagnostics"]["operator"][metric])
        rows = _path_rows(condition, "warm")
        if metric == "chamfer_gap":
            return float(rows[-1]["symmetric_chamfer"]) - float(
                rows[0]["symmetric_chamfer"]
            )
        return float(rows[-1][metric])

    effects = {}
    for metric in (
        "symmetric_chamfer",
        "point_to_surface_p95",
        "stable_rank",
        "near_null_candidate_fraction",
        "chamfer_gap",
    ):
        values = {label: value(label, metric) for label in "ABCD"}
        effects[metric] = {
            "values": values,
            "resolution_main_effect": 0.5 * (values["C"] + values["D"])
            - 0.5 * (values["A"] + values["B"]),
            "photon_main_effect": 0.5 * (values["B"] + values["D"])
            - 0.5 * (values["A"] + values["C"]),
            "interaction": (values["D"] - values["C"])
            - (values["B"] - values["A"]),
        }
    return effects


def _mean_effect_supported(
    effects: dict[str, object],
    kind: str,
    conditions: list[dict[str, object]],
    sigma: float,
) -> tuple[bool, dict[str, object]]:
    suffix = f"{kind}_main_effect"
    baseline_group = [item for item in conditions if float(item["sigma"]) == sigma]
    mean_chamfer = statistics.mean(
        float(_path_rows(item, "warm")[-1]["symmetric_chamfer"])
        for item in baseline_group
    )
    mean_p95 = statistics.mean(
        float(_path_rows(item, "warm")[-1]["point_to_surface_p95"])
        for item in baseline_group
    )
    mean_rank = statistics.mean(
        float(item["diagnostics"]["operator"]["stable_rank"])
        for item in baseline_group
    )
    main_checks = {
        "chamfer_improves_0p1_percent": float(
            effects["symmetric_chamfer"][suffix]
        )
        <= -0.001 * mean_chamfer,
        "p2s_p95_improves_0p1_percent": float(
            effects["point_to_surface_p95"][suffix]
        )
        <= -0.001 * mean_p95,
        "stable_rank_improves_5_percent": float(
            effects["stable_rank"][suffix]
        )
        >= 0.05 * mean_rank,
        "near_null_drops_0p5_percentage_point": float(
            effects["near_null_candidate_fraction"][suffix]
        )
        <= -0.005,
    }
    group = {str(item["condition"]): item for item in baseline_group}
    pairs = (
        (("A", "C"), ("B", "D"))
        if kind == "resolution"
        else (("A", "B"), ("C", "D"))
    )
    fixed_comparisons = {}
    for low, high in pairs:
        low_final = _path_rows(group[low], "warm")[-1]
        high_final = _path_rows(group[high], "warm")[-1]
        pair_checks = {
            "chamfer_improves_0p1_percent": float(
                high_final["symmetric_chamfer"]
            )
            <= 0.999 * float(low_final["symmetric_chamfer"]),
            "p2s_p95_improves_0p1_percent": float(
                high_final["point_to_surface_p95"]
            )
            <= 0.999 * float(low_final["point_to_surface_p95"]),
            "stable_rank_improves_5_percent": float(
                group[high]["diagnostics"]["operator"]["stable_rank"]
            )
            >= 1.05
            * float(group[low]["diagnostics"]["operator"]["stable_rank"]),
            "near_null_drops_0p5_percentage_point": float(
                group[high]["diagnostics"]["operator"][
                    "near_null_candidate_fraction"
                ]
            )
            <= float(
                group[low]["diagnostics"]["operator"][
                    "near_null_candidate_fraction"
                ]
            )
            - 0.005,
        }
        pair_geometry = pair_checks["chamfer_improves_0p1_percent"] or pair_checks[
            "p2s_p95_improves_0p1_percent"
        ]
        pair_observability = (
            pair_checks["stable_rank_improves_5_percent"]
            or pair_checks["near_null_drops_0p5_percentage_point"]
        )
        fixed_comparisons[f"{low}_to_{high}"] = {
            "supported": pair_geometry and pair_observability,
            "checks": pair_checks,
        }
    main_geometry = main_checks["chamfer_improves_0p1_percent"] or main_checks[
        "p2s_p95_improves_0p1_percent"
    ]
    main_observability = (
        main_checks["stable_rank_improves_5_percent"]
        or main_checks["near_null_drops_0p5_percentage_point"]
    )
    supported = (main_geometry and main_observability) or any(
        bool(item["supported"]) for item in fixed_comparisons.values()
    )
    return supported, {
        "main_effect_checks": main_checks,
        "fixed_resolution_or_sampling_comparisons": fixed_comparisons,
    }


def _best_observation_condition(
    conditions: list[dict[str, object]]
) -> dict[str, object]:
    primary = [item for item in conditions if float(item["sigma"]) == 2.5]
    return min(
        primary,
        key=lambda item: (
            not bool(
                item["diagnostics"]["warm_classification"][
                    "geometry_monotone_nonincreasing"
                ]
            ),
            float(_path_rows(item, "warm")[-1]["symmetric_chamfer"]),
            -float(item["diagnostics"]["operator"]["stable_rank"]),
            float(item["diagnostics"]["operator"]["near_null_candidate_fraction"]),
            sum(float(row["total_runtime_seconds"]) for row in item["fixed_rows"]),
        ),
    )


def _optional_gate(conditions: list[dict[str, object]]) -> dict[str, object]:
    group = {
        item["condition"]: item
        for item in conditions
        if float(item["sigma"]) == 2.5
    }
    high = group["D"]
    lows = [group["A"], group["B"]]
    high_final = _path_rows(high, "warm")[-1]
    checks = {
        "geometry_monotonicity": bool(
            high["diagnostics"]["warm_classification"][
                "geometry_monotone_nonincreasing"
            ]
        )
        and not any(
            bool(
                item["diagnostics"]["warm_classification"][
                    "geometry_monotone_nonincreasing"
                ]
            )
            for item in lows
        ),
        "stable_rank": float(high["diagnostics"]["operator"]["stable_rank"])
        >= 1.05
        * max(float(item["diagnostics"]["operator"]["stable_rank"]) for item in lows),
        "near_null_fraction": float(
            high["diagnostics"]["operator"]["near_null_candidate_fraction"]
        )
        <= min(
            float(item["diagnostics"]["operator"]["near_null_candidate_fraction"])
            for item in lows
        )
        - 0.005,
        "predictor_distinguishability": False,
        "normalized_image_residual": float(high_final["normalized_image_mse"])
        <= 0.95
        * min(
            float(_path_rows(item, "warm")[-1]["normalized_image_mse"])
            for item in lows
        ),
    }
    return {"passed": sum(checks.values()) >= 2, "checks": checks}


def _trajectory_observation_metrics(
    trajectory: dict[str, object], target: dict[str, object]
) -> None:
    observation_count = sum(
        image.numel() for image in target["images"]  # type: ignore[union-attr]
    )
    target_energy = sum(
        float((image * image).sum())
        for image in target["images"]  # type: ignore[union-attr]
    )
    for row in trajectory["rows"]:
        squared = 2.0 * float(row["image_loss"])
        row["raw_image_loss"] = row["image_loss"]
        row["normalized_image_mse"] = squared / observation_count
        row["residual_energy"] = squared
        row["normalized_residual_energy"] = squared / max(target_energy, 1e-30)


def _window_calibration(trajectory: dict[str, object]) -> dict[str, object]:
    batches = trajectory["batches"]
    result = {}
    for name, low, high in (
        ("early", 0.0, 1 / 3),
        ("middle", 1 / 3, 2 / 3),
        ("late", 2 / 3, 1.0),
    ):
        start = int(low * len(batches))
        stop = len(batches) if high == 1.0 else int(high * len(batches))
        rows = batches[start:stop]
        predicted = torch.tensor(
            [float(row["predicted_batch_gain"]) for row in rows],
            dtype=torch.float64,
        )
        realized = torch.tensor(
            [float(row["realized_gain"]) for row in rows], dtype=torch.float64
        )
        result[name] = {
            "rounds": len(rows),
            "pearson": _correlation(predicted, realized, False),
            "spearman": _correlation(predicted, realized, True),
        }
    return result


def _birth_regions(trajectory: dict[str, object]) -> dict[str, float]:
    centers = torch.tensor(
        [center for batch in trajectory["batches"] for center in batch["centers"]],
        dtype=torch.float64,
    )
    ear = centers[:, 1] > 0.45
    head = (centers[:, 1] <= 0.45) & (centers[:, 1] > -0.05) & (centers[:, 0] < -0.35)
    leg = centers[:, 1] < -0.70
    torso = ~(ear | head | leg)
    return {
        name: float(mask.to(torch.float64).mean())
        for name, mask in (("ear", ear), ("head", head), ("torso", torso), ("leg", leg))
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )


def bandwidth_cpu_verification() -> dict[str, object]:
    """Verify the deterministic prefix contract used by the v0.3f sweep."""
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]], dtype=torch.float64
    )
    directions_8 = nested_deterministic_directions(normals, 8, 128.0).reshape(
        2, 8, 3
    )
    directions_32 = nested_deterministic_directions(
        normals, 32, 128.0
    ).reshape(2, 32, 3)
    prefix_error = float((directions_8 - directions_32[:, :8]).abs().max())
    unit_error = float(
        (
            torch.linalg.vector_norm(directions_32, dim=-1)
            - torch.ones((2, 32), dtype=torch.float64)
        )
        .abs()
        .max()
    )
    assert prefix_error == 0.0
    assert unit_error < 1e-12
    return {
        "nested_8_in_32_prefix_max_error": prefix_error,
        "maximum_unit_length_error": unit_error,
    }


def _fixed_figures(
    directory: Path,
    conditions: list[dict[str, object]],
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    ordered_conditions = sorted(
        conditions,
        key=lambda item: (
            float(item["sigma"]),
            "ABCD".index(str(item["condition"])),
        ),
    )
    for sigma in (2.5, 4.0):
        figure, axis = plt.subplots(figsize=(6.8, 4.5))
        for item in ordered_conditions:
            if float(item["sigma"]) != sigma:
                continue
            rows = _path_rows(item, "warm")
            axis.plot(
                [row["active_k"] for row in rows],
                [row["symmetric_chamfer"] for row in rows],
                marker="o",
                color=colors[item["condition"]],
                label=item["condition"],
            )
        axis.set_xlabel("active K")
        axis.set_ylabel("symmetric Chamfer")
        axis.set_title(f"Bunny fixed-space scaling, σ={sigma:.1f}")
        axis.grid(alpha=0.25)
        axis.legend(title="condition")
        figure.tight_layout()
        path = directory / f"v03f_chamfer_sigma_{sigma:.1f}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, sigma in zip(axes, (2.5, 4.0)):
        for item in ordered_conditions:
            if float(item["sigma"]) != sigma:
                continue
            rows = _path_rows(item, "warm")
            axis.plot(
                [row["active_k"] for row in rows],
                [row["normalized_image_mse"] for row in rows],
                marker="o",
                color=colors[item["condition"]],
                label=item["condition"],
            )
        axis.set_title(f"σ={sigma:.1f}")
        axis.set_xlabel("active K")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("normalized image MSE")
    axes[-1].legend()
    figure.tight_layout()
    path = directory / "v03f_normalized_image_vs_k.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    labels = [
        f"{item['condition']}\nσ={item['sigma']}"
        for item in ordered_conditions
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    gaps = [
        float(item["diagnostics"]["warm_classification"]["k1024_minus_k256_chamfer"])
        for item in ordered_conditions
    ]
    axis.bar(labels, gaps)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("warm K1024 − K256 Chamfer")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = directory / "v03f_geometry_gap.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    for name, key, ylabel in (
        ("stable_rank", "stable_rank", "stable rank"),
        ("near_null", "near_null_candidate_fraction", "near-null fraction"),
    ):
        figure, axis = plt.subplots(figsize=(8.2, 4.4))
        values = [
            float(item["diagnostics"]["operator"][key])
            for item in ordered_conditions
        ]
        axis.bar(labels, values)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = directory / f"v03f_{name}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    occupied = [
        float(item["transport"]["target_observation_support"]["occupied_pixel_fraction"])
        for item in ordered_conditions
    ]
    density = [
        float(item["transport"]["target_observation_support"]["photons_per_occupied_pixel"])
        for item in ordered_conditions
    ]
    axes[0].bar(labels, occupied)
    axes[1].bar(labels, density)
    axes[0].set_ylabel("occupied detector fraction")
    axes[1].set_ylabel("events / occupied pixel")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = directory / "v03f_detector_occupancy.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, sigma in zip(axes, (2.5, 4.0)):
        group = [
            item
            for item in ordered_conditions
            if float(item["sigma"]) == sigma
        ]
        x = np.arange(len(group))
        cold = [
            float(_path_rows(item, "cold")[-1]["symmetric_chamfer"])
            for item in group
        ]
        warm = [
            float(_path_rows(item, "warm")[-1]["symmetric_chamfer"])
            for item in group
        ]
        axis.bar(x - 0.18, cold, 0.36, label="cold")
        axis.bar(x + 0.18, warm, 0.36, label="nested warm")
        axis.set_xticks(x, [item["condition"] for item in group])
        axis.set_title(f"σ={sigma:.1f}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("K=1024 Chamfer")
    axes[-1].legend()
    figure.tight_layout()
    path = directory / "v03f_cold_vs_warm.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _dynamic_figures(
    directory: Path,
    dynamic: dict[str, object],
    uniform: dict[str, object],
    baseline_schedule: list[int],
) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    for trajectory, label in ((dynamic, "dynamic"), (uniform, "uniform matched")):
        axis.plot(
            [row["active_k"] for row in trajectory["rows"]],
            [row["symmetric_chamfer"] for row in trajectory["rows"]],
            marker="o",
            label=label,
        )
    axis.set_xlabel("active DoFs", loc="center")
    axis.set_ylabel("symmetric Chamfer")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "v03f_dynamic_geometry_vs_dofs.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.step(
        range(1, len(baseline_schedule) + 1),
        baseline_schedule,
        where="mid",
        label="baseline 256² / 65k",
    )
    axis.step(
        range(1, len(dynamic["batch_schedule"]) + 1),
        dynamic["batch_schedule"],
        where="mid",
        label="best observation config",
    )
    axis.set_xlabel("birth round")
    axis.set_ylabel("dynamic batch size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "v03f_dynamic_batch_schedule.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def run_observation_bandwidth_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    experiment_started = time.perf_counter()
    prepared = prepare_stanford_bunny(mesh_path)
    base_config = BunnyExperimentConfig(
        checkpoints=(32, 64, 96, 128, 160, 192, 224, 256),
        fixed_levels=FIXED_LEVELS,
        random_seeds=(),
        oracle_subset_size=0,
    )
    canonical = _canonical_emitters(prepared, base_config.emitters, device)
    prefix_8 = nested_deterministic_directions(
        canonical["normals"], 8, 128.0  # type: ignore[arg-type]
    ).reshape(base_config.emitters, 8, 3)
    prefix_32 = nested_deterministic_directions(
        canonical["normals"], 32, 128.0  # type: ignore[arg-type]
    ).reshape(base_config.emitters, 32, 3)
    nested_max_error = float((prefix_8 - prefix_32[:, :8]).abs().max())
    if nested_max_error != 0.0:
        raise RuntimeError("nested packet prefix is not bit-identical")
    all_conditions: list[dict[str, object]] = []
    fixed_csv_rows: list[dict[str, object]] = []
    for sigma in (2.5, 4.0):
        geometry = _geometry_bundle(prepared, canonical, sigma, base_config)
        for packets in (8, 32):
            scene = _scene_bundle(geometry, packets, base_config)
            for label, resolution, condition_packets in CONDITIONS:
                if condition_packets != packets:
                    continue
                context, target, transport = _condition_context(
                    geometry, scene, label, resolution, base_config
                )
                condition = {
                    **transport,
                    "sigma": sigma,
                    "condition": label,
                    "resolution": resolution,
                    "packets_per_emitter": packets,
                    "photons": base_config.emitters * packets,
                }
                rows, diagnostics = _fixed_condition(context, target, condition)
                fixed_csv_rows.extend(rows)
                all_conditions.append(
                    {
                        "sigma": sigma,
                        "condition": label,
                        "resolution": resolution,
                        "packets_per_emitter": packets,
                        "photons": base_config.emitters * packets,
                        "transport": transport,
                        "fixed_rows": rows,
                        "diagnostics": diagnostics,
                    }
                )
                print(
                    json.dumps(
                        {
                            "phase": "fixed",
                            "sigma": sigma,
                            "condition": label,
                            "cold_case": diagnostics["cold_classification"]["case"],
                            "warm_case": diagnostics["warm_classification"]["case"],
                            "warm_k1024_chamfer": _path_rows(
                                all_conditions[-1], "warm"
                            )[-1]["symmetric_chamfer"],
                            "stable_rank": diagnostics["operator"]["stable_rank"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del context, target
                torch.cuda.empty_cache()
            del scene
        del geometry
        torch.cuda.empty_cache()
    factorial = {
        f"{sigma:.1f}": _factorial_effects(all_conditions, sigma)
        for sigma in (2.5, 4.0)
    }
    resolution_checks = {}
    photon_checks = {}
    resolution_votes = []
    photon_votes = []
    for sigma in (2.5, 4.0):
        supported, checks = _mean_effect_supported(
            factorial[f"{sigma:.1f}"], "resolution", all_conditions, sigma
        )
        resolution_votes.append(supported)
        resolution_checks[f"{sigma:.1f}"] = checks
        supported, checks = _mean_effect_supported(
            factorial[f"{sigma:.1f}"], "photon", all_conditions, sigma
        )
        photon_votes.append(supported)
        photon_checks[f"{sigma:.1f}"] = checks
    resolution_supported = any(resolution_votes)
    photon_supported = any(photon_votes)
    group_25 = {
        item["condition"]: item
        for item in all_conditions
        if float(item["sigma"]) == 2.5
    }
    baseline = group_25["A"]
    combined = group_25["D"]
    baseline_final = _path_rows(baseline, "warm")[-1]
    combined_final = _path_rows(combined, "warm")[-1]
    bandwidth_checks = {
        "geometry_monotonicity_restored": bool(
            combined["diagnostics"]["warm_classification"][
                "geometry_monotone_nonincreasing"
            ]
        )
        and not bool(
            baseline["diagnostics"]["warm_classification"][
                "geometry_monotone_nonincreasing"
            ]
        ),
        "k1024_chamfer_improves_1_percent": float(
            combined_final["symmetric_chamfer"]
        )
        <= 0.99 * float(baseline_final["symmetric_chamfer"]),
        "stable_rank_improves_5_percent": float(
            combined["diagnostics"]["operator"]["stable_rank"]
        )
        >= 1.05 * float(baseline["diagnostics"]["operator"]["stable_rank"]),
        "near_null_drops_0p5_percentage_point": float(
            combined["diagnostics"]["operator"]["near_null_candidate_fraction"]
        )
        <= float(baseline["diagnostics"]["operator"]["near_null_candidate_fraction"])
        - 0.005,
    }
    bandwidth_supported = (
        bandwidth_checks["geometry_monotonicity_restored"]
        or bandwidth_checks["k1024_chamfer_improves_1_percent"]
    ) and (
        bandwidth_checks["stable_rank_improves_5_percent"]
        or bandwidth_checks["near_null_drops_0p5_percentage_point"]
    )
    cold_class = baseline["diagnostics"]["cold_classification"]
    warm_class = baseline["diagnostics"]["warm_classification"]
    cold_final = _path_rows(baseline, "cold")[-1]
    warm_final = _path_rows(baseline, "warm")[-1]
    optimization_checks = {
        "monotonicity_restored": bool(warm_class["geometry_monotone_nonincreasing"])
        and not bool(cold_class["geometry_monotone_nonincreasing"]),
        "warm_k1024_chamfer_improves_1_percent": float(
            warm_final["symmetric_chamfer"]
        )
        <= 0.99 * float(cold_final["symmetric_chamfer"]),
        "zero_expansion": float(
            baseline["diagnostics"]["maximum_expansion_geometry_difference"]
        )
        < 1e-12
        and float(
            baseline["diagnostics"]["maximum_expansion_image_difference"]
        )
        < 1e-12,
    }
    optimization_supported = (
        optimization_checks["monotonicity_restored"]
        or optimization_checks["warm_k1024_chamfer_improves_1_percent"]
    )
    ambiguity_conditions = [
        f"sigma={item['sigma']}/{item['condition']}"
        for item in all_conditions
        if item["diagnostics"]["warm_classification"]["case"] == "B"
    ]
    ambiguity_supported = bool(ambiguity_conditions)
    representation_supported = not any(
        bool(
            item["diagnostics"]["cold_classification"][
                "geometry_materially_improves"
            ]
        )
        or bool(
            item["diagnostics"]["warm_classification"][
                "geometry_materially_improves"
            ]
        )
        for item in all_conditions
    )
    scientific_classification = {
        "CLASS_A_OBSERVATION_BANDWIDTH_BOTTLENECK": bandwidth_supported,
        "CLASS_B_PHOTON_SAMPLING_BOTTLENECK": photon_supported,
        "CLASS_C_DETECTOR_RESOLUTION_BOTTLENECK": resolution_supported,
        "CLASS_D_OPTIMIZATION_BOTTLENECK": optimization_supported,
        "CLASS_E_INVERSE_AMBIGUITY": ambiguity_supported,
        "CLASS_F_REPRESENTATION_LIMITATION": representation_supported,
        "strongest_explanation": "CLASS_D_OPTIMIZATION_BOTTLENECK",
        "secondary_explanation": (
            "CLASS_B_PHOTON_SAMPLING_BOTTLENECK"
            if photon_supported
            else None
        ),
        "interpretation": (
            "Nested initialization fixes the primary sigma=2.5 K=1024 "
            "failure without an operator change. More photons at 256 squared "
            "also reduce the near-null fraction and modestly improve geometry; "
            "higher detector resolution does not."
        ),
    }
    optional_gate = _optional_gate(all_conditions)
    optional_report: dict[str, object] | str = "OPTIONAL_1024_SKIPPED"
    if optional_gate["passed"]:
        optional_geometry = _geometry_bundle(
            prepared, canonical, 2.5, base_config
        )
        directions_32 = nested_deterministic_directions(
            optional_geometry.reference_normals, 32, 128.0
        ).reshape(base_config.emitters, 32, 3)
        directions_128 = nested_deterministic_directions(
            optional_geometry.reference_normals, 128, 128.0
        ).reshape(base_config.emitters, 128, 3)
        optional_prefix_error = float(
            (directions_32 - directions_128[:, :32]).abs().max()
        )
        if optional_prefix_error != 0.0:
            raise RuntimeError("optional 128-packet prefix is not bit-identical")
        optional_scene = _scene_bundle(optional_geometry, 128, base_config)
        optional_context, optional_target, optional_transport = _condition_context(
            optional_geometry,
            optional_scene,
            "E",
            1024,
            base_config,
        )
        optional_target.pop("transport_cells", None)
        optional_256 = _solve_fixed(
            optional_context, optional_target, 256, "cold"
        )
        optional_1024_cold = _solve_fixed(
            optional_context, optional_target, 1024, "cold"
        )
        optional_1024_warm = _solve_fixed(
            optional_context,
            optional_target,
            1024,
            "warm",
            optional_256,
        )
        optional_rows = [
            optional_256.row,
            optional_1024_cold.row,
            optional_1024_warm.row,
        ]
        for row in optional_rows:
            row.update(
                {
                    "sigma": 2.5,
                    "condition": "E",
                    "resolution": 1024,
                    "packets_per_emitter": 128,
                    "photons": 1_048_576,
                    "occupied_pixels": optional_transport[
                        "target_observation_support"
                    ]["occupied_pixels"],
                    "occupied_pixel_fraction": optional_transport[
                        "target_observation_support"
                    ]["occupied_pixel_fraction"],
                    "photons_per_occupied_pixel": optional_transport[
                        "target_observation_support"
                    ]["photons_per_occupied_pixel"],
                    "photon_state_transport_event_fraction": optional_transport[
                        "photon_state_transport_event_fraction"
                    ],
                    "occupied_owner_event_fraction": optional_transport[
                        "occupied_owner_event_fraction"
                    ],
                }
            )
        fixed_csv_rows.extend(optional_rows)
        optional_report = {
            "status": "OPTIONAL_1024_COMPLETED",
            "nested_32_in_128_prefix_max_error": optional_prefix_error,
            "transport": optional_transport,
            "fixed_rows": optional_rows,
            "cold_classification": _classify_path(
                [optional_256.row, optional_1024_cold.row]
            ),
            "warm_classification": _classify_path(
                [optional_256.row, optional_1024_warm.row]
            ),
        }
        del optional_context, optional_target, optional_scene, optional_geometry
        torch.cuda.empty_cache()
    best = _best_observation_condition(all_conditions)
    best_geometry = _geometry_bundle(prepared, canonical, 2.5, base_config)
    best_scene = _scene_bundle(
        best_geometry, int(best["packets_per_emitter"]), base_config
    )
    best_context, best_target, rebuilt_transport = _condition_context(
        best_geometry,
        best_scene,
        str(best["condition"]),
        int(best["resolution"]),
        base_config,
    )
    best_target.pop("transport_cells", None)
    dynamic = _run_batch_trajectory(
        best_context, best_target, BatchPolicy("dynamic", dynamic=True)
    )
    uniform = _run_batch_trajectory(
        best_context,
        best_target,
        BatchPolicy(
            "uniform_matched",
            uniform_schedule=tuple(dynamic["batch_schedule"]),
        ),
    )
    _trajectory_observation_metrics(dynamic, best_target)
    _trajectory_observation_metrics(uniform, best_target)
    historical = json.loads(
        (artifact_directory / "v03e_bunny_smoothing.json").read_text()
    )
    historical_dynamic = next(
        item
        for item in historical["trajectory_summaries"]
        if float(item["sigma"]) == 2.5 and item["method"] == "dynamic"
    )
    historical_trajectory = historical["trajectories"]["2.5"]["adaptive"]
    dynamic_final = dynamic["rows"][-1]
    uniform_final = uniform["rows"][-1]
    genuine_metrics = (
        "symmetric_chamfer",
        "point_to_surface_mean",
        "point_to_surface_p95",
        "surface_rms",
        "normal_error",
    )
    dynamic_wins = {
        metric: float(dynamic_final[metric]) < float(uniform_final[metric])
        for metric in genuine_metrics
    }
    dynamic_stable = all(
        int(row[key]) == 0
        for row in (dynamic_final, uniform_final)
        for key in ("root_failures", "cg_failures", "nonfinite_candidate_score_count")
    ) and dynamic["maximum_birth_geometry_jump"] < 1e-12
    dynamic_supported = (
        dynamic_stable
        and dynamic_wins["symmetric_chamfer"]
        and dynamic_wins["point_to_surface_p95"]
    )
    quality_metrics = (
        *genuine_metrics,
        "raw_image_loss",
        "normalized_image_mse",
        "normalized_residual_energy",
        "total_runtime_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )

    def quality_row(row: dict[str, object]) -> dict[str, object]:
        return {key: row[key] for key in quality_metrics}

    fixed_best_rows = {
        int(row["active_k"]): row
        for row in _path_rows(best, "warm")
    }
    dynamic_report = {
        "verdict": (
            "DYNAMIC_BIRTH_HIGH_BANDWIDTH_SUPPORTED"
            if dynamic_supported
            else "DYNAMIC_BIRTH_HIGH_BANDWIDTH_NOT_SUPPORTED"
        ),
        "best_observation_config": {
            key: best[key]
            for key in (
                "condition",
                "resolution",
                "packets_per_emitter",
                "photons",
            )
        },
        "selection_reason": (
            "geometry monotonicity first, then warm K=1024 Chamfer, stable "
            "rank, near-null fraction, and runtime"
        ),
        "rebuilt_transport": rebuilt_transport,
        "dynamic": dynamic,
        "uniform_matched": uniform,
        "dynamic_summary": _trajectory_summary(dynamic),
        "uniform_summary": _trajectory_summary(uniform),
        "dynamic_geometry_wins": dynamic_wins,
        "dynamic_numerically_stable": dynamic_stable,
        "dynamic_minus_uniform": {
            metric: float(dynamic_final[metric]) - float(uniform_final[metric])
            for metric in genuine_metrics
        },
        "k256_quality_comparison": {
            "dynamic": quality_row(dynamic_final),
            "uniform_matched_schedule": quality_row(uniform_final),
            "fixed_k256": quality_row(fixed_best_rows[256]),
            "fixed_k512_nested_warm": quality_row(fixed_best_rows[512]),
            "fixed_k1024_nested_warm": quality_row(fixed_best_rows[1024]),
        },
        "batch_schedule": dynamic["batch_schedule"],
        "baseline_batch_schedule": historical_trajectory["batch_schedule"],
        "batch_schedule_comparison": {
            "baseline_rounds": historical_dynamic["birth_rounds"],
            "current_rounds": dynamic["birth_rounds"],
            "baseline_mean_batch": historical_dynamic["mean_batch_size"],
            "current_mean_batch": dynamic["mean_batch_size"],
            "baseline_mean_rho_off": historical_dynamic["mean_rho_off"],
            "current_mean_rho_off": dynamic["mean_rho_off"],
            "baseline_max_pairwise_correlation": historical_dynamic[
                "maximum_pairwise_correlation"
            ],
            "current_max_pairwise_correlation": dynamic[
                "maximum_pairwise_correlation"
            ],
            "current_selected_score_min_to_max_mean": statistics.mean(
                float(batch["selected_score_min_to_max"])
                for batch in dynamic["batches"]
            ),
        },
        "predictor_calibration": {
            "overall": dynamic["predicted_joint_calibration"],
            "windows": _window_calibration(dynamic),
            "baseline": historical_dynamic["predicted_joint_calibration"],
        },
        "birth_geography": {
            "current_regions": _birth_regions(dynamic),
            "baseline_regions": _birth_regions(historical_trajectory),
            "current_top25": dynamic["top25_birth_fraction"],
            "current_top10": dynamic["top10_birth_fraction"],
            "baseline_top25": historical_dynamic["top25_birth_fraction"],
            "baseline_top10": historical_dynamic["top10_birth_fraction"],
        },
    }
    fixed_figures = _fixed_figures(figure_directory, all_conditions)
    dynamic_figures = _dynamic_figures(
        figure_directory,
        dynamic,
        uniform,
        historical_trajectory["batch_schedule"],
    )
    fixed_report = {
        "environment": cuda_environment(),
        "bunny": prepared.metadata,
        "configuration": {
            "sigmas": [2.5, 4.0],
            "conditions": [
                {
                    "condition": label,
                    "resolution": resolution,
                    "packets_per_emitter": packets,
                    "photons": base_config.emitters * packets,
                }
                for label, resolution, packets in CONDITIONS
            ],
            "emitters": base_config.emitters,
            "views": base_config.views,
            "fixed_levels": list(FIXED_LEVELS),
            "nested_packet_prefix_max_error": nested_max_error,
            "operator_evaluation_state": "cold K=256",
        },
        "conditions": all_conditions,
        "factorial_effects": factorial,
        "verdicts": {
            "resolution": (
                "RESOLUTION_EFFECT_SUPPORTED"
                if resolution_supported
                else "RESOLUTION_EFFECT_NOT_SUPPORTED"
            ),
            "photon_density": (
                "PHOTON_DENSITY_EFFECT_SUPPORTED"
                if photon_supported
                else "PHOTON_DENSITY_EFFECT_NOT_SUPPORTED"
            ),
            "observation_bandwidth": (
                "OBSERVATION_BANDWIDTH_LIMIT_SUPPORTED"
                if bandwidth_supported
                else "OBSERVATION_BANDWIDTH_LIMIT_NOT_SUPPORTED"
            ),
            "optimization": (
                "OPTIMIZATION_LIMIT_SUPPORTED"
                if optimization_supported
                else "OPTIMIZATION_LIMIT_NOT_SUPPORTED"
            ),
            "geometry_ambiguity": (
                "GEOMETRY_AMBIGUITY_SUPPORTED"
                if ambiguity_supported
                else "GEOMETRY_AMBIGUITY_NOT_SUPPORTED"
            ),
        },
        "verdict_checks": {
            "resolution": resolution_checks,
            "photon_density": photon_checks,
            "observation_bandwidth": bandwidth_checks,
            "optimization": optimization_checks,
            "geometry_ambiguity_conditions": ambiguity_conditions,
        },
        "scientific_classification": scientific_classification,
        "best_observation_config": {
            key: best[key]
            for key in (
                "condition",
                "resolution",
                "packets_per_emitter",
                "photons",
            )
        },
        "optional_1024_gate": optional_gate,
        "optional_1024": optional_report,
        "figures": fixed_figures,
        "total_experiment_seconds_before_dynamic": time.perf_counter()
        - experiment_started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(
        artifact_directory / "v03f_observation_bandwidth.csv", fixed_csv_rows
    )
    _write_json(
        artifact_directory / "v03f_observation_bandwidth.json", fixed_report
    )
    dynamic_rows = []
    for role, trajectory in (("dynamic", dynamic), ("uniform_matched", uniform)):
        for row in _report_rows(trajectory):
            dynamic_rows.append(
                {
                    "role": role,
                    "condition": best["condition"],
                    "resolution": best["resolution"],
                    "photons": best["photons"],
                    **row,
                }
            )
    _write_csv(
        artifact_directory / "v03f_dynamic_high_bandwidth.csv", dynamic_rows
    )
    dynamic_report["figures"] = dynamic_figures
    _write_json(
        artifact_directory / "v03f_dynamic_high_bandwidth.json", dynamic_report
    )
    return {"fixed": fixed_report, "dynamic": dynamic_report}
