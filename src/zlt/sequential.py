"""Repeated observation-driven growth of a sparse zero-set function space."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch

from .benchmark import BenchmarkGeometry, cuda_environment
from .birth import (
    KScalingConfig,
    _ScalingOptimization,
    _candidate_image_deltas,
    _gradient,
    _image_loss,
    _images,
    _local_sparse_jacobian,
    _new_only_oracle,
    _normal_matvec,
    _sparse_column_statistics,
)
from .fields import SphereField, unit_normals
from .jacobian import _photon_batch, deterministic_directions
from .locality import (
    BasisLayout,
    LocalZeroSet,
    SupportPairs,
    hierarchical_surface_points,
    make_support,
)
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    multiview_cameras,
    project_camera,
)
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor
SEQUENTIAL_METHODS = ("quadratic", "raw", "local_residual", "random", "uniform")


@dataclass(frozen=True)
class RepeatedBirthConfig:
    master_count: int = 1024
    initial_count: int = 32
    budget: int = 256
    checkpoints: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256)
    target_regimes: tuple[str, ...] = ("sparse", "distributed")
    target_seeds: tuple[int, ...] = (5, 17)
    random_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    views: int = 8
    resolution: int = 256
    emitters_per_master_basis: int = 8
    packets_per_emitter: int = 8
    base_support_radius: float = 0.60
    support_margin: float = 0.005
    target_rms_displacement: float = 0.003
    coefficient_limit: float = 0.03
    damping: float = 1e-8
    cg_iterations: int = 6
    initial_optimization_steps: int = 6
    post_birth_steps: int = 1
    fixed_optimization_steps: int = 16
    line_evaluations: int = 3
    convergence_tolerance: float = 1e-7
    root_samples: int = 32
    root_bisection_steps: int = 20
    deformation_iterations: int = 10
    deformation_max_offset: float = 0.03
    oracle_master_count: int = 256
    oracle_births: int = 32
    oracle_evaluations: int = 5
    run_oracle: bool = True
    oracle_subset_size: int = 0
    oracle_checkpoints: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not (32 <= self.initial_count < self.budget <= self.master_count):
            raise ValueError("require 32 <= initial_count < budget <= master_count")
        if self.master_count & (self.master_count - 1):
            raise ValueError("master_count must be a power of two")
        if any(
            regime not in {"sparse", "distributed", "bunny"}
            for regime in self.target_regimes
        ):
            raise ValueError("target regimes must be sparse, distributed, or bunny")


@dataclass
class SequentialContext:
    config: RepeatedBirthConfig
    base: SphereField
    master_layout: BasisLayout
    master_support: SupportPairs
    hierarchy_levels: Tensor
    reference_points: Tensor
    reference_normals: Tensor
    colors: Tensor
    cells: list
    state: SceneTransportState
    geometry: BenchmarkGeometry
    geometry_evaluator: Callable[[Tensor], dict[str, float]] | None = None


@dataclass
class ActiveState:
    active_ids: Tensor
    coefficients: Tensor
    points: Tensor
    denominator: Tensor
    images: list[Tensor]
    loss: float
    optimizer_steps: int
    last_relative_improvement: float
    gradient_norm: float
    root_failures: int
    line_search_failures: int = 0
    cg_failures: int = 0
    accepted_step_sizes: tuple[float, ...] = ()
    optimization_loss_pairs: tuple[tuple[float, float], ...] = ()
    cg_iterations_total: int = 0


def _hierarchical_layout(
    base: SphereField,
    master_count: int,
    initial_count: int,
    base_radius: float,
    device: torch.device,
) -> tuple[BasisLayout, Tensor]:
    centers = hierarchical_surface_points(base, master_count, device)
    radii = torch.empty(master_count, dtype=torch.float64, device=device)
    levels = torch.empty(master_count, dtype=torch.long, device=device)
    radii[:initial_count] = base_radius
    levels[:initial_count] = 0
    start = initial_count
    level = 1
    while start < master_count:
        stop = min(2 * start, master_count)
        radii[start:stop] = base_radius * math.sqrt(initial_count / stop)
        levels[start:stop] = level
        start = stop
        level += 1
    return BasisLayout(centers, radii), levels


def _multiscale_support(
    points: Tensor,
    layout: BasisLayout,
    levels: Tensor,
    margin: float,
) -> SupportPairs:
    point_parts: list[Tensor] = []
    basis_parts: list[Tensor] = []
    for level in range(int(levels.max()) + 1):
        ids = torch.nonzero(levels == level, as_tuple=False).flatten()
        block = BasisLayout(layout.centers[ids], layout.radii[ids])
        _, support = make_support(points, block, margin=margin)
        point_parts.append(support.point_ids)
        basis_parts.append(ids[support.basis_ids])
    return SupportPairs.from_pairs(
        torch.cat(point_parts),
        torch.cat(basis_parts),
        points.shape[0],
        layout.count,
    )


def _build_context(
    config: RepeatedBirthConfig, device: torch.device
) -> SequentialContext:
    base = SphereField()
    layout, levels = _hierarchical_layout(
        base,
        config.master_count,
        config.initial_count,
        config.base_support_radius,
        device,
    )
    emitter_count = config.emitters_per_master_basis * config.master_count
    points = hierarchical_surface_points(base, emitter_count, device)
    normals = unit_normals(base, points)
    colors = base.color(points)
    support = _multiscale_support(points, layout, levels, config.support_margin)
    directions = deterministic_directions(
        normals, config.packets_per_emitter, 128.0
    )
    photons = _photon_batch(
        points, directions, colors, config.packets_per_emitter
    )
    maximum_times = torch.full(
        (photons.count,), 4.0, dtype=torch.float64, device=device
    )
    surface_hits, surface_times = first_zero_set_intersections(
        base,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
        chunk_size=8192,
    )
    state = SceneTransportState(
        points,
        normals,
        photons,
        directions,
        surface_hits,
        surface_times,
        torch.ones(emitter_count, dtype=torch.bool, device=device),
    )
    cameras = multiview_cameras(
        "sphere", (config.resolution, config.resolution), device, count=config.views
    )
    geometry = BenchmarkGeometry(base, points, normals, colors)
    view_config = MultiviewConfig(
        resolution=(config.resolution, config.resolution),
        emitters=emitter_count,
        packets_per_emitter=config.packets_per_emitter,
        parameter_count=config.master_count,
        cone_power=128.0,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )
    cells = [project_camera(geometry, camera, state, view_config).cell for camera in cameras]
    return SequentialContext(
        config,
        base,
        layout,
        support,
        levels,
        points,
        normals,
        colors,
        cells,
        state,
        geometry,
    )


def _active_components(
    context: SequentialContext, active_ids: Tensor
) -> tuple[BasisLayout, SupportPairs]:
    inverse = torch.full(
        (context.config.master_count,),
        -1,
        dtype=torch.long,
        device=active_ids.device,
    )
    inverse[active_ids] = torch.arange(active_ids.numel(), device=active_ids.device)
    remapped = inverse[context.master_support.basis_ids]
    keep = remapped >= 0
    support = SupportPairs.from_pairs(
        context.master_support.point_ids[keep],
        remapped[keep],
        context.reference_points.shape[0],
        active_ids.numel(),
    )
    return (
        BasisLayout(
            context.master_layout.centers[active_ids],
            context.master_layout.radii[active_ids],
        ),
        support,
    )


def _evaluate_active(
    context: SequentialContext,
    active_ids: Tensor,
    coefficients: Tensor,
    targets: list[Tensor] | None = None,
    components: tuple[BasisLayout, SupportPairs] | None = None,
) -> ActiveState:
    layout, support = components or _active_components(context, active_ids)
    field = LocalZeroSet(
        context.base,
        layout,
        coefficients,
        context.reference_points,
        context.reference_normals,
        support,
    )
    points, success, _, denominator = field.deform(
        iterations=context.config.deformation_iterations,
        max_offset=context.config.deformation_max_offset,
    )
    images = _images(context.cells, points)
    loss = _image_loss(images, targets) if targets is not None else math.nan
    return ActiveState(
        active_ids,
        coefficients,
        points,
        denominator,
        images,
        loss,
        0,
        0.0,
        math.inf,
        int((~success).sum()),
    )


def _normalize_target(
    context: SequentialContext,
    raw: Tensor,
) -> tuple[Tensor, Tensor, float]:
    template = LocalZeroSet(
        context.base,
        context.master_layout,
        raw,
        context.reference_points,
        context.reference_normals,
        context.master_support,
    )

    def evaluate(scale: float) -> tuple[Tensor, Tensor, float, bool]:
        coefficients = scale * raw
        field = template.with_coefficients(coefficients)
        points, success, displacement, _ = field.deform(
            max_offset=context.config.deformation_max_offset
        )
        rms = float(torch.sqrt((displacement * displacement).mean()))
        return coefficients, points, rms, bool(success.all())

    desired = context.config.target_rms_displacement
    low = 0.0
    high = 0.003
    best: tuple[Tensor, Tensor, float, bool] | None = None
    endpoint = evaluate(high)
    while endpoint[3] and endpoint[2] < desired:
        low = high
        best = endpoint
        high *= 2.0
        endpoint = evaluate(high)
        if high > 0.192:
            raise RuntimeError("sequential target normalization could not find a bracket")
    for _ in range(24):
        middle = 0.5 * (low + high)
        trial = evaluate(middle)
        if trial[3]:
            if best is None or abs(trial[2] - desired) < abs(best[2] - desired):
                best = trial
            if trial[2] < desired:
                low = middle
            else:
                high = middle
        else:
            high = middle
    if best is None or abs(best[2] - desired) > 1e-6 * desired:
        raise RuntimeError("fixed sequential target RMS is unreachable")
    return best[0], best[1], best[2]


def _make_target(
    context: SequentialContext, regime: str, seed: int
) -> dict[str, object]:
    config = context.config
    raw = torch.zeros(
        config.master_count,
        dtype=torch.float64,
        device=context.reference_points.device,
    )
    inactive = torch.arange(
        config.initial_count, config.master_count, device=raw.device
    )
    if regime == "sparse":
        generator = torch.Generator().manual_seed(1907 + seed)
        count = max(32, inactive.numel() // 8)
        selected = inactive[
            torch.randperm(inactive.numel(), generator=generator)[:count].to(raw.device)
        ]
    elif regime == "distributed":
        selected = inactive
    else:
        raise ValueError(f"unknown target regime {regime}")
    phase = 0.41 * seed
    values = torch.sin(1.37 * selected.double() + phase) + 0.45 * torch.cos(
        0.61 * selected.double() - 0.3 * phase
    )
    values /= torch.sqrt((values * values).mean())
    raw[selected] = values
    coefficients, points, rms = _normalize_target(context, raw)
    targets = _images(context.cells, points)
    return {
        "regime": regime,
        "seed": seed,
        "coefficients": coefficients,
        "points": points,
        "images": targets,
        "detail_ids": selected,
        "rms": rms,
        "initial_loss": _image_loss(
            _images(context.cells, context.reference_points), targets
        ),
    }


def _active_jacobians(
    context: SequentialContext,
    state: ActiveState,
    components: tuple[BasisLayout, SupportPairs] | None = None,
) -> list[Tensor]:
    layout, support = components or _active_components(context, state.active_ids)
    return [
        _local_sparse_jacobian(
            layout,
            state.points,
            context.reference_normals,
            state.denominator,
            support,
            cell,
        )
        for cell in context.cells
    ]


def _conjugate_gradient_with_count(
    matrices: list[Tensor],
    right: Tensor,
    config: RepeatedBirthConfig,
) -> tuple[Tensor, int]:
    """Run the existing stateless CG recurrence and expose its iteration count."""
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


def _optimize(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
    steps: int,
) -> tuple[ActiveState, float]:
    started = time.perf_counter()
    coefficients = state.coefficients.clone()
    result = state
    completed = 0
    line_search_failures = state.line_search_failures
    cg_failures = state.cg_failures
    accepted_step_sizes = list(state.accepted_step_sizes)
    loss_pairs = list(state.optimization_loss_pairs)
    cg_iterations_total = state.cg_iterations_total
    components = _active_components(context, state.active_ids)
    for _ in range(steps):
        result = _evaluate_active(
            context, state.active_ids, coefficients, targets, components
        )
        matrices = _active_jacobians(context, result, components)
        gradient = _gradient(
            matrices, result.images, targets, state.active_ids.numel()
        )
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        step, cg_iterations = _conjugate_gradient_with_count(
            matrices, -gradient, context.config
        )
        cg_iterations_total += cg_iterations
        if not bool(torch.isfinite(step).all()):
            cg_failures += 1
            break
        accepted = coefficients
        accepted_loss = result.loss
        accepted_scale = 0.0
        for trial in range(context.config.line_evaluations):
            scale = 0.5**trial
            proposal = (coefficients + scale * step).clamp(
                -context.config.coefficient_limit,
                context.config.coefficient_limit,
            )
            proposal_state = _evaluate_active(
                context, state.active_ids, proposal, targets, components
            )
            if proposal_state.root_failures == 0 and proposal_state.loss < accepted_loss:
                accepted = proposal
                accepted_loss = proposal_state.loss
                accepted_scale = scale
        relative = (result.loss - accepted_loss) / max(result.loss, 1e-30)
        if accepted_loss >= result.loss:
            line_search_failures += 1
        accepted_step_sizes.append(accepted_scale)
        loss_pairs.append((result.loss, accepted_loss))
        coefficients = accepted
        completed += 1
        result.last_relative_improvement = relative
        result.gradient_norm = gradient_norm
        if (
            accepted_loss >= result.loss
            or relative < context.config.convergence_tolerance
            or gradient_norm < 1e-10
        ):
            break
    result = _evaluate_active(
        context, state.active_ids, coefficients, targets, components
    )
    result.optimizer_steps = state.optimizer_steps + completed
    result.last_relative_improvement = relative if completed else 0.0
    result.gradient_norm = gradient_norm if completed else math.inf
    result.line_search_failures = line_search_failures
    result.cg_failures = cg_failures
    result.accepted_step_sizes = tuple(accepted_step_sizes)
    result.optimization_loss_pairs = tuple(loss_pairs)
    result.cg_iterations_total = cg_iterations_total
    torch.cuda.synchronize() if result.points.is_cuda else None
    return result, time.perf_counter() - started


def _candidate_scores(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
) -> tuple[dict[str, Tensor], dict[str, float], float]:
    started = time.perf_counter()
    count = context.config.master_count
    device = state.points.device
    alignment = torch.zeros(count, dtype=torch.float64, device=device)
    norm_squared = torch.zeros_like(alignment)
    affected_pixels = torch.zeros(count, dtype=torch.long, device=device)
    local_residual_squared = torch.zeros_like(alignment)
    affected_views = torch.zeros(count, dtype=torch.long, device=device)
    nnz = 0
    active_pairs = 0
    active_pixels = 0
    for cell, image, target in zip(context.cells, state.images, targets):
        matrix = _local_sparse_jacobian(
            context.master_layout,
            state.points,
            context.reference_normals,
            state.denominator,
            context.master_support,
            cell,
        )
        local = _sparse_column_statistics(matrix, image - target, count)
        alignment += local[0]
        norm_squared += local[1]
        affected_pixels += local[2]
        local_residual_squared += local[3]
        affected_views += local[4].long()
        nnz += matrix._nnz()
        coalesced = matrix.coalesce()
        columns = coalesced.indices()[1]
        active_mask = torch.zeros(count, dtype=torch.bool, device=device)
        active_mask[state.active_ids] = True
        selected = active_mask[columns]
        if bool(selected.any()):
            pixels = torch.div(
                coalesced.indices()[0, selected], 3, rounding_mode="floor"
            )
            encoded = pixels * count + columns[selected]
            active_pairs += int(torch.unique(encoded).numel())
            active_pixels += int(torch.unique(pixels).numel())
        del matrix
    scores = {
        "raw": alignment.abs(),
        "quadratic": alignment.square() / (norm_squared + context.config.damping),
        "local_residual": torch.sqrt(local_residual_squared),
        "jacobian_norm": torch.sqrt(norm_squared),
    }
    inactive = torch.ones(count, dtype=torch.bool, device=device)
    inactive[state.active_ids] = False
    responsive = (scores["jacobian_norm"] > 1e-12) & inactive
    maximum_norm = float(scores["jacobian_norm"][inactive].max())
    diagnostics = {
        "responsive_fraction": float(
            responsive.sum() / max(int(inactive.sum()), 1)
        ),
        "zero_jacobian_fraction": float(
            ((scores["jacobian_norm"] <= 1e-12) & inactive).sum()
            / max(int(inactive.sum()), 1)
        ),
        "candidate_jacobian_nnz": float(nnz),
        "active_parameters_per_affected_pixel": active_pairs
        / max(active_pixels, 1),
        "near_null_candidate_fraction": float(
            (
                (scores["jacobian_norm"] <= 1e-8 * maximum_norm)
                & inactive
            ).sum()
            / max(int(inactive.sum()), 1)
        ),
        "nonfinite_candidate_score_count": float(
            sum(int((~torch.isfinite(value[inactive])).sum()) for value in scores.values())
        ),
        "best_remaining_quadratic": float(scores["quadratic"][inactive].max()),
        "median_affected_pixels": float(
            affected_pixels[responsive].double().median()
        )
        if bool(responsive.any())
        else 0.0,
        "median_affected_views": float(
            affected_views[responsive].double().median()
        )
        if bool(responsive.any())
        else 0.0,
    }
    torch.cuda.synchronize() if state.points.is_cuda else None
    return scores, diagnostics, time.perf_counter() - started


def _choose_candidate(
    method: str,
    scores: dict[str, Tensor],
    active_ids: Tensor,
    generator: torch.Generator,
) -> int:
    count = scores["quadratic"].numel()
    inactive = torch.ones(count, dtype=torch.bool, device=active_ids.device)
    inactive[active_ids] = False
    ids = torch.nonzero(inactive, as_tuple=False).flatten()
    if method == "uniform":
        return int(ids[0])
    if method == "random":
        position = int(torch.randint(ids.numel(), (1,), generator=generator))
        return int(ids[position])
    key = {"quadratic": "quadratic", "raw": "raw", "local_residual": "local_residual"}[method]
    values = torch.where(inactive, scores[key], torch.full_like(scores[key], -math.inf))
    return int(torch.argmax(values))


def _new_only_gain(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
    candidate: int,
) -> float:
    adapter = _oracle_context_adapter(context, state)
    baseline = _ScalingOptimization(
        state.coefficients,
        state.points,
        state.denominator,
        state.images,
        state.loss,
        state.loss,
        0,
        0.0,
        state.gradient_norm,
        state.root_failures,
    )
    oracle_config = KScalingConfig(
        k_values=(state.active_ids.numel(),),
        coefficient_limit=context.config.coefficient_limit,
        oracle_evaluations=9,
    )
    gains, _, _ = _new_only_oracle(
        adapter,
        baseline,
        targets,
        torch.tensor([candidate], device=state.active_ids.device),
        oracle_config,
    )
    return float(gains[candidate])


def _candidate_oracle_checkpoint(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
    scores: dict[str, Tensor],
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    inactive = torch.ones(
        context.config.master_count,
        dtype=torch.bool,
        device=state.active_ids.device,
    )
    inactive[state.active_ids] = False
    available = torch.nonzero(inactive, as_tuple=False).flatten()
    count = min(context.config.oracle_subset_size, available.numel())
    positions = torch.div(
        torch.arange(count, device=available.device) * available.numel(),
        max(count, 1),
        rounding_mode="floor",
    )
    candidate_ids = available[positions]
    adapter = _oracle_context_adapter(context, state)
    baseline = _ScalingOptimization(
        state.coefficients,
        state.points,
        state.denominator,
        state.images,
        state.loss,
        state.loss,
        0,
        0.0,
        state.gradient_norm,
        state.root_failures,
    )
    oracle_config = KScalingConfig(
        k_values=(state.active_ids.numel(),),
        coefficient_limit=context.config.coefficient_limit,
        oracle_evaluations=context.config.oracle_evaluations,
    )
    gains, _, failures = _new_only_oracle(
        adapter, baseline, targets, candidate_ids, oracle_config
    )
    actual = gains[candidate_ids]
    report = {
        "active_k": int(state.active_ids.numel()),
        "candidate_ids": candidate_ids.detach().cpu().tolist(),
        "actual_new_only_gains": actual.detach().cpu().tolist(),
        "oracle_failures": failures,
        "quadratic_spearman": _correlation(
            scores["quadratic"][candidate_ids], actual, True
        ),
        "quadratic_pearson": _correlation(
            scores["quadratic"][candidate_ids], actual, False
        ),
        "raw_spearman": _correlation(scores["raw"][candidate_ids], actual, True),
        "raw_pearson": _correlation(scores["raw"][candidate_ids], actual, False),
    }
    torch.cuda.synchronize() if state.points.is_cuda else None
    return report, time.perf_counter() - started


def _checkpoint_row(
    context: SequentialContext,
    target: dict[str, object],
    method: str,
    selection_seed: int,
    state: ActiveState,
    cumulative_scoring: float,
    cumulative_optimization: float,
    cumulative_total: float,
    peak_allocated: float,
    peak_reserved: float,
    last_birth: dict[str, object] | None,
    diagnostics: dict[str, float],
    continuation_gain: float,
) -> dict[str, object]:
    geometry_rms = float(
        torch.sqrt(
            ((state.points - target["points"]) ** 2).sum(dim=1).mean()  # type: ignore[operator]
        )
    )
    row = {
        "method": method,
        "target_regime": target["regime"],
        "target_seed": target["seed"],
        "selection_seed": selection_seed,
        "active_k": int(state.active_ids.numel()),
        "birth_count": int(state.active_ids.numel() - context.config.initial_count),
        "image_loss": state.loss,
        "residual_norm": math.sqrt(2.0 * state.loss),
        "geometry_rms_error": geometry_rms,
        "optimizer_steps": state.optimizer_steps,
        "last_step_relative_improvement": state.last_relative_improvement,
        "gradient_norm": state.gradient_norm,
        "cumulative_scoring_seconds": cumulative_scoring,
        "cumulative_optimization_seconds": cumulative_optimization,
        "total_runtime_seconds": cumulative_total,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "best_remaining_candidate_score": diagnostics["best_remaining_quadratic"],
        "last_predicted_quadratic": last_birth["predicted_quadratic"]
        if last_birth
        else None,
        "last_predicted_raw": last_birth["predicted_raw"] if last_birth else None,
        "last_realized_birth_gain": last_birth["realized_gain"] if last_birth else None,
        "last_new_only_gain": last_birth["new_only_gain"] if last_birth else None,
        "responsive_candidate_fraction": diagnostics["responsive_fraction"],
        "zero_jacobian_candidate_fraction": diagnostics["zero_jacobian_fraction"],
        "median_affected_pixels": diagnostics["median_affected_pixels"],
        "median_affected_views": diagnostics["median_affected_views"],
        "candidate_jacobian_nnz": diagnostics["candidate_jacobian_nnz"],
        "active_parameters_per_affected_pixel": diagnostics[
            "active_parameters_per_affected_pixel"
        ],
        "active_bases_per_query": _active_components(
            context, state.active_ids
        )[1].count
        / context.reference_points.shape[0],
        "near_null_candidate_fraction": diagnostics[
            "near_null_candidate_fraction"
        ],
        "nonfinite_candidate_score_count": diagnostics[
            "nonfinite_candidate_score_count"
        ],
        "no_birth_continuation_gain": continuation_gain,
        "root_failures": state.root_failures,
        "minimum_denominator_magnitude": float(state.denominator.abs().min()),
        "line_search_failures": state.line_search_failures,
        "cg_failures": state.cg_failures,
        "cg_iterations": state.cg_iterations_total,
    }
    if context.geometry_evaluator is not None:
        geometry_started = time.perf_counter()
        row.update(context.geometry_evaluator(state.points))
        row["geometry_evaluation_seconds"] = time.perf_counter() - geometry_started
    return row


def _run_trajectory(
    context: SequentialContext,
    target: dict[str, object],
    method: str,
    selection_seed: int,
) -> dict[str, object]:
    device = context.reference_points.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    active_ids = torch.arange(context.config.initial_count, device=device)
    initial = _evaluate_active(
        context,
        active_ids,
        torch.zeros(context.config.initial_count, dtype=torch.float64, device=device),
        target["images"],  # type: ignore[arg-type]
    )
    state, optimization_time = _optimize(
        context,
        initial,
        target["images"],  # type: ignore[arg-type]
        context.config.initial_optimization_steps,
    )
    scoring_time = 0.0
    generator = torch.Generator().manual_seed(selection_seed)
    births: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    selected_ids: list[int] = []
    checkpoint_oracles: list[dict[str, object]] = []
    oracle_diagnostic_time = 0.0
    scores, diagnostics, score_seconds = _candidate_scores(
        context, state, target["images"]  # type: ignore[arg-type]
    )
    scoring_time += score_seconds
    continuation, _ = _optimize(
        context, state, target["images"], context.config.post_birth_steps  # type: ignore[arg-type]
    )
    checkpoints.append(
        _checkpoint_row(
            context,
            target,
            method,
            selection_seed,
            state,
            scoring_time,
            optimization_time,
            time.perf_counter() - started,
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
            torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0,
            None,
            diagnostics,
            max(0.0, state.loss - continuation.loss),
        )
    )
    if (
        method == "quadratic"
        and context.config.oracle_subset_size
        and int(state.active_ids.numel()) in context.config.oracle_checkpoints
    ):
        oracle_report, seconds = _candidate_oracle_checkpoint(
            context, state, target["images"], scores  # type: ignore[arg-type]
        )
        checkpoint_oracles.append(oracle_report)
        oracle_diagnostic_time += seconds
    while state.active_ids.numel() < context.config.budget:
        before_loss = state.loss
        candidate = _choose_candidate(
            method, scores, state.active_ids, generator
        )
        new_only_gain = _new_only_gain(
            context, state, target["images"], candidate  # type: ignore[arg-type]
        )
        birth = {
            "step": len(births) + 1,
            "active_k_before": int(state.active_ids.numel()),
            "candidate_id": candidate,
            "hierarchy_level": int(context.hierarchy_levels[candidate]),
            "center": context.master_layout.centers[candidate].detach().cpu().tolist(),
            "support_radius": float(context.master_layout.radii[candidate]),
            "predicted_quadratic": float(scores["quadratic"][candidate]),
            "predicted_raw": float(scores["raw"][candidate]),
            "predicted_local_residual": float(scores["local_residual"][candidate]),
            "jacobian_norm": float(scores["jacobian_norm"][candidate]),
            "new_only_gain": new_only_gain,
            "near_target_detail": bool(
                (target["detail_ids"] == candidate).any()  # type: ignore[union-attr]
            ),
        }
        born_ids = torch.cat(
            (
                state.active_ids,
                torch.tensor([candidate], dtype=torch.long, device=device),
            )
        )
        born_coefficients = torch.cat(
            (state.coefficients, state.coefficients.new_zeros(1))
        )
        at_birth = _evaluate_active(
            context, born_ids, born_coefficients, target["images"]  # type: ignore[arg-type]
        )
        at_birth.optimizer_steps = state.optimizer_steps
        at_birth.line_search_failures = state.line_search_failures
        at_birth.cg_failures = state.cg_failures
        birth["birth_geometry_jump"] = float(
            torch.linalg.vector_norm(at_birth.points - state.points)
        )
        state, seconds = _optimize(
            context,
            at_birth,
            target["images"],  # type: ignore[arg-type]
            context.config.post_birth_steps,
        )
        optimization_time += seconds
        birth["realized_gain"] = max(0.0, before_loss - state.loss)
        birth["loss_after"] = state.loss
        births.append(birth)
        selected_ids.append(candidate)
        scores, diagnostics, score_seconds = _candidate_scores(
            context, state, target["images"]  # type: ignore[arg-type]
        )
        scoring_time += score_seconds
        active_k = int(state.active_ids.numel())
        if active_k in context.config.checkpoints or active_k == context.config.budget:
            continuation, _ = _optimize(
                context,
                state,
                target["images"],  # type: ignore[arg-type]
                context.config.post_birth_steps,
            )
            checkpoints.append(
                _checkpoint_row(
                    context,
                    target,
                    method,
                    selection_seed,
                    state,
                    scoring_time,
                    optimization_time,
                    time.perf_counter() - started - oracle_diagnostic_time,
                    torch.cuda.max_memory_allocated() / 2**20
                    if device.type == "cuda"
                    else 0.0,
                    torch.cuda.max_memory_reserved() / 2**20
                    if device.type == "cuda"
                    else 0.0,
                    birth,
                    diagnostics,
                    max(0.0, state.loss - continuation.loss),
                )
            )
            if (
                method == "quadratic"
                and context.config.oracle_subset_size
                and active_k in context.config.oracle_checkpoints
            ):
                oracle_report, oracle_seconds = _candidate_oracle_checkpoint(
                    context,
                    state,
                    target["images"],  # type: ignore[arg-type]
                    scores,
                )
                checkpoint_oracles.append(oracle_report)
                oracle_diagnostic_time += oracle_seconds
        if state.root_failures:
            break
    predicted_quad = torch.tensor(
        [birth["predicted_quadratic"] for birth in births], dtype=torch.float64
    )
    predicted_raw = torch.tensor(
        [birth["predicted_raw"] for birth in births], dtype=torch.float64
    )
    realized = torch.tensor(
        [birth["realized_gain"] for birth in births], dtype=torch.float64
    )
    correlations = {
        "quadratic_spearman": _correlation(predicted_quad, realized, True),
        "quadratic_pearson": _correlation(predicted_quad, realized, False),
        "raw_spearman": _correlation(predicted_raw, realized, True),
        "raw_pearson": _correlation(predicted_raw, realized, False),
    }
    row = {
        "method": method,
        "target_regime": target["regime"],
        "target_seed": target["seed"],
        "selection_seed": selection_seed,
        "checkpoints": checkpoints,
        "births": births,
        "selected_ids": selected_ids,
        "checkpoint_candidate_oracles": checkpoint_oracles,
        "oracle_diagnostic_seconds": oracle_diagnostic_time,
        "predictor_correlations": correlations,
        "candidate_scoring_fraction": scoring_time
        / max(time.perf_counter() - started - oracle_diagnostic_time, 1e-30),
        "cumulative_scoring_seconds": scoring_time,
        "cumulative_optimization_seconds": optimization_time,
        "total_runtime_seconds": time.perf_counter()
        - started
        - oracle_diagnostic_time,
        "duplicate_births": len(selected_ids) - len(set(selected_ids)),
        "invisible_birth_fraction": statistics.mean(
            float(birth["jacobian_norm"] <= 1e-12) for birth in births
        )
        if births
        else 0.0,
        "births_near_target_detail_fraction": statistics.mean(
            float(birth["near_target_detail"]) for birth in births
        )
        if births
        else 0.0,
        "maximum_birth_geometry_jump": max(
            (float(birth["birth_geometry_jump"]) for birth in births), default=0.0
        ),
    }
    return row


def _rank(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(
        values.numel(), dtype=torch.float64, device=values.device
    )
    return ranks


def _correlation(x: Tensor, y: Tensor, ranked: bool) -> float:
    if x.numel() < 2:
        return math.nan
    if ranked:
        x, y = _rank(x), _rank(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y) / denominator) if denominator > 0 else 0.0


def _run_fixed(
    context: SequentialContext,
    target: dict[str, object],
    active_k: int,
) -> dict[str, object]:
    device = context.reference_points.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state = _evaluate_active(
        context,
        torch.arange(active_k, device=device),
        torch.zeros(active_k, dtype=torch.float64, device=device),
        target["images"],  # type: ignore[arg-type]
    )
    state, optimization_time = _optimize(
        context,
        state,
        target["images"],  # type: ignore[arg-type]
        context.config.fixed_optimization_steps,
    )
    geometry_error = float(
        torch.sqrt(
            ((state.points - target["points"]) ** 2).sum(dim=1).mean()  # type: ignore[operator]
        )
    )
    row = {
        "method": "fixed_space",
        "target_regime": target["regime"],
        "target_seed": target["seed"],
        "selection_seed": 0,
        "active_k": active_k,
        "birth_count": 0,
        "image_loss": state.loss,
        "residual_norm": math.sqrt(2.0 * state.loss),
        "geometry_rms_error": geometry_error,
        "optimizer_steps": state.optimizer_steps,
        "cumulative_scoring_seconds": 0.0,
        "cumulative_optimization_seconds": optimization_time,
        "total_runtime_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else 0.0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if device.type == "cuda"
        else 0.0,
        "best_remaining_candidate_score": None,
        "last_predicted_quadratic": None,
        "last_predicted_raw": None,
        "last_realized_birth_gain": None,
        "last_new_only_gain": None,
        "responsive_candidate_fraction": None,
        "zero_jacobian_candidate_fraction": None,
        "median_affected_pixels": None,
        "median_affected_views": None,
        "candidate_jacobian_nnz": None,
        "active_parameters_per_affected_pixel": None,
        "active_bases_per_query": _active_components(
            context, state.active_ids
        )[1].count
        / context.reference_points.shape[0],
        "near_null_candidate_fraction": None,
        "nonfinite_candidate_score_count": None,
        "no_birth_continuation_gain": None,
        "root_failures": state.root_failures,
        "minimum_denominator_magnitude": float(state.denominator.abs().min()),
        "line_search_failures": state.line_search_failures,
        "cg_failures": state.cg_failures,
    }
    if context.geometry_evaluator is not None:
        row.update(context.geometry_evaluator(state.points))
    return row


def _normalized_auc(rows: list[dict[str, object]], x_key: str, y_key: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    x = torch.tensor([float(row[x_key]) for row in ordered], dtype=torch.float64)
    y = torch.tensor([float(row[y_key]) for row in ordered], dtype=torch.float64)
    if x.numel() < 2 or float(x[-1] - x[0]) <= 0:
        return math.nan
    return float(torch.trapezoid(y / max(float(y[0]), 1e-30), x) / (x[-1] - x[0]))


def _trajectory_summary(
    trajectory: dict[str, object],
    full_reference: dict[str, object],
) -> dict[str, object]:
    rows = trajectory["checkpoints"]
    start_error = float(rows[0]["geometry_rms_error"])
    full_error = float(full_reference["geometry_rms_error"])
    thresholds: dict[str, int | None] = {}
    for fraction in (0.90, 0.95, 0.99):
        threshold = start_error - fraction * (start_error - full_error)
        reached = [
            int(row["active_k"])
            for row in rows
            if float(row["geometry_rms_error"]) <= threshold
        ]
        thresholds[f"{int(100 * fraction)}_percent"] = min(reached) if reached else None
    return {
        "method": trajectory["method"],
        "target_regime": trajectory["target_regime"],
        "target_seed": trajectory["target_seed"],
        "selection_seed": trajectory["selection_seed"],
        "geometry_dof_auc": _normalized_auc(rows, "active_k", "geometry_rms_error"),
        "image_dof_auc": _normalized_auc(rows, "active_k", "image_loss"),
        "geometry_time_auc": _normalized_auc(
            rows, "total_runtime_seconds", "geometry_rms_error"
        ),
        "image_time_auc": _normalized_auc(rows, "total_runtime_seconds", "image_loss"),
        "k_for_full_model_improvement": thresholds,
        "final_geometry_rms_error": rows[-1]["geometry_rms_error"],
        "final_image_loss": rows[-1]["image_loss"],
        "predictor_correlations": trajectory["predictor_correlations"],
        "candidate_scoring_fraction": trajectory["candidate_scoring_fraction"],
    }


def _oracle_context_adapter(
    context: SequentialContext, state: ActiveState
) -> SimpleNamespace:
    active_layout, active_support = _active_components(context, state.active_ids)
    return SimpleNamespace(
        base=context.base,
        current_layout=active_layout,
        current_support=active_support,
        candidate_layout=context.master_layout,
        candidate_support=context.master_support,
        reference_points=context.reference_points,
        reference_normals=context.reference_normals,
        cells=context.cells,
    )


def _run_oracle_greedy(
    config: RepeatedBirthConfig,
) -> dict[str, object] | None:
    if not config.run_oracle:
        return None
    oracle_config = replace(
        config,
        master_count=config.oracle_master_count,
        budget=config.initial_count + config.oracle_births,
        checkpoints=tuple(
            sorted(
                {
                    config.initial_count,
                    config.initial_count + 16,
                    config.initial_count + config.oracle_births,
                }
            )
        ),
        target_regimes=("sparse",),
        target_seeds=(config.target_seeds[0],),
        random_seeds=(0,),
        emitters_per_master_basis=8,
        run_oracle=False,
    )
    context = _build_context(oracle_config, torch.device("cuda"))
    target = _make_target(context, "sparse", oracle_config.target_seeds[0])
    score_comparisons: list[dict[str, object]] = []
    for method in ("quadratic", "raw"):
        trajectory = _run_trajectory(context, target, method, 0)
        score_comparisons.append(
            {
                "method": method,
                "checkpoints": [
                    {
                        "active_k": row["active_k"],
                        "image_loss": row["image_loss"],
                        "geometry_rms_error": row["geometry_rms_error"],
                    }
                    for row in trajectory["checkpoints"]
                ],
                "predictor_correlations": trajectory["predictor_correlations"],
            }
        )
    state = _evaluate_active(
        context,
        torch.arange(oracle_config.initial_count, device="cuda"),
        torch.zeros(oracle_config.initial_count, dtype=torch.float64, device="cuda"),
        target["images"],  # type: ignore[arg-type]
    )
    state, _ = _optimize(
        context,
        state,
        target["images"],  # type: ignore[arg-type]
        oracle_config.initial_optimization_steps,
    )
    births: list[dict[str, object]] = []
    for step in range(oracle_config.oracle_births):
        scores, _, _ = _candidate_scores(context, state, target["images"])  # type: ignore[arg-type]
        inactive = torch.ones(
            oracle_config.master_count, dtype=torch.bool, device="cuda"
        )
        inactive[state.active_ids] = False
        candidate_ids = torch.nonzero(inactive, as_tuple=False).flatten()
        adapter = _oracle_context_adapter(context, state)
        baseline = _ScalingOptimization(
            state.coefficients,
            state.points,
            state.denominator,
            state.images,
            state.loss,
            state.loss,
            0,
            0.0,
            state.gradient_norm,
            state.root_failures,
        )
        oracle_cfg = KScalingConfig(
            k_values=(state.active_ids.numel(),),
            coefficient_limit=oracle_config.coefficient_limit,
            oracle_evaluations=oracle_config.oracle_evaluations,
        )
        gains, best_coefficients, failures = _new_only_oracle(
            adapter, baseline, target["images"], candidate_ids, oracle_cfg  # type: ignore[arg-type]
        )
        selected = int(torch.argmax(torch.nan_to_num(gains, nan=-math.inf)))
        before = state.loss
        state = _evaluate_active(
            context,
            torch.cat((state.active_ids, torch.tensor([selected], device="cuda"))),
            torch.cat(
                (
                    state.coefficients,
                    best_coefficients[selected].reshape(1),
                )
            ),
            target["images"],  # type: ignore[arg-type]
        )
        state, _ = _optimize(
            context,
            state,
            target["images"],  # type: ignore[arg-type]
            oracle_config.post_birth_steps,
        )
        births.append(
            {
                "step": step + 1,
                "candidate_id": selected,
                "actual_new_only_gain": float(gains[selected]),
                "quadratic_score": float(scores["quadratic"][selected]),
                "raw_score": float(scores["raw"][selected]),
                "realized_gain": max(0.0, before - state.loss),
                "oracle_failures": failures,
                "image_loss": state.loss,
                "geometry_rms_error": float(
                    torch.sqrt(
                        ((state.points - target["points"]) ** 2)  # type: ignore[operator]
                        .sum(dim=1)
                        .mean()
                    )
                ),
            }
        )
    return {
        "master_count": oracle_config.master_count,
        "initial_count": oracle_config.initial_count,
        "final_count": int(state.active_ids.numel()),
        "target_regime": "sparse",
        "target_seed": oracle_config.target_seeds[0],
        "score_greedy_comparisons": score_comparisons,
        "births": births,
        "final_image_loss": state.loss,
        "final_geometry_rms_error": births[-1]["geometry_rms_error"],
        "quadratic_vs_realized_spearman": _correlation(
            torch.tensor([row["quadratic_score"] for row in births]),
            torch.tensor([row["realized_gain"] for row in births]),
            True,
        ),
    }


def _aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    keys = sorted(
        {
            (str(row["method"]), str(row["target_regime"]))
            for row in summaries
        }
    )
    for method, regime in keys:
        selected = [
            row
            for row in summaries
            if row["method"] == method and row["target_regime"] == regime
        ]
        result[f"{method}:{regime}"] = {
            "runs": len(selected),
            "geometry_dof_auc_mean": statistics.mean(
                float(row["geometry_dof_auc"]) for row in selected
            ),
            "geometry_dof_auc_std": statistics.pstdev(
                float(row["geometry_dof_auc"]) for row in selected
            ),
            "geometry_time_auc_mean": statistics.mean(
                float(row["geometry_time_auc"]) for row in selected
            ),
            "final_geometry_rms_mean": statistics.mean(
                float(row["final_geometry_rms_error"]) for row in selected
            ),
            "final_geometry_rms_std": statistics.pstdev(
                float(row["final_geometry_rms_error"]) for row in selected
            ),
            "final_image_loss_mean": statistics.mean(
                float(row["final_image_loss"]) for row in selected
            ),
            "candidate_scoring_fraction_mean": statistics.mean(
                float(row["candidate_scoring_fraction"]) for row in selected
            ),
        }
    return result


def _quality_threshold_report(
    targets: list[dict[str, object]],
    trajectories: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
    master_count: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in targets:
        regime = str(target["regime"])
        seed = int(target["seed"])
        fixed = sorted(
            (
                row
                for row in fixed_rows
                if row["target_regime"] == regime and row["target_seed"] == seed
            ),
            key=lambda row: int(row["active_k"]),
        )
        start = next(
            float(row["geometry_rms_error"])
            for row in fixed
            if row["active_k"] == 32
        )
        full = next(
            float(row["geometry_rms_error"])
            for row in fixed
            if row["active_k"] == master_count
        )
        thresholds: dict[str, object] = {}
        for percent in (90, 95, 99):
            threshold = start - percent / 100.0 * (start - full)
            fixed_k = next(
                (
                    int(row["active_k"])
                    for row in fixed
                    if float(row["geometry_rms_error"]) <= threshold
                ),
                None,
            )
            method_k: dict[str, list[int | None]] = {}
            for trajectory in trajectories:
                if (
                    trajectory["target_regime"] != regime
                    or trajectory["target_seed"] != seed
                ):
                    continue
                reached = next(
                    (
                        int(row["active_k"])
                        for row in trajectory["checkpoints"]
                        if float(row["geometry_rms_error"]) <= threshold
                    ),
                    None,
                )
                method_k.setdefault(str(trajectory["method"]), []).append(reached)
            thresholds[str(percent)] = {
                "geometry_rms": threshold,
                "fixed_k": fixed_k,
                "method_k": method_k,
            }
        result[f"{regime}:{seed}"] = {
            "start_geometry_rms": start,
            "full_geometry_rms": full,
            "thresholds": thresholds,
        }
    return result


def _predictor_window_report(
    trajectories: list[dict[str, object]],
) -> dict[str, object]:
    """Measure score calibration early, midway, and late in each trajectory."""
    result: dict[str, object] = {}
    for method in ("quadratic", "raw"):
        selected = [row for row in trajectories if row["method"] == method]
        windows: dict[str, object] = {}
        for name, low, high in (
            ("early", 0.0, 1 / 3),
            ("middle", 1 / 3, 2 / 3),
            ("late", 2 / 3, 1.0),
        ):
            births = []
            for trajectory in selected:
                rows = trajectory["births"]
                start = int(low * len(rows))
                stop = len(rows) if high == 1.0 else int(high * len(rows))
                births.extend(rows[start:stop])
            realized = torch.tensor(
                [float(row["realized_gain"]) for row in births], dtype=torch.float64
            )
            windows[name] = {
                "birth_count": len(births),
                "quadratic_spearman": _correlation(
                    torch.tensor(
                        [float(row["predicted_quadratic"]) for row in births],
                        dtype=torch.float64,
                    ),
                    realized,
                    True,
                ),
                "raw_spearman": _correlation(
                    torch.tensor(
                        [float(row["predicted_raw"]) for row in births],
                        dtype=torch.float64,
                    ),
                    realized,
                    True,
                ),
            }
        result[method] = windows
    return result


def _sequential_stability_report(
    trajectories: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize pathologies without allocating a dense active-set Gram matrix."""
    groups: dict[str, object] = {}
    keys = sorted(
        {(str(row["method"]), str(row["target_regime"])) for row in trajectories}
    )
    for method, regime in keys:
        selected = [
            row
            for row in trajectories
            if row["method"] == method and row["target_regime"] == regime
        ]
        near_duplicate_count = 0
        birth_count = 0
        score_ratios: list[float] = []
        for trajectory in selected:
            births = trajectory["births"]
            birth_count += len(births)
            for index, birth in enumerate(births):
                center = birth["center"]
                radius = float(birth["support_radius"])
                if any(
                    math.dist(center, earlier["center"])
                    < 0.25 * min(radius, float(earlier["support_radius"]))
                    for earlier in births[:index]
                ):
                    near_duplicate_count += 1
            checkpoints = trajectory["checkpoints"]
            initial_score = float(checkpoints[0]["best_remaining_candidate_score"])
            final_score = float(checkpoints[-1]["best_remaining_candidate_score"])
            score_ratios.append(final_score / max(initial_score, 1e-30))
        groups[f"{method}:{regime}"] = {
            "trajectory_count": len(selected),
            "birth_count": birth_count,
            "duplicate_births": sum(int(row["duplicate_births"]) for row in selected),
            "near_duplicate_births": near_duplicate_count,
            "near_duplicate_fraction": near_duplicate_count / max(birth_count, 1),
            "invisible_birth_fraction_mean": statistics.mean(
                float(row["invisible_birth_fraction"]) for row in selected
            ),
            "births_near_target_detail_fraction_mean": statistics.mean(
                float(row["births_near_target_detail_fraction"]) for row in selected
            ),
            "final_to_initial_best_score_ratio_mean": statistics.mean(score_ratios),
            "maximum_birth_geometry_jump": max(
                float(row["maximum_birth_geometry_jump"]) for row in selected
            ),
            "root_failures": sum(
                int(row["checkpoints"][-1]["root_failures"]) for row in selected
            ),
        }
    return {
        "near_duplicate_definition": (
            "a born center lies within 0.25 times the smaller support radius "
            "of an earlier born center"
        ),
        "active_set_conditioning": (
            "not explicitly estimated; the sparse active Jacobian is used only "
            "as an operator and a dense Gram matrix is never materialized"
        ),
        "groups": groups,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    directory: Path,
    trajectories: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
    context: SequentialContext,
    targets: list[dict[str, object]],
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    all_rows = [
        row
        for trajectory in trajectories
        for row in trajectory["checkpoints"]
    ]

    def mean_curve(
        method: str, key: str, regime: str
    ) -> tuple[list[int], list[float]]:
        source = fixed_rows if method == "fixed_space" else all_rows
        levels = sorted(
            {
                int(row["active_k"])
                for row in source
                if row["method"] == method and row["target_regime"] == regime
            }
        )
        return levels, [
            statistics.mean(
                float(row[key])
                for row in source
                if row["method"] == method
                and row["target_regime"] == regime
                and int(row["active_k"]) == level
            )
            for level in levels
        ]

    regimes = tuple(context.config.target_regimes)
    for filename, key, ylabel in (
        ("v03c_geometry_vs_dofs.png", "geometry_rms_error", "geometry RMS error"),
        ("v03c_image_vs_dofs.png", "image_loss", "image loss"),
    ):
        figure, axes = plt.subplots(
            1, len(regimes), figsize=(11.0, 4.4), squeeze=False
        )
        for axis, regime in zip(axes[0], regimes):
            for method in (*SEQUENTIAL_METHODS, "fixed_space"):
                x, y = mean_curve(method, key, regime)
                if x:
                    axis.plot(x, y, marker="o", label=method)
            axis.set_title(regime)
            axis.set_xlabel("active geometry DoFs")
            axis.set_ylabel(ylabel)
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
        axes[0][0].legend(fontsize=7)
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))

    figure, axes = plt.subplots(
        1, len(regimes), figsize=(11.0, 4.4), squeeze=False
    )
    for axis, regime in zip(axes[0], regimes):
        for method in (*SEQUENTIAL_METHODS, "fixed_space"):
            _, x = mean_curve(method, "total_runtime_seconds", regime)
            _, geometry = mean_curve(method, "geometry_rms_error", regime)
            if x:
                axis.plot(x, geometry, marker="o", label=method)
        axis.set_title(regime)
        axis.set_xlabel("cumulative wall-clock time (s)")
        axis.set_ylabel("geometry RMS error")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    path = directory / "v03c_geometry_vs_time.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for axis, key, title in (
        (axes[0], "predicted_quadratic", "quadratic score"),
        (axes[1], "predicted_raw", "raw alignment"),
    ):
        for method in ("quadratic", "raw"):
            births = [
                birth
                for trajectory in trajectories
                if trajectory["method"] == method
                for birth in trajectory["births"]
            ]
            axis.scatter(
                [birth[key] for birth in births],
                [birth["realized_gain"] for birth in births],
                s=8,
                alpha=0.35,
                label=method,
            )
        axis.set_xlabel(title)
        axis.set_ylabel("realized joint gain")
        axis.set_xscale("symlog", linthresh=1e-10)
        axis.set_yscale("symlog", linthresh=1e-10)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "v03c_predicted_vs_realized.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    representative = next(
        trajectory
        for trajectory in trajectories
        if trajectory["method"] == "quadratic"
        and trajectory["target_regime"] == "sparse"
        and trajectory["target_seed"] == context.config.target_seeds[0]
    )
    target = next(
        item
        for item in targets
        if item["regime"] == "sparse"
        and item["seed"] == context.config.target_seeds[0]
    )
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    detail = (
        context.master_layout.centers[target["detail_ids"]]  # type: ignore[index]
        .detach()
        .cpu()
    )
    born = representative["births"]
    for axis, active_k in zip(axes, (32, 64, 128, context.config.budget)):
        count = max(0, active_k - context.config.initial_count)
        ids = [int(row["candidate_id"]) for row in born[:count]]
        axis.scatter(detail[:, 0], detail[:, 2], s=5, alpha=0.25, label="target detail")
        if ids:
            selected = context.master_layout.centers[ids].detach().cpu()
            axis.scatter(selected[:, 0], selected[:, 2], s=10, label="born")
        axis.set_title(f"K={active_k}")
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    path = directory / "v03c_birth_geography.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))
    return paths


def run_repeated_birth_experiment(
    config: RepeatedBirthConfig = RepeatedBirthConfig(),
    *,
    csv_path: Path | None = None,
    json_path: Path | None = None,
    figure_directory: Path | None = None,
) -> dict[str, object]:
    """Run matched repeated-birth and fixed-space CUDA experiments."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    context = _build_context(config, torch.device("cuda"))
    targets = [
        _make_target(context, regime, seed)
        for regime in config.target_regimes
        for seed in config.target_seeds
    ]
    trajectories: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    for target in targets:
        for method in ("quadratic", "raw", "local_residual", "uniform"):
            trajectory = _run_trajectory(context, target, method, 0)
            trajectories.append(trajectory)
            print(
                json.dumps(
                    {
                        "method": method,
                        "regime": target["regime"],
                        "seed": target["seed"],
                        "final_k": trajectory["checkpoints"][-1]["active_k"],
                        "final_geometry_rms": trajectory["checkpoints"][-1][
                            "geometry_rms_error"
                        ],
                        "runtime_seconds": trajectory["total_runtime_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        is_primary_target = (
            target["regime"] == config.target_regimes[0]
            and target["seed"] == config.target_seeds[0]
        )
        random_seeds = (
            config.random_seeds if is_primary_target else config.random_seeds[:1]
        )
        for random_seed in random_seeds:
            trajectory = _run_trajectory(
                context, target, "random", random_seed
            )
            trajectories.append(trajectory)
            print(
                json.dumps(
                    {
                        "method": "random",
                        "regime": target["regime"],
                        "seed": target["seed"],
                        "selection_seed": random_seed,
                        "final_k": trajectory["checkpoints"][-1]["active_k"],
                        "final_geometry_rms": trajectory["checkpoints"][-1][
                            "geometry_rms_error"
                        ],
                        "runtime_seconds": trajectory["total_runtime_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        fixed_levels = sorted(
            set(config.checkpoints).union((config.budget, 512, config.master_count))
        )
        for active_k in fixed_levels:
            if active_k <= config.master_count:
                fixed_rows.append(_run_fixed(context, target, active_k))
    full_lookup = {
        (row["target_regime"], row["target_seed"]): row
        for row in fixed_rows
        if row["active_k"] == config.master_count
    }
    summaries = [
        _trajectory_summary(
            trajectory,
            full_lookup[(trajectory["target_regime"], trajectory["target_seed"])],
        )
        for trajectory in trajectories
    ]
    aggregates = _aggregate_summaries(summaries)
    quadratic_auc = statistics.mean(
        float(row["geometry_dof_auc"])
        for row in summaries
        if row["method"] == "quadratic"
    )
    competitors = [
        float(row["geometry_dof_auc"])
        for row in summaries
        if row["method"] in {"local_residual", "random", "uniform"}
    ]
    repeated_supported = quadratic_auc < statistics.mean(competitors)
    fixed_256 = statistics.mean(
        float(row["geometry_rms_error"])
        for row in fixed_rows
        if row["active_k"] == config.budget
    )
    quadratic_256 = statistics.mean(
        float(row["final_geometry_rms_error"])
        for row in summaries
        if row["method"] == "quadratic"
    )
    parameter_supported = quadratic_256 < fixed_256
    oracle = _run_oracle_greedy(config)
    report: dict[str, object] = {
        "experiment": "v0.3c repeated observation-driven geometry DoF birth",
        "environment": cuda_environment(),
        "configuration": {
            **{
                key: value
                for key, value in config.__dict__.items()
                if key
                not in {
                    "checkpoints",
                    "target_regimes",
                    "target_seeds",
                    "random_seeds",
                }
            },
            "checkpoints": list(config.checkpoints),
            "target_regimes": list(config.target_regimes),
            "target_seeds": list(config.target_seeds),
            "random_seeds": list(config.random_seeds),
            "optimizer": (
                "stateless damped Gauss-Newton/CG; old coefficients preserved and "
                "one zero coefficient appended at birth"
            ),
            "candidate_scoring": (
                "view-streamed sparse local jTr and jTj accumulation; candidate "
                "columns discarded"
            ),
        },
        "targets": [
            {
                "regime": target["regime"],
                "seed": target["seed"],
                "target_rms_displacement": target["rms"],
                "nonzero_target_details": int(
                    target["detail_ids"].numel()  # type: ignore[union-attr]
                ),
                "initial_loss": target["initial_loss"],
            }
            for target in targets
        ],
        "trajectories": trajectories,
        "fixed_space_rows": fixed_rows,
        "trajectory_summaries": summaries,
        "aggregate_metrics": aggregates,
        "fixed_reference_quality_thresholds": _quality_threshold_report(
            targets, trajectories, fixed_rows, config.master_count
        ),
        "predictor_calibration_windows": _predictor_window_report(trajectories),
        "sequential_stability": _sequential_stability_report(trajectories),
        "small_oracle_greedy": oracle,
        "primary_verdict": (
            "REPEATED_OBSERVATION_DRIVEN_BIRTH_SUPPORTED"
            if repeated_supported
            else "REPEATED_OBSERVATION_DRIVEN_BIRTH_NOT_SUPPORTED"
        ),
        "parameter_efficiency_verdict": (
            "PARAMETER_EFFICIENCY_SUPPORTED"
            if parameter_supported
            else "PARAMETER_EFFICIENCY_NOT_SUPPORTED"
        ),
        "verdict_basis": {
            "quadratic_geometry_dof_auc_mean": quadratic_auc,
            "non_derivative_selector_geometry_dof_auc_mean": statistics.mean(competitors),
            "quadratic_final_geometry_rms_mean": quadratic_256,
            "fixed_space_same_k_geometry_rms_mean": fixed_256,
        },
    }
    rows = [
        row
        for trajectory in trajectories
        for row in trajectory["checkpoints"]
    ] + fixed_rows
    if csv_path is not None:
        _write_csv(csv_path, rows)
    if figure_directory is not None:
        report["figures"] = _write_figures(
            figure_directory, trajectories, fixed_rows, context, targets
        )
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = {
            "csv": str(csv_path) if csv_path else None,
            "json": str(json_path),
            "figures": report.get("figures", []),
        }
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def sequential_cpu_verification() -> dict[str, object]:
    """Exercise true zero-jump birth and inactive-pool semantics on a tiny case."""
    config = RepeatedBirthConfig(
        master_count=64,
        initial_count=32,
        budget=34,
        checkpoints=(32, 33, 34),
        target_regimes=("sparse",),
        target_seeds=(5,),
        random_seeds=(1,),
        views=2,
        resolution=16,
        emitters_per_master_basis=2,
        packets_per_emitter=2,
        initial_optimization_steps=1,
        post_birth_steps=1,
        fixed_optimization_steps=1,
        line_evaluations=2,
        cg_iterations=2,
        run_oracle=False,
    )
    context = _build_context(config, torch.device("cpu"))
    target = _make_target(context, "sparse", 5)
    trajectory = _run_trajectory(context, target, "quadratic", 0)
    selected = trajectory["selected_ids"]
    report = {
        "births_completed": len(selected),
        "unique_births": len(set(selected)),
        "maximum_birth_geometry_jump": trajectory["maximum_birth_geometry_jump"],
        "final_active_k": trajectory["checkpoints"][-1]["active_k"],
        "root_failures": trajectory["checkpoints"][-1]["root_failures"],
    }
    if not (
        report["births_completed"] == 2
        and report["unique_births"] == 2
        and report["maximum_birth_geometry_jump"] < 1e-12
        and report["final_active_k"] == 34
        and report["root_failures"] == 0
    ):
        raise AssertionError(f"sequential birth verification failed: {report}")
    return report
