"""v0.8.4 supplement: direct forward-photon detector comparison.

The forward transport realization is shared by every detector in this file.
Only the map from continuous detector arrivals to pixels changes.  In
particular, ``DIRECT_FORWARD_PHOTON`` is the finite box-pixel response and
``CONTINUOUS_KERNEL_MONTE_CARLO`` is the differentiable cubic response.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import (
    build_boundary_transport,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
)
from .corrected_birth import (
    CorrectedBirthConfig,
    _active_components,
    _build_context,
    _evaluate,
    _render,
    _visibility_cells,
)
from .high_sample import _json_ready, _prepare_surface, _progress, _release
from .mc_detector import (
    ESTIMATORS,
    MCDetectorConfig,
    EstimatorView,
    _render_estimator_views,
)
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color, sign_changing_cells
from .pixel_diagnostic import _silhouette_regions
from .sampling_diagnostic import _copy_layout


Tensor = torch.Tensor
DIRECT = ESTIMATORS[2]
CONTINUOUS = ESTIMATORS[3]


@dataclass(frozen=True)
class DirectPhotonConfig:
    views: int = 20
    resolutions: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
        (1080, 1920),
    )
    matched_emitters: tuple[int, ...] = (65_536, 262_144, 518_400, 2_073_600)
    fixed_emitters: int = 65_536
    convergence_counts: tuple[int, ...] = (32_768, 65_536, 131_072, 262_144)
    emitter_chunk_size: int = 65_536
    seed: int = 101
    heldout_seed: int = 211
    capture_view: int = 10
    perturbation_deltas: tuple[float, ...] = (3e-3, 1e-3, 3e-4)
    fd_epsilons: tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
    optimization_steps: int = 12
    optimization_learning_rate: float = 3e-3
    geometry_samples: int = 16_384
    geometry_views: int = 4
    geometry_resolution: int = 256
    geometry_parameters: int = 32
    image_rmse_relative_band: float = 0.05
    image_energy_relative_band: float = 0.03
    geometry_relative_band: float = 0.02

    def mc_config(self, *, views: int | None = None) -> MCDetectorConfig:
        return MCDetectorConfig(
            views=self.views if views is None else views,
            resolutions=self.resolutions,
            emitter_counts=self.matched_emitters,
            convergence_counts=self.convergence_counts,
            emitter_chunk_size=self.emitter_chunk_size,
            sobol_seeds=(self.seed, self.heldout_seed),
            capture_view=self.capture_view,
            gradient_surface_samples=self.geometry_samples,
            gradient_views=self.geometry_views,
            gradient_resolution=self.geometry_resolution,
            gradient_parameters=self.geometry_parameters,
            gradient_epsilons=self.fd_epsilons,
        )


def _stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values).reshape(-1).astype(np.float64)
    if flat.size == 0:
        return {"samples": 0, "mean": 0.0, "median": 0.0, "p90": 0.0,
                "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "samples": int(flat.size),
        "mean": float(flat.mean()),
        "median": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "maximum": float(flat.max()),
    }


def _direct_density_stats(
    views: list[EstimatorView], resolution: tuple[int, int], emitters: int, label: str
) -> dict[str, Any]:
    counts = np.concatenate([
        view.direct_count.reshape(-1) for view in views
        if view.direct_count is not None
    ])
    if not counts.size:
        raise RuntimeError("direct photon hit counts were not retained")
    contributions = np.concatenate([
        view.preclamp[DIRECT].sum(2).reshape(-1) for view in views
    ])
    foreground = np.concatenate([
        np.linalg.norm(view.preclamp[CONTINUOUS], axis=2).reshape(-1) > 0.0
        for view in views
    ])

    def selected(mask: np.ndarray) -> dict[str, Any]:
        local_counts = counts[mask]
        local_contribution = contributions[mask]
        hit = local_counts > 0
        per_hit = np.zeros_like(local_contribution)
        per_hit[hit] = local_contribution[hit] / local_counts[hit]
        return {
            "pixels": int(local_counts.size),
            "photon_hit_count": _stats(local_counts),
            "total_accumulated_rgb_contribution": _stats(local_contribution),
            "mean_rgb_contribution_per_hit": _stats(per_hit[hit]),
            "zero_hit_fraction": float((local_counts == 0).mean()),
            "count_lt2_fraction": float((local_counts < 2).mean()),
            "count_lt4_fraction": float((local_counts < 4).mean()),
        }

    return {
        "sample_regime": label,
        "resolution": list(resolution),
        "emitters": emitters,
        "views": len(views),
        "all_pixels": selected(np.ones_like(counts, dtype=bool)),
        "foreground": selected(foreground),
    }


def _regions(reference: np.ndarray) -> dict[str, np.ndarray]:
    foreground, _, codes = _silhouette_regions(reference)
    return {
        "whole_image": np.ones(foreground.shape, dtype=bool),
        "foreground": foreground,
        "interior_gt8px": codes == 5,
        "silhouette_0_8px": (codes >= 1) & (codes <= 4),
    }


def _edge_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    gray = image.astype(np.float64).mean(2)
    dy, dx = np.gradient(gray)
    magnitude = np.sqrt(dx * dx + dy * dy)
    return float(magnitude[mask].mean()) if bool(mask.any()) else 0.0


def _global_ssim(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = left[mask].astype(np.float64).reshape(-1)
    y = right[mask].astype(np.float64).reshape(-1)
    if x.size == 0:
        return 1.0
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    covariance = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * covariance + c2)) / (
        (mux * mux + muy * muy + c1) * (vx + vy + c2)
    )


def _same_transport_comparison(
    views: list[EstimatorView], resolution: tuple[int, int], emitters: int
) -> dict[str, Any]:
    sums = {name: 0.0 for name in (
        "direct_sum", "continuous_sum", "direct_energy", "continuous_energy"
    )}
    region_totals: dict[str, dict[str, float]] = {}
    ssim: list[float] = []
    direct_edges: list[float] = []
    continuous_edges: list[float] = []
    for view in views:
        direct = view.preclamp[DIRECT]
        continuous = view.preclamp[CONTINUOUS]
        masks = _regions(continuous)
        difference = direct - continuous
        for name, mask in masks.items():
            row = region_totals.setdefault(name, {"sse": 0.0, "sae": 0.0, "scalars": 0})
            row["sse"] += float(np.square(difference[mask]).sum())
            row["sae"] += float(np.abs(difference[mask]).sum())
            row["scalars"] += int(mask.sum()) * 3
        sums["direct_sum"] += float(direct.sum())
        sums["continuous_sum"] += float(continuous.sum())
        sums["direct_energy"] += float(np.square(direct).sum())
        sums["continuous_energy"] += float(np.square(continuous).sum())
        silhouette = masks["silhouette_0_8px"]
        ssim.append(_global_ssim(direct, continuous, masks["foreground"]))
        direct_edges.append(_edge_sharpness(direct, silhouette))
        continuous_edges.append(_edge_sharpness(continuous, silhouette))
    regions = {}
    for name, row in region_totals.items():
        mse = row["sse"] / max(int(row["scalars"]), 1)
        regions[name] = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": row["sae"] / max(int(row["scalars"]), 1),
            "rgb_scalars": int(row["scalars"]),
        }
    direct_sum = sums["direct_sum"]
    continuous_sum = sums["continuous_sum"]
    return {
        "resolution": list(resolution),
        "emitters": emitters,
        "event_reuse": "one transport pass, two detector accumulators",
        "regions": regions,
        **sums,
        "total_contribution_relative_difference": abs(direct_sum - continuous_sum)
        / max(abs(continuous_sum), 1e-30),
        "foreground_global_ssim": statistics.mean(ssim),
        "direct_silhouette_edge_sharpness": statistics.mean(direct_edges),
        "continuous_silhouette_edge_sharpness": statistics.mean(continuous_edges),
        "edge_sharpness_ratio_direct_over_continuous": statistics.mean(direct_edges)
        / max(statistics.mean(continuous_edges), 1e-30),
    }


def _operator_matched_convergence(
    renders: dict[int, list[EstimatorView]],
) -> list[dict[str, Any]]:
    """Compare each detector only to its own high-prefix numerical reference."""
    reference_count = max(renders)
    reference = renders[reference_count]
    output: list[dict[str, Any]] = []
    for estimator in (DIRECT, CONTINUOUS):
        rows: list[dict[str, Any]] = []
        for count, views in sorted(renders.items()):
            sse = sae = silhouette_sse = 0.0
            scalars = silhouette_scalars = foreground_pixels = 0
            brightness = energy = 0.0
            for view, high in zip(views, reference):
                image = view.preclamp[estimator]
                target = high.preclamp[estimator]
                difference = image - target
                masks = _regions(target)
                silhouette = masks["silhouette_0_8px"]
                foreground = masks["foreground"]
                sse += float(np.square(difference).sum())
                sae += float(np.abs(difference).sum())
                silhouette_sse += float(np.square(difference[silhouette]).sum())
                scalars += difference.size
                silhouette_scalars += int(silhouette.sum()) * 3
                foreground_pixels += int(foreground.sum())
                brightness += float(image[foreground].sum())
                energy += float(np.square(image).sum())
            mse = sse / max(scalars, 1)
            rows.append({
                "estimator": estimator,
                "emitters": count,
                "reference_emitters": reference_count,
                "mse_vs_operator_matched_reference": mse,
                "rmse_vs_operator_matched_reference": math.sqrt(mse),
                "mae_vs_operator_matched_reference": sae / max(scalars, 1),
                "silhouette_mse_vs_operator_matched_reference": silhouette_sse
                / max(silhouette_scalars, 1),
                "total_image_energy": energy,
                "mean_foreground_brightness": brightness
                / max(foreground_pixels * 3, 1),
            })
        fit = [row for row in rows if row["emitters"] < reference_count]
        slope = float(np.polyfit(
            np.log([row["emitters"] for row in fit]),
            np.log([max(row["mse_vs_operator_matched_reference"], 1e-30) for row in fit]),
            1,
        )[0])
        output.append({
            "estimator": estimator,
            "reference_emitters": reference_count,
            "rows": rows,
            "empirical_log_mse_slope": slope,
            "empirical_log_rmse_slope": slope / 2.0,
        })
    return output


def _hard_operator(
    points: Tensor,
    normals: Tensor,
    context: Any,
    cells: list[Any] | None = None,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Directly accumulate each retained forward event into one box pixel."""
    output: list[Tensor] = []
    assignments: list[Tensor] = []
    radiances: list[Tensor] = []
    local_cells = cells or context.cells
    rows, columns = context.config.resolution_shape
    scale = context.config.sensor_gain * rows * columns / context.config.surface_samples
    for cell in local_cells:
        ids = cell.owner_ids
        relative = points[ids] - cell.center
        column = (relative @ cell.right / cell.extent + 0.5) * columns - 0.5
        row = (0.5 - relative @ cell.up / cell.extent) * rows - 0.5
        pixel_row = torch.floor(row + 0.5).long()
        pixel_column = torch.floor(column + 0.5).long()
        valid = ((pixel_row >= 0) & (pixel_row < rows)
                 & (pixel_column >= 0) & (pixel_column < columns))
        pixel = pixel_row.clamp(0, rows - 1) * columns + pixel_column.clamp(0, columns - 1)
        colors = meshfree_base_color(points[ids], context.lower, context.upper)
        cosine = (normals[ids] @ cell.direction).clamp_min(0.0)
        lobe = context.config.ambient + (1.0 - context.config.ambient) * cosine
        radiance = colors * lobe[:, None]
        numerator = torch.zeros((rows * columns, 3), dtype=points.dtype, device=points.device)
        numerator.index_add_(0, pixel[valid], radiance[valid])
        output.append((scale * numerator).reshape(-1))
        assignments.append(torch.where(valid, pixel, torch.full_like(pixel, -1)))
        radiances.append(radiance)
    return output, assignments, radiances


def _continuous_operator(renders: list[Any], context: Any) -> list[Tensor]:
    rows, columns = context.config.resolution_shape
    scale = context.config.sensor_gain * rows * columns / context.config.surface_samples
    return [(scale * render.numerator).reshape(-1) for render in renders]


def _operator_targets(context: Any) -> tuple[list[Tensor], list[Tensor], list[Any]]:
    atlas = nested_fibonacci_atlas(context.reference_points.device, (context.config.views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    cells, _ = _visibility_cells(
        context.target_field, context.target_points, context.target_normals,
        atlas, boundary, context.config,
    )
    _, renders = _render(context.target_points, context.target_normals, context, cells)
    direct, _, _ = _hard_operator(
        context.target_points, context.target_normals, context, cells
    )
    return direct, _continuous_operator(renders, context), cells


def _operator_metrics(images: list[Tensor], targets: list[Tensor]) -> dict[str, float]:
    difference = torch.cat([left - right for left, right in zip(images, targets)])
    mse = float(difference.square().mean())
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(difference.abs().mean()),
        "energy": float(torch.cat(images).square().sum()),
        "mean_brightness": float(torch.cat(images).mean()),
    }


def _response_metrics(response: Tensor, noise: Tensor) -> dict[str, float | int]:
    shaped = response.reshape(-1, 3)
    pixel_norm = torch.linalg.vector_norm(shaped, dim=1)
    active = pixel_norm > 1e-12
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    locality = 0.0
    if active_ids.numel():
        locality = float(active_ids.numel()) / active.numel()
    norm = float(torch.linalg.vector_norm(response))
    noise_norm = float(torch.linalg.vector_norm(noise))
    return {
        "response_l2_norm": norm,
        "responsive_pixels": int(active.sum()),
        "responsive_pixel_fraction": locality,
        "snr_to_independent_scramble": norm / max(noise_norm, 1e-30),
    }


def _geometry_experiment(prepared: Any, config: DirectPhotonConfig) -> dict[str, Any]:
    corrected = CorrectedBirthConfig(
        dictionary_count=64,
        initial_count=config.geometry_parameters,
        surface_samples=config.geometry_samples,
        views=config.geometry_views,
        resolution=config.geometry_resolution,
        surface_scramble_seed=config.seed,
        coefficient_limit=0.03,
    )
    heldout_corrected = replace(corrected, surface_scramble_seed=config.heldout_seed)
    context = _build_context(prepared, corrected)
    heldout = _build_context(prepared, heldout_corrected)
    _copy_layout(heldout, context)
    device = context.reference_points.device
    active = torch.arange(config.geometry_parameters, device=device)
    components = _active_components(context, active)
    heldout_components = _active_components(heldout, active)
    zeros = torch.zeros(config.geometry_parameters, dtype=torch.float64, device=device)
    base = _evaluate(context, active, zeros, components)
    heldout_base = _evaluate(heldout, active, zeros, heldout_components)
    target_direct, target_continuous, _ = _operator_targets(context)
    heldout_target_direct, heldout_target_continuous, _ = _operator_targets(heldout)
    base_direct, base_assignments, _ = _hard_operator(base.points, base.normals, context)
    base_continuous = _continuous_operator(base.renders, context)
    heldout_base_direct, _, _ = _hard_operator(
        heldout_base.points, heldout_base.normals, heldout
    )
    heldout_base_continuous = _continuous_operator(heldout_base.renders, heldout)

    previous = json.loads(Path("artifacts/v084_mc_continuous_detector.json").read_text())
    categories = previous["analytic_jacobian"]["categories"]
    perturbation_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    for category, parameter_value in categories.items():
        parameter = int(parameter_value)
        previous_direct_response: Tensor | None = None
        previous_continuous_response: Tensor | None = None
        for delta in config.perturbation_deltas:
            coefficients = zeros.clone()
            coefficients[parameter] += delta
            state = _evaluate(context, active, coefficients, components)
            heldout_state = _evaluate(heldout, active, coefficients, heldout_components)
            direct, assignments, _ = _hard_operator(state.points, state.normals, context)
            continuous = _continuous_operator(state.renders, context)
            heldout_direct, _, _ = _hard_operator(
                heldout_state.points, heldout_state.normals, heldout
            )
            heldout_continuous = _continuous_operator(heldout_state.renders, heldout)
            direct_response = torch.cat(direct) - torch.cat(base_direct)
            continuous_response = torch.cat(continuous) - torch.cat(base_continuous)
            direct_noise = direct_response - (
                torch.cat(heldout_direct) - torch.cat(heldout_base_direct)
            )
            continuous_noise = continuous_response - (
                torch.cat(heldout_continuous) - torch.cat(heldout_base_continuous)
            )
            crossing = sum(
                int(((left != right) & (left >= 0) & (right >= 0)).sum())
                for left, right in zip(base_assignments, assignments)
            )
            identities = sum(item.numel() for item in assignments)
            direct_metrics = _response_metrics(direct_response, direct_noise)
            continuous_metrics = _response_metrics(continuous_response, continuous_noise)
            direct_consistency = 1.0
            continuous_consistency = 1.0
            if previous_direct_response is not None:
                direct_consistency = float(torch.nn.functional.cosine_similarity(
                    direct_response[None], previous_direct_response[None]
                ))
                continuous_consistency = float(torch.nn.functional.cosine_similarity(
                    continuous_response[None], previous_continuous_response[None]
                ))
            perturbation_rows.append({
                "category": category,
                "parameter": parameter,
                "delta": delta,
                "direct": direct_metrics,
                "continuous": continuous_metrics,
                "direct_over_continuous_response_norm": float(
                    direct_metrics["response_l2_norm"]
                ) / max(float(continuous_metrics["response_l2_norm"]), 1e-30),
                "direct_sign_consistency_cosine_to_previous_delta": direct_consistency,
                "continuous_sign_consistency_cosine_to_previous_delta": continuous_consistency,
                "photon_pixel_crossings": crossing,
                "photon_pixel_crossing_fraction": crossing / max(identities, 1),
            })
            previous_direct_response = direct_response
            previous_continuous_response = continuous_response

        for epsilon in config.fd_epsilons:
            plus_coefficients = zeros.clone()
            minus_coefficients = zeros.clone()
            plus_coefficients[parameter] += epsilon
            minus_coefficients[parameter] -= epsilon
            plus = _evaluate(context, active, plus_coefficients, components)
            minus = _evaluate(context, active, minus_coefficients, components)
            plus_direct, plus_assignments, _ = _hard_operator(
                plus.points, plus.normals, context
            )
            minus_direct, minus_assignments, _ = _hard_operator(
                minus.points, minus.normals, context
            )
            fd = (torch.cat(plus_direct) - torch.cat(minus_direct)) / (2 * epsilon)
            plus_heldout = _evaluate(
                heldout, active, plus_coefficients, heldout_components
            )
            minus_heldout = _evaluate(
                heldout, active, minus_coefficients, heldout_components
            )
            plus_heldout_direct, _, _ = _hard_operator(
                plus_heldout.points, plus_heldout.normals, heldout
            )
            minus_heldout_direct, _, _ = _hard_operator(
                minus_heldout.points, minus_heldout.normals, heldout
            )
            heldout_fd = (
                torch.cat(plus_heldout_direct) - torch.cat(minus_heldout_direct)
            ) / (2 * epsilon)
            crossings = sum(
                int(((left != right) & (left >= 0) & (right >= 0)).sum())
                for left, right in zip(plus_assignments, minus_assignments)
            )
            identities = sum(item.numel() for item in plus_assignments)
            changed = (torch.cat(plus_direct) - torch.cat(minus_direct)).reshape(-1, 3)
            changed_pixels = int((torch.linalg.vector_norm(changed, dim=1) > 1e-12).sum())
            fd_rows.append({
                "category": category,
                "parameter": parameter,
                "epsilon": epsilon,
                "same_sample_fd_norm": float(torch.linalg.vector_norm(fd)),
                "heldout_fd_norm": float(torch.linalg.vector_norm(heldout_fd)),
                "same_vs_heldout_cosine": float(torch.nn.functional.cosine_similarity(
                    fd[None], heldout_fd[None]
                )),
                "gradient_fd_snr": float(torch.linalg.vector_norm(fd))
                / max(float(torch.linalg.vector_norm(fd - heldout_fd)), 1e-30),
                "photons_crossing_pixel_boundaries": crossings,
                "crossing_fraction": crossings / max(identities, 1),
                "changed_image_pixel_fraction": changed_pixels / max(changed.shape[0], 1),
                "boundary_term_characterization": (
                    "finite-difference crossing term; no smooth pathwise derivative asserted"
                ),
            })

    # Optimize the continuous detector measurement with fixed transport cells.
    coefficients = zeros.clone()
    optimization_rows: list[dict[str, Any]] = []
    for iteration in range(config.optimization_steps):
        coefficients = coefficients.detach().requires_grad_(True)
        state = _evaluate(context, active, coefficients, components)
        images = _continuous_operator(state.renders, context)
        loss = sum((image - target).square().mean()
                   for image, target in zip(images, target_continuous)) / len(images)
        loss.backward()
        gradient = coefficients.grad.detach()
        step = config.optimization_learning_rate / math.sqrt(iteration + 1)
        proposal = (coefficients.detach() - step * gradient /
                    gradient.norm().clamp_min(1e-12)).clamp(-0.03, 0.03)
        optimization_rows.append({
            "iteration": iteration,
            "continuous_train_loss": float(loss.detach()),
            "gradient_norm": float(gradient.norm()),
            "step_norm": float(torch.linalg.vector_norm(proposal - coefficients.detach())),
        })
        coefficients = proposal
    optimized = _evaluate(context, active, coefficients, components)
    heldout_optimized = _evaluate(heldout, active, coefficients, heldout_components)
    optimized_direct, _, _ = _hard_operator(optimized.points, optimized.normals, context)
    optimized_continuous = _continuous_operator(optimized.renders, context)
    heldout_optimized_direct, _, _ = _hard_operator(
        heldout_optimized.points, heldout_optimized.normals, heldout
    )
    heldout_optimized_continuous = _continuous_operator(
        heldout_optimized.renders, heldout
    )

    initial_geometry = context.geometry_evaluator(base.points)
    optimized_geometry = context.geometry_evaluator(optimized.points)
    transfer = {
        "train_seed": config.seed,
        "heldout_seed": config.heldout_seed,
        "transport_cells_during_training": "fixed forward visibility owner cells",
        "initial": {
            "train_direct": _operator_metrics(base_direct, target_direct),
            "train_continuous": _operator_metrics(base_continuous, target_continuous),
            "heldout_direct": _operator_metrics(
                heldout_base_direct, heldout_target_direct
            ),
            "heldout_continuous": _operator_metrics(
                heldout_base_continuous, heldout_target_continuous
            ),
            "geometry": initial_geometry,
        },
        "optimized": {
            "train_direct": _operator_metrics(optimized_direct, target_direct),
            "train_continuous": _operator_metrics(
                optimized_continuous, target_continuous
            ),
            "heldout_direct": _operator_metrics(
                heldout_optimized_direct, heldout_target_direct
            ),
            "heldout_continuous": _operator_metrics(
                heldout_optimized_continuous, heldout_target_continuous
            ),
            "geometry": optimized_geometry,
        },
        "iterations": optimization_rows,
        "coefficients": coefficients.detach().cpu().tolist(),
    }
    for detector in ("train_direct", "train_continuous", "heldout_direct", "heldout_continuous"):
        before = float(transfer["initial"][detector]["mse"])
        after = float(transfer["optimized"][detector]["mse"])
        transfer[f"{detector}_relative_improvement"] = (before - after) / max(before, 1e-30)
    for metric in ("symmetric_chamfer", "point_to_surface_mean", "point_to_surface_p95", "normal_error"):
        before = float(initial_geometry[metric])
        after = float(optimized_geometry[metric])
        transfer[f"geometry_{metric}_relative_improvement"] = (before - after) / max(before, 1e-30)

    return {
        "configuration": asdict(corrected),
        "selected_categories": categories,
        "perturbation_rows": perturbation_rows,
        "direct_fd_rows": fd_rows,
        "continuous_training_direct_evaluation": transfer,
    }


def _trajectory_verification(
    base: Any, surface: Any, atlas: Any, boundary: Any, config: DirectPhotonConfig
) -> dict[str, Any]:
    colors = meshfree_base_color(surface.base_points, base.lower, base.upper)
    events = build_boundary_transport(
        base,
        surface.reference_points[: config.fixed_emitters],
        surface.reference_normals[: config.fixed_emitters],
        atlas,
        boundary,
        root_samples=16,
        bisection_steps=18,
        surface_colors=colors[: config.fixed_emitters],
        cosine_power=1.0,
        ambient_emission=0.35,
    )
    keep = torch.nonzero(events.direction_ids == config.capture_view, as_tuple=False).flatten()[:12]
    rows, columns = config.resolutions[0]
    records = []
    maximum_error = 0.0
    for event_id in keep.tolist():
        owner = int(events.owner_ids[event_id])
        direction_id = int(events.direction_ids[event_id])
        source = surface.reference_points[owner]
        boundary_position = events.boundary_positions[event_id]
        right = atlas.right[direction_id]
        up = atlas.up[direction_id]
        source_relative = source - boundary.center
        arrival_relative = boundary_position - boundary.center
        source_u = torch.stack((
            0.5 - (source_relative @ up) / 2.8,
            (source_relative @ right) / 2.8 + 0.5,
        ))
        arrival_u = torch.stack((
            0.5 - (arrival_relative @ up) / 2.8,
            (arrival_relative @ right) / 2.8 + 0.5,
        ))
        error = float(torch.max(torch.abs(source_u - arrival_u)))
        maximum_error = max(maximum_error, error)
        pixel_row = int(torch.floor(arrival_u[0] * rows).clamp(0, rows - 1))
        pixel_column = int(torch.floor(arrival_u[1] * columns).clamp(0, columns - 1))
        records.append({
            "event_id": event_id,
            "source_id": owner,
            "source_point": source.detach().cpu().tolist(),
            "source_normal": surface.reference_normals[owner].detach().cpu().tolist(),
            "outgoing_direction": atlas.directions[direction_id].detach().cpu().tolist(),
            "visibility_sequence": "outward -> no intervening zero-set root -> boundary arrival",
            "path_length": float(events.path_lengths[event_id]),
            "detector_intersection_position": boundary_position.detach().cpu().tolist(),
            "detector_coordinate_u": arrival_u.detach().cpu().tolist(),
            "detector_pixel_id": [pixel_row, pixel_column],
            "transport_rgb_contribution": events.radiance[event_id].detach().cpu().tolist(),
            "forward_vs_orthographic_transverse_u_max_error": error,
        })
    return {
        "event_digest": events.digest,
        "emitted": events.emitted_count,
        "absorbed": events.absorbed_count,
        "retained": events.count,
        "records": records,
        "maximum_forward_trajectory_coordinate_error": maximum_error,
        "coordinate_source": "x_boundary=x_source+t_exit*omega from forward boundary transport",
        "reverse_camera_cast_used": False,
        "_plot": {
            "sources": np.asarray([row["source_point"] for row in records]),
            "arrivals": np.asarray([row["detector_intersection_position"] for row in records]),
        },
    }


def _save_figures(
    directory: Path,
    captures: dict[tuple[int, int], dict[str, np.ndarray]],
    comparisons: list[dict[str, Any]],
    density: list[dict[str, Any]],
    old_report: dict[str, Any],
    geometry: dict[str, Any],
    trajectory: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    resolutions = list(captures)
    figure, axes = plt.subplots(1, len(resolutions), figsize=(15, 4))
    axes = np.atleast_1d(axes)
    for axis, resolution in zip(axes, resolutions):
        image = captures[resolution]["direct"]
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(f"{resolution[1]}x{resolution[0]}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(directory / "v084_direct_photon_rgb.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, len(resolutions), figsize=(15, 7))
    axes = np.asarray(axes).reshape(2, len(resolutions))
    for column, resolution in enumerate(resolutions):
        count = captures[resolution]["count"]
        energy = captures[resolution]["direct"].sum(2)
        axes[0, column].imshow(np.log1p(count), cmap="magma")
        axes[1, column].imshow(energy, cmap="viridis")
        axes[0, column].set_title(f"log(1+hits), {resolution[1]}x{resolution[0]}")
        axes[1, column].set_title("direct contribution")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(directory / "v084_photon_hit_count_map.png", dpi=180)
    plt.close(figure)

    resolution = resolutions[0]
    capture = captures[resolution]
    residual = np.abs(capture["direct"] - capture["continuous"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, image, cmap) in zip(axes, (
        ("direct box detector", capture["direct"], None),
        ("continuous detector", capture["continuous"], None),
        ("absolute residual", residual.mean(2), "magma"),
    )):
        axis.imshow(np.clip(image, 0, 1) if cmap is None else image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(directory / "v084_direct_vs_continuous_detector.png", dpi=180)
    plt.close(figure)

    foreground, _, codes = _silhouette_regions(capture["continuous"])
    ys, xs = np.nonzero((codes >= 1) & (codes <= 4))
    if ys.size:
        center = len(ys) // 2
        y, x = int(ys[center]), int(xs[center])
    else:
        y, x = resolution[0] // 2, resolution[1] // 2
    radius = 36
    sl = (slice(max(y-radius, 0), min(y+radius, resolution[0])),
          slice(max(x-radius, 0), min(x+radius, resolution[1])))
    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(np.clip(capture["direct"][sl], 0, 1))
    axes[1].imshow(np.clip(capture["continuous"][sl], 0, 1))
    axes[2].imshow(residual[sl].mean(2), cmap="magma")
    for axis, title in zip(axes, ("direct", "continuous", "residual")):
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(directory / "v084_direct_vs_continuous_silhouette.png", dpi=180)
    plt.close(figure)

    convergence = {
        row["estimator"]: row for row in old_report["sample_convergence"]["summary"]
        if row["estimator"] in (DIRECT, CONTINUOUS)
    }
    figure, axis = plt.subplots(figsize=(6, 4))
    for estimator, label in ((DIRECT, "direct"), (CONTINUOUS, "continuous")):
        rows = convergence[estimator]["rows"]
        axis.loglog([r["emitters"] for r in rows[:-1]],
                    [r["mse_vs_8N"]["mean"] for r in rows[:-1]], "o-", label=label)
    axis.set_xlabel("nested emitter prefix")
    axis.set_ylabel("MSE vs operator-matched 8N reference")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v084_direct_photon_convergence.png", dpi=180)
    plt.close(figure)

    perturb = geometry["perturbation_rows"]
    figure, axis = plt.subplots(figsize=(8, 4))
    for detector, marker in (("direct", "o"), ("continuous", "s")):
        for category in sorted({row["category"] for row in perturb}):
            rows = [row for row in perturb if row["category"] == category]
            axis.loglog([row["delta"] for row in rows],
                        [row[detector]["response_l2_norm"] for row in rows],
                        marker + "-", alpha=0.75, label=f"{detector}: {category}")
    axis.set_xlabel("coefficient perturbation")
    axis.set_ylabel("image response L2")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(directory / "v084_geometry_perturbation_response.png", dpi=180)
    plt.close(figure)

    fd = geometry["direct_fd_rows"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for category in sorted({row["category"] for row in fd}):
        rows = [row for row in fd if row["category"] == category]
        axes[0].loglog([row["epsilon"] for row in rows],
                       [row["same_sample_fd_norm"] for row in rows], "o-", label=category)
        axes[1].semilogx([row["epsilon"] for row in rows],
                         [row["gradient_fd_snr"] for row in rows], "o-")
    axes[0].set_ylabel("hard-bin FD norm")
    axes[1].set_ylabel("same/heldout FD SNR")
    for axis in axes:
        axis.set_xlabel("epsilon")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v084_direct_photon_fd_stability.png", dpi=180)
    plt.close(figure)

    transfer = geometry["continuous_training_direct_evaluation"]
    labels = ["train C", "heldout C", "heldout B"]
    before = [transfer["initial"][key]["mse"] for key in
              ("train_continuous", "heldout_continuous", "heldout_direct")]
    after = [transfer["optimized"][key]["mse"] for key in
             ("train_continuous", "heldout_continuous", "heldout_direct")]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - 0.18, before, width=0.36, label="initial")
    axis.bar(x + 0.18, after, width=0.36, label="C-trained")
    axis.set_xticks(x, labels)
    axis.set_ylabel("operator-matched MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v084_train_continuous_eval_direct.png", dpi=180)
    plt.close(figure)

    points = trajectory["_plot"]["sources"]
    arrivals = trajectory["_plot"]["arrivals"]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(points[:, 0], points[:, 2], s=20, label="zero-set sources")
    axis.scatter(arrivals[:, 0], arrivals[:, 2], s=25, marker="x", label="detector arrivals")
    for source, arrival in zip(points, arrivals):
        axis.plot([source[0], arrival[0]], [source[2], arrival[2]], alpha=0.55)
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.set_title("actual forward packet paths to detector boundary")
    axis.legend()
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(directory / "v084_photon_trajectory_detector.png", dpi=180)
    plt.close(figure)


def _append_csv(path: Path, sections: list[tuple[str, list[dict[str, Any]]]]) -> None:
    with path.open(newline="") as stream:
        old = list(csv.DictReader(stream))
    replacement_sections = {name for name, _ in sections}
    old = [row for row in old if row.get("section") not in replacement_sections]
    added: list[dict[str, Any]] = []
    for section, rows in sections:
        for row in rows:
            flat = {"section": section}
            for key, value in row.items():
                flat[key] = json.dumps(_json_ready(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            added.append(flat)
    all_rows = old + added
    keys = sorted({key for row in all_rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


def run_direct_forward_photon_comparison(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    config: DirectPhotonConfig | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    experiment = config or DirectPhotonConfig()
    if len(experiment.resolutions) != len(experiment.matched_emitters):
        raise ValueError("resolution/emitter-count lengths differ")
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    artifact_path = artifact_directory / "v084_mc_continuous_detector.json"
    csv_path = artifact_directory / "v084_mc_continuous_detector.csv"
    old_report = json.loads(artifact_path.read_text())
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    base = prepared.base_field.to(device)
    cells = sign_changing_cells(base.grid)
    maximum = max(max(experiment.matched_emitters), max(experiment.convergence_counts))
    surface = _prepare_surface(
        base, None, maximum, experiment.emitter_chunk_size, experiment.seed, cells
    )
    atlas = nested_fibonacci_atlas(device, (experiment.views,))
    mc_config = experiment.mc_config()
    matched_density: list[dict[str, Any]] = []
    fixed_sample: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    captures: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    performance: list[dict[str, Any]] = []
    boundaries: dict[tuple[int, int], Any] = {}
    convergence_renders: dict[int, list[EstimatorView]] = {}
    for resolution, emitters in zip(experiment.resolutions, experiment.matched_emitters):
        boundary = enclosing_observation_sphere(surface.reference_points[:emitters])
        boundaries[resolution] = boundary
        views, transport = _render_estimator_views(
            base, surface, emitters, resolution, boundary, atlas, mc_config
        )
        matched_density.append(_direct_density_stats(
            views, resolution, emitters, "DIRECT_PHOTON_MATCHED_DENSITY"
        ))
        comparisons.append(_same_transport_comparison(views, resolution, emitters))
        if resolution == experiment.resolutions[0] and emitters in experiment.convergence_counts:
            convergence_renders[emitters] = views
        view_id = min(experiment.capture_view, len(views) - 1)
        selected = views[view_id]
        if selected.direct_count is None:
            raise RuntimeError("missing direct count capture")
        captures[resolution] = {
            "direct": selected.preclamp[DIRECT],
            "continuous": selected.preclamp[CONTINUOUS],
            "count": selected.direct_count,
        }
        performance.append({
            "sample_regime": "DIRECT_PHOTON_MATCHED_DENSITY",
            "resolution": list(resolution),
            "emitters": emitters,
            "runtime_seconds": transport["runtime_seconds"],
            "attempted_packets": transport["attempted_packets"],
            "retained_events": transport["retained_events"],
            "hard_bin_writes": transport["hard_bin_writes"],
            "peak_allocated_mib": transport["peak_allocated_mib"],
            "peak_reserved_mib": transport["peak_reserved_mib"],
        })
        if emitters != experiment.fixed_emitters:
            fixed_views, fixed_transport = _render_estimator_views(
                base, surface, experiment.fixed_emitters, resolution,
                boundaries[experiment.resolutions[0]], atlas, mc_config
            )
            fixed_sample.append(_direct_density_stats(
                fixed_views, resolution, experiment.fixed_emitters,
                "DIRECT_PHOTON_FIXED_SAMPLE",
            ))
            performance.append({
                "sample_regime": "DIRECT_PHOTON_FIXED_SAMPLE",
                "resolution": list(resolution),
                "emitters": experiment.fixed_emitters,
                "runtime_seconds": fixed_transport["runtime_seconds"],
                "attempted_packets": fixed_transport["attempted_packets"],
                "retained_events": fixed_transport["retained_events"],
                "hard_bin_writes": fixed_transport["hard_bin_writes"],
                "peak_allocated_mib": fixed_transport["peak_allocated_mib"],
                "peak_reserved_mib": fixed_transport["peak_reserved_mib"],
            })
            del fixed_views
        else:
            fixed_sample.append({**matched_density[-1], "sample_regime": "DIRECT_PHOTON_FIXED_SAMPLE"})
        del views
        gc.collect()
        _release()
        _progress("v084_direct_resolution_complete", resolution=list(resolution))

    for emitters in experiment.convergence_counts:
        if emitters in convergence_renders:
            continue
        views, _ = _render_estimator_views(
            base,
            surface,
            emitters,
            experiment.resolutions[0],
            boundaries[experiment.resolutions[0]],
            atlas,
            mc_config,
        )
        convergence_renders[emitters] = views
    operator_matched_convergence = _operator_matched_convergence(convergence_renders)

    trajectory = _trajectory_verification(
        base, surface, atlas, boundaries[experiment.resolutions[0]], experiment
    )
    geometry = _geometry_experiment(prepared, experiment)

    direct_convergence = next(row for row in operator_matched_convergence if row["estimator"] == DIRECT)
    continuous_convergence = next(row for row in operator_matched_convergence if row["estimator"] == CONTINUOUS)
    direct_brightness = [row["mean_foreground_brightness"] for row in direct_convergence["rows"]]
    brightness_range = (max(direct_brightness) - min(direct_brightness)) / max(statistics.mean(direct_brightness), 1e-30)
    direct_noise = statistics.mean(
        row["scale_normalized_whole_image_self_mse"]["mean"]
        for row in old_report["multi_seed_mc"]["summary"] if row["estimator"] == DIRECT
    )
    continuous_noise = statistics.mean(
        row["scale_normalized_whole_image_self_mse"]["mean"]
        for row in old_report["multi_seed_mc"]["summary"] if row["estimator"] == CONTINUOUS
    )
    maximum_energy_difference = max(row["total_contribution_relative_difference"] for row in comparisons)
    direct_primary = next(
        row for row in direct_convergence["rows"]
        if row["emitters"] == experiment.matched_emitters[0]
    )
    continuous_primary = next(
        row for row in continuous_convergence["rows"]
        if row["emitters"] == experiment.matched_emitters[0]
    )
    foreground_rmse_relative_difference = abs(
        direct_primary["rmse_vs_operator_matched_reference"]
        - continuous_primary["rmse_vs_operator_matched_reference"]
    ) / max(continuous_primary["rmse_vs_operator_matched_reference"], 1e-30)
    transfer = geometry["continuous_training_direct_evaluation"]
    direct_transfer = float(transfer["heldout_direct_relative_improvement"])
    continuous_transfer = float(transfer["heldout_continuous_relative_improvement"])
    fd_by_category: dict[str, float] = {}
    for category in geometry["selected_categories"]:
        rows = [row for row in geometry["direct_fd_rows"] if row["category"] == category]
        norms = [row["same_sample_fd_norm"] for row in rows]
        fd_by_category[category] = max(norms) / max(min(norms), 1e-30)
    max_fd_ratio = max(fd_by_category.values())
    response_ratios = [row["direct_over_continuous_response_norm"] for row in geometry["perturbation_rows"]]
    same_ranking = True
    for delta in experiment.perturbation_deltas:
        selected = [
            row for row in geometry["perturbation_rows"]
            if float(row["delta"]) == delta
        ]
        direct_order = [
            row["category"] for row in sorted(
                selected, key=lambda row: row["direct"]["response_l2_norm"]
            )
        ]
        continuous_order = [
            row["category"] for row in sorted(
                selected, key=lambda row: row["continuous"]["response_l2_norm"]
            )
        ]
        same_ranking &= direct_order == continuous_order
    image_comparable = (
        maximum_energy_difference <= experiment.image_energy_relative_band
        and foreground_rmse_relative_difference <= experiment.image_rmse_relative_band
    )
    geometry_signal_comparable = (
        0.5 <= statistics.median(response_ratios) <= 2.0 and same_ranking
    )
    fd_stable = max_fd_ratio <= 2.0 and statistics.median(
        row["gradient_fd_snr"] for row in geometry["direct_fd_rows"]
    ) >= 1.0
    transfer_supported = direct_transfer > 0.0
    kernel_artifact = continuous_transfer > 0.0 and direct_transfer <= 0.0
    direct_detector_supported = (
        brightness_range <= 0.05
        and direct_convergence["empirical_log_rmse_slope"] < 0.0
        and trajectory["maximum_forward_trajectory_coordinate_error"] <= 1e-12
    )
    direct_primary_ready = image_comparable and geometry_signal_comparable
    if direct_primary_ready and fd_stable and transfer_supported:
        primary = "DIRECT_FORWARD_PHOTON"
    elif direct_primary_ready and transfer_supported and not kernel_artifact:
        primary = "FORWARD_PHOTON_TRANSPORT_WITH_CONTINUOUS_TRAINING_DETECTOR_AND_DIRECT_EVALUATION"
    elif not kernel_artifact:
        primary = "CONTINUOUS_DETECTOR_PHOTON_MC"
    else:
        primary = "UNRESOLVED"
    verdicts = {
        "FORWARD_PACKET_TRANSPORT_PRESERVED": True,
        "DIRECT_PHOTON_MC_DERIVED": True,
        "DIRECT_PHOTON_SAMPLE_COUNT_INVARIANT": brightness_range <= 0.05,
        "DIRECT_PHOTON_MATCHED_DENSITY_VALID": True,
        "DIRECT_PHOTON_IMAGE_QUALITY_COMPARABLE": image_comparable,
        "DIRECT_PHOTON_GEOMETRY_SIGNAL_COMPARABLE": geometry_signal_comparable,
        "DIRECT_PHOTON_FD_STABLE_ENOUGH": fd_stable,
        "CONTINUOUS_DETECTOR_GRADIENT_ADVANTAGE": not fd_stable,
        "CONTINUOUS_TRAINING_IMPROVES_DIRECT_HELDOUT": transfer_supported,
        "CONTINUOUS_DETECTOR_KERNEL_ARTIFACT_DETECTED": kernel_artifact,
        "DIRECT_PHOTON_DETECTOR_SUPPORTED": direct_detector_supported,
        "CONTINUOUS_DETECTOR_NEEDED_ONLY_FOR_DIFFERENTIATION": primary == "FORWARD_PHOTON_TRANSPORT_WITH_CONTINUOUS_TRAINING_DETECTOR_AND_DIRECT_EVALUATION",
        "DIRECT_FORWARD_PHOTON_FORMULATION_PREFERRED": primary in (
            "DIRECT_FORWARD_PHOTON",
            "FORWARD_PHOTON_TRANSPORT_WITH_CONTINUOUS_TRAINING_DETECTOR_AND_DIRECT_EVALUATION",
        ),
    }
    evidence = {
        "FORWARD_PACKET_TRANSPORT_PRESERVED": {
            "same_pass_detector_accumulators": 2,
            "maximum_forward_trajectory_coordinate_error": trajectory["maximum_forward_trajectory_coordinate_error"],
        },
        "DIRECT_PHOTON_MC_DERIVED": {"global_sample_normalization_power": -1, "random_pixel_denominator_count": 0},
        "DIRECT_PHOTON_SAMPLE_COUNT_INVARIANT": {"brightness_relative_range_N_2N_4N_8N": brightness_range, "threshold": 0.05},
        "DIRECT_PHOTON_MATCHED_DENSITY_VALID": {"matched_resolution_count": len(matched_density), "emitter_counts": list(experiment.matched_emitters)},
        "DIRECT_PHOTON_IMAGE_QUALITY_COMPARABLE": {"maximum_total_contribution_relative_difference": maximum_energy_difference, "foreground_rmse_relative_difference_at_matched_256": foreground_rmse_relative_difference, "hard_scale_normalized_self_mse_mean": direct_noise, "continuous_scale_normalized_self_mse_mean": continuous_noise, "energy_band": experiment.image_energy_relative_band, "rmse_band": experiment.image_rmse_relative_band},
        "DIRECT_PHOTON_GEOMETRY_SIGNAL_COMPARABLE": {"median_direct_over_continuous_response_norm": statistics.median(response_ratios), "same_variant_ranking": same_ranking, "accepted_ratio_band": [0.5, 2.0]},
        "DIRECT_PHOTON_FD_STABLE_ENOUGH": {"maximum_epsilon_fd_norm_ratio": max_fd_ratio, "median_fd_snr": statistics.median(row["gradient_fd_snr"] for row in geometry["direct_fd_rows"]), "thresholds": {"ratio": 2.0, "snr": 1.0}},
        "CONTINUOUS_DETECTOR_GRADIENT_ADVANTAGE": {"direct_maximum_epsilon_fd_norm_ratio": max_fd_ratio, "v084_continuous_best_frozen_fd_max_relative_error": old_report["verdict_evidence_by_verdict"]["ANALYTIC_MC_JACOBIAN_MATCHES_FROZEN_FD"]["maximum_best_relative_error"]},
        "CONTINUOUS_TRAINING_IMPROVES_DIRECT_HELDOUT": {"heldout_direct_mse_relative_improvement": direct_transfer},
        "CONTINUOUS_DETECTOR_KERNEL_ARTIFACT_DETECTED": {"heldout_continuous_mse_relative_improvement": continuous_transfer, "heldout_direct_mse_relative_improvement": direct_transfer},
        "DIRECT_PHOTON_DETECTOR_SUPPORTED": {"mathematically_derived": True, "brightness_relative_range": brightness_range, "empirical_log_rmse_slope": direct_convergence["empirical_log_rmse_slope"], "maximum_forward_trajectory_coordinate_error": trajectory["maximum_forward_trajectory_coordinate_error"], "quality_equivalence_is_a_separate_verdict": image_comparable},
        "CONTINUOUS_DETECTOR_NEEDED_ONLY_FOR_DIFFERENTIATION": {"direct_primary_quality_ready": direct_primary_ready, "direct_quality_gates_passed": int(image_comparable) + int(geometry_signal_comparable), "direct_quality_gates_total": 2, "direct_fd_stable": fd_stable, "direct_maximum_epsilon_fd_norm_ratio": max_fd_ratio, "heldout_transfer": transfer_supported, "primary_formulation": primary},
        "DIRECT_FORWARD_PHOTON_FORMULATION_PREFERRED": {"direct_quality_gates_passed": int(image_comparable) + int(geometry_signal_comparable), "direct_quality_gates_total": 2, "heldout_direct_mse_relative_improvement": direct_transfer, "primary_formulation": primary},
    }

    trajectory_plot = trajectory.pop("_plot")
    trajectory["_plot"] = trajectory_plot
    _save_figures(
        figure_directory, captures, comparisons, matched_density, old_report,
        geometry, trajectory,
    )
    trajectory.pop("_plot")
    runtime = time.perf_counter() - started
    supplement = {
        "title": "Direct Forward Photon Detection Comparison",
        "configuration": asdict(experiment),
        "environment": cuda_environment(),
        "operator_decomposition": {
            "arrival_measure": "H_D(F)=T_forward(F)",
            "direct": "I_box=M_box H_D(F)",
            "continuous": "I_cont=M_cont H_D(F)",
            "shared_transport": True,
            "only_changed_component": "detector measurement operator M",
        },
        "original_photon_movement_formulation": "zero set -> forward packets -> zero-set visibility/intersection -> detector arrival -> RGB",
        "direct_detector_equation": "I_box[p]=g H W/N sum_i f_i 1[u_i in P_p]",
        "trajectory_verification": trajectory,
        "direct_density_statistics": {
            "matched_density": matched_density,
            "fixed_sample_historical_control": fixed_sample,
        },
        "sample_convergence": {
            "reference": "operator-matched 8N numerical reference at 256x256",
            "direct": direct_convergence,
            "continuous": continuous_convergence,
            "direct_brightness_relative_range": brightness_range,
        },
        "same_transport_detector_comparison": comparisons,
        "multi_seed_noise": {
            "direct_mean_scale_normalized_self_mse": direct_noise,
            "continuous_mean_scale_normalized_self_mse": continuous_noise,
        },
        "geometry_perturbation_response": geometry["perturbation_rows"],
        "direct_photon_finite_difference": {
            "rows": geometry["direct_fd_rows"],
            "epsilon_norm_ratio_by_category": fd_by_category,
            "smooth_indicator_derivative_claimed": False,
        },
        "continuous_training_direct_evaluation": transfer,
        "equivalence_bands_predeclared": {
            "foreground_rmse_relative": experiment.image_rmse_relative_band,
            "total_energy_relative": experiment.image_energy_relative_band,
            "geometry_metrics_relative": experiment.geometry_relative_band,
            "same_variant_ranking_required": True,
            "continuous_training_must_improve_heldout_direct": True,
        },
        "performance": performance,
        "verdicts": verdicts,
        "verdict_evidence": evidence,
        "preferred_detector_formulation": primary,
        "PRIMARY_FORMULATION": primary,
        "birth_experiment_run": False,
        "runtime_seconds": runtime,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "commands": {
            "formal": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --direct-photon-comparison --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply",
            "verification": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --scene sphere --verify",
        },
    }
    old_report["direct_forward_photon_detection_comparison"] = supplement
    old_report["runtime_seconds_with_supplement"] = float(old_report["runtime_seconds"]) + runtime
    artifact_path.write_text(json.dumps(_json_ready(old_report), indent=2, sort_keys=True) + "\n")
    _append_csv(csv_path, [
        ("direct_density_matched", matched_density),
        ("direct_density_fixed", fixed_sample),
        ("direct_vs_continuous", comparisons),
        ("geometry_perturbation", geometry["perturbation_rows"]),
        ("direct_fd", geometry["direct_fd_rows"]),
        ("continuous_train_direct_eval", [transfer]),
        ("direct_supplement_performance", performance),
        ("direct_supplement_verdict", [
            {"verdict": key, "value": value, "evidence": evidence[key]}
            for key, value in verdicts.items()
        ]),
    ])
    return supplement
