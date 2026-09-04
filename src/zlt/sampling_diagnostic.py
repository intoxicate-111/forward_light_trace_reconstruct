"""v0.8.1 diagnostics for Full-HD transport coverage and RGB loss scaling."""

from __future__ import annotations

import csv
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import cuda_environment
from .corrected_birth import (
    CorrectedBirthConfig,
    CorrectedContext,
    CorrectedState,
    _active_components,
    _batch_gram,
    _build_context,
    _correlation,
    _evaluate,
    _finite_difference_gate,
    _joint_prediction,
    _json_ready,
    _metrics,
    _multiscale_support,
    _optimize,
    _render,
    _score_candidates,
)
from .mesh_field import prepare_stanford_bunny


Tensor = torch.Tensor


@dataclass(frozen=True)
class SamplingDiagnosticConfig:
    views: int = 20
    fixed_emitters: int = 65_536
    dictionary_count: int = 512
    initial_count: int = 32
    resolution_sweep: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
        (1080, 1920),
    )
    matched_density: tuple[tuple[int, int, int], ...] = (
        (256, 256, 4096),
        (512, 512, 16_384),
        (1024, 1024, 65_536),
    )
    footprint_emitters: int = 16_384
    footprint_resolutions: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
    )
    mc_low_resolution: tuple[int, int] = (256, 256)
    mc_high_resolution: tuple[int, int] = (1080, 1920)
    mc_low_emitters: int = 16_384
    mc_high_emitters: int = 65_536
    mc_seeds: tuple[int, int] = (101, 211)
    birth_resolution: tuple[int, int] = (512, 512)
    birth_emitters: int = 16_384
    birth_dictionary_count: int = 256
    birth_batch_sizes: tuple[int, ...] = (1, 4, 16, 32)
    random_seed: int = 1707


def _config(
    experiment: SamplingDiagnosticConfig,
    resolution: tuple[int, int],
    emitters: int,
    *,
    dictionary_count: int | None = None,
    scramble_seed: int | None = None,
    resolution_aware_footprint: bool = False,
) -> CorrectedBirthConfig:
    return CorrectedBirthConfig(
        dictionary_count=dictionary_count or experiment.dictionary_count,
        initial_count=experiment.initial_count,
        surface_samples=emitters,
        views=experiment.views,
        resolution=resolution[0],
        resolution_width=resolution[1],
        surface_scramble_seed=scramble_seed,
        footprint_reference_resolution=(
            256 if resolution_aware_footprint else None
        ),
        initial_optimization_steps=3,
        post_birth_steps=2,
        fixed_optimization_steps=3,
        line_evaluations=14,
    )


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _progress(phase: str, label: str) -> None:
    print(json.dumps({"phase": phase, "label": label}), flush=True)


def _contribution_map(render: object) -> tuple[Tensor, dict[str, float]]:
    positive = render.valid & (render.weights > 0.0)
    flat_pixels = render.pixels[positive]
    counts = torch.zeros(
        render.mass.numel(),
        dtype=torch.int32,
        device=render.mass.device,
    )
    counts.scatter_add_(
        0,
        flat_pixels,
        torch.ones_like(flat_pixels, dtype=torch.int32),
    )
    per_event = positive.sum(1)
    return counts, {
        "projected_in_frame_events": int((per_event > 0).sum()),
        "total_footprint_pixel_writes": int(positive.sum()),
        "average_effective_footprint_area_pixels": float(
            per_event.double().mean()
        )
        if per_event.numel()
        else 0.0,
    }


def _histogram_add(total: Tensor | None, values: Tensor) -> Tensor:
    current = torch.bincount(values.to(torch.long).cpu())
    if total is None:
        return current
    if current.numel() > total.numel():
        total = torch.nn.functional.pad(
            total, (0, current.numel() - total.numel())
        )
    elif total.numel() > current.numel():
        current = torch.nn.functional.pad(
            current, (0, total.numel() - current.numel())
        )
    return total + current


def _histogram_quantile(histogram: Tensor, quantile: float) -> float:
    threshold = quantile * max(int(histogram.sum()) - 1, 0)
    return float(
        torch.searchsorted(
            torch.cumsum(histogram, 0),
            torch.tensor(threshold, dtype=histogram.dtype),
        )
    )


def _pair_metrics(
    first: list[np.ndarray], second: list[np.ndarray]
) -> dict[str, float]:
    raw_sse = 0.0
    absolute = 0.0
    scalar_count = 0
    magnitudes: list[np.ndarray] = []
    for left, right in zip(first, second):
        difference = left.astype(np.float64) - right.astype(np.float64)
        raw_sse += float(np.square(difference).sum())
        absolute += float(np.abs(difference).sum())
        scalar_count += difference.size
        magnitudes.append(np.linalg.norm(difference.reshape(-1, 3), axis=1))
    spatial = np.concatenate(magnitudes)
    mse = raw_sse / scalar_count
    return {
        "raw_sse": raw_sse,
        "half_sse_loss": 0.5 * raw_sse,
        "mse_per_rgb_scalar": mse,
        "rmse_per_rgb_scalar": math.sqrt(mse),
        "mae_per_rgb_scalar": absolute / scalar_count,
        "spatial_residual_mean": float(spatial.mean()),
        "spatial_residual_p95": float(np.quantile(spatial, 0.95)),
        "spatial_residual_p99": float(np.quantile(spatial, 0.99)),
        "spatial_residual_max": float(spatial.max()),
    }


def _cpu_images(state: CorrectedState) -> list[np.ndarray]:
    return [
        image.detach().reshape(-1, 3).cpu().to(torch.float32).numpy()
        for image in state.images
    ]


def _canonical_detector_images(
    context: CorrectedContext, state: CorrectedState
) -> tuple[list[np.ndarray], list[float]]:
    """Render a sample realization with an identical scene-fixed detector.

    The legacy cell builder centers its orthographic detector on the sampled
    point cloud.  Independent Sobol scrambles therefore induce a sub-pixel
    detector translation as well as a sampling change.  The Monte-Carlo
    control must vary only the sample realization, so it uses the fixed domain
    midpoint and recomputes the support mask for that detector.
    """
    center = 0.5 * (context.lower + context.upper)
    cells = [
        replace(cell, center=center, support_mask=None)
        for cell in context.cells
    ]
    images, renders = _render(state.points, state.normals, context, cells)
    cpu = [
        image.detach().reshape(-1, 3).cpu().to(torch.float32).numpy()
        for image in images
    ]
    center_list = center.detach().cpu().tolist()
    del images, renders, cells
    return cpu, center_list


def _audit_state(
    context: CorrectedContext,
    state: CorrectedState,
    label: str,
    capture_view: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    rows, columns = context.config.resolution_shape
    pixels_per_view = rows * columns
    scalar_count = context.config.views * pixels_per_view * 3
    raw_sse = 0.0
    absolute = 0.0
    target_energy = 0.0
    predicted_energy = 0.0
    histogram: Tensor | None = None
    footprint_writes = 0
    projected = 0
    footprint_area_sum = 0.0
    retained = 0
    per_view: list[dict[str, object]] = []
    decomposition = {
        threshold: {
            "covered_raw_sse": 0.0,
            "uncovered_raw_sse": 0.0,
            "uncovered_target_energy": 0.0,
        }
        for threshold in (1, 2, 4)
    }
    capture: dict[str, np.ndarray] | None = None
    visibility = context.visibility_report["base"]
    for view_id, (image_flat, target_flat, render, cell) in enumerate(
        zip(
            state.images,
            context.target_images,
            state.renders,
            context.cells,
        )
    ):
        image = image_flat.reshape(-1, 3)
        target = target_flat.reshape(-1, 3)
        residual = image - target
        residual_energy = residual.square().sum(1)
        target_pixel_energy = target.square().sum(1)
        counts, footprint = _contribution_map(render)
        histogram = _histogram_add(histogram, counts)
        writes = int(footprint["total_footprint_pixel_writes"])
        footprint_writes += writes
        projected += int(footprint["projected_in_frame_events"])
        view_retained = int(visibility["per_view_retained"][view_id])
        retained += view_retained
        footprint_area_sum += (
            float(footprint["average_effective_footprint_area_pixels"])
            * view_retained
        )
        view_sse = float(residual_energy.sum())
        view_target_energy = float(target_pixel_energy.sum())
        view_predicted_energy = float(image.square().sum())
        raw_sse += view_sse
        absolute += float(residual.abs().sum())
        target_energy += view_target_energy
        predicted_energy += view_predicted_energy
        row: dict[str, object] = {
            "label": label,
            "view": view_id,
            "attempted_packets": context.config.surface_samples,
            "outward_events": int(visibility["per_view_outward"][view_id]),
            "absorbed_events": int(visibility["per_view_absorbed"][view_id]),
            "retained_visible_events": view_retained,
            **footprint,
            "total_pixels": pixels_per_view,
            "unique_contribution_pixels": int((counts >= 1).sum()),
            "coverage_ratio": float((counts >= 1).double().mean()),
            "renderer_support_ratio": float(
                cell.support_mask.double().mean()
            ),
            "pixels_ge_2": int((counts >= 2).sum()),
            "pixels_ge_4": int((counts >= 4).sum()),
            "pixels_ge_8": int((counts >= 8).sum()),
            "zero_contribution_fraction": float((counts == 0).double().mean()),
            "mean_contributions_per_pixel": float(counts.double().mean()),
            "median_contributions_per_pixel": float(
                counts.double().median()
            ),
            "contribution_p90": float(torch.quantile(counts.double(), 0.90)),
            "contribution_p95": float(torch.quantile(counts.double(), 0.95)),
            "contribution_p99": float(torch.quantile(counts.double(), 0.99)),
            "maximum_contributions": int(counts.max()),
            "effective_coverage_density": writes / pixels_per_view,
            "raw_sse": view_sse,
            "mse_per_rgb_scalar": view_sse / (pixels_per_view * 3),
            "target_rgb_energy": view_target_energy,
            "predicted_rgb_energy": view_predicted_energy,
        }
        for threshold in (1, 2, 4):
            covered = counts >= threshold
            covered_sse = float(residual_energy[covered].sum())
            uncovered_sse = float(residual_energy[~covered].sum())
            uncovered_target = float(target_pixel_energy[~covered].sum())
            decomposition[threshold]["covered_raw_sse"] += covered_sse
            decomposition[threshold]["uncovered_raw_sse"] += uncovered_sse
            decomposition[threshold]["uncovered_target_energy"] += (
                uncovered_target
            )
            row[f"raw_sse_coverage_ge_{threshold}"] = covered_sse
            row[f"raw_sse_coverage_lt_{threshold}"] = uncovered_sse
        per_view.append(row)
        if view_id == capture_view:
            residual_magnitude = torch.sqrt(residual_energy)
            covered = counts >= 1
            capture = {
                "target": target.reshape(rows, columns, 3)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "render": image.reshape(rows, columns, 3)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "coverage": covered.reshape(rows, columns)
                .detach()
                .cpu()
                .numpy(),
                "counts": counts.reshape(rows, columns)
                .detach()
                .cpu()
                .numpy(),
                "weights": render.mass.reshape(rows, columns)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "residual": residual_magnitude.reshape(rows, columns)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "covered_residual": torch.where(
                    covered, residual_magnitude, torch.zeros_like(residual_magnitude)
                )
                .reshape(rows, columns)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "uncovered_residual": torch.where(
                    ~covered,
                    residual_magnitude,
                    torch.zeros_like(residual_magnitude),
                )
                .reshape(rows, columns)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
            }
    assert histogram is not None
    residual_energy = raw_sse
    mse = raw_sse / scalar_count
    aggregate_decomposition: dict[str, object] = {}
    for threshold, values in decomposition.items():
        covered_sse = float(values["covered_raw_sse"])
        uncovered_sse = float(values["uncovered_raw_sse"])
        uncovered_target = float(values["uncovered_target_energy"])
        aggregate_decomposition[f"coverage_ge_{threshold}"] = {
            "covered_raw_sse": covered_sse,
            "uncovered_raw_sse": uncovered_sse,
            "fraction_loss_uncovered": uncovered_sse
            / max(residual_energy, 1e-30),
            "fraction_target_energy_uncovered": uncovered_target
            / max(target_energy, 1e-30),
        }
    total_pixels = context.config.views * pixels_per_view
    aggregate = {
        "label": label,
        "resolution": [rows, columns],
        "views": context.config.views,
        "emitters": context.config.surface_samples,
        "attempted_packets": context.config.surface_samples
        * context.config.views,
        "outward_events": int(visibility["emitted"]),
        "absorbed_events": int(visibility["absorbed"]),
        "retained_visible_events": retained,
        "projected_in_frame_events": projected,
        "total_pixels": total_pixels,
        "pixels_ge_1": int(histogram[1:].sum()),
        "pixels_ge_2": int(histogram[2:].sum()),
        "pixels_ge_4": int(histogram[4:].sum()),
        "pixels_ge_8": int(histogram[8:].sum()),
        "unique_coverage_ratio": float(histogram[1:].sum() / total_pixels),
        "zero_contribution_fraction": float(histogram[0] / total_pixels),
        "mean_contributions_per_pixel": footprint_writes / total_pixels,
        "median_contributions_per_pixel": _histogram_quantile(histogram, 0.5),
        "contribution_p90": _histogram_quantile(histogram, 0.90),
        "contribution_p95": _histogram_quantile(histogram, 0.95),
        "contribution_p99": _histogram_quantile(histogram, 0.99),
        "maximum_contributions": int(histogram.numel() - 1),
        "average_effective_footprint_area_pixels": footprint_area_sum
        / max(retained, 1),
        "total_footprint_pixel_writes": footprint_writes,
        "effective_coverage_density": footprint_writes / total_pixels,
        "raw_sse": raw_sse,
        "half_sse_loss": 0.5 * raw_sse,
        "sse_per_view": raw_sse / context.config.views,
        "sse_per_pixel": raw_sse / total_pixels,
        "sse_per_rgb_scalar": mse,
        "mse_per_rgb_scalar": mse,
        "rmse_per_rgb_scalar": math.sqrt(mse),
        "mae_per_rgb_scalar": absolute / scalar_count,
        "target_rgb_energy": target_energy,
        "predicted_rgb_energy": predicted_energy,
        "residual_rgb_energy": residual_energy,
        "residual_to_target_energy_ratio": residual_energy
        / max(target_energy, 1e-30),
        "loss_reduction": "0.5 * SUM over views, pixels, and RGB channels",
        "footprint_scale": list(context.config.footprint_scale),
        "footprint_reference_resolution": (
            context.config.footprint_reference_resolution
        ),
        "residual_decomposition": aggregate_decomposition,
        "per_view": per_view,
    }
    return aggregate, capture


def _compact_coverage(
    context: CorrectedContext, state: CorrectedState
) -> dict[str, float | int]:
    audit, _ = _audit_state(context, state, "birth_coverage")
    return {
        "unique_coverage_ratio": float(audit["unique_coverage_ratio"]),
        "zero_contribution_fraction": float(
            audit["zero_contribution_fraction"]
        ),
        "effective_coverage_density": float(
            audit["effective_coverage_density"]
        ),
        "pixels_ge_1": int(audit["pixels_ge_1"]),
        "pixels_ge_2": int(audit["pixels_ge_2"]),
        "pixels_ge_4": int(audit["pixels_ge_4"]),
        "pixels_ge_8": int(audit["pixels_ge_8"]),
    }


def _score_statistics(values: Tensor) -> dict[str, float]:
    values = values.detach().double()
    mean = float(values.mean())
    median = float(values.median())
    std = float(values.std(unbiased=False))
    maximum = float(values.max())
    top_count = min(10, values.numel())
    return {
        "minimum": float(values.min()),
        "mean": mean,
        "median": median,
        "standard_deviation": std,
        "maximum": maximum,
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "top1_to_median": maximum / max(median, 1e-30),
        "top10_mean_to_global_mean": float(torch.topk(values, top_count).values.mean())
        / max(mean, 1e-30),
        "coefficient_of_variation": std / max(abs(mean), 1e-30),
        "fraction_positive": float((values > 0.0).double().mean()),
        "fraction_within_50pct_best": float(
            (values >= 0.50 * maximum).double().mean()
        ),
        "fraction_within_75pct_best": float(
            (values >= 0.75 * maximum).double().mean()
        ),
        "fraction_within_90pct_best": float(
            (values >= 0.90 * maximum).double().mean()
        ),
    }


def _spatial_entropy(context: CorrectedContext, ids: Tensor) -> float:
    centers = context.master_layout.centers[ids]
    lower = context.master_layout.centers.amin(0)
    upper = context.master_layout.centers.amax(0)
    normalized = ((centers - lower) / (upper - lower).clamp_min(1e-30)).clamp(
        0.0, 1.0 - 1e-12
    )
    bins = torch.floor(normalized * 6).to(torch.long)
    encoded = (bins[:, 0] * 6 + bins[:, 1]) * 6 + bins[:, 2]
    probability = torch.bincount(encoded, minlength=216).double()
    probability /= probability.sum().clamp_min(1.0)
    nonzero = probability > 0.0
    entropy = -(probability[nonzero] * torch.log(probability[nonzero])).sum()
    return float(entropy / math.log(216.0))


def _candidate_audit(
    context: CorrectedContext, state: CorrectedState, label: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scores, diagnostics, seconds = _score_candidates(context, state)
    count = context.config.dictionary_count
    inactive = torch.ones(count, dtype=torch.bool, device=state.points.device)
    inactive[state.active_ids] = False
    local_coverage = torch.zeros(
        count, dtype=torch.float64, device=state.points.device
    )
    local_residual = torch.zeros_like(local_coverage)
    local_hole = torch.zeros_like(local_coverage)
    observations = torch.zeros_like(local_coverage)
    centers = context.master_layout.centers
    for cell, render, image_flat, target_flat in zip(
        context.cells, state.renders, state.images, context.target_images
    ):
        counts, _ = _contribution_map(render)
        rows, columns = cell.resolution
        relative = centers - cell.center
        column = torch.round(
            (relative @ cell.right / cell.extent + 0.5) * columns - 0.5
        ).to(torch.long)
        row = torch.round(
            (0.5 - relative @ cell.up / cell.extent) * rows - 0.5
        ).to(torch.long)
        valid = (
            (row >= 0)
            & (row < rows)
            & (column >= 0)
            & (column < columns)
        )
        pixel = row.clamp(0, rows - 1) * columns + column.clamp(
            0, columns - 1
        )
        target = target_flat.reshape(-1, 3)
        residual = image_flat.reshape(-1, 3) - target
        pixel_residual = residual.square().sum(1)
        target_signal = target.square().sum(1) > 0.0
        local_coverage += torch.where(
            valid, counts[pixel].double(), torch.zeros_like(local_coverage)
        )
        local_residual += torch.where(
            valid, pixel_residual[pixel], torch.zeros_like(local_residual)
        )
        local_hole += torch.where(
            valid & target_signal[pixel] & (counts[pixel] == 0),
            torch.ones_like(local_hole),
            torch.zeros_like(local_hole),
        )
        observations += valid.double()
    local_coverage /= observations.clamp_min(1.0)
    local_residual /= observations.clamp_min(1.0)
    local_hole /= observations.clamp_min(1.0)
    geometry_error = context.target_displacement.abs()
    ids = torch.nonzero(inactive, as_tuple=False).flatten()
    report: dict[str, object] = {
        "label": label,
        "resolution": list(context.config.resolution_shape),
        "emitters": context.config.surface_samples,
        "candidate_count": int(ids.numel()),
        "scoring_seconds": seconds,
        **diagnostics,
        "scores": {},
    }
    for policy in ("raw", "quadratic"):
        values = scores[policy][ids]
        top_count = max(1, math.ceil(0.10 * ids.numel()))
        top_ids = ids[torch.topk(values, top_count).indices]
        report["scores"][policy] = {
            **_score_statistics(values),
            "top10pct_spatial_entropy": _spatial_entropy(context, top_ids),
            "spearman_local_coverage": _correlation(
                values, local_coverage[ids], True
            ),
            "spearman_local_zero_coverage_fraction": _correlation(
                values, local_hole[ids], True
            ),
            "spearman_local_rgb_residual_energy": _correlation(
                values, local_residual[ids], True
            ),
            "spearman_geometry_error": _correlation(
                values, geometry_error[ids], True
            ),
            "pearson_local_coverage": _correlation(
                values, local_coverage[ids], False
            ),
            "pearson_geometry_error": _correlation(
                values, geometry_error[ids], False
            ),
        }
    scatter = {
        "raw": scores["raw"][ids].detach().cpu().numpy(),
        "quadratic": scores["quadratic"][ids].detach().cpu().numpy(),
        "coverage": local_coverage[ids].detach().cpu().numpy(),
        "holes": local_hole[ids].detach().cpu().numpy(),
        "residual": local_residual[ids].detach().cpu().numpy(),
        "geometry": geometry_error[ids].detach().cpu().numpy(),
    }
    return report, scatter


def _save_image_buffers(path: Path, capture: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="black")
    panels = (
        ("target RGB", capture["target"], None),
        ("predicted RGB", capture["render"], None),
        ("coverage mask", capture["coverage"], "gray"),
        ("contribution count", np.log1p(capture["counts"]), "magma"),
        ("accumulated weight", capture["weights"], "viridis"),
        ("residual magnitude", capture["residual"], "inferno"),
    )
    for axis, (title, image, cmap) in zip(axes.reshape(-1), panels):
        axis.imshow(np.clip(image, 0.0, 1.0) if cmap is None else image, cmap=cmap)
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="black")
    plt.close(figure)


def _save_residual_buffers(path: Path, capture: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    maximum = float(np.quantile(capture["residual"], 0.995))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="black")
    for axis, title, key in zip(
        axes,
        ("all residual", "covered residual", "uncovered residual"),
        ("residual", "covered_residual", "uncovered_residual"),
    ):
        axis.imshow(capture[key], cmap="inferno", vmin=0.0, vmax=maximum)
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="black")
    plt.close(figure)


def _save_resolution_plots(
    figure_directory: Path,
    rows: list[dict[str, object]],
    mc_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    x = [int(row["resolution"][0] * row["resolution"][1]) for row in rows]
    specifications = (
        (
            "v081_resolution_vs_unique_coverage.png",
            [float(row["unique_coverage_ratio"]) for row in rows],
            "unique coverage ratio",
        ),
        (
            "v081_resolution_vs_footprint_density.png",
            [float(row["effective_coverage_density"]) for row in rows],
            "effective footprint writes / pixel",
        ),
        (
            "v081_resolution_vs_normalized_rmse.png",
            [float(row["rmse_per_rgb_scalar"]) for row in rows],
            "normalized RGB RMSE",
        ),
    )
    for name, y, ylabel in specifications:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(x, y, "o-")
        axis.set_xscale("log")
        axis.set_xlabel("pixels per view")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(figure_directory / name, dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = [str(row["label"]) for row in mc_rows]
    axis.bar(labels, [float(row["mse_per_rgb_scalar"]) for row in mc_rows])
    axis.set_ylabel("independent-sample MC self MSE")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_directory / "v081_resolution_vs_mc_self_loss.png", dpi=180
    )
    plt.close(figure)


def _save_candidate_plots(
    figure_directory: Path,
    scatters: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    for target, filename, xlabel in (
        ("coverage", "v081_candidate_score_vs_coverage.png", "local count"),
        (
            "geometry",
            "v081_candidate_score_vs_geometry_error.png",
            "independent geometry error",
        ),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, policy in zip(axes, ("raw", "quadratic")):
            for label, values in scatters.items():
                axis.scatter(
                    values[target],
                    values[policy],
                    s=8,
                    alpha=0.5,
                    label=label,
                )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(f"{policy} score")
            axis.set_yscale("symlog", linthresh=1e-12)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(figure_directory / filename, dpi=180)
        plt.close(figure)


def _save_matched_density_plot(
    path: Path, rows: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(item["label"]) for item in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(
        x, [float(item["unique_coverage_ratio"]) for item in rows], "o-"
    )
    axes[1].plot(
        x, [float(item["rmse_per_rgb_scalar"]) for item in rows], "o-"
    )
    axes[2].plot(
        x, [float(item["mc_self_mse"]) for item in rows], "o-"
    )
    for axis, ylabel in zip(
        axes, ("unique coverage", "geometry RGB RMSE", "MC self MSE")
    ):
        axis.set_xticks(x, labels, rotation=15)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_gain_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for policy in ("raw", "quadratic"):
        selected = [row for row in rows if row["policy"] == policy]
        predicted = [float(row["predicted_joint_gain"]) for row in selected]
        same = [float(row["same_sample_realized_gain"]) for row in selected]
        held = [float(row["heldout_realized_gain"]) for row in selected]
        axes[0].scatter(predicted, same, label=policy)
        axes[1].scatter(predicted, held, label=policy)
    axes[0].set_title("same-sample")
    axes[1].set_title("independent held-out")
    for axis in axes:
        axis.set_xlabel("predicted joint gain")
        axis.set_ylabel("realized gain")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _copy_layout(
    heldout: CorrectedContext, training: CorrectedContext
) -> None:
    heldout.master_layout = training.master_layout
    heldout.levels = training.levels
    heldout.master_support = _multiscale_support(
        heldout.reference_points,
        training.master_layout,
        training.levels,
        heldout.config.support_margin,
    )


def _birth_retest(
    prepared: object,
    experiment: SamplingDiagnosticConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = _config(
        experiment,
        experiment.birth_resolution,
        experiment.birth_emitters,
        dictionary_count=experiment.birth_dictionary_count,
        scramble_seed=experiment.mc_seeds[0],
        resolution_aware_footprint=True,
    )
    heldout_config = replace(config, surface_scramble_seed=experiment.mc_seeds[1])
    training = _build_context(prepared, config)
    heldout = _build_context(prepared, heldout_config)
    _copy_layout(heldout, training)
    finite_difference = _finite_difference_gate(training)
    device = training.reference_points.device
    active = torch.arange(config.initial_count, device=device)
    initial = _evaluate(
        training,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device=device),
    )
    initial, initial_seconds = _optimize(
        training, initial, config.initial_optimization_steps
    )
    heldout_initial = _evaluate(
        heldout, active, initial.coefficients
    )
    scores, score_diagnostics, score_seconds = _score_candidates(
        training, initial
    )
    inactive = torch.ones(
        config.dictionary_count, dtype=torch.bool, device=device
    )
    inactive[active] = False
    generator = torch.Generator().manual_seed(experiment.random_seed)
    rows: list[dict[str, object]] = []
    final_states: dict[str, CorrectedState] = {}
    final_timings: dict[str, tuple[float, float]] = {}
    for policy in ("raw", "quadratic"):
        values = torch.where(
            inactive,
            scores[policy],
            torch.full_like(scores[policy], -math.inf),
        )
        order = torch.argsort(values, descending=True, stable=True)
        for batch_size in experiment.birth_batch_sizes:
            selected = order[:batch_size]
            gram, gram_seconds = _batch_gram(training, initial, selected)
            predicted_joint = _joint_prediction(
                scores, selected, gram, config.damping
            )
            born = _evaluate(
                training,
                torch.cat((active, selected)),
                torch.cat(
                    (
                        initial.coefficients,
                        initial.coefficients.new_zeros(batch_size),
                    )
                ),
            )
            fitted, optimize_seconds = _optimize(
                training, born, config.post_birth_steps
            )
            heldout_fitted = _evaluate(
                heldout, fitted.active_ids, fitted.coefficients
            )
            row = {
                "policy": policy,
                "batch_size": batch_size,
                "selected_ids": selected.detach().cpu().tolist(),
                "predicted_joint_gain": predicted_joint,
                "predicted_raw_alignment": float(
                    scores["raw"][selected].sum()
                ),
                "same_sample_realized_gain": initial.loss - fitted.loss,
                "heldout_realized_gain": heldout_initial.loss
                - heldout_fitted.loss,
                "same_sample_before_loss": initial.loss,
                "same_sample_after_loss": fitted.loss,
                "heldout_before_loss": heldout_initial.loss,
                "heldout_after_loss": heldout_fitted.loss,
                "score_seconds": score_seconds,
                "gram_seconds": gram_seconds,
                "optimization_seconds": optimize_seconds,
                **_metrics(
                    training,
                    fitted,
                    f"{policy}_birth",
                    1,
                    score_seconds + gram_seconds,
                    optimize_seconds,
                    optimize_seconds,
                ),
            }
            rows.append(row)
            if batch_size == max(experiment.birth_batch_sizes):
                final_states[policy] = fitted
                final_timings[policy] = (gram_seconds, optimize_seconds)
            else:
                del fitted
            del born, heldout_fitted, gram
            torch.cuda.empty_cache()
    random_ids = torch.nonzero(inactive, as_tuple=False).flatten()
    permutation = torch.randperm(random_ids.numel(), generator=generator)
    random_selected = random_ids[
        permutation[: max(experiment.birth_batch_sizes)].to(device)
    ]
    fixed_selected = torch.arange(
        config.initial_count,
        config.initial_count + max(experiment.birth_batch_sizes),
        device=device,
    )
    controls: list[dict[str, object]] = [
        {
            **_metrics(
                training,
                initial,
                "no_birth_k32",
                0,
                0.0,
                initial_seconds,
                initial_seconds,
            ),
            "heldout_image_loss": heldout_initial.loss,
            **_compact_coverage(training, initial),
        }
    ]
    for method, selected in (
        ("random_matched_k64", random_selected),
        ("fixed_space_k64", fixed_selected),
    ):
        state = _evaluate(
            training,
            torch.cat((active, selected)),
            torch.cat(
                (
                    initial.coefficients,
                    initial.coefficients.new_zeros(selected.numel()),
                )
            ),
        )
        state, seconds = _optimize(training, state, config.post_birth_steps)
        heldout_state = _evaluate(
            heldout, state.active_ids, state.coefficients
        )
        controls.append(
            {
                **_metrics(
                    training,
                    state,
                    method,
                    1,
                    0.0,
                    seconds,
                    seconds,
                ),
                "heldout_image_loss": heldout_state.loss,
                "same_sample_realized_gain": initial.loss - state.loss,
                "heldout_realized_gain": heldout_initial.loss
                - heldout_state.loss,
                "selected_ids": selected.detach().cpu().tolist(),
                **_compact_coverage(training, state),
            }
        )
        del state, heldout_state
        torch.cuda.empty_cache()
    for policy, state in final_states.items():
        gram_seconds, optimize_seconds = final_timings[policy]
        heldout_state = _evaluate(
            heldout, state.active_ids, state.coefficients
        )
        controls.append(
            {
                **_metrics(
                    training,
                    state,
                    f"{policy}_birth_k64",
                    1,
                    score_seconds + gram_seconds,
                    optimize_seconds,
                    score_seconds + gram_seconds + optimize_seconds,
                ),
                "heldout_image_loss": heldout_state.loss,
                "same_sample_realized_gain": initial.loss - state.loss,
                "heldout_realized_gain": heldout_initial.loss
                - heldout_state.loss,
                **_compact_coverage(training, state),
            }
        )
        del heldout_state
    result = {
        "configuration": asdict(config),
        "training_surface_seed": experiment.mc_seeds[0],
        "heldout_surface_seed": experiment.mc_seeds[1],
        "finite_difference": finite_difference,
        "score_diagnostics": score_diagnostics,
        "small_batch_gain_rows": rows,
        "method_comparison": controls,
    }
    del training, heldout, initial, heldout_initial, scores, final_states
    _release()
    return result, rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def run_sampling_diagnostic(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    config: SamplingDiagnosticConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    experiment = config or SamplingDiagnosticConfig()
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prepared = prepare_stanford_bunny(
        mesh_path, build_surface_scaffold=False
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    resolution_rows: list[dict[str, object]] = []
    candidate_reports: list[dict[str, object]] = []
    candidate_scatters: dict[str, dict[str, np.ndarray]] = {}
    common_sample_test: dict[str, object] = {}
    full_capture: dict[str, np.ndarray] | None = None
    for resolution in experiment.resolution_sweep:
        label = f"{resolution[1]}x{resolution[0]}_fixed_samples"
        context = _build_context(
            prepared,
            _config(
                experiment,
                resolution,
                experiment.fixed_emitters,
            ),
        )
        active = torch.arange(
            experiment.initial_count, device=context.reference_points.device
        )
        state = _evaluate(
            context,
            active,
            torch.zeros(
                active.numel(), dtype=torch.float64, device=active.device
            ),
        )
        row, capture = _audit_state(
            context,
            state,
            label,
            capture_view=(
                min(10, experiment.views - 1)
                if resolution == experiment.resolution_sweep[-1]
                else None
            ),
        )
        row.update(context.geometry_evaluator(state.points))
        resolution_rows.append(row)
        if resolution in (
            experiment.resolution_sweep[0],
            experiment.resolution_sweep[-1],
        ):
            report, scatter = _candidate_audit(context, state, label)
            candidate_reports.append(report)
            candidate_scatters[label] = scatter
        if resolution == experiment.resolution_sweep[-1]:
            repeated = _evaluate(context, active, state.coefficients)
            common_sample_test = _pair_metrics(
                _cpu_images(state), _cpu_images(repeated)
            )
            common_sample_test["surface_seed"] = None
            full_capture = capture
            del repeated
        _progress("resolution_sweep", label)
        del context, state
        _release()
    assert full_capture is not None
    _save_image_buffers(
        figure_directory / "v081_target_render_coverage_contributions.png",
        full_capture,
    )
    _save_residual_buffers(
        figure_directory / "v081_covered_uncovered_residual.png",
        full_capture,
    )

    mc_rows: list[dict[str, object]] = []
    for resolution, emitters, label in (
        (
            experiment.mc_low_resolution,
            experiment.mc_low_emitters,
            "low_resolution_independent_samples",
        ),
        (
            experiment.mc_high_resolution,
            experiment.mc_high_emitters,
            "high_resolution_independent_samples",
        ),
    ):
        images: list[list[np.ndarray]] = []
        geometry_losses: list[float] = []
        centers: list[list[float]] = []
        for seed in experiment.mc_seeds:
            context = _build_context(
                prepared,
                _config(
                    experiment,
                    resolution,
                    emitters,
                    scramble_seed=seed,
                ),
            )
            active = torch.arange(
                experiment.initial_count,
                device=context.reference_points.device,
            )
            state = _evaluate(
                context,
                active,
                torch.zeros(
                    active.numel(), dtype=torch.float64, device=active.device
                ),
            )
            canonical_images, canonical_center = _canonical_detector_images(
                context, state
            )
            images.append(canonical_images)
            geometry_losses.append(2.0 * state.loss)
            centers.append(canonical_center)
            del context, state
            _release()
        row = {
            "label": label,
            "resolution": list(resolution),
            "emitters": emitters,
            "surface_seeds": list(experiment.mc_seeds),
            "detector_center_max_difference": float(
                np.max(np.abs(np.asarray(centers[0]) - np.asarray(centers[1])))
            ),
            "geometry_reconstruction_raw_sse_mean": statistics.mean(
                geometry_losses
            ),
            **_pair_metrics(images[0], images[1]),
        }
        row["mc_self_to_geometry_sse_ratio"] = float(row["raw_sse"]) / max(
            float(row["geometry_reconstruction_raw_sse_mean"]), 1e-30
        )
        mc_rows.append(row)
        _progress("mc_self", label)
        del images
        gc.collect()

    matched_rows: list[dict[str, object]] = []
    for rows, columns, emitters in experiment.matched_density:
        resolution = (rows, columns)
        label = f"{columns}x{rows}_{emitters}_matched_density"
        pair_images: list[list[np.ndarray]] = []
        audit: dict[str, object] | None = None
        for seed_index, seed in enumerate(experiment.mc_seeds):
            context = _build_context(
                prepared,
                _config(
                    experiment,
                    resolution,
                    emitters,
                    scramble_seed=seed,
                ),
            )
            active = torch.arange(
                experiment.initial_count,
                device=context.reference_points.device,
            )
            state = _evaluate(
                context,
                active,
                torch.zeros(
                    active.numel(), dtype=torch.float64, device=active.device
                ),
            )
            canonical_images, _ = _canonical_detector_images(context, state)
            pair_images.append(canonical_images)
            if seed_index == 0:
                audit, _ = _audit_state(context, state, label)
                audit.update(context.geometry_evaluator(state.points))
            del context, state
            _release()
        assert audit is not None
        mc = _pair_metrics(pair_images[0], pair_images[1])
        audit["mc_self_mse"] = mc["mse_per_rgb_scalar"]
        audit["mc_self_rmse"] = mc["rmse_per_rgb_scalar"]
        audit["surface_seeds"] = list(experiment.mc_seeds)
        matched_rows.append(audit)
        _progress("matched_density", label)
        del pair_images
        gc.collect()

    footprint_rows: list[dict[str, object]] = []
    for resolution in experiment.footprint_resolutions:
        for aware in (False, True):
            label = (
                f"{resolution[1]}x{resolution[0]}_"
                f"{'resolution_aware' if aware else 'legacy_pixel'}"
            )
            context = _build_context(
                prepared,
                _config(
                    experiment,
                    resolution,
                    experiment.footprint_emitters,
                    resolution_aware_footprint=aware,
                ),
            )
            active = torch.arange(
                experiment.initial_count,
                device=context.reference_points.device,
            )
            state = _evaluate(
                context,
                active,
                torch.zeros(
                    active.numel(), dtype=torch.float64, device=active.device
                ),
            )
            row, _ = _audit_state(context, state, label)
            row.update(context.geometry_evaluator(state.points))
            footprint_rows.append(row)
            _progress("footprint", label)
            del context, state
            _release()

    birth_report, gain_rows = _birth_retest(prepared, experiment)
    _progress("birth_retest", "complete")

    _save_resolution_plots(figure_directory, resolution_rows, mc_rows)
    _save_candidate_plots(figure_directory, candidate_scatters)
    _save_matched_density_plot(
        figure_directory / "v081_matched_density_resolution.png",
        matched_rows,
    )
    _save_gain_plot(
        figure_directory / "v081_predicted_realized_same_heldout.png",
        gain_rows,
    )

    full = resolution_rows[-1]
    low = resolution_rows[0]
    full_mc = next(
        row
        for row in mc_rows
        if "high_resolution" in str(row["label"])
    )
    candidate_full = candidate_reports[-1]
    quadratic_full = candidate_full["scores"]["quadratic"]
    matched_coverage = [
        float(row["unique_coverage_ratio"]) for row in matched_rows
    ]
    matched_rmse = [float(row["rmse_per_rgb_scalar"]) for row in matched_rows]
    footprint_pairs: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    for row in footprint_rows:
        key = tuple(row["resolution"])
        mode = "aware" if row["footprint_reference_resolution"] else "legacy"
        footprint_pairs.setdefault(key, {})[mode] = row
    post_methods = {
        str(row["method"]): row for row in birth_report["method_comparison"]
    }
    raw_post = post_methods["raw_birth_k64"]
    random_post = post_methods["random_matched_k64"]
    high_res_undersampled = float(full["unique_coverage_ratio"]) < 0.50
    loss_scale_dependent = (
        float(full["half_sse_loss"]) > 10.0 * float(low["half_sse_loss"])
        and float(full["mse_per_rgb_scalar"])
        < float(full["half_sse_loss"])
    )
    uncovered_dominates = (
        float(
            full["residual_decomposition"]["coverage_ge_1"][
                "fraction_loss_uncovered"
            ]
        )
        > 0.50
    )
    mc_dominates = float(full_mc["mc_self_to_geometry_sse_ratio"]) >= 0.75
    footprint_bug = all(
        float(pair["aware"]["average_effective_footprint_area_pixels"])
        > float(pair["legacy"]["average_effective_footprint_area_pixels"])
        for resolution, pair in footprint_pairs.items()
        if resolution != (256, 256)
    )
    score_holes = abs(
        float(quadratic_full["spearman_local_zero_coverage_fraction"])
    ) >= 0.50
    score_geometry = float(quadratic_full["spearman_geometry_error"]) >= 0.30
    matched_density_supported = (
        max(matched_coverage) - min(matched_coverage) < 0.02
        and matched_rmse[-1] > 1.50 * matched_rmse[0]
    )
    birth_recovers = (
        float(raw_post["image_loss"])
        < float(random_post["image_loss"])
        and float(raw_post["symmetric_chamfer"])
        < float(random_post["symmetric_chamfer"])
        and float(raw_post["heldout_realized_gain"]) > 0.0
    )
    common_repeat_sse = float(common_sample_test["raw_sse"])
    independent_mc_would_dominate = mc_dominates
    current_mc_dominates = common_repeat_sse > 1e-12
    birth_sampling_explained = footprint_bug and birth_recovers
    optimizer_still_needed = not birth_sampling_explained
    verdicts = {
        "HIGH_RES_IMAGE_UNDERSAMPLED": high_res_undersampled,
        "RGB_LOSS_SCALE_RESOLUTION_DEPENDENT": loss_scale_dependent,
        "RGB_RESIDUAL_DOMINATED_BY_UNCOVERED_PIXELS": uncovered_dominates,
        "MC_NOISE_FLOOR_DOMINATES_RGB_OBJECTIVE": current_mc_dominates,
        "INDEPENDENT_MC_WOULD_DOMINATE_RGB_OBJECTIVE": (
            independent_mc_would_dominate
        ),
        "FOOTPRINT_RESOLUTION_SCALING_BUG_FOUND": footprint_bug,
        "COMMON_RANDOM_NUMBERS_NEEDED": independent_mc_would_dominate,
        "COMMON_RANDOM_NUMBERS_ALREADY_USED": common_repeat_sse <= 1e-12,
        "CANDIDATE_SCORE_CORRELATES_WITH_COVERAGE_HOLES": score_holes,
        "CANDIDATE_SCORE_CORRELATES_WITH_GEOMETRY_ERROR": score_geometry,
        "BIRTH_FAILURE_EXPLAINED_BY_IMAGE_SAMPLING": (
            birth_sampling_explained
        ),
        "OPTIMIZER_FAILURE_STILL_NEEDED_TO_EXPLAIN_RESULTS": (
            optimizer_still_needed
        ),
        "MATCHED_DENSITY_RESOLUTION_EFFECT_SUPPORTED": (
            matched_density_supported
        ),
        "BIRTH_SIGNAL_RECOVERS_AFTER_IMAGE_FIX": birth_recovers,
    }
    full_decomposition = full["residual_decomposition"]
    aware_coverage = [
        float(pair["aware"]["unique_coverage_ratio"])
        for pair in footprint_pairs.values()
    ]
    aware_rmse = [
        float(pair["aware"]["rmse_per_rgb_scalar"])
        for pair in footprint_pairs.values()
    ]
    verdict_evidence = {
        "HIGH_RES_IMAGE_UNDERSAMPLED": {
            "fullhd_unique_coverage_ratio": full["unique_coverage_ratio"],
            "fullhd_effective_coverage_density": full[
                "effective_coverage_density"
            ],
            "threshold_unique_coverage_ratio": 0.50,
        },
        "RGB_LOSS_SCALE_RESOLUTION_DEPENDENT": {
            "half_sse_ratio_fullhd_to_256": float(full["half_sse_loss"])
            / float(low["half_sse_loss"]),
            "pixel_ratio_fullhd_to_256": float(full["total_pixels"])
            / float(low["total_pixels"]),
            "mse_ratio_fullhd_to_256": float(full["mse_per_rgb_scalar"])
            / float(low["mse_per_rgb_scalar"]),
            "reduction": "0.5 * SUM",
        },
        "RGB_RESIDUAL_DOMINATED_BY_UNCOVERED_PIXELS": {
            "fraction_loss_count_lt_1": full_decomposition[
                "coverage_ge_1"
            ]["fraction_loss_uncovered"],
            "fraction_loss_count_lt_2": full_decomposition[
                "coverage_ge_2"
            ]["fraction_loss_uncovered"],
            "fraction_loss_count_lt_4": full_decomposition[
                "coverage_ge_4"
            ]["fraction_loss_uncovered"],
            "dominance_threshold": 0.50,
            "verdict_coverage_definition": "count >= 1",
        },
        "MC_NOISE_FLOOR_DOMINATES_RGB_OBJECTIVE": {
            "same_sample_repeat_raw_sse": common_repeat_sse,
            "note": (
                "the production objective uses common samples, so the "
                "independent-sample control is counterfactual"
            ),
        },
        "INDEPENDENT_MC_WOULD_DOMINATE_RGB_OBJECTIVE": {
            "fullhd_independent_mc_to_geometry_sse_ratio": full_mc[
                "mc_self_to_geometry_sse_ratio"
            ],
            "dominance_threshold": 0.75,
        },
        "FOOTPRINT_RESOLUTION_SCALING_BUG_FOUND": {
            "resolution_aware_coverage_range": max(aware_coverage)
            - min(aware_coverage),
            "resolution_aware_rmse_range": max(aware_rmse)
            - min(aware_rmse),
            "legacy_960x540_coverage": footprint_pairs[(540, 960)][
                "legacy"
            ]["unique_coverage_ratio"],
            "aware_960x540_coverage": footprint_pairs[(540, 960)][
                "aware"
            ]["unique_coverage_ratio"],
        },
        "COMMON_RANDOM_NUMBERS_NEEDED": {
            "independent_mc_to_geometry_sse_ratio": full_mc[
                "mc_self_to_geometry_sse_ratio"
            ],
            "same_sample_repeat_raw_sse": common_repeat_sse,
        },
        "CANDIDATE_SCORE_CORRELATES_WITH_COVERAGE_HOLES": {
            "fullhd_quadratic_spearman_zero_coverage": quadratic_full[
                "spearman_local_zero_coverage_fraction"
            ],
            "absolute_correlation_threshold": 0.50,
        },
        "CANDIDATE_SCORE_CORRELATES_WITH_GEOMETRY_ERROR": {
            "fullhd_quadratic_spearman_geometry_error": quadratic_full[
                "spearman_geometry_error"
            ],
            "positive_correlation_threshold": 0.30,
        },
        "BIRTH_FAILURE_EXPLAINED_BY_IMAGE_SAMPLING": {
            "footprint_bug": footprint_bug,
            "post_fix_birth_recovers": birth_recovers,
            "scope": (
                "material explanation of the v0.8 high-resolution result, "
                "not proof that every large-batch optimization effect vanishes"
            ),
        },
        "OPTIMIZER_FAILURE_STILL_NEEDED_TO_EXPLAIN_RESULTS": {
            "sampling_cause_not_excluded": birth_sampling_explained,
            "post_fix_raw_beats_random": birth_recovers,
        },
        "MATCHED_DENSITY_RESOLUTION_EFFECT_SUPPORTED": {
            "coverage_range": max(matched_coverage) - min(matched_coverage),
            "rmse_first": matched_rmse[0],
            "rmse_last": matched_rmse[-1],
            "rmse_last_to_first": matched_rmse[-1] / matched_rmse[0],
        },
        "BIRTH_SIGNAL_RECOVERS_AFTER_IMAGE_FIX": {
            "raw_birth_image_loss": raw_post["image_loss"],
            "random_control_image_loss": random_post["image_loss"],
            "raw_birth_chamfer": raw_post["symmetric_chamfer"],
            "random_control_chamfer": random_post["symmetric_chamfer"],
            "raw_birth_heldout_gain": raw_post["heldout_realized_gain"],
        },
    }
    report: dict[str, Any] = {
        "version": "0.8.1",
        "phase": "high_resolution_sampling_diagnostic",
        "environment": cuda_environment(),
        "configuration": asdict(experiment),
        "seeds": {
            "legacy_surface": "unscrambled Sobol",
            "independent_surface": list(experiment.mc_seeds),
            "random_policy": experiment.random_seed,
        },
        "commands": {
            "formal": (
                "PYTHONPATH=src conda run --no-capture-output -n test "
                "python demo.py --sampling-diagnostic --bunny-mesh "
                "data/stanford_bunny/cache/bun_zipper.ply"
            ),
            "verification": (
                "PYTHONPATH=src conda run --no-capture-output -n test "
                "python demo.py --scene sphere --verify"
            ),
        },
        "image_formation_audit": {
            "packet_emission": (
                "one deterministic atlas direction per emitter/view; only "
                "positive normal dot products become outward events"
            ),
            "visibility": (
                "earliest positive sign-changing zero-set intersection before "
                "the enclosing-sphere exit absorbs the event"
            ),
            "projection": (
                "orthographic transverse coordinates are converted to pixel "
                "row/column coordinates by multiplying by H/W"
            ),
            "legacy_footprint": (
                "separable cubic B-spline B(row-r)B(column-c), support radius "
                "2 pixels per axis, at most 16 pixel writes per event"
            ),
            "rgb_accumulation": (
                "I_p = clamp(sensor_gain * sum_i(w_ip C_i lobe_i) / "
                "sum_i(w_ip), 0, 1) on supported pixels"
            ),
            "intensity_semantics": (
                "normalized weighted average, not an energy sum or pixel-area "
                "radiometric integral; no packet-count or view-count factor"
            ),
            "legacy_support": "sum_i(w_ip) >= 0.05",
            "target_vs_prediction": (
                "same deterministic emitter identities and atlas, but each "
                "geometry has its own outward/visibility events and frozen "
                "support mask"
            ),
            "loss": "L = 0.5 * sum_{view,pixel,channel}(I-T)^2",
            "jacobian_residual": "g = sum_view J_view^T (I_view-T_view)",
            "sample_sharing": (
                "v0.8 uses common deterministic surface identities; target "
                "and prediction are not independently Monte-Carlo sampled"
            ),
            "implemented_equations": {
                "legacy_weight": (
                    "w_ip = B(row_p-row_i) B(col_p-col_i), each cubic B "
                    "has support |delta| < 2 pixels"
                ),
                "resolution_aware_weight": (
                    "w_ip = B((row_p-row_i)/s_h) "
                    "B((col_p-col_i)/s_w)/(s_h s_w), with "
                    "s_h=H/256 and s_w=W/256"
                ),
                "pixel_rgb": (
                    "I_p = clamp(gain * sum_i w_ip radiance_i / "
                    "sum_i w_ip, 0, 1)"
                ),
                "loss": "L = 0.5 * sum_v sum_p sum_c (I_vpc-T_vpc)^2",
                "gradient": "dL/da = sum_v J_v^T (I_v-T_v)",
            },
        },
        "resolution_sweep": resolution_rows,
        "common_sample_repeat": common_sample_test,
        "independent_sample_mc_self_loss": mc_rows,
        "matched_sample_density": matched_rows,
        "footprint_scaling_control": footprint_rows,
        "candidate_score_diagnostic": candidate_reports,
        "predicted_realized_and_minimal_birth": birth_report,
        "verdicts": verdicts,
        "verdict_evidence": verdict_evidence,
        "interpretation": {
            "observation": (
                "Full-HD coverage, residual decomposition, independent-sample "
                "noise, score distributions, and held-out gains are measured "
                "separately."
            ),
            "inference": (
                "The verdict booleans are computed from declared numerical "
                "thresholds and must be interpreted with their evidence rows."
            ),
            "fix": (
                "Optional resolution-aware detector-space cubic footprint: "
                "kernel coordinate scales with H/256 and W/256, weights carry "
                "the reciprocal pixel-scale Jacobian, and the support threshold "
                "scales by pixel area. Legacy behavior remains the default."
            ),
            "post_fix_result": (
                "See the minimal 512-square common-sample/independent-held-out "
                "birth comparison; no full v0.8 rerun is performed."
            ),
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "json": str(
                artifact_directory / "v081_highres_sampling_diagnostic.json"
            ),
            "csv": str(
                artifact_directory / "v081_highres_sampling_diagnostic.csv"
            ),
            "figure_glob": str(figure_directory / "v081_*.png"),
        },
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v081_highres_sampling_diagnostic.json"
    json_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    csv_rows: list[dict[str, object]] = []
    for section, rows in (
        ("resolution_sweep", resolution_rows),
        ("mc_self", mc_rows),
        ("matched_density", matched_rows),
        ("footprint", footprint_rows),
        ("gain", gain_rows),
        ("birth", birth_report["method_comparison"]),
    ):
        csv_rows.extend({"section": section, **row} for row in rows)
    _write_csv(
        artifact_directory / "v081_highres_sampling_diagnostic.csv",
        csv_rows,
    )
    return report
