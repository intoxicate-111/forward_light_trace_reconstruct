"""v0.8.3 pixel accumulation, support-switching, and gradient diagnostics.

This module deliberately diagnoses the legacy fixed four-by-four pixel cubic
estimator before changing it.  Full-resolution transport is streamed; only
per-pixel sufficient statistics are retained on the CPU.
"""

from __future__ import annotations

import gc
import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage, stats

from .benchmark import cuda_environment
from .boundary_transport import (
    DirectionAtlas,
    ObservationSphere,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
)
from .high_sample import (
    HighSampleConfig,
    SurfaceState,
    _cubic,
    _finalize_image,
    _json_ready,
    _prepare_surface,
    _progress,
    _release,
    _sync,
    _visible_owner_ids,
    _write_csv,
)
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color, sign_changing_cells
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor


@dataclass(frozen=True)
class PixelDiagnosticConfig:
    views: int = 20
    resolutions: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
        (1080, 1920),
    )
    emitter_counts: tuple[int, ...] = (
        65_536,
        262_144,
        518_400,
        2_073_600,
    )
    emitter_chunk_size: int = 65_536
    sobol_seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    pseudorandom_seed: int = 9001
    root_samples: int = 16
    bisection_steps: int = 18
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    support_threshold: float = 0.05
    ambient: float = 0.35
    capture_view: int = 10
    gradient_surface_samples: int = 16_384
    gradient_views: int = 4
    gradient_resolution: int = 256
    gradient_parameters: int = 32
    gradient_epsilons: tuple[float, ...] = (
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
        3e-5,
    )

    def transport(self) -> HighSampleConfig:
        return HighSampleConfig(
            views=self.views,
            resolutions=self.resolutions,
            initial_emitter_counts=self.emitter_counts,
            emitter_chunk_size=self.emitter_chunk_size,
            root_samples=self.root_samples,
            bisection_steps=self.bisection_steps,
            detector_extent=self.detector_extent,
            sensor_gain=self.sensor_gain,
            support_threshold=self.support_threshold,
            ambient=self.ambient,
        )


@dataclass
class PixelView:
    image: np.ndarray
    counts: np.ndarray
    mass: np.ndarray
    squared_mass: np.ndarray
    numerator: np.ndarray | None
    hard_bin_numerator: np.ndarray | None = None

    @property
    def support(self) -> np.ndarray:
        return self.mass >= 0.05


@dataclass
class ReferenceView:
    image: np.ndarray
    support: np.ndarray


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"pearson": 0.0, "spearman": 0.0, "samples": int(x.size)}
    return {
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "samples": int(x.size),
    }


def _spatial_gradient(image: np.ndarray) -> np.ndarray:
    rows, columns, _ = image.shape
    dy = np.gradient(image.astype(np.float64), axis=0)
    dx = np.gradient(image.astype(np.float64), axis=1)
    return np.sqrt(np.square(dx).sum(2) + np.square(dy).sum(2)).reshape(
        rows * columns
    )


def _diagnostic_splat(
    points: Tensor,
    normals: Tensor,
    owners: Tensor,
    direction: Tensor,
    right: Tensor,
    up: Tensor,
    center: Tensor,
    lower: Tensor,
    upper: Tensor,
    resolution: tuple[int, int],
    extent: float,
    mass: Tensor,
    squared_mass: Tensor,
    numerator: Tensor,
    counts: Tensor,
    ambient: float,
    hard_bin_numerator: Tensor | None = None,
) -> dict[str, object]:
    rows, columns = resolution
    device = points.device
    relative = points[owners] - center
    column = (relative @ right / extent + 0.5) * columns - 0.5
    row = (0.5 - relative @ up / extent) * rows - 0.5
    base_row = torch.floor(row).to(torch.long)
    base_column = torch.floor(column).to(torch.long)
    offsets = torch.arange(-1, 3, dtype=torch.long, device=device)
    row_ids = base_row[:, None] + offsets[None]
    column_ids = base_column[:, None] + offsets[None]
    row_distance = row[:, None] - row_ids
    column_distance = column[:, None] - column_ids
    row_weights = _cubic(row_distance)
    column_weights = _cubic(column_distance)
    pixel_rows = row_ids[:, :, None].expand(-1, 4, 4).reshape(-1, 16)
    pixel_columns = column_ids[:, None, :].expand(-1, 4, 4).reshape(-1, 16)
    weights = (
        row_weights[:, :, None] * column_weights[:, None, :]
    ).reshape(-1, 16)
    in_frame = (
        (pixel_rows >= 0)
        & (pixel_rows < rows)
        & (pixel_columns >= 0)
        & (pixel_columns < columns)
    )
    positive = weights > 0.0
    valid = in_frame & positive
    pixels = (
        pixel_rows.clamp(0, rows - 1) * columns
        + pixel_columns.clamp(0, columns - 1)
    )
    effective_weights = weights * valid
    colors = meshfree_base_color(points[owners], lower, upper)
    cosine = (normals[owners] @ direction).clamp_min(0.0)
    lobe = ambient + (1.0 - ambient) * cosine
    radiance = colors * lobe[:, None]
    flat_pixels = pixels.reshape(-1)
    flat_weights = effective_weights.reshape(-1)
    mass.scatter_add_(0, flat_pixels, flat_weights)
    squared_mass.scatter_add_(0, flat_pixels, flat_weights.square())
    numerator.index_add_(
        0,
        flat_pixels,
        (effective_weights[..., None] * radiance[:, None, :]).reshape(-1, 3),
    )
    counts.scatter_add_(
        0,
        pixels[valid],
        torch.ones(int(valid.sum()), dtype=torch.int32, device=device),
    )
    hard_bin_writes = 0
    if hard_bin_numerator is not None:
        hard_rows = torch.floor(row + 0.5).to(torch.long)
        hard_columns = torch.floor(column + 0.5).to(torch.long)
        hard_valid = (
            (hard_rows >= 0)
            & (hard_rows < rows)
            & (hard_columns >= 0)
            & (hard_columns < columns)
        )
        hard_pixels = hard_rows[hard_valid] * columns + hard_columns[hard_valid]
        hard_bin_numerator.index_add_(0, hard_pixels, radiance[hard_valid])
        hard_bin_writes = int(hard_valid.sum())
    row_gap = (2.0 - row_distance.abs()).abs()[:, :, None].expand(-1, 4, 4)
    column_gap = (
        (2.0 - column_distance.abs()).abs()[:, None, :].expand(-1, 4, 4)
    )
    boundary_gap = torch.minimum(row_gap, column_gap).reshape(-1, 16)
    fractional_row = row - torch.floor(row)
    fractional_column = column - torch.floor(column)
    anchor_gap = torch.minimum(
        torch.minimum(fractional_row, 1.0 - fractional_row),
        torch.minimum(fractional_column, 1.0 - fractional_column),
    )
    thresholds = (1e-6, 1e-5, 1e-4, 1e-3)
    report = {
        "projected_events": int(in_frame.any(1).sum()),
        "footprint_writes": int(valid.sum()),
        "candidate_pairs": int(in_frame.sum()),
        "near_kernel_boundary_pairs": {
            str(value): int((in_frame & (boundary_gap <= value)).sum())
            for value in thresholds
        },
        "near_integer_anchor_events": {
            str(value): int((anchor_gap <= value).sum()) for value in thresholds
        },
        "event_kernel_weight_sum": float(effective_weights.sum()),
        "event_kernel_weight_squared_sum": float(effective_weights.square().sum()),
    }
    if hard_bin_numerator is not None:
        report["hard_bin_writes"] = hard_bin_writes
    return report


def _merge_event_report(total: dict[str, object], row: dict[str, object]) -> None:
    for key in (
        "projected_events",
        "footprint_writes",
        "candidate_pairs",
    ):
        total[key] = int(total.get(key, 0)) + int(row[key])
    if "hard_bin_writes" in row:
        total["hard_bin_writes"] = int(total.get("hard_bin_writes", 0)) + int(
            row["hard_bin_writes"]
        )
    for key in ("event_kernel_weight_sum", "event_kernel_weight_squared_sum"):
        total[key] = float(total.get(key, 0.0)) + float(row[key])
    for key in ("near_kernel_boundary_pairs", "near_integer_anchor_events"):
        target = total.setdefault(key, {})
        assert isinstance(target, dict)
        source = row[key]
        assert isinstance(source, dict)
        for threshold, count in source.items():
            target[threshold] = int(target.get(threshold, 0)) + int(count)


def _render_diagnostics(
    field: object,
    surface: SurfaceState,
    emitter_count: int,
    resolution: tuple[int, int],
    boundary: ObservationSphere,
    atlas: DirectionAtlas,
    config: PixelDiagnosticConfig,
    *,
    geometry: str = "base",
    keep_numerator: bool = False,
    keep_hard_bin: bool = False,
) -> tuple[list[PixelView], dict[str, object]]:
    device = surface.reference_points.device
    rows, columns = resolution
    pixels = rows * columns
    transport = config.transport()
    views: list[PixelView] = []
    outward = absorbed = retained = 0
    event_report: dict[str, object] = {}
    visibility_seconds = splat_seconds = 0.0
    started = time.perf_counter()
    if geometry == "base":
        visibility_points = surface.reference_points
        visibility_normals = surface.reference_normals
        render_points = surface.base_points
        render_normals = surface.base_normals
    elif geometry == "target":
        if surface.target_points is None or surface.target_normals is None:
            raise ValueError("target surface state is unavailable")
        visibility_points = render_points = surface.target_points
        visibility_normals = render_normals = surface.target_normals
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    for view_id in range(config.views):
        mass = torch.zeros(pixels, dtype=torch.float64, device=device)
        squared_mass = torch.zeros_like(mass)
        numerator = torch.zeros((pixels, 3), dtype=torch.float64, device=device)
        hard_bin_numerator = (
            torch.zeros_like(numerator) if keep_hard_bin else None
        )
        counts = torch.zeros(pixels, dtype=torch.int32, device=device)
        direction = atlas.directions[view_id]
        for start in range(0, emitter_count, config.emitter_chunk_size):
            stop = min(start + config.emitter_chunk_size, emitter_count)
            owners, chunk_outward, chunk_absorbed, seconds = _visible_owner_ids(
                field,
                visibility_points[start:stop],
                visibility_normals[start:stop],
                direction,
                boundary,
                transport,
            )
            outward += chunk_outward
            absorbed += chunk_absorbed
            retained += int(owners.numel())
            visibility_seconds += seconds
            _sync(device)
            splat_started = time.perf_counter()
            row = _diagnostic_splat(
                render_points[start:stop],
                render_normals[start:stop],
                owners,
                direction,
                atlas.right[view_id],
                atlas.up[view_id],
                boundary.center,
                field.lower,
                field.upper,
                resolution,
                config.detector_extent,
                mass,
                squared_mass,
                numerator,
                counts,
                config.ambient,
                hard_bin_numerator,
            )
            _sync(device)
            splat_seconds += time.perf_counter() - splat_started
            _merge_event_report(event_report, row)
        image = _finalize_image(
            mass, numerator, config.sensor_gain, config.support_threshold
        ).reshape(rows, columns, 3)
        views.append(
            PixelView(
                image.detach().cpu().to(torch.float32).numpy(),
                counts.detach().cpu().numpy(),
                mass.detach().cpu().to(torch.float32).numpy(),
                squared_mass.detach().cpu().to(torch.float32).numpy(),
                numerator.detach().cpu().to(torch.float32).numpy()
                if keep_numerator
                else None,
                hard_bin_numerator.detach().cpu().to(torch.float32).numpy()
                if hard_bin_numerator is not None
                else None,
            )
        )
        del mass, squared_mass, numerator, counts, image, hard_bin_numerator
        _progress(
            "v083_render_view",
            geometry=geometry,
            resolution=list(resolution),
            emitters=emitter_count,
            view=view_id,
        )
    candidate_pairs = max(int(event_report.get("candidate_pairs", 0)), 1)
    projected_events = max(int(event_report.get("projected_events", 0)), 1)
    for key in ("near_kernel_boundary_pairs", "near_integer_anchor_events"):
        values = event_report[key]
        assert isinstance(values, dict)
        denominator = candidate_pairs if "boundary" in key else projected_events
        event_report[f"{key}_fractions"] = {
            threshold: int(count) / denominator
            for threshold, count in values.items()
        }
    report = {
        "geometry": geometry,
        "resolution": list(resolution),
        "emitters": emitter_count,
        "attempted_packets": emitter_count * config.views,
        "outward_events": outward,
        "absorbed_events": absorbed,
        "retained_events": retained,
        **event_report,
        "runtime_seconds": time.perf_counter() - started,
        "visibility_seconds": visibility_seconds,
        "splat_and_accumulation_seconds": splat_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    return views, report


COUNT_BINS: tuple[tuple[str, float, float], ...] = (
    ("N=0", 0.0, 0.5),
    ("N=1", 0.5, 1.5),
    ("N=2-3", 1.5, 3.5),
    ("N=4-7", 3.5, 7.5),
    ("N=8-15", 7.5, 15.5),
    ("N=16-31", 15.5, 31.5),
    ("N=32-63", 31.5, 63.5),
    ("N>=64", 63.5, math.inf),
)


REGION_NAMES = (
    "BACKGROUND",
    "SILHOUETTE_0_1PX",
    "SILHOUETTE_1_2PX",
    "SILHOUETTE_2_4PX",
    "SILHOUETTE_4_8PX",
    "INTERIOR_GT_8PX",
)


def _silhouette_regions(
    target_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    foreground = np.linalg.norm(target_image, axis=2) > 0.0
    inside = ndimage.distance_transform_edt(foreground)
    outside = ndimage.distance_transform_edt(~foreground)
    signed = np.where(foreground, inside, -outside)
    absolute = np.abs(signed)
    code = np.zeros(foreground.shape, dtype=np.uint8)
    code[absolute <= 1.0] = 1
    code[(absolute > 1.0) & (absolute <= 2.0)] = 2
    code[(absolute > 2.0) & (absolute <= 4.0)] = 3
    code[(absolute > 4.0) & (absolute <= 8.0)] = 4
    code[foreground & (inside > 8.0)] = 5
    return foreground, signed.astype(np.float32), code


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _sampling_stats(mask: np.ndarray, counts: np.ndarray) -> dict[str, object]:
    selected = counts[mask]
    if selected.size == 0:
        return {
            "pixels": 0,
            "coverage": 0.0,
            "writes_per_pixel": 0.0,
            "zero_fraction": 0.0,
            "count_lt2_fraction": 0.0,
            "count_lt4_fraction": 0.0,
            **_quantiles(selected),
        }
    return {
        "pixels": int(selected.size),
        "coverage": float(np.mean(selected > 0)),
        "writes_per_pixel": float(np.mean(selected)),
        "zero_fraction": float(np.mean(selected == 0)),
        "count_lt2_fraction": float(np.mean(selected < 2)),
        "count_lt4_fraction": float(np.mean(selected < 4)),
        **_quantiles(selected),
    }


def _support_class_row(
    name: str,
    mask: np.ndarray,
    foreground: np.ndarray,
    error_sse: np.ndarray,
    total_sse: float,
) -> dict[str, object]:
    pixels = int(mask.sum())
    foreground_pixels = max(int(foreground.sum()), 1)
    class_sse = float(error_sse[mask].sum())
    return {
        "class": name,
        "pixels": pixels,
        "total_pixel_fraction": pixels / mask.size,
        "foreground_pixel_fraction": int((mask & foreground).sum())
        / foreground_pixels,
        "mc_self_sse": class_sse,
        "mc_self_mse_per_rgb_scalar": class_sse / max(3 * pixels, 1),
        "fraction_total_mc_self_sse": class_sse / max(total_sse, 1e-30),
    }


def _append_sample(
    target: dict[str, list[np.ndarray]],
    name: str,
    values: np.ndarray,
    *,
    maximum: int = 65_536,
) -> None:
    flat = values.reshape(-1)
    if flat.size > maximum:
        stride = max(flat.size // maximum, 1)
        flat = flat[::stride][:maximum]
    target.setdefault(name, []).append(flat.astype(np.float32, copy=False))


def _case_records(
    resolution: tuple[int, int],
    view_id: int,
    a: PixelView,
    b: PixelView,
    target: np.ndarray,
    foreground: np.ndarray,
    signed_distance: np.ndarray,
    region_code: np.ndarray,
) -> list[dict[str, object]]:
    rows, columns = resolution
    error = np.square(a.image.astype(np.float64) - b.image).sum(2)
    gradient = 0.5 * (
        _spatial_gradient(a.image) + _spatial_gradient(b.image)
    ).reshape(rows, columns)
    support_a = a.support.reshape(rows, columns)
    support_b = b.support.reshape(rows, columns)
    common = support_a & support_b
    switch = support_a ^ support_b
    mean_count = 0.5 * (
        a.counts.reshape(rows, columns) + b.counts.reshape(rows, columns)
    )
    definitions = {
        "stable_interior": common & (region_code == 5),
        "high_count": common & (mean_count >= 64),
        "low_count_foreground": foreground & (mean_count > 0) & (mean_count < 4),
        "support_switch": switch,
        "silhouette": region_code == 1,
    }
    records: list[dict[str, object]] = []
    for category, mask in definitions.items():
        ids = np.flatnonzero(mask.reshape(-1))
        if ids.size == 0:
            continue
        local_error = error.reshape(-1)[ids]
        order = np.argsort(local_error)
        if category in ("support_switch", "silhouette"):
            order = order[::-1]
        selected = ids[order[:5]]
        for pixel in selected:
            row, column = divmod(int(pixel), columns)
            records.append(
                {
                    "category": category,
                    "resolution": list(resolution),
                    "view": view_id,
                    "pixel": [row, column],
                    "target_rgb": target[row, column].tolist(),
                    "sample_a_rgb": a.image[row, column].tolist(),
                    "sample_b_rgb": b.image[row, column].tolist(),
                    "N_A": int(a.counts[pixel]),
                    "N_B": int(b.counts[pixel]),
                    "W_A": float(a.mass[pixel]),
                    "W_B": float(b.mass[pixel]),
                    "Q_A": float(a.squared_mass[pixel]),
                    "Q_B": float(b.squared_mass[pixel]),
                    "support_A": bool(a.support[pixel]),
                    "support_B": bool(b.support[pixel]),
                    "distance_to_silhouette": float(signed_distance[row, column]),
                    "local_mc_sse": float(error[row, column]),
                    "spatial_rgb_gradient_norm": float(gradient[row, column]),
                    "near_final_support_threshold": min(
                        abs(float(a.mass[pixel]) - 0.05),
                        abs(float(b.mass[pixel]) - 0.05),
                    )
                    <= 1e-3,
                }
            )
    return records


def _analyze_pair(
    resolution: tuple[int, int],
    a_views: list[PixelView],
    b_views: list[PixelView],
    target_images: list[np.ndarray],
    capture_view: int,
) -> tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]]:
    total_pixels = 0
    total_sse = 0.0
    target_foreground_pixels = 0
    support_switch_pixels = 0
    support_switch_foreground = 0
    support_switch_sse = 0.0
    support_rows: dict[str, dict[str, float]] = {}
    region_rows: dict[str, dict[str, float]] = {}
    foreground_rows: dict[str, list[dict[str, object]]] = {}
    count_errors: dict[str, list[np.ndarray]] = {}
    count_gradients: dict[str, list[np.ndarray]] = {}
    correlation_samples: dict[str, list[np.ndarray]] = {}
    case_records: list[dict[str, object]] = []
    capture: dict[str, np.ndarray] = {}
    for view_id, (a, b, target) in enumerate(
        zip(a_views, b_views, target_images)
    ):
        rows, columns, _ = a.image.shape
        foreground, signed_distance, region_code = _silhouette_regions(target)
        error_sse = np.square(a.image.astype(np.float64) - b.image).sum(2)
        error_mse = error_sse / 3.0
        view_sse = float(error_sse.sum())
        total_sse += view_sse
        total_pixels += rows * columns
        target_foreground_pixels += int(foreground.sum())
        support_a = a.support.reshape(rows, columns)
        support_b = b.support.reshape(rows, columns)
        a_only = support_a & ~support_b
        b_only = ~support_a & support_b
        neither = ~support_a & ~support_b
        common = support_a & support_b
        switch = a_only | b_only
        support_switch_pixels += int(switch.sum())
        support_switch_foreground += int((switch & foreground).sum())
        support_switch_sse += float(error_sse[switch].sum())
        for name, mask in (
            ("UNSUPPORTED_BOTH", neither),
            ("COMMON_SUPPORT", common),
            ("SUPPORT_SWITCH_A_ONLY", a_only),
            ("SUPPORT_SWITCH_B_ONLY", b_only),
        ):
            row = _support_class_row(
                name, mask, foreground, error_sse, view_sse
            )
            aggregate = support_rows.setdefault(
                name,
                {"pixels": 0.0, "foreground_pixels": 0.0, "sse": 0.0},
            )
            aggregate["pixels"] += int(mask.sum())
            aggregate["foreground_pixels"] += int((mask & foreground).sum())
            aggregate["sse"] += float(row["mc_self_sse"])
        gradient = 0.5 * (
            _spatial_gradient(a.image) + _spatial_gradient(b.image)
        ).reshape(rows, columns)
        mean_count = 0.5 * (
            a.counts.reshape(rows, columns)
            + b.counts.reshape(rows, columns)
        )
        for label, lower, upper in COUNT_BINS:
            mask = (mean_count >= lower) & (mean_count < upper)
            count_errors.setdefault(label, []).append(error_mse[mask].astype(np.float32))
            count_gradients.setdefault(label, []).append(gradient[mask].astype(np.float32))
        for code, name in enumerate(REGION_NAMES):
            mask = region_code == code
            aggregate = region_rows.setdefault(
                name,
                {
                    "pixels": 0.0,
                    "sse": 0.0,
                    "support_switch_pixels": 0.0,
                    "count_sum": 0.0,
                    "intensity_sum": 0.0,
                    "gradient_sum": 0.0,
                },
            )
            aggregate["pixels"] += int(mask.sum())
            aggregate["sse"] += float(error_sse[mask].sum())
            aggregate["support_switch_pixels"] += int((switch & mask).sum())
            aggregate["count_sum"] += float(a.counts.reshape(rows, columns)[mask].sum())
            aggregate["intensity_sum"] += float(np.mean(a.image, axis=2)[mask].sum())
            aggregate["gradient_sum"] += float(gradient[mask].sum())
        predicted = a.support.reshape(rows, columns)
        masks = {
            "target_foreground": foreground,
            "predicted_foreground": predicted,
            "union_foreground": foreground | predicted,
            "intersection_foreground": foreground & predicted,
            "interior_foreground": region_code == 5,
        }
        count_image = a.counts.reshape(rows, columns)
        for name, mask in masks.items():
            foreground_rows.setdefault(name, []).append(
                _sampling_stats(mask, count_image)
            )
        delta_count = np.abs(a.counts.astype(np.float64) - b.counts)
        delta_weight = np.abs(a.mass.astype(np.float64) - b.mass)
        mean_weight = 0.5 * (a.mass.astype(np.float64) + b.mass)
        _append_sample(correlation_samples, "delta_count", delta_count)
        _append_sample(correlation_samples, "delta_weight", delta_weight)
        _append_sample(correlation_samples, "mean_count", mean_count)
        _append_sample(correlation_samples, "mean_weight", mean_weight)
        _append_sample(correlation_samples, "error", error_mse)
        _append_sample(correlation_samples, "common", common.astype(np.float32))
        if view_id == min(capture_view, len(a_views) - 1):
            capture = {
                "count": count_image,
                "mass": a.mass.reshape(rows, columns),
                "delta_count": delta_count.reshape(rows, columns),
                "delta_weight": delta_weight.reshape(rows, columns),
                "support_switch": switch,
                "mc_error": error_mse,
                "silhouette_distance": signed_distance,
                "spatial_gradient": gradient,
                "sample_a": a.image,
                "sample_b": b.image,
                "target": target,
            }
            case_records = _case_records(
                resolution,
                view_id,
                a,
                b,
                target,
                foreground,
                signed_distance,
                region_code,
            )
    support_decomposition = []
    for name, values in support_rows.items():
        pixels = int(values["pixels"])
        sse = float(values["sse"])
        support_decomposition.append(
            {
                "class": name,
                "pixels": pixels,
                "total_pixel_fraction": pixels / total_pixels,
                "foreground_pixel_fraction": float(values["foreground_pixels"])
                / max(target_foreground_pixels, 1),
                "mc_self_sse": sse,
                "mc_self_mse_per_rgb_scalar": sse / max(3 * pixels, 1),
                "fraction_total_mc_self_sse": sse / max(total_sse, 1e-30),
            }
        )
    count_rows = []
    for label, _, _ in COUNT_BINS:
        errors = np.concatenate(count_errors[label])
        gradients = np.concatenate(count_gradients[label])
        count_rows.append(
            {
                "count_bin": label,
                "pixels": int(errors.size),
                "mean_mc_self_mse": float(errors.mean()) if errors.size else 0.0,
                "median_mc_self_mse": float(np.median(errors)) if errors.size else 0.0,
                "p90_mc_self_mse": float(np.quantile(errors, 0.90)) if errors.size else 0.0,
                "p95_mc_self_mse": float(np.quantile(errors, 0.95)) if errors.size else 0.0,
                "mean_spatial_rgb_gradient": float(gradients.mean()) if gradients.size else 0.0,
                "spatial_rgb_gradient_variance": float(gradients.var()) if gradients.size else 0.0,
            }
        )
    regions = []
    for name in REGION_NAMES:
        values = region_rows[name]
        pixels = int(values["pixels"])
        sse = float(values["sse"])
        regions.append(
            {
                "region": name,
                "pixels": pixels,
                "mean_contribution_count": float(values["count_sum"]) / max(pixels, 1),
                "support_switch_fraction": float(values["support_switch_pixels"])
                / max(pixels, 1),
                "mc_self_mse_per_rgb_scalar": sse / max(3 * pixels, 1),
                "mc_self_sse": sse,
                "fraction_total_mc_self_sse": sse / max(total_sse, 1e-30),
                "mean_rgb_intensity": float(values["intensity_sum"]) / max(pixels, 1),
                "mean_spatial_rgb_gradient": float(values["gradient_sum"])
                / max(pixels, 1),
            }
        )
    foreground_statistics = {}
    for name, rows_for_mask in foreground_rows.items():
        pixels = sum(int(row["pixels"]) for row in rows_for_mask)
        if pixels == 0:
            foreground_statistics[name] = _sampling_stats(
                np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
            )
            continue
        # Weighted means are exact; quantiles are summarized per view to keep
        # full-HD memory bounded.
        foreground_statistics[name] = {
            "pixels": pixels,
            "coverage": sum(float(row["coverage"]) * int(row["pixels"]) for row in rows_for_mask) / pixels,
            "writes_per_pixel": sum(float(row["writes_per_pixel"]) * int(row["pixels"]) for row in rows_for_mask) / pixels,
            "zero_fraction": sum(float(row["zero_fraction"]) * int(row["pixels"]) for row in rows_for_mask) / pixels,
            "count_lt2_fraction": sum(float(row["count_lt2_fraction"]) * int(row["pixels"]) for row in rows_for_mask) / pixels,
            "count_lt4_fraction": sum(float(row["count_lt4_fraction"]) * int(row["pixels"]) for row in rows_for_mask) / pixels,
            "median_of_view_medians": float(statistics.median(float(row["median"]) for row in rows_for_mask)),
            "median_of_view_p90": float(statistics.median(float(row["p90"]) for row in rows_for_mask)),
            "median_of_view_p95": float(statistics.median(float(row["p95"]) for row in rows_for_mask)),
        }
    samples = {
        key: np.concatenate(value) for key, value in correlation_samples.items()
    }
    common_sample = samples["common"] > 0.5
    correlations = {}
    for predictor in ("delta_count", "delta_weight", "mean_count", "mean_weight"):
        correlations[predictor] = {
            "all_pixels": _safe_correlation(samples[predictor], samples["error"]),
            "common_support": _safe_correlation(
                samples[predictor][common_sample], samples["error"][common_sample]
            ),
        }
    return (
        {
            "resolution": list(resolution),
            "pixels": total_pixels,
            "mc_self_sse": total_sse,
            "mc_self_mse_per_rgb_scalar": total_sse / (3 * total_pixels),
            "mc_self_rmse": math.sqrt(total_sse / (3 * total_pixels)),
            "support_switch_pixel_fraction": support_switch_pixels / total_pixels,
            "support_switch_fraction_on_target_foreground": support_switch_foreground
            / max(target_foreground_pixels, 1),
            "support_switch_mc_self_sse": support_switch_sse,
            "fraction_self_sse_from_support_switch_pixels": support_switch_sse
            / max(total_sse, 1e-30),
            "support_decomposition": support_decomposition,
            "count_bins": count_rows,
            "silhouette_regions": regions,
            "foreground_sampling": foreground_statistics,
            "correlations": correlations,
        },
        capture,
        case_records,
    )


def _compare_reference(
    resolution: tuple[int, int],
    reference: list[ReferenceView],
    candidate: list[PixelView],
    target_images: list[np.ndarray],
    seed: int | str,
) -> dict[str, object]:
    totals = {
        "whole_sse": 0.0,
        "whole_scalars": 0,
        "common_sse": 0.0,
        "common_scalars": 0,
        "switch_sse": 0.0,
        "switch_scalars": 0,
        "foreground_sse": 0.0,
        "foreground_scalars": 0,
        "interior_sse": 0.0,
        "interior_scalars": 0,
        "silhouette_sse": 0.0,
        "silhouette_scalars": 0,
        "pixels": 0,
        "switch_pixels": 0,
        "foreground_pixels": 0,
        "foreground_switch_pixels": 0,
    }
    band_sse = {name: 0.0 for name in REGION_NAMES[1:5]}
    band_scalars = {name: 0 for name in REGION_NAMES[1:5]}
    for left, right, target in zip(reference, candidate, target_images):
        rows, columns, _ = target.shape
        foreground, _, region = _silhouette_regions(target)
        right_support = right.support.reshape(rows, columns)
        common = left.support & right_support
        switch = left.support ^ right_support
        error = np.square(left.image.astype(np.float64) - right.image).sum(2)
        totals["whole_sse"] += float(error.sum())
        totals["whole_scalars"] += error.size * 3
        for name, mask in (
            ("common", common),
            ("switch", switch),
            ("foreground", foreground),
            ("interior", region == 5),
            ("silhouette", (region >= 1) & (region <= 4)),
        ):
            totals[f"{name}_sse"] += float(error[mask].sum())
            totals[f"{name}_scalars"] += int(mask.sum()) * 3
        totals["pixels"] += error.size
        totals["switch_pixels"] += int(switch.sum())
        totals["foreground_pixels"] += int(foreground.sum())
        totals["foreground_switch_pixels"] += int((switch & foreground).sum())
        for code, name in enumerate(REGION_NAMES[1:5], start=1):
            mask = region == code
            band_sse[name] += float(error[mask].sum())
            band_scalars[name] += int(mask.sum()) * 3
    result: dict[str, object] = {
        "resolution": list(resolution),
        "reference_seed": 101,
        "comparison_seed": seed,
        "whole_image_mse": totals["whole_sse"] / totals["whole_scalars"],
        "whole_image_sse": totals["whole_sse"],
        "common_support_mse": totals["common_sse"]
        / max(totals["common_scalars"], 1),
        "foreground_mse": totals["foreground_sse"]
        / max(totals["foreground_scalars"], 1),
        "interior_mse": totals["interior_sse"]
        / max(totals["interior_scalars"], 1),
        "silhouette_0_8px_mse": totals["silhouette_sse"]
        / max(totals["silhouette_scalars"], 1),
        "support_switch_mse": totals["switch_sse"]
        / max(totals["switch_scalars"], 1),
        "support_switch_fraction": totals["switch_pixels"] / totals["pixels"],
        "support_switch_foreground_fraction": totals["foreground_switch_pixels"]
        / max(totals["foreground_pixels"], 1),
        "fraction_sse_from_support_switch": totals["switch_sse"]
        / max(totals["whole_sse"], 1e-30),
    }
    result["silhouette_band_mse"] = {
        name: band_sse[name] / max(band_scalars[name], 1)
        for name in band_sse
    }
    return result


def _multiseed_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "whole_image_mse",
        "common_support_mse",
        "foreground_mse",
        "interior_mse",
        "silhouette_0_8px_mse",
        "support_switch_mse",
        "support_switch_fraction",
        "fraction_sse_from_support_switch",
    )
    summaries: list[dict[str, object]] = []
    resolutions = sorted({tuple(row["resolution"]) for row in rows})
    for resolution in resolutions:
        selected = [row for row in rows if tuple(row["resolution"]) == resolution]
        summary: dict[str, object] = {
            "resolution": list(resolution),
            "comparisons": len(selected),
            "sample_sets": len(selected) + 1,
            "reference_seed": 101,
            "comparison_seeds": [row["comparison_seed"] for row in selected],
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            standard = float(values.std(ddof=1)) if values.size > 1 else 0.0
            half = 1.96 * standard / math.sqrt(max(values.size, 1))
            summary[metric] = {
                "mean": float(values.mean()),
                "std": standard,
                "median": float(np.median(values)),
                "ci95": [float(values.mean() - half), float(values.mean() + half)],
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        summaries.append(summary)
    return summaries


def _paired_resolution_trend(rows: list[dict[str, object]]) -> dict[str, object]:
    low = {
        row["comparison_seed"]: float(row["whole_image_mse"])
        for row in rows
        if tuple(row["resolution"]) == (256, 256)
    }
    high = {
        row["comparison_seed"]: float(row["whole_image_mse"])
        for row in rows
        if tuple(row["resolution"]) == (1080, 1920)
    }
    seeds = sorted(set(low) & set(high), key=str)
    low_values = np.asarray([low[seed] for seed in seeds])
    high_values = np.asarray([high[seed] for seed in seeds])
    test = stats.ttest_rel(high_values, low_values)
    differences = high_values - low_values
    return {
        "comparison_seeds": seeds,
        "low_values": low_values.tolist(),
        "fullhd_values": high_values.tolist(),
        "paired_differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "fullhd_to_low_mean_ratio": float(high_values.mean() / low_values.mean()),
        "paired_t_statistic": float(test.statistic),
        "paired_p_value": float(test.pvalue),
        "significant_at_0_05": bool(test.pvalue < 0.05),
    }


def _cubic_with_derivative(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(value)
    inner = 2.0 / 3.0 - absolute**2 + 0.5 * absolute**3
    outer = np.maximum(2.0 - absolute, 0.0) ** 3 / 6.0
    weights = np.where(absolute < 1.0, inner, outer)
    derivative = np.where(
        absolute < 1.0,
        np.sign(value) * (-2.0 * absolute + 1.5 * absolute**2),
        -0.5 * np.sign(value) * np.maximum(2.0 - absolute, 0.0) ** 2,
    )
    return weights, derivative


def _stencil_vector(
    row: float, column: float, dtype: np.dtype[Any]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[float, float]]]:
    row_value = dtype.type(row)
    column_value = dtype.type(column)
    base_row = math.floor(float(row_value))
    base_column = math.floor(float(column_value))
    row_ids = np.arange(base_row - 1, base_row + 3)
    column_ids = np.arange(base_column - 1, base_column + 3)
    rw, rd = _cubic_with_derivative(
        (row_value - row_ids).astype(dtype)
    )
    cw, cd = _cubic_with_derivative(
        (column_value - column_ids).astype(dtype)
    )
    values: dict[tuple[int, int], float] = {}
    derivatives: dict[tuple[int, int], tuple[float, float]] = {}
    for row_index, pixel_row in enumerate(row_ids):
        for column_index, pixel_column in enumerate(column_ids):
            key = (int(pixel_row), int(pixel_column))
            weight = float(rw[row_index] * cw[column_index])
            if weight > 0.0:
                values[key] = weight
            derivatives[key] = (
                float(rd[row_index] * cw[column_index]),
                float(rw[row_index] * cd[column_index]),
            )
    return values, derivatives


def _stencil_continuity() -> dict[str, object]:
    epsilons = (1e-5, 1e-4, 1e-3, 1e-2, 0.1)
    cases = {
        "far_from_integer": (12.37, 8.41),
        "near_integer": (12.0 - 1e-8, 8.0 + 1e-8),
    }
    rows: list[dict[str, object]] = []
    for dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        for name, (row, column) in cases.items():
            for epsilon in epsilons:
                minus, minus_derivative = _stencil_vector(
                    row - epsilon, column - epsilon, dtype
                )
                plus, plus_derivative = _stencil_vector(
                    row + epsilon, column + epsilon, dtype
                )
                keys = sorted(set(minus) | set(plus))
                minus_vector = np.asarray([minus.get(key, 0.0) for key in keys])
                plus_vector = np.asarray([plus.get(key, 0.0) for key in keys])
                derivative_keys = sorted(set(minus_derivative) | set(plus_derivative))
                minus_d = np.asarray(
                    [minus_derivative.get(key, (0.0, 0.0)) for key in derivative_keys]
                )
                plus_d = np.asarray(
                    [plus_derivative.get(key, (0.0, 0.0)) for key in derivative_keys]
                )
                rows.append(
                    {
                        "dtype": dtype.name,
                        "case": name,
                        "epsilon_pixels": epsilon,
                        "minus_members": len(minus),
                        "plus_members": len(plus),
                        "membership_symmetric_difference": len(set(minus) ^ set(plus)),
                        "minus_total_weight": float(minus_vector.sum()),
                        "plus_total_weight": float(plus_vector.sum()),
                        "value_jump_l2": float(np.linalg.norm(plus_vector - minus_vector)),
                        "central_value_derivative_l2": float(
                            np.linalg.norm((plus_vector - minus_vector) / (2 * epsilon))
                        ),
                        "analytic_derivative_change_l2": float(
                            np.linalg.norm(plus_d - minus_d)
                        ),
                    }
                )
    return {
        "implemented_anchor": "floor(continuous coordinate)",
        "implemented_offsets": [-1, 0, 1, 2],
        "support_membership_changes_at_integer_anchor": True,
        "mathematical_value_continuity": "cubic B-spline entering/leaving weights are zero",
        "mathematical_first_derivative_continuity": "entering/leaving derivatives are zero",
        "rows": rows,
    }


def _visibility_codes(
    field: object,
    points: Tensor,
    normals: Tensor,
    direction: Tensor,
    boundary: ObservationSphere,
    config: PixelDiagnosticConfig,
) -> Tensor:
    dot = normals @ direction
    outward = dot > 1e-8
    owners = torch.nonzero(outward, as_tuple=False).flatten()
    codes = torch.zeros(points.shape[0], dtype=torch.int8, device=points.device)
    if owners.numel() == 0:
        return codes
    origins = points[owners]
    directions = direction.expand_as(origins)
    maximum = boundary.exit_times(origins, directions)
    hit, _ = first_zero_set_intersections(
        field,
        origins,
        directions,
        maximum,
        epsilon=2e-4,
        samples=config.root_samples,
        bisection_steps=config.bisection_steps,
        chunk_size=8192,
    )
    codes[owners] = torch.where(
        hit,
        torch.ones_like(hit, dtype=torch.int8),
        torch.full_like(hit, 2, dtype=torch.int8),
    )
    return codes


def _controlled_visibility_switching(
    base: object,
    surface: SurfaceState,
    atlas: DirectionAtlas,
    boundary: ObservationSphere,
    config: PixelDiagnosticConfig,
) -> dict[str, object]:
    count = min(65_536, surface.reference_points.shape[0])
    points = surface.reference_points[:count]
    normals = surface.reference_normals[:count]
    rows: list[dict[str, object]] = []
    for view_id in (0, 5, 10, 15):
        direction = atlas.directions[view_id]
        base_codes = _visibility_codes(
            base, points, normals, direction, boundary, config
        )
        dot = normals @ direction
        for epsilon in (1e-5, 1e-4, 1e-3, 1e-2):
            perturbed = direction + epsilon * atlas.right[view_id]
            perturbed /= torch.linalg.vector_norm(perturbed)
            codes = _visibility_codes(
                base, points, normals, perturbed, boundary, config
            )
            changed = codes != base_codes
            visibility_changed = ((codes == 1) & (base_codes == 2)) | (
                (codes == 2) & (base_codes == 1)
            )
            outward_changed = (codes == 0) != (base_codes == 0)
            rows.append(
                {
                    "view": view_id,
                    "direction_perturbation": epsilon,
                    "identities": count,
                    "any_state_switches": int(changed.sum()),
                    "any_state_switch_fraction": float(changed.float().mean()),
                    "visible_occluded_switches": int(visibility_changed.sum()),
                    "visible_occluded_switch_fraction": float(
                        visibility_changed.float().mean()
                    ),
                    "outward_switches": int(outward_changed.sum()),
                    "grazing_fraction_among_switches": float(
                        (dot[changed].abs() < 0.05).float().mean()
                    )
                    if bool(changed.any())
                    else 0.0,
                }
            )
    return {
        "same_emitter_identities": count,
        "perturbation": "view direction along transverse right vector",
        "thresholds": {
            "outward_dot": 1e-8,
            "zero_set_intersection_epsilon": 2e-4,
        },
        "rows": rows,
    }


def _normalization_audit(
    emitter_counts: list[int],
    renders: list[list[PixelView]],
    target_images: list[np.ndarray],
    sensor_gain: float,
) -> dict[str, object]:
    highest = renders[-1]
    rows: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for count, views in zip(emitter_counts, renders):
        energy = foreground_sum = 0.0
        foreground_pixels = 0
        comparison_sse = 0.0
        comparison_scalars = 0
        masses: list[np.ndarray] = []
        for view, reference, target in zip(views, highest, target_images):
            foreground = np.linalg.norm(target, axis=2) > 0.0
            energy += float(np.square(view.image.astype(np.float64)).sum())
            foreground_sum += float(np.mean(view.image, axis=2)[foreground].sum())
            foreground_pixels += int(foreground.sum())
            common = view.support.reshape(foreground.shape) & reference.support.reshape(
                foreground.shape
            )
            difference = view.image.astype(np.float64) - reference.image
            comparison_sse += float(np.square(difference[common]).sum())
            comparison_scalars += int(common.sum()) * 3
            masses.append(view.mass.reshape(foreground.shape)[foreground])
            assert view.numerator is not None
            numerator = view.numerator.reshape(*foreground.shape, 3).astype(np.float64)
            mean_mass = float(view.mass.reshape(foreground.shape)[foreground].mean())
            global_image = sensor_gain * numerator / count
            expected_density_image = sensor_gain * numerator / max(mean_mass, 1e-30)
            controls.append(
                {
                    "emitters": count,
                    "view": len([row for row in controls if row["emitters"] == count]),
                    "current_foreground_mean": float(
                        np.mean(view.image, axis=2)[foreground].mean()
                    ),
                    "global_N_normalized_foreground_mean": float(
                        np.mean(global_image, axis=2)[foreground].mean()
                    ),
                    "expected_W_normalized_foreground_mean": float(
                        np.mean(expected_density_image, axis=2)[foreground].mean()
                    ),
                    "mean_foreground_W": mean_mass,
                }
            )
        mass = np.concatenate(masses)
        rows.append(
            {
                "emitters": count,
                "relative_to_base": count / emitter_counts[0],
                "mean_rgb_energy": energy / len(views),
                "total_rgb_energy": energy,
                "foreground_mean_intensity": foreground_sum
                / max(foreground_pixels, 1),
                "rmse_against_highest_on_common_support": math.sqrt(
                    comparison_sse / max(comparison_scalars, 1)
                ),
                "foreground_W_mean": float(mass.mean()),
                "foreground_W_std": float(mass.std()),
                "foreground_W_cv": float(mass.std() / max(mass.mean(), 1e-30)),
            }
        )
    return {
        "equation": (
            "I_p = 1[W_p >= 0.05] clamp(1.5 A_p / max(W_p,1e-30), 0, 1); "
            "A_p=sum_i w_ip C_i l_i; W_p=sum_i w_ip"
        ),
        "expected_sample_count_behavior": (
            "On stable support, multiplying sample count multiplies A_p and W_p "
            "together, so brightness is invariant while variance decreases."
        ),
        "rows": rows,
        "alternative_estimator_controls_per_view": controls,
        "control_interpretation": (
            "Global-N and expected-W estimators are diagnostic only; the current "
            "production estimator is already per-pixel weight normalized."
        ),
    }


def _spatial_structure(error: np.ndarray) -> dict[str, float]:
    values = error.astype(np.float64)
    centered = values - values.mean()
    variance = float(np.square(centered).mean())
    horizontal = float((centered[:, :-1] * centered[:, 1:]).mean()) / max(
        variance, 1e-30
    )
    vertical = float((centered[:-1] * centered[1:]).mean()) / max(
        variance, 1e-30
    )
    checker = np.fromfunction(lambda y, x: 1.0 - 2.0 * ((x + y) % 2), values.shape)
    checker_correlation = float(
        abs(np.sum(centered * checker))
        / math.sqrt(max(np.square(centered).sum() * np.square(checker).sum(), 1e-30))
    )
    return {
        "horizontal_lag1_correlation": horizontal,
        "vertical_lag1_correlation": vertical,
        "checkerboard_correlation": checker_correlation,
    }


def _vector_metrics(analytic: Tensor, finite: Tensor, mask: Tensor) -> dict[str, float]:
    left = analytic[mask]
    right = finite[mask]
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    relative = torch.linalg.vector_norm(left - right) / right_norm.clamp_min(1e-30)
    cosine = (left @ right) / (left_norm * right_norm).clamp_min(1e-30)
    active = (left.abs() + right.abs()) > 1e-12
    sign = (
        (torch.sign(left[active]) == torch.sign(right[active])).to(torch.float64).mean()
        if bool(active.any())
        else torch.tensor(1.0, device=left.device)
    )
    return {
        "relative_error": float(relative),
        "cosine_similarity": float(cosine),
        "analytic_norm": float(left_norm),
        "finite_difference_norm": float(right_norm),
        "sign_agreement": float(sign),
    }


def _gradient_stability(prepared: object, config: PixelDiagnosticConfig) -> dict[str, object]:
    from .corrected_birth import (
        CorrectedBirthConfig,
        _active_components,
        _build_context,
        _evaluate,
        _jacobians,
        _render,
        _visibility_cells,
    )
    from .sampling_diagnostic import _copy_layout
    from .locality import wendland_values

    corrected = CorrectedBirthConfig(
        dictionary_count=64,
        initial_count=config.gradient_parameters,
        surface_samples=config.gradient_surface_samples,
        views=config.gradient_views,
        resolution=config.gradient_resolution,
        surface_scramble_seed=config.sobol_seeds[0],
    )
    heldout_config = replace(
        corrected, surface_scramble_seed=config.sobol_seeds[1]
    )
    context = _build_context(prepared, corrected)
    heldout = _build_context(prepared, heldout_config)
    _copy_layout(heldout, context)
    device = context.reference_points.device
    active = torch.arange(corrected.initial_count, device=device)
    coefficients = torch.zeros(active.numel(), dtype=torch.float64, device=device)
    components = _active_components(context, active)
    state = _evaluate(context, active, coefficients, components)
    matrices = _jacobians(context, state, *components)
    heldout_components = _active_components(heldout, active)
    heldout_state = _evaluate(heldout, active, coefficients, heldout_components)
    heldout_matrices = _jacobians(
        heldout, heldout_state, *heldout_components
    )

    region_masks: dict[str, list[Tensor]] = {
        "all_pixels": [],
        "common_support": [],
        "silhouette_0_2px": [],
        "interior_gt8px": [],
    }
    region_codes: list[np.ndarray] = []
    for image, render in zip(context.target_images, state.renders):
        rows, columns = corrected.resolution_shape
        target = image.reshape(rows, columns, 3).detach().cpu().numpy()
        _, _, codes = _silhouette_regions(target)
        region_codes.append(codes)
        support = render.mass >= corrected.support_threshold
        pixel_masks = {
            "all_pixels": np.ones((rows, columns), dtype=bool),
            "common_support": support.detach().cpu().numpy().reshape(rows, columns),
            "silhouette_0_2px": (codes == 1) | (codes == 2),
            "interior_gt8px": codes == 5,
        }
        for name, mask in pixel_masks.items():
            region_masks[name].append(
                torch.from_numpy(np.repeat(mask.reshape(-1), 3)).to(device)
            )
    concatenated_masks = {
        name: torch.cat(parts) for name, parts in region_masks.items()
    }

    def column_for(parameter: int, source: list[Tensor]) -> Tensor:
        one = torch.zeros(active.numel(), dtype=torch.float64, device=device)
        one[parameter] = 1.0
        return torch.cat(
            [torch.sparse.mm(matrix, one[:, None]).flatten() for matrix in source]
        )

    analytic_columns = [column_for(parameter, matrices) for parameter in range(active.numel())]
    energy = torch.stack([column.square().sum() for column in analytic_columns])
    silhouette_energy = []
    interior_energy = []
    for column in analytic_columns:
        silhouette_energy.append(
            column[concatenated_masks["silhouette_0_2px"]].square().sum()
        )
        interior_energy.append(
            column[concatenated_masks["interior_gt8px"]].square().sum()
        )
    silhouette_energy_tensor = torch.stack(silhouette_energy)
    interior_energy_tensor = torch.stack(interior_energy)
    categories: dict[str, int] = {}
    categories["deep_interior"] = int(torch.argmax(interior_energy_tensor))
    categories["silhouette"] = int(torch.argmax(silhouette_energy_tensor))
    excluded = set(categories.values())
    order = torch.argsort(energy, descending=True).tolist()
    categories["smooth_visible_surface"] = next(item for item in order if item not in excluded)
    excluded.add(categories["smooth_visible_surface"])
    categories["occlusion_boundary"] = next(
        item
        for item in torch.argsort(silhouette_energy_tensor, descending=True).tolist()
        if item not in excluded
    )
    excluded.add(categories["occlusion_boundary"])
    centers = components[0].centers
    epsilon = 2e-3
    normal_plus = context.base.gradient(centers + epsilon)
    normal_minus = context.base.gradient(centers - epsilon)
    curvature_proxy = torch.linalg.vector_norm(normal_plus - normal_minus, dim=1)
    categories["high_curvature"] = next(
        item
        for item in torch.argsort(curvature_proxy, descending=True).tolist()
        if item not in excluded
    )

    atlas = nested_fibonacci_atlas(device, (corrected.views,))
    boundary = enclosing_observation_sphere(context.reference_points)

    class GenericLocalField:
        def __init__(self, coefficients_for_field: Tensor) -> None:
            self.coefficients = coefficients_for_field
            self.lower = context.base.lower
            self.upper = context.base.upper

        def value(self, points: Tensor) -> Tensor:
            shape = points.shape[:-1]
            flat_points = points.reshape(-1, 3)
            values = context.base.value(flat_points)
            for start in range(0, flat_points.shape[0], 8192):
                stop = min(start + 8192, flat_points.shape[0])
                offsets = (
                    flat_points[start:stop, None, :]
                    - components[0].centers[None, :, :]
                )
                basis = wendland_values(
                    offsets.reshape(-1, 3),
                    components[0].radii[None, :]
                    .expand(stop - start, -1)
                    .reshape(-1),
                ).reshape(stop - start, -1)
                values[start:stop] += basis @ self.coefficients
            return values.reshape(shape)

    def full_images(coefficients_for_render: Tensor, evaluated: object) -> list[Tensor]:
        cells, _ = _visibility_cells(
            GenericLocalField(coefficients_for_render),
            evaluated.points,
            evaluated.normals,
            atlas,
            boundary,
            corrected,
        )
        images, _ = _render(
            evaluated.points, evaluated.normals, context, cells
        )
        return images

    parameter_rows: list[dict[str, object]] = []
    for category, parameter in categories.items():
        analytic = analytic_columns[parameter]
        heldout_analytic = column_for(parameter, heldout_matrices)
        sample_gradient = _vector_metrics(
            analytic,
            heldout_analytic,
            concatenated_masks["all_pixels"],
        )
        for step in config.gradient_epsilons:
            plus_coefficients = coefficients.clone()
            minus_coefficients = coefficients.clone()
            plus_coefficients[parameter] += step
            minus_coefficients[parameter] -= step
            plus = _evaluate(context, active, plus_coefficients, components)
            minus = _evaluate(context, active, minus_coefficients, components)
            frozen = torch.cat(
                [
                    (right - left) / (2.0 * step)
                    for right, left in zip(plus.images, minus.images)
                ]
            )
            plus_full = full_images(plus_coefficients, plus)
            minus_full = full_images(minus_coefficients, minus)
            full = torch.cat(
                [
                    (right - left) / (2.0 * step)
                    for right, left in zip(plus_full, minus_full)
                ]
            )
            support_switch_parts = []
            for right, left in zip(plus_full, minus_full):
                right_support = right.reshape(-1, 3).abs().sum(1) > 0.0
                left_support = left.reshape(-1, 3).abs().sum(1) > 0.0
                support_switch_parts.append(
                    (right_support ^ left_support).repeat_interleave(3)
                )
            masks = dict(concatenated_masks)
            masks["support_switch_pixels"] = torch.cat(support_switch_parts)
            parameter_rows.append(
                {
                    "category": category,
                    "parameter": parameter,
                    "epsilon": step,
                    "sample_realization_gradient": sample_gradient,
                    "regions": {
                        name: {
                            "analytic_vs_frozen": _vector_metrics(
                                analytic, frozen, mask
                            ),
                            "analytic_vs_full": _vector_metrics(
                                analytic, full, mask
                            ),
                            "frozen_vs_full": _vector_metrics(
                                frozen, full, mask
                            ),
                            "pixels_or_scalars": int(mask.sum()),
                        }
                        for name, mask in masks.items()
                    },
                    "full_rerender_support_switch_scalar_fraction": float(
                        masks["support_switch_pixels"].to(torch.float64).mean()
                    ),
                }
            )
    best_rows = []
    for category in categories:
        selected = [row for row in parameter_rows if row["category"] == category]
        best_rows.append(
            min(
                selected,
                key=lambda row: float(
                    row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"]
                ),
            )
        )
    return {
        "configuration": asdict(corrected),
        "categories": categories,
        "epsilon_sweep": list(config.gradient_epsilons),
        "rows": parameter_rows,
        "best_frozen_rows": best_rows,
        "selection": (
            "Parameters are selected from analytic energy in deep interior/"
            "silhouette bands, total visible energy, a second boundary mode, "
            "and an independent field-gradient curvature proxy."
        ),
    }


def _labels(rows: list[dict[str, object]]) -> list[str]:
    return [f"{row['resolution'][1]}x{row['resolution'][0]}" for row in rows]


def _save_figures(
    directory: Path,
    detailed: list[dict[str, object]],
    captures: list[dict[str, np.ndarray]],
    multiseed_rows: list[dict[str, object]],
    multiseed_summary: list[dict[str, object]],
    gradient: dict[str, object],
    pixel_cases: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    labels = _labels(detailed)
    count_labels = [row[0] for row in COUNT_BINS]
    figure, axis = plt.subplots(figsize=(9, 4))
    for label, result in zip(labels, detailed):
        axis.plot(
            count_labels,
            [row["mean_mc_self_mse"] for row in result["count_bins"]],
            marker="o",
            label=label,
        )
    axis.set_yscale("log")
    axis.set_ylabel("mean MC self-MSE")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v083_mc_error_vs_count.png", dpi=180)
    plt.close(figure)

    for key, filename, xlabel in (
        ("delta_count", "v083_mc_error_vs_delta_count.png", "|Delta N|"),
        ("delta_weight", "v083_mc_error_vs_weight_change.png", "|Delta W|"),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        for axis, label, capture in zip(axes.flat, labels, captures):
            x = capture[key]
            y = capture["mc_error"]
            stride = max(x.size // 30_000, 1)
            axis.scatter(x.reshape(-1)[::stride], y.reshape(-1)[::stride], s=1, alpha=0.15)
            axis.set_yscale("symlog", linthresh=1e-10)
            axis.set_title(label)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("pixel MC MSE")
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, label, capture in zip(axes.flat, labels, captures):
        axis.imshow(capture["support_switch"], cmap="gray")
        axis.set_title(label)
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "v083_support_switch_map.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        labels,
        [row["fraction_self_sse_from_support_switch_pixels"] for row in detailed],
    )
    axis.set_ylabel("fraction of total MC SSE")
    axis.set_title("support-switch pixels")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v083_support_switch_sse_fraction.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(6)
    for label, row in zip(labels, detailed):
        axis.plot(
            x,
            [region["mc_self_mse_per_rgb_scalar"] for region in row["silhouette_regions"]],
            marker="o",
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, REGION_NAMES, rotation=25)
    axis.set_ylabel("MC self-MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v083_silhouette_band_error.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    for label, capture in zip(labels, captures):
        foreground = np.linalg.norm(capture["target"], axis=2) > 0.0
        values = capture["count"][foreground]
        histogram = np.bincount(values, minlength=65).astype(np.float64)
        if histogram.size > 65:
            histogram[64] = histogram[64:].sum()
        histogram /= max(histogram.sum(), 1)
        axis.plot(np.arange(65), histogram[:65], label=label)
    axis.set_yscale("log")
    axis.set_xlabel("contribution count (64 includes truncated tail)")
    axis.set_ylabel("foreground pixel probability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v083_foreground_count_histograms.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    grouped = [
        [float(row["whole_image_mse"]) for row in multiseed_rows if tuple(row["resolution"]) == tuple(result["resolution"])]
        for result in detailed
    ]
    axis.violinplot(grouped, showmeans=True, showmedians=True)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
    axis.set_ylabel("reference-vs-seed MC MSE")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v083_multiseed_mc_distribution.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    for metric, title in (
        ("whole_image_mse", "whole"),
        ("common_support_mse", "common support"),
        ("interior_mse", "interior"),
        ("silhouette_0_8px_mse", "silhouette 0-8px"),
    ):
        axis.plot(
            x,
            [row[metric]["mean"] for row in multiseed_summary],
            marker="o",
            label=title,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("multi-seed mean MC MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "v083_resolution_mc_components.png", dpi=180)
    plt.close(figure)

    gradient_rows = gradient["rows"]
    categories = list(gradient["categories"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for category in categories:
        selected = [row for row in gradient_rows if row["category"] == category]
        eps = [row["epsilon"] for row in selected]
        axes[0].plot(
            eps,
            [row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"] for row in selected],
            marker="o",
            label=category,
        )
        axes[1].plot(
            eps,
            [row["regions"]["all_pixels"]["analytic_vs_full"]["relative_error"] for row in selected],
            marker="o",
            label=category,
        )
    for axis, title in zip(axes, ("analytic vs frozen", "analytic vs full rerender")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("coefficient epsilon")
        axis.set_ylabel("relative error")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "v083_gradient_fd_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    best = gradient["best_frozen_rows"]
    x = np.arange(len(best))
    axis.bar(
        x - 0.18,
        [row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"] for row in best],
        width=0.36,
        label="frozen",
    )
    axis.bar(
        x + 0.18,
        [row["regions"]["all_pixels"]["analytic_vs_full"]["relative_error"] for row in best],
        width=0.36,
        label="full rerender",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, [row["category"] for row in best], rotation=25)
    axis.set_ylabel("analytic relative error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "v083_frozen_vs_full_support_fd.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.axis("off")
    columns = ("category", "resolution", "pixel", "N_A", "N_B", "W_A", "W_B", "local_mc_sse")
    selected_cases = pixel_cases[:30]
    cells = [[str(row[column])[:18] for column in columns] for row in selected_cases]
    table = axis.table(cellText=cells, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.25)
    figure.tight_layout()
    figure.savefig(directory / "v083_pixel_case_studies.png", dpi=180)
    plt.close(figure)

    for label, capture in zip(labels, captures):
        figure, axes = plt.subplots(1, 5, figsize=(16, 3))
        panels = (
            (np.log1p(capture["count"]), "log count", "magma"),
            (capture["support_switch"], "support switch", "gray"),
            (np.log10(capture["mc_error"] + 1e-12), "log10 MC MSE", "inferno"),
            (capture["silhouette_distance"], "signed silhouette distance", "coolwarm"),
            (capture["spatial_gradient"], "spatial RGB gradient", "viridis"),
        )
        for axis, (image, title, cmap) in zip(axes, panels):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(directory / f"v083_spatial_maps_{label}.png", dpi=180)
        plt.close(figure)


def _estimator_audit(config: PixelDiagnosticConfig) -> dict[str, object]:
    return {
        "definitions": {
            "N_p": "sum_i 1[event i has an in-frame strictly-positive cubic weight for p]",
            "W_p": "sum_i w_ip",
            "Q_p": "sum_i w_ip^2",
            "A_p": "sum_i w_ip C_i (0.35 + 0.65 max(0,n_i dot omega_v))",
            "I_p": "1[W_p >= 0.05] clamp(1.5 A_p / max(W_p,1e-30), 0, 1)",
        },
        "equations": [
            "b(x)=2/3-|x|^2+|x|^3/2 for |x|<1",
            "b(x)=(2-|x|)^3/6 for 1<=|x|<2, else 0",
            "w_ip=b(r_p-r_i)b(c_p-c_i)",
            "N_p=sum_i 1[w_ip>0 and p in frame]",
            "W_p=sum_i w_ip; Q_p=sum_i w_ip^2",
            "I_p=1[W_p>=0.05] clamp(1.5 sum_i(w_ip C_i l_i)/max(W_p,1e-30),0,1)",
        ],
        "normalization_factors": {
            "total_emitters": "not an explicit denominator; affects numerator and W_p together",
            "packets_per_emitter": "one direct packet per view; no additional factor",
            "views": "separate images; not a pixel denominator",
            "retained_events": "not an explicit denominator",
            "splat_weights": "present in numerator and W_p",
            "kernel_integral": "unit-sum cardinal cubic on an infinite grid; edge clipping loses mass",
            "pixel_area": "absent",
            "detector_area": "encoded only by coordinate projection extent",
            "accumulated_weight": "W_p is the production denominator",
            "RGB_channels": "independent numerator channels; no image-formation normalization",
        },
        "exact_conditions": {
            "outward": "normal dot direction > 1e-8",
            "visibility_intersection_epsilon": 2e-4,
            "cubic_inner_branch": "abs(distance) < 1",
            "candidate_anchor": "floor(continuous_pixel_coordinate)",
            "candidate_offsets": [-1, 0, 1, 2],
            "write_membership": "in frame and weight > 0.0",
            "final_support": f"W_p >= {config.support_threshold}",
            "mass_denominator_clamp": 1e-30,
            "RGB_clamp": [0.0, 1.0],
        },
    }


def _strip_reference(views: list[PixelView]) -> list[ReferenceView]:
    return [
        ReferenceView(
            view.image,
            view.support.reshape(view.image.shape[:2]).copy(),
        )
        for view in views
    ]


def run_pixel_support_gradient_diagnostic(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    config: PixelDiagnosticConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    experiment = config or PixelDiagnosticConfig()
    if len(experiment.resolutions) != len(experiment.emitter_counts):
        raise ValueError("resolution/emitter-count lengths differ")
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    device = torch.device("cuda")
    base = prepared.base_field.to(device)
    target = prepared.gt_field.to(device)
    cells = sign_changing_cells(base.grid)
    maximum = max(experiment.emitter_counts)
    atlas = nested_fibonacci_atlas(device, (experiment.views,))
    a_surface = _prepare_surface(
        base,
        None,
        maximum,
        experiment.emitter_chunk_size,
        experiment.sobol_seeds[0],
        cells,
    )
    b_surface = _prepare_surface(
        base,
        None,
        maximum,
        experiment.emitter_chunk_size,
        experiment.sobol_seeds[1],
        cells,
    )
    target_surface = _prepare_surface(
        base,
        target,
        maximum,
        experiment.emitter_chunk_size,
        None,
        cells,
    )
    reference_by_resolution: dict[tuple[int, int], list[ReferenceView]] = {}
    target_by_resolution: dict[tuple[int, int], list[np.ndarray]] = {}
    detailed_rows: list[dict[str, object]] = []
    primary_transport: list[dict[str, object]] = []
    primary_multiseed_rows: list[dict[str, object]] = []
    captures: list[dict[str, np.ndarray]] = []
    pixel_cases: list[dict[str, object]] = []
    normalization_base: list[PixelView] | None = None
    boundaries: dict[tuple[int, int], ObservationSphere] = {}
    for resolution, emitter_count in zip(
        experiment.resolutions, experiment.emitter_counts
    ):
        boundary = enclosing_observation_sphere(
            a_surface.reference_points[:emitter_count]
        )
        boundaries[resolution] = boundary
        a_views, a_report = _render_diagnostics(
            base,
            a_surface,
            emitter_count,
            resolution,
            boundary,
            atlas,
            experiment,
            keep_numerator=resolution == (256, 256),
        )
        b_views, b_report = _render_diagnostics(
            base,
            b_surface,
            emitter_count,
            resolution,
            boundary,
            atlas,
            experiment,
        )
        target_views, target_report = _render_diagnostics(
            target,
            target_surface,
            emitter_count,
            resolution,
            boundary,
            atlas,
            experiment,
            geometry="target",
        )
        target_images = [view.image for view in target_views]
        del target_views
        gc.collect()
        detailed, capture, cases = _analyze_pair(
            resolution,
            a_views,
            b_views,
            target_images,
            experiment.capture_view,
        )
        detailed["emitters"] = emitter_count
        detailed["writes_per_pixel"] = float(
            a_report["footprint_writes"]
        ) / (experiment.views * math.prod(resolution))
        detailed["foreground_writes_per_pixel"] = detailed[
            "foreground_sampling"
        ]["target_foreground"]["writes_per_pixel"]
        detailed["independent_visibility_rate_difference"] = abs(
            float(a_report["retained_events"])
            / float(a_report["attempted_packets"])
            - float(b_report["retained_events"])
            / float(b_report["attempted_packets"])
        )
        detailed_rows.append(detailed)
        captures.append(capture)
        pixel_cases.extend(cases)
        reference = _strip_reference(a_views)
        reference_by_resolution[resolution] = reference
        target_by_resolution[resolution] = target_images
        primary_multiseed_rows.append(
            _compare_reference(
                resolution,
                reference,
                b_views,
                target_images,
                experiment.sobol_seeds[1],
            )
        )
        primary_transport.append(
            {
                "resolution": list(resolution),
                "sample_a": a_report,
                "sample_b": b_report,
                "target": target_report,
            }
        )
        if resolution == (256, 256):
            normalization_base = a_views
        else:
            del a_views
        del b_views
        gc.collect()
        _progress("v083_primary_pair", resolution=list(resolution))

    assert normalization_base is not None
    visibility = _controlled_visibility_switching(
        base,
        a_surface,
        atlas,
        boundaries[(256, 256)],
        experiment,
    )

    normalization_counts = [65_536, 131_072, 262_144]
    normalization_renders = [normalization_base]
    for count in normalization_counts[1:]:
        views, _ = _render_diagnostics(
            base,
            a_surface,
            count,
            (256, 256),
            enclosing_observation_sphere(a_surface.reference_points[:count]),
            atlas,
            experiment,
            keep_numerator=True,
        )
        normalization_renders.append(views)
    normalization = _normalization_audit(
        normalization_counts,
        normalization_renders,
        target_by_resolution[(256, 256)],
        experiment.sensor_gain,
    )
    del normalization_renders, normalization_base

    multiseed_rows = list(primary_multiseed_rows)
    del b_surface, target_surface
    _release()
    for seed in experiment.sobol_seeds[2:]:
        surface = _prepare_surface(
            base,
            None,
            maximum,
            experiment.emitter_chunk_size,
            seed,
            cells,
        )
        for resolution, emitter_count in zip(
            experiment.resolutions, experiment.emitter_counts
        ):
            views, _ = _render_diagnostics(
                base,
                surface,
                emitter_count,
                resolution,
                boundaries[resolution],
                atlas,
                experiment,
            )
            multiseed_rows.append(
                _compare_reference(
                    resolution,
                    reference_by_resolution[resolution],
                    views,
                    target_by_resolution[resolution],
                    seed,
                )
            )
            del views
        del surface
        _release()
        _progress("v083_multiseed_complete", seed=seed)
    multiseed_summary = _multiseed_summary(multiseed_rows)
    resolution_trend = _paired_resolution_trend(multiseed_rows)

    grid_rows: list[dict[str, object]] = []
    grid_captures: dict[str, list[dict[str, float]]] = {}
    for method in ("unscrambled_sobol", "pseudorandom_uniform"):
        unit_sequence = None
        if method == "pseudorandom_uniform":
            generator = torch.Generator(device="cpu").manual_seed(
                experiment.pseudorandom_seed
            )
            unit_sequence = torch.rand(
                (maximum, 4), generator=generator, dtype=torch.float32
            )
        surface = _prepare_surface(
            base,
            None,
            maximum,
            experiment.emitter_chunk_size,
            None,
            cells,
            unit_sequence=unit_sequence,
        )
        method_captures: list[dict[str, float]] = []
        for resolution, emitter_count in zip(
            experiment.resolutions, experiment.emitter_counts
        ):
            views, _ = _render_diagnostics(
                base,
                surface,
                emitter_count,
                resolution,
                boundaries[resolution],
                atlas,
                experiment,
            )
            row = _compare_reference(
                resolution,
                reference_by_resolution[resolution],
                views,
                target_by_resolution[resolution],
                method,
            )
            row["sampling_method"] = method
            grid_rows.append(row)
            capture_view = min(experiment.capture_view, experiment.views - 1)
            error = np.square(
                reference_by_resolution[resolution][capture_view].image.astype(np.float64)
                - views[capture_view].image
            ).mean(2)
            structure = _spatial_structure(error)
            structure["resolution"] = list(resolution)
            method_captures.append(structure)
            del views
        grid_captures[method] = method_captures
        del surface, unit_sequence
        _release()
        _progress("v083_grid_control_complete", method=method)

    stencil = _stencil_continuity()
    del a_surface
    _release()
    gradient = _gradient_stability(prepared, experiment)

    high_count_mse = [
        float(row["count_bins"][-1]["mean_mc_self_mse"])
        for row in detailed_rows
    ]
    medium_count_mse = [
        float(row["count_bins"][3]["mean_mc_self_mse"])
        for row in detailed_rows
    ]
    high_count_stable = statistics.median(high_count_mse) < statistics.median(
        medium_count_mse
    )
    count_correlations = [
        abs(float(row["correlations"]["delta_count"]["common_support"]["spearman"]))
        for row in detailed_rows
    ]
    weight_correlations = [
        abs(float(row["correlations"]["delta_weight"]["common_support"]["spearman"]))
        for row in detailed_rows
    ]
    switch_sse_fractions = [
        float(row["fraction_self_sse_from_support_switch_pixels"])
        for row in detailed_rows
    ]
    silhouette_sse_fractions = [
        sum(
            float(region["fraction_total_mc_self_sse"])
            for region in row["silhouette_regions"][1:5]
        )
        for row in detailed_rows
    ]
    visibility_switch_fraction = max(
        float(row["visible_occluded_switch_fraction"])
        for row in visibility["rows"]
        if float(row["direction_perturbation"]) <= 1e-3
    )
    low_summary = next(
        row for row in multiseed_summary if tuple(row["resolution"]) == (256, 256)
    )
    high_summary = next(
        row for row in multiseed_summary if tuple(row["resolution"]) == (1080, 1920)
    )
    interior_ratio = float(high_summary["interior_mse"]["mean"]) / max(
        float(low_summary["interior_mse"]["mean"]), 1e-30
    )
    foreground_ratio = float(high_summary["foreground_mse"]["mean"]) / max(
        float(low_summary["foreground_mse"]["mean"]), 1e-30
    )
    best_gradient = gradient["best_frozen_rows"]
    frozen_errors = [
        float(row["regions"]["all_pixels"]["analytic_vs_frozen"]["relative_error"])
        for row in best_gradient
    ]
    full_errors = [
        float(row["regions"]["all_pixels"]["analytic_vs_full"]["relative_error"])
        for row in best_gradient
    ]
    silhouette_full_errors = [
        float(row["regions"]["silhouette_0_2px"]["analytic_vs_full"]["relative_error"])
        for row in best_gradient
    ]
    interior_full_errors = [
        float(row["regions"]["interior_gt8px"]["analytic_vs_full"]["relative_error"])
        for row in best_gradient
    ]
    brightness = [
        float(row["foreground_mean_intensity"]) for row in normalization["rows"]
    ]
    brightness_relative_range = (max(brightness) - min(brightness)) / max(
        statistics.mean(brightness), 1e-30
    )
    scrambled_high_values = np.asarray(
        [
            float(row["whole_image_mse"])
            for row in multiseed_rows
            if tuple(row["resolution"]) == (1080, 1920)
        ]
    )
    high_grid = [
        row for row in grid_rows if tuple(row["resolution"]) == (1080, 1920)
    ]
    grid_z_scores = {
        str(row["sampling_method"]): abs(
            float(row["whole_image_mse"]) - float(scrambled_high_values.mean())
        )
        / max(float(scrambled_high_values.std(ddof=1)), 1e-30)
        for row in high_grid
    }

    verdicts = {
        "PIXEL_ESTIMATOR_NORMALIZATION_CORRECT": brightness_relative_range <= 0.05,
        "MULTIPLE_EMITTER_ACCUMULATION_UNSTABLE": not high_count_stable,
        "HIGH_COUNT_PIXELS_MORE_STABLE": high_count_stable,
        "COUNT_VARIATION_DRIVES_MC_ERROR": statistics.median(count_correlations) >= 0.30,
        "WEIGHT_VARIATION_DRIVES_MC_ERROR": statistics.median(weight_correlations) >= 0.30,
        "RANDOM_DENOMINATOR_DRIVES_MC_ERROR": (
            statistics.median(weight_correlations) >= 0.30
            and statistics.median(weight_correlations)
            > statistics.median(count_correlations)
        ),
        "COMPACT_SUPPORT_SWITCHING_DETECTED": max(
            float(row["support_switch_pixel_fraction"]) for row in detailed_rows
        )
        > 0.0,
        "SUPPORT_SWITCHING_DOMINATES_MC_SSE": statistics.median(
            switch_sse_fractions
        )
        >= 0.50,
        "VISIBILITY_SWITCHING_DETECTED": visibility_switch_fraction > 0.0,
        "SILHOUETTE_REGION_DOMINATES_MC_SSE": statistics.median(
            silhouette_sse_fractions
        )
        >= 0.50,
        "INTERIOR_MC_STABLE": interior_ratio <= 1.50,
        "FOREGROUND_MC_STABLE": foreground_ratio <= 1.50,
        "SOBOL_GRID_RESONANCE_DETECTED": max(grid_z_scores.values()) >= 3.0,
        "MC_NOISE_RESOLUTION_TREND_SIGNIFICANT": bool(
            resolution_trend["significant_at_0_05"]
        ),
        "ANALYTIC_JACOBIAN_MATCHES_FROZEN_SUPPORT": max(frozen_errors) < 0.08,
        "ANALYTIC_JACOBIAN_MATCHES_FULL_RERENDER": max(full_errors) < 0.08,
        "GRADIENT_INSTABILITY_LOCALIZED_TO_BOUNDARIES": statistics.median(
            silhouette_full_errors
        )
        > 2.0 * statistics.median(interior_full_errors),
        "MORE_SAMPLES_STILL_REQUIRED": not (
            interior_ratio <= 1.50 and foreground_ratio <= 1.50
        ),
        "ESTIMATOR_CHANGE_REQUIRED": statistics.median(switch_sse_fractions) >= 0.50,
        "HIGH_RES_BIRTH_READY_TO_TEST": False,
    }
    verdict_evidence = {
        "brightness_relative_range_N_2N_4N": brightness_relative_range,
        "high_count_mse_by_resolution": high_count_mse,
        "medium_count_mse_by_resolution": medium_count_mse,
        "median_common_support_abs_spearman_deltaN": statistics.median(count_correlations),
        "median_common_support_abs_spearman_deltaW": statistics.median(weight_correlations),
        "support_switch_sse_fractions": switch_sse_fractions,
        "silhouette_0_8px_sse_fractions": silhouette_sse_fractions,
        "maximum_small_perturbation_visibility_switch_fraction": visibility_switch_fraction,
        "fullhd_to_low_interior_multiseed_mse_ratio": interior_ratio,
        "fullhd_to_low_foreground_multiseed_mse_ratio": foreground_ratio,
        "grid_control_fullhd_z_scores": grid_z_scores,
        "resolution_trend": resolution_trend,
        "best_analytic_vs_frozen_relative_errors": frozen_errors,
        "best_analytic_vs_full_relative_errors": full_errors,
        "best_silhouette_full_relative_errors": silhouette_full_errors,
        "best_interior_full_relative_errors": interior_full_errors,
        "thresholds": {
            "correlation_driver": 0.30,
            "dominant_sse_fraction": 0.50,
            "interior_foreground_stability_ratio": 1.50,
            "grid_outlier_z": 3.0,
            "jacobian_relative_error": 0.08,
            "boundary_localization_ratio": 2.0,
            "brightness_relative_range": 0.05,
        },
    }
    prerequisites = {
        "MC_SELF_NOISE_STATISTICALLY_CHARACTERIZED": len(experiment.sobol_seeds) >= 8,
        "INTERIOR_MC_STABLE": verdicts["INTERIOR_MC_STABLE"],
        "SUPPORT_SWITCH_ROOT_CAUSE_UNDERSTOOD": verdicts[
            "SUPPORT_SWITCHING_DOMINATES_MC_SSE"
        ],
        "GRADIENT_STABILITY_ACCEPTABLE": verdicts[
            "ANALYTIC_JACOBIAN_MATCHES_FULL_RERENDER"
        ],
        "NORMALIZATION_VALIDATED": verdicts[
            "PIXEL_ESTIMATOR_NORMALIZATION_CORRECT"
        ],
    }
    verdicts["HIGH_RES_BIRTH_READY_TO_TEST"] = all(prerequisites.values())
    root_cause = {
        "CASE_1_MULTIPLE_ACCUMULATION": verdicts[
            "MULTIPLE_EMITTER_ACCUMULATION_UNSTABLE"
        ],
        "CASE_2_COUNT_VARIATION": verdicts["COUNT_VARIATION_DRIVES_MC_ERROR"],
        "CASE_3_RANDOM_WEIGHT_DENOMINATOR": verdicts[
            "RANDOM_DENOMINATOR_DRIVES_MC_ERROR"
        ],
        "CASE_4_COMPACT_SUPPORT_SWITCHING": verdicts[
            "SUPPORT_SWITCHING_DOMINATES_MC_SSE"
        ],
        "CASE_5_VISIBILITY_SILHOUETTE_SWITCHING": (
            verdicts["VISIBILITY_SWITCHING_DETECTED"]
            and verdicts["SILHOUETTE_REGION_DOMINATES_MC_SSE"]
        ),
        "CASE_6_SOBOL_RASTER_RESONANCE": verdicts[
            "SOBOL_GRID_RESONANCE_DETECTED"
        ],
        "interpretation": (
            "Booleans are thresholded summaries; detailed pixel, band, seed, "
            "visibility, and finite-difference rows retain the continuous evidence."
        ),
    }

    _save_figures(
        figure_directory,
        detailed_rows,
        captures,
        multiseed_rows,
        multiseed_summary,
        gradient,
        pixel_cases,
    )
    runtime = time.perf_counter() - started
    recorded_allocated_mib = max(
        torch.cuda.max_memory_allocated(device) / 2**20,
        *(
            float(sample["peak_allocated_mib"])
            for group in primary_transport
            for sample in (group["sample_a"], group["sample_b"], group["target"])
        ),
    )
    recorded_reserved_mib = max(
        torch.cuda.max_memory_reserved(device) / 2**20,
        *(
            float(sample["peak_reserved_mib"])
            for group in primary_transport
            for sample in (group["sample_a"], group["sample_b"], group["target"])
        ),
    )
    sobol_ranges = [
        {
            "resolution": list(resolution),
            "emitter_count": count,
            "surface_sobol_dimensions": 4,
            "start_inclusive": 0,
            "stop_exclusive": count,
            "policy": "nested global Sobol prefix; chunking preserves global indices",
        }
        for resolution, count in zip(
            experiment.resolutions, experiment.emitter_counts
        )
    ]
    report: dict[str, object] = {
        "version": "0.8.3",
        "question": "Why is independent-sample MC self-noise resolution dependent after matched transport density?",
        "configuration": {
            **asdict(experiment),
            "sobol_ranges": sobol_ranges,
        },
        "environment": cuda_environment(),
        "commands": {
            "formal": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --pixel-support-diagnostic --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply",
            "verification": "PYTHONPATH=src conda run --no-capture-output -n test python demo.py --scene sphere --verify",
        },
        "exact_pixel_estimator": _estimator_audit(experiment),
        "accumulation_statistics": detailed_rows,
        "normalization_audit": normalization,
        "support_switching": {
            "primary_seed_pair": [experiment.sobol_seeds[0], experiment.sobol_seeds[1]],
            "rows": detailed_rows,
            "stencil_continuity": stencil,
            "event_threshold_statistics": primary_transport,
        },
        "visibility_switching": visibility,
        "foreground_interior_silhouette": [
            {
                "resolution": row["resolution"],
                "foreground_sampling": row["foreground_sampling"],
                "silhouette_regions": row["silhouette_regions"],
            }
            for row in detailed_rows
        ],
        "multi_seed_mc": {
            "design": "seed 101 reference versus seven independent scrambled Sobol seeds",
            "sample_sets": len(experiment.sobol_seeds),
            "seeds": list(experiment.sobol_seeds),
            "sobol_ranges": sobol_ranges,
            "rows": multiseed_rows,
            "summary": multiseed_summary,
            "resolution_trend": resolution_trend,
        },
        "sobol_grid_test": {
            "rows": grid_rows,
            "spatial_structure": grid_captures,
            "pseudorandom_seed": experiment.pseudorandom_seed,
        },
        "gradient_stability": gradient,
        "frozen_support_vs_full_rerender_fd": gradient,
        "pixel_case_studies": pixel_cases,
        "root_cause": root_cause,
        "fix": {
            "implemented": False,
            "reason": "This stage isolates causes; no estimator or renderer behavior is changed before the complete decomposition is reviewed.",
        },
        "remaining_unknowns": [
            "The multi-seed design is reference-versus-independent rather than all 28 seed pairs.",
            "Full-resolution geometry Jacobians remain infeasible; gradient tests use a controlled 256-square, four-view, 16,384-emitter case.",
            "Independent sample identities cannot define emitter-ID Jaccard overlap, so support/count/weight proxies are used.",
        ],
        "birth_prerequisites": prerequisites,
        "verdicts": verdicts,
        "verdict_evidence": verdict_evidence,
        "runtime_seconds": runtime,
        "peak_allocated_mib": recorded_allocated_mib,
        "peak_reserved_mib": recorded_reserved_mib,
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "artifacts": {
            "json": str(artifact_directory / "v083_pixel_support_gradient_diagnostic.json"),
            "csv": str(artifact_directory / "v083_pixel_support_gradient_diagnostic.csv"),
            "figures": str(figure_directory / "v083_*.png"),
        },
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v083_pixel_support_gradient_diagnostic.json"
    json_path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n")
    csv_rows: list[dict[str, object]] = []
    for section, rows in (
        ("accumulation", detailed_rows),
        ("multiseed", multiseed_rows),
        ("multiseed_summary", multiseed_summary),
        ("grid", grid_rows),
        ("gradient", gradient["rows"]),
        ("pixel_cases", pixel_cases),
    ):
        csv_rows.extend({"section": section, **row} for row in rows)
    _write_csv(
        artifact_directory / "v083_pixel_support_gradient_diagnostic.csv",
        csv_rows,
    )
    return report
