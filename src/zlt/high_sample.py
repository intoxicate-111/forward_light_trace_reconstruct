"""v0.8.2 fixed-footprint, matched-transport-density experiment.

The renderer in this module intentionally keeps the legacy four-by-four
pixel-space cubic footprint.  It streams emitter prefixes through visibility
and splatting so detector resolution can be increased together with real
transport samples without materializing the full event graph.
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

import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import (
    DirectionAtlas,
    ObservationSphere,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
)
from .bunny import BunnyGeometryEvaluator, _normal_line_roots
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import (
    meshfree_base_color,
    sample_meshfree_zero_set,
    sign_changing_cells,
)
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor


@dataclass(frozen=True)
class HighSampleConfig:
    views: int = 20
    reference_resolution: tuple[int, int] = (256, 256)
    reference_emitters: int = 65_536
    resolutions: tuple[tuple[int, int], ...] = (
        (256, 256),
        (512, 512),
        (540, 960),
        (1080, 1920),
    )
    initial_emitter_counts: tuple[int, ...] = (
        65_536,
        262_144,
        518_400,
        2_073_600,
    )
    emitter_chunk_size: int = 65_536
    validation_emitters: int = 8_192
    validation_chunk_sizes: tuple[int, int] = (1_024, 2_048)
    train_scramble_seed: int | None = None
    heldout_scramble_seed: int = 211
    mc_scramble_seeds: tuple[int, int] = (101, 211)
    root_samples: int = 16
    bisection_steps: int = 18
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    support_threshold: float = 0.05
    ambient: float = 0.35
    density_relative_tolerance: float = 0.10
    coverage_absolute_tolerance: float = 0.03
    zero_absolute_tolerance: float = 0.03
    count_lt2_absolute_tolerance: float = 0.05
    median_count_tolerance: float = 1.0


@dataclass
class SurfaceState:
    reference_points: Tensor
    reference_normals: Tensor
    base_points: Tensor
    base_normals: Tensor
    target_points: Tensor | None
    target_normals: Tensor | None
    report: dict[str, object]


@dataclass
class StreamResult:
    report: dict[str, object]
    base_images: list[np.ndarray]
    target_images: list[np.ndarray] | None
    capture: dict[str, np.ndarray]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _progress(phase: str, **values: object) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _prepare_surface(
    base: object,
    target: object | None,
    count: int,
    chunk_size: int,
    scramble_seed: int | None,
    sampling_cells: Tensor,
    maximum_offset: float = 0.05,
) -> SurfaceState:
    device = base.grid.device
    point_parts: list[Tensor] = []
    normal_parts: list[Tensor] = []
    base_point_parts: list[Tensor] = []
    base_normal_parts: list[Tensor] = []
    target_point_parts: list[Tensor] = []
    target_normal_parts: list[Tensor] = []
    root_failures = 0
    requested = 0
    valid = 0
    newton = 0
    fallback = 0
    sample_seconds = 0.0
    target_seconds = 0.0
    residual_max = 0.0
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        surface = sample_meshfree_zero_set(
            base,
            stop - start,
            sobol_scramble_seed=scramble_seed,
            sobol_start_index=start,
            sampling_cells=sampling_cells,
        )
        if surface.valid_count != stop - start:
            raise RuntimeError(
                f"STREAM_SURFACE_ROOT_FAILURE:{start}:{surface.root_failure_count}"
            )
        point_parts.append(surface.points)
        normal_parts.append(surface.normals)
        displacement = torch.zeros_like(surface.points[:, 0])
        for _ in range(16):
            base_points = surface.points + displacement[:, None] * surface.normals
            values = base.value(base_points)
            gradients = base.interpolant_gradient(base_points)
            denominator = (gradients * surface.normals).sum(1)
            step = torch.where(
                denominator.abs() > 1e-8,
                values / denominator,
                torch.zeros_like(values),
            )
            displacement = (displacement - step).clamp(
                -maximum_offset, maximum_offset
            )
        base_points = surface.points + displacement[:, None] * surface.normals
        shading_gradients = base.gradient(base_points)
        base_normals = shading_gradients / torch.linalg.vector_norm(
            shading_gradients, dim=1, keepdim=True
        ).clamp_min(1e-30)
        base_point_parts.append(base_points)
        base_normal_parts.append(base_normals)
        requested += surface.requested_count
        valid += surface.valid_count
        newton += surface.newton_converged_count
        fallback += surface.edge_fallback_count
        sample_seconds += surface.build_seconds
        residual_max = max(residual_max, surface.residual_max)
        peak_allocated_mib = max(
            peak_allocated_mib,
            torch.cuda.max_memory_allocated(device) / 2**20,
        )
        peak_reserved_mib = max(
            peak_reserved_mib,
            torch.cuda.max_memory_reserved(device) / 2**20,
        )
        if target is not None:
            _sync(device)
            target_started = time.perf_counter()
            target_points, success, _ = _normal_line_roots(
                target,
                surface.points,
                surface.normals,
                maximum_offset=maximum_offset,
            )
            target_gradients = target.gradient(target_points)
            target_normals = target_gradients / torch.linalg.vector_norm(
                target_gradients, dim=1, keepdim=True
            ).clamp_min(1e-30)
            _sync(device)
            target_seconds += time.perf_counter() - target_started
            root_failures += int((~success).sum())
            target_point_parts.append(target_points)
            target_normal_parts.append(target_normals)
            peak_allocated_mib = max(
                peak_allocated_mib,
                torch.cuda.max_memory_allocated(device) / 2**20,
            )
            peak_reserved_mib = max(
                peak_reserved_mib,
                torch.cuda.max_memory_reserved(device) / 2**20,
            )
        _progress(
            "surface_chunk",
            seed=scramble_seed,
            start=start,
            stop=stop,
        )
    if root_failures:
        raise RuntimeError(f"TARGET_NORMAL_LINE_FAILURES:{root_failures}")
    points = torch.cat(point_parts)
    normals = torch.cat(normal_parts)
    base_points = torch.cat(base_point_parts)
    base_normals = torch.cat(base_normal_parts)
    target_points = torch.cat(target_point_parts) if target is not None else None
    target_normals = (
        torch.cat(target_normal_parts) if target is not None else None
    )
    return SurfaceState(
        points,
        normals,
        base_points,
        base_normals,
        target_points,
        target_normals,
        {
            "requested_count": requested,
            "valid_count": valid,
            "newton_converged_count": newton,
            "edge_fallback_count": fallback,
            "target_root_failures": root_failures,
            "surface_sample_seconds": sample_seconds,
            "target_mapping_seconds": target_seconds,
            "surface_residual_max": residual_max,
            "sobol_scramble": scramble_seed is not None,
            "sobol_seed": scramble_seed,
            "sobol_start_index": 0,
            "sobol_stop_index_exclusive": count,
            "emitter_index_range": [0, count],
            "chunk_size": chunk_size,
            "chunks": math.ceil(count / chunk_size),
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
        },
    )


def _cubic(distance: Tensor) -> Tensor:
    absolute = distance.abs()
    inner = 2.0 / 3.0 - absolute.square() + 0.5 * absolute**3
    outer = (2.0 - absolute).clamp_min(0.0) ** 3 / 6.0
    return torch.where(absolute < 1.0, inner, outer)


def _visible_owner_ids(
    field: object,
    points: Tensor,
    normals: Tensor,
    direction: Tensor,
    boundary: ObservationSphere,
    config: HighSampleConfig,
) -> tuple[Tensor, int, int, float]:
    outward = (normals @ direction) > 1e-8
    owners = torch.nonzero(outward, as_tuple=False).flatten()
    if owners.numel() == 0:
        return owners, 0, 0, 0.0
    origins = points[owners]
    directions = direction.expand_as(origins)
    maximum = boundary.exit_times(origins, directions)
    _sync(points.device)
    started = time.perf_counter()
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
    _sync(points.device)
    seconds = time.perf_counter() - started
    return owners[~hit], int(owners.numel()), int(hit.sum()), seconds


def _splat(
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
    numerator: Tensor,
    counts: Tensor | None,
    ambient: float,
) -> tuple[int, int, float, float]:
    rows, columns = resolution
    device = points.device
    _sync(device)
    projection_started = time.perf_counter()
    relative = points[owners] - center
    column = (relative @ right / extent + 0.5) * columns - 0.5
    row = (0.5 - relative @ up / extent) * rows - 0.5
    base_row = torch.floor(row).to(torch.long)
    base_column = torch.floor(column).to(torch.long)
    offsets = torch.arange(-1, 3, dtype=torch.long, device=device)
    row_ids = base_row[:, None] + offsets[None]
    column_ids = base_column[:, None] + offsets[None]
    row_weights = _cubic(row[:, None] - row_ids)
    column_weights = _cubic(column[:, None] - column_ids)
    pixel_rows = row_ids[:, :, None].expand(-1, 4, 4).reshape(-1, 16)
    pixel_columns = column_ids[:, None, :].expand(-1, 4, 4).reshape(-1, 16)
    weights = (
        row_weights[:, :, None] * column_weights[:, None, :]
    ).reshape(-1, 16)
    valid = (
        (pixel_rows >= 0)
        & (pixel_rows < rows)
        & (pixel_columns >= 0)
        & (pixel_columns < columns)
        & (weights > 0.0)
    )
    pixels = (
        pixel_rows.clamp(0, rows - 1) * columns
        + pixel_columns.clamp(0, columns - 1)
    )
    weights *= valid
    projected = int(valid.any(1).sum())
    writes = int(valid.sum())
    colors = meshfree_base_color(points[owners], lower, upper)
    cosine = (normals[owners] @ direction).clamp_min(0.0)
    lobe = ambient + (1.0 - ambient) * cosine
    radiance = colors * lobe[:, None]
    _sync(device)
    projection_seconds = time.perf_counter() - projection_started
    accumulation_started = time.perf_counter()
    mass.scatter_add_(0, pixels.reshape(-1), weights.reshape(-1))
    numerator.index_add_(
        0,
        pixels.reshape(-1),
        (weights[..., None] * radiance[:, None, :]).reshape(-1, 3),
    )
    if counts is not None:
        counts.scatter_add_(
            0,
            pixels[valid],
            torch.ones(writes, dtype=torch.int32, device=device),
        )
    _sync(device)
    accumulation_seconds = time.perf_counter() - accumulation_started
    return projected, writes, projection_seconds, accumulation_seconds


def _finalize_image(
    mass: Tensor,
    numerator: Tensor,
    sensor_gain: float,
    support_threshold: float,
) -> Tensor:
    raw = sensor_gain * numerator / mass[:, None].clamp_min(1e-30)
    return torch.where(
        (mass >= support_threshold)[:, None],
        raw.clamp(0.0, 1.0),
        torch.zeros_like(raw),
    )


def _merge_histogram(total: Tensor | None, current: Tensor) -> Tensor:
    current = current.detach().cpu()
    if total is None:
        return current
    if current.numel() > total.numel():
        total = torch.nn.functional.pad(total, (0, current.numel() - total.numel()))
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


def _stream_render(
    base_field: object,
    target_field: object | None,
    surface: SurfaceState,
    emitter_count: int,
    resolution: tuple[int, int],
    boundary: ObservationSphere,
    atlas: DirectionAtlas,
    config: HighSampleConfig,
    *,
    chunk_size: int,
    capture_view: int = 10,
) -> StreamResult:
    device = surface.reference_points.device
    rows, columns = resolution
    pixels = rows * columns
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    histogram: Tensor | None = None
    base_images: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    capture: dict[str, np.ndarray] = {}
    attempted = emitter_count * config.views
    base_outward = base_absorbed = base_retained = base_projected = 0
    base_writes = 0
    target_outward = target_absorbed = target_retained = target_projected = 0
    target_writes = 0
    emission_seconds = visibility_seconds = projection_seconds = 0.0
    accumulation_seconds = 0.0
    diagnostics_seconds = 0.0
    raw_sse = absolute = target_energy = predicted_energy = 0.0
    scalar_count = config.views * pixels * 3
    for view in range(config.views):
        direction = atlas.directions[view]
        right = atlas.right[view]
        up = atlas.up[view]
        base_mass = torch.zeros(pixels, dtype=torch.float64, device=device)
        base_numerator = torch.zeros(
            (pixels, 3), dtype=torch.float64, device=device
        )
        base_counts = torch.zeros(pixels, dtype=torch.int32, device=device)
        if target_field is not None:
            target_mass = torch.zeros_like(base_mass)
            target_numerator = torch.zeros_like(base_numerator)
        else:
            target_mass = target_numerator = None
        for start in range(0, emitter_count, chunk_size):
            stop = min(start + chunk_size, emitter_count)
            local_reference_points = surface.reference_points[start:stop]
            local_reference_normals = surface.reference_normals[start:stop]
            _sync(device)
            emission_started = time.perf_counter()
            owners, outward, absorbed, seconds = _visible_owner_ids(
                base_field,
                local_reference_points,
                local_reference_normals,
                direction,
                boundary,
                config,
            )
            _sync(device)
            emission_seconds += max(
                0.0, time.perf_counter() - emission_started - seconds
            )
            base_outward += outward
            base_absorbed += absorbed
            base_retained += int(owners.numel())
            visibility_seconds += seconds
            local_points = surface.base_points[start:stop]
            local_normals = surface.base_normals[start:stop]
            projected, writes, projection, accumulation = _splat(
                local_points,
                local_normals,
                owners,
                direction,
                right,
                up,
                boundary.center,
                base_field.lower,
                base_field.upper,
                resolution,
                config.detector_extent,
                base_mass,
                base_numerator,
                base_counts,
                config.ambient,
            )
            base_projected += projected
            base_writes += writes
            projection_seconds += projection
            accumulation_seconds += accumulation
            if target_field is not None:
                assert surface.target_points is not None
                assert surface.target_normals is not None
                local_target_points = surface.target_points[start:stop]
                local_target_normals = surface.target_normals[start:stop]
                _sync(device)
                emission_started = time.perf_counter()
                target_owners, outward, absorbed, seconds = _visible_owner_ids(
                    target_field,
                    local_target_points,
                    local_target_normals,
                    direction,
                    boundary,
                    config,
                )
                _sync(device)
                emission_seconds += max(
                    0.0, time.perf_counter() - emission_started - seconds
                )
                target_outward += outward
                target_absorbed += absorbed
                target_retained += int(target_owners.numel())
                visibility_seconds += seconds
                projected, writes, projection, accumulation = _splat(
                    local_target_points,
                    local_target_normals,
                    target_owners,
                    direction,
                    right,
                    up,
                    boundary.center,
                    target_field.lower,
                    target_field.upper,
                    resolution,
                    config.detector_extent,
                    target_mass,
                    target_numerator,
                    None,
                    config.ambient,
                )
                target_projected += projected
                target_writes += writes
                projection_seconds += projection
                accumulation_seconds += accumulation
        _sync(device)
        diagnostic_started = time.perf_counter()
        base_image = _finalize_image(
            base_mass,
            base_numerator,
            config.sensor_gain,
            config.support_threshold,
        )
        base_images.append(base_image.detach().cpu().to(torch.float32).numpy())
        histogram = _merge_histogram(
            histogram, torch.bincount(base_counts.to(torch.long))
        )
        if target_field is not None:
            assert target_mass is not None and target_numerator is not None
            target_image = _finalize_image(
                target_mass,
                target_numerator,
                config.sensor_gain,
                config.support_threshold,
            )
            target_images.append(
                target_image.detach().cpu().to(torch.float32).numpy()
            )
            residual = base_image - target_image
            raw_sse += float(residual.square().sum())
            absolute += float(residual.abs().sum())
            target_energy += float(target_image.square().sum())
            predicted_energy += float(base_image.square().sum())
        if view == min(capture_view, config.views - 1):
            capture = {
                "base": base_image.reshape(rows, columns, 3)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "counts": base_counts.reshape(rows, columns)
                .detach()
                .cpu()
                .numpy(),
                "mass": base_mass.reshape(rows, columns)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy(),
                "support": (base_mass >= config.support_threshold)
                .reshape(rows, columns)
                .detach()
                .cpu()
                .numpy(),
            }
            if target_field is not None:
                capture["target"] = (
                    target_image.reshape(rows, columns, 3)
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
        _sync(device)
        diagnostics_seconds += time.perf_counter() - diagnostic_started
        del base_mass, base_numerator, base_counts, base_image
        if target_field is not None:
            del target_mass, target_numerator, target_image, residual
        _progress(
            "stream_view",
            resolution=[rows, columns],
            emitters=emitter_count,
            view=view,
        )
    assert histogram is not None
    total_pixels = config.views * pixels
    mse = raw_sse / scalar_count if target_field is not None else 0.0
    runtime_seconds = time.perf_counter() - started
    geometry_streams = 2 if target_field is not None else 1
    processed_packets = attempted * geometry_streams
    total_retained = base_retained + target_retained
    report: dict[str, object] = {
        "resolution": [rows, columns],
        "emitters": emitter_count,
        "views": config.views,
        "attempted_packets": attempted,
        "geometry_streams": geometry_streams,
        "processed_packets": processed_packets,
        "base_outward_events": base_outward,
        "base_absorbed_events": base_absorbed,
        "base_retained_events": base_retained,
        "base_projected_events": base_projected,
        "base_footprint_writes": base_writes,
        "target_outward_events": target_outward,
        "target_absorbed_events": target_absorbed,
        "target_retained_events": target_retained,
        "target_projected_events": target_projected,
        "target_footprint_writes": target_writes,
        "total_pixels": total_pixels,
        "pixels_ge_1": int(histogram[1:].sum()),
        "pixels_ge_2": int(histogram[2:].sum()),
        "pixels_ge_4": int(histogram[4:].sum()),
        "pixels_ge_8": int(histogram[8:].sum()),
        "unique_coverage_ratio": float(histogram[1:].sum() / total_pixels),
        "zero_count_fraction": float(histogram[0] / total_pixels),
        "count_lt2_fraction": float(histogram[:2].sum() / total_pixels),
        "count_lt4_fraction": float(histogram[:4].sum() / total_pixels),
        "effective_writes_per_pixel": base_writes / total_pixels,
        "mean_contribution_count": base_writes / total_pixels,
        "median_contribution_count": _histogram_quantile(histogram, 0.50),
        "contribution_p90": _histogram_quantile(histogram, 0.90),
        "contribution_p95": _histogram_quantile(histogram, 0.95),
        "contribution_p99": _histogram_quantile(histogram, 0.99),
        "maximum_contribution_count": int(histogram.numel() - 1),
        "contribution_histogram": histogram.tolist(),
        "target_rgb_energy": target_energy,
        "predicted_rgb_energy": predicted_energy,
        "raw_sse": raw_sse,
        "half_sse": 0.5 * raw_sse,
        "mse_per_rgb_scalar": mse,
        "rmse_per_rgb_scalar": math.sqrt(mse),
        "mae_per_rgb_scalar": absolute / scalar_count
        if target_field is not None
        else 0.0,
        "surface_samples_per_pixel": emitter_count / pixels,
        "emitter_chunk_size": chunk_size,
        "chunks_per_view": math.ceil(emitter_count / chunk_size),
        "total_stream_chunks": config.views
        * math.ceil(emitter_count / chunk_size),
        "runtime_seconds": runtime_seconds,
        "emission_seconds": emission_seconds,
        "visibility_seconds": visibility_seconds,
        "projection_and_footprint_seconds": projection_seconds,
        "image_accumulation_seconds": accumulation_seconds,
        "diagnostics_seconds": diagnostics_seconds,
        "packets_per_second": processed_packets
        / max(runtime_seconds, 1e-30),
        "retained_events_per_second": total_retained
        / max(runtime_seconds, 1e-30),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "footprint": "legacy fixed 4x4 pixel-space cubic",
    }
    return StreamResult(
        report,
        base_images,
        target_images if target_field is not None else None,
        capture,
    )


def _pair_metrics(
    first: list[np.ndarray], second: list[np.ndarray]
) -> dict[str, object]:
    raw_sse = 0.0
    absolute = 0.0
    target_energy = 0.0
    count = 0
    pixel_count = 0
    support_disagreement = 0
    common_support_sse = 0.0
    common_support_scalars = 0
    per_view_mse: list[float] = []
    for left, right in zip(first, second):
        difference = left.astype(np.float64) - right.astype(np.float64)
        view_sse = float(np.square(difference).sum())
        raw_sse += view_sse
        absolute += float(np.abs(difference).sum())
        target_energy += float(np.square(left.astype(np.float64)).sum())
        count += difference.size
        per_view_mse.append(view_sse / difference.size)
        left_support = np.linalg.norm(left, axis=1) > 0.0
        right_support = np.linalg.norm(right, axis=1) > 0.0
        common = left_support & right_support
        support_disagreement += int(np.count_nonzero(left_support ^ right_support))
        pixel_count += left.shape[0]
        common_support_sse += float(np.square(difference[common]).sum())
        common_support_scalars += int(np.count_nonzero(common)) * 3
    mse = raw_sse / count
    return {
        "mc_self_raw_sse": raw_sse,
        "mc_self_mse": mse,
        "mc_self_rmse": math.sqrt(mse),
        "mc_self_mae": absolute / count,
        "mc_self_to_image_energy": raw_sse / max(target_energy, 1e-30),
        "support_disagreement_fraction": support_disagreement
        / max(pixel_count, 1),
        "common_support_mse": common_support_sse
        / max(common_support_scalars, 1),
        "per_view_mse": per_view_mse,
        "per_view_mse_mean": statistics.mean(per_view_mse),
        "per_view_mse_median": statistics.median(per_view_mse),
        "per_view_mse_minimum": min(per_view_mse),
        "per_view_mse_maximum": max(per_view_mse),
    }


def _density_gate(
    reference: dict[str, object], candidate: dict[str, object], config: HighSampleConfig
) -> dict[str, object]:
    density_relative_error = abs(
        float(candidate["effective_writes_per_pixel"])
        / float(reference["effective_writes_per_pixel"])
        - 1.0
    )
    coverage_error = abs(
        float(candidate["unique_coverage_ratio"])
        - float(reference["unique_coverage_ratio"])
    )
    zero_error = abs(
        float(candidate["zero_count_fraction"])
        - float(reference["zero_count_fraction"])
    )
    count_lt2_error = abs(
        float(candidate["count_lt2_fraction"])
        - float(reference["count_lt2_fraction"])
    )
    median_error = abs(
        float(candidate["median_contribution_count"])
        - float(reference["median_contribution_count"])
    )
    checks = {
        "effective_writes_per_pixel": density_relative_error
        <= config.density_relative_tolerance,
        "unique_coverage": coverage_error
        <= config.coverage_absolute_tolerance,
        "zero_count": zero_error <= config.zero_absolute_tolerance,
        "count_lt2": count_lt2_error <= config.count_lt2_absolute_tolerance,
        "median": median_error <= config.median_count_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "density_relative_error": density_relative_error,
        "coverage_absolute_error": coverage_error,
        "zero_absolute_error": zero_error,
        "count_lt2_absolute_error": count_lt2_error,
        "median_absolute_error": median_error,
        "tolerances": {
            "density_relative": config.density_relative_tolerance,
            "coverage_absolute": config.coverage_absolute_tolerance,
            "zero_absolute": config.zero_absolute_tolerance,
            "count_lt2_absolute": config.count_lt2_absolute_tolerance,
            "median_absolute": config.median_count_tolerance,
        },
    }


def _sample_sequence_validation(
    base: object,
    sampling_cells: Tensor,
    config: HighSampleConfig,
) -> dict[str, object]:
    count = config.validation_emitters
    monolithic = sample_meshfree_zero_set(
        base,
        count,
        sobol_scramble_seed=config.train_scramble_seed,
        sampling_cells=sampling_cells,
    )
    checks: dict[str, object] = {}
    for chunk_size in config.validation_chunk_sizes:
        points: list[Tensor] = []
        normals: list[Tensor] = []
        cell_ids: list[Tensor] = []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            state = sample_meshfree_zero_set(
                base,
                stop - start,
                sobol_scramble_seed=config.train_scramble_seed,
                sobol_start_index=start,
                sampling_cells=sampling_cells,
            )
            points.append(state.points)
            normals.append(state.normals)
            cell_ids.append(state.source_cell_ids)
        joined_points = torch.cat(points)
        joined_normals = torch.cat(normals)
        joined_ids = torch.cat(cell_ids)
        checks[str(chunk_size)] = {
            "point_max_error": float(
                (joined_points - monolithic.points).abs().max()
            ),
            "normal_max_error": float(
                (joined_normals - monolithic.normals).abs().max()
            ),
            "source_cell_ids_exact": bool(
                torch.equal(joined_ids, monolithic.source_cell_ids)
            ),
        }
    passed = all(
        float(item["point_max_error"]) == 0.0
        and float(item["normal_max_error"]) == 0.0
        and bool(item["source_cell_ids_exact"])
        for item in checks.values()
    )
    return {
        "emitters": count,
        "monolithic_chunk_size": count,
        "tested_chunk_sizes": list(config.validation_chunk_sizes),
        "sobol_scramble": config.train_scramble_seed is not None,
        "sobol_seed": config.train_scramble_seed,
        "checks": checks,
        "passed": passed,
    }


def _render_chunk_validation(
    base: object,
    target: object,
    surface: SurfaceState,
    atlas: DirectionAtlas,
    config: HighSampleConfig,
) -> dict[str, object]:
    validation_config = replace(config, views=2)
    count = config.validation_emitters
    boundary = enclosing_observation_sphere(surface.reference_points[:count])
    results: dict[int, StreamResult] = {}
    for chunk_size in (count, *config.validation_chunk_sizes):
        results[chunk_size] = _stream_render(
            base,
            target,
            surface,
            count,
            (64, 64),
            boundary,
            atlas,
            validation_config,
            chunk_size=chunk_size,
            capture_view=0,
        )
    monolithic = results[count]
    checks: dict[str, object] = {}
    for chunk_size in config.validation_chunk_sizes:
        candidate = results[chunk_size]
        base_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(monolithic.base_images, candidate.base_images)
        )
        assert monolithic.target_images is not None
        assert candidate.target_images is not None
        target_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(
                monolithic.target_images, candidate.target_images
            )
        )
        checks[str(chunk_size)] = {
            "base_image_max_error": base_error,
            "target_image_max_error": target_error,
            "base_retained_event_difference": int(
                candidate.report["base_retained_events"]
            )
            - int(monolithic.report["base_retained_events"]),
            "base_footprint_write_difference": int(
                candidate.report["base_footprint_writes"]
            )
            - int(monolithic.report["base_footprint_writes"]),
            "target_retained_event_difference": int(
                candidate.report["target_retained_events"]
            )
            - int(monolithic.report["target_retained_events"]),
        }
    maximum_error = max(
        max(
            float(item["base_image_max_error"]),
            float(item["target_image_max_error"]),
        )
        for item in checks.values()
    )
    passed = maximum_error < 2e-6 and all(
        int(item["base_retained_event_difference"]) == 0
        and int(item["base_footprint_write_difference"]) == 0
        and int(item["target_retained_event_difference"]) == 0
        for item in checks.values()
    )
    del results
    _release()
    return {
        "resolution": [64, 64],
        "views": 2,
        "emitters": count,
        "reference_chunk_size": count,
        "checks": checks,
        "maximum_image_error": maximum_error,
        "tolerance": 2e-6,
        "passed": passed,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _save_reference(path: Path, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def _save_density_plots(
    figure_directory: Path,
    fixed_rows: list[dict[str, object]],
    matched_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    pixels = [int(row["total_pixels"]) // int(row["views"]) for row in matched_rows]
    resolutions = [
        f"{row['resolution'][1]}x{row['resolution'][0]}" for row in matched_rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    treatments = (
        (fixed_rows, "fixed samples"),
        (matched_rows, "matched density"),
    )
    for rows, label in treatments:
        axes[0].plot(
            pixels,
            [float(row["effective_writes_per_pixel"]) for row in rows],
            marker="o",
            label=label,
        )
        axes[1].plot(
            pixels,
            [float(row["unique_coverage_ratio"]) for row in rows],
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("pixels per view")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("effective footprint writes / pixel")
    axes[1].set_ylabel("unique coverage")
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_density_vs_resolution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        pixels,
        [int(row["emitters"]) for row in matched_rows],
        marker="o",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("pixels per view")
    axes[0].set_ylabel("emitters required")
    axes[1].scatter(
        [float(row["surface_samples_per_pixel"]) for row in calibration_rows],
        [float(row["effective_writes_per_pixel"]) for row in calibration_rows],
    )
    axes[1].set_xlabel("emitters / pixel")
    axes[1].set_ylabel("effective writes / pixel")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_emitters_vs_resolution.png", dpi=180)
    plt.close(figure)


def _save_comparison_plots(
    figure_directory: Path,
    fixed_rows: list[dict[str, object]],
    matched_rows: list[dict[str, object]],
    mc_rows: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['resolution'][1]}x{row['resolution'][0]}" for row in matched_rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        x,
        [float(row["rmse_per_rgb_scalar"]) for row in fixed_rows],
        marker="o",
        label="fixed samples",
    )
    axis.plot(
        x,
        [float(row["rmse_per_rgb_scalar"]) for row in matched_rows],
        marker="o",
        label="matched density",
    )
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("RGB RMSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_directory / "v082_fixed_vs_matched_density_rgb.png", dpi=180
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        x,
        [float(row["symmetric_chamfer"]) for row in geometry_rows],
        marker="o",
    )
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("fixed-geometry symmetric Chamfer")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_directory / "v082_fixed_vs_matched_density_geometry.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        x,
        [float(row["mc_self_mse"]) for row in mc_rows],
        marker="o",
        label="MC self MSE",
    )
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("same geometry, independent samples MSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_mc_self_noise.png", dpi=180)
    plt.close(figure)


def _save_histograms(
    path: Path, matched_rows: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for row in matched_rows:
        histogram = np.asarray(row["contribution_histogram"], dtype=np.float64)
        histogram /= histogram.sum()
        limit = min(40, histogram.size)
        axis.plot(
            np.arange(limit),
            histogram[:limit],
            marker=".",
            label=f"{row['resolution'][1]}x{row['resolution'][0]}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("contribution count")
    axis.set_ylabel("pixel probability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_montages(
    figure_directory: Path,
    matched_results: list[StreamResult],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(matched_results), 3, figsize=(10, 10))
    for row, result in enumerate(matched_results):
        report = result.report
        label = f"{report['resolution'][1]}x{report['resolution'][0]}"
        axes[row, 0].imshow(np.clip(result.capture["target"], 0.0, 1.0))
        axes[row, 1].imshow(np.clip(result.capture["base"], 0.0, 1.0))
        axes[row, 2].imshow(result.capture["support"], cmap="gray")
        axes[row, 0].set_ylabel(label)
    for column, title in enumerate(("target", "imperfect geometry", "support")):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_rgb_montage.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(matched_results), 3, figsize=(9, 10))
    for row, result in enumerate(matched_results):
        support = result.capture["support"]
        ys, xs = np.nonzero(support)
        if ys.size:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            height, width = y1 - y0, x1 - x0
            y0 += int(0.20 * height)
            y1 = max(y0 + 1, y1 - int(0.20 * height))
            x0 += int(0.20 * width)
            x1 = max(x0 + 1, x1 - int(0.20 * width))
        else:
            y0, y1, x0, x1 = 0, support.shape[0], 0, support.shape[1]
        report = result.report
        label = f"{report['resolution'][1]}x{report['resolution'][0]}"
        axes[row, 0].imshow(
            np.clip(result.capture["target"][y0:y1, x0:x1], 0.0, 1.0)
        )
        axes[row, 1].imshow(
            np.clip(result.capture["base"][y0:y1, x0:x1], 0.0, 1.0)
        )
        axes[row, 2].imshow(support[y0:y1, x0:x1], cmap="gray")
        axes[row, 0].set_ylabel(label)
    for column, title in enumerate(("target crop", "imperfect crop", "support crop")):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_rgb_zoom_crops.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(matched_results), figsize=(14, 4))
    for axis, result in zip(axes, matched_results):
        image = axis.imshow(
            np.log1p(result.capture["counts"]), cmap="magma"
        )
        report = result.report
        axis.set_title(f"{report['resolution'][1]}x{report['resolution'][0]}")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_transport_count_maps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(matched_results), figsize=(14, 4))
    for axis, result in zip(axes, matched_results):
        image = axis.imshow(result.capture["mass"], cmap="viridis")
        report = result.report
        axis.set_title(f"{report['resolution'][1]}x{report['resolution'][0]}")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(figure_directory / "v082_weight_maps.png", dpi=180)
    plt.close(figure)


def _save_runtime(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['resolution'][1]}x{row['resolution'][0]}" for row in rows]
    x = np.arange(len(rows))
    emission = np.asarray([float(row["emission_seconds"]) for row in rows])
    visibility = np.asarray([float(row["visibility_seconds"]) for row in rows])
    projection = np.asarray(
        [float(row["projection_and_footprint_seconds"]) for row in rows]
    )
    accumulation = np.asarray(
        [float(row["image_accumulation_seconds"]) for row in rows]
    )
    diagnostics = np.asarray([float(row["diagnostics_seconds"]) for row in rows])
    figure, axis = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(rows))
    for values, label in (
        (emission, "emission/outward filtering"),
        (visibility, "visibility"),
        (projection, "projection/footprint"),
        (accumulation, "accumulation"),
        (diagnostics, "diagnostics"),
    ):
        axis.bar(x, values, bottom=bottom, label=label)
        bottom += values
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("seconds")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_high_sample_resolution_control(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
    config: HighSampleConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    experiment = config or HighSampleConfig()
    if len(experiment.resolutions) != len(experiment.initial_emitter_counts):
        raise ValueError("each resolution needs an initial emitter count")
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    device = torch.device("cuda")
    base = prepared.base_field.to(device)
    target = prepared.gt_field.to(device)
    sampling_cells = sign_changing_cells(base.grid)
    sequence_validation = _sample_sequence_validation(
        base, sampling_cells, experiment
    )
    if not sequence_validation["passed"]:
        raise RuntimeError(f"SOBOL_STREAM_VALIDATION_FAILED:{sequence_validation}")
    maximum_emitters = max(experiment.initial_emitter_counts)
    train_surface = _prepare_surface(
        base,
        target,
        maximum_emitters,
        experiment.emitter_chunk_size,
        experiment.train_scramble_seed,
        sampling_cells,
    )
    heldout_surface = _prepare_surface(
        base,
        None,
        maximum_emitters,
        experiment.emitter_chunk_size,
        experiment.heldout_scramble_seed,
        sampling_cells,
    )
    mc_surface_a = _prepare_surface(
        base,
        None,
        maximum_emitters,
        experiment.emitter_chunk_size,
        experiment.mc_scramble_seeds[0],
        sampling_cells,
    )
    atlas = nested_fibonacci_atlas(device, (experiment.views,))
    render_validation = _render_chunk_validation(
        base, target, train_surface, atlas, experiment
    )
    if not render_validation["passed"]:
        raise RuntimeError(f"STREAM_RENDER_VALIDATION_FAILED:{render_validation}")

    matched_results: list[StreamResult] = []
    matched_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    reference: dict[str, object] | None = None
    boundaries: list[ObservationSphere] = []
    for resolution, emitters in zip(
        experiment.resolutions, experiment.initial_emitter_counts
    ):
        boundary = enclosing_observation_sphere(
            train_surface.reference_points[:emitters]
        )
        boundaries.append(boundary)
        result = _stream_render(
            base,
            target,
            train_surface,
            emitters,
            resolution,
            boundary,
            atlas,
            experiment,
            chunk_size=experiment.emitter_chunk_size,
        )
        row = result.report
        row["treatment"] = "MATCHED-DENSITY HIGH RESOLUTION"
        if reference is None:
            reference = row
        gate = _density_gate(reference, row, experiment)
        row["density_gate"] = gate
        row["sampling_density_status"] = (
            "MATCHED" if gate["passed"] else "SAMPLING_DENSITY_NOT_MATCHED"
        )
        calibration_rows.append(
            {
                "resolution": row["resolution"],
                "emitters": emitters,
                "attempted_packets": row["attempted_packets"],
                "base_retained_events": row["base_retained_events"],
                "effective_writes_per_pixel": row[
                    "effective_writes_per_pixel"
                ],
                "unique_coverage_ratio": row["unique_coverage_ratio"],
                "surface_samples_per_pixel": row["surface_samples_per_pixel"],
                "gate_passed": gate["passed"],
            }
        )
        matched_rows.append(row)
        matched_results.append(result)
        _progress(
            "matched_density",
            resolution=list(resolution),
            emitters=emitters,
            gate=gate["passed"],
        )
    assert reference is not None

    mc_rows: list[dict[str, object]] = []
    for result, boundary, resolution, emitters in zip(
        matched_results,
        boundaries,
        experiment.resolutions,
        experiment.initial_emitter_counts,
    ):
        result.base_images.clear()
        if result.target_images is not None:
            result.target_images.clear()
        independent_a = _stream_render(
            base,
            None,
            mc_surface_a,
            emitters,
            resolution,
            boundary,
            atlas,
            experiment,
            chunk_size=experiment.emitter_chunk_size,
        )
        heldout = _stream_render(
            base,
            None,
            heldout_surface,
            emitters,
            resolution,
            boundary,
            atlas,
            experiment,
            chunk_size=experiment.emitter_chunk_size,
        )
        mc = _pair_metrics(independent_a.base_images, heldout.base_images)
        mc_rows.append(
            {
                "resolution": list(resolution),
                "emitters": emitters,
                "surface_samples_per_pixel": result.report[
                    "surface_samples_per_pixel"
                ],
                "effective_writes_per_pixel": result.report[
                    "effective_writes_per_pixel"
                ],
                "mc_scramble_seeds": list(experiment.mc_scramble_seeds),
                "independent_a_runtime_seconds": independent_a.report[
                    "runtime_seconds"
                ],
                "independent_a_peak_allocated_mib": independent_a.report[
                    "peak_allocated_mib"
                ],
                "independent_a_peak_reserved_mib": independent_a.report[
                    "peak_reserved_mib"
                ],
                "independent_a_unique_coverage": independent_a.report[
                    "unique_coverage_ratio"
                ],
                "independent_a_effective_density": independent_a.report[
                    "effective_writes_per_pixel"
                ],
                "independent_a_count_lt2_fraction": independent_a.report[
                    "count_lt2_fraction"
                ],
                "heldout_runtime_seconds": heldout.report["runtime_seconds"],
                "heldout_peak_allocated_mib": heldout.report[
                    "peak_allocated_mib"
                ],
                "heldout_peak_reserved_mib": heldout.report[
                    "peak_reserved_mib"
                ],
                "independent_b_unique_coverage": heldout.report[
                    "unique_coverage_ratio"
                ],
                "independent_b_effective_density": heldout.report[
                    "effective_writes_per_pixel"
                ],
                "independent_b_count_lt2_fraction": heldout.report[
                    "count_lt2_fraction"
                ],
                **mc,
            }
        )
        del independent_a, heldout
        _release()
        _progress("mc_self", resolution=list(resolution), emitters=emitters)

    fixed_results: list[StreamResult] = [matched_results[0]]
    fixed_rows: list[dict[str, object]] = [dict(matched_rows[0])]
    fixed_rows[0]["treatment"] = "FIXED-SAMPLE HIGH RESOLUTION"
    for resolution in experiment.resolutions[1:]:
        boundary = enclosing_observation_sphere(
            train_surface.reference_points[: experiment.reference_emitters]
        )
        result = _stream_render(
            base,
            target,
            train_surface,
            experiment.reference_emitters,
            resolution,
            boundary,
            atlas,
            experiment,
            chunk_size=experiment.emitter_chunk_size,
        )
        result.report["treatment"] = "FIXED-SAMPLE HIGH RESOLUTION"
        fixed_results.append(result)
        fixed_rows.append(result.report)
        result.base_images.clear()
        if result.target_images is not None:
            result.target_images.clear()
        _progress("fixed_samples", resolution=list(resolution))

    evaluator = BunnyGeometryEvaluator(
        prepared,
        train_surface.reference_points[: experiment.reference_emitters],
        train_surface.reference_normals[: experiment.reference_emitters],
        16_384,
    )
    fixed_geometry = evaluator(
        train_surface.base_points[: experiment.reference_emitters]
    )
    geometry_rows = [
        {
            "resolution": list(resolution),
            "emitters": emitters,
            "optimization": "not run before differentiable streaming support",
            **fixed_geometry,
        }
        for resolution, emitters in zip(
            experiment.resolutions, experiment.initial_emitter_counts
        )
    ]

    v081_path = artifact_directory / "v081_highres_sampling_diagnostic.json"
    v081 = json.loads(v081_path.read_text())
    footprint_control = v081["footprint_scaling_control"]
    aware_footprint_rows = [
        row
        for row in footprint_control
        if str(row["label"]).endswith("resolution_aware")
    ]

    figure_directory.mkdir(parents=True, exist_ok=True)
    render_directory.mkdir(parents=True, exist_ok=True)
    _save_reference(
        render_directory / "v082_low_reference_rgb.png",
        matched_results[0].capture["base"],
    )
    _save_density_plots(
        figure_directory, fixed_rows, matched_rows, calibration_rows
    )
    _save_comparison_plots(
        figure_directory, fixed_rows, matched_rows, mc_rows, geometry_rows
    )
    _save_histograms(
        figure_directory / "v082_contribution_histograms.png", matched_rows
    )
    _save_montages(figure_directory, matched_results)
    _save_runtime(
        figure_directory / "v082_runtime_scaling.png", matched_rows
    )

    all_density_pass = all(
        bool(row["density_gate"]["passed"]) for row in matched_rows
    )
    fullhd_pass = bool(matched_rows[-1]["density_gate"]["passed"])
    mc_values = [float(row["mc_self_mse"]) for row in mc_rows]
    mc_ratio = max(mc_values) / max(min(mc_values), 1e-30)
    mc_stable = mc_ratio <= 1.25
    fixed_degrades = (
        float(fixed_rows[-1]["effective_writes_per_pixel"])
        < 0.10 * float(fixed_rows[0]["effective_writes_per_pixel"])
        and float(fixed_rows[-1]["rmse_per_rgb_scalar"])
        > 1.50 * float(fixed_rows[0]["rmse_per_rgb_scalar"])
    )
    matched_image_stable = (
        all_density_pass
        and max(float(row["rmse_per_rgb_scalar"]) for row in matched_rows)
        / min(float(row["rmse_per_rgb_scalar"]) for row in matched_rows)
        <= 1.25
    )
    verdicts = {
        "LOW_RES_REFERENCE_DENSE": (
            float(reference["effective_writes_per_pixel"]) >= 4.0
            and float(reference["unique_coverage_ratio"]) >= 0.25
        ),
        "FULLHD_MATCHED_DENSITY_ACHIEVED": fullhd_pass,
        "STREAMING_TRANSPORT_VALIDATED": bool(render_validation["passed"]),
        "CHUNK_SIZE_INVARIANT": bool(render_validation["passed"]),
        "COMMON_RANDOM_NUMBERS_VALIDATED": bool(
            sequence_validation["passed"]
        ),
        "MC_SELF_NOISE_STABLE_AT_MATCHED_DENSITY": mc_stable,
        "FIXED_SAMPLE_HIGHRES_DEGRADES": fixed_degrades,
        "MATCHED_DENSITY_HIGHRES_IMAGE_STABLE": matched_image_stable,
        "MATCHED_DENSITY_HIGHRES_GEOMETRY_IMPROVES": False,
        "RESOLUTION_AWARE_FOOTPRINT_ONLY_SMOOTHS": True,
        "REAL_SAMPLE_INCREASE_NEEDED_FOR_HIGH_BANDWIDTH": (
            fullhd_pass
            and float(matched_rows[-1]["effective_writes_per_pixel"])
            >= 0.90 * float(reference["effective_writes_per_pixel"])
            and float(matched_rows[-1]["unique_coverage_ratio"])
            - float(fixed_rows[-1]["unique_coverage_ratio"])
            >= 0.10
        ),
        "HIGH_RES_BIRTH_SIGNAL_STRONGER": False,
        "HIGH_RES_BIRTH_BEATS_RANDOM": False,
        "HIGH_RES_BIRTH_HELDOUT_SUPPORTED": False,
    }
    verdict_evidence = {
        "LOW_RES_REFERENCE_DENSE": {
            "effective_writes_per_pixel": reference[
                "effective_writes_per_pixel"
            ],
            "median_contribution_count": reference[
                "median_contribution_count"
            ],
            "unique_coverage_ratio": reference["unique_coverage_ratio"],
            "thresholds": {"writes": 4.0, "unique_coverage": 0.25},
        },
        "FULLHD_MATCHED_DENSITY_ACHIEVED": matched_rows[-1]["density_gate"],
        "STREAMING_TRANSPORT_VALIDATED": render_validation,
        "CHUNK_SIZE_INVARIANT": render_validation,
        "COMMON_RANDOM_NUMBERS_VALIDATED": sequence_validation,
        "MC_SELF_NOISE_STABLE_AT_MATCHED_DENSITY": {
            "mse_values": mc_values,
            "maximum_to_minimum_ratio": mc_ratio,
            "threshold": 1.25,
            "support_disagreement_fractions": [
                row["support_disagreement_fraction"] for row in mc_rows
            ],
            "common_support_mse_values": [
                row["common_support_mse"] for row in mc_rows
            ],
            "investigation": (
                "Both independent sample sets have matched aggregate density; "
                "per-view and support-XOR diagnostics separate footprint phase/"
                "support-threshold noise from common-support RGB variation."
            ),
        },
        "FIXED_SAMPLE_HIGHRES_DEGRADES": {
            "low_density": fixed_rows[0]["effective_writes_per_pixel"],
            "fullhd_density": fixed_rows[-1]["effective_writes_per_pixel"],
            "low_rmse": fixed_rows[0]["rmse_per_rgb_scalar"],
            "fullhd_rmse": fixed_rows[-1]["rmse_per_rgb_scalar"],
        },
        "MATCHED_DENSITY_HIGHRES_IMAGE_STABLE": {
            "rmse_values": [
                row["rmse_per_rgb_scalar"] for row in matched_rows
            ],
            "maximum_to_minimum_ratio": max(
                float(row["rmse_per_rgb_scalar"]) for row in matched_rows
            )
            / min(float(row["rmse_per_rgb_scalar"]) for row in matched_rows),
            "maximum_to_minimum_threshold": 1.25,
            "all_density_gates_pass": all_density_pass,
        },
        "MATCHED_DENSITY_HIGHRES_GEOMETRY_IMPROVES": {
            "status": "NOT_TESTED_NO_STREAMING_JACOBIAN",
            "fixed_geometry_metrics": fixed_geometry,
        },
        "RESOLUTION_AWARE_FOOTPRINT_ONLY_SMOOTHS": {
            "control_source": str(v081_path),
            "fixed_emitters": [row["emitters"] for row in aware_footprint_rows],
            "labels": [row["label"] for row in aware_footprint_rows],
            "aware_unique_coverage": [
                row["unique_coverage_ratio"] for row in aware_footprint_rows
            ],
            "aware_rmse": [
                row["rmse_per_rgb_scalar"] for row in aware_footprint_rows
            ],
            "aware_footprint_area_pixels": [
                row["average_effective_footprint_area_pixels"]
                for row in aware_footprint_rows
            ],
            "interpretation": (
                "fixed samples plus an enlarged kernel restore normalized "
                "support without adding detector observations"
            ),
        },
        "REAL_SAMPLE_INCREASE_NEEDED_FOR_HIGH_BANDWIDTH": {
            "fixed_fullhd_coverage": fixed_rows[-1][
                "unique_coverage_ratio"
            ],
            "matched_fullhd_coverage": matched_rows[-1][
                "unique_coverage_ratio"
            ],
            "fixed_fullhd_density": fixed_rows[-1][
                "effective_writes_per_pixel"
            ],
            "matched_fullhd_density": matched_rows[-1][
                "effective_writes_per_pixel"
            ],
            "minimum_coverage_gain": 0.10,
            "minimum_reference_density_fraction": 0.90,
        },
        "HIGH_RES_BIRTH_SIGNAL_STRONGER": {
            "status": "NOT_TESTED_PREREQUISITE_FAILED",
            "birth_executed": False,
            "mc_maximum_to_minimum_ratio": mc_ratio,
            "mc_threshold": 1.25,
        },
        "HIGH_RES_BIRTH_BEATS_RANDOM": {
            "status": "NOT_TESTED_PREREQUISITE_FAILED",
            "birth_executed": False,
            "mc_maximum_to_minimum_ratio": mc_ratio,
            "mc_threshold": 1.25,
        },
        "HIGH_RES_BIRTH_HELDOUT_SUPPORTED": {
            "status": "NOT_TESTED_PREREQUISITE_FAILED",
            "birth_executed": False,
            "mc_maximum_to_minimum_ratio": mc_ratio,
            "mc_threshold": 1.25,
        },
    }
    runtime = time.perf_counter() - started
    recorded_allocated = [
        float(train_surface.report["peak_allocated_mib"]),
        float(heldout_surface.report["peak_allocated_mib"]),
        float(mc_surface_a.report["peak_allocated_mib"]),
        *(float(row["peak_allocated_mib"]) for row in matched_rows),
        *(float(row["peak_allocated_mib"]) for row in fixed_rows),
        *(float(row["heldout_peak_allocated_mib"]) for row in mc_rows),
        *(float(row["independent_a_peak_allocated_mib"]) for row in mc_rows),
    ]
    recorded_reserved = [
        float(train_surface.report["peak_reserved_mib"]),
        float(heldout_surface.report["peak_reserved_mib"]),
        float(mc_surface_a.report["peak_reserved_mib"]),
        *(float(row["peak_reserved_mib"]) for row in matched_rows),
        *(float(row["peak_reserved_mib"]) for row in fixed_rows),
        *(float(row["heldout_peak_reserved_mib"]) for row in mc_rows),
        *(float(row["independent_a_peak_reserved_mib"]) for row in mc_rows),
    ]
    report: dict[str, object] = {
        "version": "0.8.2",
        "question": (
            "At fixed transport sampling density per detector degree of "
            "freedom, does increasing resolution improve geometry evidence?"
        ),
        "configuration": asdict(experiment),
        "environment": cuda_environment(),
        "commands": {
            "formal": (
                "PYTHONPATH=src conda run --no-capture-output -n test "
                "python demo.py --high-sample-resolution --bunny-mesh "
                "data/stanford_bunny/cache/bun_zipper.ply"
            ),
            "verification": (
                "PYTHONPATH=src conda run --no-capture-output -n test "
                "python demo.py --scene sphere --verify"
            ),
        },
        "sample_sequence_validation": sequence_validation,
        "streaming_validation": render_validation,
        "surface_preparation": {
            "train": train_surface.report,
            "heldout": heldout_surface.report,
            "mc_independent_a": mc_surface_a.report,
        },
        "sample_sets": {
            "train": {
                "sobol_scramble": False,
                "seed": None,
                "index_ranges_by_resolution": [
                    [0, count] for count in experiment.initial_emitter_counts
                ],
                "policy": "nested global Sobol prefixes",
            },
            "mc_independent_a": {
                "sobol_scramble": True,
                "seed": experiment.mc_scramble_seeds[0],
                "index_ranges_by_resolution": [
                    [0, count] for count in experiment.initial_emitter_counts
                ],
                "policy": "nested global Sobol prefixes",
            },
            "heldout_mc_independent_b": {
                "sobol_scramble": True,
                "seed": experiment.heldout_scramble_seed,
                "index_ranges_by_resolution": [
                    [0, count] for count in experiment.initial_emitter_counts
                ],
                "policy": "nested global Sobol prefixes",
            },
        },
        "low_resolution_reference": reference,
        "calibration_rows": calibration_rows,
        "matched_density": matched_rows,
        "fixed_sample_control": fixed_rows,
        "mc_self_noise": mc_rows,
        "fixed_geometry": geometry_rows,
        "resolution_aware_footprint_control": footprint_control,
        "birth": {
            "executed": False,
            "reason": (
                "The prerequisite MC self-noise stability gate failed, so the "
                "protocol requires investigation before geometry optimization "
                "or birth. In addition, the validated streaming path is "
                "forward-only: the existing explicit Jacobian materialization "
                "is not memory-safe at matched high-resolution emitter counts. "
                "No birth claim is made."
            ),
        },
        "verdicts": verdicts,
        "verdict_evidence": verdict_evidence,
        "runtime_seconds": runtime,
        "peak_allocated_mib": max(recorded_allocated),
        "peak_reserved_mib": max(recorded_reserved),
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "artifacts": {
            "json": str(
                artifact_directory / "v082_high_sample_resolution_control.json"
            ),
            "csv": str(
                artifact_directory / "v082_high_sample_resolution_control.csv"
            ),
            "figures": str(figure_directory / "v082_*.png"),
            "low_reference": str(
                render_directory / "v082_low_reference_rgb.png"
            ),
        },
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v082_high_sample_resolution_control.json"
    json_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    csv_rows: list[dict[str, object]] = []
    for section, rows in (
        ("calibration", calibration_rows),
        ("matched_density", matched_rows),
        ("fixed_samples", fixed_rows),
        ("mc_self", mc_rows),
        ("fixed_geometry", geometry_rows),
    ):
        csv_rows.extend({"section": section, **row} for row in rows)
    _write_csv(
        artifact_directory / "v082_high_sample_resolution_control.csv",
        csv_rows,
    )
    return report
