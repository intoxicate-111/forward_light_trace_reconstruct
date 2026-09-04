"""Multiview transport evidence and natural geometry-birth capacity."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bandwidth import (
    FIXED_LEVELS,
    FixedSolution,
    GeometryBundle,
    _birth_regions,
    _canonical_emitters,
    _geometry_bundle,
    _image_digest,
    _operator_diagnostics,
    _physical_camera_digest,
    _repeated_config,
    _solve_fixed,
    _sync,
)
from .batch import BatchPolicy, _run_batch_trajectory
from .benchmark import BenchmarkGeometry, cuda_environment
from .bunny import BunnyExperimentConfig
from .fields import LocalBasisField
from .jacobian import (
    FixedTransportCell,
    _continuous_sensor,
    _footprint,
    nested_deterministic_direction_block,
    nested_deterministic_directions,
    nested_deterministic_directions_for_ids,
    render_fixed_transport_cell,
    sparse_bilinear_transport,
)
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    _compare_camera,
    camera_pose,
    multiview_cameras,
    project_camera,
)
from .mesh_field import prepare_stanford_bunny
from .sequential import ActiveState, SequentialContext, _correlation
from .tracer import PhotonBatch, first_zero_set_intersections


Tensor = torch.Tensor
PACKET_LEVELS = {"P1": 32, "P10": 320, "P100": 3200}
VIEW_LEVELS = (8, 20, 40)
CAPTURE_CURVE_VIEWS = (1, 2, 4, 8, 16, 20, 40)
MAJOR_CONDITIONS = (
    ("V8_P1", 8, "P1"),
    ("V8_P10", 8, "P10"),
    ("V8_P100", 8, "P100"),
    ("V20_P10", 20, "P10"),
    ("V40_P10", 40, "P10"),
    ("V40_P100", 40, "P100"),
)


def natural_cpu_verification() -> dict[str, object]:
    """Check the v0.3g nested photon and camera contracts on CPU."""
    device = torch.device("cpu")
    direction_report = _nested_direction_verification(device)
    cameras_8 = nested_bunny_cameras(256, device, 8)
    cameras_40 = nested_bunny_cameras(256, device, 40)
    camera_error = max(
        float((left.center - right.center).abs().max())
        for left, right in zip(cameras_8, cameras_40[:8])
    )
    assert camera_error == 0.0
    return {
        **direction_report,
        "V8_in_V40_camera_prefix_max_error": camera_error,
        "camera_count": len(cameras_40),
    }


@dataclass
class StreamedTransport:
    packet_level: str
    packets_per_emitter: int
    emitted: int
    cameras: list[object]
    base_cells: list[FixedTransportCell]
    target_cells: list[FixedTransportCell]
    base_images: list[Tensor]
    target_images: list[Tensor]
    base_valid_hits: list[int]
    target_valid_hits: list[int]
    base_survivors: list[int]
    target_survivors: list[int]
    base_event_counts: list[Tensor]
    target_event_counts: list[Tensor]
    capture_by_prefix: dict[int, dict[str, object]]
    state_event_by_view: list[float]
    owner_event_by_view: list[float]
    scene_seconds_base: float
    scene_seconds_target: float
    detector_seconds_by_view: list[float]
    chunks: int
    peak_allocated_mib: float
    peak_reserved_mib: float


def nested_bunny_cameras(
    resolution: int, device: torch.device, count: int = 40
) -> list[object]:
    """Keep the validated V8 set, then append deterministic surrounding views."""
    if count < 1 or count > 40:
        raise ValueError("nested Bunny camera count must be in [1, 40]")
    cameras = multiview_cameras(
        "bunny", (resolution, resolution), device, count=8
    )
    if count <= 8:
        return cameras[:count]
    pool = multiview_cameras(
        "bunny", (resolution, resolution), device, count=64
    )
    for camera in pool:
        if all(
            float(torch.linalg.vector_norm(camera.center - old.center)) > 1e-12
            for old in cameras
        ):
            cameras.append(camera)
        if len(cameras) == 40:
            break
    if len(cameras) != 40:
        raise RuntimeError("could not construct 40 unique nested Bunny cameras")
    return cameras[:count]


def _empty_photons(device: torch.device) -> PhotonBatch:
    vector = torch.empty((0, 3), dtype=torch.float64, device=device)
    scalar = torch.empty(0, dtype=torch.float64, device=device)
    return PhotonBatch(
        vector,
        vector,
        vector,
        scalar,
        scalar,
        torch.empty(0, dtype=torch.long, device=device),
    )


def _owner_update(
    owner_ids: Tensor,
    owner_times: Tensor,
    pixels: Tensor,
    arrivals: Tensor,
    global_ids: Tensor,
) -> None:
    if pixels.numel() == 0:
        return
    local_times = torch.full_like(owner_times, torch.inf)
    local_times.scatter_reduce_(0, pixels, arrivals, reduce="amin")
    winners = arrivals == local_times[pixels]
    missing = torch.iinfo(torch.long).max
    local_ids = torch.full_like(owner_ids, missing)
    local_ids.scatter_reduce_(
        0, pixels[winners], global_ids[winners], reduce="amin"
    )
    better = (local_times < owner_times) | (
        (local_times == owner_times) & (local_ids < owner_ids)
    )
    owner_times[better] = local_times[better]
    owner_ids[better] = local_ids[better]


def _state_values(valid: Tensor, absorbed: Tensor, pixels: Tensor) -> Tensor:
    state = torch.full(
        valid.shape, -1, dtype=torch.int32, device=valid.device
    )
    state[valid] = pixels[valid].to(torch.int32)
    state[absorbed] = -2
    return state


def _capture_summary(
    histogram: list[int], emitted: int, total_detector_hits: int
) -> dict[str, object]:
    unique = emitted - histogram[0]
    captured_histogram = histogram[1:]
    captured = max(unique, 1)
    weighted = sum(index * count for index, count in enumerate(histogram))
    captured_cumulative = 0
    captured_median = 0
    captured_p95 = 0
    for multiplicity, count in enumerate(captured_histogram, start=1):
        captured_cumulative += count
        if captured_median == 0 and captured_cumulative >= math.ceil(0.5 * captured):
            captured_median = multiplicity
        if captured_p95 == 0 and captured_cumulative >= math.ceil(0.95 * captured):
            captured_p95 = multiplicity
    maximum = max(
        (index for index, count in enumerate(histogram) if count), default=0
    )
    return {
        "emitted_photons": emitted,
        "valid_outward_non_degenerate_packets": emitted,
        "unique_capture_count": unique,
        "eta_capture_emitted": unique / max(emitted, 1),
        "eta_capture_valid": unique / max(emitted, 1),
        "total_detector_hit_count": total_detector_hits,
        "mean_capture_multiplicity_all_emitted": weighted / max(emitted, 1),
        "mean_capture_multiplicity_captured": weighted / captured,
        "median_capture_multiplicity_captured": captured_median,
        "p95_capture_multiplicity_captured": captured_p95,
        "maximum_capture_multiplicity": maximum,
        "fraction_captured_ge_1": unique / max(emitted, 1),
        "fraction_captured_ge_2": sum(histogram[2:]) / max(emitted, 1),
        "fraction_captured_ge_4": sum(histogram[4:]) / max(emitted, 1),
        "fraction_captured_ge_8": sum(histogram[8:]) / max(emitted, 1),
        "multiplicity_histogram": histogram,
    }


def _make_cell(
    geometry: GeometryBundle,
    camera: object,
    points: Tensor,
    owner_ids: Tensor,
    packets_per_emitter: int,
) -> FixedTransportCell:
    occupied = owner_ids >= 0
    photon_ids = owner_ids[occupied]
    emitter_ids = torch.div(
        photon_ids, packets_per_emitter, rounding_mode="floor"
    )
    packet_ids = torch.remainder(photon_ids, packets_per_emitter)
    directions = nested_deterministic_directions_for_ids(
        geometry.reference_normals,
        emitter_ids,
        packet_ids,
        packets_per_emitter,
        128.0,
    )
    sensor_xy, _ = _continuous_sensor(
        camera, points[emitter_ids], directions  # type: ignore[arg-type]
    )
    footprint_pixels, footprint_valid, footprint_base = _footprint(
        sensor_xy, camera.resolution  # type: ignore[attr-defined]
    )
    return FixedTransportCell(
        camera,  # type: ignore[arg-type]
        packets_per_emitter,
        128.0,
        "reference",
        geometry.colors,
        photon_ids,
        emitter_ids,
        directions,
        footprint_pixels,
        footprint_valid,
        footprint_base,
        torch.empty(0, dtype=torch.int32, device=points.device),
        owner_ids,
    )


def _stream_transport(
    geometry: GeometryBundle,
    cameras: list[object],
    packet_level: str,
    config: BunnyExperimentConfig,
    *,
    maximum_chunk_photons: int = 819_200,
) -> StreamedTransport:
    device = geometry.reference_points.device
    packets = PACKET_LEVELS[packet_level]
    emitted = config.emitters * packets
    views = len(cameras)
    pixels = config.resolution**2
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    missing = torch.iinfo(torch.long).max
    owner_base = [
        torch.full((pixels,), missing, dtype=torch.long, device=device)
        for _ in cameras
    ]
    owner_target = [item.clone() for item in owner_base]
    time_base = [
        torch.full((pixels,), torch.inf, dtype=torch.float64, device=device)
        for _ in cameras
    ]
    time_target = [item.clone() for item in time_base]
    event_counts_base = [
        torch.zeros(pixels, dtype=torch.int32, device=device) for _ in cameras
    ]
    event_counts_target = [item.clone() for item in event_counts_base]
    valid_base = [0] * views
    valid_target = [0] * views
    survivors_base = [0] * views
    survivors_target = [0] * views
    state_event_counts = [0] * views
    prefix_histograms = {
        count: [0] * (count + 1)
        for count in CAPTURE_CURVE_VIEWS
        if count <= views
    }
    prefix_detector_hits = {count: 0 for count in prefix_histograms}
    scene_seconds_base = 0.0
    scene_seconds_target = 0.0
    detector_events: list[list[tuple[torch.cuda.Event, torch.cuda.Event]]] = [
        [] for _ in cameras
    ]
    emitters_per_chunk = max(1, maximum_chunk_photons // packets)
    chunks = 0
    for emitter_start in range(0, config.emitters, emitters_per_chunk):
        emitter_stop = min(config.emitters, emitter_start + emitters_per_chunk)
        count = (emitter_stop - emitter_start) * packets
        directions = nested_deterministic_direction_block(
            geometry.reference_normals[emitter_start:emitter_stop],
            packets,
            128.0,
            emitter_offset=emitter_start,
        )
        emitter_ids = torch.arange(
            emitter_start, emitter_stop, device=device
        ).repeat_interleave(packets)
        global_ids = torch.arange(
            emitter_start * packets,
            emitter_stop * packets,
            dtype=torch.long,
            device=device,
        )
        base_origins = geometry.reference_points[emitter_ids]
        target_origins = geometry.target_points[emitter_ids]
        maximum_times = torch.full(
            (count,), 3.0, dtype=torch.float64, device=device
        )
        _sync(device)
        started = time.perf_counter()
        base_hits, base_hit_times = first_zero_set_intersections(
            geometry.base,
            base_origins,
            directions,
            maximum_times,
            samples=config.root_samples,
            bisection_steps=config.root_bisection_steps,
            chunk_size=8192,
        )
        _sync(device)
        scene_seconds_base += time.perf_counter() - started
        started = time.perf_counter()
        target_hits, target_hit_times = first_zero_set_intersections(
            geometry.gt,
            target_origins,
            directions,
            maximum_times,
            samples=config.root_samples,
            bisection_steps=config.root_bisection_steps,
            chunk_size=8192,
        )
        _sync(device)
        scene_seconds_target += time.perf_counter() - started
        multiplicity = torch.zeros(count, dtype=torch.uint8, device=device)
        cumulative_detector_hits = 0
        for view, camera in enumerate(cameras):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            base_valid, base_times, base_pixels, _ = camera.intersect(
                base_origins, directions
            )
            base_absorbed = (
                base_valid & base_hits & (base_hit_times < base_times)
            )
            base_survived = base_valid & ~base_absorbed
            target_valid_mask, target_times, target_pixels, _ = camera.intersect(
                target_origins, directions
            )
            target_absorbed = (
                target_valid_mask
                & target_hits
                & (target_hit_times < target_times)
            )
            target_survived = target_valid_mask & ~target_absorbed
            base_survivor_ids = torch.nonzero(
                base_survived, as_tuple=False
            ).flatten()
            target_survivor_ids = torch.nonzero(
                target_survived, as_tuple=False
            ).flatten()
            _owner_update(
                owner_base[view],
                time_base[view],
                base_pixels[base_survivor_ids],
                base_times[base_survivor_ids],
                global_ids[base_survivor_ids],
            )
            _owner_update(
                owner_target[view],
                time_target[view],
                target_pixels[target_survivor_ids],
                target_times[target_survivor_ids],
                global_ids[target_survivor_ids],
            )
            event_counts_base[view].scatter_add_(
                0,
                base_pixels[base_survivor_ids],
                torch.ones_like(base_survivor_ids, dtype=torch.int32),
            )
            event_counts_target[view].scatter_add_(
                0,
                target_pixels[target_survivor_ids],
                torch.ones_like(target_survivor_ids, dtype=torch.int32),
            )
            multiplicity += target_survived.to(torch.uint8)
            base_state = _state_values(
                base_valid, base_absorbed, base_pixels
            )
            target_state = _state_values(
                target_valid_mask, target_absorbed, target_pixels
            )
            state_event_counts[view] += int((base_state != target_state).sum())
            valid_base[view] += int(base_valid.sum())
            valid_target[view] += int(target_valid_mask.sum())
            survivors_base[view] += int(base_survived.sum())
            survived_count = int(target_survived.sum())
            survivors_target[view] += survived_count
            cumulative_detector_hits += survived_count
            prefix = view + 1
            if prefix in prefix_histograms:
                histogram = torch.bincount(
                    multiplicity.to(torch.long), minlength=prefix + 1
                ).detach().cpu().tolist()
                prefix_histograms[prefix] = [
                    old + int(new)
                    for old, new in zip(prefix_histograms[prefix], histogram)
                ]
                prefix_detector_hits[prefix] += cumulative_detector_hits
            end_event.record()
            detector_events[view].append((start_event, end_event))
        chunks += 1
        del (
            directions,
            emitter_ids,
            global_ids,
            base_origins,
            target_origins,
            maximum_times,
            base_hits,
            base_hit_times,
            target_hits,
            target_hit_times,
            multiplicity,
        )
    _sync(device)
    detector_seconds_by_view = [
        sum(start.elapsed_time(end) for start, end in events) / 1000.0
        for events in detector_events
    ]
    for owners in (*owner_base, *owner_target):
        owners[owners == missing] = -1
    base_cells = [
        _make_cell(geometry, camera, geometry.reference_points, owners, packets)
        for camera, owners in zip(cameras, owner_base)
    ]
    target_cells = [
        _make_cell(geometry, camera, geometry.target_points, owners, packets)
        for camera, owners in zip(cameras, owner_target)
    ]
    base_images = [
        sparse_bilinear_transport(cell, geometry.reference_points)[1].reshape(-1)
        for cell in base_cells
    ]
    target_images = [
        sparse_bilinear_transport(cell, geometry.target_points)[1].reshape(-1)
        for cell in target_cells
    ]
    owner_event_by_view = []
    for left, right in zip(owner_base, owner_target):
        occupied = (left >= 0) | (right >= 0)
        owner_event_by_view.append(
            float((left[occupied] != right[occupied]).to(torch.float64).mean())
            if bool(occupied.any())
            else 0.0
        )
    capture_by_prefix = {
        prefix: _capture_summary(
            histogram, emitted, prefix_detector_hits[prefix]
        )
        for prefix, histogram in prefix_histograms.items()
    }
    return StreamedTransport(
        packet_level,
        packets,
        emitted,
        cameras,
        base_cells,
        target_cells,
        base_images,
        target_images,
        valid_base,
        valid_target,
        survivors_base,
        survivors_target,
        event_counts_base,
        event_counts_target,
        capture_by_prefix,
        [count / emitted for count in state_event_counts],
        owner_event_by_view,
        scene_seconds_base,
        scene_seconds_target,
        detector_seconds_by_view,
        chunks,
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
    )


def _event_distribution(counts: list[Tensor]) -> dict[str, object]:
    nonempty = torch.cat(
        [item[item > 0].to(torch.float64) for item in counts]
    )
    return {
        "events_per_nonempty_pixel_mean": float(nonempty.mean())
        if nonempty.numel()
        else 0.0,
        "events_per_nonempty_pixel_median": float(nonempty.median())
        if nonempty.numel()
        else 0.0,
        "events_per_nonempty_pixel_p95": float(torch.quantile(nonempty, 0.95))
        if nonempty.numel()
        else 0.0,
    }


def _prefix_context(
    geometry: GeometryBundle,
    stream: StreamedTransport,
    views: int,
    config: BunnyExperimentConfig,
    condition: str,
) -> tuple[SequentialContext, dict[str, object], dict[str, object]]:
    device = geometry.reference_points.device
    scale = 1.0 / math.sqrt(views)
    cameras = stream.cameras[:views]
    cells = [
        replace(cell, emitter_colors=geometry.colors * scale)
        for cell in stream.base_cells[:views]
    ]
    target_images = [image * scale for image in stream.target_images[:views]]
    current_config = BunnyExperimentConfig(
        master_count=config.master_count,
        initial_count=config.initial_count,
        budget=config.budget,
        checkpoints=config.checkpoints,
        fixed_levels=FIXED_LEVELS,
        views=views,
        resolution=config.resolution,
        emitters=config.emitters,
        packets_per_emitter=stream.packets_per_emitter,
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
    empty_photons = _empty_photons(device)
    dummy_state = SceneTransportState(
        geometry.reference_points,
        geometry.reference_normals,
        empty_photons,
        empty_photons.directions,
        torch.empty(0, dtype=torch.bool, device=device),
        torch.empty(0, dtype=torch.float64, device=device),
        torch.ones(config.emitters, dtype=torch.bool, device=device),
    )
    context = SequentialContext(
        repeated,
        geometry.base,
        geometry.layout,
        geometry.support,
        geometry.levels,
        geometry.reference_points,
        geometry.reference_normals,
        geometry.colors,
        cells,
        dummy_state,
        benchmark_geometry,
        geometry.evaluator,
    )
    target = {
        "regime": "bunny",
        "seed": 0,
        "points": geometry.target_points,
        "images": target_images,
        "detail_ids": geometry.detail_ids,
        "detail_ids_top_ten": geometry.detail_ids_top_ten,
    }
    occupied = [
        int((cell.owner_map >= 0).sum())
        for cell in stream.target_cells[:views]
    ]
    total_pixels = views * config.resolution**2
    total_occupied = sum(occupied)
    target_counts = stream.target_event_counts[:views]
    capture = stream.capture_by_prefix[views]
    per_view = [
        {
            "view": index,
            "packet_hits": stream.target_valid_hits[index],
            "unique_packet_ids": stream.target_survivors[index],
            "occupied_pixels": occupied[index],
            "occupied_fraction": occupied[index] / config.resolution**2,
            "packets_per_occupied_pixel": stream.target_survivors[index]
            / max(occupied[index], 1),
        }
        for index in range(views)
    ]
    diagnostics = {
        "condition": condition,
        "views": views,
        "resolution": config.resolution,
        "packet_level": stream.packet_level,
        "packets_per_emitter": stream.packets_per_emitter,
        "photons": stream.emitted,
        "equal_view_weight": 1.0 / views,
        "objective_cell_color_scale": scale,
        "nested_camera_prefix": True,
        "physical_camera_sha256": _physical_camera_digest(cameras),
        "camera_poses": [camera_pose(camera) for camera in cameras],
        "target_image_sha256_unscaled": _image_digest(
            stream.target_images[:views]
        ),
        "capture": capture,
        "per_view_coverage": per_view,
        "union_occupied_pixels": total_occupied,
        "union_occupied_fraction": total_occupied / max(total_pixels, 1),
        "packets_per_occupied_pixel": sum(stream.target_survivors[:views])
        / max(total_occupied, 1),
        **_event_distribution(target_counts),
        "photon_state_transport_event_fraction": statistics.mean(
            stream.state_event_by_view[:views]
        ),
        "occupied_owner_event_fraction": statistics.mean(
            stream.owner_event_by_view[:views]
        ),
        "current_scene_transport_seconds": stream.scene_seconds_base,
        "target_scene_transport_seconds": stream.scene_seconds_target,
        "detector_transport_seconds": sum(
            stream.detector_seconds_by_view[:views]
        ),
        "detector_transport_seconds_by_view": stream.detector_seconds_by_view[
            :views
        ],
        "stream_chunks": stream.chunks,
        "stream_peak_allocated_mib": stream.peak_allocated_mib,
        "stream_peak_reserved_mib": stream.peak_reserved_mib,
        **geometry.diagnostics,
    }
    return context, target, diagnostics


def _per_view_metrics(
    state: ActiveState, target: dict[str, object], views: int
) -> dict[str, object]:
    scale = 1.0 / math.sqrt(views)
    per_view_mse = []
    per_view_half_sse = []
    squared_total = 0.0
    for image, target_image in zip(state.images, target["images"]):
        residual = (image - target_image) / scale
        squared = float((residual * residual).sum())
        squared_total += squared
        per_view_half_sse.append(0.5 * squared)
        per_view_mse.append(squared / residual.numel())
    return {
        "joint_equal_view_objective": state.loss,
        "joint_normalized_multiview_mse": statistics.mean(per_view_mse),
        "per_view_normalized_mse": per_view_mse,
        "per_view_half_sse": per_view_half_sse,
        "mean_per_view_normalized_mse": statistics.mean(per_view_mse),
        "worst_view_normalized_mse": max(per_view_mse),
        "variance_per_view_normalized_mse": statistics.pvariance(per_view_mse),
        "unscaled_multiview_residual_norm": math.sqrt(squared_total),
    }


def _annotate_trajectory_rows(
    trajectory: dict[str, object], views: int, resolution: int
) -> None:
    observations_per_view = resolution**2 * 3
    for row in trajectory["rows"]:
        row["joint_equal_view_objective"] = row["image_loss"]
        row["joint_normalized_multiview_mse"] = (
            2.0 * float(row["image_loss"]) / observations_per_view
        )


def _fixed_warm_references(
    context: SequentialContext,
    target: dict[str, object],
    condition: str,
    views: int,
    photons: int,
) -> tuple[list[dict[str, object]], dict[str, object], ActiveState]:
    solved_256 = _solve_fixed(context, target, 256, "shared_cold_warm")
    solved_512 = _solve_fixed(
        context, target, 512, "nested_warm", solved_256
    )
    solved_1024 = _solve_fixed(
        context, target, 1024, "nested_warm", solved_512
    )
    solutions: tuple[FixedSolution, ...] = (
        solved_256,
        solved_512,
        solved_1024,
    )
    rows = []
    for solution in solutions:
        row = dict(solution.row)
        row.update(
            {
                "condition": condition,
                "views": views,
                "photons": photons,
                **_per_view_metrics(solution.state, target, views),
            }
        )
        rows.append(row)
    operator = _operator_diagnostics(context, solved_256.state)
    expansion_image = max(
        float(row["expansion_image_max_difference"]) for row in rows[1:]
    )
    expansion_geometry = max(
        float(row["expansion_geometry_max_difference"]) for row in rows[1:]
    )
    return rows, {
        "operator": operator,
        "maximum_expansion_image_difference": expansion_image,
        "maximum_expansion_geometry_difference": expansion_geometry,
        "zero_padding_invariant": expansion_image < 1e-12
        and expansion_geometry < 1e-12,
    }, solved_256.state


def _trajectory_report(
    context: SequentialContext,
    target: dict[str, object],
    condition: str,
    views: int,
    photons: int,
) -> tuple[dict[str, object], dict[str, object]]:
    dynamic = _run_batch_trajectory(
        context,
        target,
        BatchPolicy(
            "natural_dynamic",
            dynamic=True,
            maximum_size=None,
            precompute_gram=True,
        ),
        retain_final_state=True,
    )
    schedule = tuple(dynamic["batch_schedule"])
    uniform = _run_batch_trajectory(
        context,
        target,
        BatchPolicy(
            "uniform_matched",
            uniform_schedule=schedule,
            precompute_gram=True,
        ),
        retain_final_state=True,
    )
    reports = []
    for trajectory in (dynamic, uniform):
        state = trajectory.pop("_final_state")
        _annotate_trajectory_rows(trajectory, views, context.config.resolution)
        trajectory["condition"] = condition
        trajectory["views"] = views
        trajectory["photons"] = photons
        trajectory["packets_per_emitter"] = photons // 8192
        trajectory["first_batch_size"] = trajectory["batch_schedule"][0]
        trajectory["p_max_reached"] = any(
            bool(batch["p_max_reached"]) for batch in trajectory["batches"]
        )
        trajectory["stopping_reasons"] = [
            batch["stopping_reason"] for batch in trajectory["batches"]
        ]
        trajectory["final_view_metrics"] = _per_view_metrics(
            state, target, views
        )
        trajectory["accepted_step_sizes"] = list(state.accepted_step_sizes)
        trajectory["optimization_loss_pairs"] = [
            list(pair) for pair in state.optimization_loss_pairs
        ]
        trajectory["cg_iterations"] = state.cg_iterations_total
        realized = [float(batch["realized_gain"]) for batch in trajectory["batches"]]
        calibration_valid = (
            len(realized) >= 3
            and max(realized, default=0.0) - min(realized, default=0.0) > 1e-15
        )
        trajectory["predicted_realized_calibration_valid"] = calibration_valid
        trajectory["predicted_realized_calibration_note"] = (
            "DEFINED"
            if calibration_valid
            else "UNDEFINED_ZERO_REALIZED_VARIANCE_OR_TOO_FEW_BATCHES"
        )
        reports.append(trajectory)
    return reports[0], reports[1]


def _nested_direction_verification(device: torch.device) -> dict[str, object]:
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [-0.3, 0.4, 0.8660254]],
        dtype=torch.float64,
        device=device,
    )
    normals = normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True)
    p1 = nested_deterministic_directions(normals, 32, 128.0).reshape(3, 32, 3)
    p10 = nested_deterministic_directions(normals, 320, 128.0).reshape(
        3, 320, 3
    )
    p100 = nested_deterministic_directions(normals, 3200, 128.0).reshape(
        3, 3200, 3
    )
    p1_p10 = float((p1 - p10[:, :32]).abs().max())
    p1_p100 = float((p1 - p100[:, :32]).abs().max())
    p10_p100 = float((p10 - p100[:, :320]).abs().max())
    block = nested_deterministic_direction_block(
        normals[1:], 320, 128.0, emitter_offset=1
    ).reshape(2, 320, 3)
    block_error = float((block - p10[1:]).abs().max())
    if max(p1_p10, p1_p100, p10_p100, block_error) != 0.0:
        raise RuntimeError("nested photon prefix verification failed")
    return {
        "P1_in_P10_max_error": p1_p10,
        "P1_in_P100_max_error": p1_p100,
        "P10_in_P100_max_error": p10_p100,
        "streamed_emitter_block_max_error": block_error,
    }


def _streaming_equivalence(
    geometry: GeometryBundle,
    stream: StreamedTransport,
    config: BunnyExperimentConfig,
) -> dict[str, object]:
    from .bandwidth import _condition_context, _scene_bundle

    scene = _scene_bundle(geometry, 32, config)
    monolithic_context, monolithic_target, _ = _condition_context(
        geometry, scene, "P1_MONOLITHIC", 256, config
    )
    base_image_errors = []
    target_image_errors = []
    owner_exact = []
    direction_errors = []
    for index in range(8):
        monolithic_base = monolithic_context.cells[index]
        streamed_base = stream.base_cells[index]
        owner_exact.append(
            torch.equal(monolithic_base.owner_map, streamed_base.owner_map)
        )
        direction_errors.append(
            float(
                (
                    monolithic_base.fixed_directions
                    - streamed_base.fixed_directions
                )
                .abs()
                .max()
            )
            if streamed_base.fixed_directions.numel()
            else 0.0
        )
        monolithic_image = render_fixed_transport_cell(
            monolithic_base, geometry.reference_points
        ).reshape(-1)
        base_image_errors.append(
            float((monolithic_image - stream.base_images[index]).abs().max())
        )
        target_image_errors.append(
            float(
                (
                    monolithic_target["images"][index]
                    - stream.target_images[index]
                )
                .abs()
                .max()
            )
        )
    camera = stream.cameras[0]
    view_config = MultiviewConfig(
        resolution=(256, 256),
        emitters=config.emitters,
        packets_per_emitter=32,
        parameter_count=config.master_count,
        cone_power=128.0,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )
    shared = project_camera(
        monolithic_context.geometry, camera, scene.base_state, view_config
    )
    rebuilt_scene = _scene_bundle(geometry, 32, config)
    independent = project_camera(
        monolithic_context.geometry,
        camera,
        rebuilt_scene.base_state,
        view_config,
    )
    independent_report = _compare_camera(shared, independent)
    report = {
        "streamed_vs_monolithic_owner_maps_exact": all(owner_exact),
        "streamed_vs_monolithic_base_image_max_error": max(base_image_errors),
        "streamed_vs_monolithic_target_image_max_error": max(
            target_image_errors
        ),
        "streamed_vs_monolithic_direction_max_error": max(direction_errors),
        "shared_vs_independent_one_camera": independent_report,
    }
    report["passed"] = (
        report["streamed_vs_monolithic_owner_maps_exact"]
        and report["streamed_vs_monolithic_base_image_max_error"] < 1e-12
        and report["streamed_vs_monolithic_target_image_max_error"] < 1e-12
        and report["streamed_vs_monolithic_direction_max_error"] < 1e-12
        and independent_report["passed"]
    )
    if not report["passed"]:
        raise RuntimeError("streamed/shared transport equivalence failed")
    del monolithic_context, monolithic_target, scene, rebuilt_scene
    torch.cuda.empty_cache()
    return report


def _condition_summary(
    condition: str,
    views: int,
    packet_level: str,
    diagnostics: dict[str, object],
    fixed_rows: list[dict[str, object]],
    operator: dict[str, object],
    dynamic: dict[str, object],
    uniform: dict[str, object],
    base_chamfer: float,
) -> dict[str, object]:
    dynamic_final = dynamic["rows"][-1]
    uniform_final = uniform["rows"][-1]
    fixed = {int(row["active_k"]): row for row in fixed_rows}
    capture = diagnostics["capture"]
    emitted_millions = int(diagnostics["photons"]) / 1_000_000.0
    captured_millions = int(capture["unique_capture_count"]) / 1_000_000.0
    responsive_count = round(
        float(operator["responsive_candidate_fraction"]) * 1024
    )
    rank = int(operator["numerical_rank_relative_1e-6"])
    stable_rank = float(operator["stable_rank"])
    improvement = base_chamfer - float(dynamic_final["symmetric_chamfer"])
    return {
        "condition": condition,
        "views": views,
        "packet_level": packet_level,
        "packets_per_emitter": PACKET_LEVELS[packet_level],
        "photons": int(diagnostics["photons"]),
        "natural_schedule": dynamic["batch_schedule"],
        "birth_rounds": dynamic["birth_rounds"],
        "mean_batch_size": dynamic["mean_batch_size"],
        "median_batch_size": dynamic["median_batch_size"],
        "first_batch_size": dynamic["first_batch_size"],
        "maximum_batch_size": dynamic["maximum_batch_size"],
        "p_max_reached": dynamic["p_max_reached"],
        "stopping_reasons": dynamic["stopping_reasons"],
        "unique_capture_count": capture["unique_capture_count"],
        "unique_capture_percent": 100.0
        * float(capture["eta_capture_emitted"]),
        "capture_multiplicity_mean": capture[
            "mean_capture_multiplicity_captured"
        ],
        "capture_multiplicity_median": capture[
            "median_capture_multiplicity_captured"
        ],
        "capture_multiplicity_p95": capture[
            "p95_capture_multiplicity_captured"
        ],
        "capture_multiplicity_max": capture["maximum_capture_multiplicity"],
        "responsive_candidate_fraction": operator[
            "responsive_candidate_fraction"
        ],
        "near_null_candidate_fraction": operator[
            "near_null_candidate_fraction"
        ],
        "gram_rank_1e-6": rank,
        "gram_rank_1e-8": operator["numerical_rank_relative_1e-8"],
        "gram_rank_1e-10": operator["numerical_rank_relative_1e-10"],
        "stable_rank": stable_rank,
        "candidate_cosine_median": operator["mutual_column_cosine_median"],
        "candidate_cosine_p95": operator["mutual_column_cosine_p95"],
        "candidate_cosine_p99": operator["mutual_column_cosine_p99"],
        "candidate_cosine_max": operator["mutual_column_cosine_max"],
        "candidate_pair_fraction_cosine_gt_0.05": operator[
            "candidate_pair_fraction_cosine_gt_0.05"
        ],
        "candidate_pair_fraction_cosine_gt_0.10": operator[
            "candidate_pair_fraction_cosine_gt_0.10"
        ],
        "candidate_pair_fraction_cosine_gt_0.20": operator[
            "candidate_pair_fraction_cosine_gt_0.20"
        ],
        "dynamic_chamfer": dynamic_final["symmetric_chamfer"],
        "uniform_chamfer": uniform_final["symmetric_chamfer"],
        "fixed256_chamfer": fixed[256]["symmetric_chamfer"],
        "fixed512_chamfer": fixed[512]["symmetric_chamfer"],
        "fixed1024_chamfer": fixed[1024]["symmetric_chamfer"],
        "dynamic_p2s_mean": dynamic_final["point_to_surface_mean"],
        "dynamic_p95": dynamic_final["point_to_surface_p95"],
        "dynamic_surface_rms": dynamic_final["surface_rms"],
        "dynamic_normal_error": dynamic_final["normal_error"],
        "uniform_p95": uniform_final["point_to_surface_p95"],
        "joint_image_loss": dynamic_final["joint_equal_view_objective"],
        "joint_normalized_multiview_mse": dynamic_final[
            "joint_normalized_multiview_mse"
        ],
        "worst_view_normalized_mse": dynamic["final_view_metrics"][
            "worst_view_normalized_mse"
        ],
        "variance_per_view_normalized_mse": dynamic["final_view_metrics"][
            "variance_per_view_normalized_mse"
        ],
        "candidate_scoring_seconds": dynamic[
            "cumulative_scoring_seconds"
        ],
        "candidate_scoring_seconds_excluding_batch_gram": float(
            dynamic["cumulative_scoring_seconds"]
        )
        - float(dynamic["cumulative_batch_gram_seconds"]),
        "batch_gram_seconds": dynamic["cumulative_batch_gram_seconds"],
        "optimization_seconds": dynamic["cumulative_optimization_seconds"],
        "geometry_evaluation_seconds": sum(
            float(row.get("geometry_evaluation_seconds", 0.0))
            for row in dynamic["rows"]
        ),
        "fixed_reference_geometry_evaluation_seconds": sum(
            float(row["geometry_evaluation_seconds"]) for row in fixed_rows
        ),
        "scene_transport_seconds": float(
            diagnostics["current_scene_transport_seconds"]
        )
        + float(diagnostics["target_scene_transport_seconds"]),
        "detector_transport_seconds": diagnostics[
            "detector_transport_seconds"
        ],
        "dynamic_total_seconds": dynamic["total_runtime_seconds"],
        "uniform_total_seconds": uniform["total_runtime_seconds"],
        "peak_allocated_mib": max(
            float(dynamic["peak_allocated_mib"]),
            float(uniform["peak_allocated_mib"]),
            max(float(row["peak_allocated_mib"]) for row in fixed_rows),
            float(diagnostics["stream_peak_allocated_mib"]),
        ),
        "maximum_birth_geometry_jump": dynamic[
            "maximum_birth_geometry_jump"
        ],
        "maximum_batch_pairwise_cosine": dynamic[
            "maximum_pairwise_correlation"
        ],
        "maximum_batch_rho_off": dynamic["maximum_rho_off"],
        "predicted_realized_pearson": dynamic[
            "predicted_joint_calibration"
        ]["pearson"],
        "predicted_realized_spearman": dynamic[
            "predicted_joint_calibration"
        ]["spearman"],
        "predicted_realized_calibration_valid": dynamic[
            "predicted_realized_calibration_valid"
        ],
        "mean_joint_to_independent_ratio": statistics.mean(
            float(batch["joint_to_independent_ratio"])
            for batch in dynamic["batches"]
        ),
        "top25_birth_fraction": dynamic["top25_birth_fraction"],
        "top10_birth_fraction": dynamic["top10_birth_fraction"],
        "birth_regions": _birth_regions(dynamic),
        "photon_efficiency": {
            "unique_captured_per_emitted": capture["eta_capture_emitted"],
            "responsive_candidates_per_million_emitted": responsive_count
            / emitted_millions,
            "gram_rank_per_million_emitted": rank / emitted_millions,
            "stable_rank_per_million_emitted": stable_rank / emitted_millions,
            "geometry_improvement_per_million_emitted": improvement
            / emitted_millions,
            "responsive_candidates_per_million_captured": responsive_count
            / max(captured_millions, 1e-30),
            "gram_rank_per_million_captured": rank
            / max(captured_millions, 1e-30),
            "stable_rank_per_million_captured": stable_rank
            / max(captured_millions, 1e-30),
            "geometry_improvement_per_million_captured": improvement
            / max(captured_millions, 1e-30),
        },
    }


def _spearman(left: list[float], right: list[float]) -> float:
    return _correlation(
        torch.tensor(left, dtype=torch.float64),
        torch.tensor(right, dtype=torch.float64),
        True,
    )


def _correlation_report(summaries: list[dict[str, object]]) -> dict[str, float]:
    eta = [float(item["unique_capture_percent"]) for item in summaries]
    mean_batch = [float(item["mean_batch_size"]) for item in summaries]
    first = [float(item["first_batch_size"]) for item in summaries]
    near_null = [float(item["near_null_candidate_fraction"]) for item in summaries]
    responsive = [
        float(item["responsive_candidate_fraction"]) for item in summaries
    ]
    photons = [math.log10(float(item["photons"])) for item in summaries]
    views = [float(item["views"]) for item in summaries]
    return {
        "eta_capture_vs_mean_batch": _spearman(eta, mean_batch),
        "eta_capture_vs_first_batch": _spearman(eta, first),
        "near_null_vs_mean_batch": _spearman(near_null, mean_batch),
        "responsive_vs_mean_batch": _spearman(responsive, mean_batch),
        "photon_count_vs_mean_batch": _spearman(photons, mean_batch),
        "view_count_vs_eta_capture": _spearman(views, eta),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    keys = sorted({key for row in flattened for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


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


def _plots(
    directory: Path,
    summaries: list[dict[str, object]],
    trajectories: dict[str, dict[str, object]],
    capture_curve: list[dict[str, object]],
    geometry: GeometryBundle,
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    schedule_conditions = (
        "V8_P1",
        "V8_P10",
        "V8_P100",
        "V40_P10",
        "V40_P100",
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for condition in schedule_conditions:
        schedule = trajectories[condition]["dynamic"]["batch_schedule"]
        axis.step(
            range(1, len(schedule) + 1),
            schedule,
            where="mid",
            marker="o",
            label=condition.replace("_", "/"),
        )
    axis.set_xlabel("birth round")
    axis.set_ylabel("natural batch size")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "v03g_natural_batch_schedule.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    v8 = sorted(
        [item for item in summaries if int(item["views"]) == 8],
        key=lambda item: int(item["photons"]),
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    photons = [int(item["photons"]) for item in v8]
    axis.plot(
        photons,
        [item["mean_batch_size"] for item in v8],
        marker="o",
        label="mean p",
    )
    axis.plot(
        photons,
        [item["first_batch_size"] for item in v8],
        marker="s",
        label="first p",
    )
    axis.set_xscale("log")
    axis.set_xlabel("emitted photons")
    axis.set_ylabel("natural batch size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "v03g_batch_vs_photons.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    axis.plot(
        [item["views"] for item in capture_curve],
        [item["eta_capture_emitted"] for item in capture_curve],
        marker="o",
    )
    axis.set_xlabel("detector views")
    axis.set_ylabel("unique capture fraction")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = directory / "v03g_capture_curve.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    for suffix, group, xlabel in (
        ("photons", v8, "emitted photons"),
        (
            "views",
            sorted(
                [item for item in summaries if item["packet_level"] == "P10"],
                key=lambda item: int(item["views"]),
            ),
            "detector views",
        ),
    ):
        figure, axis = plt.subplots(figsize=(6.8, 4.3))
        x = [
            int(item["photons"] if suffix == "photons" else item["views"])
            for item in group
        ]
        axis.plot(
            x,
            [item["near_null_candidate_fraction"] for item in group],
            marker="o",
        )
        if suffix == "photons":
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("near-null candidate fraction")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = directory / f"v03g_near_null_vs_{suffix}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))

    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    for item in summaries:
        axis.scatter(
            item["unique_capture_percent"],
            item["mean_batch_size"],
            label=str(item["condition"]).replace("_", "/"),
        )
    axis.set_xlabel("unique capture (%)")
    axis.set_ylabel("mean natural batch")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = directory / "v03g_batch_vs_capture.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    for suffix, x_key, xlabel in (
        ("dofs", "active_k", "active DoFs"),
        ("time", "total_runtime_seconds", "wall-clock time (s)"),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.5))
        for axis, condition in zip(
            axes.flat, ("V8_P1", "V8_P10", "V40_P10", "V40_P100")
        ):
            for method, style in (("dynamic", "-"), ("uniform", "--")):
                rows = trajectories[condition][method]["rows"]
                axis.plot(
                    [row[x_key] for row in rows],
                    [row["symmetric_chamfer"] for row in rows],
                    style,
                    marker="o",
                    label=method,
                )
            axis.set_title(condition.replace("_", "/"))
            axis.set_xlabel(xlabel)
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Chamfer")
        axes[1, 0].set_ylabel("Chamfer")
        axes[0, 0].legend()
        figure.tight_layout()
        path = directory / f"v03g_chamfer_vs_{suffix}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))

    labels = [str(item["condition"]).replace("_", "/") for item in summaries]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.bar(x - 0.18, [item["gram_rank_1e-6"] for item in summaries], 0.36)
    stable_axis = axis.twinx()
    stable_axis.bar(
        x + 0.18,
        [item["stable_rank"] for item in summaries],
        0.36,
        color="tab:orange",
    )
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("Gram rank at 1e-6")
    stable_axis.set_ylabel("stable rank")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = directory / "v03g_gram_rank.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    p10 = sorted(
        [item for item in summaries if item["packet_level"] == "P10"],
        key=lambda item: int(item["views"]),
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    axis.plot(
        [item["views"] for item in p10],
        [item["scene_transport_seconds"] for item in p10],
        marker="o",
        label="shared scene (current + target)",
    )
    axis.plot(
        [item["views"] for item in p10],
        [item["detector_transport_seconds"] for item in p10],
        marker="s",
        label="detectors",
    )
    axis.set_xlabel("detector views")
    axis.set_ylabel("transport time (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "v03g_shared_detector_runtime.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(4, 3, figsize=(10.0, 11.0))
    selected_conditions = ("V8_P1", "V8_P10", "V40_P10", "V40_P100")
    for row_index, condition in enumerate(selected_conditions):
        selected = trajectories[condition]["dynamic"]["selected_ids"]
        for column, active_k in enumerate((64, 128, 256)):
            birth_count = active_k - 32
            ids = torch.tensor(
                selected[:birth_count],
                dtype=torch.long,
                device=geometry.layout.centers.device,
            )
            centers = geometry.layout.centers[ids].detach().cpu()
            axis = axes[row_index, column]
            axis.scatter(centers[:, 0], centers[:, 1], s=4, alpha=0.75)
            axis.set_aspect("equal")
            axis.set_title(f"{condition.replace('_', '/')} K={active_k}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    path = directory / "v03g_birth_geography.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _historical_cap_summary(
    artifact_directory: Path,
    current_capture: dict[str, object],
) -> dict[str, object]:
    dynamic_report = json.loads(
        (artifact_directory / "v03f_dynamic_high_bandwidth.json").read_text()
    )
    fixed_report = json.loads(
        (artifact_directory / "v03f_observation_bandwidth.json").read_text()
    )
    historical = dynamic_report["dynamic"]
    final = historical["rows"][-1]
    fixed_condition = next(
        item
        for item in fixed_report["conditions"]
        if float(item["sigma"]) == 2.5 and item["condition"] == "B"
    )
    operator = fixed_condition["diagnostics"]["operator"]
    fixed = {
        int(row["active_k"]): row
        for row in fixed_condition["fixed_rows"]
        if row["start_mode"] == "warm" or int(row["active_k"]) == 256
    }
    capture = current_capture["capture"]
    return {
        "condition": "V8_P1_CAP32_HISTORICAL",
        "views": 8,
        "packet_level": "P1",
        "packets_per_emitter": 32,
        "photons": 262_144,
        "natural_schedule": historical["batch_schedule"],
        "birth_rounds": historical["birth_rounds"],
        "mean_batch_size": historical["mean_batch_size"],
        "median_batch_size": historical["median_batch_size"],
        "first_batch_size": historical["batch_schedule"][0],
        "maximum_batch_size": historical["maximum_batch_size"],
        "p_max_reached": True,
        "stopping_reasons": ["P_MAX_REACHED"] * 6
        + ["TARGET_K_REACHED"],
        "unique_capture_count": capture["unique_capture_count"],
        "unique_capture_percent": 100.0
        * float(capture["eta_capture_emitted"]),
        "capture_multiplicity_mean": capture[
            "mean_capture_multiplicity_captured"
        ],
        "responsive_candidate_fraction": operator[
            "responsive_candidate_fraction"
        ],
        "near_null_candidate_fraction": operator[
            "near_null_candidate_fraction"
        ],
        "gram_rank_1e-6": operator["numerical_rank_relative_1e-6"],
        "gram_rank_1e-8": operator["numerical_rank_relative_1e-8"],
        "gram_rank_1e-10": operator["numerical_rank_relative_1e-10"],
        "stable_rank": operator["stable_rank"],
        "candidate_cosine_median": operator[
            "mutual_column_cosine_median"
        ],
        "candidate_cosine_max": operator["mutual_column_cosine_max"],
        "dynamic_chamfer": final["symmetric_chamfer"],
        "dynamic_p2s_mean": final["point_to_surface_mean"],
        "dynamic_surface_rms": final["surface_rms"],
        "dynamic_normal_error": final["normal_error"],
        "fixed256_chamfer": fixed[256]["symmetric_chamfer"],
        "fixed1024_chamfer": fixed[1024]["symmetric_chamfer"],
        "dynamic_p95": final["point_to_surface_p95"],
        "joint_image_loss": final["raw_image_loss"],
        "candidate_scoring_seconds": historical[
            "cumulative_scoring_seconds"
        ],
        "optimization_seconds": historical[
            "cumulative_optimization_seconds"
        ],
        "dynamic_total_seconds": historical["total_runtime_seconds"],
        "peak_allocated_mib": historical["peak_allocated_mib"],
        "maximum_birth_geometry_jump": historical[
            "maximum_birth_geometry_jump"
        ],
        "maximum_batch_pairwise_cosine": historical[
            "maximum_pairwise_correlation"
        ],
        "maximum_batch_rho_off": historical["maximum_rho_off"],
        "historical_source": "v0.3f byte-identical artifact",
    }


def _condition_is_stable(data: dict[str, object]) -> bool:
    dynamic = data["dynamic"]
    return (
        float(dynamic["maximum_birth_geometry_jump"]) < 1e-12
        and float(dynamic["maximum_pairwise_correlation"]) <= 0.10 + 1e-12
        and float(dynamic["maximum_rho_off"]) <= 0.15 + 1e-12
        and not bool(dynamic["p_max_reached"])
        and all(
            int(row["root_failures"]) == 0
            and int(row["cg_failures"]) == 0
            and int(row["nonfinite_candidate_score_count"]) == 0
            for row in dynamic["rows"]
        )
    )


def run_natural_multiview_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
) -> dict[str, object]:
    """Run the staged v0.3g local-CUDA experiment."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    experiment_started = time.perf_counter()
    prepared = prepare_stanford_bunny(mesh_path)
    config = BunnyExperimentConfig(
        checkpoints=(32, 64, 96, 128, 160, 192, 224, 256),
        fixed_levels=FIXED_LEVELS,
        views=8,
        resolution=256,
        emitters=8192,
        packets_per_emitter=32,
        random_seeds=(),
        oracle_subset_size=0,
    )
    canonical = _canonical_emitters(prepared, config.emitters, device)
    geometry = _geometry_bundle(prepared, canonical, 2.5, config)
    nested_directions = _nested_direction_verification(device)
    cameras_8 = nested_bunny_cameras(256, device, 8)
    cameras_40 = nested_bunny_cameras(256, device, 40)
    camera_prefix_error = max(
        float((left.center - right.center).abs().max())
        for left, right in zip(cameras_8, cameras_40[:8])
    )
    if camera_prefix_error != 0.0:
        raise RuntimeError("V8 is not an exact V40 camera prefix")
    streams = {
        "P1": _stream_transport(geometry, cameras_8, "P1", config),
    }
    print(
        json.dumps(
            {
                "phase": "stream_P1",
                "chunks": streams["P1"].chunks,
                "capture_V8": streams["P1"].capture_by_prefix[8][
                    "eta_capture_emitted"
                ],
            }
        ),
        flush=True,
    )
    streaming_equivalence = _streaming_equivalence(
        geometry, streams["P1"], config
    )
    print(json.dumps({"phase": "equivalence", **streaming_equivalence}), flush=True)
    streams["P10"] = _stream_transport(
        geometry, cameras_40, "P10", config
    )
    print(
        json.dumps(
            {
                "phase": "stream_P10",
                "chunks": streams["P10"].chunks,
                "capture_V40": streams["P10"].capture_by_prefix[40][
                    "eta_capture_emitted"
                ],
            }
        ),
        flush=True,
    )
    streams["P100"] = _stream_transport(
        geometry, cameras_40, "P100", config
    )
    print(
        json.dumps(
            {
                "phase": "stream_P100",
                "chunks": streams["P100"].chunks,
                "capture_V40": streams["P100"].capture_by_prefix[40][
                    "eta_capture_emitted"
                ],
            }
        ),
        flush=True,
    )
    condition_data: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    for condition, views, packet_level in MAJOR_CONDITIONS:
        context, target, diagnostics = _prefix_context(
            geometry,
            streams[packet_level],
            views,
            config,
            condition,
        )
        fixed_rows, fixed_diagnostics, _ = _fixed_warm_references(
            context,
            target,
            condition,
            views,
            streams[packet_level].emitted,
        )
        dynamic, uniform = _trajectory_report(
            context,
            target,
            condition,
            views,
            streams[packet_level].emitted,
        )
        operator = fixed_diagnostics["operator"]
        base_chamfer = float(
            geometry.evaluator(geometry.reference_points)[
                "symmetric_chamfer"
            ]
        )
        summary = _condition_summary(
            condition,
            views,
            packet_level,
            diagnostics,
            fixed_rows,
            operator,
            dynamic,
            uniform,
            base_chamfer,
        )
        condition_data[condition] = {
            "diagnostics": diagnostics,
            "fixed_rows": fixed_rows,
            "fixed_diagnostics": fixed_diagnostics,
            "dynamic": dynamic,
            "uniform": uniform,
            "summary": summary,
            "numerically_stable": _condition_is_stable(
                {"dynamic": dynamic}
            ),
        }
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "condition": condition,
                    "schedule": dynamic["batch_schedule"],
                    "capture": diagnostics["capture"][
                        "eta_capture_emitted"
                    ],
                    "near_null": operator[
                        "near_null_candidate_fraction"
                    ],
                    "stable_rank": operator["stable_rank"],
                    "dynamic_chamfer": summary["dynamic_chamfer"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del context, target
        torch.cuda.empty_cache()
    by_name = {item["condition"]: item for item in summaries}
    historical = _historical_cap_summary(
        artifact_directory,
        condition_data["V8_P1"]["diagnostics"],
    )
    natural_checks = {
        "one_or_more_batches_above_32": max(
            condition_data["V8_P1"]["dynamic"]["batch_schedule"]
        )
        > 32,
        "numerically_stable_and_weakly_coupled": condition_data["V8_P1"][
            "numerically_stable"
        ],
        "all_five_geometry_metrics_no_worse_than_cap32": all(
            float(by_name["V8_P1"][current])
            <= float(historical[reference]) + 1e-12
            for current, reference in (
                ("dynamic_chamfer", "dynamic_chamfer"),
                ("dynamic_p2s_mean", "dynamic_p2s_mean"),
                ("dynamic_p95", "dynamic_p95"),
                ("dynamic_surface_rms", "dynamic_surface_rms"),
                ("dynamic_normal_error", "dynamic_normal_error"),
            )
        ),
    }
    natural_supported = all(natural_checks.values())
    photon_checks = {
        "P10_or_P100_mean_batch_increases_5_percent": max(
            float(by_name[name]["mean_batch_size"])
            for name in ("V8_P10", "V8_P100")
        )
        >= 1.05 * float(by_name["V8_P1"]["mean_batch_size"]),
        "P10_or_P100_birth_rounds_decrease": min(
            int(by_name[name]["birth_rounds"])
            for name in ("V8_P10", "V8_P100")
        )
        < int(by_name["V8_P1"]["birth_rounds"]),
        "higher_photon_geometry_within_1_percent": all(
            float(by_name[name]["dynamic_chamfer"])
            <= 1.01 * float(by_name["V8_P1"]["dynamic_chamfer"])
            for name in ("V8_P10", "V8_P100")
        ),
    }
    photon_supported = (
        photon_checks["P10_or_P100_mean_batch_increases_5_percent"]
        or photon_checks["P10_or_P100_birth_rounds_decrease"]
    ) and photon_checks["higher_photon_geometry_within_1_percent"]
    capture_checks = {
        "V40_minus_V8_capture_at_P10_at_least_1_percentage_point": float(
            by_name["V40_P10"]["unique_capture_percent"]
        )
        >= float(by_name["V8_P10"]["unique_capture_percent"]) + 1.0
    }
    capture_supported = all(capture_checks.values())
    view_observability_checks = {
        "near_null_drops_0p5_percentage_point": float(
            by_name["V40_P10"]["near_null_candidate_fraction"]
        )
        <= float(by_name["V8_P10"]["near_null_candidate_fraction"])
        - 0.005,
        "gram_rank_increases": int(by_name["V40_P10"]["gram_rank_1e-6"])
        > int(by_name["V8_P10"]["gram_rank_1e-6"]),
        "geometry_improves_0p1_percent": float(
            by_name["V40_P10"]["dynamic_chamfer"]
        )
        <= 0.999 * float(by_name["V8_P10"]["dynamic_chamfer"]),
        "p95_cosine_drops_10_percent": float(
            by_name["V40_P10"]["candidate_cosine_p95"]
        )
        <= 0.9 * float(by_name["V8_P10"]["candidate_cosine_p95"]),
    }
    observability_supported = any(view_observability_checks.values())
    view_birth_checks = {
        "mean_batch_increases_5_percent": float(
            by_name["V40_P10"]["mean_batch_size"]
        )
        >= 1.05 * float(by_name["V8_P10"]["mean_batch_size"]),
        "birth_rounds_decrease": int(by_name["V40_P10"]["birth_rounds"])
        < int(by_name["V8_P10"]["birth_rounds"]),
        "geometry_within_1_percent": float(
            by_name["V40_P10"]["dynamic_chamfer"]
        )
        <= 1.01 * float(by_name["V8_P10"]["dynamic_chamfer"]),
    }
    view_birth_supported = (
        view_birth_checks["mean_batch_increases_5_percent"]
        or view_birth_checks["birth_rounds_decrease"]
    ) and view_birth_checks["geometry_within_1_percent"]
    unified_supported = (
        capture_supported and observability_supported and view_birth_supported
    )
    verdicts = {
        "natural_batch": (
            "NATURAL_BATCH_GROWTH_SUPPORTED"
            if natural_supported
            else "NATURAL_BATCH_GROWTH_NOT_SUPPORTED"
        ),
        "photon_evidence": (
            "PHOTON_EVIDENCE_EXPANDS_SAFE_BIRTH_SUPPORTED"
            if photon_supported
            else "PHOTON_EVIDENCE_EXPANDS_SAFE_BIRTH_NOT_SUPPORTED"
        ),
        "multiview_capture": (
            "MULTIVIEW_CAPTURE_UTILIZATION_SUPPORTED"
            if capture_supported
            else "MULTIVIEW_CAPTURE_UTILIZATION_NOT_SUPPORTED"
        ),
        "multiview_observability": (
            "MULTIVIEW_OBSERVABILITY_GAIN_SUPPORTED"
            if observability_supported
            else "MULTIVIEW_OBSERVABILITY_GAIN_NOT_SUPPORTED"
        ),
        "multiview_birth": (
            "MULTIVIEW_EVIDENCE_EXPANDS_SAFE_BIRTH_SUPPORTED"
            if view_birth_supported
            else "MULTIVIEW_EVIDENCE_EXPANDS_SAFE_BIRTH_NOT_SUPPORTED"
        ),
        "unified_chain": (
            "SHARED_MULTIVIEW_TO_ADAPTIVE_GROWTH_CHAIN_SUPPORTED"
            if unified_supported
            else "SHARED_MULTIVIEW_TO_ADAPTIVE_GROWTH_CHAIN_NOT_SUPPORTED"
        ),
    }
    checks = {
        "natural_batch": natural_checks,
        "photon_evidence": photon_checks,
        "multiview_capture": capture_checks,
        "multiview_observability": view_observability_checks,
        "multiview_birth": view_birth_checks,
    }
    correlations = _correlation_report(summaries)
    capture_curve = [
        {
            "views": views,
            "packet_level": "P10",
            "photons": streams["P10"].emitted,
            **streams["P10"].capture_by_prefix[views],
        }
        for views in CAPTURE_CURVE_VIEWS
    ]
    trajectories_for_plots = {
        name: {
            "dynamic": data["dynamic"],
            "uniform": data["uniform"],
        }
        for name, data in condition_data.items()
    }
    figures = _plots(
        figure_directory,
        summaries,
        trajectories_for_plots,
        capture_curve,
        geometry,
    )
    common = {
        "environment": cuda_environment(),
        "bunny": prepared.metadata,
        "configuration": {
            "sigma": 2.5,
            "resolution": 256,
            "emitters": 8192,
            "packet_levels": PACKET_LEVELS,
            "view_levels": list(VIEW_LEVELS),
            "master_k": 1024,
            "initial_k": 32,
            "target_k": 256,
            "alpha": 0.10,
            "tau": 0.10,
            "rho_max": 0.15,
            "p_max_raw": "100 * K_current",
            "p_max": (
                "min(100 * K_current, K_target - K_current, "
                "inactive_candidates)"
            ),
            "equal_view_objective": "mean of per-view half-SSE",
            "camera_independent_scene_transport_amortized": True,
        },
        "nested_direction_verification": nested_directions,
        "nested_camera_prefix_max_error": camera_prefix_error,
        "streaming_and_shared_equivalence": streaming_equivalence,
        "optional_low_photon_view_sentinels": (
            "SKIPPED_P100_DOMINATED_TOTAL_RUNTIME"
        ),
        "verdicts": verdicts,
        "verdict_checks": checks,
        "correlations": correlations,
        "figures": figures,
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    natural_names = {"V8_P1", "V8_P10", "V8_P100"}
    natural_report = {
        **common,
        "historical_cap32": historical,
        "condition_summaries": [
            item for item in summaries if item["condition"] in natural_names
        ],
        "conditions": {
            name: condition_data[name] for name in sorted(natural_names)
        },
    }
    capture_report = {
        **common,
        "capture_curve_P10": capture_curve,
        "condition_capture": {
            name: data["diagnostics"] for name, data in condition_data.items()
        },
    }
    birth_report = {
        **common,
        "condition_summaries": summaries,
        "conditions": condition_data,
        "historical_cap32": historical,
    }
    natural_rows = [historical] + [
        item for item in summaries if item["condition"] in natural_names
    ]
    capture_rows = [
        {
            "row_type": "capture_curve",
            **row,
        }
        for row in capture_curve
    ] + [
        {
            "row_type": "major_condition",
            "condition": item["condition"],
            "views": item["views"],
            "photons": item["photons"],
            "unique_capture_count": item["unique_capture_count"],
            "unique_capture_percent": item["unique_capture_percent"],
            "capture_multiplicity_mean": item["capture_multiplicity_mean"],
            "capture_multiplicity_median": item[
                "capture_multiplicity_median"
            ],
            "capture_multiplicity_p95": item["capture_multiplicity_p95"],
            "capture_multiplicity_max": item["capture_multiplicity_max"],
        }
        for item in summaries
    ]
    artifact_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(artifact_directory / "v03g_natural_batch.csv", natural_rows)
    _write_json(
        artifact_directory / "v03g_natural_batch.json", natural_report
    )
    _write_csv(
        artifact_directory / "v03g_multiview_capture.csv", capture_rows
    )
    _write_json(
        artifact_directory / "v03g_multiview_capture.json", capture_report
    )
    _write_csv(artifact_directory / "v03g_multiview_birth.csv", summaries)
    _write_json(
        artifact_directory / "v03g_multiview_birth.json", birth_report
    )
    return {
        "verdicts": verdicts,
        "summaries": summaries,
        "capture_curve": capture_curve,
        "figures": figures,
    }
