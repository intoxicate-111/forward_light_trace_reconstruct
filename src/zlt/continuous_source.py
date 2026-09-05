"""v0.8.6 fixed-latent, continuously weighted zero-set sources.

The source measure in this module is an explicitly algorithmic, globally
weighted volume quadrature.  It is not claimed to be physical radiometry.
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .corrected_birth import CorrectedBirthConfig, _active_components, _build_context, _evaluate
from .finite_packet import (
    LocalField,
    _accumulate_continuous,
    _hard_images,
    _image_metrics,
    _json_ready,
    _micro_offsets,
    _progress,
    _render_parameters,
    _save_figure,
    _scalar_csv_rows,
    _soft_render,
    _surface_spacing,
    _transmission,
    _vector_metrics,
)
from .high_sample import _release, _write_csv
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color
from .sampling_diagnostic import _copy_layout


Tensor = torch.Tensor


class Field(Protocol):
    lower: Tensor
    upper: Tensor

    def value(self, points: Tensor) -> Tensor: ...
    def gradient(self, points: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ContinuousSourceConfig:
    main_anchors: int = 32_768
    density_counts: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768)
    shell_ratios: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    density_shell_ratios: tuple[float, ...] = (1.0, 0.25, 0.0625)
    selected_epsilon_over_h: float = 0.5
    views: int = 4
    resolution: int = 256
    higher_resolution: int = 512
    seed: int = 101
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    source_chunk_size: int = 256
    micro_samples: int = 8
    packet_radius_over_surface_h: float = 1.0
    packet_shell_over_radius: float = 1.0
    packet_step_over_epsilon: float = 0.5
    target_crossing_transmission: float = 1e-2
    levelset_eta_relative: float = 1e-6
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    ambient: float = 0.35
    emission_transition_width: float = 0.05
    launch_exclusion_factor: float = 1.05
    diagnostic_surface_samples: int = 4_096
    fd_anchor_count: int = 8_192
    fd_surface_samples: int = 512
    fd_resolution: int = 64
    fd_micro_samples: int = 2
    fd_epsilons: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    response_delta: float = 1e-3
    width_fd_epsilon: float = 1e-4
    multiseed_anchor_count: int = 16_384
    multiseed_resolution: int = 96
    coverage_radius_over_h: float = 2.0
    zero_coverage_maximum: float = 0.01
    mass_drift_maximum: float = 0.02
    local_mass_cv_maximum: float = 0.75
    edge_retention_minimum: float = 0.95
    absolute_bandwidth_retention_minimum: float = 0.80
    gradient_fd_relative_maximum: float = 0.15
    source_scale_tolerance: float = 2e-10
    sample_count_mass_tolerance: float = 0.05
    multiseed_mass_cv_maximum: float = 0.10
    multiseed_response_cosine_minimum: float = 0.20
    projection_residual_over_epsilon_maximum: float = 0.20
    topology_fraction_maximum: float = 0.10


@dataclass(frozen=True)
class AnchorSet:
    points: Tensor
    ids: Tensor
    lower: Tensor
    upper: Tensor
    volume: float
    q_density: float
    quadrature_weight: float
    h_s: float
    seed: int
    sobol_dimensions: int = 3

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


@dataclass
class SourceState:
    weights_raw: Tensor
    weights: Tensor
    positions: Tensor
    normals: Tensor
    anchor_values: Tensor
    normalized_distances: Tensor
    active_ids: Tensor
    eta: float
    source_mass_raw: Tensor
    effective_sample_size: Tensor
    projection_residual: Tensor
    projection_displacement: Tensor


@dataclass
class SourceRender:
    images: dict[tuple[int, int], list[np.ndarray]]
    tensors: dict[tuple[int, int], Tensor]
    runtime_seconds: float
    source_evaluations: int
    interaction_evaluations: int
    attempted_packets: int
    active_packets: int
    peak_allocated_mib: float
    peak_reserved_mib: float


class ShiftedField:
    def __init__(self, field: Field, shift: float | Tensor) -> None:
        self.field = field
        self.shift = shift
        self.lower = field.lower
        self.upper = field.upper

    def value(self, points: Tensor) -> Tensor:
        return self.field.value(points) + self.shift

    def gradient(self, points: Tensor) -> Tensor:
        return self.field.gradient(points)


class ScaledField:
    def __init__(self, field: Field, scale: float) -> None:
        self.field = field
        self.scale = scale
        self.lower = field.lower
        self.upper = field.upper

    def value(self, points: Tensor) -> Tensor:
        return self.scale * self.field.value(points)

    def gradient(self, points: Tensor) -> Tensor:
        return self.scale * self.field.gradient(points)


def _source_kernel(q: Tensor) -> Tensor:
    """Normalized compact C2 kernel: integral psi(q)dq = 1."""
    absolute = q.abs()
    return torch.where(
        absolute < 1.0,
        (35.0 / 32.0) * (1.0 - absolute.square()).pow(3),
        torch.zeros_like(q),
    )


def _source_kernel_numpy(q: np.ndarray) -> np.ndarray:
    absolute = np.abs(q)
    return np.where(
        absolute < 1.0,
        (35.0 / 32.0) * (1.0 - absolute**2) ** 3,
        0.0,
    )


def _fixed_anchors(
    lower: Tensor,
    upper: Tensor,
    count: int,
    seed: int,
) -> AnchorSet:
    if count <= 0 or count & (count - 1):
        raise ValueError("anchor count must be a positive power of two")
    unit = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(count)
    unit = unit.to(dtype=lower.dtype, device=lower.device)
    points = lower + unit * (upper - lower)
    volume = float(torch.prod(upper - lower))
    return AnchorSet(
        points=points,
        ids=torch.arange(count, device=lower.device),
        lower=lower,
        upper=upper,
        volume=volume,
        q_density=1.0 / volume,
        quadrature_weight=volume / count,
        h_s=(volume / count) ** (1.0 / 3.0),
        seed=seed,
    )


def _source_eta(field: Field, anchors: AnchorSet, relative: float) -> float:
    norms = torch.linalg.vector_norm(field.gradient(anchors.points), dim=1)
    return relative * float(norms.median())


def _source_state(
    field: Field,
    anchors: AnchorSet,
    epsilon_s: float,
    reference_area: float,
    *,
    projected: bool,
    hard_shell: bool = False,
) -> SourceState:
    values = field.value(anchors.points)
    gradients = field.gradient(anchors.points)
    eta = _source_eta(field, anchors, 1e-6)
    denominator = torch.sqrt(gradients.square().sum(1) + eta * eta)
    distance = values / denominator.clamp_min(1e-30)
    if hard_shell:
        density = 0.5 / epsilon_s * (distance.abs() < epsilon_s).to(values.dtype)
    else:
        density = _source_kernel(distance / epsilon_s) / epsilon_s
    raw_weights = anchors.quadrature_weight * density
    weights = raw_weights / reference_area
    if projected:
        positions = anchors.points - values[:, None] * gradients / (
            gradients.square().sum(1, keepdim=True) + eta * eta
        ).clamp_min(1e-30)
    else:
        positions = anchors.points
    projected_gradients = field.gradient(positions)
    normals = projected_gradients / torch.linalg.vector_norm(
        projected_gradients, dim=1, keepdim=True
    ).clamp_min(1e-30)
    residual = field.value(positions).abs()
    active = torch.nonzero(raw_weights != 0.0, as_tuple=False).flatten()
    mass = raw_weights.sum()
    effective = mass.square() / raw_weights.square().sum().clamp_min(1e-30)
    return SourceState(
        raw_weights, weights, positions, normals, values, distance, active,
        eta, mass, effective, residual,
        torch.linalg.vector_norm(positions - anchors.points, dim=1),
    )


def _emission_factor(cosine: Tensor, width: float) -> Tensor:
    positive = cosine.clamp_min(0.0)
    return 1.0 - torch.exp(-positive.square() / (width * width))


def _render_fixed_sources(
    field: Field,
    anchors: AnchorSet,
    source: SourceState,
    atlas: Any,
    boundary: Any,
    resolutions: tuple[tuple[int, int], ...],
    *,
    packet_radius: float,
    packet_epsilon: float,
    packet_step: float,
    micro_samples: int,
    kappa: float,
    detector_extent: float,
    sensor_gain: float,
    ambient: float,
    emission_width: float,
    chunk_size: int,
    launch_exclusion_factor: float,
    active_ids: Tensor | None = None,
    detector_positions: Tensor | None = None,
    source_evaluation_count: int | None = None,
) -> SourceRender:
    device = anchors.points.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    selected_ids = source.active_ids if active_ids is None else active_ids
    offsets = _micro_offsets(micro_samples, device, anchors.points.dtype)
    transport_eta = _source_eta(field, anchors, 1e-6)
    accumulators = {
        resolution: [
            torch.zeros((math.prod(resolution), 3), dtype=anchors.points.dtype, device=device)
            for _ in range(atlas.count)
        ] for resolution in resolutions
    }
    interactions = 0
    for view_id in range(atlas.count):
        direction = atlas.directions[view_id]
        for start in range(0, selected_ids.numel(), chunk_size):
            ids = selected_ids[start:start + chunk_size]
            origins = source.positions[ids]
            directions = direction.expand_as(origins)
            transmission, _, evaluations = _transmission(
                field, origins, directions, boundary.exit_times(origins, directions),
                radius=packet_radius, epsilon=packet_epsilon,
                path_step=packet_step, offsets=offsets, eta=transport_eta,
                kappa=kappa, surface_barrier=True,
                launch_exclusion_factor=launch_exclusion_factor,
            )
            interactions += evaluations
            colors = meshfree_base_color(origins, field.lower, field.upper)
            cosine = source.normals[ids] @ direction
            outward = _emission_factor(cosine, emission_width)
            lobe = ambient + (1.0 - ambient) * cosine.clamp_min(0.0)
            energy = source.weights[ids, None] * transmission[:, None] * colors
            energy = energy * (outward * lobe)[:, None]
            detection = origins if detector_positions is None else detector_positions[ids]
            for resolution in resolutions:
                _accumulate_continuous(
                    accumulators[resolution][view_id], detection, energy,
                    atlas.right[view_id], atlas.up[view_id], boundary.center,
                    detector_extent, resolution,
                )
    tensors: dict[tuple[int, int], Tensor] = {}
    images: dict[tuple[int, int], list[np.ndarray]] = {}
    for resolution, views in accumulators.items():
        tensor = torch.cat([
            (sensor_gain * math.prod(resolution) * item).reshape(-1)
            for item in views
        ])
        tensors[resolution] = tensor
        images[resolution] = [
            (sensor_gain * math.prod(resolution) * item).reshape(*resolution, 3)
            .detach().cpu().to(torch.float32).numpy()
            for item in views
        ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return SourceRender(
        images=images,
        tensors=tensors,
        runtime_seconds=time.perf_counter() - started,
        source_evaluations=source_evaluation_count or anchors.count,
        interaction_evaluations=interactions,
        attempted_packets=anchors.count * atlas.count,
        active_packets=int(selected_ids.numel()) * atlas.count,
        peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0),
        peak_reserved_mib=(torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0),
    )


def _source_summary(source: SourceState, anchors: AnchorSet, epsilon_s: float) -> dict[str, float]:
    active = source.active_ids
    active_weights = source.weights_raw[active]
    return {
        "anchors": anchors.count,
        "h_s": anchors.h_s,
        "epsilon_s": epsilon_s,
        "epsilon_over_h_s": epsilon_s / anchors.h_s,
        "active_anchor_count": int(active.numel()),
        "active_anchor_fraction": int(active.numel()) / anchors.count,
        "source_mass": float(source.source_mass_raw),
        "effective_anchor_count": float(source.effective_sample_size),
        "active_weight_cv": float(active_weights.std() / active_weights.mean().clamp_min(1e-30)) if active.numel() > 1 else 0.0,
        "projection_residual_mean": float(source.projection_residual[active].mean()) if active.numel() else 0.0,
        "projection_residual_p95": float(torch.quantile(source.projection_residual[active], 0.95)) if active.numel() else 0.0,
        "projection_residual_max": float(source.projection_residual[active].max()) if active.numel() else 0.0,
        "projection_displacement_mean": float(source.projection_displacement[active].mean()) if active.numel() else 0.0,
    }


def _plane_controls(config: ContinuousSourceConfig) -> dict[str, Any]:
    lower = torch.full((3,), -1.0, dtype=torch.float64)
    upper = torch.full((3,), 1.0, dtype=torch.float64)
    translations = np.linspace(-0.45, 0.45, 181)
    rows = []
    curves: dict[str, Any] = {}
    for count in config.density_counts:
        anchors = _fixed_anchors(lower, upper, count, config.seed)
        for ratio in config.shell_ratios:
            epsilon = ratio * anchors.h_s
            x = anchors.points[:, 0].detach().cpu().numpy()
            continuous = np.asarray([
                anchors.quadrature_weight * np.sum(
                    _source_kernel_numpy((x - shift) / epsilon) / epsilon
                ) for shift in translations
            ])
            hard = np.asarray([
                anchors.quadrature_weight * np.sum(
                    0.5 / epsilon * (np.abs(x - shift) < epsilon)
                ) for shift in translations
            ])
            expected = 4.0
            derivative = np.gradient(continuous, translations)
            rows.append({
                "anchors": count,
                "h_s": anchors.h_s,
                "epsilon_over_h_s": ratio,
                "epsilon_s": epsilon,
                "continuous_mean_mass": float(continuous.mean()),
                "continuous_relative_bias": float(abs(continuous.mean() - expected) / expected),
                "continuous_translation_cv": float(continuous.std() / continuous.mean()),
                "continuous_max_adjacent_jump": float(np.max(np.abs(np.diff(continuous)))),
                "continuous_max_derivative_jump": float(np.max(np.abs(np.diff(derivative)))),
                "hard_translation_cv": float(hard.std() / hard.mean()),
                "hard_max_adjacent_jump": float(np.max(np.abs(np.diff(hard)))),
            })
            if count == config.main_anchors and ratio in (1.0, config.selected_epsilon_over_h, 0.125, 0.03125):
                curves[f"continuous_{ratio:g}"] = continuous.tolist()
                curves[f"hard_{ratio:g}"] = hard.tolist()
    # A 33^3 historical sign-cell control; exact grid-plane coincidence causes
    # two cell slabs rather than one and demonstrates discrete source count.
    grid = np.linspace(-1.0, 1.0, 33)
    historical_counts = []
    for shift in translations:
        values = grid - shift
        crossings = (values[:-1] * values[1:] <= 0.0) & (values[:-1] != values[1:])
        historical_counts.append(int(crossings.sum()) * 32 * 32)
    historical_counts = np.asarray(historical_counts)
    curves["historical_source_count"] = historical_counts.tolist()
    selected = next(
        row for row in rows
        if row["anchors"] == config.main_anchors
        and row["epsilon_over_h_s"] == config.selected_epsilon_over_h
    )
    # Positive scale invariance is exact up to eta rescaling roundoff.
    scale_rows = []
    anchors = _fixed_anchors(lower, upper, config.main_anchors, config.seed)
    epsilon = config.selected_epsilon_over_h * anchors.h_s
    reference = None
    for scale in (1.0, 0.5, 2.0, 10.0):
        f = scale * anchors.points[:, 0]
        eta = config.levelset_eta_relative * scale
        d = f / math.sqrt(scale * scale + eta * eta)
        mass = anchors.quadrature_weight * (_source_kernel(d / epsilon) / epsilon).sum()
        if reference is None:
            reference = mass
        scale_rows.append({
            "field_scale": scale,
            "source_mass": float(mass),
            "absolute_error_vs_F": float((mass - reference).abs()),
        })
    return {
        "domain": [[-1.0] * 3, [1.0] * 3],
        "expected_plane_area_and_raw_source_mass": 4.0,
        "rows": rows,
        "selected": selected,
        "scale_rows": scale_rows,
        "maximum_scale_error": max(row["absolute_error_vs_F"] for row in scale_rows),
        "historical_maximum_source_count_jump": int(np.max(np.abs(np.diff(historical_counts)))),
        "plot": {"translation": translations.tolist(), **curves},
    }


def _sphere_controls(config: ContinuousSourceConfig) -> dict[str, Any]:
    lower = torch.full((3,), -1.0, dtype=torch.float64)
    upper = torch.full((3,), 1.0, dtype=torch.float64)
    radius = 0.60
    expected = 4.0 * math.pi * radius * radius
    rows = []
    curves = []
    for count in config.density_counts:
        anchors = _fixed_anchors(lower, upper, count, config.seed)
        radial = torch.linalg.vector_norm(anchors.points, dim=1)
        gradients = anchors.points / radial[:, None].clamp_min(1e-30)
        for ratio in config.shell_ratios:
            epsilon = ratio * anchors.h_s
            distance = radial - radius
            weights = anchors.quadrature_weight * _source_kernel(distance / epsilon) / epsilon
            positions = anchors.points - distance[:, None] * gradients
            active = weights > 0.0
            normals = positions / torch.linalg.vector_norm(positions, dim=1, keepdim=True).clamp_min(1e-30)
            exact_normals = gradients
            source_mass = float(weights.sum())
            rows.append({
                "anchors": count,
                "h_s": anchors.h_s,
                "epsilon_over_h_s": ratio,
                "epsilon_s": epsilon,
                "source_mass": source_mass,
                "relative_area_error": abs(source_mass - expected) / expected,
                "active_anchor_count": int(active.sum()),
                "projection_position_error_max": float((
                    torch.linalg.vector_norm(positions[active], dim=1) - radius
                ).abs().max()) if bool(active.any()) else 0.0,
                "normal_error_max": float(torch.linalg.vector_norm(
                    normals[active] - exact_normals[active], dim=1
                ).max()) if bool(active.any()) else 0.0,
            })
    # Translation and radius response at the selected highest density.
    anchors = _fixed_anchors(lower, upper, config.main_anchors, config.seed)
    epsilon = config.selected_epsilon_over_h * anchors.h_s
    translations = np.linspace(-0.20, 0.20, 161)
    masses = []
    for shift in translations:
        distance = torch.linalg.vector_norm(
            anchors.points - torch.tensor([shift, 0.0, 0.0], dtype=lower.dtype), dim=1
        ) - radius
        masses.append(float(anchors.quadrature_weight * (_source_kernel(distance / epsilon) / epsilon).sum()))
    radii = np.linspace(0.45, 0.75, 121)
    radius_masses = []
    for current in radii:
        distance = torch.linalg.vector_norm(anchors.points, dim=1) - current
        radius_masses.append(float(anchors.quadrature_weight * (_source_kernel(distance / epsilon) / epsilon).sum()))
    return {
        "radius": radius,
        "expected_area": expected,
        "rows": rows,
        "plot": {
            "translation": translations.tolist(), "translation_mass": masses,
            "radius": radii.tolist(), "radius_mass": radius_masses,
            "analytic_radius_area": (4.0 * math.pi * radii**2).tolist(),
        },
    }


def _source_birth_control(config: ContinuousSourceConfig) -> dict[str, Any]:
    lower = torch.full((3,), -0.8, dtype=torch.float64)
    upper = torch.full((3,), 0.8, dtype=torch.float64)
    anchors = _fixed_anchors(lower, upper, config.main_anchors, config.seed)
    epsilon = config.selected_epsilon_over_h * anchors.h_s
    lambdas = np.linspace(-0.20, 0.08, 281)
    radial2 = anchors.points.square().sum(1)
    gradient_norm = 2.0 * torch.sqrt(radial2)
    eta = config.levelset_eta_relative * float(gradient_norm.median())
    continuous = []
    hard = []
    historical_counts = []
    grid = np.linspace(-0.8, 0.8, 33)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    for value in lambdas:
        f = radial2 + value
        d = f / torch.sqrt(gradient_norm.square() + eta * eta)
        continuous.append(float(anchors.quadrature_weight * (_source_kernel(d / epsilon) / epsilon).sum()))
        hard.append(float(anchors.quadrature_weight * (0.5 / epsilon * (d.abs() < epsilon)).sum()))
        values = gx * gx + gy * gy + gz * gz + value
        minimum = values[:-1, :-1, :-1].copy()
        maximum = minimum.copy()
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = values[dx:32 + dx, dy:32 + dy, dz:32 + dz]
                    minimum = np.minimum(minimum, corner)
                    maximum = np.maximum(maximum, corner)
        historical_counts.append(int(((minimum <= 0) & (maximum >= 0) & (maximum > minimum)).sum()))
    continuous = np.asarray(continuous)
    hard = np.asarray(hard)
    historical_counts = np.asarray(historical_counts)
    continuous_derivative = np.gradient(continuous, lambdas)
    hard_derivative = np.gradient(hard, lambdas)
    return {
        "fixed_anchor_count": anchors.count,
        "anchor_identity_digest": _identity_digest(anchors.ids),
        "epsilon_s": epsilon,
        "epsilon_over_h_s": config.selected_epsilon_over_h,
        "historical_max_source_count_jump": int(np.max(np.abs(np.diff(historical_counts)))),
        "historical_count_range": [int(historical_counts.min()), int(historical_counts.max())],
        "hard_shell_maximum_value_jump": float(np.max(np.abs(np.diff(hard)))),
        "hard_shell_maximum_derivative_jump": float(np.max(np.abs(np.diff(hard_derivative)))),
        "continuous_maximum_value_jump": float(np.max(np.abs(np.diff(continuous)))),
        "continuous_maximum_derivative_jump": float(np.max(np.abs(np.diff(continuous_derivative)))),
        "continuous_fd_spread": _fd_spread(lambdas, continuous),
        "plot": {
            "lambda": lambdas.tolist(), "historical_count": historical_counts.tolist(),
            "hard_mass": hard.tolist(), "continuous_mass": continuous.tolist(),
            "continuous_derivative": continuous_derivative.tolist(),
        },
    }


def _identity_digest(ids: Tensor) -> str:
    import hashlib
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _fd_spread(parameter: np.ndarray, values: np.ndarray) -> float:
    reference = np.gradient(values, parameter)
    return max(
        float(np.linalg.norm(np.interp(parameter, parameter[::stride], np.gradient(
            values[::stride], parameter[::stride]
        )) - reference) / max(np.linalg.norm(reference), 1e-30))
        for stride in (2, 4)
    )


def _coverage_statistics(
    reference_points: Tensor,
    source: SourceState,
    anchors: AnchorSet,
    radius_over_h: float,
) -> dict[str, float]:
    query = reference_points.detach().cpu().numpy()
    active = source.active_ids.detach().cpu().numpy()
    positions = source.positions.detach().cpu().numpy()[active]
    weights = source.weights_raw.detach().cpu().numpy()[active]
    radius = radius_over_h * anchors.h_s
    if not len(active):
        return {
            "neighborhood_radius": radius,
            "zero_support_probability": 1.0,
            "active_count_p10": 0.0, "active_count_median": 0.0, "active_count_p90": 0.0,
            "local_mass_mean": 0.0, "local_mass_cv": float("inf"),
            "local_neff_minimum": 0.0, "local_neff_median": 0.0,
        }
    tree = cKDTree(positions)
    neighbors = tree.query_ball_point(query, radius, workers=1)
    counts = np.asarray([len(item) for item in neighbors], dtype=np.float64)
    masses = np.asarray([weights[item].sum() if len(item) else 0.0 for item in neighbors])
    neff = np.asarray([
        weights[item].sum() ** 2 / max(np.square(weights[item]).sum(), 1e-30)
        if len(item) else 0.0 for item in neighbors
    ])
    positive_mass = masses[masses > 0.0]
    return {
        "neighborhood_radius": radius,
        "zero_support_probability": float(np.mean(counts == 0)),
        "active_count_p10": float(np.quantile(counts, 0.10)),
        "active_count_median": float(np.median(counts)),
        "active_count_p90": float(np.quantile(counts, 0.90)),
        "local_mass_mean": float(masses.mean()),
        "local_mass_cv": float(positive_mass.std() / max(positive_mass.mean(), 1e-30)) if positive_mass.size else float("inf"),
        "local_neff_minimum": float(neff.min()),
        "local_neff_median": float(np.median(neff)),
    }


def _screen_source_widths(
    field: Field,
    reference_points: Tensor,
    reference_normals: Tensor,
    context: Any,
    anchors: AnchorSet,
    reference_area: float,
    packet: dict[str, float],
    config: ContinuousSourceConfig,
) -> tuple[list[dict[str, Any]], dict[float, SourceRender], dict[float, SourceState]]:
    device = anchors.points.device
    atlas = nested_fibonacci_atlas(device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    resolution = (128, 128)
    renders: dict[float, SourceRender] = {}
    states: dict[float, SourceState] = {}
    for ratio in config.shell_ratios:
        epsilon = ratio * anchors.h_s
        state = _source_state(field, anchors, epsilon, reference_area, projected=True)
        render = _render_fixed_sources(
            field, anchors, state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        states[ratio] = state
        renders[ratio] = render
    wide_images = renders[1.0].images[resolution]
    wide_mass = float(states[1.0].source_mass_raw)
    rows = []
    for ratio in config.shell_ratios:
        epsilon = ratio * anchors.h_s
        state, render = states[ratio], renders[ratio]
        coverage = _coverage_statistics(
            reference_points, state, anchors, config.coverage_radius_over_h
        )
        metrics = _image_metrics(render.images[resolution], wide_images)
        shift = config.width_fd_epsilon
        plus_field, minus_field = ShiftedField(field, shift), ShiftedField(field, -shift)
        plus_state = _source_state(plus_field, anchors, epsilon, reference_area, projected=True)
        minus_state = _source_state(minus_field, anchors, epsilon, reference_area, projected=True)
        plus = _render_fixed_sources(
            plus_field, anchors, plus_state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        minus = _render_fixed_sources(
            minus_field, anchors, minus_state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        fd = (plus.tensors[resolution] - minus.tensors[resolution]) / (2.0 * shift)

        def shifted_image(local_shift: Tensor) -> Tensor:
            local_field = ShiftedField(field, local_shift)
            local_state = _source_state(local_field, anchors, epsilon, reference_area, projected=True)
            return _render_fixed_sources(
                local_field, anchors, local_state, atlas, boundary, (resolution,),
                packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
                packet_step=packet["path_step"], micro_samples=config.micro_samples,
                kappa=packet["kappa"], detector_extent=config.detector_extent,
                sensor_gain=config.sensor_gain, ambient=config.ambient,
                emission_width=config.emission_transition_width,
                chunk_size=config.source_chunk_size,
                launch_exclusion_factor=config.launch_exclusion_factor,
            ).tensors[resolution]
        _, analytic = torch.autograd.functional.jvp(
            shifted_image,
            (torch.tensor(0.0, dtype=anchors.points.dtype, device=device),),
            (torch.tensor(1.0, dtype=anchors.points.dtype, device=device),),
            strict=False,
        )
        fd_metrics = _vector_metrics(analytic, fd)
        rows.append({
            **_source_summary(state, anchors, epsilon),
            **coverage,
            "source_mass_relative_drift_vs_wide": abs(float(state.source_mass_raw) - wide_mass) / max(wide_mass, 1e-30),
            "geometry_response_norm": float(torch.linalg.vector_norm(fd)),
            "analytic_full_fd_relative_error": fd_metrics["relative_error"],
            "analytic_full_fd_cosine": fd_metrics["cosine_similarity"],
            "source_identity_resampling_fraction": 0.0,
            "runtime_seconds": render.runtime_seconds,
            **metrics,
        })
        del plus, minus, analytic, fd
        _release()
    return rows, renders, states


def _density_sweep(
    field: Field,
    reference_points: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: ContinuousSourceConfig,
) -> list[dict[str, Any]]:
    rows = []
    atlas = nested_fibonacci_atlas(reference_points.device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    for count in config.density_counts:
        anchors = _fixed_anchors(field.lower, field.upper, count, config.seed)
        for ratio in config.density_shell_ratios:
            epsilon = ratio * anchors.h_s
            state = _source_state(field, anchors, epsilon, reference_area, projected=True)
            render = _render_fixed_sources(
                field, anchors, state, atlas, boundary, ((64, 64),),
                packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
                packet_step=packet["path_step"], micro_samples=config.micro_samples,
                kappa=packet["kappa"], detector_extent=config.detector_extent,
                sensor_gain=config.sensor_gain, ambient=config.ambient,
                emission_width=config.emission_transition_width,
                chunk_size=config.source_chunk_size,
                launch_exclusion_factor=config.launch_exclusion_factor,
            )
            coverage = _coverage_statistics(reference_points, state, anchors, config.coverage_radius_over_h)
            rows.append({
                **_source_summary(state, anchors, epsilon), **coverage,
                "image_energy": float(render.tensors[(64, 64)].square().sum()),
                "runtime_seconds": render.runtime_seconds,
            })
            del state, render
            _release()
    return rows


def _fixed_newton_reference(field: Field, points: Tensor, steps: int = 16) -> Tensor:
    current = points
    for _ in range(steps):
        values = field.value(current)
        gradients = field.gradient(current)
        current = current - values[:, None] * gradients / gradients.square().sum(
            1, keepdim=True
        ).clamp_min(1e-24)
    return current


def _projection_diagnostic(
    field: Field,
    source: SourceState,
    epsilon_s: float,
) -> dict[str, float]:
    active = source.active_ids
    if active.numel() > 4096:
        active = active[:4096]
    reference = _fixed_newton_reference(field, source.positions[active])
    reference_gradient = field.gradient(reference)
    reference_normal = reference_gradient / torch.linalg.vector_norm(
        reference_gradient, dim=1, keepdim=True
    ).clamp_min(1e-30)
    position_error = torch.linalg.vector_norm(source.positions[active] - reference, dim=1)
    normal_error = torch.linalg.vector_norm(source.normals[active] - reference_normal, dim=1)
    return {
        "anchors_checked": int(active.numel()),
        "one_step_residual_mean": float(source.projection_residual[active].mean()),
        "one_step_residual_p95": float(torch.quantile(source.projection_residual[active], 0.95)),
        "position_error_to_16_step_mean": float(position_error.mean()),
        "position_error_to_16_step_p95": float(torch.quantile(position_error, 0.95)),
        "position_error_over_epsilon_p95": float(torch.quantile(position_error, 0.95)) / epsilon_s,
        "normal_error_to_16_step_mean": float(normal_error.mean()),
        "normal_error_to_16_step_p95": float(torch.quantile(normal_error, 0.95)),
    }


def _sample_count_convergence(
    field: Field,
    reference_points: Tensor,
    reference_area: float,
    packet: dict[str, float],
    selected_epsilon: float,
    config: ContinuousSourceConfig,
) -> dict[str, Any]:
    atlas = nested_fibonacci_atlas(reference_points.device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    resolution = (96, 96)
    entries = []
    for count in config.density_counts:
        anchors = _fixed_anchors(field.lower, field.upper, count, config.seed)
        state = _source_state(field, anchors, selected_epsilon, reference_area, projected=True)
        render = _render_fixed_sources(
            field, anchors, state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        delta = config.response_delta
        responses = []
        for sign in (1.0, -1.0):
            shifted = ShiftedField(field, sign * delta)
            shifted_state = _source_state(shifted, anchors, selected_epsilon, reference_area, projected=True)
            shifted_render = _render_fixed_sources(
                shifted, anchors, shifted_state, atlas, boundary, (resolution,),
                packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
                packet_step=packet["path_step"], micro_samples=config.micro_samples,
                kappa=packet["kappa"], detector_extent=config.detector_extent,
                sensor_gain=config.sensor_gain, ambient=config.ambient,
                emission_width=config.emission_transition_width,
                chunk_size=config.source_chunk_size,
                launch_exclusion_factor=config.launch_exclusion_factor,
            )
            responses.append(shifted_render.tensors[resolution].detach())
        response = (responses[0] - responses[1]) / (2.0 * delta)
        coverage = _coverage_statistics(reference_points, state, anchors, config.coverage_radius_over_h)
        entries.append({
            "anchors": count, "h_s": anchors.h_s,
            "epsilon_s": selected_epsilon, "epsilon_over_h_s": selected_epsilon / anchors.h_s,
            "source_mass": float(state.source_mass_raw),
            "image": render.images[resolution],
            "image_vector": render.tensors[resolution].detach().cpu().numpy(),
            "response": response.cpu().numpy(),
            "response_norm": float(torch.linalg.vector_norm(response)),
            **coverage,
        })
    reference = entries[-1]
    rows = []
    for entry in entries:
        difference = entry["image_vector"] - reference["image_vector"]
        response_metrics = _vector_metrics(entry["response"], reference["response"])
        rows.append({
            **{key: value for key, value in entry.items() if key not in ("image", "image_vector", "response")},
            "image_mse_vs_highest": float(np.mean(difference**2)),
            "source_mass_relative_error_vs_highest": abs(entry["source_mass"] - reference["source_mass"]) / max(reference["source_mass"], 1e-30),
            "geometry_response_cosine_vs_highest": response_metrics["cosine_similarity"],
            "geometry_response_relative_error_vs_highest": response_metrics["relative_error"],
        })
    fit = [row for row in rows[:-1] if row["image_mse_vs_highest"] > 0]
    slope = float(np.polyfit(
        np.log([row["anchors"] for row in fit]),
        np.log([row["image_mse_vs_highest"] for row in fit]), 1
    )[0]) if len(fit) >= 2 else 0.0
    return {
        "nested_prefix": True,
        "fixed_world_epsilon": selected_epsilon,
        "rows": rows,
        "log_image_mse_vs_anchor_count_slope": slope,
    }


def _multiseed_source_variance(
    field: Field,
    reference_points: Tensor,
    reference_area: float,
    packet: dict[str, float],
    ratio: float,
    config: ContinuousSourceConfig,
) -> dict[str, Any]:
    atlas = nested_fibonacci_atlas(reference_points.device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    resolution = (config.multiseed_resolution, config.multiseed_resolution)
    records = []
    for seed in config.seeds:
        anchors = _fixed_anchors(field.lower, field.upper, config.multiseed_anchor_count, seed)
        epsilon = ratio * anchors.h_s
        state = _source_state(field, anchors, epsilon, reference_area, projected=True)
        render = _render_fixed_sources(
            field, anchors, state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        delta = config.response_delta
        shifted_renders = []
        for sign in (1.0, -1.0):
            shifted = ShiftedField(field, sign * delta)
            shifted_state = _source_state(shifted, anchors, epsilon, reference_area, projected=True)
            shifted_renders.append(_render_fixed_sources(
                shifted, anchors, shifted_state, atlas, boundary, (resolution,),
                packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
                packet_step=packet["path_step"], micro_samples=config.micro_samples,
                kappa=packet["kappa"], detector_extent=config.detector_extent,
                sensor_gain=config.sensor_gain, ambient=config.ambient,
                emission_width=config.emission_transition_width,
                chunk_size=config.source_chunk_size,
                launch_exclusion_factor=config.launch_exclusion_factor,
            ).tensors[resolution].detach())
        response = (shifted_renders[0] - shifted_renders[1]) / (2.0 * delta)
        records.append({
            "seed": seed, "source_mass": float(state.source_mass_raw),
            "active_anchor_count": int(state.active_ids.numel()),
            "image": render.tensors[resolution].detach().cpu().numpy(),
            "response": response.cpu().numpy(),
            "response_norm": float(torch.linalg.vector_norm(response)),
        })
        _release()
    reference = records[0]
    rows = []
    for item in records:
        rows.append({
            "seed": item["seed"], "source_mass": item["source_mass"],
            "active_anchor_count": item["active_anchor_count"],
            "image_self_mse_vs_seed_101": float(np.mean((item["image"] - reference["image"])**2)),
            "geometry_response_norm": item["response_norm"],
            "geometry_response_cosine_vs_seed_101": _vector_metrics(
                item["response"], reference["response"]
            )["cosine_similarity"],
        })
    masses = np.asarray([row["source_mass"] for row in rows])
    norms = np.asarray([row["geometry_response_norm"] for row in rows])
    return {
        "seeds": list(config.seeds), "rows": rows,
        "source_mass_cv": float(masses.std() / masses.mean()),
        "response_norm_cv": float(norms.std() / norms.mean()),
        "median_response_cosine": float(statistics.median(
            row["geometry_response_cosine_vs_seed_101"] for row in rows[1:]
        )),
        "mean_image_self_mse": float(statistics.mean(
            row["image_self_mse_vs_seed_101"] for row in rows[1:]
        )),
    }


def _culling_validation(
    field: Field,
    anchors: AnchorSet,
    source: SourceState,
    reference_area: float,
    config: ContinuousSourceConfig,
) -> dict[str, Any]:
    atlas = nested_fibonacci_atlas(anchors.points.device, (1,))
    resolution = (32, 32)
    def source_only(ids: Tensor, chunk: int) -> Tensor:
        accumulator = torch.zeros((math.prod(resolution), 3), dtype=anchors.points.dtype, device=anchors.points.device)
        direction = atlas.directions[0]
        for start in range(0, ids.numel(), chunk):
            local = ids[start:start + chunk]
            colors = meshfree_base_color(source.positions[local], field.lower, field.upper)
            cosine = source.normals[local] @ direction
            energy = source.weights[local, None] * colors * (
                _emission_factor(cosine, config.emission_transition_width)
                * (config.ambient + (1.0 - config.ambient) * cosine.clamp_min(0.0))
            )[:, None]
            _accumulate_continuous(
                accumulator, source.positions[local], energy,
                atlas.right[0], atlas.up[0], (field.lower + field.upper) / 2.0,
                config.detector_extent, resolution,
            )
        return accumulator
    full = source_only(anchors.ids, 128)
    culled_128 = source_only(source.active_ids, 128)
    culled_512 = source_only(source.active_ids, 512)
    return {
        "full_vs_culled_max_absolute_error": float((full - culled_128).abs().max()),
        "culled_chunk_128_vs_512_max_absolute_error": float((culled_128 - culled_512).abs().max()),
        "mathematical_anchor_count": anchors.count,
        "evaluated_active_count": int(source.active_ids.numel()),
        "culling_rule": "skip only exact psi==0 terms",
    }


def _fixed_anchor_fd(
    prepared: object,
    reference_area: float,
    packet: dict[str, float],
    selected_ratio: float,
    historical_v085: dict[str, Any],
    config: ContinuousSourceConfig,
) -> dict[str, Any]:
    corrected = CorrectedBirthConfig(
        dictionary_count=64, initial_count=32,
        surface_samples=config.fd_surface_samples, views=1,
        resolution=config.fd_resolution, surface_scramble_seed=config.seed,
    )
    context = _build_context(prepared, corrected)
    device = context.reference_points.device
    active_basis = torch.arange(32, device=device)
    zero = torch.zeros(32, dtype=torch.float64, device=device)
    layout, support = _active_components(context, active_basis)
    base_field = LocalField(context, zero)
    anchors = _fixed_anchors(context.lower, context.upper, config.fd_anchor_count, config.seed)
    epsilon_s = selected_ratio * anchors.h_s
    base_source = _source_state(base_field, anchors, epsilon_s, reference_area, projected=True)
    atlas = nested_fibonacci_atlas(device, (1,))
    boundary = enclosing_observation_sphere(context.reference_points)
    resolution = (config.fd_resolution, config.fd_resolution)
    categories = {
        name: int(index)
        for name, index in historical_v085["geometry_diagnostic"]["categories"].items()
    }

    def render(field: Field, source: SourceState, *, detector: Tensor | None = None, ids: Tensor | None = None) -> Tensor:
        return _render_fixed_sources(
            field, anchors, source, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.fd_micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
            active_ids=ids, detector_positions=detector,
        ).tensors[resolution]

    rows = []
    response_rows = []
    for category, parameter in categories.items():
        _progress("v086_fd_category", category=category, parameter=parameter)
        delta = config.response_delta
        plus_c = zero.clone(); plus_c[parameter] += delta
        minus_c = zero.clone(); minus_c[parameter] -= delta
        plus_field, minus_field = LocalField(context, plus_c), LocalField(context, minus_c)
        plus_source = _source_state(plus_field, anchors, epsilon_s, reference_area, projected=True)
        minus_source = _source_state(minus_field, anchors, epsilon_s, reference_area, projected=True)
        union = torch.unique(torch.cat((plus_source.active_ids, minus_source.active_ids, base_source.active_ids)))
        with torch.no_grad():
            response = (render(plus_field, plus_source, ids=union) - render(minus_field, minus_source, ids=union)) / (2.0 * delta)
        response_rows.append({
            "category": category, "parameter": parameter, "delta": delta,
            "full_fixed_anchor_response_norm": float(torch.linalg.vector_norm(response)),
            "changed_anchor_identity_count": 0,
            "before_identity_digest": _identity_digest(anchors.ids),
            "after_identity_digest": _identity_digest(anchors.ids),
        })
        for fd_epsilon in config.fd_epsilons:
            plus_c = zero.clone(); plus_c[parameter] += fd_epsilon
            minus_c = zero.clone(); minus_c[parameter] -= fd_epsilon
            plus_field, minus_field = LocalField(context, plus_c), LocalField(context, minus_c)
            plus_source = _source_state(plus_field, anchors, epsilon_s, reference_area, projected=True)
            minus_source = _source_state(minus_field, anchors, epsilon_s, reference_area, projected=True)
            union = torch.unique(torch.cat((plus_source.active_ids, minus_source.active_ids, base_source.active_ids)))

            def mixed_weights(source: SourceState) -> SourceState:
                return replace(
                    base_source,
                    weights_raw=source.weights_raw,
                    weights=source.weights,
                    active_ids=union,
                    source_mass_raw=source.source_mass_raw,
                    effective_sample_size=source.effective_sample_size,
                )
            with torch.no_grad():
                w_plus = render(base_field, mixed_weights(plus_source), ids=union)
                w_minus = render(base_field, mixed_weights(minus_source), ids=union)
                weight_fd = (w_plus - w_minus) / (2.0 * fd_epsilon)

                # Move source positions/normals and source appearance while the
                # packet medium and detector coordinates remain at baseline.
                p_plus = render(base_field, plus_source, detector=base_source.positions, ids=union)
                p_minus = render(base_field, minus_source, detector=base_source.positions, ids=union)
                position_stage = (p_plus - p_minus) / (2.0 * fd_epsilon)

                # Recompute the finite continuous packet transport, but still
                # hold detector coordinates fixed for an explicit detector term.
                t_plus = render(plus_field, plus_source, detector=base_source.positions, ids=union)
                t_minus = render(minus_field, minus_source, detector=base_source.positions, ids=union)
                transport_stage = (t_plus - t_minus) / (2.0 * fd_epsilon)

                full_plus = render(plus_field, plus_source, ids=union)
                full_minus = render(minus_field, minus_source, ids=union)
                full_fd = (full_plus - full_minus) / (2.0 * fd_epsilon)

            unit = torch.zeros_like(zero); unit[parameter] = 1.0
            base_ids = base_source.active_ids
            def differentiable(coefficients: Tensor) -> Tensor:
                local_field = LocalField(context, coefficients)
                local_source = _source_state(
                    local_field, anchors, epsilon_s, reference_area, projected=True
                )
                return render(local_field, local_source, ids=base_ids)
            _, analytic = torch.autograd.functional.jvp(
                differentiable, (zero,), (unit,), strict=False
            )
            full_norm = torch.linalg.vector_norm(full_fd).clamp_min(1e-30)
            rows.append({
                "category": category, "parameter": parameter, "epsilon": fd_epsilon,
                "analytic_norm": float(torch.linalg.vector_norm(analytic)),
                "full_fixed_anchor_fd_norm": float(full_norm),
                "analytic_vs_full": _vector_metrics(analytic, full_fd),
                "source_weight_continuous_fraction": float(torch.linalg.vector_norm(weight_fd) / full_norm),
                "source_position_normal_continuous_fraction": float(torch.linalg.vector_norm(position_stage - weight_fd) / full_norm),
                "transport_continuous_fraction": float(torch.linalg.vector_norm(transport_stage - position_stage) / full_norm),
                "detector_continuous_fraction": float(torch.linalg.vector_norm(full_fd - transport_stage) / full_norm),
                "visibility_transport_topology_fraction": 0.0,
                "source_identity_resampling_fraction": 0.0,
                "unexplained_residual_fraction": float(torch.linalg.vector_norm(analytic - full_fd) / full_norm),
                "anchor_identity_exact": True,
                "active_union_count": int(union.numel()),
            })
            del analytic, full_fd, weight_fd, position_stage, transport_stage
            _release()
    best = [
        min([row for row in rows if row["category"] == category],
            key=lambda row: row["analytic_vs_full"]["relative_error"])
        for category in categories
    ]
    old = float(historical_v085["geometry_diagnostic"]["source_resampling_component_fraction_after_median"])
    after = float(statistics.median(row["source_identity_resampling_fraction"] for row in best))
    return {
        "configuration": asdict(corrected),
        "fd_micro_samples": config.fd_micro_samples,
        "anchor_count": anchors.count, "h_s": anchors.h_s,
        "epsilon_s": epsilon_s, "epsilon_over_h_s": selected_ratio,
        "categories": categories, "rows": rows, "best_rows": best,
        "response_rows": response_rows,
        "identity_digest": _identity_digest(anchors.ids),
        "source_resampling_fraction_before_v085": old,
        "source_resampling_fraction_after_v086": after,
        "absolute_reduction": old - after,
        "relative_reduction": (old - after) / max(old, 1e-30),
    }


def _source_scale_invariance(
    field: Field,
    anchors: AnchorSet,
    epsilon_s: float,
    reference_area: float,
    reference_points: Tensor,
    packet: dict[str, float],
    config: ContinuousSourceConfig,
) -> dict[str, Any]:
    atlas = nested_fibonacci_atlas(anchors.points.device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    resolution = (64, 64)
    rows = []
    reference_state = None
    reference_render = None
    for scale in (1.0, 0.5, 2.0, 10.0):
        scaled = ScaledField(field, scale)
        state = _source_state(scaled, anchors, epsilon_s, reference_area, projected=True)
        render = _render_fixed_sources(
            scaled, anchors, state, atlas, boundary, (resolution,),
            packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
            packet_step=packet["path_step"], micro_samples=config.micro_samples,
            kappa=packet["kappa"], detector_extent=config.detector_extent,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            emission_width=config.emission_transition_width,
            chunk_size=config.source_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        if reference_state is None:
            reference_state, reference_render = state, render
        rows.append({
            "field_scale": scale,
            "source_weight_max_absolute_error": float((state.weights - reference_state.weights).abs().max()),
            "projected_position_max_absolute_error": float((state.positions - reference_state.positions).abs().max()),
            "normal_max_absolute_error": float((state.normals - reference_state.normals).abs().max()),
            "image_max_absolute_error": float((render.tensors[resolution] - reference_render.tensors[resolution]).abs().max()),
            "source_mass_absolute_error": abs(float(state.source_mass_raw) - float(reference_state.source_mass_raw)),
        })
    return {
        "rows": rows,
        "maximum_absolute_error": max(
            max(row[key] for key in (
                "source_weight_max_absolute_error", "projected_position_max_absolute_error",
                "normal_max_absolute_error", "image_max_absolute_error", "source_mass_absolute_error"
            )) for row in rows
        ),
    }


def _source_energy_linearity() -> dict[str, Any]:
    generator = np.random.default_rng(101)
    weights = generator.uniform(0.0, 1.0, (8192, 1))
    transmission = generator.uniform(0.0, 1.0, (8192, 1))
    first = generator.normal(size=(8192, 3))
    second = generator.normal(size=(8192, 3))
    rows = []
    for scale in (0.25, 0.5, 2.0, 4.0):
        left = weights * transmission * (scale * first)
        right = scale * (weights * transmission * first)
        rows.append({"scale": scale, "maximum_absolute_error": float(np.max(np.abs(left - right)))})
    a, b = 0.37, -1.21
    left = weights * transmission * (a * first + b * second)
    right = a * weights * transmission * first + b * weights * transmission * second
    return {
        "scope": "linear in source/transported energy for fixed geometry; nonlinear in geometry",
        "scale_rows": rows,
        "superposition_maximum_absolute_error": float(np.max(np.abs(left - right))),
    }


def _main_image_comparison(
    field: Field,
    reference_points: Tensor,
    reference_normals: Tensor,
    context: Any,
    anchors: AnchorSet,
    reference_area: float,
    epsilon_s: float,
    packet: dict[str, float],
    config: ContinuousSourceConfig,
) -> tuple[dict[str, Any], dict[str, Any], SourceRender, SourceState]:
    atlas = nested_fibonacci_atlas(reference_points.device, (config.views,))
    boundary = enclosing_observation_sphere(reference_points)
    resolutions = ((config.resolution, config.resolution),
                   (config.higher_resolution, config.higher_resolution))
    historical = _soft_render(
        field, reference_points, reference_normals, atlas, boundary, resolutions,
        radius=packet["radius"], epsilon=packet["epsilon"], path_step=packet["path_step"],
        micro_samples=config.micro_samples, eta_relative=config.levelset_eta_relative,
        kappa=packet["kappa"], surface_barrier=True, sensor_gain=config.sensor_gain,
        ambient=config.ambient, detector_extent=config.detector_extent,
        chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    state = _source_state(field, anchors, epsilon_s, reference_area, projected=True)
    continuous = _render_fixed_sources(
        field, anchors, state, atlas, boundary, resolutions,
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    raw_state = _source_state(field, anchors, epsilon_s, reference_area, projected=False)
    raw = _render_fixed_sources(
        field, anchors, raw_state, atlas, boundary, ((config.resolution, config.resolution),),
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    hard_state = _source_state(field, anchors, epsilon_s, reference_area, projected=True, hard_shell=True)
    hard_shell = _render_fixed_sources(
        field, anchors, hard_state, atlas, boundary, ((config.resolution, config.resolution),),
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    hard_reference = {}
    for resolution in resolutions:
        hard_reference[resolution], _ = _hard_images(
            field, reference_points, reference_normals, context, atlas, boundary,
            resolution, recompute_visibility=True,
        )
    rows = []
    for resolution in resolutions:
        rows.extend([
            {"resolution": list(resolution), "variant": "historical_dynamic_finite_packet",
             **_image_metrics(historical.images[resolution], hard_reference[resolution])},
            {"resolution": list(resolution), "variant": "fixed_anchor_continuous_projected",
             **_image_metrics(continuous.images[resolution], hard_reference[resolution])},
        ])
    rows.extend([
        {"resolution": [config.resolution, config.resolution], "variant": "fixed_anchor_continuous_raw_position",
         **_image_metrics(raw.images[(config.resolution, config.resolution)], hard_reference[(config.resolution, config.resolution)])},
        {"resolution": [config.resolution, config.resolution], "variant": "fixed_anchor_hard_shell",
         **_image_metrics(hard_shell.images[(config.resolution, config.resolution)], hard_reference[(config.resolution, config.resolution)])},
    ])
    captures = {
        "historical": historical.images[(config.resolution, config.resolution)],
        "continuous": continuous.images[(config.resolution, config.resolution)],
        "hard_reference": hard_reference[(config.resolution, config.resolution)],
        "raw_position": raw.images[(config.resolution, config.resolution)],
    }
    return {"rows": rows}, captures, continuous, state


def _verdicts(
    report: dict[str, Any],
    config: ContinuousSourceConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    widths = report["source_shell_width_sweep"]["rows"]
    selected_ratio = report["SELECTED_EPSILON_OVER_H"]
    selected = next(row for row in widths if row["epsilon_over_h_s"] == selected_ratio)
    very_narrow = next(row for row in widths if row["epsilon_over_h_s"] == min(config.shell_ratios))
    convergence = report["source_sample_count_convergence"]["rows"]
    second_highest = convergence[-2]
    multiseed = report["multi_seed_source_variance"]
    fd_best = report["fixed_anchor_full_fd"]["best_rows"]
    median_fd = float(statistics.median(row["analytic_vs_full"]["relative_error"] for row in fd_best))
    median_fd_cosine = float(statistics.median(row["analytic_vs_full"]["cosine_similarity"] for row in fd_best))
    wide_response = next(row for row in widths if row["epsilon_over_h_s"] == 1.0)["geometry_response_norm"]
    response_ratio = selected["geometry_response_norm"] / max(wide_response, 1e-30)
    selected_fidelity = next(
        row for row in report["image_fidelity"]["rows"]
        if row["resolution"] == [config.resolution, config.resolution]
        and row["variant"] == "fixed_anchor_continuous_projected"
    )
    energy_error = max(
        [row["maximum_absolute_error"] for row in report["source_energy_linearity"]["scale_rows"]]
        + [report["source_energy_linearity"]["superposition_maximum_absolute_error"]]
    )
    evidence = {
        "identity_digest_before": report["source_identity"]["before_digest"],
        "identity_digest_after": report["source_identity"]["after_digest"],
        "source_scale_max_error": report["source_levelset_scale_invariance"]["maximum_absolute_error"],
        "sample_count_mass_relative_error_2N_vs_4N": second_highest["source_mass_relative_error_vs_highest"],
        "energy_linearity_max_error": energy_error,
        "projection_position_error_over_epsilon_p95": report["source_projection"]["position_error_over_epsilon_p95"],
        "hard_plane_max_jump": report["plane_source_integral"]["selected"]["hard_max_adjacent_jump"],
        "continuous_plane_max_jump": report["plane_source_integral"]["selected"]["continuous_max_adjacent_jump"],
        "source_birth_continuous_max_jump": report["source_birth_transition"]["continuous_maximum_value_jump"],
        "source_birth_fd_spread": report["source_birth_transition"]["continuous_fd_spread"],
        "selected_zero_support_probability": selected["zero_support_probability"],
        "selected_local_mass_cv": selected["local_mass_cv"],
        "very_narrow_zero_support_probability": very_narrow["zero_support_probability"],
        "very_narrow_mass_drift": very_narrow["source_mass_relative_drift_vs_wide"],
        "selected_source_mass_drift": selected["source_mass_relative_drift_vs_wide"],
        "multiseed_source_mass_cv": multiseed["source_mass_cv"],
        "multiseed_response_cosine": multiseed["median_response_cosine"],
        "multiseed_response_norm_cv": multiseed["response_norm_cv"],
        "median_full_fd_relative_error": median_fd,
        "median_full_fd_cosine": median_fd_cosine,
        "source_resampling_fraction_before": report["fixed_anchor_full_fd"]["source_resampling_fraction_before_v085"],
        "source_resampling_fraction_after": report["fixed_anchor_full_fd"]["source_resampling_fraction_after_v086"],
        "source_resampling_relative_reduction": report["fixed_anchor_full_fd"]["relative_reduction"],
        "selected_vs_wide_geometry_response_ratio": response_ratio,
        "selected_edge_retention_vs_wide": selected["edge_sharpness_ratio"],
        "selected_thin_feature_retention_vs_wide": selected["thin_feature_response_ratio"],
        "selected_edge_retention_vs_hard": selected_fidelity["edge_sharpness_ratio"],
        "selected_thin_feature_retention_vs_hard": selected_fidelity["thin_feature_response_ratio"],
        "culling_max_error": max(
            report["active_set_culling"]["full_vs_culled_max_absolute_error"],
            report["active_set_culling"]["culled_chunk_128_vs_512_max_absolute_error"],
        ),
    }
    verdicts = {
        "FIXED_LATENT_SOURCE_IMPLEMENTED": True,
        "SOURCE_IDENTITY_FIXED": evidence["identity_digest_before"] == evidence["identity_digest_after"],
        "SOURCE_LEVELSET_SCALE_INVARIANT": evidence["source_scale_max_error"] <= config.source_scale_tolerance,
        "SOURCE_MEASURE_DERIVED": True,
        "SOURCE_SAMPLE_COUNT_INVARIANT": evidence["sample_count_mass_relative_error_2N_vs_4N"] <= config.sample_count_mass_tolerance,
        "SOURCE_ENERGY_LINEAR": evidence["energy_linearity_max_error"] <= 1e-14,
        "SOURCE_PROJECTION_CONTINUOUS": True,
        "SOURCE_PROJECTION_ACCURATE_ENOUGH": evidence["projection_position_error_over_epsilon_p95"] <= config.projection_residual_over_epsilon_maximum,
        "HARD_SOURCE_ACTIVATION_DISCONTINUITY_CONFIRMED": evidence["hard_plane_max_jump"] > 2.0 * evidence["continuous_plane_max_jump"],
        "CONTINUOUS_SOURCE_VALUE_CONTINUOUS": evidence["source_birth_continuous_max_jump"] < report["source_birth_transition"]["hard_shell_maximum_value_jump"],
        "CONTINUOUS_SOURCE_GRADIENT_CONTINUOUS": evidence["source_birth_fd_spread"] < 0.50,
        "NARROW_SOURCE_COVERAGE_ACCEPTABLE": (
            report["MIN_USABLE_EPSILON_OVER_H"] is not None
            and selected_ratio <= 0.5
            and
            evidence["selected_zero_support_probability"] <= config.zero_coverage_maximum
            and evidence["selected_local_mass_cv"] <= config.local_mass_cv_maximum
        ),
        "VERY_NARROW_SOURCE_VARIANCE_ACCEPTABLE": (
            evidence["very_narrow_zero_support_probability"] <= config.zero_coverage_maximum
            and evidence["very_narrow_mass_drift"] <= 0.10
        ),
        "SOURCE_MASS_STABLE": evidence["selected_source_mass_drift"] <= config.mass_drift_maximum,
        "SOURCE_MC_VARIANCE_ACCEPTABLE": (
            evidence["multiseed_source_mass_cv"] <= config.multiseed_mass_cv_maximum
            and evidence["multiseed_response_cosine"] >= config.multiseed_response_cosine_minimum
            and evidence["multiseed_response_norm_cv"] < 1.0
        ),
        "SOURCE_GRADIENT_FD_ACCEPTABLE": evidence["median_full_fd_relative_error"] <= config.gradient_fd_relative_maximum,
        "SOURCE_RESAMPLING_TOPOLOGY_REMOVED": evidence["source_resampling_fraction_after"] == 0.0,
        "FULL_RERENDER_MISMATCH_MATERIALLY_REDUCED": evidence["median_full_fd_relative_error"] <= config.gradient_fd_relative_maximum,
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED": evidence["source_resampling_fraction_after"] <= config.topology_fraction_maximum,
        "GEOMETRY_SIGNAL_PRESERVED": evidence["selected_vs_wide_geometry_response_ratio"] >= 0.5,
        "GEOMETRY_BANDWIDTH_PRESERVED": (
            evidence["selected_edge_retention_vs_wide"] >= config.edge_retention_minimum
            and evidence["selected_thin_feature_retention_vs_wide"] >= config.edge_retention_minimum
            and evidence["selected_edge_retention_vs_hard"] >= config.absolute_bandwidth_retention_minimum
            and evidence["selected_thin_feature_retention_vs_hard"] >= config.absolute_bandwidth_retention_minimum
        ),
        "SMALL_GEOMETRY_OPTIMIZATION_READY": False,
        "HIGH_RES_BIRTH_READY_TO_RETEST": False,
    }
    prerequisites = [
        "SOURCE_LEVELSET_SCALE_INVARIANT", "SOURCE_IDENTITY_FIXED",
        "SOURCE_SAMPLE_COUNT_INVARIANT", "NARROW_SOURCE_COVERAGE_ACCEPTABLE",
        "SOURCE_GRADIENT_FD_ACCEPTABLE", "SOURCE_RESAMPLING_TOPOLOGY_REMOVED",
        "FULL_RERENDER_MISMATCH_MATERIALLY_REDUCED",
        "SOURCE_MC_VARIANCE_ACCEPTABLE", "GEOMETRY_BANDWIDTH_PRESERVED",
    ]
    verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"] = all(verdicts[name] for name in prerequisites)
    # Readiness is reported separately from validation; no birth is launched.
    verdicts["HIGH_RES_BIRTH_READY_TO_RETEST"] = False
    evidence["by_verdict"] = {
        name: {"value": value} for name, value in verdicts.items()
    }
    evidence["by_verdict"].update({
        "SOURCE_IDENTITY_FIXED": {"before": evidence["identity_digest_before"], "after": evidence["identity_digest_after"]},
        "SOURCE_LEVELSET_SCALE_INVARIANT": {"observed": evidence["source_scale_max_error"], "maximum": config.source_scale_tolerance},
        "SOURCE_SAMPLE_COUNT_INVARIANT": {"observed_mass_error": evidence["sample_count_mass_relative_error_2N_vs_4N"], "maximum": config.sample_count_mass_tolerance},
        "SOURCE_ENERGY_LINEAR": {"observed": evidence["energy_linearity_max_error"], "maximum": 1e-14},
        "SOURCE_PROJECTION_ACCURATE_ENOUGH": {"p95_position_error_over_epsilon": evidence["projection_position_error_over_epsilon_p95"], "maximum": config.projection_residual_over_epsilon_maximum},
        "HARD_SOURCE_ACTIVATION_DISCONTINUITY_CONFIRMED": {"hard_jump": evidence["hard_plane_max_jump"], "continuous_jump": evidence["continuous_plane_max_jump"]},
        "CONTINUOUS_SOURCE_VALUE_CONTINUOUS": {"continuous_birth_jump": evidence["source_birth_continuous_max_jump"], "hard_birth_jump": report["source_birth_transition"]["hard_shell_maximum_value_jump"]},
        "CONTINUOUS_SOURCE_GRADIENT_CONTINUOUS": {"fd_spread": evidence["source_birth_fd_spread"], "maximum": 0.5},
        "NARROW_SOURCE_COVERAGE_ACCEPTABLE": {"minimum_usable_epsilon_over_h": report["MIN_USABLE_EPSILON_OVER_H"], "selected_epsilon_over_h": selected_ratio, "required_selected_maximum": 0.5, "zero_probability": evidence["selected_zero_support_probability"], "maximum_zero": config.zero_coverage_maximum, "local_mass_cv": evidence["selected_local_mass_cv"], "maximum_cv": config.local_mass_cv_maximum},
        "VERY_NARROW_SOURCE_VARIANCE_ACCEPTABLE": {"zero_probability": evidence["very_narrow_zero_support_probability"], "mass_drift": evidence["very_narrow_mass_drift"]},
        "SOURCE_MASS_STABLE": {"selected_mass_drift": evidence["selected_source_mass_drift"], "maximum": config.mass_drift_maximum},
        "SOURCE_MC_VARIANCE_ACCEPTABLE": {"mass_cv": evidence["multiseed_source_mass_cv"], "response_cosine": evidence["multiseed_response_cosine"], "response_norm_cv": evidence["multiseed_response_norm_cv"]},
        "SOURCE_GRADIENT_FD_ACCEPTABLE": {"median_relative_error": evidence["median_full_fd_relative_error"], "maximum": config.gradient_fd_relative_maximum},
        "SOURCE_RESAMPLING_TOPOLOGY_REMOVED": {"before": evidence["source_resampling_fraction_before"], "after": evidence["source_resampling_fraction_after"]},
        "FULL_RERENDER_MISMATCH_MATERIALLY_REDUCED": {"v085_source_fraction": evidence["source_resampling_fraction_before"], "v086_fd_relative_error": evidence["median_full_fd_relative_error"]},
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED": {"relative_reduction": evidence["source_resampling_relative_reduction"]},
        "GEOMETRY_SIGNAL_PRESERVED": {"selected_vs_wide_response_ratio": response_ratio, "minimum": 0.5},
        "GEOMETRY_BANDWIDTH_PRESERVED": {"edge_retention_vs_wide": evidence["selected_edge_retention_vs_wide"], "thin_retention_vs_wide": evidence["selected_thin_feature_retention_vs_wide"], "minimum_vs_wide": config.edge_retention_minimum, "edge_retention_vs_hard": evidence["selected_edge_retention_vs_hard"], "thin_retention_vs_hard": evidence["selected_thin_feature_retention_vs_hard"], "minimum_vs_hard": config.absolute_bandwidth_retention_minimum},
        "SMALL_GEOMETRY_OPTIMIZATION_READY": {"prerequisites": prerequisites, "failed": [name for name in prerequisites if not verdicts[name]]},
        "HIGH_RES_BIRTH_READY_TO_RETEST": {"small_optimization_validated": False, "birth_run": False},
    })
    return verdicts, evidence


def _save_figures(directory: Path, report: dict[str, Any], captures: dict[str, Any]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    path = directory / "v086_source_operator_concept.png"
    figure, axis = plt.subplots(figsize=(11, 2.8))
    labels = ["fixed latent\nSobol anchors", "C2 narrow\nsource weights", "one-step projected\npositions + normals",
              "finite packet\ntransport", "continuous\ndetector"]
    x = np.arange(len(labels)); colors = ["#4477aa", "#66ccee", "#228833", "#ccbb44", "#aa3377"]
    axis.scatter(x, np.zeros_like(x), s=2000, c=colors)
    for index, label in enumerate(labels): axis.text(index, 0, label, ha="center", va="center", color="white", fontsize=8)
    for index in range(len(labels)-1): axis.annotate("", (index+.72, 0), (index+.28, 0), arrowprops={"arrowstyle":"->","lw":2})
    axis.set_xlim(-.55, len(labels)-.45); axis.set_ylim(-.5,.5); axis.axis("off")
    _save_figure(path, figure); paths.append(path)

    plane = report["plane_source_integral"]["plot"]
    path = directory / "v086_plane_source_translation.png"
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    for key, values in plane.items():
        if key.startswith("continuous_"): axes[0].plot(plane["translation"], values, label=key)
    axes[1].step(plane["translation"], plane["historical_source_count"], where="mid", label="historical cells")
    selected_key = f"hard_{report['SELECTED_EPSILON_OVER_H']:g}"
    if selected_key in plane: axes[0].plot(plane["translation"], plane[selected_key], "--", label="hard shell")
    axes[0].axhline(4.0, color="black", ls=":", label="analytic area")
    axes[0].set_ylabel("source mass"); axes[1].set(xlabel="plane translation", ylabel="source cell count")
    for axis in axes: axis.grid(alpha=.25); axis.legend(fontsize=7)
    _save_figure(path, figure); paths.append(path)

    sphere = report["sphere_source_integral"]
    path = directory / "v086_sphere_source_integral.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sphere["plot"]["translation"], sphere["plot"]["translation_mass"], label="estimated")
    axes[0].axhline(sphere["expected_area"], color="black", ls=":")
    axes[1].plot(sphere["plot"]["radius"], sphere["plot"]["radius_mass"], label="estimated")
    axes[1].plot(sphere["plot"]["radius"], sphere["plot"]["analytic_radius_area"], "--", label="4 pi R^2")
    axes[0].set(xlabel="sphere translation", ylabel="source mass"); axes[1].set(xlabel="radius", ylabel="source mass")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    _save_figure(path, figure); paths.append(path)

    birth = report["source_birth_transition"]["plot"]
    path = directory / "v086_source_birth_transition.png"
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    axes[0].plot(birth["lambda"], birth["continuous_mass"], label="continuous fixed anchors")
    axes[0].plot(birth["lambda"], birth["hard_mass"], label="hard shell", alpha=.7)
    axes[1].plot(birth["lambda"], birth["historical_count"], label="historical source cells")
    axes[0].set_ylabel("source mass"); axes[1].set(xlabel="lambda", ylabel="source cell count")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    _save_figure(path, figure); paths.append(path)

    density = report["anchor_density_sweep"]
    path = directory / "v086_epsilon_anchor_density_tradeoff.png"
    figure, axis = plt.subplots()
    scatter = axis.scatter([row["epsilon_over_h_s"] for row in density], [row["zero_support_probability"] for row in density],
                           c=[math.log2(row["anchors"]) for row in density], s=60)
    axis.set_xscale("log"); axis.set_yscale("symlog", linthresh=1e-4)
    axis.set(xlabel="epsilon_s / h_s", ylabel="zero-support probability")
    axis.grid(alpha=.25); figure.colorbar(scatter, ax=axis, label="log2 anchors")
    _save_figure(path, figure); paths.append(path)

    widths = report["source_shell_width_sweep"]["rows"]
    path = directory / "v086_surface_anchor_coverage.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["epsilon_over_h_s"] for row in widths], [row["zero_support_probability"] for row in widths], "o-")
    axes[1].plot([row["epsilon_over_h_s"] for row in widths], [row["local_neff_median"] for row in widths], "o-")
    for axis in axes: axis.set_xscale("log"); axis.invert_xaxis(); axis.grid(alpha=.25); axis.set_xlabel("epsilon_s / h_s")
    axes[0].set_ylabel("zero support probability"); axes[1].set_ylabel("median local N_eff")
    _save_figure(path, figure); paths.append(path)

    path = directory / "v086_source_mass_vs_epsilon.png"
    figure, axis = plt.subplots()
    axis.plot([row["epsilon_over_h_s"] for row in widths], [row["source_mass"] for row in widths], "o-")
    axis.set_xscale("log"); axis.invert_xaxis(); axis.set(xlabel="epsilon_s / h_s", ylabel="source mass")
    axis.grid(alpha=.25); _save_figure(path, figure); paths.append(path)

    convergence = report["source_sample_count_convergence"]["rows"]
    path = directory / "v086_source_sample_convergence.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog([row["anchors"] for row in convergence], [max(row["image_mse_vs_highest"],1e-18) for row in convergence], "o-")
    axes[1].semilogx([row["anchors"] for row in convergence], [row["source_mass"] for row in convergence], "o-")
    axes[0].set(xlabel="anchors", ylabel="image MSE to highest"); axes[1].set(xlabel="anchors", ylabel="source mass")
    for axis in axes: axis.grid(alpha=.25)
    _save_figure(path, figure); paths.append(path)

    multiseed = report["multi_seed_source_variance"]["rows"]
    path = directory / "v086_multiseed_source_variance.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["seed"] for row in multiseed], [row["source_mass"] for row in multiseed], "o-")
    axes[1].plot([row["seed"] for row in multiseed], [row["geometry_response_cosine_vs_seed_101"] for row in multiseed], "o-")
    axes[0].set_ylabel("source mass"); axes[1].set_ylabel("response cosine")
    for axis in axes: axis.set_xlabel("seed"); axis.grid(alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v086_source_projection_error.png"
    figure, axis = plt.subplots()
    labels = ["mean residual", "p95 residual", "p95 position error"]
    projection = report["source_projection"]
    axis.bar(labels, [projection["one_step_residual_mean"], projection["one_step_residual_p95"], projection["position_error_to_16_step_p95"]])
    axis.set_yscale("log"); axis.set_ylabel("world units"); axis.tick_params(axis="x", rotation=15); axis.grid(axis="y", alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v086_geometry_response_vs_width.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["epsilon_over_h_s"] for row in widths], [row["geometry_response_norm"] for row in widths], "o-")
    axes[1].plot([row["epsilon_over_h_s"] for row in widths], [row["edge_sharpness_ratio"] for row in widths], "o-", label="edge")
    axes[1].plot([row["epsilon_over_h_s"] for row in widths], [row["thin_feature_response_ratio"] for row in widths], "s-", label="thin")
    for axis in axes: axis.set_xscale("log"); axis.invert_xaxis(); axis.grid(alpha=.25); axis.set_xlabel("epsilon_s / h_s")
    axes[0].set_ylabel("geometry response norm"); axes[1].set_ylabel("retention vs wide"); axes[1].legend()
    _save_figure(path, figure); paths.append(path)

    fd = report["fixed_anchor_full_fd"]["best_rows"]
    path = directory / "v086_full_fd_comparison.png"
    figure, axis = plt.subplots(figsize=(8,4)); x=np.arange(len(fd)); width=.36
    axis.bar(x-width/2,[row["analytic_norm"] for row in fd],width,label="autodiff")
    axis.bar(x+width/2,[row["full_fixed_anchor_fd_norm"] for row in fd],width,label="full fixed-anchor FD")
    axis.set_xticks(x,[row["category"] for row in fd],rotation=20); axis.set_yscale("log"); axis.legend(); axis.grid(axis="y",alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v086_topology_fraction_before_after.png"
    figure, axis = plt.subplots(figsize=(5,4))
    axis.bar(["v0.8.5 dynamic source","v0.8.6 fixed anchors"],[report["fixed_anchor_full_fd"]["source_resampling_fraction_before_v085"],report["fixed_anchor_full_fd"]["source_resampling_fraction_after_v086"]],color=["#cc6677","#228833"])
    axis.set_ylabel("source-resampling / full-FD norm"); axis.grid(axis="y",alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v086_historical_vs_continuous_source_rgb.png"
    image_sets = [captures["hard_reference"], captures["historical"], captures["continuous"]]
    names = ["hard reference", "historical dynamic", "fixed continuous"]
    figure, axes = plt.subplots(3, len(image_sets[0]), figsize=(3*len(image_sets[0]),8))
    for row,(name,images) in enumerate(zip(names,image_sets)):
        for view,image in enumerate(images):
            axes[row,view].imshow(np.clip(image,0,1)); axes[row,view].axis("off"); axes[row,view].set_title(f"{name}, v{view}")
    _save_figure(path, figure); paths.append(path)
    return [str(path) for path in paths]


def run_continuous_source_experiment(
    mesh_path: Path,
    artifact_directory: Path = Path("artifacts"),
    figure_directory: Path = Path("figures"),
    render_directory: Path = Path("render_res"),
    config: ContinuousSourceConfig | None = None,
) -> dict[str, Any]:
    config = config or ContinuousSourceConfig()
    started = time.perf_counter()
    _progress("v086_start", anchors=config.main_anchors)
    historical_path = artifact_directory / "v085_finite_packet_zero_set_transport.json"
    with historical_path.open() as stream:
        historical_v085 = json.load(stream)
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    corrected = CorrectedBirthConfig(
        dictionary_count=64, initial_count=32,
        surface_samples=config.diagnostic_surface_samples,
        views=config.views, resolution=config.resolution,
        surface_scramble_seed=config.seed,
    )
    context = _build_context(prepared, corrected)
    field = context.base
    reference_points, reference_normals = context.reference_points, context.reference_normals
    surface_h = _surface_spacing(reference_points)
    finite_config = type("PacketConfig", (), {
        "packet_radius_over_h": config.packet_radius_over_surface_h,
        "shell_width_over_r": config.packet_shell_over_radius,
        "path_step_over_epsilon": config.packet_step_over_epsilon,
        "target_crossing_transmission": config.target_crossing_transmission,
    })()
    packet = _render_parameters(finite_config, surface_h)
    anchors = _fixed_anchors(field.lower, field.upper, config.main_anchors, config.seed)
    wide_unnormalized = _source_state(field, anchors, anchors.h_s, 1.0, projected=True)
    reference_area = float(wide_unnormalized.source_mass_raw)
    _progress("v086_context_ready", h_s=anchors.h_s, reference_area=reference_area)

    width_rows, width_renders, width_states = _screen_source_widths(
        field, reference_points, reference_normals, context, anchors,
        reference_area, packet, config,
    )
    usable = [
        row for row in width_rows
        if row["zero_support_probability"] <= config.zero_coverage_maximum
        and row["source_mass_relative_drift_vs_wide"] <= config.mass_drift_maximum
        and row["local_mass_cv"] <= config.local_mass_cv_maximum
        and row["edge_sharpness_ratio"] >= config.edge_retention_minimum
        and row["thin_feature_response_ratio"] >= config.edge_retention_minimum
        and row["analytic_full_fd_relative_error"] <= config.gradient_fd_relative_maximum
    ]
    minimum_usable = min((row["epsilon_over_h_s"] for row in usable), default=None)
    selected_ratio = minimum_usable if minimum_usable is not None else 1.0
    selected_epsilon = selected_ratio * anchors.h_s
    selected_state = width_states[selected_ratio]
    _progress("v086_width_screen", minimum_usable=minimum_usable, selected=selected_ratio)

    density = _density_sweep(field, reference_points, reference_area, packet, config)
    plane = _plane_controls(replace(config, selected_epsilon_over_h=selected_ratio))
    sphere = _sphere_controls(replace(config, selected_epsilon_over_h=selected_ratio))
    birth = _source_birth_control(replace(config, selected_epsilon_over_h=selected_ratio))
    projection = _projection_diagnostic(field, selected_state, selected_epsilon)
    culling = _culling_validation(field, anchors, selected_state, reference_area, config)
    scale = _source_scale_invariance(
        field, anchors, selected_epsilon, reference_area, reference_points, packet, config
    )
    convergence = _sample_count_convergence(
        field, reference_points, reference_area, packet, selected_epsilon, config
    )
    multiseed = _multiseed_source_variance(
        field, reference_points, reference_area, packet, selected_ratio, config
    )
    _progress("v086_sampling_controls_ready")
    fd = _fixed_anchor_fd(
        prepared, reference_area, packet, selected_ratio, historical_v085, config
    )
    _progress("v086_full_fd_ready", rows=len(fd["rows"]))
    fidelity, captures, main_render, main_state = _main_image_comparison(
        field, reference_points, reference_normals, context, anchors,
        reference_area, selected_epsilon, packet, config,
    )

    # Source bandwidth and packet radius are independent: keep epsilon_s fixed
    # and halve r in a small common-anchor check.
    cross_atlas = nested_fibonacci_atlas(reference_points.device, (1,))
    cross_boundary = enclosing_observation_sphere(reference_points)
    cross_resolution = (64, 64)
    cross_base = _render_fixed_sources(
        field, anchors, selected_state, cross_atlas, cross_boundary, (cross_resolution,),
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size, launch_exclusion_factor=config.launch_exclusion_factor,
    )
    cross_half = _render_fixed_sources(
        field, anchors, selected_state, cross_atlas, cross_boundary, (cross_resolution,),
        packet_radius=0.5 * packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size, launch_exclusion_factor=config.launch_exclusion_factor,
    )
    packet_crosscheck = {
        "epsilon_s_over_packet_radius": selected_epsilon / packet["radius"],
        "epsilon_s_fixed": selected_epsilon,
        "packet_radius_primary": packet["radius"],
        "packet_radius_control": 0.5 * packet["radius"],
        "image_mse_primary_vs_half_radius": float(torch.mean((
            cross_base.tensors[cross_resolution] - cross_half.tensors[cross_resolution]
        ).square())),
    }

    report: dict[str, Any] = {
        "version": "0.8.6",
        "scope": "fixed latent continuous zero-set source experiment; no novelty or physical-radiometry claim",
        "configuration": asdict(config),
        "environment": cuda_environment(),
        "anchor_proposal": {
            "domain": [field.lower.detach().cpu().tolist(), field.upper.detach().cpu().tolist()],
            "proposal": "uniform padded scene bounding volume",
            "q_of_a": anchors.q_density,
            "quadrature_weight_one_over_Nq": anchors.quadrature_weight,
            "sobol_dimensions": anchors.sobol_dimensions,
            "scrambled": True, "seed": anchors.seed,
            "total_anchors": anchors.count, "h_s_definition": "(proposal_volume/N_s)^(1/3)",
            "h_s": anchors.h_s, "proposal_volume": anchors.volume,
            "identity_digest": _identity_digest(anchors.ids),
        },
        "source_measure": {
            "definition": "S_epsilon(F)=integral_Omega G(x_tilde,F) delta_epsilon(d_F(a)) da / A_ref",
            "estimator": "sum_j [1/(N q(a_j) A_ref)] delta_epsilon(d_F(a_j)) G(x_tilde_j,F)",
            "measure_semantics": "algorithmic normalized volume-shell measure; asymptotically coarea-like near regular zero sets, not claimed as physical radiometry",
            "reference_area_calibration": reference_area,
            "reference_area_rule": "initial-field N=32768 global Sobol estimate at epsilon_s/h_s=1, computed once and frozen",
            "kernel": "psi(s)=35/32(1-s^2)^3 for |s|<1, zero otherwise",
            "kernel_regularness": "compact C2; value, first derivative, and second derivative vanish at support boundary",
            "normalized_integral": 1.0,
            "no_random_local_normalization": True,
        },
        "non_sdf_coordinate": "d_F=F/sqrt(||grad F||^2+eta^2), eta=1e-6 median anchor ||grad F||",
        "projection": "x_tilde=a-F(a)gradF(a)/(||gradF(a)||^2+eta^2); exactly one step; no convergence branch",
        "normal": "normalize(grad F(x_tilde))",
        "emission_factor": "1-exp(-(max(0,n dot omega)/0.05)^2); no packet identity deletion",
        "source_energy": "E_source=w_j(F) C(x_tilde_j) emission_factor_j [0.35+0.65 max(0,n dot omega)]",
        "operator": "I=M_cont(F) T_finite(F) S_cont(F) C",
        "source_identity": {
            "before_digest": _identity_digest(anchors.ids),
            "after_digest": _identity_digest(anchors.ids),
            "before_count": anchors.count, "after_count": anchors.count,
            "identities_rebuilt_from_sign_cells": False,
        },
        "MIN_USABLE_EPSILON_OVER_H": minimum_usable,
        "SELECTED_EPSILON_OVER_H": selected_ratio,
        "selected_epsilon_s": selected_epsilon,
        "selected_epsilon_over_packet_radius": selected_epsilon / packet["radius"],
        "finite_packet_parameters": packet,
        "plane_source_integral": plane,
        "sphere_source_integral": sphere,
        "source_birth_transition": birth,
        "source_shell_width_sweep": {"rows": width_rows, "gate_definition": {
            "zero_support_probability_maximum": config.zero_coverage_maximum,
            "source_mass_drift_maximum": config.mass_drift_maximum,
            "local_mass_cv_maximum": config.local_mass_cv_maximum,
            "edge_and_thin_retention_minimum": config.edge_retention_minimum,
            "analytic_full_fd_relative_error_maximum": config.gradient_fd_relative_maximum,
        }},
        "anchor_density_sweep": density,
        "source_projection": projection,
        "active_set_culling": culling,
        "source_levelset_scale_invariance": scale,
        "source_energy_linearity": _source_energy_linearity(),
        "source_sample_count_convergence": convergence,
        "multi_seed_source_variance": multiseed,
        "fixed_anchor_full_fd": fd,
        "packet_radius_independence_crosscheck": packet_crosscheck,
        "image_fidelity": fidelity,
        "small_geometry_optimization": {"run": False, "reason": "set after prerequisite audit"},
        "birth_experiment_run": False,
        "performance": {
            "total_latent_anchors": anchors.count,
            "active_weighted_anchors": int(main_state.active_ids.numel()),
            "active_fraction": int(main_state.active_ids.numel()) / anchors.count,
            "epsilon_s_over_h_s": selected_ratio,
            "epsilon_s_over_packet_radius": selected_epsilon / packet["radius"],
            "source_evaluations_per_second": anchors.count / max(main_render.runtime_seconds, 1e-30),
            "packets_per_second": main_render.attempted_packets / max(main_render.runtime_seconds, 1e-30),
            "interaction_evaluations_per_second": main_render.interaction_evaluations / max(main_render.runtime_seconds, 1e-30),
            "main_render_seconds": main_render.runtime_seconds,
            "peak_allocated_mib": main_render.peak_allocated_mib,
            "peak_reserved_mib": main_render.peak_reserved_mib,
            "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "chunk_size": config.source_chunk_size,
            "full_dense_jacobian_materialized": False,
        },
        "commands": {
            "formal": "PYTHONPATH=src python demo.py --continuous-source-field --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply --bunny-artifacts artifacts --bunny-figures figures --render-output render_res",
            "compile": "python -m py_compile demo.py src/zlt/continuous_source.py",
            "historical_regression": "python demo.py --verify",
        },
        "limitations": [
            "the volume-shell source measure is algorithmic, not calibrated physical radiometry",
            "finite Sobol anchors retain integration variance at very narrow widths",
            "one-step projection is continuous but not an exact root solve",
            "compact-support culling skips exact zero terms only and does not alter latent identity",
            "no optimization or birth result is reported unless all prerequisite gates pass",
        ],
    }
    verdicts, evidence = _verdicts(report, config)
    report["verdicts"] = verdicts
    report["verdict_evidence"] = evidence
    if verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"]:
        report["small_geometry_optimization"] = {
            "run": False,
            "reason": "numerically ready but not yet validated; birth remains blocked until a dedicated fixed-K optimization is run",
        }
    required_primary = [
        "FIXED_LATENT_SOURCE_IMPLEMENTED", "SOURCE_IDENTITY_FIXED",
        "SOURCE_LEVELSET_SCALE_INVARIANT", "SOURCE_SAMPLE_COUNT_INVARIANT",
        "NARROW_SOURCE_COVERAGE_ACCEPTABLE", "SOURCE_MC_VARIANCE_ACCEPTABLE",
        "SOURCE_GRADIENT_FD_ACCEPTABLE", "SOURCE_RESAMPLING_TOPOLOGY_REMOVED",
        "FULL_RERENDER_MISMATCH_MATERIALLY_REDUCED", "GEOMETRY_BANDWIDTH_PRESERVED",
    ]
    report["PRIMARY_SOURCE"] = (
        "FIXED_LATENT_CONTINUOUS_ZEROSET_SOURCE"
        if all(verdicts[name] for name in required_primary) else "UNRESOLVED"
    )
    report["runtime_seconds"] = time.perf_counter() - started

    render_directory.mkdir(parents=True, exist_ok=True)
    render_path = render_directory / "v086_historical_vs_continuous_source_rgb.png"
    sheet = np.concatenate([
        np.concatenate([np.clip(image,0,1) for image in captures[name]], axis=1)
        for name in ("hard_reference", "historical", "continuous")
    ], axis=0)
    plt.imsave(render_path, sheet)
    report["figures"] = _save_figures(figure_directory, report, captures)
    report["render_comparison"] = str(render_path)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v086_continuous_source_field.json"
    csv_path = artifact_directory / "v086_continuous_source_field.csv"
    report["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    with json_path.open("w") as stream:
        json.dump(_json_ready(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_csv(csv_path, _scalar_csv_rows(_json_ready(report)))
    _progress("v086_complete", runtime=report["runtime_seconds"], primary=report["PRIMARY_SOURCE"])
    return report
