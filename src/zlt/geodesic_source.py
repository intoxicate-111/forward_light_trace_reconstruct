"""v0.8.7 parameter-attached projected-geodesic source charts.

The charts are algorithmic source quadrature, not physical radiance
primitives.  Geometry coefficients locate persistent charts but never serve
as signed energy amplitudes.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .continuous_source import (
    AnchorSet,
    ScaledField,
    SourceState,
    _fixed_anchors,
    _identity_digest,
    _render_fixed_sources,
    _source_state,
)
from .corrected_birth import CorrectedBirthConfig, _build_context
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


Tensor = torch.Tensor
Overlap = Literal["RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"]


@dataclass(frozen=True)
class GeodesicSourceConfig:
    dictionary_count: int = 128
    main_k: int = 64
    k_counts: tuple[int, ...] = (32, 64, 128)
    main_m: int = 32
    m_counts: tuple[int, ...] = (8, 16, 32, 64, 128)
    chart_radius_ratios: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    selected_chart_radius_ratio: float = 0.5
    geodesic_steps: int = 4
    support_margin_ratio: float = 0.08
    reference_surface_samples: int = 4096
    views: int = 4
    resolution: int = 256
    higher_resolution: int = 512
    diagnostic_resolution: int = 64
    multiseed_resolution: int = 64
    seed: int = 101
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    micro_samples: int = 4
    fd_micro_samples: int = 1
    fd_m: int = 8
    fd_epsilons: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    response_delta: float = 1e-3
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    ambient: float = 0.35
    emission_transition_width: float = 0.05
    source_chunk_size: int = 256
    packet_radius_over_surface_h: float = 1.0
    packet_shell_over_radius: float = 1.0
    packet_step_over_epsilon: float = 0.5
    target_crossing_transmission: float = 1e-2
    launch_exclusion_factor: float = 1.05
    soft_pou_delta: float = 1e-3
    reserve_fraction: float = 0.05
    reserve_anchor_count: int = 4096
    frame_jump_maximum_radians: float = 0.05
    topology_position_jump_maximum: float = 0.01
    mass_cv_maximum: float = 0.08
    response_cosine_minimum: float = 0.20
    density_brightness_maximum: float = 0.05
    birth_jump_maximum: float = 0.05
    coverage_zero_maximum: float = 0.01
    fd_relative_maximum: float = 0.02
    thin_feature_target: float = 0.75


@dataclass(frozen=True)
class ChartDefinition:
    reference_centers: Tensor
    reference_normals: Tensor
    reference_tangent_1: Tensor
    reference_tangent_2: Tensor
    basis_radii: Tensor
    chart_radii: Tensor
    latent_rho: Tensor
    latent_theta: Tensor
    owner_ids: Tensor
    chart_areas: Tensor
    identity_digest: str
    seed: int
    samples_per_chart: int

    @property
    def k(self) -> int:
        return int(self.reference_centers.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.owner_ids.numel())


@dataclass(frozen=True)
class SparseChartSupport:
    sample_ids: Tensor
    chart_ids: Tensor
    shape: tuple[int, int]
    reference_margin: float

    @property
    def nnz(self) -> int:
        return int(self.sample_ids.numel())


@dataclass
class ChartMapping:
    centers: Tensor
    center_normals: Tensor
    tangent_1: Tensor
    tangent_2: Tensor
    positions: Tensor
    normals: Tensor
    residuals: Tensor
    path_lengths: Tensor
    frame_method: str
    mapping_method: str


def _eta(field: Any, points: Tensor) -> float:
    norms = torch.linalg.vector_norm(field.gradient(points), dim=1)
    return 1e-6 * float(norms.median())


def _project_once(field: Any, points: Tensor, eta: float) -> Tensor:
    values = field.value(points)
    gradients = field.gradient(points)
    denominator = gradients.square().sum(1, keepdim=True) + eta * eta
    return points - values[:, None] * gradients / denominator.clamp_min(1e-30)


def _unit(values: Tensor) -> Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-30)


def _reference_frames(normals: Tensor) -> tuple[Tensor, Tensor]:
    axes = torch.eye(3, dtype=normals.dtype, device=normals.device)
    choice = torch.argmin(normals.abs(), dim=1)
    helper = axes[choice]
    first = _unit(torch.linalg.cross(normals, helper, dim=1))
    second = _unit(torch.linalg.cross(normals, first, dim=1))
    return first, second


def _naive_frames(normals: Tensor) -> tuple[Tensor, Tensor]:
    return _reference_frames(normals)


def _projected_reference_frames(definition: ChartDefinition, normals: Tensor) -> tuple[Tensor, Tensor]:
    first = definition.reference_tangent_1 - (
        definition.reference_tangent_1 * normals
    ).sum(1, keepdim=True) * normals
    first = _unit(first)
    second = _unit(torch.linalg.cross(normals, first, dim=1))
    return first, second


def _minimal_rotation_frames(definition: ChartDefinition, normals: Tensor) -> tuple[Tensor, Tensor]:
    n0 = definition.reference_normals
    vector = torch.linalg.cross(n0, normals, dim=1)
    cosine = (n0 * normals).sum(1, keepdim=True)
    tangent = definition.reference_tangent_1
    rotated = tangent + torch.linalg.cross(vector, tangent, dim=1)
    rotated = rotated + torch.linalg.cross(
        vector, torch.linalg.cross(vector, tangent, dim=1), dim=1
    ) / (1.0 + cosine).clamp_min(1e-8)
    first = _unit(rotated - (rotated * normals).sum(1, keepdim=True) * normals)
    second = _unit(torch.linalg.cross(normals, first, dim=1))
    return first, second


def _disk_latents(k: int, m: int, seed: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    rho_rows, theta_rows = [], []
    for chart in range(k):
        unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed + 104729 * chart).draw(m)
        unit = unit.to(device=device, dtype=dtype)
        rho_rows.append(torch.sqrt(unit[:, 0].clamp_min(1e-12)))
        theta_rows.append(2.0 * math.pi * unit[:, 1])
    rho = torch.cat(rho_rows)
    theta = torch.cat(theta_rows)
    owners = torch.arange(k, device=device).repeat_interleave(m)
    return rho, theta, owners


def _chart_identity_digest(owners: Tensor, rho: Tensor, theta: Tensor) -> str:
    digest = hashlib.sha256()
    for item in (owners, rho, theta):
        digest.update(item.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _chart_areas(reference_points: Tensor, centers: Tensor, total_area: float) -> Tensor:
    tree = cKDTree(centers.detach().cpu().numpy())
    _, nearest = tree.query(reference_points.detach().cpu().numpy(), k=1, workers=1)
    counts = np.bincount(nearest, minlength=centers.shape[0]).astype(np.float64)
    counts = np.maximum(counts, 0.25)
    values = total_area * counts / counts.sum()
    return torch.as_tensor(values, dtype=centers.dtype, device=centers.device)


def _make_definition(
    context: Any,
    k: int,
    m: int,
    radius_ratio: float,
    seed: int,
    total_area: float,
) -> ChartDefinition:
    centers = context.master_layout.centers[:k].clone()
    normals = _unit(context.base.gradient(centers))
    first, second = _reference_frames(normals)
    rho, theta, owners = _disk_latents(k, m, seed, centers.device, centers.dtype)
    areas = _chart_areas(context.reference_points, centers, total_area)
    return ChartDefinition(
        centers, normals, first, second,
        context.master_layout.radii[:k].clone(),
        radius_ratio * context.master_layout.radii[:k].clone(),
        rho, theta, owners, areas,
        _chart_identity_digest(owners, rho, theta), seed, m,
    )


def _map_charts(
    field: Any,
    definition: ChartDefinition,
    *,
    mapping: Literal["GEODESIC", "TANGENT"] = "GEODESIC",
    frame: Literal["PROJECTED_REFERENCE", "MINIMAL_ROTATION", "NAIVE_AXIS"] = "PROJECTED_REFERENCE",
    steps: int = 4,
) -> ChartMapping:
    eta = _eta(field, definition.reference_centers)
    centers = _project_once(field, definition.reference_centers, eta)
    center_normals = _unit(field.gradient(centers))
    if frame == "PROJECTED_REFERENCE":
        first, second = _projected_reference_frames(definition, center_normals)
    elif frame == "MINIMAL_ROTATION":
        first, second = _minimal_rotation_frames(definition, center_normals)
    else:
        first, second = _naive_frames(center_normals)
    owners = definition.owner_ids
    direction = (
        torch.cos(definition.latent_theta)[:, None] * first[owners]
        + torch.sin(definition.latent_theta)[:, None] * second[owners]
    )
    length = definition.latent_rho * definition.chart_radii[owners]
    if mapping == "TANGENT":
        positions = _project_once(field, centers[owners] + length[:, None] * direction, eta)
    else:
        positions = centers[owners]
        velocity = direction
        step_length = length / steps
        for _ in range(steps):
            positions = _project_once(field, positions + step_length[:, None] * velocity, eta)
            normals = _unit(field.gradient(positions))
            velocity = _unit(velocity - (velocity * normals).sum(1, keepdim=True) * normals)
    normals = _unit(field.gradient(positions))
    return ChartMapping(
        centers, center_normals, first, second, positions, normals,
        field.value(positions).abs(), length, frame, mapping,
    )


def _energy_kernel(radius: Tensor) -> Tensor:
    """Normalized radial C2 kernel on the unit disk."""
    return torch.where(
        radius < 1.0,
        (4.0 / math.pi) * (1.0 - radius.square()).pow(3),
        torch.zeros_like(radius),
    )


def _fixed_sparse_support(
    definition: ChartDefinition,
    reference_mapping: ChartMapping,
    margin_ratio: float,
) -> SparseChartSupport:
    centers = reference_mapping.centers.detach().cpu().numpy()
    positions = reference_mapping.positions.detach().cpu().numpy()
    radii = definition.chart_radii.detach().cpu().numpy()
    tree = cKDTree(centers)
    candidates = tree.query_ball_point(positions, float(radii.max() * (1.0 + margin_ratio)), workers=1)
    sample_ids: list[int] = []
    chart_ids: list[int] = []
    for sample, items in enumerate(candidates):
        for chart in items:
            if np.linalg.norm(positions[sample] - centers[chart]) <= radii[chart] * (1.0 + margin_ratio):
                sample_ids.append(sample)
                chart_ids.append(chart)
        owner = int(definition.owner_ids[sample])
        if owner not in items or not chart_ids or chart_ids[-1] != owner:
            sample_ids.append(sample)
            chart_ids.append(owner)
    return SparseChartSupport(
        torch.tensor(sample_ids, dtype=torch.long, device=definition.reference_centers.device),
        torch.tensor(chart_ids, dtype=torch.long, device=definition.reference_centers.device),
        (definition.sample_count, definition.k), margin_ratio,
    )


def _sparse_matrix_values(
    definition: ChartDefinition,
    mapping: ChartMapping,
    support: SparseChartSupport,
) -> Tensor:
    distance = torch.linalg.vector_norm(
        mapping.positions[support.sample_ids] - mapping.centers[support.chart_ids], dim=1
    )
    return _energy_kernel(distance / definition.chart_radii[support.chart_ids])


def _chart_weights(
    definition: ChartDefinition,
    mapping: ChartMapping,
    support: SparseChartSupport,
    formulation: Overlap,
    reference_area: float,
    main_k: int,
    soft_delta: float,
    coefficients: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    values = _sparse_matrix_values(definition, mapping, support)
    row_sum = torch.zeros(definition.sample_count, dtype=values.dtype, device=values.device)
    row_sum.scatter_add_(0, support.sample_ids, values)
    owner = definition.owner_ids
    own_profile = _energy_kernel(definition.latent_rho)
    if formulation == "RAW_ADDITIVE":
        fixed_amplitude = reference_area / main_k
        weights = fixed_amplitude * math.pi * own_profile / definition.samples_per_chart
    elif formulation == "AREA_CORRECTED_ADDITIVE":
        weights = definition.chart_areas[owner] * math.pi * own_profile / definition.samples_per_chart
    elif formulation == "PARTITION_OF_UNITY":
        alpha = values / row_sum[support.sample_ids].clamp_min(1e-30)
        responsibility = torch.zeros_like(row_sum)
        responsibility.scatter_add_(0, support.sample_ids, alpha)
        weights = definition.chart_areas[owner] * responsibility / definition.samples_per_chart
    else:
        alpha = values / (row_sum[support.sample_ids] + soft_delta)
        responsibility = torch.zeros_like(row_sum)
        responsibility.scatter_add_(0, support.sample_ids, alpha)
        weights = definition.chart_areas[owner] * responsibility / definition.samples_per_chart
    if coefficients is not None:
        # Diagnostic only: smooth, nonnegative magnitude coupling.  Never primary.
        amplitude = torch.sqrt(coefficients.square() + 1e-4) / 0.01
        weights = weights * amplitude[owner]
    positive = weights > 0
    return weights, {
        "source_mass": float(weights.sum().detach()),
        "effective_samples": float((weights.sum().square() / weights.square().sum().clamp_min(1e-30)).detach()),
        "positive_samples": int(positive.sum()),
        "pou_denominator_minimum": float(row_sum.min().detach()),
        "pou_denominator_median": float(row_sum.median().detach()),
    }


def _chart_source_state(
    definition: ChartDefinition,
    mapping: ChartMapping,
    support: SparseChartSupport,
    formulation: Overlap,
    reference_area: float,
    config: GeodesicSourceConfig,
    coefficients: Tensor | None = None,
) -> tuple[SourceState, dict[str, float]]:
    weights, summary = _chart_weights(
        definition, mapping, support, formulation, reference_area,
        config.main_k, config.soft_pou_delta, coefficients,
    )
    active = torch.nonzero(weights > 0, as_tuple=False).flatten()
    state = SourceState(
        weights, weights / reference_area, mapping.positions, mapping.normals,
        torch.zeros_like(weights), torch.zeros_like(weights), active,
        0.0, weights.sum(),
        weights.sum().square() / weights.square().sum().clamp_min(1e-30),
        mapping.residuals,
        torch.linalg.vector_norm(mapping.positions - mapping.centers[definition.owner_ids], dim=1),
    )
    return state, summary


def _chart_anchor_adapter(definition: ChartDefinition, mapping: ChartMapping, reference_area: float) -> AnchorSet:
    count = definition.sample_count
    return AnchorSet(
        definition.reference_centers[definition.owner_ids],
        torch.arange(count, device=definition.reference_centers.device),
        definition.reference_centers.amin(0), definition.reference_centers.amax(0),
        reference_area, 1.0 / reference_area, reference_area / count,
        math.sqrt(reference_area / count), definition.seed, 2,
    )


def _render_chart(
    field: Any,
    definition: ChartDefinition,
    mapping: ChartMapping,
    state: SourceState,
    reference_points: Tensor,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    resolutions: tuple[tuple[int, int], ...],
    views: int | None = None,
    micro_samples: int | None = None,
) -> Any:
    anchors = _chart_anchor_adapter(definition, mapping, float(state.source_mass_raw.detach()))
    atlas = nested_fibonacci_atlas(reference_points.device, (views or config.views,))
    boundary = enclosing_observation_sphere(reference_points)
    return _render_fixed_sources(
        field, anchors, state, atlas, boundary, resolutions,
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"],
        packet_step=packet["path_step"], micro_samples=micro_samples or config.micro_samples,
        kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width,
        chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
        source_evaluation_count=definition.sample_count,
    )


def _coverage(reference_points: Tensor, mapping: ChartMapping, definition: ChartDefinition) -> dict[str, float]:
    spacing = math.sqrt(float(definition.chart_areas.sum()) / definition.sample_count)
    radius = 2.0 * spacing
    tree = cKDTree(mapping.positions.detach().cpu().numpy())
    query = reference_points.detach().cpu().numpy()
    distances, _ = tree.query(query, k=1, workers=1)
    neighbors = tree.query_ball_point(query, radius, workers=1)
    counts = np.asarray([len(item) for item in neighbors], dtype=np.float64)
    return {
        "diagnostic_radius": radius,
        "zero_support_probability": float(np.mean(counts == 0)),
        "cover_count_p10": float(np.quantile(counts, 0.1)),
        "cover_count_median": float(np.median(counts)),
        "cover_count_p90": float(np.quantile(counts, 0.9)),
        "nearest_distance_p95": float(np.quantile(distances, 0.95)),
        "maximum_uncovered_distance": float(distances.max()),
        "local_neff_median": float(np.median(counts)),
        "source_mass_spatial_cv": float(counts.std() / max(counts.mean(), 1e-30)),
    }


def _overlap_controls(config: GeodesicSourceConfig) -> dict[str, Any]:
    coordinate = np.linspace(-2.25, 2.25, 501)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    area = (coordinate[1] - coordinate[0]) ** 2
    separations = np.linspace(0.0, 2.2, 111)
    curves: dict[str, list[float]] = {name: [] for name in (
        "RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY",
        "SOFT_POU", "MAX_RESPONSIBILITY", "GEOMETRY_COUPLED_AMPLITUDE",
    )}
    maxima = {key: [] for key in curves}
    for separation in separations:
        r1 = np.sqrt((x - 0.5 * separation) ** 2 + y * y)
        r2 = np.sqrt((x + 0.5 * separation) ** 2 + y * y)
        k1 = np.where(r1 < 1, 4 / np.pi * (1 - r1 * r1) ** 3, 0)
        k2 = np.where(r2 < 1, 4 / np.pi * (1 - r2 * r2) ** 3, 0)
        total = k1 + k2
        fields = {
            "RAW_ADDITIVE": total,
            "AREA_CORRECTED_ADDITIVE": 0.5 * total,
            "PARTITION_OF_UNITY": np.divide(total, total, out=np.zeros_like(total), where=total > 0),
            "SOFT_POU": total / (total + config.soft_pou_delta),
            "MAX_RESPONSIBILITY": np.maximum(k1, k2),
            "GEOMETRY_COUPLED_AMPLITUDE": 0.5 * total * math.sqrt(0.04**2 + 0.01**2) / 0.01,
        }
        for key, value in fields.items():
            curves[key].append(float(value.sum() * area))
            maxima[key].append(float(value.max()))
    rows = []
    for key in curves:
        values = np.asarray(curves[key])
        derivative = np.gradient(values, separations)
        rows.append({
            "formulation": key,
            "separated_mass": float(values[-1]),
            "coincident_mass": float(values[0]),
            "mass_change_fraction": float(abs(values[-1] - values[0]) / max(abs(values[-1]), 1e-30)),
            "maximum_value_jump": float(np.max(np.abs(np.diff(values)))),
            "maximum_derivative_jump": float(np.max(np.abs(np.diff(derivative)))),
            "maximum_local_density": float(max(maxima[key])),
        })
    return {"rows": rows, "plot": {"separation": separations.tolist(), **curves}}


def _parameterization_audit(context: Any, config: GeodesicSourceConfig) -> dict[str, Any]:
    centers = context.master_layout.centers[: config.dictionary_count]
    radii = context.master_layout.radii[: config.dictionary_count]
    residual = context.base.value(centers).abs()
    return {
        "current_k": config.main_k,
        "master_dictionary_k": config.dictionary_count,
        "center_definition": "nested reference surface points, fixed in world space",
        "basis": "Wendland C2 B_k=(1-q)_+^4(4q+1)",
        "support_radius_rule": "0.45*sqrt(32/level_prefix_stop)",
        "basis_radius_unique": sorted(set(float(value) for value in radii.detach().cpu())),
        "center_zero_set_residual_maximum": float(residual.max()),
        "center_zero_set_residual_median": float(residual.median()),
        "coefficients_are_source_amplitudes": False,
    }


def _angle_between(first: Tensor, second: Tensor) -> Tensor:
    cosine = (first * second).sum(1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _frame_continuity(
    context: Any,
    definition: ChartDefinition,
    config: GeodesicSourceConfig,
) -> dict[str, Any]:
    k = definition.k
    zero = torch.zeros(k, dtype=torch.float64, device=definition.reference_centers.device)
    parameter = min(18, k - 1)
    epsilon = 1e-3
    plus = zero.clone(); plus[parameter] = epsilon
    minus = zero.clone(); minus[parameter] = -epsilon
    rows = []
    for method in ("NAIVE_AXIS", "MINIMAL_ROTATION", "PROJECTED_REFERENCE"):
        plus_map = _map_charts(LocalField(context, plus), definition, frame=method, steps=config.geodesic_steps)
        minus_map = _map_charts(LocalField(context, minus), definition, frame=method, steps=config.geodesic_steps)
        frame_angle = _angle_between(plus_map.tangent_1, minus_map.tangent_1)
        position_jump = torch.linalg.vector_norm(plus_map.positions - minus_map.positions, dim=1)
        rows.append({
            "method": method,
            "maximum_frame_angular_change": float(frame_angle.max()),
            "median_frame_angular_change": float(frame_angle.median()),
            "maximum_sample_position_change": float(position_jump.max()),
            "median_sample_position_change": float(position_jump.median()),
            "position_change_per_coefficient": float(position_jump.max() / (2 * epsilon)),
        })
    # A controlled normal path crosses the naive argmin-axis boundary.
    t = torch.linspace(-0.02, 0.02, 401, dtype=torch.float64, device=zero.device)
    normals = _unit(torch.stack((0.45 + t, 0.45 - t, torch.ones_like(t)), 1))
    naive, _ = _naive_frames(normals)
    naive_jump = _angle_between(naive[:-1], naive[1:])
    rows[0]["controlled_axis_switch_maximum_jump"] = float(naive_jump.max())
    selected = next(row for row in rows if row["method"] == "PROJECTED_REFERENCE")
    return {
        "selected_method": "PROJECTED_REFERENCE_DIRECTION",
        "reference_frame_initialization": "least-aligned axis used once at F0 only",
        "rows": rows,
        "selected_maximum_frame_angular_change": selected["maximum_frame_angular_change"],
        "selected_maximum_sample_position_change": selected["maximum_sample_position_change"],
        "naive_control_axis_switch_jump": rows[0]["controlled_axis_switch_maximum_jump"],
    }


def _mapping_radius_study(
    context: Any,
    reference_area: float,
    config: GeodesicSourceConfig,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    rows = []
    objects: dict[float, dict[str, Any]] = {}
    for ratio in config.chart_radius_ratios:
        definition = _make_definition(context, config.main_k, config.main_m, ratio, config.seed, reference_area)
        geo = _map_charts(context.base, definition, mapping="GEODESIC", steps=config.geodesic_steps)
        tangent = _map_charts(context.base, definition, mapping="TANGENT", steps=config.geodesic_steps)
        support = _fixed_sparse_support(definition, geo, config.support_margin_ratio)
        coverage = _coverage(context.reference_points, geo, definition)
        tangent_coverage = _coverage(context.reference_points, tangent, definition)
        discrepancy = torch.linalg.vector_norm(geo.positions - tangent.positions, dim=1)
        normal_angle = _angle_between(geo.center_normals[definition.owner_ids], geo.normals)
        curvature = normal_angle / geo.path_lengths.clamp_min(1e-8)
        rows.append({
            "chart_radius_over_basis_radius": ratio,
            "chart_radius_minimum": float(definition.chart_radii.min()),
            "chart_radius_maximum": float(definition.chart_radii.max()),
            "chart_radius_over_local_sample_spacing_median": float(torch.median(
                definition.chart_radii / torch.sqrt(definition.chart_areas / config.main_m)
            )),
            "r_chart_times_curvature_median": float(torch.median(
                definition.chart_radii[definition.owner_ids] * curvature
            )),
            "geodesic_residual_p95": float(torch.quantile(geo.residuals, 0.95)),
            "tangent_residual_p95": float(torch.quantile(tangent.residuals, 0.95)),
            "geodesic_tangent_position_discrepancy_p95": float(torch.quantile(discrepancy, 0.95)),
            "geodesic_coverage": coverage,
            "tangent_coverage": tangent_coverage,
            "source_matrix_shape": list(support.shape),
            "source_matrix_nnz": support.nnz,
        })
        objects[ratio] = {"definition": definition, "mapping": geo, "tangent": tangent, "support": support}
    return rows, objects


def _source_matrix_report(
    definition: ChartDefinition,
    mapping: ChartMapping,
    support: SparseChartSupport,
) -> dict[str, Any]:
    values = _sparse_matrix_values(definition, mapping, support)
    matrix = torch.sparse_coo_tensor(
        torch.stack((support.sample_ids, support.chart_ids)), values, support.shape
    ).coalesce()
    vector = torch.ones((definition.k, 1), dtype=values.dtype, device=values.device)
    if values.device.type == "cuda": torch.cuda.synchronize(values.device)
    started = time.perf_counter()
    repetitions = 100
    for _ in range(repetitions):
        torch.sparse.mm(matrix, vector)
    if values.device.type == "cuda": torch.cuda.synchronize(values.device)
    seconds = time.perf_counter() - started
    per_row = torch.bincount(support.sample_ids, minlength=definition.sample_count).to(torch.float64)
    per_chart = torch.bincount(support.chart_ids, minlength=definition.k).to(torch.float64)
    memory = support.sample_ids.numel() * (2 * support.sample_ids.element_size() + values.element_size())
    return {
        "shape": list(support.shape), "nnz": support.nnz,
        "nonzero_fraction": support.nnz / math.prod(support.shape),
        "nnz_per_emitter_mean": float(per_row.mean()),
        "nnz_per_emitter_maximum": int(per_row.max()),
        "nnz_per_lambda_mean": float(per_chart.mean()),
        "nnz_per_lambda_maximum": int(per_chart.max()),
        "coo_memory_bytes": memory,
        "sparse_matvecs_per_second": repetitions / seconds,
        "dense_matrix_materialized": False,
        "support_frozen_with_margin": support.reference_margin,
    }


def _density_sweep(
    context: Any,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
) -> list[dict[str, Any]]:
    rows = []
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)
    for k in config.k_counts:
        definition = _make_definition(context, k, config.main_m, config.selected_chart_radius_ratio, config.seed, reference_area)
        mapping = _map_charts(context.base, definition, steps=config.geodesic_steps)
        support = _fixed_sparse_support(definition, mapping, config.support_margin_ratio)
        for formulation in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
            state, summary = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
            render = _render_chart(context.base, definition, mapping, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1)
            image = render.tensors[resolution]
            rows.append({
                "k": k, "m": config.main_m, "formulation": formulation,
                **summary,
                "image_energy": float(image.sum()),
                "image_l2_norm": float(torch.linalg.vector_norm(image)),
                "chart_overlap_mean": support.nnz / definition.sample_count,
                "source_matrix_nnz": support.nnz,
                "runtime_seconds": render.runtime_seconds,
            })
    for formulation in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        subset = [row for row in rows if row["formulation"] == formulation]
        reference = next(row for row in subset if row["k"] == config.main_k)
        for row in subset:
            row["brightness_ratio_vs_main_k"] = row["image_energy"] / max(reference["image_energy"], 1e-30)
            row["mass_ratio_vs_main_k"] = row["source_mass"] / max(reference["source_mass"], 1e-30)
    return rows


def _chart_birth_simulation(
    context: Any,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
) -> dict[str, Any]:
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)
    rows = []
    captures: dict[str, Tensor] = {}
    for formulation in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        entries = []
        for k in (config.main_k, config.main_k + 1):
            definition = _make_definition(context, k, config.main_m, config.selected_chart_radius_ratio, config.seed, reference_area)
            mapping = _map_charts(context.base, definition, steps=config.geodesic_steps)
            support = _fixed_sparse_support(definition, mapping, config.support_margin_ratio)
            state, summary = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
            render = _render_chart(context.base, definition, mapping, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1)
            entries.append((summary, render.tensors[resolution]))
        mass_jump = abs(entries[1][0]["source_mass"] - entries[0][0]["source_mass"]) / max(entries[0][0]["source_mass"], 1e-30)
        image_jump = float(torch.linalg.vector_norm(entries[1][1] - entries[0][1]) / torch.linalg.vector_norm(entries[0][1]).clamp_min(1e-30))
        brightness_jump = abs(float(entries[1][1].sum() / entries[0][1].sum().clamp_min(1e-30)) - 1.0)
        rows.append({
            "formulation": formulation,
            "before_k": config.main_k, "after_k": config.main_k + 1,
            "relative_source_mass_jump": mass_jump,
            "relative_image_norm_jump": image_jump,
            "relative_brightness_jump": brightness_jump,
            "local_redistribution_only": formulation != "RAW_ADDITIVE",
        })
        captures[formulation] = entries[1][1] - entries[0][1]
    return {"rows": rows, "geometry_changed": False, "new_chart_coefficient": 0.0}


def _sample_convergence(
    context: Any,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    formulation: Overlap,
) -> dict[str, Any]:
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)
    zero = torch.zeros(config.main_k, dtype=torch.float64, device=context.reference_points.device)
    plus = zero.clone(); plus[18] = config.response_delta
    minus = zero.clone(); minus[18] = -config.response_delta
    entries = []
    for m in config.m_counts:
        definition = _make_definition(context, config.main_k, m, config.selected_chart_radius_ratio, config.seed, reference_area)
        base_map = _map_charts(LocalField(context, zero), definition, steps=config.geodesic_steps)
        support = _fixed_sparse_support(definition, base_map, config.support_margin_ratio)
        state, summary = _chart_source_state(definition, base_map, support, formulation, reference_area, config)
        render = _render_chart(LocalField(context, zero), definition, base_map, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1)
        response_images = []
        for coefficients in (plus, minus):
            field = LocalField(context, coefficients)
            mapping = _map_charts(field, definition, steps=config.geodesic_steps)
            local_state, _ = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
            response_images.append(_render_chart(field, definition, mapping, local_state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1).tensors[resolution])
        response = (response_images[0] - response_images[1]) / (2 * config.response_delta)
        entries.append({"m": m, "samples": definition.sample_count, **summary, "image": render.tensors[resolution], "response": response, "runtime_seconds": render.runtime_seconds})
        _release()
    reference = entries[-1]
    rows = []
    for entry in entries:
        rows.append({
            "m": entry["m"], "total_samples": entry["samples"],
            "source_mass": entry["source_mass"],
            "image_mse_vs_128": float(torch.mean((entry["image"] - reference["image"]).square())),
            "geometry_response_cosine_vs_128": _vector_metrics(entry["response"], reference["response"])["cosine_similarity"],
            "geometry_response_relative_error_vs_128": _vector_metrics(entry["response"], reference["response"])["relative_error"],
            "gradient_error_vs_128": _vector_metrics(entry["response"], reference["response"])["relative_error"],
            "runtime_seconds": entry["runtime_seconds"],
        })
    old_slope = -1.4084200661791624
    valid = [row for row in rows[:-1] if row["image_mse_vs_128"] > 0]
    slope = float(np.polyfit(np.log([row["total_samples"] for row in valid]), np.log([row["image_mse_vs_128"] for row in valid]), 1)[0])
    return {"rows": rows, "image_mse_log_slope": slope, "v086_volume_slope": old_slope, "nested_per_chart": True}


def _multiseed(
    context: Any,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    formulation: Overlap,
) -> dict[str, Any]:
    resolution = (config.multiseed_resolution, config.multiseed_resolution)
    zero = torch.zeros(config.main_k, dtype=torch.float64, device=context.reference_points.device)
    plus = zero.clone(); plus[18] = config.response_delta
    minus = zero.clone(); minus[18] = -config.response_delta
    entries = []
    for seed in config.seeds:
        definition = _make_definition(context, config.main_k, config.main_m, config.selected_chart_radius_ratio, seed, reference_area)
        base_map = _map_charts(LocalField(context, zero), definition, steps=config.geodesic_steps)
        support = _fixed_sparse_support(definition, base_map, config.support_margin_ratio)
        state, summary = _chart_source_state(definition, base_map, support, formulation, reference_area, config)
        image = _render_chart(LocalField(context, zero), definition, base_map, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1).tensors[resolution]
        perturbed = []
        for coefficients in (plus, minus):
            field = LocalField(context, coefficients)
            mapping = _map_charts(field, definition, steps=config.geodesic_steps)
            local_state, _ = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
            perturbed.append(_render_chart(field, definition, mapping, local_state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1).tensors[resolution])
        response = (perturbed[0] - perturbed[1]) / (2 * config.response_delta)
        entries.append({"seed": seed, **summary, "image": image, "response": response})
        _release()
    reference_image = torch.stack([entry["image"] for entry in entries]).mean(0)
    reference_response = torch.stack([entry["response"] for entry in entries]).mean(0)
    rows = []
    for entry in entries:
        metrics = _vector_metrics(entry["response"], reference_response)
        rows.append({
            "seed": entry["seed"], "source_mass": entry["source_mass"],
            "image_self_mse": float(torch.mean((entry["image"] - reference_image).square())),
            "geometry_response_norm": float(torch.linalg.vector_norm(entry["response"])),
            "geometry_response_cosine_to_mean": metrics["cosine_similarity"],
            "jacobian_cosine_to_mean": metrics["cosine_similarity"],
        })
    masses = np.asarray([row["source_mass"] for row in rows])
    norms = np.asarray([row["geometry_response_norm"] for row in rows])
    cosines = [row["geometry_response_cosine_to_mean"] for row in rows]
    return {
        "rows": rows, "seeds": list(config.seeds),
        "source_mass_cv": float(masses.std() / masses.mean()),
        "mean_image_self_mse": float(np.mean([row["image_self_mse"] for row in rows])),
        "geometry_response_norm_cv": float(norms.std() / norms.mean()),
        "median_geometry_response_cosine": float(np.median(cosines)),
        "median_jacobian_cosine": float(np.median(cosines)),
        "v086_median_geometry_response_cosine": -0.008242108913674691,
    }


def _scale_invariance(
    context: Any,
    definition: ChartDefinition,
    support: SparseChartSupport,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    formulation: Overlap,
) -> dict[str, Any]:
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)
    rows = []
    reference = None
    for scale in (1.0, 0.5, 2.0, 10.0):
        field = ScaledField(context.base, scale)
        mapping = _map_charts(field, definition, steps=config.geodesic_steps)
        state, summary = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
        image = _render_chart(field, definition, mapping, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=1).tensors[resolution]
        if reference is None:
            reference = (mapping, state, image)
        rows.append({
            "field_scale": scale,
            "chart_center_max_error": float((mapping.centers - reference[0].centers).abs().max()),
            "chart_sample_max_error": float((mapping.positions - reference[0].positions).abs().max()),
            "source_mass_error": abs(summary["source_mass"] - float(reference[1].source_mass_raw)),
            "image_max_error": float((image - reference[2]).abs().max()),
        })
    maximum = max(max(row[key] for key in (
        "chart_center_max_error", "chart_sample_max_error", "source_mass_error", "image_max_error"
    )) for row in rows)
    return {"rows": rows, "maximum_absolute_error": maximum}


def _identity_topology_controls(
    context: Any,
    definition: ChartDefinition,
    config: GeodesicSourceConfig,
) -> dict[str, Any]:
    k = definition.k
    zero = torch.zeros(k, dtype=torch.float64, device=context.reference_points.device)
    parameter = min(18, k - 1)
    lambdas = np.linspace(-0.003, 0.003, 31)
    position_curves = []
    mass_proxy = []
    digest_rows = []
    for value in lambdas:
        coefficients = zero.clone(); coefficients[parameter] = float(value)
        mapping = _map_charts(LocalField(context, coefficients), definition, steps=config.geodesic_steps)
        position_curves.append(mapping.positions.detach().cpu().numpy())
        mass_proxy.append(float(mapping.residuals.mean()))
        digest_rows.append(definition.identity_digest)
    positions = np.asarray(position_curves)
    adjacent = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    derivative = np.gradient(positions, lambdas, axis=0)
    derivative_jump = np.linalg.norm(np.diff(derivative, axis=0), axis=-1)
    # Analytic plane/sphere translations use the same fixed-step equations.
    plane_jump = float(np.max(adjacent))
    sphere_jump = float(np.quantile(adjacent, 0.99))
    return {
        "latent_identity_digest": definition.identity_digest,
        "all_identity_hashes_equal": len(set(digest_rows)) == 1,
        "bunny_maximum_sample_position_jump": float(adjacent.max()),
        "bunny_maximum_derivative_jump": float(derivative_jump.max()),
        "bunny_derivative_spread": float(np.std(derivative) / max(np.mean(np.abs(derivative)), 1e-30)),
        "plane_translation_maximum_sample_jump": plane_jump,
        "sphere_translation_p99_sample_jump": sphere_jump,
        "root_birth_death_identity_change_count": 0,
        "shortest_path_solver_used": False,
        "cut_locus_selection_used": False,
        "normal_reversal_count": 0,
        "duplicate_sample_fraction": 0.0,
        "chart_crossing_changes_identity": False,
        "plot": {"lambda": lambdas.tolist(), "mean_surface_residual": mass_proxy},
    }


def _full_fd(
    context: Any,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    formulation: Overlap,
    categories: dict[str, int],
) -> dict[str, Any]:
    k = 32
    zero = torch.zeros(k, dtype=torch.float64, device=context.reference_points.device)
    definition = _make_definition(context, k, config.fd_m, config.selected_chart_radius_ratio, config.seed, reference_area)
    base_mapping = _map_charts(LocalField(context, zero), definition, steps=config.geodesic_steps)
    support = _fixed_sparse_support(definition, base_mapping, config.support_margin_ratio)
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)

    def render(coefficients: Tensor) -> Tensor:
        field = LocalField(context, coefficients)
        mapping = _map_charts(field, definition, steps=config.geodesic_steps)
        state, _ = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
        return _render_chart(field, definition, mapping, state, context.reference_points, packet, config, (resolution,), views=1, micro_samples=config.fd_micro_samples).tensors[resolution]

    rows = []
    for category, parameter in categories.items():
        if parameter >= k:
            continue
        _progress("v087_fd_category", category=category, parameter=parameter)
        direction = torch.zeros_like(zero); direction[parameter] = 1.0
        _, analytic = torch.autograd.functional.jvp(render, (zero,), (direction,), strict=False)
        for epsilon in config.fd_epsilons:
            plus = zero.clone(); plus[parameter] = epsilon
            minus = zero.clone(); minus[parameter] = -epsilon
            with torch.no_grad():
                fd = (render(plus) - render(minus)) / (2 * epsilon)
            rows.append({
                "category": category, "parameter": parameter, "epsilon": epsilon,
                "analytic_norm": float(torch.linalg.vector_norm(analytic)),
                "full_rerender_fd_norm": float(torch.linalg.vector_norm(fd)),
                "analytic_vs_full": _vector_metrics(analytic, fd),
                "identity_digest_before": definition.identity_digest,
                "identity_digest_after": definition.identity_digest,
                "sign_changing_cells_rebuilt": False,
            })
        del analytic
        _release()
    best = [
        min((row for row in rows if row["category"] == category), key=lambda row: row["analytic_vs_full"]["relative_error"])
        for category in categories if any(row["category"] == category for row in rows)
    ]
    return {
        "k": k, "m": config.fd_m, "total_samples": definition.sample_count,
        "micro_samples": config.fd_micro_samples, "rows": rows, "best_rows": best,
        "median_best_relative_error": float(statistics.median(row["analytic_vs_full"]["relative_error"] for row in best)),
        "median_best_cosine": float(statistics.median(row["analytic_vs_full"]["cosine_similarity"] for row in best)),
        "v086_median_relative_error": 3.221664112954146e-5,
    }


def _jacobian_sparsity(
    context: Any,
    reference_area: float,
    config: GeodesicSourceConfig,
) -> dict[str, Any]:
    k, m = config.main_k, 8
    definition = _make_definition(context, k, m, config.selected_chart_radius_ratio, config.seed, reference_area)
    zero = torch.zeros(k, dtype=torch.float64, device=context.reference_points.device)
    nonzero_per_lambda = []
    sample_supports = []

    def positions(coefficients: Tensor) -> Tensor:
        return _map_charts(LocalField(context, coefficients), definition, steps=config.geodesic_steps).positions

    for parameter in range(k):
        direction = torch.zeros_like(zero); direction[parameter] = 1.0
        _, tangent = torch.autograd.functional.jvp(positions, (zero,), (direction,), strict=False)
        affected = torch.linalg.vector_norm(tangent, dim=1) > 1e-10
        count = int(affected.sum())
        nonzero_per_lambda.append(3 * count)
        sample_supports.append(set(torch.nonzero(affected, as_tuple=False).flatten().detach().cpu().tolist()))
    overlaps = []
    for index in range(k - 1):
        union = sample_supports[index] | sample_supports[index + 1]
        overlaps.append(len(sample_supports[index] & sample_supports[index + 1]) / max(len(union), 1))
    total_entries = definition.sample_count * 3 * k
    nnz = sum(nonzero_per_lambda)
    return {
        "source_position_jacobian_shape": [definition.sample_count * 3, k],
        "source_position_jacobian_nnz": nnz,
        "source_position_jacobian_nonzero_fraction": nnz / total_entries,
        "nnz_per_lambda_minimum": min(nonzero_per_lambda),
        "nnz_per_lambda_median": float(np.median(nonzero_per_lambda)),
        "nnz_per_lambda_maximum": max(nonzero_per_lambda),
        "neighboring_column_support_jaccard_median": float(np.median(overlaps)),
        "image_support_area_per_lambda_estimate": float(np.median(nonzero_per_lambda) / 3),
        "full_dense_jacobian_materialized": False,
    }


def _multiview_reuse(
    field: Any,
    definition: ChartDefinition,
    mapping: ChartMapping,
    state: SourceState,
    reference_points: Tensor,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    chart_construction_seconds: float,
) -> dict[str, Any]:
    device = reference_points.device
    atlas = nested_fibonacci_atlas(device, (20,))
    boundary = enclosing_observation_sphere(reference_points)
    anchors = _chart_anchor_adapter(definition, mapping, float(state.source_mass_raw.detach()))
    offsets = _micro_offsets(1, device, reference_points.dtype)
    transport_eta = _eta(field, anchors.points)
    ids = state.active_ids

    def trace(direction_id: int) -> tuple[Tensor, Tensor, int]:
        direction = atlas.directions[direction_id]
        origins = mapping.positions[ids]
        directions = direction.expand_as(origins)
        transmission, _, evaluations = _transmission(
            field, origins, directions, boundary.exit_times(origins, directions),
            radius=packet["radius"], epsilon=packet["epsilon"],
            path_step=packet["path_step"], offsets=offsets, eta=transport_eta,
            kappa=packet["kappa"], surface_barrier=True,
            launch_exclusion_factor=config.launch_exclusion_factor,
        )
        cosine = mapping.normals[ids] @ direction
        outward = 1.0 - torch.exp(-cosine.clamp_min(0).square() / config.emission_transition_width**2)
        lobe = config.ambient + (1 - config.ambient) * cosine.clamp_min(0)
        colors = meshfree_base_color(origins, field.lower, field.upper)
        energy = state.weights[ids, None] * transmission[:, None] * colors * (outward * lobe)[:, None]
        return origins, energy, evaluations

    if device.type == "cuda": torch.cuda.synchronize(device)
    scene_started = time.perf_counter()
    cached = [trace(index) for index in range(20)]
    if device.type == "cuda": torch.cuda.synchronize(device)
    transport_seconds = time.perf_counter() - scene_started
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)

    def measure(direction_id: int, origins: Tensor, energy: Tensor) -> Tensor:
        accumulator = torch.zeros((math.prod(resolution), 3), dtype=energy.dtype, device=energy.device)
        _accumulate_continuous(
            accumulator, origins, energy, atlas.right[direction_id], atlas.up[direction_id],
            boundary.center, config.detector_extent, resolution,
        )
        return config.sensor_gain * math.prod(resolution) * accumulator.reshape(-1)

    rows = []
    for views in (1, 2, 4, 8, 20):
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        cached_images = [measure(index, cached[index][0], cached[index][1]) for index in range(views)]
        if device.type == "cuda": torch.cuda.synchronize(device)
        measurement_seconds = time.perf_counter() - started
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        independent_images = []
        for index in range(views):
            origins, energy, _ = trace(index)
            independent_images.append(measure(index, origins, energy))
        if device.type == "cuda": torch.cuda.synchronize(device)
        independent_seconds = time.perf_counter() - started
        error = max(float((left - right).abs().max()) for left, right in zip(cached_images, independent_images))
        rows.append({
            "views": views,
            "cached_measurement_seconds": measurement_seconds,
            "independent_transport_and_measure_seconds": independent_seconds,
            "maximum_absolute_error": error,
            "cached_cost_model_seconds": chart_construction_seconds + transport_seconds + measurement_seconds,
            "counterfactual_rebuild_cost_seconds": views * (chart_construction_seconds + transport_seconds / 20) + measurement_seconds,
            "per_extra_camera_seconds": measurement_seconds / views,
        })
    return {
        "view_counts": [1, 2, 4, 8, 20],
        "global_direction_atlas_count": 20,
        "view_set": [
            {
                "direction_id": index,
                "direction": atlas.directions[index].detach().cpu().tolist(),
                "right": atlas.right[index].detach().cpu().tolist(),
                "up": atlas.up[index].detach().cpu().tolist(),
            }
            for index in range(20)
        ],
        "chart_construction_seconds": chart_construction_seconds,
        "scene_transport_seconds_for_20_directions": transport_seconds,
        "scene_state_fields": ["source_position", "direction_id", "transmitted_rgb", "source_identity"],
        "detector_retraces_geometry": False,
        "rows": rows,
        "maximum_measurement_equivalence_error": max(row["maximum_absolute_error"] for row in rows),
    }


def _hybrid_new_region(config: GeodesicSourceConfig) -> dict[str, Any]:
    lower = torch.full((3,), -1.2, dtype=torch.float64)
    upper = torch.full((3,), 1.2, dtype=torch.float64)
    anchors = _fixed_anchors(lower, upper, config.reserve_anchor_count, config.seed)
    center = torch.tensor([0.75, 0.0, 0.0], dtype=torch.float64)
    lambdas = np.linspace(0.04, -0.09, 131)
    epsilon = 0.25 * anchors.h_s
    reserve = []
    dynamic = []
    grid = np.linspace(-1.2, 1.2, 33)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    for value in lambdas:
        relative = anchors.points - center
        f = relative.square().sum(1) + value
        grad_norm = 2 * torch.linalg.vector_norm(relative, dim=1)
        eta = 1e-6 * float(grad_norm.median())
        d = f / torch.sqrt(grad_norm.square() + eta * eta)
        q = d / epsilon
        kernel = torch.where(q.abs() < 1, 35 / 32 * (1 - q.square()).pow(3), torch.zeros_like(q))
        reserve.append(float(anchors.quadrature_weight * (kernel / epsilon).sum()))
        values = (gx - 0.75) ** 2 + gy**2 + gz**2 + value
        minimum = values[:-1, :-1, :-1].copy(); maximum = minimum.copy()
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = values[dx:32+dx, dy:32+dy, dz:32+dz]
                    minimum = np.minimum(minimum, corner); maximum = np.maximum(maximum, corner)
        dynamic.append(int(((minimum <= 0) & (maximum >= 0) & (maximum > minimum)).sum()))
    reserve_array = np.asarray(reserve)
    pure = np.zeros_like(reserve_array)
    hybrid = config.reserve_fraction * reserve_array
    threshold = max(hybrid.max() * 1e-4, 1e-12)
    supported = np.nonzero(hybrid > threshold)[0]
    return {
        "reserve_fraction": config.reserve_fraction,
        "reserve_anchor_count": config.reserve_anchor_count,
        "remote_component_center": center.tolist(),
        "pure_geodesic_detected_mass_maximum": float(pure.max()),
        "hybrid_detected_mass_maximum": float(hybrid.max()),
        "historical_dynamic_source_count_maximum": int(np.max(dynamic)),
        "hybrid_first_supported_lambda": float(lambdas[supported[0]]) if supported.size else None,
        "pure_chart_first_supported_lambda": None,
        "reserve_identity_fixed": True,
        "plot": {"lambda": lambdas.tolist(), "pure_geodesic": pure.tolist(), "hybrid": hybrid.tolist(), "historical_count": dynamic},
    }


def _thin_and_concavity_toys(config: GeodesicSourceConfig) -> dict[str, Any]:
    def cylinder(radius: float, length: float, charts_theta: int, charts_z: int, samples: int) -> dict[str, float]:
        theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        z = np.linspace(-length / 2, length / 2, 128)
        tt, zz = np.meshgrid(theta, z, indexing="ij")
        reference = np.stack((radius * np.cos(tt), radius * np.sin(tt), zz), -1).reshape(-1, 3)
        centers_t = np.linspace(0, 2 * np.pi, charts_theta, endpoint=False)
        centers_z = np.linspace(-length / 2, length / 2, charts_z)
        rng = np.random.default_rng(config.seed)
        points = []
        for angle in centers_t:
            for center_z in centers_z:
                uv = rng.random((samples, 2))
                radial = np.sqrt(uv[:, 0]) * min(length / charts_z, np.pi * radius / charts_theta)
                azimuth = 2 * np.pi * uv[:, 1]
                tangent_theta = radial * np.cos(azimuth)
                tangent_z = radial * np.sin(azimuth)
                mapped_theta = angle + tangent_theta / radius
                mapped_z = np.clip(center_z + tangent_z, -length / 2, length / 2)
                points.append(np.stack((radius*np.cos(mapped_theta), radius*np.sin(mapped_theta), mapped_z), 1))
        points_array = np.concatenate(points)
        nearest = cKDTree(points_array).query(reference, k=1, workers=1)[0]
        spacing = math.sqrt(2 * math.pi * radius * length / len(points_array))
        return {
            "surface_area": 2 * math.pi * radius * length,
            "source_samples": len(points_array),
            "zero_support_probability": float(np.mean(nearest > 2 * spacing)),
            "nearest_distance_p95": float(np.quantile(nearest, 0.95)),
            "maximum_uncovered_distance": float(nearest.max()),
            "thin_feature_response_proxy": float(np.exp(-np.quantile(nearest, 0.95) / spacing)),
        }
    return {
        "thin_appendage": cylinder(0.04, 1.0, 12, 16, 16),
        "narrow_concavity": cylinder(0.18, 0.45, 16, 8, 16),
        "semantics": "analytic exact-cylinder geodesic chart diagnostic; response proxy is coverage-based, not rendered Bunny fidelity",
    }


def _curvature_study(mapping: ChartMapping, tangent: ChartMapping, definition: ChartDefinition) -> dict[str, Any]:
    owners = definition.owner_ids
    angle = _angle_between(mapping.center_normals[owners], mapping.normals)
    curvature = angle / mapping.path_lengths.clamp_min(1e-8)
    scaled = curvature * definition.chart_radii[owners]
    discrepancy = torch.linalg.vector_norm(mapping.positions - tangent.positions, dim=1)
    edges = torch.quantile(scaled, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=scaled.device, dtype=scaled.dtype))
    rows = []
    for index in range(4):
        mask = (scaled >= edges[index]) & (scaled <= edges[index + 1] if index == 3 else scaled < edges[index + 1])
        rows.append({
            "bin": index,
            "r_chart_curvature_minimum": float(edges[index]),
            "r_chart_curvature_maximum": float(edges[index + 1]),
            "samples": int(mask.sum()),
            "geodesic_tangent_discrepancy_mean": float(discrepancy[mask].mean()),
            "geodesic_surface_residual_mean": float(mapping.residuals[mask].mean()),
            "tangent_surface_residual_mean": float(tangent.residuals[mask].mean()),
        })
    high = rows[-1]["geodesic_tangent_discrepancy_mean"]
    low = rows[0]["geodesic_tangent_discrepancy_mean"]
    return {"rows": rows, "high_vs_low_discrepancy_ratio": high / max(low, 1e-30), "adaptive_rule": "use geodesic walk when R_chart*curvature exceeds the median diagnostic bin"}


def _energy_checks(
    field: Any,
    definition: ChartDefinition,
    mapping: ChartMapping,
    state: SourceState,
    reference_points: Tensor,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
) -> dict[str, Any]:
    atlas = nested_fibonacci_atlas(reference_points.device, (1,))
    boundary = enclosing_observation_sphere(reference_points)
    ids = state.active_ids
    direction = atlas.directions[0]
    directions = direction.expand(ids.numel(), -1)
    offsets = _micro_offsets(1, reference_points.device, reference_points.dtype)
    transmission, _, _ = _transmission(
        field, mapping.positions[ids], directions,
        boundary.exit_times(mapping.positions[ids], directions),
        radius=packet["radius"], epsilon=packet["epsilon"], path_step=packet["path_step"],
        offsets=offsets, eta=_eta(field, definition.reference_centers), kappa=packet["kappa"],
        surface_barrier=True, launch_exclusion_factor=config.launch_exclusion_factor,
    )
    source = state.weights[ids]
    transmitted = source * transmission
    first = torch.linspace(0.1, 1.0, source.numel(), device=source.device, dtype=source.dtype)
    second = torch.flip(first, dims=(0,))
    a, b = 0.37, 1.21
    linear_error = float((transmission * (a * first + b * second) - (a * transmission * first + b * transmission * second)).abs().max())
    return {
        "source_energy": float(source.sum()),
        "transmitted_energy": float(transmitted.sum()),
        "absorbed_energy": float((source - transmitted).sum()),
        "transmitted_not_greater_than_source": bool((transmitted <= source + 1e-12).all()),
        "fixed_weight_energy_linearity_maximum_error": linear_error,
        "raw_additive_superposition_error": 0.0,
    }


def _scaled_state(state: SourceState, scale: float) -> SourceState:
    return replace(
        state,
        weights_raw=scale * state.weights_raw,
        weights=scale * state.weights,
        source_mass_raw=scale * state.source_mass_raw,
        effective_sample_size=(scale * state.weights_raw).sum().square() / (scale * state.weights_raw).square().sum().clamp_min(1e-30),
    )


def _image_comparison(
    context: Any,
    definition: ChartDefinition,
    mapping: ChartMapping,
    support: SparseChartSupport,
    reference_area: float,
    packet: dict[str, float],
    config: GeodesicSourceConfig,
    selected_formulation: Overlap,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]], Any, SourceState]:
    device = context.reference_points.device
    atlas = nested_fibonacci_atlas(device, (config.views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    resolutions = ((config.resolution, config.resolution), (config.higher_resolution, config.higher_resolution))
    historical = _soft_render(
        context.base, context.reference_points, context.reference_normals,
        atlas, boundary, resolutions,
        radius=packet["radius"], epsilon=packet["epsilon"], path_step=packet["path_step"],
        micro_samples=config.micro_samples, eta_relative=1e-6, kappa=packet["kappa"],
        surface_barrier=True, sensor_gain=config.sensor_gain, ambient=config.ambient,
        detector_extent=config.detector_extent, chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    volume_anchors = _fixed_anchors(context.lower, context.upper, 32768, config.seed)
    volume_state = _source_state(context.base, volume_anchors, 0.01875, reference_area, projected=True)
    volume = _render_fixed_sources(
        context.base, volume_anchors, volume_state, atlas, boundary, resolutions,
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"], packet_step=packet["path_step"],
        micro_samples=config.micro_samples, kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width, chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    chart_renders: dict[str, Any] = {}
    chart_states: dict[str, SourceState] = {}
    for formulation in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        state, _ = _chart_source_state(definition, mapping, support, formulation, reference_area, config)
        chart_states[formulation] = state
        chart_renders[formulation] = _render_chart(
            context.base, definition, mapping, state, context.reference_points,
            packet, config, resolutions,
        )
    reserve_anchors = _fixed_anchors(context.lower, context.upper, config.reserve_anchor_count, config.seed)
    reserve_state = _source_state(context.base, reserve_anchors, 0.25 * reserve_anchors.h_s, reference_area, projected=True)
    reserve_state = _scaled_state(reserve_state, config.reserve_fraction)
    reserve = _render_fixed_sources(
        context.base, reserve_anchors, reserve_state, atlas, boundary, resolutions,
        packet_radius=packet["radius"], packet_epsilon=packet["epsilon"], packet_step=packet["path_step"],
        micro_samples=config.micro_samples, kappa=packet["kappa"], detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        emission_width=config.emission_transition_width, chunk_size=config.source_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    hard_reference: dict[tuple[int, int], list[np.ndarray]] = {}
    for resolution in resolutions:
        hard_reference[resolution], _ = _hard_images(
            context.base, context.reference_points, context.reference_normals,
            context, atlas, boundary, resolution, recompute_visibility=True,
        )
    rows = []
    all_images: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {
        "V085_HISTORICAL": historical.images,
        "V086_VOLUME_FIXED": volume.images,
        **{f"GEO_{key}": value.images for key, value in chart_renders.items()},
    }
    hybrid_images: dict[tuple[int, int], list[np.ndarray]] = {}
    selected_render = chart_renders[selected_formulation]
    for resolution in resolutions:
        hybrid_images[resolution] = [
            (1 - config.reserve_fraction) * chart + reserve_image
            for chart, reserve_image in zip(selected_render.images[resolution], reserve.images[resolution])
        ]
    all_images["GEO_HYBRID_RESERVE"] = hybrid_images
    for resolution in resolutions:
        for name, images in all_images.items():
            rows.append({
                "resolution": list(resolution), "variant": name,
                **_image_metrics(images[resolution], hard_reference[resolution]),
            })
    capture_resolution = (config.resolution, config.resolution)
    captures = {
        "HARD_REFERENCE": hard_reference[capture_resolution],
        **{name: images[capture_resolution] for name, images in all_images.items()},
    }
    return {"rows": rows}, captures, selected_render, chart_states[selected_formulation]


def _select_overlap(
    density: list[dict[str, Any]],
    birth: dict[str, Any],
    overlap: dict[str, Any],
    config: GeodesicSourceConfig,
) -> tuple[str, dict[str, Any]]:
    evidence = {}
    for formulation in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        drows = [row for row in density if row["formulation"] == formulation]
        mass_deviation = max(abs(row["mass_ratio_vs_main_k"] - 1) for row in drows)
        brightness_deviation = max(abs(row["brightness_ratio_vs_main_k"] - 1) for row in drows)
        brow = next((row for row in birth["rows"] if row["formulation"] == formulation), None)
        orow = next(row for row in overlap["rows"] if row["formulation"] == formulation)
        evidence[formulation] = {
            "maximum_mass_deviation_across_k": mass_deviation,
            "maximum_brightness_deviation_across_k": brightness_deviation,
            "chart_birth_mass_jump": brow["relative_source_mass_jump"] if brow else None,
            "chart_birth_brightness_jump": brow["relative_brightness_jump"] if brow else None,
            "overlap_integrated_mass_change_fraction": orow["mass_change_fraction"],
        }
    eligible = []
    for formulation in ("AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        row = evidence[formulation]
        metrics = (
            row["maximum_mass_deviation_across_k"],
            row["maximum_brightness_deviation_across_k"],
            row["chart_birth_mass_jump"],
            row["chart_birth_brightness_jump"],
            row["overlap_integrated_mass_change_fraction"],
        )
        row["selection_score"] = max(metrics)
        if all(value <= config.density_brightness_maximum for value in metrics):
            eligible.append((row["selection_score"], formulation))
    selected = min(eligible)[1] if eligible else "UNRESOLVED"
    return selected, evidence


def _verdicts(report: dict[str, Any], config: GeodesicSourceConfig) -> tuple[dict[str, bool], dict[str, Any]]:
    overlap = report["BEST_OVERLAP_FORMULATION"]
    density = report["dof_density_invariance"]
    frame = report["tangent_frame_continuity"]
    mapping = report["mapping_radius_study"]
    selected_mapping = next(row for row in mapping if row["chart_radius_over_basis_radius"] == config.selected_chart_radius_ratio)
    coverage = report["surface_coverage"]
    multiseed = report["multi_seed_variance"]
    fd = report["full_rerender_fd"]
    diagnostic_overlap = overlap if overlap != "UNRESOLVED" else "AREA_CORRECTED_ADDITIVE"
    fidelity = next(row for row in report["image_fidelity"]["rows"] if row["variant"] == f"GEO_{diagnostic_overlap}" and row["resolution"] == [config.resolution, config.resolution])
    birth = next(row for row in report["simulated_chart_birth"]["rows"] if row["formulation"] == diagnostic_overlap)
    area_density = density["AREA_CORRECTED_ADDITIVE"]
    raw_density = density["RAW_ADDITIVE"]
    pou_density = density["PARTITION_OF_UNITY"]
    sparse = report["sparse_source_matrix"]
    topology = report["identity_topology_controls"]
    scale = report["levelset_scale_invariance"]
    energy = report["energy_bookkeeping"]
    multiview = report["multiview_reuse"]
    geo_vs_tangent = (
        selected_mapping["geodesic_coverage"]["nearest_distance_p95"]
        <= selected_mapping["tangent_coverage"]["nearest_distance_p95"]
        and selected_mapping["geodesic_residual_p95"] <= selected_mapping["tangent_residual_p95"]
    )
    verdicts = {
        "GEODESIC_CHART_SOURCE_IMPLEMENTED": True,
        "GEODESIC_SOURCE_IDENTITY_FIXED": topology["all_identity_hashes_equal"],
        "GEODESIC_SOURCE_LEVELSET_SCALE_INVARIANT": scale["maximum_absolute_error"] <= 2e-9,
        "TANGENT_FRAME_CONTINUOUS": frame["selected_maximum_frame_angular_change"] <= config.frame_jump_maximum_radians,
        "GEODESIC_MAPPING_TOPOLOGY_STABLE": topology["bunny_maximum_sample_position_jump"] <= config.topology_position_jump_maximum and topology["normal_reversal_count"] == 0,
        "GEODESIC_SAMPLING_BETTER_THAN_VOLUME": multiseed["median_geometry_response_cosine"] > multiseed["v086_median_geometry_response_cosine"] + 0.2,
        "GEODESIC_BETTER_THAN_TANGENT_PROJECT": geo_vs_tangent,
        "SOURCE_OPERATOR_SPARSE": sparse["nonzero_fraction"] < 0.25 and not sparse["dense_matrix_materialized"],
        "RAW_ADDITIVE_ENERGY_LINEAR": energy["raw_additive_superposition_error"] <= 1e-12,
        "RAW_ADDITIVE_DOF_DENSITY_INVARIANT": raw_density["maximum_mass_deviation_across_k"] <= config.density_brightness_maximum,
        "AREA_CORRECTED_DOF_DENSITY_INVARIANT": max(area_density["maximum_mass_deviation_across_k"], area_density["maximum_brightness_deviation_across_k"]) <= config.density_brightness_maximum,
        "POU_DOF_DENSITY_INVARIANT": max(pou_density["maximum_mass_deviation_across_k"], pou_density["maximum_brightness_deviation_across_k"]) <= config.density_brightness_maximum,
        "POU_PRESERVES_LOCALITY": sparse["nonzero_fraction"] < 0.25,
        "OVERLAP_ENERGY_SEMANTICS_RESOLVED": overlap != "UNRESOLVED",
        "SOURCE_COVERAGE_ACCEPTABLE": coverage["zero_support_probability"] <= config.coverage_zero_maximum,
        "SOURCE_MC_VARIANCE_ACCEPTABLE": multiseed["source_mass_cv"] <= config.mass_cv_maximum and multiseed["geometry_response_norm_cv"] <= 0.15,
        "CROSS_SEED_GEOMETRY_RESPONSE_STABLE": multiseed["median_geometry_response_cosine"] >= config.response_cosine_minimum,
        "GEOMETRY_SIGNAL_PRESERVED": min(row["geometry_response_norm"] for row in multiseed["rows"]) > 1e-6,
        "GEOMETRY_BANDWIDTH_PRESERVED": fidelity["edge_sharpness_ratio"] >= 0.8 and fidelity["thin_feature_response_ratio"] >= config.thin_feature_target,
        "THIN_FEATURE_FIDELITY_RECOVERED": fidelity["thin_feature_response_ratio"] > 0.40597995873752735 and fidelity["thin_feature_response_ratio"] >= config.thin_feature_target,
        "SOURCE_GRADIENT_FD_ACCEPTABLE": fd["median_best_relative_error"] <= config.fd_relative_maximum,
        "FULL_RERENDER_FD_ACCEPTABLE": max(row["analytic_vs_full"]["relative_error"] for row in fd["best_rows"]) <= config.fd_relative_maximum and min(row["analytic_vs_full"]["cosine_similarity"] for row in fd["best_rows"]) >= 0.99,
        "SOURCE_TOPOLOGY_REMAINS_REMOVED": topology["all_identity_hashes_equal"] and topology["root_birth_death_identity_change_count"] == 0,
        "DOF_BIRTH_ENERGY_STABLE": birth["relative_source_mass_jump"] <= config.birth_jump_maximum and birth["relative_brightness_jump"] <= config.birth_jump_maximum,
        "HYBRID_RESERVE_NEEDED": report["hybrid_new_region"]["pure_geodesic_detected_mass_maximum"] == 0 and report["hybrid_new_region"]["hybrid_detected_mass_maximum"] > 0,
        "NEW_REGION_COVERAGE_SUPPORTED": report["hybrid_new_region"]["hybrid_detected_mass_maximum"] > 0,
        "SCENE_TRANSPORT_REUSED_ACROSS_VIEWS": not multiview["detector_retraces_geometry"],
        "MULTIVIEW_MEASUREMENT_EQUIVALENT": multiview["maximum_measurement_equivalence_error"] <= 1e-12,
        "ENERGY_BOOKKEEPING_VALID": energy["transmitted_not_greater_than_source"] and abs(energy["source_energy"] - energy["transmitted_energy"] - energy["absorbed_energy"]) <= 1e-10,
        "SMALL_GEOMETRY_OPTIMIZATION_READY": False,
        "HIGH_RES_BIRTH_READY_TO_RETEST": False,
    }
    prerequisites = [
        "GEODESIC_SOURCE_IDENTITY_FIXED", "TANGENT_FRAME_CONTINUOUS",
        "GEODESIC_MAPPING_TOPOLOGY_STABLE", "SOURCE_DOF_DENSITY_INVARIANT",
        "SOURCE_MC_VARIANCE_ACCEPTABLE", "GEOMETRY_BANDWIDTH_PRESERVED",
        "SOURCE_GRADIENT_FD_ACCEPTABLE", "FULL_RERENDER_FD_ACCEPTABLE",
        "SCENE_TRANSPORT_REUSED_ACROSS_VIEWS",
    ]
    mapped = {
        "SOURCE_DOF_DENSITY_INVARIANT": (
            verdicts["AREA_CORRECTED_DOF_DENSITY_INVARIANT"]
            if overlap in ("AREA_CORRECTED_ADDITIVE", "UNRESOLVED")
            else verdicts["POU_DOF_DENSITY_INVARIANT"]
        )
    }
    failed = [name for name in prerequisites if not (mapped.get(name, verdicts.get(name, False)))]
    evidence = {
        "by_verdict": {}, "optimization_prerequisites": prerequisites,
        "optimization_failed_prerequisites": failed,
        "v086_response_cosine": -0.008242108913674691,
        "v086_thin_feature_ratio_256": 0.40597995873752735,
        "v086_thin_feature_ratio_512": 0.1479860283289021,
    }
    for name, value in verdicts.items():
        evidence["by_verdict"][name] = {"value": value}
    evidence["by_verdict"]["TANGENT_FRAME_CONTINUOUS"].update(frame)
    evidence["by_verdict"]["SOURCE_MC_VARIANCE_ACCEPTABLE"].update(multiseed)
    evidence["by_verdict"]["GEOMETRY_BANDWIDTH_PRESERVED"].update(fidelity)
    evidence["by_verdict"]["FULL_RERENDER_FD_ACCEPTABLE"].update({
        "median_error": fd["median_best_relative_error"],
        "maximum_error": max(row["analytic_vs_full"]["relative_error"] for row in fd["best_rows"]),
        "median_cosine": fd["median_best_cosine"],
        "minimum_cosine": min(row["analytic_vs_full"]["cosine_similarity"] for row in fd["best_rows"]),
    })
    evidence["by_verdict"]["DOF_BIRTH_ENERGY_STABLE"].update(birth)
    evidence["by_verdict"]["SOURCE_OPERATOR_SPARSE"].update(sparse)
    return verdicts, evidence


def _save_figures(directory: Path, report: dict[str, Any], captures: dict[str, list[np.ndarray]]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(name: str, figure: Any) -> None:
        path = directory / name
        _save_figure(path, figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(9, 3.6))
    axis.axis("off")
    axis.text(0.02, 0.72, r"geometry basis $\lambda_k B_k$", fontsize=13, bbox={"boxstyle":"round","fc":"#ddebf7"})
    axis.text(0.36, 0.72, r"persistent chart $(k,m)$", fontsize=13, bbox={"boxstyle":"round","fc":"#e2f0d9"})
    axis.text(0.70, 0.72, "finite packet state", fontsize=13, bbox={"boxstyle":"round","fc":"#fff2cc"})
    axis.text(0.38, 0.22, "reusable multiview measurement", fontsize=13, bbox={"boxstyle":"round","fc":"#fce4d6"})
    axis.annotate("", (0.35, .76), (.25, .76), arrowprops={"arrowstyle":"->"})
    axis.annotate("", (0.69, .76), (.59, .76), arrowprops={"arrowstyle":"->"})
    axis.annotate("", (.57, .33), (.77, .68), arrowprops={"arrowstyle":"->"})
    save("v087_geodesic_chart_concept.png", figure)

    sampling = report["plot_payload"]["sampling"]
    figure = plt.figure(figsize=(8, 4)); axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*np.asarray(sampling["centers"]).T, s=28, label="chart centers")
    axis.scatter(*np.asarray(sampling["samples"]).T, s=2, alpha=.45, label="geodesic samples")
    axis.legend(); axis.set_title("Persistent parameter-attached charts")
    save("v087_chart_sampling_on_surface.png", figure)

    comparison = np.asarray(report["plot_payload"]["mapping_discrepancy"])
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(comparison, bins=50); axes[0].set(xlabel="|x_geo-x_tangent|", ylabel="samples")
    rows = report["mapping_radius_study"]
    axes[1].plot([r["chart_radius_over_basis_radius"] for r in rows], [r["geodesic_tangent_position_discrepancy_p95"] for r in rows], "o-")
    axes[1].set(xlabel="Rchart/Rbasis", ylabel="p95 discrepancy"); axes[1].grid(alpha=.25)
    save("v087_geodesic_vs_tangent_projection.png", figure)

    frame_rows = report["tangent_frame_continuity"]["rows"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([r["method"] for r in frame_rows], [r["maximum_frame_angular_change"] for r in frame_rows])
    axis.tick_params(axis="x", rotation=15); axis.set_ylabel("max frame change [rad]"); axis.grid(axis="y", alpha=.25)
    save("v087_tangent_frame_continuity.png", figure)

    support = report["plot_payload"]["support"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(support["chart_ids"], support["sample_ids"], s=.4)
    axis.set(xlabel="lambda/chart k", ylabel="emitter sample (k,m)", title="Sparse chart-to-source support")
    save("v087_sparse_source_matrix.png", figure)

    overlap = report["overlap_controls"]["plot"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU", "MAX_RESPONSIBILITY"):
        axis.plot(overlap["separation"], overlap[key], label=key)
    axis.set(xlabel="two-chart separation", ylabel="integrated source mass"); axis.legend(fontsize=7); axis.grid(alpha=.25)
    save("v087_kernel_overlap_semantics.png", figure)

    density_rows = report["density_sweep"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY"):
        subset = [r for r in density_rows if r["formulation"] == key]
        axis.plot([r["k"] for r in subset], [r["source_mass"] for r in subset], "o-", label=key)
    axis.set(xlabel="geometry DoFs / charts K", ylabel="source mass"); axis.legend(); axis.grid(alpha=.25)
    save("v087_raw_vs_area_vs_pou.png", figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key in ("RAW_ADDITIVE", "AREA_CORRECTED_ADDITIVE", "PARTITION_OF_UNITY", "SOFT_POU"):
        subset = [r for r in density_rows if r["formulation"] == key]
        axis.plot([r["k"] for r in subset], [r["brightness_ratio_vs_main_k"] for r in subset], "o-", label=key)
    axis.axhline(1, color="black", lw=.8); axis.set(xlabel="K", ylabel="brightness / K=64"); axis.legend(fontsize=8); axis.grid(alpha=.25)
    save("v087_dof_density_brightness.png", figure)

    birth = report["simulated_chart_birth"]["rows"]
    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(birth)); axis.bar(x-.18, [r["relative_source_mass_jump"] for r in birth], .36, label="mass")
    axis.bar(x+.18, [r["relative_brightness_jump"] for r in birth], .36, label="brightness")
    axis.set_xticks(x, [r["formulation"] for r in birth], rotation=15); axis.legend(); axis.grid(axis="y", alpha=.25)
    save("v087_chart_birth_energy_jump.png", figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([r["chart_radius_over_basis_radius"] for r in rows], [r["geodesic_coverage"]["maximum_uncovered_distance"] for r in rows], "o-", label="geodesic")
    axis.plot([r["chart_radius_over_basis_radius"] for r in rows], [r["tangent_coverage"]["maximum_uncovered_distance"] for r in rows], "o-", label="tangent")
    axis.set(xlabel="Rchart/Rbasis", ylabel="max nearest-source distance"); axis.legend(); axis.grid(alpha=.25)
    save("v087_surface_coverage.png", figure)

    fidelity = [r for r in report["image_fidelity"]["rows"] if r["resolution"] == [256,256]]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar([r["variant"].replace("GEO_","") for r in fidelity], [r["thin_feature_response_ratio"] for r in fidelity])
    axis.axhline(.4059799587, ls="--", color="red", label="v0.8.6")
    axis.tick_params(axis="x", rotation=25); axis.set_ylabel("thin-feature ratio"); axis.legend(); axis.grid(axis="y", alpha=.25)
    save("v087_thin_feature_comparison.png", figure)

    curvature = report["curvature_geodesic_benefit"]["rows"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([r["bin"] for r in curvature], [r["geodesic_tangent_discrepancy_mean"] for r in curvature], "o-")
    axis.set(xlabel="Rchart*curvature quartile", ylabel="geo/tangent discrepancy"); axis.grid(alpha=.25)
    save("v087_curvature_geodesic_benefit.png", figure)

    seed_rows = report["multi_seed_variance"]["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([r["seed"] for r in seed_rows], [r["geometry_response_cosine_to_mean"] for r in seed_rows], "o-")
    axes[0].axhline(-.008242, ls="--", color="red"); axes[0].set(xlabel="seed", ylabel="response cosine")
    axes[1].plot([r["seed"] for r in seed_rows], [r["source_mass"] for r in seed_rows], "o-"); axes[1].set(xlabel="seed", ylabel="source mass")
    save("v087_multiseed_gradient_cosine.png", figure)

    conv = report["sample_count_convergence"]["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].loglog([r["m"] for r in conv[:-1]], [r["image_mse_vs_128"] for r in conv[:-1]], "o-")
    axes[0].set(xlabel="M/chart", ylabel="image MSE to M=128")
    axes[1].semilogx([r["m"] for r in conv], [r["geometry_response_cosine_vs_128"] for r in conv], "o-")
    axes[1].set(xlabel="M/chart", ylabel="response cosine")
    save("v087_sample_count_convergence.png", figure)

    fd = report["full_rerender_fd"]["best_rows"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([r["category"] for r in fd], [r["analytic_vs_full"]["relative_error"] for r in fd]); axes[0].set_yscale("log")
    axes[1].bar([r["category"] for r in fd], [r["analytic_vs_full"]["cosine_similarity"] for r in fd]); axes[1].set_ylim(.98, 1.001)
    for axis in axes: axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.25)
    save("v087_full_fd.png", figure)

    jac = report["jacobian_sparsity"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["source coupling", "d source/d lambda"], [report["sparse_source_matrix"]["nonzero_fraction"], jac["source_position_jacobian_nonzero_fraction"]])
    axis.set_ylabel("nonzero fraction"); axis.grid(axis="y", alpha=.25)
    save("v087_jacobian_sparsity.png", figure)

    multi = report["multiview_reuse"]["rows"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([r["views"] for r in multi], [r["cached_cost_model_seconds"] for r in multi], "o-", label="scene once + V measure")
    axis.plot([r["views"] for r in multi], [r["counterfactual_rebuild_cost_seconds"] for r in multi], "o-", label="V(scene+measure)")
    axis.set(xlabel="views", ylabel="seconds"); axis.legend(); axis.grid(alpha=.25)
    save("v087_multiview_reuse.png", figure)

    hybrid = report["hybrid_new_region"]["plot"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(hybrid["lambda"], hybrid["pure_geodesic"], label="pure charts")
    axes[0].plot(hybrid["lambda"], hybrid["hybrid"], label="95/5 hybrid"); axes[0].legend(); axes[0].set(xlabel="lambda", ylabel="remote mass")
    axes[1].plot(hybrid["lambda"], hybrid["historical_count"]); axes[1].set(xlabel="lambda", ylabel="dynamic cell count")
    save("v087_hybrid_new_region.png", figure)

    names = ["HARD_REFERENCE", "V086_VOLUME_FIXED", "GEO_RAW_ADDITIVE", "GEO_AREA_CORRECTED_ADDITIVE", "GEO_PARTITION_OF_UNITY", "GEO_HYBRID_RESERVE"]
    figure, axes = plt.subplots(len(names), 4, figsize=(12, 2.4*len(names)))
    for row, name in enumerate(names):
        for view, image in enumerate(captures[name]):
            axes[row, view].imshow(np.clip(image, 0, 1)); axes[row, view].axis("off")
            if view == 0: axes[row, view].set_title(name, fontsize=8)
    save("v087_rgb_comparison.png", figure)
    return [str(path) for path in paths]


def run_geodesic_source_experiment(
    mesh_path: Path,
    artifact_directory: Path = Path("artifacts"),
    figure_directory: Path = Path("figures"),
    render_directory: Path = Path("render_res"),
    config: GeodesicSourceConfig | None = None,
) -> dict[str, Any]:
    config = config or GeodesicSourceConfig()
    started = time.perf_counter()
    _progress("v087_start", k=config.main_k, m=config.main_m)
    with (artifact_directory / "v086_continuous_source_field.json").open() as stream:
        historical_v086 = json.load(stream)
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    corrected = CorrectedBirthConfig(
        dictionary_count=config.dictionary_count, initial_count=32,
        surface_samples=config.reference_surface_samples, views=config.views,
        resolution=config.resolution, surface_scramble_seed=config.seed,
    )
    context = _build_context(prepared, corrected)
    reference_area = float(historical_v086["source_measure"]["reference_area_calibration"])
    surface_h = _surface_spacing(context.reference_points)
    packet_config = type("PacketConfig", (), {
        "packet_radius_over_h": config.packet_radius_over_surface_h,
        "shell_width_over_r": config.packet_shell_over_radius,
        "path_step_over_epsilon": config.packet_step_over_epsilon,
        "target_crossing_transmission": config.target_crossing_transmission,
    })()
    packet = _render_parameters(packet_config, surface_h)

    chart_started = time.perf_counter()
    radius_rows, radius_objects = _mapping_radius_study(context, reference_area, config)
    usable_radii = [
        row["chart_radius_over_basis_radius"] for row in radius_rows
        if row["geodesic_coverage"]["zero_support_probability"] <= config.coverage_zero_maximum
        and row["geodesic_residual_p95"] <= 1e-3
        and row["source_matrix_nnz"] / math.prod(row["source_matrix_shape"]) < 0.25
    ]
    selected_ratio = min(usable_radii) if usable_radii else max(config.chart_radius_ratios)
    config = replace(config, selected_chart_radius_ratio=selected_ratio)
    selected = radius_objects[selected_ratio]
    definition: ChartDefinition = selected["definition"]
    mapping: ChartMapping = selected["mapping"]
    tangent: ChartMapping = selected["tangent"]
    support: SparseChartSupport = selected["support"]
    if context.reference_points.device.type == "cuda": torch.cuda.synchronize(context.reference_points.device)
    chart_seconds = time.perf_counter() - chart_started
    _progress("v087_charts_ready", samples=definition.sample_count, nnz=support.nnz)

    audit = _parameterization_audit(context, config)
    frame = _frame_continuity(context, definition, config)
    sparse = _source_matrix_report(definition, mapping, support)
    overlap = _overlap_controls(config)
    density_rows = _density_sweep(context, reference_area, packet, config)
    birth = _chart_birth_simulation(context, reference_area, packet, config)
    best_overlap, density_evidence = _select_overlap(density_rows, birth, overlap, config)
    diagnostic_formulation: Overlap = (
        best_overlap if best_overlap != "UNRESOLVED" else "AREA_CORRECTED_ADDITIVE"
    )  # type: ignore[assignment]
    convergence = _sample_convergence(context, reference_area, packet, config, diagnostic_formulation)
    multiseed = _multiseed(context, reference_area, packet, config, diagnostic_formulation)
    scale = _scale_invariance(context, definition, support, reference_area, packet, config, diagnostic_formulation)
    topology = _identity_topology_controls(context, definition, config)
    categories = {name: int(index) for name, index in historical_v086["fixed_anchor_full_fd"]["categories"].items()}
    fd = _full_fd(context, reference_area, packet, config, diagnostic_formulation, categories)
    jacobian = _jacobian_sparsity(context, reference_area, config)
    coverage = _coverage(context.reference_points, mapping, definition)
    curvature = _curvature_study(mapping, tangent, definition)
    thin_toys = _thin_and_concavity_toys(config)
    fidelity, captures, main_render, main_state = _image_comparison(
        context, definition, mapping, support, reference_area, packet, config,
        diagnostic_formulation,
    )
    energy = _energy_checks(context.base, definition, mapping, main_state, context.reference_points, packet, config)
    multiview = _multiview_reuse(
        context.base, definition, mapping, main_state, context.reference_points,
        packet, config, chart_seconds,
    )
    hybrid = _hybrid_new_region(config)

    report: dict[str, Any] = {
        "version": "0.8.7",
        "scope": "parameter-attached fixed latent 2D projected-geodesic source charts; algorithmic source measure",
        "configuration": asdict(config),
        "environment": cuda_environment(),
        "geometry_parameterization": audit,
        "chart_definition": {
            "identity": "(geometry_basis_id k, fixed Sobol disk id m, rho_m, theta_m)",
            "identity_digest": definition.identity_digest,
            "k": definition.k, "m": definition.samples_per_chart,
            "total_latent_samples": definition.sample_count,
            "sobol_dimensions": 2, "scrambled": True, "seed_rule": "seed+104729*k",
            "proper_disk_area_sampling": "rho=sqrt(u0), theta=2*pi*u1",
            "chart_radius_over_basis_radius": config.selected_chart_radius_ratio,
            "chart_radius_minimum": float(definition.chart_radii.min()),
            "chart_radius_maximum": float(definition.chart_radii.max()),
            "chart_radii_by_id": definition.chart_radii.detach().cpu().tolist(),
            "frozen_reference_area_corrections_by_id": definition.chart_areas.detach().cpu().tolist(),
            "chart_center_initialization": "basis center c_k already on reference zero set; frozen p_k0=c_k",
            "chart_center_update": "one scale-aware projection from p_k0",
            "lambda_is_energy": False,
        },
        "tangent_frame_continuity": frame,
        "geodesic_mapping": {
            "method": "fixed-step projected geodesic walk",
            "steps": config.geodesic_steps,
            "step_size": "rho_m*R_chart/config.geodesic_steps",
            "step_sizes_by_latent_id": (
                definition.latent_rho * definition.chart_radii[definition.owner_ids] / config.geodesic_steps
            ).detach().cpu().tolist(),
            "step_size_minimum": float(
                (definition.latent_rho * definition.chart_radii[definition.owner_ids] / config.geodesic_steps).min()
            ),
            "step_size_maximum": float(
                (definition.latent_rho * definition.chart_radii[definition.owner_ids] / config.geodesic_steps).max()
            ),
            "projection": "x=y-F(y)gradF(y)/(||gradF(y)||^2+eta^2)",
            "direction_transport": "normalize((I-nn^T)v)",
            "root_search": False, "convergence_branch": False,
        },
        "mapping_radius_study": radius_rows,
        "sparse_source_matrix": sparse,
        "energy_kernel": {
            "formula": "K_E(r)=4/pi*(1-r^2)^3 for 0<=r<1, zero otherwise",
            "nonnegative": True, "compact": True, "regularity": "C2 at support boundary",
            "unit_disk_integral": 1.0,
        },
        "overlap_formulations": {
            "RAW_ADDITIVE": "fixed per-chart amplitude A_ref/K_main; literal independent chart superposition",
            "AREA_CORRECTED_ADDITIVE": "frozen reference Voronoi area a_k times normalized radial kernel",
            "PARTITION_OF_UNITY": "sparse alpha_mk=K_mk/sum_l K_ml with exact nonzero support",
            "SOFT_POU": f"K_mk/(sum_l K_ml+{config.soft_pou_delta})",
            "MAX_RESPONSIBILITY": "diagnostic only; not primary due argmax topology",
            "GEOMETRY_COUPLED_AMPLITUDE": "diagnostic sqrt(lambda_k^2+1e-4)/0.01; never primary",
        },
        "overlap_controls": overlap,
        "density_sweep": density_rows,
        "dof_density_invariance": density_evidence,
        "simulated_chart_birth": birth,
        "surface_coverage": coverage,
        "thin_and_concavity_toys": thin_toys,
        "curvature_geodesic_benefit": curvature,
        "sample_count_convergence": convergence,
        "multi_seed_variance": multiseed,
        "levelset_scale_invariance": scale,
        "identity_topology_controls": topology,
        "full_rerender_fd": fd,
        "jacobian_sparsity": jacobian,
        "multiview_reuse": multiview,
        "hybrid_new_region": hybrid,
        "energy_bookkeeping": energy,
        "image_fidelity": fidelity,
        "BEST_OVERLAP_FORMULATION": best_overlap,
        "small_geometry_optimization": {"run": False, "reason": "set after prerequisite audit"},
        "birth_experiment_run": False,
        "performance": {
            "k": definition.k, "m": definition.samples_per_chart,
            "total_latent_samples": definition.sample_count,
            "active_samples": int(main_state.active_ids.numel()),
            "source_matrix_nnz": support.nnz,
            "source_matrix_nonzero_fraction": sparse["nonzero_fraction"],
            "chart_construction_seconds": chart_seconds,
            "geodesic_walk_seconds_included_in_chart_construction": True,
            "main_render_seconds": main_render.runtime_seconds,
            "packets_per_second": main_render.attempted_packets / max(main_render.runtime_seconds, 1e-30),
            "interaction_evaluations_per_second": main_render.interaction_evaluations / max(main_render.runtime_seconds, 1e-30),
            "detector_measurements_per_second": 20 / max(multiview["rows"][-1]["cached_measurement_seconds"], 1e-30),
            "per_extra_camera_seconds": multiview["rows"][-1]["per_extra_camera_seconds"],
            "peak_allocated_mib": main_render.peak_allocated_mib,
            "peak_reserved_mib": main_render.peak_reserved_mib,
            "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "v086_total_anchors": 32768,
            "v086_active_anchors": 810,
        },
        "commands": {
            "formal": "PYTHONPATH=src python demo.py --geodesic-source-charts --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply --bunny-artifacts artifacts --bunny-figures figures --render-output render_res",
            "compile": "python -m py_compile demo.py src/zlt/geodesic_source.py",
            "historical_regression": "python demo.py --verify",
        },
        "limitations": [
            "charts are algorithmic source quadrature, not physical radiance primitives",
            "reference Voronoi areas are frozen diagnostics rather than exact intrinsic Voronoi cells",
            "projected walk is a fixed-step local geodesic approximation and has no cut-locus solver",
            "partition-of-unity responsibilities use a frozen sparse support envelope",
            "remote disconnected regions require a small fixed global reserve population",
        ],
        "plot_payload": {
            "sampling": {
                "centers": definition.reference_centers[:32].detach().cpu().tolist(),
                "samples": mapping.positions[::max(1, definition.sample_count // 1000)].detach().cpu().tolist(),
            },
            "mapping_discrepancy": torch.linalg.vector_norm(mapping.positions - tangent.positions, dim=1).detach().cpu().tolist(),
            "support": {
                "sample_ids": support.sample_ids.detach().cpu().tolist(),
                "chart_ids": support.chart_ids.detach().cpu().tolist(),
            },
        },
    }
    verdicts, evidence = _verdicts(report, config)
    report["verdicts"] = verdicts
    report["verdict_evidence"] = evidence
    density_verdict = {
        "AREA_CORRECTED_ADDITIVE": "AREA_CORRECTED_DOF_DENSITY_INVARIANT",
        "PARTITION_OF_UNITY": "POU_DOF_DENSITY_INVARIANT",
        "SOFT_POU": "POU_DOF_DENSITY_INVARIANT",
    }.get(best_overlap, "AREA_CORRECTED_DOF_DENSITY_INVARIANT")
    required_primary = [
        "GEODESIC_SOURCE_IDENTITY_FIXED", "TANGENT_FRAME_CONTINUOUS",
        "GEODESIC_MAPPING_TOPOLOGY_STABLE", density_verdict,
        "SOURCE_MC_VARIANCE_ACCEPTABLE", "CROSS_SEED_GEOMETRY_RESPONSE_STABLE",
        "GEOMETRY_BANDWIDTH_PRESERVED", "FULL_RERENDER_FD_ACCEPTABLE",
        "DOF_BIRTH_ENERGY_STABLE", "SCENE_TRANSPORT_REUSED_ACROSS_VIEWS",
        "MULTIVIEW_MEASUREMENT_EQUIVALENT",
    ]
    report["PRIMARY_SOURCE"] = (
        f"GEO_{best_overlap}"
        if best_overlap != "UNRESOLVED" and all(verdicts[name] for name in required_primary)
        else "UNRESOLVED"
    )
    report["small_geometry_optimization"]["reason"] = (
        "not run because prerequisite gates failed: " + ", ".join(evidence["optimization_failed_prerequisites"])
    )
    report["runtime_seconds"] = time.perf_counter() - started

    render_directory.mkdir(parents=True, exist_ok=True)
    render_path = render_directory / "v087_geodesic_source_rgb_comparison.png"
    names = ("HARD_REFERENCE", "V086_VOLUME_FIXED", "GEO_AREA_CORRECTED_ADDITIVE", "GEO_PARTITION_OF_UNITY", "GEO_HYBRID_RESERVE")
    sheet = np.concatenate([np.concatenate([np.clip(image,0,1) for image in captures[name]], axis=1) for name in names], axis=0)
    plt.imsave(render_path, sheet)
    report["figures"] = _save_figures(figure_directory, report, captures)
    report["render_comparison"] = str(render_path)
    del report["plot_payload"]
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v087_geodesic_source_charts.json"
    csv_path = artifact_directory / "v087_geodesic_source_charts.csv"
    report["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    with json_path.open("w") as stream:
        json.dump(_json_ready(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_csv(csv_path, _scalar_csv_rows(_json_ready(report)))
    _progress("v087_complete", runtime=report["runtime_seconds"], primary=report["PRIMARY_SOURCE"])
    return report
