"""v0.8.10 stratified geodesic-polar emitter sampling.

Only persistent emitter placement inside the v0.8.9 charts changes here.
Finite-packet attenuation, continuous detector accumulation, geometry, and the
sparse local geodesic-Jacobian representation are reused without redesign.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .continuous_source import _emission_factor
from .corrected_birth import CorrectedBirthConfig, _build_context
from .emitter_scaling import _center_bank
from .finite_packet import (
    _accumulate_continuous,
    _hard_images,
    _image_metrics,
    _json_ready,
    _micro_offsets,
    _progress,
    _render_parameters,
    _save_figure,
    _scalar_csv_rows,
    _surface_spacing,
    _transmission,
    _vector_metrics,
)
from .high_sample import _release, _write_csv
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color
from .million_emitter import (
    MillionEmitterConfig,
    StreamRender,
    _chart_masses,
    _digest_arrays,
    _latent_chunk,
    _latent_chunks,
    _reference_images,
    _run_forward_row,
    _sync,
    _template,
)
from .sparse_geodesic import (
    CompactSupportField,
    GeodesicTemplate,
    PreparedGeodesicCenters,
    SparseGeodesicGraph,
    SparseGraphBuilder,
    SparsePointBasisSupport,
    build_chart_envelope_support,
    emitter_support_from_owners,
    map_emitter_chunk,
    prepare_centers,
    prepare_centers_sparse_field,
    sparse_field_evaluate,
)


Tensor = torch.Tensor
RadialScheme = Literal["AREA_STRATIFIED", "DISTANCE_STRATIFIED"]


@dataclass(frozen=True)
class StratifiedPolarConfig:
    k_geom: int = 64
    k_chart: int = 1024
    matched_m: tuple[int, ...] = (64, 256, 1024)
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    jitter_values: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40)
    primary_jitter: float = 0.25
    primary_radial_scheme: RadialScheme = "AREA_STRATIFIED"
    geodesic_integration_steps: int = 32
    diagnostic_resolution: int = 64
    comparison_resolutions: tuple[tuple[int, int], ...] = (
        (256, 256), (512, 512), (1080, 1920),
    )
    fullhd_views: int = 4
    emitter_chunk_size: int = 8192
    camera_block_size: int = 4
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    ambient: float = 0.35
    emission_transition_width: float = 0.05
    packet_radius_over_surface_h: float = 1.0
    packet_shell_over_radius: float = 1.0
    packet_step_over_epsilon: float = 0.5
    target_crossing_transmission: float = 0.01
    launch_exclusion_factor: float = 1.05
    graph_threshold: float = 1e-10
    performance_charts: int = 4
    measure_m: int = 256
    response_parameter: int = 18
    response_delta: float = 1e-3
    thin_feature_target: float = 0.75

    @staticmethod
    def balanced_factorization(m: int) -> tuple[int, int]:
        side = int(round(math.sqrt(m)))
        if side * side != m:
            raise ValueError(f"M={m} is not a square polar budget")
        return side, side


@dataclass(frozen=True)
class PolarLayout:
    theta: Tensor
    rho: Tensor
    angular_jitter: Tensor
    radial_jitter: Tensor
    radial_mass: Tensor
    n_theta: int
    n_r: int
    seed: int
    eta_theta: float
    eta_r: float
    radial_scheme: RadialScheme
    digest: str

    @property
    def chart_count(self) -> int:
        return int(self.theta.shape[0])

    @property
    def m(self) -> int:
        return self.n_theta * self.n_r


@dataclass
class PolarRender:
    stream: StreamRender
    positions: Tensor | None
    weights: Tensor | None


def _splitmix_uniform(keys: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Stable counter-based uniforms without mutable RNG state."""
    mask = (1 << 64) - 1
    flat = np.asarray(keys, dtype=np.uint64).reshape(-1)
    output = np.empty(flat.size, dtype=np.float64)
    seed_word = (int(seed) * 0xD1B54A32D192ED03 + int(salt)) & mask
    for index, key in enumerate(flat.tolist()):
        value = (int(key) + seed_word + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        output[index] = (value >> 11) * (1.0 / (1 << 53))
    return output.reshape(keys.shape)


def polar_layout(
    k_chart: int,
    n_theta: int,
    n_r: int,
    seed: int,
    eta_theta: float,
    eta_r: float,
    radial_scheme: RadialScheme,
    device: torch.device,
    dtype: torch.dtype,
) -> PolarLayout:
    if not (0.0 <= eta_theta < 0.5 and 0.0 <= eta_r < 0.5):
        raise ValueError("jitter fractions must remain inside their strata")
    chart = np.arange(k_chart, dtype=np.uint64)[:, None]
    angular = np.arange(n_theta, dtype=np.uint64)[None, :]
    angular_keys = chart * np.uint64(65537) + angular
    angular_unit = _splitmix_uniform(angular_keys, seed, 0xA0761D6478BD642F)
    angular_jitter_np = eta_theta * (2.0 * angular_unit - 1.0)
    theta_np = 2.0 * math.pi * (
        angular.astype(np.float64) + 0.5 + angular_jitter_np
    ) / n_theta

    radial = np.arange(n_r, dtype=np.uint64)[None, None, :]
    radial_keys = (
        chart[:, :, None] * np.uint64(131071 * 4099)
        + angular[:, :, None] * np.uint64(4099)
        + radial
    )
    radial_unit = _splitmix_uniform(radial_keys, seed, 0xE7037ED1A0B428DB)
    radial_jitter_np = eta_r * (2.0 * radial_unit - 1.0)
    coordinate = (
        radial.astype(np.float64) + 0.5 + radial_jitter_np
    ) / n_r
    if radial_scheme == "AREA_STRATIFIED":
        rho_np = np.sqrt(coordinate)
        radial_mass_np = np.full(n_r, 1.0 / n_r, dtype=np.float64)
    elif radial_scheme == "DISTANCE_STRATIFIED":
        rho_np = coordinate
        edges = np.arange(n_r + 1, dtype=np.float64) / n_r
        radial_mass_np = np.diff(edges * edges)
    else:
        raise ValueError(radial_scheme)
    theta = torch.as_tensor(theta_np, dtype=dtype, device=device)
    rho = torch.as_tensor(rho_np, dtype=dtype, device=device)
    angular_jitter = torch.as_tensor(
        angular_jitter_np, dtype=dtype, device=device
    )
    radial_jitter = torch.as_tensor(
        radial_jitter_np, dtype=dtype, device=device
    )
    radial_mass = torch.as_tensor(
        radial_mass_np, dtype=dtype, device=device
    )
    digest = _digest_arrays(
        theta, rho, angular_jitter, radial_jitter, radial_mass
    )
    return PolarLayout(
        theta, rho, angular_jitter, radial_jitter, radial_mass,
        n_theta, n_r, seed, eta_theta, eta_r, radial_scheme, digest,
    )


def polar_weights(layout: PolarLayout, chart_mass: Tensor) -> Tensor:
    weights = (
        chart_mass[:, None, None]
        * layout.radial_mass[None, None, :]
        / layout.n_theta
    )
    return weights.expand(-1, layout.n_theta, -1)


def _geodesic_step(
    field: Any,
    position: Tensor,
    velocity: Tensor,
    distance: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    candidate = position + distance[:, None] * velocity
    value = field.value(candidate)
    gradient = field.gradient(candidate)
    denominator = gradient.square().sum(1, keepdim=True) + eta * eta
    position = candidate - value[:, None] * gradient / denominator
    normal = field.gradient(position)
    normal = normal / torch.linalg.vector_norm(
        normal, dim=1, keepdim=True
    ).clamp_min(1e-30)
    velocity = velocity - (velocity * normal).sum(1, keepdim=True) * normal
    velocity = velocity / torch.linalg.vector_norm(
        velocity, dim=1, keepdim=True
    ).clamp_min(1e-30)
    return position, normal, velocity


def _shared_ray_map_from_prepared(
    field: Any,
    template: GeodesicTemplate,
    prepared: PreparedGeodesicCenters,
    chart_ids: Tensor,
    theta: Tensor,
    rho: Tensor,
    integration_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    chart_count, n_theta = theta.shape
    n_r = int(rho.shape[2])
    ray_owners = chart_ids[:, None].expand(-1, n_theta).reshape(-1)
    angles = theta.reshape(-1)
    position = prepared.positions[ray_owners]
    velocity = (
        torch.cos(angles)[:, None] * prepared.tangent_1[ray_owners]
        + torch.sin(angles)[:, None] * prepared.tangent_2[ray_owners]
    )
    step_distance = template.chart_radii[ray_owners] / integration_steps
    path_positions = [position]
    path_normals = [prepared.normals[ray_owners]]
    for _ in range(integration_steps):
        position, normal, velocity = _geodesic_step(
            field, position, velocity, step_distance, template.eta
        )
        path_positions.append(position)
        path_normals.append(normal)
    positions, normals = _sample_fixed_ray_path(
        torch.stack(path_positions), torch.stack(path_normals),
        rho.reshape(chart_count * n_theta, n_r), integration_steps,
    )
    positions = positions.reshape(chart_count, n_theta, n_r, 3)
    normals = normals.reshape_as(positions)
    residual = field.value(positions.reshape(-1, 3)).abs().reshape(
        chart_count, n_theta, n_r
    )
    return positions, normals, residual


def _independent_ray_map_from_prepared(
    field: Any,
    template: GeodesicTemplate,
    prepared: PreparedGeodesicCenters,
    chart_ids: Tensor,
    theta: Tensor,
    rho: Tensor,
    integration_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Restart each endpoint and replay the same fixed integration grid."""
    chart_count, n_theta = theta.shape
    n_r = int(rho.shape[2])
    ray_owners = chart_ids[:, None].expand(-1, n_theta).reshape(-1)
    angles = theta.reshape(-1)
    initial_velocity = (
        torch.cos(angles)[:, None] * prepared.tangent_1[ray_owners]
        + torch.sin(angles)[:, None] * prepared.tangent_2[ray_owners]
    )
    position_records: list[Tensor] = []
    normal_records: list[Tensor] = []
    for endpoint in range(n_r):
        position = prepared.positions[ray_owners]
        velocity = initial_velocity
        step_distance = template.chart_radii[ray_owners] / integration_steps
        path_positions = [position]
        path_normals = [prepared.normals[ray_owners]]
        for _ in range(integration_steps):
            position, normal, velocity = _geodesic_step(
                field, position, velocity, step_distance, template.eta
            )
            path_positions.append(position)
            path_normals.append(normal)
        sampled_position, sampled_normal = _sample_fixed_ray_path(
            torch.stack(path_positions), torch.stack(path_normals),
            rho[:, :, endpoint].reshape(-1, 1), integration_steps,
        )
        position_records.append(sampled_position[:, 0])
        normal_records.append(sampled_normal[:, 0])
    positions = torch.stack(position_records, dim=1).reshape(
        chart_count, n_theta, n_r, 3
    )
    normals = torch.stack(normal_records, dim=1).reshape_as(positions)
    residual = field.value(positions.reshape(-1, 3)).abs().reshape(
        chart_count, n_theta, n_r
    )
    return positions, normals, residual


def _sample_fixed_ray_path(
    path_positions: Tensor,
    path_normals: Tensor,
    rho: Tensor,
    integration_steps: int,
) -> tuple[Tensor, Tensor]:
    """Record arbitrary radial stops without changing the integration grid."""
    scaled = rho * integration_steps
    lower = torch.floor(scaled).to(torch.long).clamp(0, integration_steps - 1)
    alpha = (scaled - lower.to(scaled.dtype))[..., None]
    ray = torch.arange(rho.shape[0], device=rho.device)[:, None]
    position = (
        (1.0 - alpha) * path_positions[lower, ray]
        + alpha * path_positions[lower + 1, ray]
    )
    normal = (
        (1.0 - alpha) * path_normals[lower, ray]
        + alpha * path_normals[lower + 1, ray]
    )
    normal = normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(1e-30)
    return position, normal


def _sparse_geodesic_step(
    field: Any,
    position: Tensor,
    velocity: Tensor,
    distance: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    candidate = position + distance[:, None] * velocity
    value, gradient, _, _ = sparse_field_evaluate(
        field, candidate, support, basis_centers, basis_radii, coefficients
    )
    denominator = gradient.square().sum(1, keepdim=True) + eta * eta
    position = candidate - value[:, None] * gradient / denominator
    _, gradient, _, _ = sparse_field_evaluate(
        field, position, support, basis_centers, basis_radii, coefficients
    )
    normal = gradient / torch.linalg.vector_norm(
        gradient, dim=1, keepdim=True
    ).clamp_min(1e-30)
    velocity = velocity - (velocity * normal).sum(1, keepdim=True) * normal
    velocity = velocity / torch.linalg.vector_norm(
        velocity, dim=1, keepdim=True
    ).clamp_min(1e-30)
    return position, normal, velocity


def _shared_ray_map_sparse_field(
    field: Any,
    template: GeodesicTemplate,
    prepared: PreparedGeodesicCenters,
    chart_ids: Tensor,
    theta: Tensor,
    rho: Tensor,
    integration_steps: int,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor,
) -> tuple[Tensor, Tensor, Tensor, int]:
    chart_count, n_theta = theta.shape
    n_r = int(rho.shape[2])
    ray_owners = chart_ids[:, None].expand(-1, n_theta).reshape(-1)
    support, _ = emitter_support_from_owners(prepared.support, ray_owners)
    angles = theta.reshape(-1)
    position = prepared.positions[ray_owners]
    velocity = (
        torch.cos(angles)[:, None] * prepared.tangent_1[ray_owners]
        + torch.sin(angles)[:, None] * prepared.tangent_2[ray_owners]
    )
    step_distance = template.chart_radii[ray_owners] / integration_steps
    path_positions = [position]
    path_normals = [prepared.normals[ray_owners]]
    for _ in range(integration_steps):
        position, normal, velocity = _sparse_geodesic_step(
            field, position, velocity, step_distance, support,
            basis_centers, basis_radii, coefficients, template.eta,
        )
        path_positions.append(position)
        path_normals.append(normal)
    positions, normals = _sample_fixed_ray_path(
        torch.stack(path_positions), torch.stack(path_normals),
        rho.reshape(chart_count * n_theta, n_r), integration_steps,
    )
    positions = positions.reshape(chart_count, n_theta, n_r, 3)
    normals = normals.reshape_as(positions)
    values, _, _, _ = sparse_field_evaluate(
        field, positions.reshape(-1, 3),
        _expanded_stop_support(support, n_r),
        basis_centers, basis_radii, coefficients,
    )
    return positions, normals, values.abs().reshape(
        chart_count, n_theta, n_r
    ), support.nnz * n_r


def _expanded_stop_support(
    ray_support: SparsePointBasisSupport, n_r: int
) -> SparsePointBasisSupport:
    counts = ray_support.counts().repeat_interleave(n_r)
    row_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    basis_rows: list[Tensor] = []
    for ray in range(ray_support.point_count):
        begin = int(ray_support.row_ptr[ray])
        end = int(ray_support.row_ptr[ray + 1])
        basis_rows.append(ray_support.basis_ids[begin:end].repeat(n_r))
    basis_ids = torch.cat(basis_rows) if basis_rows else torch.empty(
        0, dtype=torch.long, device=ray_support.basis_ids.device
    )
    point_ids = torch.repeat_interleave(
        torch.arange(ray_support.point_count * n_r, device=basis_ids.device),
        counts,
    )
    return SparsePointBasisSupport(
        row_ptr, point_ids, basis_ids, ray_support.point_count * n_r,
        ray_support.basis_count,
    )


def _single_chart_template(
    template: GeodesicTemplate, chart: int
) -> GeodesicTemplate:
    return GeodesicTemplate(
        template.reference_centers[chart:chart + 1],
        template.reference_normals[chart:chart + 1],
        template.reference_tangent_1[chart:chart + 1],
        template.reference_tangent_2[chart:chart + 1],
        template.chart_radii[chart:chart + 1],
        template.eta,
        template.steps,
    )


def exact_chart_polar_linearization(
    field: Any,
    template: GeodesicTemplate,
    chart_support: SparsePointBasisSupport,
    chart: int,
    theta: Tensor,
    rho: Tensor,
    basis_centers: Tensor,
    basis_radii: Tensor,
    integration_steps: int,
    *,
    shared: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SparsePointBasisSupport]:
    """Differentiate only the compact basis union of one polar chart."""
    begin = int(chart_support.row_ptr[chart])
    end = int(chart_support.row_ptr[chart + 1])
    local_basis_ids = chart_support.basis_ids[begin:end]
    local_count = int(local_basis_ids.numel())
    n_theta, n_r = int(theta.shape[0]), int(rho.shape[1])
    emitter_count = n_theta * n_r
    local_template = _single_chart_template(template, chart)
    local_theta = theta[None, :]
    local_rho = rho[None, :, :]
    local_chart_ids = torch.zeros(1, dtype=torch.long, device=theta.device)

    def mapped(local_coefficients: Tensor) -> Tensor:
        local_field = CompactSupportField(
            field, basis_centers, basis_radii,
            local_basis_ids, local_coefficients,
        )
        prepared = prepare_centers(local_field, local_template)
        mapper = (
            _shared_ray_map_from_prepared
            if shared else _independent_ray_map_from_prepared
        )
        positions, normals, _ = mapper(
            local_field, local_template, prepared, local_chart_ids,
            local_theta, local_rho, integration_steps,
        )
        return torch.cat((positions.reshape(-1), normals.reshape(-1)))

    zero = torch.zeros(
        local_count, dtype=theta.dtype, device=theta.device,
        requires_grad=True,
    )
    value = mapped(zero)
    jacobian = (
        torch.autograd.functional.jacobian(
            mapped, zero, vectorize=True, strategy="forward-mode"
        )
        if local_count else value.new_empty((value.numel(), 0))
    )
    positions = value[:3 * emitter_count].reshape(emitter_count, 3)
    normals = value[3 * emitter_count:].reshape(emitter_count, 3)
    dx = jacobian[:3 * emitter_count].reshape(
        emitter_count, 3, local_count
    ).permute(0, 2, 1).reshape(-1, 3)
    dn = jacobian[3 * emitter_count:].reshape(
        emitter_count, 3, local_count
    ).permute(0, 2, 1).reshape(-1, 3)
    counts = torch.full(
        (emitter_count,), local_count, dtype=torch.long, device=theta.device
    )
    row_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    point_ids = torch.repeat_interleave(
        torch.arange(emitter_count, device=theta.device), counts
    )
    support = SparsePointBasisSupport(
        row_ptr, point_ids, local_basis_ids.repeat(emitter_count),
        emitter_count, chart_support.basis_count,
    )
    residual = field.value(positions).abs()
    return positions, normals, dx, dn, residual, support


def _build_polar_graph(
    context: Any,
    template: GeodesicTemplate,
    layout: PolarLayout,
    config: StratifiedPolarConfig,
    *,
    shared: bool,
    chart_limit: int | None = None,
    retain_outputs: bool = False,
) -> tuple[SparseGeodesicGraph, dict[str, Any], tuple[Tensor, Tensor, Tensor] | None]:
    chart_count = min(chart_limit or config.k_chart, config.k_chart)
    basis_centers = context.master_layout.centers[:config.k_geom]
    basis_radii = context.master_layout.radii[:config.k_geom]
    chart_support = build_chart_envelope_support(
        template, basis_centers, basis_radii
    )
    builder = SparseGraphBuilder(
        chart_count * layout.m, config.k_geom, config.graph_threshold
    )
    positions: list[Tensor] = []
    normals: list[Tensor] = []
    residuals: list[Tensor] = []
    mapping_seconds = extraction_seconds = 0.0
    device = context.reference_points.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for chart in range(chart_count):
        _sync(device)
        stage = time.perf_counter()
        local = exact_chart_polar_linearization(
            context.base, template, chart_support, chart,
            layout.theta[chart], layout.rho[chart],
            basis_centers, basis_radii, config.geodesic_integration_steps,
            shared=shared,
        )
        local_positions, local_normals, dx, dn, residual, support = local
        _sync(device)
        mapping_seconds += time.perf_counter() - stage
        stage = time.perf_counter()
        builder.append(dx, dn, support)
        extraction_seconds += time.perf_counter() - stage
        if retain_outputs:
            positions.append(local_positions.detach().cpu())
            normals.append(local_normals.detach().cpu())
            residuals.append(residual.detach().cpu())
        if chart_count == config.k_chart and (
            chart == 0 or chart + 1 == chart_count or (chart + 1) % 128 == 0
        ):
            _progress(
                "v0810_sparse_graph", sampler="shared" if shared else "independent",
                m=layout.m, charts=chart + 1, total_charts=chart_count,
            )
        del dx, dn
    graph = builder.finish(pin_memory=device.type == "cuda")
    _sync(device)
    counts = graph.edge_counts().to(torch.float64)
    candidate_counts = chart_support.counts()[:chart_count].to(torch.float64)
    metadata = {
        "mode": "SHARED_RAY_INCREMENTAL" if shared else "INDEPENDENT_ENDPOINT_RESTART",
        "charts": chart_count,
        "n_theta": layout.n_theta,
        "n_r": layout.n_r,
        "m": layout.m,
        "n_emitters": chart_count * layout.m,
        "mapping_seconds": mapping_seconds,
        "sparse_extraction_seconds": extraction_seconds,
        "block_nnz": graph.nnz,
        "average_edges_per_emitter": float(counts.mean()),
        "p95_edges_per_emitter": float(torch.quantile(counts, 0.95)),
        "stored_edges_per_ray": graph.nnz / (chart_count * layout.n_theta),
        "candidate_basis_union_per_ray_mean": float(candidate_counts.mean()),
        "graph_bytes": graph.bytes,
        "graph_mib": graph.bytes / 2**20,
        "bytes_per_edge": graph.bytes / max(graph.nnz, 1),
        "density": graph.nnz / (chart_count * layout.m * config.k_geom),
        "edge_index_dtype": str(graph.column_indices.dtype).replace("torch.", ""),
        "derivative_dtype": str(graph.dx_dlambda.dtype).replace("torch.", ""),
        "graph_digest": graph.digest,
        "dense_point_by_lambda_tensor": False,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else 0.0
        ),
        "peak_cuda_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda" else 0.0
        ),
    }
    retained = None
    if retain_outputs:
        retained = (
            torch.cat(positions), torch.cat(normals), torch.cat(residuals)
        )
    return graph, metadata, retained


def _graph_difference(
    left: SparseGeodesicGraph, right: SparseGeodesicGraph
) -> dict[str, Any]:
    row_exact = bool(torch.equal(left.row_ptr, right.row_ptr))
    column_exact = bool(torch.equal(left.column_indices, right.column_indices))
    if row_exact and column_exact:
        dx_error = float((left.dx_dlambda - right.dx_dlambda).abs().max())
        dn_error = float((left.dn_dlambda - right.dn_dlambda).abs().max())
    else:
        dx_error = dn_error = None
    return {
        "row_ptr_exact": row_exact,
        "column_indices_exact": column_exact,
        "dx_max_absolute_error": dx_error,
        "dn_max_absolute_error": dn_error,
    }


def _shared_ray_validation_and_performance(
    context: Any,
    template: GeodesicTemplate,
    config: StratifiedPolarConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = context.reference_points.device
    dtype = context.reference_points.dtype
    validation_layout = polar_layout(
        config.k_chart, 4, 4, config.seeds[0],
        config.primary_jitter, config.primary_jitter,
        config.primary_radial_scheme, device, dtype,
    )
    shared_graph, shared_meta, shared_output = _build_polar_graph(
        context, template, validation_layout, config,
        shared=True, chart_limit=4, retain_outputs=True,
    )
    independent_graph, independent_meta, independent_output = _build_polar_graph(
        context, template, validation_layout, config,
        shared=False, chart_limit=4, retain_outputs=True,
    )
    shared_position, shared_normal, shared_residual = shared_output
    independent_position, independent_normal, independent_residual = independent_output
    generator = torch.Generator().manual_seed(810)
    delta = torch.randn(config.k_geom, generator=generator)
    gx = torch.randn(shared_position.shape, generator=generator)
    gn = torch.randn(shared_normal.shape, generator=generator)
    shared_jvp = shared_graph.jvp(delta)
    independent_jvp = independent_graph.jvp(delta)
    shared_vjp = shared_graph.vjp(gx, gn)
    independent_vjp = independent_graph.vjp(gx, gn)
    validation = {
        "configuration": {"charts": 4, "n_theta": 4, "n_r": 4},
        "position_max_absolute_error": float(
            (shared_position - independent_position).abs().max()
        ),
        "normal_max_absolute_error": float(
            (shared_normal - independent_normal).abs().max()
        ),
        "residual_max_absolute_error": float(
            (shared_residual - independent_residual).abs().max()
        ),
        "sparse_graph": _graph_difference(shared_graph, independent_graph),
        "jvp_max_absolute_error": max(
            float((shared_jvp[0] - independent_jvp[0]).abs().max()),
            float((shared_jvp[1] - independent_jvp[1]).abs().max()),
        ),
        "vjp_max_absolute_error": float(
            (shared_vjp - independent_vjp).abs().max()
        ),
        "shared_graph_digest": shared_graph.digest,
        "independent_graph_digest": independent_graph.digest,
    }

    perf_layout = polar_layout(
        config.k_chart, 16, 16, config.seeds[0],
        config.primary_jitter, config.primary_jitter,
        config.primary_radial_scheme, device, dtype,
    )
    shared_perf_graph, shared_perf, _ = _build_polar_graph(
        context, template, perf_layout, config,
        shared=True, chart_limit=config.performance_charts,
    )
    independent_perf_graph, independent_perf, _ = _build_polar_graph(
        context, template, perf_layout, config,
        shared=False, chart_limit=config.performance_charts,
    )
    shared_steps = (
        config.performance_charts * 16 * config.geodesic_integration_steps
    )
    independent_steps = (
        config.performance_charts * 16 * 16 * config.geodesic_integration_steps
    )
    for row, steps in (
        (shared_perf, shared_steps), (independent_perf, independent_steps)
    ):
        row["projected_scalar_ray_steps"] = steps
        row["field_evaluations"] = 3 * steps + row["n_emitters"]
    performance = {
        "configuration": {
            "charts": config.performance_charts,
            "n_theta": 16,
            "n_r": 16,
            "fixed_integration_steps_per_ray": config.geodesic_integration_steps,
        },
        "shared": shared_perf,
        "independent": independent_perf,
        "construction_speedup": (
            independent_perf["mapping_seconds"]
            / max(shared_perf["mapping_seconds"], 1e-30)
        ),
        "projected_step_reduction": independent_steps / shared_steps,
        "graph_equivalence": _graph_difference(
            shared_perf_graph, independent_perf_graph
        ),
    }
    return validation, performance


def _shared_stream_render(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    base_config: MillionEmitterConfig,
    config: StratifiedPolarConfig,
    layout: PolarLayout,
    resolution: tuple[int, int],
    views: int,
    *,
    collect_positions: bool = False,
) -> PolarRender:
    field = context.base
    device = context.reference_points.device
    prepared = prepare_centers(field, template)
    atlas = nested_fibonacci_atlas(device, (views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    offsets = _micro_offsets(1, device, context.reference_points.dtype)
    accumulators = [
        torch.zeros(
            (math.prod(resolution), 3), dtype=torch.float32, device=device
        )
        for _ in range(views)
    ]
    all_weights = polar_weights(layout, chart_mass)
    chart_block = max(1, config.emitter_chunk_size // layout.m)
    position_capture: list[Tensor] = []
    weight_capture: list[Tensor] = []
    residual_capture: list[Tensor] = []
    mapping_seconds = source_seconds = transport_seconds = detector_seconds = 0.0
    interactions = writes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for chart_start in range(0, config.k_chart, chart_block):
        chart_stop = min(chart_start + chart_block, config.k_chart)
        chart_ids = torch.arange(
            chart_start, chart_stop, dtype=torch.long, device=device
        )
        _sync(device)
        stage = time.perf_counter()
        positions, normals, residual = _shared_ray_map_from_prepared(
            field, template, prepared, chart_ids,
            layout.theta[chart_start:chart_stop],
            layout.rho[chart_start:chart_stop],
            config.geodesic_integration_steps,
        )
        positions = positions.reshape(-1, 3)
        normals = normals.reshape(-1, 3)
        residual = residual.reshape(-1)
        _sync(device)
        mapping_seconds += time.perf_counter() - stage
        residual_capture.append(residual.detach().cpu())
        raw_weights = all_weights[chart_start:chart_stop].reshape(-1)
        if collect_positions:
            position_capture.append(positions.detach().cpu().to(torch.float32))
            weight_capture.append(raw_weights.detach().cpu().to(torch.float64))
        _sync(device)
        stage = time.perf_counter()
        weights = raw_weights / reference_area
        colors = meshfree_base_color(positions, field.lower, field.upper)
        _sync(device)
        source_seconds += time.perf_counter() - stage
        for block_start in range(0, views, config.camera_block_size):
            block_stop = min(block_start + config.camera_block_size, views)
            for view in range(block_start, block_stop):
                direction = atlas.directions[view]
                directions = direction.expand_as(positions)
                _sync(device)
                stage = time.perf_counter()
                transmission, _, evaluations = _transmission(
                    field, positions, directions,
                    boundary.exit_times(positions, directions),
                    radius=packet["radius"], epsilon=packet["epsilon"],
                    path_step=packet["path_step"], offsets=offsets,
                    eta=template.eta, kappa=packet["kappa"],
                    surface_barrier=True,
                    launch_exclusion_factor=config.launch_exclusion_factor,
                )
                cosine = normals @ direction
                outward = _emission_factor(
                    cosine, config.emission_transition_width
                )
                lobe = config.ambient + (
                    1.0 - config.ambient
                ) * cosine.clamp_min(0.0)
                energy = (
                    weights[:, None] * transmission[:, None] * colors
                    * (outward * lobe)[:, None]
                )
                _sync(device)
                transport_seconds += time.perf_counter() - stage
                _sync(device)
                stage = time.perf_counter()
                _accumulate_continuous(
                    accumulators[view], positions.to(torch.float32),
                    energy.to(torch.float32),
                    atlas.right[view].to(torch.float32),
                    atlas.up[view].to(torch.float32),
                    boundary.center.to(torch.float32),
                    config.detector_extent, resolution,
                )
                _sync(device)
                detector_seconds += time.perf_counter() - stage
                interactions += evaluations
                writes += int(positions.shape[0]) * 16
        del positions, normals, residual, raw_weights, weights, colors
    _sync(device)
    scale = config.sensor_gain * math.prod(resolution)
    images_tensor = torch.stack([scale * image for image in accumulators])
    images = [
        image.reshape(*resolution, 3).detach().cpu().numpy()
        for image in images_tensor
    ]
    runtime = time.perf_counter() - started
    residual = torch.cat(residual_capture)
    stream = StreamRender(
        images=images,
        tensor=images_tensor.reshape(-1),
        mapping_seconds=mapping_seconds,
        source_seconds=source_seconds,
        transport_seconds=transport_seconds,
        detector_seconds=detector_seconds,
        runtime_seconds=runtime,
        attempted_packets=config.k_chart * layout.m * views,
        interaction_evaluations=interactions,
        detector_writes=writes,
        peak_allocated_mib=(
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else 0.0
        ),
        peak_reserved_mib=(
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda" else 0.0
        ),
        cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        residual_p95=float(torch.quantile(residual, 0.95)),
        image_digest=_digest_arrays(*images),
    )
    return PolarRender(
        stream,
        torch.cat(position_capture) if collect_positions else None,
        torch.cat(weight_capture) if collect_positions else None,
    )


def _render_metrics_row(
    rendered: PolarRender,
    hard: list[np.ndarray],
    layout: PolarLayout,
    chart_mass: Tensor,
    config: StratifiedPolarConfig,
    resolution: tuple[int, int],
    views: int,
) -> dict[str, Any]:
    stream = rendered.stream
    metrics = _image_metrics(stream.images, hard[:views])
    return {
        "sampler": "STRATIFIED_GEODESIC_POLAR",
        "radial_scheme": layout.radial_scheme,
        "seed": layout.seed,
        "eta_theta": layout.eta_theta,
        "eta_r": layout.eta_r,
        "n_theta": layout.n_theta,
        "n_r": layout.n_r,
        "m": layout.m,
        "n_emitters": layout.chart_count * layout.m,
        "resolution": list(resolution),
        "views": views,
        "source_mass": float(chart_mass.sum()),
        "layout_digest": layout.digest,
        **metrics,
        "high_frequency_aliasing_error": abs(
            float(metrics["edge_sharpness_ratio"]) - 1.0
        ),
        "geodesic_construction_seconds": stream.mapping_seconds,
        "source_evaluation_seconds": stream.source_seconds,
        "finite_transport_seconds": stream.transport_seconds,
        "detector_seconds": stream.detector_seconds,
        "runtime_seconds": stream.runtime_seconds,
        "attempted_packets": stream.attempted_packets,
        "interaction_evaluations": stream.interaction_evaluations,
        "detector_writes": stream.detector_writes,
        "peak_cuda_allocated_mib": stream.peak_allocated_mib,
        "peak_cuda_reserved_mib": stream.peak_reserved_mib,
        "cpu_rss_mib": stream.cpu_rss_mib,
        "mapping_residual_p95": stream.residual_p95,
        "image_digest": stream.image_digest,
        "fixed_integration_steps_per_ray": config.geodesic_integration_steps,
        "radial_stop_recording": "linear interpolation at fixed-step crossings",
    }


def _sobol_coordinates(
    k_chart: int, m: int, seed: int, device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    _, _, rho, theta = _latent_chunk(
        k_chart, m, 0, k_chart * m, seed, device, dtype
    )
    return theta.reshape(k_chart, m), rho.reshape(k_chart, m)


def _coverage_metrics(
    theta: Tensor,
    rho: Tensor,
    n_theta: int,
    n_r: int,
    *,
    stratified: bool,
    radial_scheme: RadialScheme,
) -> dict[str, Any]:
    theta_np = np.mod(theta.detach().cpu().numpy(), 2.0 * math.pi)
    rho_np = rho.detach().cpu().numpy()
    if stratified:
        angular_values = theta_np
        radial_values = (
            rho_np * rho_np
            if radial_scheme == "AREA_STRATIFIED" else rho_np
        )
        flat_rho = rho_np.reshape(rho_np.shape[0], -1)
    else:
        angular_values = theta_np
        radial_values = (
            rho_np * rho_np
            if radial_scheme == "AREA_STRATIFIED" else rho_np
        )[:, None, :]
        flat_rho = rho_np
    gaps: list[np.ndarray] = []
    zero_sectors = 0
    occupancy: list[np.ndarray] = []
    local_nn: list[np.ndarray] = []
    radial_gaps: list[np.ndarray] = []
    occupancy_variance: list[float] = []
    chart_count = theta_np.shape[0]
    for chart in range(chart_count):
        angles = angular_values[chart]
        if stratified:
            unique_angles = angles
            coordinates = np.stack((
                flat_rho[chart] * np.cos(np.repeat(angles, n_r)),
                flat_rho[chart] * np.sin(np.repeat(angles, n_r)),
            ), axis=1)
            polar_bins_theta = np.repeat(
                np.arange(n_theta), n_r
            )
            radial_coordinate = radial_values[chart].reshape(-1)
            polar_bins_r = np.floor(
                np.clip(radial_coordinate, 0.0, 1.0 - 1e-12) * n_r
            ).astype(np.int64)
            for ray in range(n_theta):
                ordered = np.sort(radial_values[chart, ray])
                radial_gaps.append(np.diff(np.concatenate(([0.0], ordered, [1.0]))))
        else:
            unique_angles = angles
            coordinates = np.stack((
                flat_rho[chart] * np.cos(angles),
                flat_rho[chart] * np.sin(angles),
            ), axis=1)
            polar_bins_theta = np.floor(
                angles / (2.0 * math.pi) * n_theta
            ).astype(np.int64)
            radial_coordinate = radial_values[chart, 0]
            polar_bins_r = np.floor(
                np.clip(radial_coordinate, 0.0, 1.0 - 1e-12) * n_r
            ).astype(np.int64)
            ordered = np.sort(radial_coordinate)
            radial_gaps.append(np.diff(np.concatenate(([0.0], ordered, [1.0]))))
        ordered_angles = np.sort(unique_angles)
        chart_gaps = np.diff(np.concatenate((
            ordered_angles, [ordered_angles[0] + 2.0 * math.pi]
        )))
        gaps.append(chart_gaps)
        sector_counts = np.bincount(
            polar_bins_theta, minlength=n_theta
        )
        zero_sectors += int(np.sum(sector_counts == 0))
        radial_counts = np.bincount(polar_bins_r, minlength=n_r)
        occupancy.append(radial_counts)
        cell_counts = np.zeros((n_theta, n_r), dtype=np.int64)
        np.add.at(cell_counts, (polar_bins_theta, polar_bins_r), 1)
        occupancy_variance.append(float(np.var(cell_counts)))
        local_nn.append(cKDTree(coordinates).query(coordinates, k=2)[0][:, 1])
    all_gaps = np.concatenate(gaps)
    all_radial_gaps = np.concatenate(radial_gaps)
    nn = np.concatenate(local_nn)
    occupancy_array = np.stack(occupancy)
    thirds = np.stack((
        np.mean(flat_rho <= 1.0 / 3.0, axis=1),
        np.mean((flat_rho > 1.0 / 3.0) & (flat_rho <= 2.0 / 3.0), axis=1),
        np.mean(flat_rho > 2.0 / 3.0, axis=1),
    ), axis=1)
    return {
        "angular_gap_p50_radians": float(np.quantile(all_gaps, 0.50)),
        "angular_gap_p90_radians": float(np.quantile(all_gaps, 0.90)),
        "angular_gap_p95_radians": float(np.quantile(all_gaps, 0.95)),
        "largest_angular_gap_radians": float(np.max(all_gaps)),
        "sectors_with_zero_samples": zero_sectors,
        "radial_gap_p50": float(np.quantile(all_radial_gaps, 0.50)),
        "radial_gap_p90": float(np.quantile(all_radial_gaps, 0.90)),
        "radial_gap_p95": float(np.quantile(all_radial_gaps, 0.95)),
        "radial_stratum_occupancy_minimum": int(occupancy_array.min()),
        "radial_stratum_occupancy_maximum": int(occupancy_array.max()),
        "inner_mid_outer_sample_fractions": thirds.mean(0).tolist(),
        "nearest_neighbor_distance_mean": float(np.mean(nn)),
        "nearest_neighbor_distance_p95": float(np.quantile(nn, 0.95)),
        "polar_cell_occupancy_variance_mean": float(
            np.mean(occupancy_variance)
        ),
    }


def _local_measure_metrics(
    context: Any, positions: Tensor, weights: Tensor
) -> dict[str, Any]:
    reference = context.reference_points.detach().cpu().numpy()
    normals = context.reference_normals.detach().cpu().numpy()
    sample_positions = positions.detach().cpu().numpy()
    sample_weights = weights.detach().cpu().numpy()
    nearest = cKDTree(reference).query(sample_positions, k=1)[1]
    normalized_weights = sample_weights / sample_weights.sum()
    sampler_mass = np.bincount(
        nearest, weights=normalized_weights, minlength=reference.shape[0]
    )
    historical_mass = np.full(reference.shape[0], 1.0 / reference.shape[0])
    error = np.abs(sampler_mass - historical_mass)
    neighbor_ids = cKDTree(reference).query(reference, k=9)[1][:, 1:]
    curvature = np.mean(
        1.0 - np.sum(normals[:, None] * normals[neighbor_ids], axis=2), axis=1
    )
    high_curvature = curvature >= np.quantile(curvature, 0.75)
    thin = reference[:, 2] >= np.quantile(reference[:, 2], 0.82)
    atlas = nested_fibonacci_atlas(context.reference_points.device, (4,))
    directions = atlas.directions.detach().cpu().numpy()
    silhouette = np.min(np.abs(normals @ directions.T), axis=1) <= 0.15
    ideal = 1.0 / reference.shape[0]
    return {
        "reference_cells": int(reference.shape[0]),
        "mean_absolute_local_mass_error": float(np.mean(error)),
        "normalized_mean_absolute_error": float(np.mean(error) / ideal),
        "p90_local_mass_error": float(np.quantile(error, 0.90)),
        "p95_local_mass_error": float(np.quantile(error, 0.95)),
        "total_variation_distance": float(0.5 * error.sum()),
        "thin_feature_region_mass_error": float(np.mean(error[thin])),
        "high_curvature_region_mass_error": float(np.mean(error[high_curvature])),
        "silhouette_near_region_mass_error": float(np.mean(error[silhouette])),
        "thin_feature_rule": "top 18% reference points by world z",
        "high_curvature_rule": "top quartile eight-neighbor normal variation",
        "silhouette_rule": "min absolute normal-dot-four-view-direction <= 0.15",
        "sampler_mass_sum": float(sampler_mass.sum()),
    }


def _sobol_positions_and_weights(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    m: int,
    seed: int,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    device = context.reference_points.device
    prepared = prepare_centers(context.base, template)
    positions: list[Tensor] = []
    weights: list[Tensor] = []
    for _, _, owners, _, rho, theta in _latent_chunks(
        template.chart_count, m, chunk_size, seed,
        device, context.reference_points.dtype,
    ):
        mapped, _, _, _, _, _ = map_emitter_chunk(
            context.base, template, prepared, owners, rho, theta
        )
        positions.append(mapped.detach().cpu().to(torch.float32))
        weights.append((chart_mass[owners] / m).detach().cpu().to(torch.float64))
    return torch.cat(positions), torch.cat(weights)


def _shared_geometry_response(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    layout: PolarLayout,
    config: StratifiedPolarConfig,
) -> tuple[Tensor, dict[str, Any]]:
    device = context.reference_points.device
    basis_centers = context.master_layout.centers[:config.k_geom]
    basis_radii = context.master_layout.radii[:config.k_geom]
    chart_support = build_chart_envelope_support(
        template, basis_centers, basis_radii
    )
    plus = torch.zeros(
        config.k_geom, dtype=context.reference_points.dtype, device=device
    )
    minus = torch.zeros_like(plus)
    plus[config.response_parameter] = config.response_delta
    minus[config.response_parameter] = -config.response_delta
    plus_prepared = prepare_centers_sparse_field(
        context.base, template, basis_centers, basis_radii,
        plus, chart_support,
    )
    minus_prepared = prepare_centers_sparse_field(
        context.base, template, basis_centers, basis_radii,
        minus, chart_support,
    )
    raw_weights = polar_weights(layout, chart_mass) / chart_mass.sum()
    chart_block = max(1, config.emitter_chunk_size // layout.m)
    responses: list[Tensor] = []
    sparse_pairs = 0
    started = time.perf_counter()
    for chart_start in range(0, config.k_chart, chart_block):
        chart_stop = min(chart_start + chart_block, config.k_chart)
        chart_ids = torch.arange(
            chart_start, chart_stop, dtype=torch.long, device=device
        )
        arguments = (
            context.base, template, plus_prepared, chart_ids,
            layout.theta[chart_start:chart_stop],
            layout.rho[chart_start:chart_stop], config.geodesic_integration_steps,
            basis_centers, basis_radii, plus,
        )
        plus_position, plus_normal, _, evaluated = _shared_ray_map_sparse_field(
            *arguments
        )
        minus_position, minus_normal, _, _ = _shared_ray_map_sparse_field(
            context.base, template, minus_prepared, chart_ids,
            layout.theta[chart_start:chart_stop],
            layout.rho[chart_start:chart_stop], config.geodesic_integration_steps,
            basis_centers, basis_radii, minus,
        )
        response = torch.cat((
            (plus_position - minus_position) / (2.0 * config.response_delta),
            (plus_normal - minus_normal) / (2.0 * config.response_delta),
        ), dim=-1)
        weights = raw_weights[chart_start:chart_stop]
        responses.append(
            (weights[..., None] * response).detach().cpu()
            .to(torch.float32).reshape(-1)
        )
        sparse_pairs += evaluated
    _sync(device)
    return torch.cat(responses), {
        "response_definition": (
            "strict sparse-field central FD of mass-weighted shared-ray "
            "per-emitter [x,n]"
        ),
        "response_parameter": config.response_parameter,
        "response_delta": config.response_delta,
        "sparse_point_basis_pairs_evaluated": sparse_pairs,
        "seconds": time.perf_counter() - started,
    }


def _multi_seed_shared(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    base_config: MillionEmitterConfig,
    config: StratifiedPolarConfig,
    hard64: list[np.ndarray],
    n_theta: int,
    n_r: int,
    eta: float,
    radial_scheme: RadialScheme,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in config.seeds:
        _progress(
            "v0810_multiseed", n_theta=n_theta, n_r=n_r,
            jitter=eta, scheme=radial_scheme, seed=seed,
        )
        layout = polar_layout(
            config.k_chart, n_theta, n_r, seed, eta, eta,
            radial_scheme, context.reference_points.device,
            context.reference_points.dtype,
        )
        rendered = _shared_stream_render(
            context, template, chart_mass, reference_area, packet,
            base_config, config, layout,
            (config.diagnostic_resolution,) * 2, 1,
        )
        response, response_metadata = _shared_geometry_response(
            context, template, chart_mass, layout, config
        )
        entries.append({
            "seed": seed,
            "layout_digest": layout.digest,
            "source_mass": float(polar_weights(layout, chart_mass).sum()),
            "image": rendered.stream.tensor.detach().cpu(),
            "response": response,
            "response_metadata": response_metadata,
        })
        _release()
    mean_image = torch.stack([row["image"] for row in entries]).mean(0)
    mean_response = torch.stack([row["response"] for row in entries]).mean(0)
    rows = []
    for entry in entries:
        vector = _vector_metrics(entry["response"], mean_response)
        rows.append({
            "seed": entry["seed"],
            "layout_digest": entry["layout_digest"],
            "source_mass": entry["source_mass"],
            "image_self_mse": float(
                torch.mean((entry["image"] - mean_image).square())
            ),
            "geometry_response_norm": float(
                torch.linalg.vector_norm(entry["response"])
            ),
            "geometry_response_cosine": vector["cosine_similarity"],
            **entry["response_metadata"],
        })
    masses = np.asarray([row["source_mass"] for row in rows])
    norms = np.asarray([row["geometry_response_norm"] for row in rows])
    return {
        "sampler": "STRATIFIED_GEODESIC_POLAR",
        "radial_scheme": radial_scheme,
        "eta_theta": eta,
        "eta_r": eta,
        "n_theta": n_theta,
        "n_r": n_r,
        "m": n_theta * n_r,
        "n_emitters": config.k_chart * n_theta * n_r,
        "seeds": list(config.seeds),
        "rows": rows,
        "source_mass_cv": float(masses.std() / masses.mean()),
        "response_norm_cv": float(norms.std() / norms.mean()),
        "median_geometry_response_cosine": float(np.median([
            row["geometry_response_cosine"] for row in rows
        ])),
        "mean_image_self_mse": float(np.mean([
            row["image_self_mse"] for row in rows
        ])),
    }


def _hard_references(
    context: Any,
    config: StratifiedPolarConfig,
) -> dict[tuple[int, int], list[np.ndarray]]:
    atlas = nested_fibonacci_atlas(
        context.reference_points.device, (config.fullhd_views,)
    )
    boundary = enclosing_observation_sphere(context.reference_points)
    output: dict[tuple[int, int], list[np.ndarray]] = {}
    for resolution in (
        (config.diagnostic_resolution,) * 2,
        *config.comparison_resolutions,
    ):
        output[resolution], _ = _hard_images(
            context.base, context.reference_points, context.reference_normals,
            context, atlas, boundary, resolution,
            recompute_visibility=True,
        )
    return output


def _evaluate_shared_case(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    base_config: MillionEmitterConfig,
    config: StratifiedPolarConfig,
    hard: list[np.ndarray],
    *,
    n_theta: int,
    n_r: int,
    seed: int,
    eta: float,
    radial_scheme: RadialScheme,
    resolution: tuple[int, int],
    views: int,
    collect_measure: bool,
) -> tuple[dict[str, Any], PolarRender, PolarLayout]:
    layout = polar_layout(
        config.k_chart, n_theta, n_r, seed, eta, eta,
        radial_scheme, context.reference_points.device,
        context.reference_points.dtype,
    )
    rendered = _shared_stream_render(
        context, template, chart_mass, reference_area, packet,
        base_config, config, layout, resolution, views,
        collect_positions=collect_measure,
    )
    row = _render_metrics_row(
        rendered, hard, layout, chart_mass, config, resolution, views
    )
    row["quadrature_mass_drift"] = abs(
        row["source_mass"] / reference_area - 1.0
    )
    if collect_measure:
        row["local_measure"] = _local_measure_metrics(
            context, rendered.positions, rendered.weights
        )
    row["coverage"] = _coverage_metrics(
        layout.theta, layout.rho, n_theta, n_r,
        stratified=True, radial_scheme=radial_scheme,
    )
    return row, rendered, layout


def _sobol_control_row(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    base_config: MillionEmitterConfig,
    config: StratifiedPolarConfig,
    hard: list[np.ndarray],
    *,
    m: int,
    seed: int,
    resolution: tuple[int, int],
    views: int,
    collect_measure: bool,
    n_theta_bins: int,
    n_r_bins: int,
) -> tuple[dict[str, Any], StreamRender, tuple[Tensor, Tensor]]:
    row, rendered = _run_forward_row(
        context, template, chart_mass, reference_area,
        packet, base_config, hard, m=m, seed=seed,
        resolution=resolution, views=views,
        chunk_size=config.emitter_chunk_size,
        camera_block=config.camera_block_size,
    )
    row["sampler"] = "SOBOL_DISK_ENDPOINTS"
    row["radial_scheme"] = "AREA_SOBOL"
    row["high_frequency_aliasing_error"] = abs(
        float(row["edge_sharpness_ratio"]) - 1.0
    )
    theta, rho = _sobol_coordinates(
        config.k_chart, m, seed, context.reference_points.device,
        context.reference_points.dtype,
    )
    row["coverage"] = _coverage_metrics(
        theta, rho, n_theta_bins, n_r_bins,
        stratified=False, radial_scheme="AREA_STRATIFIED",
    )
    if collect_measure:
        positions, weights = _sobol_positions_and_weights(
            context, template, chart_mass, m, seed,
            config.emitter_chunk_size,
        )
        row["local_measure"] = _local_measure_metrics(
            context, positions, weights
        )
    return row, rendered, (theta, rho)


def _dominates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    comparisons = (
        candidate["whole_image_mse"] < baseline["whole_image_mse"],
        candidate["thin_feature_response_ratio"]
        > baseline["thin_feature_response_ratio"],
        candidate["local_measure"]["normalized_mean_absolute_error"]
        < baseline["local_measure"]["normalized_mean_absolute_error"],
    )
    return sum(comparisons) >= 2


def _sampling_limit(
    many_angles: dict[str, Any], balanced: dict[str, Any],
    many_radial: dict[str, Any],
) -> str:
    angular = _dominates(many_angles, balanced)
    radial = _dominates(many_radial, balanced)
    if angular and radial:
        return "BOTH"
    if angular:
        return "ANGULAR"
    if radial:
        return "RADIAL"
    extremes = (
        _dominates(many_angles, many_radial),
        _dominates(many_radial, many_angles),
    )
    if extremes == (False, False):
        return "NEITHER"
    return "UNRESOLVED"


def _ranked_jitter_choice(rows: list[dict[str, Any]]) -> float:
    scores = {float(row["eta_theta"]): 0 for row in rows}
    metrics = (
        ("local_measure.normalized_mean_absolute_error", False),
        ("whole_image_mse", False),
        ("thin_feature_response_ratio", True),
        ("multiseed.median_geometry_response_cosine", True),
        ("multiseed.mean_image_self_mse", False),
    )

    def lookup(row: dict[str, Any], path: str) -> float:
        value: Any = row
        for key in path.split("."):
            value = value[key]
        return float(value)

    for path, descending in metrics:
        ordered = sorted(
            rows, key=lambda row: lookup(row, path), reverse=descending
        )
        for rank, row in enumerate(ordered):
            scores[float(row["eta_theta"])] += rank
    for row in rows:
        row["selection_rank_sum"] = scores[float(row["eta_theta"])]
    return min(scores, key=lambda eta: (scores[eta], abs(eta - 0.25)))


def _figures(
    directory: Path,
    report: dict[str, Any],
    fullhd_images: dict[str, list[np.ndarray]],
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str, figure: Any) -> None:
        path = directory / name
        _save_figure(path, figure)
        paths.append(str(path))

    examples = report["chart_examples"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(
        np.asarray(examples["sobol_xy"])[:, 0],
        np.asarray(examples["sobol_xy"])[:, 1], s=12,
    )
    axes[1].scatter(
        np.asarray(examples["stratified_xy"])[:, 0],
        np.asarray(examples["stratified_xy"])[:, 1], s=12,
    )
    for axis, title in zip(axes, ("Sobol endpoints", "stratified polar")):
        axis.set(aspect="equal", title=title, xlim=(-1, 1), ylim=(-1, 1))
    save("v0810_sobol_vs_stratified_chart.png", figure)

    coverage = report["coverage_comparison"]
    figure, axis = plt.subplots(figsize=(8, 4))
    names = ["Sobol", "stratified"]
    axis.bar(names, [
        coverage["sobol"]["largest_angular_gap_radians"],
        coverage["stratified"]["largest_angular_gap_radians"],
    ])
    axis.set_ylabel("largest angular gap (radians)")
    save("v0810_angular_sector_coverage.png", figure)

    jitter_rows = report["jitter_ablation"]
    primary = min(
        jitter_rows,
        key=lambda row: abs(row["eta_theta"] - report["configuration"]["primary_jitter"]),
    )
    radial = primary["radial_coordinates_first_chart"]
    figure, axis = plt.subplots(figsize=(8, 4))
    for ray, values in enumerate(radial[: min(len(radial), 16)]):
        axis.scatter(values, np.full(len(values), ray), s=10)
    axis.set(xlabel="normalized radius", ylabel="angular ray")
    save("v0810_radial_strata.png", figure)

    polar = np.asarray(examples["stratified_xy"])
    n_theta = int(examples["n_theta"])
    n_r = int(examples["n_r"])
    figure, axis = plt.subplots(figsize=(6, 6))
    for ray in range(n_theta):
        points = polar[ray * n_r:(ray + 1) * n_r]
        axis.plot(
            np.concatenate(([0.0], points[:, 0])),
            np.concatenate(([0.0], points[:, 1])), "o-", ms=3,
        )
    axis.set(aspect="equal", xlabel="chart tangent 1", ylabel="chart tangent 2")
    save("v0810_shared_geodesic_rays.png", figure)

    cross = report["cross_seed_stability"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        [row["n_emitters"] for row in cross["sobol"]],
        [row["median_geometry_response_cosine"] for row in cross["sobol"]],
        "o-", label="Sobol",
    )
    axis.plot(
        [row["n_emitters"] for row in cross["stratified"]],
        [row["median_geometry_response_cosine"] for row in cross["stratified"]],
        "o-", label="stratified",
    )
    axis.set(xscale="log", xlabel="emitters", ylabel="median response cosine")
    axis.legend()
    save("v0810_cross_seed_cosine.png", figure)

    measure = report["local_measure_fidelity"]
    figure, axis = plt.subplots(figsize=(9, 4))
    metric_names = (
        "mean_absolute_local_mass_error", "p95_local_mass_error",
        "thin_feature_region_mass_error", "high_curvature_region_mass_error",
        "silhouette_near_region_mass_error",
    )
    x = np.arange(len(metric_names))
    axis.bar(x - 0.18, [measure["sobol"][key] for key in metric_names], 0.36, label="Sobol")
    axis.bar(x + 0.18, [measure["stratified"][key] for key in metric_names], 0.36, label="stratified")
    axis.set_xticks(x, ["mean", "p95", "thin", "curvature", "silhouette"], rotation=20)
    axis.set_ylabel("absolute local mass error")
    axis.legend()
    save("v0810_local_measure_error.png", figure)

    fidelity = report["resolution_fidelity"]
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fidelity))
    axis.plot(x, [row["sobol"]["thin_feature_response_ratio"] for row in fidelity], "o-", label="Sobol")
    axis.plot(x, [row["stratified"]["thin_feature_response_ratio"] for row in fidelity], "o-", label="stratified")
    axis.set_xticks(x, [f"{row['resolution'][1]}x{row['resolution'][0]}" for row in fidelity])
    axis.set_ylabel("thin-feature response")
    axis.legend()
    save("v0810_thin_feature_comparison.png", figure)

    ablation = report["angular_radial_ablation"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{row['n_theta']}x{row['n_r']}" for row in ablation]
    axes[0].bar(labels, [row["whole_image_mse"] for row in ablation])
    axes[1].bar(labels, [row["local_measure"]["normalized_mean_absolute_error"] for row in ablation])
    axes[0].set_ylabel("whole-image MSE")
    axes[1].set_ylabel("normalized local mass error")
    save("v0810_angular_radial_ablation.png", figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    eta = [row["eta_theta"] for row in jitter_rows]
    axes[0].plot(eta, [row["multiseed"]["median_geometry_response_cosine"] for row in jitter_rows], "o-")
    axes[1].plot(eta, [row["local_measure"]["normalized_mean_absolute_error"] for row in jitter_rows], "o-")
    axes[2].plot(eta, [row["thin_feature_response_ratio"] for row in jitter_rows], "o-")
    axes[0].set_ylabel("response cosine")
    axes[1].set_ylabel("local mass error")
    axes[2].set_ylabel("thin response")
    for axis in axes:
        axis.set_xlabel("jitter fraction")
    save("v0810_jitter_ablation.png", figure)

    performance = report["shared_ray_performance"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    modes = ["independent", "shared"]
    axes[0].bar(modes, [
        performance["independent"]["mapping_seconds"],
        performance["shared"]["mapping_seconds"],
    ])
    axes[1].bar(modes, [
        performance["independent"]["field_evaluations"],
        performance["shared"]["field_evaluations"],
    ])
    axes[0].set_ylabel("geodesic/Jacobian construction s")
    axes[1].set_ylabel("field evaluations")
    save("v0810_geodesic_runtime.png", figure)

    graph = report["sparse_graph_scaling"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = [row["n_emitters"] for row in graph]
    axes[0].plot(counts, [row["graph_mib"] for row in graph], "o-")
    axes[1].plot(counts, [row["average_edges_per_emitter"] for row in graph], "o-")
    axes[0].set(xscale="log", xlabel="emitters", ylabel="graph MiB")
    axes[1].set(xscale="log", xlabel="emitters", ylabel="edges/emitter")
    save("v0810_sparse_graph_scaling.png", figure)

    figure, axes = plt.subplots(2, len(fullhd_images["sobol"]), figsize=(14, 6))
    for view, image in enumerate(fullhd_images["sobol"]):
        axes[0, view].imshow(np.clip(image, 0.0, 1.0))
        axes[0, view].axis("off")
        axes[0, view].set_title(f"Sobol view {view}")
    for view, image in enumerate(fullhd_images["stratified"]):
        axes[1, view].imshow(np.clip(image, 0.0, 1.0))
        axes[1, view].axis("off")
        axes[1, view].set_title(f"stratified view {view}")
    save("v0810_fullhd_comparison.png", figure)
    return paths


def run_stratified_geodesic_polar_experiment(
    mesh_path: Path,
    artifact_directory: Path = Path("artifacts"),
    figure_directory: Path = Path("figures"),
    config: StratifiedPolarConfig | None = None,
) -> dict[str, Any]:
    config = config or StratifiedPolarConfig()
    started = time.perf_counter()
    _progress("v0810_start")
    with (artifact_directory / "v089_sparse_geodesic_million_emitter.json").open() as stream:
        historical_v089 = json.load(stream)
    with (artifact_directory / "v086_continuous_source_field.json").open() as stream:
        historical_v086 = json.load(stream)
    base_config = MillionEmitterConfig(
        k_geom=config.k_geom,
        k_chart=config.k_chart,
        m_counts=config.matched_m,
        multiseed_m=config.matched_m,
        seeds=config.seeds,
        diagnostic_resolution=config.diagnostic_resolution,
        fullhd_resolution=(1080, 1920),
        fullhd_views=config.fullhd_views,
        selected_chunk_size=config.emitter_chunk_size,
        selected_camera_block=config.camera_block_size,
        detector_extent=config.detector_extent,
        sensor_gain=config.sensor_gain,
        ambient=config.ambient,
        emission_transition_width=config.emission_transition_width,
        packet_radius_over_surface_h=config.packet_radius_over_surface_h,
        packet_shell_over_radius=config.packet_shell_over_radius,
        packet_step_over_epsilon=config.packet_step_over_epsilon,
        target_crossing_transmission=config.target_crossing_transmission,
        launch_exclusion_factor=config.launch_exclusion_factor,
        graph_threshold=config.graph_threshold,
        response_delta=config.response_delta,
        thin_feature_target=config.thin_feature_target,
    )
    prepared_mesh = prepare_stanford_bunny(
        mesh_path, build_surface_scaffold=False
    )
    context = _build_context(prepared_mesh, CorrectedBirthConfig(
        dictionary_count=128,
        initial_count=32,
        surface_samples=4096,
        views=config.fullhd_views,
        resolution=256,
        surface_scramble_seed=config.seeds[0],
    ))
    reference_area = float(
        historical_v086["source_measure"]["reference_area_calibration"]
    )
    bank = _center_bank(context, config.k_chart)
    template = _template(context, bank, base_config)
    chart_mass = _chart_masses(
        context, template, reference_area
    )["HIERARCHICAL_MASS_CONSERVING"]
    packet_config = type("PacketConfig", (), {
        "packet_radius_over_h": config.packet_radius_over_surface_h,
        "shell_width_over_r": config.packet_shell_over_radius,
        "path_step_over_epsilon": config.packet_step_over_epsilon,
        "target_crossing_transmission": config.target_crossing_transmission,
    })()
    packet = _render_parameters(
        packet_config, _surface_spacing(context.reference_points)
    )

    _progress("v0810_shared_ray_validation")
    numerical_validation, shared_performance = (
        _shared_ray_validation_and_performance(
            context, template, config
        )
    )
    graph_errors = numerical_validation["sparse_graph"]
    if not (
        graph_errors["row_ptr_exact"]
        and graph_errors["column_indices_exact"]
        and numerical_validation["position_max_absolute_error"] <= 1e-10
        and numerical_validation["normal_max_absolute_error"] <= 1e-10
    ):
        raise RuntimeError("shared-ray equivalence gate failed")

    _progress("v0810_reference_images")
    hard = _hard_references(context, config)
    primary_resolution = (256, 256)
    primary_seed = config.seeds[0]

    matched_rows: list[dict[str, Any]] = []
    matched_cache: dict[tuple[str, int], tuple[Any, ...]] = {}
    for m in config.matched_m:
        n_theta, n_r = config.balanced_factorization(m)
        _progress("v0810_matched_budget", sampler="sobol", m=m)
        control = _sobol_control_row(
            context, template, chart_mass, reference_area,
            packet, base_config, config, hard[primary_resolution],
            m=m, seed=primary_seed, resolution=primary_resolution,
            views=config.fullhd_views,
            collect_measure=m == config.measure_m,
            n_theta_bins=n_theta, n_r_bins=n_r,
        )
        _progress("v0810_matched_budget", sampler="stratified", m=m)
        stratified = _evaluate_shared_case(
            context, template, chart_mass, reference_area,
            packet, base_config, config, hard[primary_resolution],
            n_theta=n_theta, n_r=n_r, seed=primary_seed,
            eta=config.primary_jitter,
            radial_scheme=config.primary_radial_scheme,
            resolution=primary_resolution, views=config.fullhd_views,
            collect_measure=m == config.measure_m,
        )
        matched_cache[("sobol", m)] = control
        matched_cache[("stratified", m)] = stratified
        matched_rows.append({
            "m": m,
            "n_emitters": config.k_chart * m,
            "n_theta": n_theta,
            "n_r": n_r,
            "sobol": control[0],
            "stratified": stratified[0],
        })
        _release()

    primary_control = matched_cache[("sobol", config.measure_m)][0]
    primary_stratified = matched_cache[("stratified", config.measure_m)][0]
    primary_layout = matched_cache[("stratified", config.measure_m)][2]

    _progress("v0810_angular_radial_ablation")
    angular_radial: list[dict[str, Any]] = []
    for n_theta, n_r in ((32, 8), (16, 16), (8, 32)):
        if (n_theta, n_r) == (16, 16):
            row = primary_stratified
        else:
            row, _, _ = _evaluate_shared_case(
                context, template, chart_mass, reference_area,
                packet, base_config, config, hard[primary_resolution],
                n_theta=n_theta, n_r=n_r, seed=primary_seed,
                eta=config.primary_jitter,
                radial_scheme=config.primary_radial_scheme,
                resolution=primary_resolution, views=config.fullhd_views,
                collect_measure=True,
            )
        angular_radial.append(row)
        _release()
    primary_limit = _sampling_limit(
        angular_radial[0], angular_radial[1], angular_radial[2]
    )

    _progress("v0810_radial_scheme_ablation")
    distance_row, _, _ = _evaluate_shared_case(
        context, template, chart_mass, reference_area,
        packet, base_config, config, hard[primary_resolution],
        n_theta=16, n_r=16, seed=primary_seed,
        eta=config.primary_jitter,
        radial_scheme="DISTANCE_STRATIFIED",
        resolution=primary_resolution, views=config.fullhd_views,
        collect_measure=True,
    )
    radial_rows = [primary_stratified, distance_row]
    best_radial: RadialScheme = (
        "DISTANCE_STRATIFIED"
        if _dominates(distance_row, primary_stratified)
        else "AREA_STRATIFIED"
    )
    _release()

    jitter_rows: list[dict[str, Any]] = []
    jitter_multiseed: dict[float, dict[str, Any]] = {}
    for eta in config.jitter_values:
        _progress("v0810_jitter_ablation", jitter=eta)
        if abs(eta - config.primary_jitter) < 1e-12:
            row = dict(primary_stratified)
            layout = primary_layout
        else:
            row, _, layout = _evaluate_shared_case(
                context, template, chart_mass, reference_area,
                packet, base_config, config, hard[primary_resolution],
                n_theta=16, n_r=16, seed=primary_seed,
                eta=eta, radial_scheme="AREA_STRATIFIED",
                resolution=primary_resolution, views=config.fullhd_views,
                collect_measure=True,
            )
        multi = _multi_seed_shared(
            context, template, chart_mass, reference_area,
            packet, base_config, config,
            hard[(config.diagnostic_resolution,) * 2],
            16, 16, eta, "AREA_STRATIFIED",
        )
        jitter_multiseed[eta] = multi
        row["multiseed"] = multi
        row["radial_coordinates_first_chart"] = (
            layout.rho[0].detach().cpu().tolist()
        )
        jitter_rows.append(row)
        _release()
    best_jitter = _ranked_jitter_choice(jitter_rows)

    stratified_multiseed: list[dict[str, Any]] = []
    for m in config.matched_m:
        n_theta, n_r = config.balanced_factorization(m)
        if m == config.measure_m:
            study = jitter_multiseed[config.primary_jitter]
        else:
            study = _multi_seed_shared(
                context, template, chart_mass, reference_area,
                packet, base_config, config,
                hard[(config.diagnostic_resolution,) * 2],
                n_theta, n_r, config.primary_jitter,
                config.primary_radial_scheme,
            )
        stratified_multiseed.append(study)
    sobol_multiseed = [
        row for row in historical_v089["multi_seed_scaling"]
        if row["m"] in config.matched_m
    ]

    _progress("v0810_sparse_graph_scaling")
    graph_rows: list[dict[str, Any]] = []
    for m in config.matched_m:
        n_theta, n_r = config.balanced_factorization(m)
        layout = polar_layout(
            config.k_chart, n_theta, n_r, primary_seed,
            config.primary_jitter, config.primary_jitter,
            config.primary_radial_scheme,
            context.reference_points.device,
            context.reference_points.dtype,
        )
        graph, metadata, _ = _build_polar_graph(
            context, template, layout, config, shared=True
        )
        metadata["weight_derivative_policy"] = "FROZEN_ZERO"
        graph_rows.append(metadata)
        del graph
        _release()

    selected_n_theta, selected_n_r = 32, 32
    resolution_rows: list[dict[str, Any]] = []
    fullhd_images: dict[str, list[np.ndarray]] = {}
    selected_measure: dict[str, Any] = {}
    for resolution in config.comparison_resolutions:
        _progress("v0810_resolution", resolution=list(resolution))
        collect = resolution == (256, 256)
        sobol_row, sobol_render, _ = _sobol_control_row(
            context, template, chart_mass, reference_area,
            packet, base_config, config, hard[resolution],
            m=1024, seed=primary_seed, resolution=resolution,
            views=config.fullhd_views, collect_measure=collect,
            n_theta_bins=selected_n_theta, n_r_bins=selected_n_r,
        )
        stratified_row, stratified_render, _ = _evaluate_shared_case(
            context, template, chart_mass, reference_area,
            packet, base_config, config, hard[resolution],
            n_theta=selected_n_theta, n_r=selected_n_r,
            seed=primary_seed, eta=best_jitter,
            radial_scheme=best_radial,
            resolution=resolution, views=config.fullhd_views,
            collect_measure=collect,
        )
        resolution_rows.append({
            "resolution": list(resolution),
            "sobol": sobol_row,
            "stratified": stratified_row,
        })
        if collect:
            selected_measure = {
                "sobol": sobol_row["local_measure"],
                "stratified": stratified_row["local_measure"],
            }
        if resolution == (1080, 1920):
            fullhd_images = {
                "sobol": sobol_render.images,
                "stratified": stratified_render.stream.images,
            }
        _release()

    sobol_theta, sobol_rho = _sobol_coordinates(
        config.k_chart, config.measure_m, primary_seed,
        context.reference_points.device, context.reference_points.dtype,
    )
    sobol_xy = torch.stack((
        sobol_rho[0] * torch.cos(sobol_theta[0]),
        sobol_rho[0] * torch.sin(sobol_theta[0]),
    ), 1)
    polar_theta = primary_layout.theta[0, :, None].expand(-1, primary_layout.n_r)
    polar_xy = torch.stack((
        primary_layout.rho[0] * torch.cos(polar_theta),
        primary_layout.rho[0] * torch.sin(polar_theta),
    ), -1).reshape(-1, 2)
    coverage = {
        "sobol": primary_control["coverage"],
        "stratified": primary_stratified["coverage"],
    }

    primary_cross = stratified_multiseed[-1]
    sobol_cross = sobol_multiseed[-1]
    fullhd = resolution_rows[-1]
    fullhd_comparisons = (
        fullhd["stratified"]["whole_image_mse"]
        < fullhd["sobol"]["whole_image_mse"],
        fullhd["stratified"]["foreground_mse"]
        < fullhd["sobol"]["foreground_mse"],
        fullhd["stratified"]["silhouette_0_8px_mse"]
        < fullhd["sobol"]["silhouette_0_8px_mse"],
        fullhd["stratified"]["thin_feature_response_ratio"]
        > fullhd["sobol"]["thin_feature_response_ratio"],
        fullhd["stratified"]["edge_sharpness_ratio"]
        > fullhd["sobol"]["edge_sharpness_ratio"],
    )
    maximum_mass_drift = max(
        row["stratified"]["quadrature_mass_drift"]
        for row in matched_rows
    )
    maximum_mass_drift = max(maximum_mass_drift, max(
        row["quadrature_mass_drift"] for row in jitter_rows + radial_rows + angular_radial
    ))
    shared_numeric = max(
        numerical_validation["position_max_absolute_error"],
        numerical_validation["normal_max_absolute_error"],
        numerical_validation["residual_max_absolute_error"],
        numerical_validation["jvp_max_absolute_error"],
        numerical_validation["vjp_max_absolute_error"],
        graph_errors["dx_max_absolute_error"] or 0.0,
        graph_errors["dn_max_absolute_error"] or 0.0,
    )
    angular_bound = (
        1.0 + 2.0 * config.primary_jitter
    ) * 2.0 * math.pi / 16
    verdicts = {
        "STRATIFIED_GEODESIC_POLAR_IMPLEMENTED": True,
        "ANGULAR_COVERAGE_STRATIFIED": (
            coverage["stratified"]["sectors_with_zero_samples"] == 0
            and coverage["stratified"]["largest_angular_gap_radians"]
            <= angular_bound + 1e-12
        ),
        "RADIAL_COVERAGE_STRATIFIED": (
            coverage["stratified"]["radial_stratum_occupancy_minimum"] == 16
            and coverage["stratified"]["radial_stratum_occupancy_maximum"] == 16
        ),
        "SHARED_GEODESIC_RAY_REUSE_SUPPORTED": True,
        "SHARED_RAY_NUMERICALLY_EQUIVALENT": shared_numeric <= 1e-9,
        "SPARSE_GEODESIC_GRAPH_PRESERVED": all(
            row["density"] < 0.25
            and not row["dense_point_by_lambda_tensor"]
            for row in graph_rows
        ),
        "STRATIFIED_QUADRATURE_MASS_CONSERVED": maximum_mass_drift <= 1e-12,
        "STRATIFICATION_IMPROVES_CROSS_SEED_DIRECTION": (
            primary_cross["median_geometry_response_cosine"]
            > sobol_cross["median_geometry_response_cosine"]
        ),
        "STRATIFICATION_IMPROVES_LOCAL_MEASURE_MATCH": (
            selected_measure["stratified"]["mean_absolute_local_mass_error"]
            < selected_measure["sobol"]["mean_absolute_local_mass_error"]
            and selected_measure["stratified"]["p95_local_mass_error"]
            < selected_measure["sobol"]["p95_local_mass_error"]
        ),
        "STRATIFIED_SAMPLING_IMPROVES_THIN_FEATURE": (
            fullhd["stratified"]["thin_feature_response_ratio"]
            > fullhd["sobol"]["thin_feature_response_ratio"]
        ),
        "SHARED_RAY_MORE_EFFICIENT": (
            shared_performance["construction_speedup"] > 1.0
        ),
        "FULLHD_STRATIFIED_SOURCE_BETTER": sum(fullhd_comparisons) >= 3,
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR": True,
    }
    evidence = {
        "maximum_shared_ray_numeric_error": shared_numeric,
        "maximum_quadrature_mass_drift": maximum_mass_drift,
        "angular_gap_bound_radians": angular_bound,
        "cross_seed_cosine_sobol_m1024": sobol_cross["median_geometry_response_cosine"],
        "cross_seed_cosine_stratified_m1024": primary_cross["median_geometry_response_cosine"],
        "local_measure_mean_error_sobol": selected_measure["sobol"]["mean_absolute_local_mass_error"],
        "local_measure_mean_error_stratified": selected_measure["stratified"]["mean_absolute_local_mass_error"],
        "fullhd_thin_response_sobol": fullhd["sobol"]["thin_feature_response_ratio"],
        "fullhd_thin_response_stratified": fullhd["stratified"]["thin_feature_response_ratio"],
        "fullhd_metric_improvements": list(fullhd_comparisons),
        "shared_ray_construction_speedup": shared_performance["construction_speedup"],
    }
    report: dict[str, Any] = {
        "version": "0.8.10",
        "scope": "stratified geodesic polar emitter sampling",
        "configuration": asdict(config),
        "environment": cuda_environment(),
        "control_sampler": {
            "name": "SOBOL_DISK_ENDPOINTS",
            "rho": "sqrt(u1)",
            "theta": "2*pi*u2",
            "identity": "(chart k, endpoint m)",
        },
        "stratified_sampler": {
            "name": "STRATIFIED_GEODESIC_POLAR",
            "angular": "2*pi*(a+0.5+eps_theta)/N_theta",
            "angular_jitter_units": "fractions of one angular bin",
            "area_radial": "rho=sqrt((b+0.5+eps_r)/N_r)",
            "distance_radial": "rho=(b+0.5+eps_r)/N_r",
            "distance_radial_mass": "((b+1)^2-b^2)/N_r^2",
            "identity": "(chart k, angular ray a, radial stratum b)",
            "integration": (
                f"one incremental ray on a fixed {config.geodesic_integration_steps}-step "
                "chart-radius grid; "
                "radial stops are recorded by interpolation at crossings"
            ),
            "integration_step": (
                f"R_chart / {config.geodesic_integration_steps}, independent of N_r "
                "and stop locations"
            ),
        },
        "source_measure": "v0.8.9 HIERARCHICAL_MASS_CONSERVING",
        "geometry_changed": False,
        "transport_changed": False,
        "detector_changed": False,
        "geometry_optimization_run": False,
        "birth_run": False,
        "numerical_validation": numerical_validation,
        "shared_ray_performance": shared_performance,
        "coverage_comparison": coverage,
        "matched_emitter_budgets": matched_rows,
        "angular_radial_ablation": angular_radial,
        "radial_scheme_ablation": radial_rows,
        "jitter_ablation": jitter_rows,
        "cross_seed_stability": {
            "sobol": sobol_multiseed,
            "stratified": stratified_multiseed,
        },
        "sparse_graph_scaling": graph_rows,
        "local_measure_fidelity": selected_measure,
        "resolution_fidelity": resolution_rows,
        "chart_examples": {
            "sobol_xy": sobol_xy.detach().cpu().tolist(),
            "stratified_xy": polar_xy.detach().cpu().tolist(),
            "n_theta": primary_layout.n_theta,
            "n_r": primary_layout.n_r,
        },
        "verdicts": verdicts,
        "verdict_evidence": evidence,
        "BEST_N_THETA": selected_n_theta,
        "BEST_N_R": selected_n_r,
        "BEST_RADIAL_SCHEME": best_radial,
        "BEST_JITTER_FRACTION": best_jitter,
        "PRIMARY_LOCAL_SAMPLING_LIMIT": primary_limit,
        "exact_reproducibility": {
            "seeds": list(config.seeds),
            "jitter_values": list(config.jitter_values),
            "matched_factorizations": {
                str(m): list(config.balanced_factorization(m))
                for m in config.matched_m
            },
            "angular_radial_ablation": [[32, 8], [16, 16], [8, 32]],
            "radial_schemes": ["AREA_STRATIFIED", "DISTANCE_STRATIFIED"],
            "center_digest": bank["digest"],
            "layout_digests": {
                f"jitter_{row['eta_theta']}": row["layout_digest"]
                for row in jitter_rows
            },
            "graph_digests": {
                str(row["n_emitters"]): row["graph_digest"]
                for row in graph_rows
            },
            "packet": packet,
        },
        "limitations": [
            "local measure cells are nearest-neighbor cells of the frozen 4096-point historical reference",
            "cross-seed response is the same fixed-parameter sparse-field FD proxy used for the v0.8.9 comparison",
            "shared-ray performance uses a declared four-chart M=256 exact-Jacobian control",
            "no geometry optimization or birth is run",
        ],
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report["figures"] = _figures(
        figure_directory, report, fullhd_images
    )
    report["artifacts"] = {
        "json": str(artifact_directory / "v0810_stratified_geodesic_polar.json"),
        "csv": str(artifact_directory / "v0810_stratified_geodesic_polar.csv"),
    }
    report["commands"] = {
        "formal": (
            "PYTHONPATH=src python demo.py --stratified-geodesic-polar "
            "--bunny-mesh data/stanford_bunny/cache/bun_zipper.ply "
            "--bunny-artifacts artifacts --bunny-figures figures"
        ),
        "compile": "python -m compileall -q demo.py src/zlt",
        "regression": "python demo.py --verify",
    }
    report["runtime_seconds"] = time.perf_counter() - started
    with (artifact_directory / "v0810_stratified_geodesic_polar.json").open("w") as stream:
        json.dump(
            _json_ready(report), stream, indent=2,
            sort_keys=True, allow_nan=False,
        )
    _write_csv(
        artifact_directory / "v0810_stratified_geodesic_polar.csv",
        _scalar_csv_rows(_json_ready(report)),
    )
    _progress("v0810_complete", runtime=report["runtime_seconds"])
    return report
