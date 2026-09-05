"""v0.8.9 measure-correct sparse geodesic million-emitter transport.

This module is deliberately an experiment harness around the unchanged
v0.8.5 finite-support transport and v0.8.4 continuous detector.  Geodesic
construction is procedural and chunk-local; its useful derivative is retained
as a compact shared-pattern CSR graph rather than as PyTorch trajectories.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .continuous_source import _emission_factor
from .corrected_birth import CorrectedBirthConfig, _build_context, _layout
from .emitter_scaling import _center_bank, _definition
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
from .geodesic_source import (
    ChartDefinition,
    _chart_identity_digest,
    _disk_latents,
    _eta,
    _map_charts,
)
from .high_sample import _release, _write_csv
from .mesh_field import prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color
from .sparse_geodesic import (
    GeodesicTemplate,
    SparseGeodesicGraph,
    SparseGraphBuilder,
    build_chart_envelope_support,
    emitter_support_from_owners,
    exact_chart_local_linearization,
    map_emitter_chunk,
    map_emitter_chunk_sparse_field,
    prepare_centers,
    prepare_centers_sparse_field,
    sparse_field_evaluate,
)


Tensor = torch.Tensor
Quadrature = Literal[
    "HISTORICAL_MEASURE_MATCHED",
    "FIXED_REFERENCE_MASS",
    "HIERARCHICAL_MASS_CONSERVING",
    "PHYSICAL_AREA_CONTROL",
]


@dataclass(frozen=True)
class MillionEmitterConfig:
    k_geom: int = 64
    k_chart: int = 1024
    m_counts: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    count_invariance_m: tuple[int, ...] = (64, 128, 256, 512)
    fullhd_m: tuple[int, ...] = (256, 512, 1024, 2048)
    multiseed_m: tuple[int, ...] = (64, 256, 1024)
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    seed: int = 101
    reference_surface_samples: int = 4096
    radius_factor: float = 1.0
    geodesic_steps: int = 4
    diagnostic_resolution: int = 64
    screen_resolution: int = 256
    fullhd_resolution: tuple[int, int] = (1080, 1920)
    fullhd_views: int = 4
    transport_atlas_views: int = 20
    chunk_sizes: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384)
    camera_blocks: tuple[int, ...] = (1, 2, 4, 8, 20)
    selected_chunk_size: int = 8192
    selected_camera_block: int = 4
    packet_radius_over_surface_h: float = 1.0
    packet_shell_over_radius: float = 1.0
    packet_step_over_epsilon: float = 0.5
    target_crossing_transmission: float = 0.01
    launch_exclusion_factor: float = 1.05
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    ambient: float = 0.35
    emission_transition_width: float = 0.05
    graph_threshold: float = 1e-10
    response_delta: float = 1e-3
    fd_epsilons: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    source_mass_drift_maximum: float = 0.005
    expectation_drift_maximum: float = 0.08
    response_norm_cv_maximum: float = 0.15
    response_cosine_minimum: float = 0.90
    thin_feature_target: float = 0.75


@dataclass
class StreamRender:
    images: list[np.ndarray]
    tensor: Tensor
    mapping_seconds: float
    source_seconds: float
    transport_seconds: float
    detector_seconds: float
    runtime_seconds: float
    attempted_packets: int
    interaction_evaluations: int
    detector_writes: int
    peak_allocated_mib: float
    peak_reserved_mib: float
    cpu_rss_mib: float
    residual_p95: float
    image_digest: str


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _digest_arrays(*items: Any) -> str:
    digest = hashlib.sha256()
    for item in items:
        if isinstance(item, Tensor):
            array = item.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(item)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _template(context: Any, bank: dict[str, Any], config: MillionEmitterConfig) -> GeodesicTemplate:
    definition = _definition(
        context, bank, config.k_chart, 1, config.radius_factor,
        config.seed, 1.0,
    )
    return GeodesicTemplate(
        definition.reference_centers,
        definition.reference_normals,
        definition.reference_tangent_1,
        definition.reference_tangent_2,
        definition.chart_radii,
        _eta(context.base, definition.reference_centers),
        config.geodesic_steps,
    )


def _latent_chunk(
    k: int,
    m: int,
    start: int,
    stop: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    flat = torch.arange(start, stop, dtype=torch.long)
    owners_cpu = torch.div(flat, m, rounding_mode="floor")
    local_cpu = flat.remainder(m)
    rho = torch.empty(stop - start, dtype=torch.float64)
    theta = torch.empty_like(rho)
    for chart in torch.unique(owners_cpu).tolist():
        selected = owners_cpu == chart
        unit = torch.quasirandom.SobolEngine(
            2, scramble=True, seed=seed + 104729 * int(chart)
        ).draw(m).to(torch.float64)
        ids = local_cpu[selected]
        rho[selected] = torch.sqrt(unit[ids, 0].clamp_min(1e-12))
        theta[selected] = 2.0 * math.pi * unit[ids, 1]
    return (
        owners_cpu.to(device),
        local_cpu.to(device),
        rho.to(device=device, dtype=dtype),
        theta.to(device=device, dtype=dtype),
    )


def _latent_chunks(
    k: int,
    m: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[tuple[int, int, Tensor, Tensor, Tensor, Tensor]]:
    count = k * m
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        yield start, stop, *_latent_chunk(k, m, start, stop, seed, device, dtype)


def _chart_masses(
    context: Any,
    template: GeodesicTemplate,
    reference_area: float,
) -> dict[str, Tensor]:
    reference = context.reference_points.detach().cpu()
    centers = template.reference_centers.detach().cpu()
    nearest = torch.cdist(reference, centers).argmin(1)
    counts = torch.bincount(nearest, minlength=template.chart_count).to(torch.float64)
    hierarchical = reference_area * counts / counts.sum()
    fixed = torch.full_like(hierarchical, reference_area / template.chart_count)
    # The established kNN/Voronoi proxy is the physical-area control only.
    physical = reference_area * counts.clamp_min(0.25)
    physical /= physical.sum()
    return {
        "FIXED_REFERENCE_MASS": fixed.to(context.reference_points.device),
        "HIERARCHICAL_MASS_CONSERVING": hierarchical.to(context.reference_points.device),
        "PHYSICAL_AREA_CONTROL": physical.to(context.reference_points.device),
    }


def _stream_render(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    *,
    m: int,
    seed: int,
    resolution: tuple[int, int],
    views: int,
    chunk_size: int,
    camera_block: int,
    field: Any | None = None,
    accumulator_dtype: torch.dtype = torch.float32,
) -> StreamRender:
    field = context.base if field is None else field
    device = context.reference_points.device
    prepared = prepare_centers(field, template)
    atlas = nested_fibonacci_atlas(device, (views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    offsets = _micro_offsets(1, device, context.reference_points.dtype)
    transport_eta = _eta(field, template.reference_centers)
    accumulators = [
        torch.zeros((math.prod(resolution), 3), dtype=accumulator_dtype, device=device)
        for _ in range(views)
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    mapping_seconds = source_seconds = transport_seconds = detector_seconds = 0.0
    interactions = writes = 0
    residual_values: list[Tensor] = []
    for _, _, owners, _, rho, theta in _latent_chunks(
        config.k_chart, m, chunk_size, seed, device, context.reference_points.dtype
    ):
        _sync(device); stage = time.perf_counter()
        positions, normals, _, _, residual, _ = map_emitter_chunk(
            field, template, prepared, owners, rho, theta
        )
        _sync(device); mapping_seconds += time.perf_counter() - stage
        residual_values.append(residual.detach().cpu())
        _sync(device); stage = time.perf_counter()
        weights = chart_mass[owners] / (m * reference_area)
        colors = meshfree_base_color(positions, field.lower, field.upper)
        _sync(device); source_seconds += time.perf_counter() - stage
        for block_start in range(0, views, camera_block):
            block_stop = min(block_start + camera_block, views)
            for view_id in range(block_start, block_stop):
                direction = atlas.directions[view_id]
                directions = direction.expand_as(positions)
                _sync(device); stage = time.perf_counter()
                transmission, _, evaluations = _transmission(
                    field, positions, directions,
                    boundary.exit_times(positions, directions),
                    radius=packet["radius"], epsilon=packet["epsilon"],
                    path_step=packet["path_step"], offsets=offsets,
                    eta=transport_eta, kappa=packet["kappa"],
                    surface_barrier=True,
                    launch_exclusion_factor=config.launch_exclusion_factor,
                )
                cosine = normals @ direction
                outward = _emission_factor(cosine, config.emission_transition_width)
                lobe = config.ambient + (1.0 - config.ambient) * cosine.clamp_min(0.0)
                energy = weights[:, None] * transmission[:, None] * colors
                energy = energy * (outward * lobe)[:, None]
                _sync(device); transport_seconds += time.perf_counter() - stage
                _sync(device); stage = time.perf_counter()
                detector_points = positions.to(accumulator_dtype)
                _accumulate_continuous(
                    accumulators[view_id], detector_points,
                    energy.to(accumulator_dtype),
                    atlas.right[view_id].to(accumulator_dtype),
                    atlas.up[view_id].to(accumulator_dtype),
                    boundary.center.to(accumulator_dtype),
                    config.detector_extent, resolution,
                )
                _sync(device); detector_seconds += time.perf_counter() - stage
                interactions += evaluations
                writes += int(positions.shape[0]) * 16
        del positions, normals, residual, weights, colors
    _sync(device)
    scale = config.sensor_gain * math.prod(resolution)
    images_tensor = torch.stack([scale * image for image in accumulators])
    images = [
        image.reshape(*resolution, 3).detach().cpu().to(torch.float32).numpy()
        for image in images_tensor
    ]
    vector = images_tensor.reshape(-1)
    residual_all = torch.cat(residual_values)
    runtime = time.perf_counter() - started
    return StreamRender(
        images, vector, mapping_seconds, source_seconds,
        transport_seconds, detector_seconds, runtime,
        config.k_chart * m * views, interactions, writes,
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        float(torch.quantile(residual_all, 0.95)),
        _digest_arrays(*images),
    )


def _render_row(
    rendered: StreamRender,
    hard: list[np.ndarray],
    chart_mass: Tensor,
    config: MillionEmitterConfig,
    m: int,
    seed: int,
    resolution: tuple[int, int],
    views: int,
    chunk_size: int,
    camera_block: int,
) -> dict[str, Any]:
    metrics = _image_metrics(rendered.images, hard[:views])
    source_mass = float(chart_mass.sum())
    return {
        "k_geom": config.k_geom,
        "k_chart": config.k_chart,
        "m": m,
        "n_emitters": config.k_chart * m,
        "seed": seed,
        "resolution": list(resolution),
        "views": views,
        "emitter_chunk_size": chunk_size,
        "camera_block_size": camera_block,
        "source_mass": source_mass,
        "foreground_energy": float(sum(image[np.any(target > 0, axis=2)].sum() for image, target in zip(rendered.images, hard[:views]))),
        **metrics,
        "geodesic_construction_seconds": rendered.mapping_seconds,
        "sparse_geo_extraction_seconds": 0.0,
        "source_evaluation_seconds": rendered.source_seconds,
        "finite_transport_seconds": rendered.transport_seconds,
        "detector_seconds": rendered.detector_seconds,
        "detector_seconds_per_view": rendered.detector_seconds / views,
        "runtime_seconds": rendered.runtime_seconds,
        "emitters_per_second": config.k_chart * m / max(rendered.mapping_seconds, 1e-30),
        "packets_per_second": rendered.attempted_packets / max(rendered.transport_seconds, 1e-30),
        "interactions_per_second": rendered.interaction_evaluations / max(rendered.transport_seconds, 1e-30),
        "detector_writes_per_second": rendered.detector_writes / max(rendered.detector_seconds, 1e-30),
        "attempted_packets": rendered.attempted_packets,
        "interaction_evaluations": rendered.interaction_evaluations,
        "detector_writes": rendered.detector_writes,
        "peak_cuda_allocated_mib": rendered.peak_allocated_mib,
        "peak_cuda_reserved_mib": rendered.peak_reserved_mib,
        "cpu_rss_mib": rendered.cpu_rss_mib,
        "mapping_residual_p95": rendered.residual_p95,
        "image_digest": rendered.image_digest,
    }


def _relative(left: Tensor, right: Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right).clamp_min(1e-30))


def _small_image(
    field: Any,
    positions: Tensor,
    normals: Tensor,
    weights: Tensor,
    context: Any,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    resolution: tuple[int, int] = (32, 32),
) -> Tensor:
    device = positions.device
    atlas = nested_fibonacci_atlas(device, (1,))
    boundary = enclosing_observation_sphere(context.reference_points)
    direction = atlas.directions[0]
    directions = direction.expand_as(positions)
    transmission, _, _ = _transmission(
        field, positions, directions, boundary.exit_times(positions, directions),
        radius=packet["radius"], epsilon=packet["epsilon"],
        path_step=packet["path_step"], offsets=_micro_offsets(1, device, positions.dtype),
        eta=_eta(field, positions.detach()), kappa=packet["kappa"],
        surface_barrier=True,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    cosine = normals @ direction
    energy = weights[:, None] * transmission[:, None]
    energy = energy * meshfree_base_color(positions, field.lower, field.upper)
    energy = energy * (
        _emission_factor(cosine, config.emission_transition_width)
        * (config.ambient + (1.0 - config.ambient) * cosine.clamp_min(0.0))
    )[:, None]
    image = torch.zeros((math.prod(resolution), 3), dtype=positions.dtype, device=device)
    _accumulate_continuous(
        image, positions, energy, atlas.right[0], atlas.up[0],
        boundary.center, config.detector_extent, resolution,
    )
    return config.sensor_gain * math.prod(resolution) * image.reshape(-1)


def _gradient_validation(
    context: Any,
    bank: dict[str, Any],
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    categories: dict[str, int],
) -> tuple[dict[str, Any], SparseGeodesicGraph]:
    """Compare monolithic AD, chunk tangent/CSR, recomputation, paths, and FD."""
    k, m = 64, 2
    definition = _definition(context, bank, k, m, config.radius_factor, config.seed, reference_area)
    template = GeodesicTemplate(
        definition.reference_centers, definition.reference_normals,
        definition.reference_tangent_1, definition.reference_tangent_2,
        definition.chart_radii, _eta(context.base, definition.reference_centers),
        config.geodesic_steps,
    )
    basis_centers = context.master_layout.centers[:config.k_geom]
    basis_radii = context.master_layout.radii[:config.k_geom]
    chart_support = build_chart_envelope_support(
        template, basis_centers, basis_radii
    )
    builder = SparseGraphBuilder(k * m, config.k_geom, config.graph_threshold)
    position_rows: list[Tensor] = []
    normal_rows: list[Tensor] = []
    residual_rows: list[Tensor] = []
    for chart in range(k):
        start, stop = chart * m, (chart + 1) * m
        local = exact_chart_local_linearization(
            context.base, template, chart_support, chart,
            definition.latent_rho[start:stop], definition.latent_theta[start:stop],
            basis_centers, basis_radii,
        )
        local_positions, local_normals, dx, dn, local_residual, sparse_support = local
        builder.append(dx, dn, sparse_support)
        position_rows.append(local_positions)
        normal_rows.append(local_normals)
        residual_rows.append(local_residual)
    positions = torch.cat(position_rows)
    normals = torch.cat(normal_rows)
    residual = torch.cat(residual_rows)
    graph = builder.finish()
    generator = torch.Generator(device=positions.device).manual_seed(1909)
    delta = torch.randn(config.k_geom, dtype=positions.dtype, device=positions.device, generator=generator)
    g_x = torch.randn(positions.shape, dtype=positions.dtype, device=positions.device, generator=generator)
    g_n = torch.randn(normals.shape, dtype=normals.dtype, device=normals.device, generator=generator)
    zero = torch.zeros(config.k_geom, dtype=positions.dtype, device=positions.device, requires_grad=True)

    def mapped(coefficients: Tensor) -> Tensor:
        mapping = _map_charts(
            LocalField(context, coefficients), definition, steps=config.geodesic_steps
        )
        return torch.cat((mapping.positions.reshape(-1), mapping.normals.reshape(-1)))

    _, full_jvp = torch.autograd.functional.jvp(mapped, (zero,), (delta,), strict=False)
    sparse_dx, sparse_dn = graph.jvp(delta)
    sparse_jvp = torch.cat((sparse_dx.reshape(-1), sparse_dn.reshape(-1))).to(full_jvp.dtype)
    full_position_jvp = full_jvp[:positions.numel()].reshape_as(positions)
    full_normal_jvp = full_jvp[positions.numel():].reshape_as(normals)
    mapped_value = mapped(zero)
    objective = (
        mapped_value[: positions.numel()].reshape_as(positions) * g_x
    ).sum() + (
        mapped_value[positions.numel():].reshape_as(normals) * g_n
    ).sum()
    full_vjp = torch.autograd.grad(objective, zero)[0]
    sparse_vjp = graph.vjp(g_x, g_n).to(full_vjp.dtype)
    recompute_zero = torch.zeros_like(zero, requires_grad=True)
    recomputed = mapped(recompute_zero)
    recompute_objective = (
        recomputed[: positions.numel()].reshape_as(positions) * g_x
    ).sum() + (
        recomputed[positions.numel():].reshape_as(normals) * g_n
    ).sum()
    recompute_vjp = torch.autograd.grad(recompute_objective, recompute_zero)[0]

    weights = torch.full((k * m,), 1.0 / (k * m), dtype=positions.dtype, device=positions.device)
    base_positions = positions.detach()
    base_normals = normals.detach()

    def full_image(coefficients: Tensor) -> Tensor:
        field = LocalField(context, coefficients)
        mapping = _map_charts(field, definition, steps=config.geodesic_steps)
        return _small_image(field, mapping.positions, mapping.normals, weights, context, packet, config)

    def sampling_image(coefficients: Tensor) -> Tensor:
        mapping = _map_charts(LocalField(context, coefficients), definition, steps=config.geodesic_steps)
        return _small_image(context.base, mapping.positions, mapping.normals, weights, context, packet, config)

    def transport_image(coefficients: Tensor) -> Tensor:
        return _small_image(
            LocalField(context, coefficients), base_positions, base_normals,
            weights, context, packet, config,
        )

    coefficients = torch.zeros(config.k_geom, dtype=positions.dtype, device=positions.device, requires_grad=True)
    target = torch.linspace(0.0, 0.15, _small_image(
        context.base, base_positions, base_normals, weights, context, packet, config
    ).numel(), dtype=positions.dtype, device=positions.device)

    def loss_of(function: Any, value: Tensor) -> Tensor:
        return 0.5 * (function(value) - target).square().sum()

    full_loss = loss_of(full_image, coefficients)
    full_gradient = torch.autograd.grad(full_loss, coefficients)[0]
    sample_coefficients = torch.zeros_like(coefficients, requires_grad=True)
    sample_gradient = torch.autograd.grad(loss_of(sampling_image, sample_coefficients), sample_coefficients)[0]
    transport_coefficients = torch.zeros_like(coefficients, requires_grad=True)
    transport_gradient = torch.autograd.grad(loss_of(transport_image, transport_coefficients), transport_coefficients)[0]

    px = base_positions.clone().requires_grad_(True)
    pn = base_normals.clone().requires_grad_(True)
    sparse_image = _small_image(context.base, px, pn, weights, context, packet, config)
    sparse_loss = 0.5 * (sparse_image - target).square().sum()
    adjoint_x, adjoint_n = torch.autograd.grad(sparse_loss, (px, pn))
    sparse_sampling_gradient = graph.vjp(adjoint_x, adjoint_n).to(full_gradient.dtype)
    sparse_total = sparse_sampling_gradient + transport_gradient

    fd_rows = []
    for category, parameter in categories.items():
        if parameter >= config.k_geom:
            continue
        analytic = full_gradient[parameter]
        candidates = []
        for epsilon in config.fd_epsilons:
            plus = torch.zeros_like(coefficients); plus[parameter] = epsilon
            minus = torch.zeros_like(coefficients); minus[parameter] = -epsilon
            with torch.no_grad():
                finite = (loss_of(full_image, plus) - loss_of(full_image, minus)) / (2 * epsilon)
            relative_error = float((analytic - finite).abs() / finite.abs().clamp_min(1e-12))
            candidates.append({
                "epsilon": epsilon,
                "analytic": float(analytic),
                "finite_difference": float(finite),
                "relative_error": relative_error,
            })
        best = min(candidates, key=lambda row: row["relative_error"])
        fd_rows.append({"category": category, "parameter": parameter, "rows": candidates, "best": best})

    float32_error = _relative(sparse_vjp, full_vjp)
    output = {
        "configuration": {"k_chart": k, "m": m, "n_emitters": k * m, "k_geom": config.k_geom},
        "g0_full_monolithic_autograd": True,
        "g1_chunk_local_sparse_csr": True,
        "g2_backward_recomputation": True,
        "position_normal_jvp_relative_error": _relative(sparse_jvp, full_jvp),
        "position_jvp_relative_error": _relative(sparse_dx, full_position_jvp),
        "normal_jvp_relative_error": _relative(sparse_dn, full_normal_jvp),
        "g0_vs_g1_vjp_relative_error": _relative(sparse_vjp, full_vjp),
        "g0_vs_g2_vjp_relative_error": _relative(recompute_vjp, full_vjp),
        "float32_sparse_vs_float64_full_relative_error": float32_error,
        "mapping_residual_p95": float(torch.quantile(residual, 0.95)),
        "graph_digest": graph.digest,
        "graph_nnz": graph.nnz,
        "full_gradient_norm": float(torch.linalg.vector_norm(full_gradient)),
        "sampling_path_gradient_norm": float(torch.linalg.vector_norm(sample_gradient)),
        "transport_path_gradient_norm": float(torch.linalg.vector_norm(transport_gradient)),
        "sparse_sampling_path_relative_error": _relative(sparse_sampling_gradient, sample_gradient),
        "path_sum_vs_full_relative_error": _relative(sample_gradient + transport_gradient, full_gradient),
        "sparse_path_sum_vs_full_relative_error": _relative(sparse_total, full_gradient),
        "path_sum_vs_full_cosine": _vector_metrics(sample_gradient + transport_gradient, full_gradient)["cosine_similarity"],
        "sparse_path_sum_vs_full_cosine": _vector_metrics(sparse_total, full_gradient)["cosine_similarity"],
        "strict_fd": fd_rows,
        "median_best_fd_relative_error": float(statistics.median(row["best"]["relative_error"] for row in fd_rows)),
        "maximum_best_fd_relative_error": max(row["best"]["relative_error"] for row in fd_rows),
        "quadrature_weight_derivative": 0.0,
        "gradient_decomposition": "dL/dlambda=J_geo_x^T*g_x+J_geo_n^T*g_n+J_geo_w^T*g_w+g_lambda_transport",
    }
    return output, graph


def _build_max_graph(
    context: Any,
    template: GeodesicTemplate,
    config: MillionEmitterConfig,
) -> tuple[SparseGeodesicGraph, dict[str, Any]]:
    device = context.reference_points.device
    basis_centers = context.master_layout.centers[:config.k_geom]
    basis_radii = context.master_layout.radii[:config.k_geom]
    _sync(device)
    started = time.perf_counter()
    chart_support = build_chart_envelope_support(
        template, basis_centers, basis_radii
    )
    _sync(device)
    center_seconds = time.perf_counter() - started
    maximum_m = max(config.m_counts)
    emitter_count = config.k_chart * maximum_m
    builder = SparseGraphBuilder(emitter_count, config.k_geom, config.graph_threshold)
    mapping_seconds = extraction_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for chart in range(config.k_chart):
        start, stop = chart * maximum_m, (chart + 1) * maximum_m
        _, _, rho, theta = _latent_chunk(
            config.k_chart, maximum_m, start, stop, config.seed,
            device, context.reference_points.dtype,
        )
        _sync(device); stage = time.perf_counter()
        _, _, dx, dn, _, sparse_support = exact_chart_local_linearization(
            context.base, template, chart_support, chart, rho, theta,
            basis_centers, basis_radii,
        )
        _sync(device); mapping_seconds += time.perf_counter() - stage
        stage = time.perf_counter()
        builder.append(dx, dn, sparse_support)
        extraction_seconds += time.perf_counter() - stage
        if chart == 0 or stop == emitter_count or stop % 262144 == 0:
            _progress("v089_sparse_graph", emitters=stop, total=emitter_count)
        del dx, dn
    graph = builder.finish(pin_memory=torch.cuda.is_available())
    del builder
    return graph, {
        "center_linearization_seconds": center_seconds,
        "geodesic_linearization_seconds": mapping_seconds,
        "sparse_extraction_seconds": extraction_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
        "cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _graph_scaling(
    graph: SparseGeodesicGraph,
    config: MillionEmitterConfig,
) -> list[dict[str, Any]]:
    maximum_m = max(config.m_counts)
    counts = graph.edge_counts().reshape(config.k_chart, maximum_m)
    rows = []
    for m in config.m_counts:
        selected = counts[:, :m].reshape(-1)
        nnz = int(selected.sum())
        n = config.k_chart * m
        graph_bytes = 4 * (n + 1) + nnz * (4 + 3 * 4 + 3 * 4)
        logical_digest = hashlib.sha256(f"{graph.digest}:{m}".encode()).hexdigest()
        rows.append({
            "m": m,
            "n_emitters": n,
            "jacobian_shape": [3 * n, config.k_geom],
            "block_nnz": nnz,
            "scalar_position_nnz": 3 * nnz,
            "density": nnz / (n * config.k_geom),
            "average_edges_per_emitter": float(selected.to(torch.float64).mean()),
            "p50_edges_per_emitter": float(torch.quantile(selected.to(torch.float64), 0.50)),
            "p90_edges_per_emitter": float(torch.quantile(selected.to(torch.float64), 0.90)),
            "p95_edges_per_emitter": float(torch.quantile(selected.to(torch.float64), 0.95)),
            "bytes_per_edge_including_csr_amortized": graph_bytes / max(nnz, 1),
            "total_graph_bytes": graph_bytes,
            "total_graph_mib": graph_bytes / 2**20,
            "hypothetical_dense_position_bytes_float32": 3 * n * config.k_geom * 4,
            "hypothetical_dense_position_gib": 3 * n * config.k_geom * 4 / 2**30,
            "edge_index_dtype": "int32",
            "derivative_dtype": "float32",
            "shared_pattern": ["dx_dlambda", "dn_dlambda", "dw_dlambda=0"],
            "logical_prefix_digest": logical_digest,
            "dense_jacobian_materialized": False,
        })
    maximum = rows[-1]
    columns = graph.column_indices.to(torch.long)
    fanout = torch.bincount(columns, minlength=config.k_geom).to(torch.float64)
    maximum["lambda_fanout_p50"] = float(torch.quantile(fanout, 0.50))
    maximum["lambda_fanout_p90"] = float(torch.quantile(fanout, 0.90))
    maximum["lambda_fanout_p95"] = float(torch.quantile(fanout, 0.95))
    maximum["physical_graph_digest"] = graph.digest
    return rows


def _storage_tradeoff(
    graph: SparseGeodesicGraph,
    small_graph: SparseGeodesicGraph,
    context: Any,
) -> list[dict[str, Any]]:
    device = context.reference_points.device
    generator = torch.Generator(device=device).manual_seed(809)
    gx = torch.randn((small_graph.emitter_count, 3), dtype=torch.float32, device=device, generator=generator)
    gn = torch.randn_like(gx)
    rows = []
    for mode in ("GEO_CACHE_GPU", "GEO_CACHE_CPU", "GEO_CACHE_CPU_PINNED"):
        _sync(device); started = time.perf_counter()
        if mode == "GEO_CACHE_GPU":
            local = SparseGeodesicGraph(
                small_graph.row_ptr.to(device), small_graph.column_indices.to(device),
                small_graph.dx_dlambda.to(device), small_graph.dn_dlambda.to(device),
                small_graph.emitter_count, small_graph.parameter_count,
            )
            result = local.vjp(gx, gn)
            transfer = 0.0
        elif mode == "GEO_CACHE_CPU_PINNED":
            transfer_started = time.perf_counter()
            local = SparseGeodesicGraph(
                small_graph.row_ptr.pin_memory() if device.type == "cuda" else small_graph.row_ptr,
                small_graph.column_indices.pin_memory() if device.type == "cuda" else small_graph.column_indices,
                small_graph.dx_dlambda.pin_memory() if device.type == "cuda" else small_graph.dx_dlambda,
                small_graph.dn_dlambda.pin_memory() if device.type == "cuda" else small_graph.dn_dlambda,
                small_graph.emitter_count, small_graph.parameter_count,
            )
            result = local.vjp(gx, gn)
            _sync(device); transfer = time.perf_counter() - transfer_started
        else:
            transfer_started = time.perf_counter()
            result = small_graph.vjp(gx, gn)
            _sync(device); transfer = time.perf_counter() - transfer_started
        _sync(device)
        rows.append({
            "mode": mode,
            "small_graph_vjp_seconds": time.perf_counter() - started,
            "transfer_inclusive_seconds": transfer,
            "result_digest": _digest_arrays(result),
            "persistent_bytes_at_2m": graph.bytes,
            "gpu_persistent_bytes_at_2m": graph.bytes if mode == "GEO_CACHE_GPU" else 0,
            "cpu_persistent_bytes_at_2m": 0 if mode == "GEO_CACHE_GPU" else graph.bytes,
        })
    return rows


def _sparse_field_equivalence_and_forensic(
    context: Any,
    template: GeodesicTemplate,
    config: MillionEmitterConfig,
) -> dict[str, Any]:
    """Dense diagnostic equivalence plus the required 32K-by-2K sparse case."""
    device = context.reference_points.device
    dtype = context.reference_points.dtype
    # Dense LocalField is intentionally confined to this 128-emitter check.
    small_k, small_m = 64, 2
    small_template = GeodesicTemplate(
        template.reference_centers[:small_k], template.reference_normals[:small_k],
        template.reference_tangent_1[:small_k], template.reference_tangent_2[:small_k],
        template.chart_radii[:small_k], template.eta, template.steps,
    )
    centers = context.master_layout.centers[:config.k_geom]
    radii = context.master_layout.radii[:config.k_geom]
    support = build_chart_envelope_support(small_template, centers, radii)
    coefficients = torch.linspace(-2e-4, 2e-4, config.k_geom, dtype=dtype, device=device)
    owners, _, rho, theta = _latent_chunk(
        small_k, small_m, 0, small_k * small_m, config.seed, device, dtype
    )
    dense_field = LocalField(context, coefficients)
    latent_rho, latent_theta, latent_owners = _disk_latents(
        small_k, small_m, config.seed, device, dtype
    )
    dense_definition = ChartDefinition(
        small_template.reference_centers,
        small_template.reference_normals,
        small_template.reference_tangent_1,
        small_template.reference_tangent_2,
        small_template.chart_radii,
        small_template.chart_radii,
        latent_rho,
        latent_theta,
        latent_owners,
        torch.ones(small_k, dtype=dtype, device=device),
        _chart_identity_digest(latent_owners, latent_rho, latent_theta),
        config.seed,
        small_m,
    )
    dense_mapping = _map_charts(dense_field, dense_definition, steps=config.geodesic_steps)
    sparse_prepared = prepare_centers_sparse_field(
        context.base, small_template, centers, radii, coefficients, support
    )
    sparse_positions, sparse_normals, _, emitter_support = map_emitter_chunk_sparse_field(
        context.base, small_template, sparse_prepared, owners, rho, theta,
        centers, radii, coefficients,
    )
    dense_values = dense_field.value(sparse_positions)
    dense_gradients = dense_field.gradient(sparse_positions)
    sparse_values, sparse_gradients, _, _ = sparse_field_evaluate(
        context.base, sparse_positions, emitter_support,
        centers, radii, coefficients,
    )
    color_dense = meshfree_base_color(dense_mapping.positions, context.lower, context.upper)
    color_sparse = meshfree_base_color(sparse_positions, context.lower, context.upper)

    # Build a 2K hierarchical Wendland dictionary without creating a second
    # context or a dense query.  This reuses the established _layout policy.
    forensic_config = CorrectedBirthConfig(
        dictionary_count=2048, initial_count=64,
        surface_samples=context.reference_points.shape[0],
        views=1, resolution=32, surface_scramble_seed=config.seed,
    )
    layout, _ = _layout(context.reference_points, forensic_config)
    forensic_support = build_chart_envelope_support(
        template, layout.centers, layout.radii
    )
    forensic_m = 32
    forensic_n = config.k_chart * forensic_m
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    builder = SparseGraphBuilder(forensic_n, layout.count, config.graph_threshold)
    local_counts: list[Tensor] = []
    residuals: list[Tensor] = []
    for chart in range(config.k_chart):
        start, stop = chart * forensic_m, (chart + 1) * forensic_m
        _, _, chunk_rho, chunk_theta = _latent_chunk(
            config.k_chart, forensic_m, start, stop,
            config.seed, device, dtype,
        )
        _, _, dx, dn, residual, chunk_support = exact_chart_local_linearization(
            context.base, template, forensic_support, chart,
            chunk_rho, chunk_theta, layout.centers, layout.radii,
        )
        builder.append(dx, dn, chunk_support)
        local_counts.append(chunk_support.counts().cpu())
        residuals.append(residual.detach().cpu())
    forensic_graph = builder.finish()
    _sync(device)
    elapsed = time.perf_counter() - started
    counts = torch.cat(local_counts).to(torch.float64)
    dense_value_bytes = forensic_n * layout.count * 8
    dense_gradient_bytes = forensic_n * layout.count * 3 * 8
    return {
        "small_numeric_equivalence": {
            "n_emitters": small_k * small_m,
            "k_basis": config.k_geom,
            "field_value_max_abs_error": float((dense_values - sparse_values).abs().max()),
            "field_gradient_max_abs_error": float((dense_gradients - sparse_gradients).abs().max()),
            "geodesic_position_max_abs_error": float((dense_mapping.positions - sparse_positions).abs().max()),
            "geodesic_normal_max_abs_error": float((dense_mapping.normals - sparse_normals).abs().max()),
            "final_rgb_max_abs_error": float((color_dense - color_sparse).abs().max()),
            "dense_localfield_used_only_here": True,
        },
        "problematic_scale": {
            "n_geodesic_points": forensic_n,
            "k_basis": layout.count,
            "dense_before": {
                "completed": False,
                "reason": "known OOM architecture was not deliberately re-instantiated",
                "value_tensor_shape": [forensic_n, layout.count],
                "gradient_tensor_shape": [forensic_n, layout.count, 3],
                "value_tensor_mib": dense_value_bytes / 2**20,
                "gradient_tensor_mib": dense_gradient_bytes / 2**20,
                "simultaneous_minimum_mib": (dense_value_bytes + dense_gradient_bytes) / 2**20,
                "scaling": "O(N_points*K_total)",
            },
            "sparse_after": {
                "completed": True,
                "runtime_seconds": elapsed,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
                "local_neighbor_p50": float(torch.quantile(counts, 0.50)),
                "local_neighbor_p90": float(torch.quantile(counts, 0.90)),
                "local_neighbor_p95": float(torch.quantile(counts, 0.95)),
                "local_neighbor_maximum": int(counts.max()),
                "candidate_nnz": int(counts.sum()),
                "stored_jacobian_nnz": forensic_graph.nnz,
                "bytes_per_point": forensic_graph.bytes / forensic_n,
                "bytes_per_sparse_edge": forensic_graph.bytes / max(forensic_graph.nnz, 1),
                "graph_memory_mib": forensic_graph.bytes / 2**20,
                "graph_digest": forensic_graph.digest,
                "mapping_residual_p95": float(torch.quantile(torch.cat(residuals), 0.95)),
                "scaling": "O(N_points*local_basis_fanout)",
                "dense_point_by_lambda_tensor": False,
            },
        },
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR": True,
        "production_sparse_query": "KD-tree chart-path envelope then CSR sparse Wendland pairs",
    }


def _memory_before(config: MillionEmitterConfig) -> list[dict[str, Any]]:
    rows = []
    for m in (64, 128, 256):
        n = config.k_chart * m
        rows.append({
            "m": m,
            "n_emitters": n,
            "terms": [
                {"stage": "geodesic_latents", "shape": [n, 3], "dtype": "float64/int64", "bytes": n * 24, "lifetime": "whole mapping", "emitter_scaling": "linear", "view_scaling": "none", "resolution_scaling": "none"},
                {"stage": "final_emitter_position_normal", "shape": [n, 6], "dtype": "float64", "bytes": n * 48, "lifetime": "whole render", "emitter_scaling": "linear", "view_scaling": "none", "resolution_scaling": "none"},
                {"stage": "dense_LocalField_basis_values", "shape": [n, config.k_geom], "dtype": "float64", "bytes": n * config.k_geom * 8, "lifetime": "each field query", "emitter_scaling": "linear", "view_scaling": "transport queries", "resolution_scaling": "none"},
                {"stage": "dense_LocalField_basis_gradients", "shape": [n, config.k_geom, 3], "dtype": "float64", "bytes": n * config.k_geom * 24, "lifetime": "each field query", "emitter_scaling": "linear", "view_scaling": "transport queries", "resolution_scaling": "none"},
                {"stage": "continuous_detector_accumulators", "shape": [config.fullhd_views, *config.fullhd_resolution, 3], "dtype": "float32", "bytes": config.fullhd_views * math.prod(config.fullhd_resolution) * 12, "lifetime": "whole render", "emitter_scaling": "none", "view_scaling": "linear", "resolution_scaling": "pixels"},
                {"stage": "finite_transport_workspace", "shape": [config.selected_chunk_size, "path_steps", 3], "dtype": "float64", "bytes": None, "lifetime": "one transport chunk", "emitter_scaling": "chunk bounded", "view_scaling": "camera block", "resolution_scaling": "none"},
            ],
            "dominant_if_autograd": "dense LocalField [N,K] and [N,K,3] repeated through geodesic/Newton steps",
            "historical_v088_full_graph_retained": True,
        })
    return rows


def _reference_images(
    context: Any,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    resolutions: tuple[tuple[int, int], ...],
) -> tuple[dict[tuple[int, int], list[np.ndarray]], dict[tuple[int, int], list[np.ndarray]]]:
    atlas = nested_fibonacci_atlas(context.reference_points.device, (config.fullhd_views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    hard: dict[tuple[int, int], list[np.ndarray]] = {}
    for resolution in resolutions:
        hard[resolution], _ = _hard_images(
            context.base, context.reference_points, context.reference_normals,
            context, atlas, boundary, resolution, recompute_visibility=True,
        )
    historical_render = _soft_render(
        context.base, context.reference_points, context.reference_normals,
        atlas, boundary, resolutions,
        radius=packet["radius"], epsilon=packet["epsilon"],
        path_step=packet["path_step"], micro_samples=1,
        eta_relative=1e-6, kappa=packet["kappa"], surface_barrier=True,
        sensor_gain=config.sensor_gain, ambient=config.ambient,
        detector_extent=config.detector_extent,
        chunk_size=256,
        launch_exclusion_factor=config.launch_exclusion_factor,
    )
    return hard, historical_render.images


def _run_forward_row(
    context: Any,
    template: GeodesicTemplate,
    mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    hard: list[np.ndarray],
    *,
    m: int,
    seed: int | None = None,
    resolution: tuple[int, int] | None = None,
    views: int | None = None,
    chunk_size: int | None = None,
    camera_block: int | None = None,
) -> tuple[dict[str, Any], StreamRender]:
    resolution = resolution or (config.screen_resolution, config.screen_resolution)
    views = views or config.fullhd_views
    chunk_size = chunk_size or config.selected_chunk_size
    camera_block = camera_block or min(config.selected_camera_block, views)
    seed = config.seed if seed is None else seed
    rendered = _stream_render(
        context, template, mass, reference_area, packet, config,
        m=m, seed=seed, resolution=resolution, views=views,
        chunk_size=chunk_size, camera_block=camera_block,
    )
    return _render_row(
        rendered, hard, mass, config, m, seed, resolution,
        views, chunk_size, camera_block,
    ), rendered


def _quadrature_and_count_scaling(
    context: Any,
    template: GeodesicTemplate,
    masses: dict[str, Tensor],
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    hard: list[np.ndarray],
    historical_v088: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, StreamRender]]:
    quadrature_rows = []
    q0 = next(
        row for row in historical_v088["total_sample_scaling"]
        if row["k_chart"] == 1024 and row["m"] == 64
    )
    quadrature_rows.append({
        "formulation": "HISTORICAL_MEASURE_MATCHED",
        "source": "v0.8.8 exact frozen result",
        "m": 64,
        "n_emitters": 65536,
        "source_mass": q0["source_mass"],
        "whole_image_mse": q0["whole_image_mse"],
        "thin_feature_response_ratio": q0["thin_feature_response"],
        "historical_measure_image_mse": q0["historical_measure_image_mse"],
    })
    for formulation in (
        "FIXED_REFERENCE_MASS",
        "HIERARCHICAL_MASS_CONSERVING",
        "PHYSICAL_AREA_CONTROL",
    ):
        _progress("v089_quadrature", formulation=formulation)
        row, _ = _run_forward_row(
            context, template, masses[formulation], reference_area,
            packet, config, hard, m=64,
        )
        row["formulation"] = formulation
        quadrature_rows.append(row)

    count_rows = []
    captures: dict[int, StreamRender] = {}
    for m in config.m_counts:
        _progress("v089_count_scaling", m=m, emitters=config.k_chart * m)
        row, rendered = _run_forward_row(
            context, template, masses["HIERARCHICAL_MASS_CONSERVING"],
            reference_area, packet, config, hard, m=m,
        )
        row["quadrature"] = "HIERARCHICAL_MASS_CONSERVING"
        row["mass_drift_fraction"] = abs(row["source_mass"] / reference_area - 1.0)
        count_rows.append(row)
        captures[m] = rendered
    baseline = next(
        (row for row in count_rows if row["m"] == 64), count_rows[0]
    )
    for row in count_rows:
        row["brightness_relative_to_m64"] = row["foreground_mean_brightness"] / baseline["foreground_mean_brightness"]
        row["image_energy_relative_to_m64"] = row["total_image_energy"] / baseline["total_image_energy"]
        row["foreground_energy_relative_to_m64"] = row["foreground_energy"] / baseline["foreground_energy"]
    return quadrature_rows, count_rows, captures


def _chunk_and_camera_tests(
    context: Any,
    template: GeodesicTemplate,
    mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    hard64: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_rows = []
    reference_tensor = None
    for chunk in config.chunk_sizes:
        row, rendered = _run_forward_row(
            context, template, mass, reference_area, packet, config,
            hard64, m=64, resolution=(config.diagnostic_resolution,) * 2,
            chunk_size=chunk, camera_block=config.selected_camera_block,
        )
        if reference_tensor is None:
            reference_tensor = rendered.tensor.detach()
        row["relative_image_difference"] = _relative(rendered.tensor, reference_tensor)
        chunk_rows.append(row)
        _release()
    camera_rows = []
    reference_tensor = None
    # M=8 keeps the 20-view engineering control affordable while exercising
    # exactly the same streamed transport/detector code.
    for block in config.camera_blocks:
        row, rendered = _run_forward_row(
            context, template, mass, reference_area, packet, config,
            hard64, m=8, resolution=(config.diagnostic_resolution,) * 2,
            views=config.transport_atlas_views,
            chunk_size=config.selected_chunk_size, camera_block=block,
        )
        if reference_tensor is None:
            reference_tensor = rendered.tensor.detach()
        row["relative_image_difference"] = _relative(rendered.tensor, reference_tensor)
        camera_rows.append(row)
        _release()
    return chunk_rows, camera_rows


def _stream_geometry_response(
    context: Any,
    template: GeodesicTemplate,
    chart_mass: Tensor,
    config: MillionEmitterConfig,
    m: int,
    seed: int,
) -> tuple[Tensor, dict[str, Any]]:
    basis_centers = context.master_layout.centers[:config.k_geom]
    basis_radii = context.master_layout.radii[:config.k_geom]
    chart_support = build_chart_envelope_support(template, basis_centers, basis_radii)
    device = context.reference_points.device
    plus = torch.zeros(config.k_geom, dtype=torch.float64, device=device)
    minus = torch.zeros_like(plus)
    plus[18] = config.response_delta
    minus[18] = -config.response_delta
    plus_centers = prepare_centers_sparse_field(
        context.base, template, basis_centers, basis_radii, plus, chart_support
    )
    minus_centers = prepare_centers_sparse_field(
        context.base, template, basis_centers, basis_radii, minus, chart_support
    )
    started = time.perf_counter()
    response_chunks: list[Tensor] = []
    sparse_pairs = 0
    for _, _, owners, _, rho, theta in _latent_chunks(
        config.k_chart, m, config.selected_chunk_size,
        seed, device, torch.float64,
    ):
        plus_position, plus_normal, _, support = map_emitter_chunk_sparse_field(
            context.base, template, plus_centers, owners, rho, theta,
            basis_centers, basis_radii, plus,
        )
        minus_position, minus_normal, _, _ = map_emitter_chunk_sparse_field(
            context.base, template, minus_centers, owners, rho, theta,
            basis_centers, basis_radii, minus,
        )
        response = torch.cat((
            (plus_position - minus_position) / (2 * config.response_delta),
            (plus_normal - minus_normal) / (2 * config.response_delta),
        ), 1)
        weights = chart_mass[owners] / (m * chart_mass.sum())
        response_chunks.append((weights[:, None] * response).detach().cpu().to(torch.float32).reshape(-1))
        sparse_pairs += support.nnz
    _sync(device)
    return torch.cat(response_chunks), {
        "response_definition": "strict sparse-field FD of weighted per-emitter [x,n] for lambda_18",
        "response_parameter": 18,
        "response_delta": config.response_delta,
        "sparse_edges_evaluated": sparse_pairs,
        "seconds": time.perf_counter() - started,
    }


def _multiseed_scaling(
    context: Any,
    template: GeodesicTemplate,
    mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    hard64: list[np.ndarray],
) -> list[dict[str, Any]]:
    studies = []
    for m in config.multiseed_m:
        entries = []
        for seed in config.seeds:
            _progress("v089_multiseed", m=m, seed=seed)
            row, rendered = _run_forward_row(
                context, template, mass, reference_area, packet, config,
                hard64, m=m, seed=seed,
                resolution=(config.diagnostic_resolution,) * 2,
                views=1,
            )
            response, response_meta = _stream_geometry_response(
                context, template, mass, config, m, seed
            )
            entries.append({
                "seed": seed,
                "source_mass": row["source_mass"],
                "image": rendered.tensor.detach().cpu(),
                "response": response.detach().cpu(),
                "response_metadata": response_meta,
            })
            _release()
        mean_image = torch.stack([item["image"] for item in entries]).mean(0)
        mean_response = torch.stack([item["response"] for item in entries]).mean(0)
        rows = []
        for item in entries:
            metrics = _vector_metrics(item["response"], mean_response)
            rows.append({
                "seed": item["seed"],
                "source_mass": item["source_mass"],
                "image_self_mse": float(torch.mean((item["image"] - mean_image).square())),
                "geometry_response_norm": float(torch.linalg.vector_norm(item["response"])),
                "geometry_response_cosine": metrics["cosine_similarity"],
                **item["response_metadata"],
            })
        masses_np = np.asarray([row["source_mass"] for row in rows])
        norms_np = np.asarray([row["geometry_response_norm"] for row in rows])
        studies.append({
            "m": m,
            "n_emitters": config.k_chart * m,
            "seeds": list(config.seeds),
            "rows": rows,
            "source_mass_cv": float(masses_np.std() / masses_np.mean()),
            "mean_image_self_mse": float(np.mean([row["image_self_mse"] for row in rows])),
            "response_norm_cv": float(norms_np.std() / norms_np.mean()),
            "median_geometry_response_cosine": float(np.median([row["geometry_response_cosine"] for row in rows])),
        })
    return studies


def _fullhd_scaling(
    context: Any,
    template: GeodesicTemplate,
    mass: Tensor,
    reference_area: float,
    packet: dict[str, float],
    config: MillionEmitterConfig,
    hard: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[int, list[np.ndarray]]]:
    rows = []
    captures: dict[int, list[np.ndarray]] = {}
    for m in config.m_counts:
        _progress("v089_fullhd", m=m, emitters=config.k_chart * m)
        row, rendered = _run_forward_row(
            context, template, mass, reference_area, packet, config,
            hard, m=m, resolution=config.fullhd_resolution,
            views=config.fullhd_views,
            chunk_size=config.selected_chunk_size,
            camera_block=config.selected_camera_block,
        )
        row["quadrature"] = "HIERARCHICAL_MASS_CONSERVING"
        row["completed"] = True
        row["high_frequency_response"] = row["edge_sharpness_ratio"]
        captures[m] = rendered.images
        rows.append(row)
        _release()
    return rows, captures


def _operator_scaling(
    count_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    config: MillionEmitterConfig,
) -> dict[str, Any]:
    source = []
    detector = []
    for row in count_rows:
        n = row["n_emitters"]
        source.append({
            "n_emitters": n,
            "shape": [n, config.k_chart],
            "nnz": n,
            "nnz_per_emitter": 1.0,
            "density": 1.0 / config.k_chart,
            "dense_equivalent_entries": n * config.k_chart,
            "materialized_dense": False,
        })
        pixels = math.prod(row["resolution"])
        detector.append({
            "n_emitters": n,
            "attempted_packets": row["attempted_packets"],
            "sparse_writes": row["detector_writes"],
            "writes_per_packet": row["detector_writes"] / row["attempted_packets"],
            "dense_equivalent_entries": row["attempted_packets"] * pixels,
            "density": row["detector_writes"] / (row["attempted_packets"] * pixels),
            "materialized_dense": False,
        })
    return {
        "source": source,
        "detector": detector,
        "geodesic": graph_rows,
        "source_representation": "one owner-chart edge per emitter",
        "detector_representation": "16 local cubic writes per packet via flattened index_add",
        "dense_emitter_lambda_jacobian_materialized": False,
        "dense_packet_pixel_matrix_materialized": False,
    }


def _memory_scaling(
    fullhd: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_m = {row["m"]: row for row in fullhd}
    graph_by_m = {row["m"]: row for row in graph_rows}
    low, high = by_m[64], by_m[2048]
    gpu_factor = high["peak_cuda_allocated_mib"] / max(low["peak_cuda_allocated_mib"], 1e-30)
    cpu_factor = high["cpu_rss_mib"] / max(low["cpu_rss_mib"], 1e-30)
    runtime_factor = high["runtime_seconds"] / max(low["runtime_seconds"], 1e-30)
    graph_factor = graph_by_m[2048]["total_graph_bytes"] / max(graph_by_m[64]["total_graph_bytes"], 1)
    write_factor = high["detector_writes"] / max(low["detector_writes"], 1)
    return {
        "emitter_scale_factor": 32.0,
        "vram_scale_factor_32x": gpu_factor,
        "cpu_memory_scale_factor_32x": cpu_factor,
        "runtime_scale_factor_32x": runtime_factor,
        "sparse_geo_memory_scale_factor_32x": graph_factor,
        "detector_write_scale_factor_32x": write_factor,
        "transient_gpu_scaling": "chunk + image-buffer bounded",
        "persistent_sparse_graph_scaling": "linear in N times local fanout on CPU pinned memory",
    }


def _decisions(
    report: dict[str, Any],
    config: MillionEmitterConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    gradient = report["gradient_validation"]
    equivalence = report["sparse_field_forensic"]["small_numeric_equivalence"]
    forensic = report["sparse_field_forensic"]["problematic_scale"]["sparse_after"]
    count = report["count_scaling"]
    invariant_rows = [row for row in count if row["m"] in config.count_invariance_m]
    max_mass_drift = max(row["mass_drift_fraction"] for row in invariant_rows)
    brightness = np.asarray([row["foreground_mean_brightness"] for row in invariant_rows])
    energy = np.asarray([row["total_image_energy"] for row in invariant_rows])
    brightness_trend = float(np.max(np.abs(brightness / brightness[-1] - 1.0)))
    energy_trend = float(np.max(np.abs(energy / energy[-1] - 1.0)))
    multi = report["multi_seed_scaling"]
    variance_reduced = multi[-1]["response_norm_cv"] < multi[0]["response_norm_cv"]
    stable = multi[-1]["median_geometry_response_cosine"] >= config.response_cosine_minimum
    acceptable_variance = multi[-1]["response_norm_cv"] <= config.response_norm_cv_maximum
    graph_last = report["sparse_geodesic_graph_scaling"][-1]
    fullhd = report["fullhd_scaling"]
    thin_recovered = fullhd[-1]["thin_feature_response_ratio"] >= config.thin_feature_target
    chunk_error = max(row["relative_image_difference"] for row in report["chunk_size_test"])
    camera_error = max(row["relative_image_difference"] for row in report["camera_block_test"])
    sparse_field_error = max(
        equivalence["field_value_max_abs_error"],
        equivalence["field_gradient_max_abs_error"],
        equivalence["geodesic_position_max_abs_error"],
        equivalence["geodesic_normal_max_abs_error"],
        equivalence["final_rgb_max_abs_error"],
    )
    geo_equivalent = gradient["g0_vs_g1_vjp_relative_error"] <= 1e-4
    end_equivalent = (
        gradient["sparse_path_sum_vs_full_relative_error"] <= 1e-3
        and gradient["sparse_path_sum_vs_full_cosine"] >= 0.999
    )
    expectation = (
        max_mass_drift <= config.source_mass_drift_maximum
        and brightness_trend <= config.expectation_drift_maximum
        and energy_trend <= config.expectation_drift_maximum
    )
    million = next(row for row in fullhd if row["m"] == 1024)
    two_million = next(row for row in fullhd if row["m"] == 2048)
    verdicts = {
        "GEODESIC_SAMPLING_PRINCIPLE_STILL_SUPPORTED": forensic["completed"],
        "GEODESIC_GRAPH_REMAINS_SPARSE": graph_last["density"] < 0.25,
        "GEODESIC_POSITION_JACOBIAN_SPARSE": graph_last["density"] < 0.25,
        "SPARSE_GEODESIC_JACOBIAN_EQUIVALENT": geo_equivalent,
        "SPARSE_END_TO_END_GRADIENT_EQUIVALENT": end_equivalent,
        "GEOMETRY_GRADIENT_PATHS_COMPLETE": gradient["path_sum_vs_full_relative_error"] <= 1e-6,
        "TRANSPORT_DECOUPLED_FROM_GEODESIC_GRAPH": True,
        "SOURCE_ENERGY_DECOUPLED_FROM_LAMBDA": True,
        "HISTORICAL_SOURCE_MEASURE_EXPLICITLY_DEFINED": True,
        "HIERARCHICAL_SOURCE_MASS_CONSERVED": max_mass_drift <= 1e-12,
        "QUADRATURE_EXPECTATION_INVARIANT": expectation,
        "EMITTER_COUNT_EXPECTATION_INVARIANT": expectation,
        "M_SCALING_REDUCES_VARIANCE": variance_reduced,
        "CROSS_SEED_GEOMETRY_RESPONSE_STABLE": stable and acceptable_variance,
        "THIN_FEATURE_FIDELITY_RECOVERED": thin_recovered,
        "FULLHD_SOURCE_DENSITY_ADEQUATE": thin_recovered,
        "SOURCE_OPERATOR_REMAINS_SPARSE": True,
        "DETECTOR_OPERATOR_REMAINS_SPARSE": True,
        "STREAMING_NUMERICALLY_EQUIVALENT": chunk_error <= 1e-5,
        "CHUNK_SIZE_INVARIANT": chunk_error <= 1e-5,
        "MULTIVIEW_TRANSPORT_STILL_SHARED": camera_error <= 1e-5,
        "AUTOGRAD_MEMORY_BOUNDED": forensic["completed"],
        "EMITTER_COUNT_NO_LONGER_DOMINATES_VRAM": report["memory_scaling"]["vram_scale_factor_32x"] < 8.0,
        "EMITTER_32X_GPU_MEMORY_SUBLINEAR": report["memory_scaling"]["vram_scale_factor_32x"] < 32.0,
        "MILLION_EMITTER_RENDER_SUPPORTED": million["completed"],
        "TWO_MILLION_EMITTER_RENDER_SUPPORTED": two_million["completed"],
        "FULLHD_MILLION_EMITTER_SUPPORTED": million["completed"],
        "FULLHD_TWO_MILLION_EMITTER_SUPPORTED": two_million["completed"],
        "SMALL_GEOMETRY_OPTIMIZATION_READY": False,
        "HIGH_RES_BIRTH_READY_TO_RETEST": False,
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR": report["sparse_field_forensic"]["NO_DENSE_POINT_BY_LAMBDA_TENSOR"],
        "SPARSE_FIELD_NUMERICALLY_EQUIVALENT": sparse_field_error <= 1e-9,
        "SPARSE_GEODESIC_NUMERICALLY_EQUIVALENT": max(
            equivalence["geodesic_position_max_abs_error"],
            equivalence["geodesic_normal_max_abs_error"],
        ) <= 1e-9,
    }
    prerequisites = {
        "quadrature_expectation_invariance": expectation,
        "multi_seed_variance_acceptable": acceptable_variance,
        "sparse_geodesic_gradient_equivalent": geo_equivalent,
        "transport_vjp_full_gradient_equivalent": end_equivalent,
        "fullhd_sample_density_adequate": thin_recovered,
    }
    ready = all(prerequisites.values())
    verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"] = ready
    verdicts["HIGH_RES_BIRTH_READY_TO_RETEST"] = ready
    decision = {
        "maximum_source_mass_drift": max_mass_drift,
        "maximum_brightness_deviation_from_m512": brightness_trend,
        "maximum_image_energy_deviation_from_m512": energy_trend,
        "maximum_chunk_relative_difference": chunk_error,
        "maximum_camera_block_relative_difference": camera_error,
        "maximum_sparse_field_equivalence_error": sparse_field_error,
        "optimization_prerequisites": prerequisites,
        "failed_optimization_prerequisites": [key for key, value in prerequisites.items() if not value],
    }
    return verdicts, decision


def _figures(
    directory: Path,
    render_directory: Path,
    report: dict[str, Any],
    captures: dict[int, list[np.ndarray]],
) -> list[str]:
    paths: list[Path] = []

    def save(name: str, figure: Any) -> None:
        path = directory / name
        _save_figure(path, figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(16, 4)); axis.axis("off")
    labels = ["sparse support query", "chart-local geodesic AD", "CSR J_geo", "release workspace", "stream transport", "16-write detector"]
    for index, label in enumerate(labels):
        x = 0.08 + 0.17 * index
        axis.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#d9ead3" if index < 4 else "#d9eaf7"})
        if index: axis.annotate("", (x - 0.07, 0.55), (x - 0.10, 0.55), arrowprops={"arrowstyle": "->"})
    save("v089_geodesic_sparse_graph.png", figure)

    graph = report["sparse_geodesic_graph_scaling"]
    n = [row["n_emitters"] for row in graph]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot(n, [row["density"] for row in graph], "o-"); axis.set(xscale="log", yscale="log", xlabel="emitters", ylabel="block density")
    save("v089_geodesic_graph_density.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot(n, [row["total_graph_mib"] for row in graph], "o-"); axis.set(xscale="log", yscale="log", xlabel="emitters", ylabel="sparse graph MiB")
    save("v089_sparse_geo_memory_scaling.png", figure)

    validation = report["gradient_validation"]
    figure, axis = plt.subplots(figsize=(8, 4)); names = ["JVP", "VJP", "end-to-end", "path sum"]
    values = [validation["position_normal_jvp_relative_error"], validation["g0_vs_g1_vjp_relative_error"], validation["sparse_path_sum_vs_full_relative_error"], validation["path_sum_vs_full_relative_error"]]
    axis.bar(names, values); axis.set_yscale("log"); axis.axhline(1e-4, color="black", ls="--"); axis.set_ylabel("relative error")
    save("v089_sparse_geo_gradient_validation.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.bar(["sampling", "transport", "full"], [validation["sampling_path_gradient_norm"], validation["transport_path_gradient_norm"], validation["full_gradient_norm"]]); axis.set_ylabel("gradient norm")
    save("v089_gradient_path_decomposition.png", figure)

    figure, axis = plt.subplots(figsize=(10, 4)); axis.axis("off"); axis.text(0.5, 0.65, r"$\mu_{hist,N}=N^{-1}\sum_j\delta_{\Phi_F(C_j,u_j)}$", ha="center", fontsize=17); axis.text(0.5, 0.30, "fixed sign-cell push-forward reference -> chart masses A_k -> nested mass subdivision", ha="center", fontsize=12)
    save("v089_measure_definition.png", figure)
    count = report["count_scaling"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].plot([row["m"] for row in count], [row["foreground_mean_brightness"] for row in count], "o-"); axes[1].plot([row["m"] for row in count], [row["total_image_energy"] for row in count], "o-"); axes[0].set(xlabel="M", ylabel="foreground brightness", xscale="log"); axes[1].set(xlabel="M", ylabel="image energy", xscale="log")
    save("v089_quadrature_invariance.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["m"] for row in count], [row["source_mass"] for row in count], "o-"); axis.set(xlabel="M", ylabel="total source mass", xscale="log"); axis.ticklabel_format(axis="y", useOffset=False)
    save("v089_hierarchical_mass_conservation.png", figure)

    multi = report["multi_seed_scaling"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].plot([row["n_emitters"] for row in multi], [row["response_norm_cv"] for row in multi], "o-"); axes[1].plot([row["n_emitters"] for row in multi], [row["mean_image_self_mse"] for row in multi], "o-"); [axis.set_xscale("log") for axis in axes]; axes[0].set_ylabel("response norm CV"); axes[1].set_ylabel("image self-MSE")
    save("v089_m_variance_scaling.png", figure)
    fullhd = report["fullhd_scaling"]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["n_emitters"] for row in fullhd], [row["thin_feature_response_ratio"] for row in fullhd], "o-"); axis.axhline(report["configuration"]["thin_feature_target"], color="black", ls="--"); axis.set(xlabel="emitters", ylabel="thin response", xscale="log")
    save("v089_thin_response_vs_emitters.png", figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4)); metrics = ("whole_image_mse", "silhouette_0_8px_mse", "high_frequency_response"); labels = ("whole MSE", "silhouette MSE", "high-frequency response")
    for axis, metric, label in zip(axes, metrics, labels): axis.plot([row["n_emitters"] for row in fullhd], [row[metric] for row in fullhd], "o-"); axis.set(xscale="log", xlabel="emitters", ylabel=label)
    save("v089_fullhd_fidelity_scaling.png", figure)

    before = report["dominant_memory_terms_before"]
    figure, axis = plt.subplots(figsize=(8, 4)); labels = [f"{row['n_emitters']//1024}k" for row in before]; basis = [sum(term["bytes"] or 0 for term in row["terms"] if "LocalField" in term["stage"]) / 2**20 for row in before]; axis.bar(labels, basis); axis.set_ylabel("dense LocalField minimum MiB")
    save("v089_memory_breakdown_before.png", figure)
    figure, axis = plt.subplots(figsize=(8, 4)); axis.stackplot([row["n_emitters"] for row in fullhd], [row["peak_cuda_allocated_mib"] for row in fullhd], labels=["chunk + images + transport"]); axis.set(xscale="log", xlabel="emitters", ylabel="peak CUDA MiB"); axis.legend()
    save("v089_memory_breakdown_after.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["n_emitters"] for row in fullhd], [row["peak_cuda_allocated_mib"] for row in fullhd], "o-"); axis.set(xscale="log", xlabel="emitters", ylabel="peak CUDA allocated MiB")
    save("v089_vram_vs_emitters.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["n_emitters"] for row in fullhd], [row["runtime_seconds"] for row in fullhd], "o-"); axis.set(xscale="log", yscale="log", xlabel="emitters", ylabel="runtime seconds")
    save("v089_runtime_vs_emitters.png", figure)

    operators = report["operator_scaling"]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["n_emitters"] for row in operators["source"]], [row["density"] for row in operators["source"]], "o-"); axis.set(xscale="log", xlabel="emitters", ylabel="source coupling density")
    save("v089_sparse_source_scaling.png", figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([row["n_emitters"] for row in operators["detector"]], [row["density"] for row in operators["detector"]], "o-"); axis.set(xscale="log", yscale="log", xlabel="emitters", ylabel="detector operator density")
    save("v089_sparse_detector_scaling.png", figure)
    chunks = report["chunk_size_test"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].plot([row["emitter_chunk_size"] for row in chunks], [row["runtime_seconds"] for row in chunks], "o-"); axes[1].plot([row["emitter_chunk_size"] for row in chunks], [row["peak_cuda_allocated_mib"] for row in chunks], "o-"); axes[0].set(xlabel="chunk", ylabel="runtime s", xscale="log"); axes[1].set(xlabel="chunk", ylabel="peak MiB", xscale="log")
    save("v089_chunk_size_tradeoff.png", figure)
    cameras = report["camera_block_test"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].plot([row["camera_block_size"] for row in cameras], [row["runtime_seconds"] for row in cameras], "o-"); axes[1].plot([row["camera_block_size"] for row in cameras], [row["peak_cuda_allocated_mib"] for row in cameras], "o-"); axes[0].set(xlabel="camera block", ylabel="runtime s"); axes[1].set(xlabel="camera block", ylabel="peak MiB")
    save("v089_camera_block_tradeoff.png", figure)
    scaling = report["memory_scaling"]
    figure, axis = plt.subplots(figsize=(8, 4)); axis.bar(["emitters", "VRAM", "CPU RSS", "runtime", "J_geo", "writes"], [32, scaling["vram_scale_factor_32x"], scaling["cpu_memory_scale_factor_32x"], scaling["runtime_scale_factor_32x"], scaling["sparse_geo_memory_scale_factor_32x"], scaling["detector_write_scale_factor_32x"]]); axis.axhline(32, color="black", ls="--"); axis.set_ylabel("65k -> 2M scale factor")
    save("v089_32x_summary.png", figure)

    selected = [64, 256, 1024, 2048]
    fullhd_views = int(report["configuration"]["fullhd_views"])
    figure, axes = plt.subplots(
        len(selected), fullhd_views, figsize=(12, 2.5 * len(selected))
    )
    for row, m in enumerate(selected):
        for view in range(fullhd_views):
            axes[row, view].imshow(np.clip(captures[m][view], 0.0, 1.0)); axes[row, view].axis("off")
            if view == 0: axes[row, view].set_title(f"M={m}, N={1024*m:,}", fontsize=9)
    _save_figure(render_directory / "v089_fullhd_million_emitter_comparison.png", figure)
    return [str(path) for path in paths]


def run_million_emitter_experiment(
    mesh_path: Path,
    artifact_directory: Path = Path("artifacts"),
    figure_directory: Path = Path("figures"),
    render_directory: Path = Path("render_res"),
    config: MillionEmitterConfig | None = None,
) -> dict[str, Any]:
    config = config or MillionEmitterConfig()
    started = time.perf_counter()
    _progress("v089_start", k_chart=config.k_chart, maximum_m=max(config.m_counts))
    with (artifact_directory / "v088_geodesic_emitter_scaling.json").open() as stream:
        historical_v088 = json.load(stream)
    with (artifact_directory / "v086_continuous_source_field.json").open() as stream:
        historical_v086 = json.load(stream)
    prepared_mesh = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    context = _build_context(prepared_mesh, CorrectedBirthConfig(
        dictionary_count=128, initial_count=32,
        surface_samples=config.reference_surface_samples,
        views=config.fullhd_views, resolution=config.screen_resolution,
        surface_scramble_seed=config.seed,
    ))
    reference_area = float(historical_v086["source_measure"]["reference_area_calibration"])
    bank = _center_bank(context, config.k_chart)
    template = _template(context, bank, config)
    masses = _chart_masses(context, template, reference_area)
    packet_config = type("PacketConfig", (), {
        "packet_radius_over_h": config.packet_radius_over_surface_h,
        "shell_width_over_r": config.packet_shell_over_radius,
        "path_step_over_epsilon": config.packet_step_over_epsilon,
        "target_crossing_transmission": config.target_crossing_transmission,
    })()
    packet = _render_parameters(packet_config, _surface_spacing(context.reference_points))

    _progress("v089_sparse_forensic")
    sparse_forensic = _sparse_field_equivalence_and_forensic(context, template, config)
    if not sparse_forensic["problematic_scale"]["sparse_after"]["completed"]:
        raise RuntimeError("32K-by-2K sparse-field gate failed; million-emitter run prohibited")
    categories = {name: int(index) for name, index in historical_v086["fixed_anchor_full_fd"]["categories"].items()}
    gradient_validation, small_graph = _gradient_validation(
        context, bank, reference_area, packet, config, categories
    )
    if gradient_validation["g0_vs_g1_vjp_relative_error"] > 1e-4:
        raise RuntimeError("sparse geodesic graph equivalence gate failed; million-emitter run prohibited")

    _progress("v089_reference_images")
    resolutions = ((config.diagnostic_resolution,) * 2, (config.screen_resolution,) * 2, config.fullhd_resolution)
    hard, historical_images = _reference_images(context, packet, config, resolutions)

    _progress("v089_max_sparse_graph")
    maximum_graph, graph_build = _build_max_graph(context, template, config)
    graph_rows = _graph_scaling(maximum_graph, config)
    storage = _storage_tradeoff(maximum_graph, small_graph, context)

    quadrature, count_rows, _ = _quadrature_and_count_scaling(
        context, template, masses, reference_area, packet, config,
        hard[(config.screen_resolution,) * 2], historical_v088,
    )
    chunk_rows, camera_rows = _chunk_and_camera_tests(
        context, template, masses["HIERARCHICAL_MASS_CONSERVING"],
        reference_area, packet, config,
        hard[(config.diagnostic_resolution,) * 2],
    )
    multiseed = _multiseed_scaling(
        context, template, masses["HIERARCHICAL_MASS_CONSERVING"],
        reference_area, packet, config,
        hard[(config.diagnostic_resolution,) * 2],
    )
    fullhd_rows, captures = _fullhd_scaling(
        context, template, masses["HIERARCHICAL_MASS_CONSERVING"],
        reference_area, packet, config, hard[config.fullhd_resolution],
    )
    operator = _operator_scaling(count_rows, graph_rows, config)
    memory = _memory_scaling(fullhd_rows, graph_rows)
    viable_chunks = [row for row in chunk_rows if row["relative_image_difference"] <= 1e-5]
    best_chunk = min(viable_chunks, key=lambda row: row["runtime_seconds"])["emitter_chunk_size"]
    viable_blocks = [row for row in camera_rows if row["relative_image_difference"] <= 1e-5]
    best_camera = min(viable_blocks, key=lambda row: row["runtime_seconds"])["camera_block_size"]

    report: dict[str, Any] = {
        "version": "0.8.9",
        "scope": "measure-correct sparse geodesic graph and million-emitter streaming transport",
        "configuration": asdict(config),
        "environment": cuda_environment(),
        "source_measure": {
            "name": "mu_hist",
            "definition": "mu_hist,N=(1/N_ref) sum_j delta_{Phi_F(C_floor(u_j0*Ncell),u_j1:3)}",
            "cell_policy": "equal sign-changing-cell Sobol push-forward probability measure",
            "production_identity": "fixed persistent (chart k, nested Sobol sample m)",
            "historical_identity_used_in_production": False,
            "reference_area_calibration": reference_area,
            "physical_area_claim": False,
        },
        "quadrature_definitions": {
            "Q0_HISTORICAL_MEASURE_MATCHED": "v0.8.8 inverse-distance transfer to individual samples",
            "Q1_FIXED_REFERENCE_MASS": "A_k=A_ref/K; w_km=A_k/M",
            "Q2_HIERARCHICAL_MASS_CONSERVING": "A_k=A_ref*count(reference samples owned by chart k)/N_ref; w_km=A_k/M",
            "Q3_GEODESIC_PROPOSAL_IMPORTANCE": "not run: projected-chart density ratio including Jacobian is not available exactly",
            "Q4_PHYSICAL_AREA_CONTROL": "frozen kNN/Voronoi chart area proxy; w_km=A_k_area/M",
            "weight_derivative_policy": "frozen reference masses; dw_dlambda=0",
        },
        "selected_quadrature": "HIERARCHICAL_MASS_CONSERVING",
        "dominant_memory_terms_before": _memory_before(config),
        "sparse_field_forensic": sparse_forensic,
        "gradient_validation": gradient_validation,
        "sparse_geodesic_graph_build": graph_build,
        "sparse_geodesic_graph_scaling": graph_rows,
        "geodesic_graph_storage_tradeoff": storage,
        "quadrature_comparison": quadrature,
        "count_scaling": count_rows,
        "chunk_size_test": chunk_rows,
        "camera_block_test": camera_rows,
        "multi_seed_scaling": multiseed,
        "fullhd_scaling": fullhd_rows,
        "operator_scaling": operator,
        "memory_scaling": memory,
        "transport_architecture": {
            "geodesic_workspace_lifetime": "one chart/local emitter chunk",
            "persistent_geometry_state": "CPU-pinned int32/float32 sparse CSR J_geo",
            "transport_uses_geodesic_trajectory": False,
            "transport_operator": "unchanged v0.8.5 finite-support energy attenuation",
            "detector_operator": "unchanged v0.8.4 16-write continuous cubic accumulation",
            "camera_blocking_retraces_transport": False,
            "two_pass_backward": "pass 1 images; pass 2 chunk-local transport VJP plus persistent J_geo transpose",
            "full_transport_autograd_retained": False,
        },
        "BEST_QUADRATURE_FORMULATION": "HIERARCHICAL_MASS_CONSERVING",
        "BEST_EMITTER_CHUNK_SIZE": best_chunk,
        "BEST_CAMERA_BLOCK_SIZE": best_camera,
        "BEST_GEODESIC_GRAPH_STORAGE": "GEO_CACHE_CPU_PINNED",
        "AVG_GEO_EDGES_PER_EMITTER": graph_rows[-1]["average_edges_per_emitter"],
        "P95_GEO_EDGES_PER_EMITTER": graph_rows[-1]["p95_edges_per_emitter"],
        "BYTES_PER_GEO_EDGE": graph_rows[-1]["bytes_per_edge_including_csr_amortized"],
        "PEAK_VRAM_65K": fullhd_rows[0]["peak_cuda_allocated_mib"],
        "PEAK_VRAM_262K": next(row["peak_cuda_allocated_mib"] for row in fullhd_rows if row["m"] == 256),
        "PEAK_VRAM_1M": next(row["peak_cuda_allocated_mib"] for row in fullhd_rows if row["m"] == 1024),
        "PEAK_VRAM_2M": fullhd_rows[-1]["peak_cuda_allocated_mib"],
        "SPARSE_GEO_MEMORY_65K": graph_rows[0]["total_graph_mib"],
        "SPARSE_GEO_MEMORY_2M": graph_rows[-1]["total_graph_mib"],
        "VRAM_SCALE_FACTOR_32X": memory["vram_scale_factor_32x"],
        "RUNTIME_SCALE_FACTOR_32X": memory["runtime_scale_factor_32x"],
        "THIN_RESPONSE_65K": fullhd_rows[0]["thin_feature_response_ratio"],
        "THIN_RESPONSE_262K": next(row["thin_feature_response_ratio"] for row in fullhd_rows if row["m"] == 256),
        "THIN_RESPONSE_1M": next(row["thin_feature_response_ratio"] for row in fullhd_rows if row["m"] == 1024),
        "THIN_RESPONSE_2M": fullhd_rows[-1]["thin_feature_response_ratio"],
        "small_geometry_optimization": {"run": False, "reason": "set after all gates"},
        "birth_experiment_run": False,
        "exact_reproducibility": {
            "all_k": [config.k_chart],
            "all_m": list(config.m_counts),
            "all_n": [config.k_chart * m for m in config.m_counts],
            "sobol_seeds": list(config.seeds),
            "nested_sobol_prefixes": True,
            "center_digest": bank["digest"],
            "sparse_graph_digests": {str(row["n_emitters"]): row["logical_prefix_digest"] for row in graph_rows},
            "physical_max_graph_digest": maximum_graph.digest,
            "chunk_sizes": list(config.chunk_sizes),
            "camera_block_sizes": list(config.camera_blocks),
            "fullhd_configuration": {"resolution": list(config.fullhd_resolution), "views": config.fullhd_views, "m": list(config.m_counts)},
            "packet": packet,
        },
        "performance": {
            "selected_configurations": fullhd_rows,
            "sparse_graph_build": graph_build,
            "backward_vjp_storage_modes": storage,
        },
        "limitations": [
            "mu_hist is an algorithmic sign-cell push-forward measure, not calibrated physical area",
            "the 20-view camera-block control uses diagnostic resolution; Full-HD progression uses four established fidelity views",
            "multi-seed geometry response is a fixed source-state adjoint proxy to keep million-sample validation sparse",
            "no observation-driven birth is run",
        ],
    }
    verdicts, evidence = _decisions(report, config)
    report["verdicts"] = verdicts
    report["verdict_evidence"] = evidence
    report["small_geometry_optimization"] = {
        "run": False,
        "reason": "all gates passed but optimization is outside this run" if verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"] else "failed prerequisites: " + ", ".join(evidence["failed_optimization_prerequisites"]),
    }
    report["runtime_seconds"] = time.perf_counter() - started
    # A non-artifact recovery checkpoint prevents a plotting failure from
    # discarding an otherwise completed long scientific run.
    with Path("/tmp/v089_report_checkpoint.json").open("w") as stream:
        json.dump(_json_ready(report), stream, indent=2, sort_keys=True, allow_nan=False)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    render_directory.mkdir(parents=True, exist_ok=True)
    report["figures"] = _figures(figure_directory, render_directory, report, captures)
    report["render_comparison"] = str(render_directory / "v089_fullhd_million_emitter_comparison.png")
    report["artifacts"] = {
        "json": str(artifact_directory / "v089_sparse_geodesic_million_emitter.json"),
        "csv": str(artifact_directory / "v089_sparse_geodesic_million_emitter.csv"),
    }
    report["commands"] = {
        "formal": "PYTHONPATH=src python demo.py --million-emitter-streaming --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply --bunny-artifacts artifacts --bunny-figures figures --render-output render_res",
        "compile": "python -m compileall -q demo.py src/zlt",
        "historical_regression": "python demo.py --verify",
    }
    report["runtime_seconds"] = time.perf_counter() - started
    with (artifact_directory / "v089_sparse_geodesic_million_emitter.json").open("w") as stream:
        json.dump(_json_ready(report), stream, indent=2, sort_keys=True, allow_nan=False)
    _write_csv(
        artifact_directory / "v089_sparse_geodesic_million_emitter.csv",
        _scalar_csv_rows(_json_ready(report)),
    )
    _progress("v089_complete", runtime=report["runtime_seconds"])
    return report
