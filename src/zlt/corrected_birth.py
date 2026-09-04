"""Observation-driven DoF birth under corrected mesh-free forward RGB images."""

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

from .benchmark import cuda_environment
from .boundary_transport import (
    DirectionAtlas,
    ObservationSphere,
    build_boundary_transport,
    enclosing_observation_sphere,
    nested_fibonacci_atlas,
)
from .bunny import BunnyGeometryEvaluator, _normal_line_roots
from .locality import (
    BasisLayout,
    SupportPairs,
    wendland_gradients,
    wendland_values,
)
from .mesh_field import GridZeroSetField, prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color, sample_meshfree_zero_set
from .sequential import _correlation, _multiscale_support


Tensor = torch.Tensor


@dataclass(frozen=True)
class CorrectedBirthConfig:
    dictionary_count: int = 4096
    initial_count: int = 32
    surface_samples: int = 32768
    views: int = 8
    resolution: int = 256
    resolution_width: int | None = None
    surface_scramble_seed: int | None = None
    footprint_reference_resolution: int | None = None
    base_support_radius: float = 0.45
    support_margin: float = 0.005
    coefficient_limit: float = 0.03
    deformation_iterations: int = 16
    deformation_max_offset: float = 0.05
    damping: float = 1e-7
    cg_iterations: int = 8
    initial_optimization_steps: int = 5
    post_birth_steps: int = 1
    fixed_optimization_steps: int = 5
    line_evaluations: int = 10
    sensor_gain: float = 1.5
    support_threshold: float = 0.05
    ambient: float = 0.35
    minimum_active_before_stopping: int = 1024
    flattening_window: int = 3
    flattening_relative_gain: float = 2e-4
    maximum_scoring_seconds_per_round: float = 90.0
    maximum_peak_allocated_mib: float = 14_000.0
    maximum_pairwise_coupling: float = 0.95
    maximum_batch_rho_off: float = 4.0
    dynamic_batch: bool = False
    dynamic_score_floor_fraction: float = 0.05
    dynamic_predicted_gain_fraction: float = 0.95
    require_geometry_flattening: bool = False
    geometry_flattening_gain: float = 1e-7

    @property
    def resolution_shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution_width or self.resolution)

    @property
    def footprint_scale(self) -> tuple[float, float]:
        reference = self.footprint_reference_resolution
        if reference is None:
            return (1.0, 1.0)
        if reference <= 0:
            raise ValueError("footprint_reference_resolution must be positive")
        rows, columns = self.resolution_shape
        return (rows / reference, columns / reference)


@dataclass(frozen=True)
class ForwardRGBCell:
    direction_id: int
    direction: Tensor
    right: Tensor
    up: Tensor
    owner_ids: Tensor
    center: Tensor
    extent: float
    resolution: tuple[int, int]
    support_mask: Tensor | None = None


@dataclass
class CellRender:
    image: Tensor
    numerator: Tensor
    mass: Tensor
    radiance: Tensor
    pixels: Tensor
    valid: Tensor
    weights: Tensor
    weight_row_derivatives: Tensor
    weight_column_derivatives: Tensor
    unclipped: Tensor


@dataclass
class CorrectedState:
    active_ids: Tensor
    coefficients: Tensor
    points: Tensor
    normals: Tensor
    gradients: Tensor
    denominator: Tensor
    images: list[Tensor]
    renders: list[CellRender]
    loss: float
    root_failures: int
    optimizer_steps: int = 0
    line_search_failures: int = 0
    cg_failures: int = 0


@dataclass
class CorrectedContext:
    config: CorrectedBirthConfig
    base: GridZeroSetField
    target_field: GridZeroSetField
    master_layout: BasisLayout
    master_support: SupportPairs
    levels: Tensor
    reference_points: Tensor
    reference_normals: Tensor
    lower: Tensor
    upper: Tensor
    cells: list[ForwardRGBCell]
    target_points: Tensor
    target_normals: Tensor
    target_images: list[Tensor]
    target_displacement: Tensor
    detail_ids: Tensor
    top_detail_ids: Tensor
    geometry_evaluator: BunnyGeometryEvaluator
    visibility_report: dict[str, object]


class CorrectedLocalZeroSet:
    """Sparse local field with the exact trilinear base gradient."""

    def __init__(
        self,
        base: GridZeroSetField,
        layout: BasisLayout,
        coefficients: Tensor,
        reference_points: Tensor,
        reference_normals: Tensor,
        support: SupportPairs,
    ) -> None:
        self.base = base
        self.layout = layout
        self.coefficients = coefficients
        self.reference_points = reference_points
        self.reference_normals = reference_normals
        self.support = support

    def values_and_gradients(self, points: Tensor) -> tuple[Tensor, Tensor]:
        values = self.base.value(points)
        gradients = self.base.interpolant_gradient(points)
        point_ids = self.support.point_ids
        basis_ids = self.support.basis_ids
        offsets = points[point_ids] - self.layout.centers[basis_ids]
        radii = self.layout.radii[basis_ids]
        basis_values = wendland_values(offsets, radii)
        basis_gradients = wendland_gradients(offsets, radii)
        values.scatter_add_(
            0, point_ids, basis_values * self.coefficients[basis_ids]
        )
        gradients.scatter_add_(
            0,
            point_ids[:, None].expand(-1, 3),
            basis_gradients * self.coefficients[basis_ids, None],
        )
        return values, gradients

    def shading_gradients(self, points: Tensor) -> Tensor:
        """Continuous normal field used by the RGB appearance operator."""
        gradients = self.base.gradient(points)
        point_ids = self.support.point_ids
        basis_ids = self.support.basis_ids
        offsets = points[point_ids] - self.layout.centers[basis_ids]
        basis_gradients = wendland_gradients(
            offsets, self.layout.radii[basis_ids]
        )
        gradients.scatter_add_(
            0,
            point_ids[:, None].expand(-1, 3),
            basis_gradients * self.coefficients[basis_ids, None],
        )
        return gradients

    def deform(self, config: CorrectedBirthConfig) -> tuple[Tensor, Tensor, Tensor]:
        displacement = torch.zeros(
            self.reference_points.shape[0],
            dtype=self.reference_points.dtype,
            device=self.reference_points.device,
        )
        for _ in range(config.deformation_iterations):
            points = (
                self.reference_points
                + displacement[:, None] * self.reference_normals
            )
            values, gradients = self.values_and_gradients(points)
            denominator = (gradients * self.reference_normals).sum(1)
            safe = denominator.abs() > 1e-8
            step = torch.where(
                safe, values / denominator, torch.zeros_like(values)
            )
            displacement = (displacement - step).clamp(
                -config.deformation_max_offset,
                config.deformation_max_offset,
            )
        points = self.reference_points + displacement[:, None] * self.reference_normals
        values, gradients = self.values_and_gradients(points)
        denominator = (gradients * self.reference_normals).sum(1)
        success = (values.abs() < 1e-8) & (denominator.abs() > 1e-8)
        return points, gradients, success


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _layout(
    points: Tensor, config: CorrectedBirthConfig
) -> tuple[BasisLayout, Tensor]:
    centers = points[: config.dictionary_count].clone()
    radii = torch.empty(
        config.dictionary_count, dtype=points.dtype, device=points.device
    )
    levels = torch.empty(
        config.dictionary_count, dtype=torch.long, device=points.device
    )
    start = 0
    stop = config.initial_count
    level = 0
    while start < config.dictionary_count:
        stop = min(stop, config.dictionary_count)
        radii[start:stop] = config.base_support_radius * math.sqrt(
            config.initial_count / stop
        )
        levels[start:stop] = level
        start = stop
        stop *= 2
        level += 1
    return BasisLayout(centers, radii), levels


def _active_components(
    context: CorrectedContext, active_ids: Tensor
) -> tuple[BasisLayout, SupportPairs]:
    inverse = torch.full(
        (context.config.dictionary_count,),
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


def _visibility_cells(
    field: GridZeroSetField,
    points: Tensor,
    normals: Tensor,
    atlas: DirectionAtlas,
    boundary: ObservationSphere,
    config: CorrectedBirthConfig,
) -> tuple[list[ForwardRGBCell], dict[str, object]]:
    events = build_boundary_transport(
        field,
        points,
        normals,
        atlas,
        boundary,
        root_samples=16,
        bisection_steps=18,
    )
    event_right = atlas.right[events.direction_ids]
    event_up = atlas.up[events.direction_ids]
    source_relative = points[events.owner_ids] - boundary.center
    boundary_relative = events.boundary_positions - boundary.center
    transverse_error = torch.maximum(
        ((source_relative - boundary_relative) * event_right).sum(1).abs(),
        ((source_relative - boundary_relative) * event_up).sum(1).abs(),
    )
    per_view_outward = [
        int(((normals @ atlas.directions[index]) > 1e-8).sum())
        for index in range(config.views)
    ]
    per_view_retained = [
        int((events.direction_ids == index).sum())
        for index in range(config.views)
    ]
    cells: list[ForwardRGBCell] = []
    for direction_id in range(config.views):
        keep = events.direction_ids == direction_id
        cells.append(
            ForwardRGBCell(
                direction_id,
                atlas.directions[direction_id],
                atlas.right[direction_id],
                atlas.up[direction_id],
                events.owner_ids[keep],
                boundary.center,
                2.8,
                config.resolution_shape,
            )
        )
    return cells, {
        "emitted": events.emitted_count,
        "absorbed": events.absorbed_count,
        "retained": events.count,
        "digest": events.digest,
        "seconds": events.scene_seconds,
        "per_view_outward": per_view_outward,
        "per_view_absorbed": [
            outward - retained
            for outward, retained in zip(
                per_view_outward, per_view_retained
            )
        ],
        "per_view_retained": per_view_retained,
        "transverse_source_boundary_max_error": float(
            transverse_error.max()
        ),
    }


def _render_cell(
    points: Tensor,
    normals: Tensor,
    lower: Tensor,
    upper: Tensor,
    cell: ForwardRGBCell,
    config: CorrectedBirthConfig,
) -> CellRender:
    ids = cell.owner_ids
    rows, columns = cell.resolution
    relative = points[ids] - cell.center
    horizontal = relative @ cell.right
    vertical = relative @ cell.up
    column = (horizontal / cell.extent + 0.5) * columns - 0.5
    row = (0.5 - vertical / cell.extent) * rows - 0.5
    base_row = torch.floor(row).to(torch.long)
    base_column = torch.floor(column).to(torch.long)
    row_scale, column_scale = config.footprint_scale
    row_radius = math.ceil(2.0 * row_scale)
    column_radius = math.ceil(2.0 * column_scale)
    row_offsets = torch.arange(
        -row_radius + 1,
        row_radius + 1,
        dtype=torch.long,
        device=points.device,
    )
    column_offsets = torch.arange(
        -column_radius + 1,
        column_radius + 1,
        dtype=torch.long,
        device=points.device,
    )
    row_ids = base_row[:, None] + row_offsets[None, :]
    column_ids = base_column[:, None] + column_offsets[None, :]

    def cubic(distance: Tensor) -> tuple[Tensor, Tensor]:
        absolute = distance.abs()
        sign = torch.sign(distance)
        inner = 2.0 / 3.0 - absolute.square() + 0.5 * absolute**3
        outer = (2.0 - absolute).clamp_min(0.0) ** 3 / 6.0
        values = torch.where(absolute < 1.0, inner, outer)
        derivatives = torch.where(
            absolute < 1.0,
            sign * (-2.0 * absolute + 1.5 * absolute.square()),
            -0.5 * sign * (2.0 - absolute).clamp_min(0.0).square(),
        )
        return values, derivatives

    row_weights, row_derivatives = cubic(
        (row[:, None] - row_ids) / row_scale
    )
    row_weights /= row_scale
    row_derivatives /= row_scale * row_scale
    column_weights, column_derivatives = cubic(
        (column[:, None] - column_ids) / column_scale
    )
    column_weights /= column_scale
    column_derivatives /= column_scale * column_scale
    row_count = int(row_ids.shape[1])
    column_count = int(column_ids.shape[1])
    footprint_area = row_count * column_count
    pixel_rows = row_ids[:, :, None].expand(
        -1, row_count, column_count
    ).reshape(-1, footprint_area)
    pixel_columns = column_ids[:, None, :].expand(
        -1, row_count, column_count
    ).reshape(-1, footprint_area)
    weights = (
        row_weights[:, :, None] * column_weights[:, None, :]
    ).reshape(-1, footprint_area)
    weight_row_derivatives = (
        row_derivatives[:, :, None] * column_weights[:, None, :]
    ).reshape(-1, footprint_area)
    weight_column_derivatives = (
        row_weights[:, :, None] * column_derivatives[:, None, :]
    ).reshape(-1, footprint_area)
    valid = (
        (pixel_rows >= 0)
        & (pixel_rows < rows)
        & (pixel_columns >= 0)
        & (pixel_columns < columns)
    )
    pixels = (
        pixel_rows.clamp(0, rows - 1) * columns
        + pixel_columns.clamp(0, columns - 1)
    )
    weights = weights * valid
    weight_row_derivatives *= valid
    weight_column_derivatives *= valid
    colors = meshfree_base_color(points[ids], lower, upper)
    cosine = (normals[ids] @ cell.direction).clamp_min(0.0)
    lobe = config.ambient + (1.0 - config.ambient) * cosine
    radiance = colors * lobe[:, None]
    mass = torch.zeros(rows * columns, dtype=points.dtype, device=points.device)
    numerator = torch.zeros(
        (rows * columns, 3), dtype=points.dtype, device=points.device
    )
    mass.scatter_add_(0, pixels.reshape(-1), weights.reshape(-1))
    numerator.index_add_(
        0,
        pixels.reshape(-1),
        (weights[..., None] * radiance[:, None, :]).reshape(-1, 3),
    )
    raw = config.sensor_gain * numerator / mass[:, None].clamp_min(1e-30)
    support_mask = (
        cell.support_mask
        if cell.support_mask is not None
        else (
            mass
            >= config.support_threshold / (row_scale * column_scale)
        )
    )
    unclipped = (raw > 0.0) & (raw < 1.0) & support_mask[:, None]
    image = torch.where(
        support_mask[:, None],
        raw.clamp(0.0, 1.0),
        torch.zeros_like(raw),
    )
    return CellRender(
        image.reshape(-1),
        numerator,
        mass,
        radiance,
        pixels,
        valid,
        weights,
        weight_row_derivatives,
        weight_column_derivatives,
        unclipped,
    )


def _render(
    points: Tensor,
    normals: Tensor,
    context: CorrectedContext,
    cells: list[ForwardRGBCell] | None = None,
) -> tuple[list[Tensor], list[CellRender]]:
    renders = [
        _render_cell(
            points,
            normals,
            context.lower,
            context.upper,
            cell,
            context.config,
        )
        for cell in (cells or context.cells)
    ]
    return [item.image for item in renders], renders


def _loss(images: list[Tensor], targets: list[Tensor]) -> float:
    return 0.5 * sum(
        float((image - target).square().sum())
        for image, target in zip(images, targets)
    )


def _evaluate(
    context: CorrectedContext,
    active_ids: Tensor,
    coefficients: Tensor,
    components: tuple[BasisLayout, SupportPairs] | None = None,
) -> CorrectedState:
    layout, support = components or _active_components(context, active_ids)
    model = CorrectedLocalZeroSet(
        context.base,
        layout,
        coefficients,
        context.reference_points,
        context.reference_normals,
        support,
    )
    points, implicit_gradients, success = model.deform(context.config)
    shading_gradients = model.shading_gradients(points)
    norms = torch.linalg.vector_norm(shading_gradients, dim=1)
    normals = shading_gradients / norms[:, None].clamp_min(1e-30)
    denominator = (implicit_gradients * context.reference_normals).sum(1)
    images, renders = _render(points, normals, context)
    return CorrectedState(
        active_ids,
        coefficients,
        points,
        normals,
        shading_gradients,
        denominator,
        images,
        renders,
        _loss(images, context.target_images),
        int((~success).sum()),
    )


def _directional_gradient_derivative(
    model: CorrectedLocalZeroSet,
    points: Tensor,
    reference_normals: Tensor,
    epsilon: float = 2e-4,
) -> Tensor:
    plus = model.shading_gradients(points + epsilon * reference_normals)
    minus = model.shading_gradients(points - epsilon * reference_normals)
    return (plus - minus) / (2.0 * epsilon)


def _sparse_jacobian(
    context: CorrectedContext,
    state: CorrectedState,
    layout: BasisLayout,
    support: SupportPairs,
    cell: ForwardRGBCell,
    render: CellRender,
    directional_hessian: Tensor,
    threshold: float = 1e-12,
) -> Tensor:
    local_events, basis_ids = support.subset_points(cell.owner_ids)
    point_ids = cell.owner_ids[local_events]
    offsets = state.points[point_ids] - layout.centers[basis_ids]
    basis_values = wendland_values(offsets, layout.radii[basis_ids])
    inside = basis_values > 0.0
    local_events = local_events[inside]
    basis_ids = basis_ids[inside]
    point_ids = point_ids[inside]
    offsets = offsets[inside]
    basis_values = basis_values[inside]
    basis_gradient = wendland_gradients(offsets, layout.radii[basis_ids])
    safe_denominator = torch.where(
        state.denominator[point_ids].abs() > 1e-8,
        state.denominator[point_ids],
        torch.full_like(state.denominator[point_ids], 1e-8),
    )
    ds = -basis_values / safe_denominator
    dp = context.reference_normals[point_ids] * ds[:, None]
    gradient_derivative = (
        basis_gradient + directional_hessian[point_ids] * ds[:, None]
    )
    normals = state.normals[point_ids]
    normal_derivative = (
        gradient_derivative
        - normals * (normals * gradient_derivative).sum(1, keepdim=True)
    ) / torch.linalg.vector_norm(
        state.gradients[point_ids], dim=1, keepdim=True
    ).clamp_min(1e-30)

    extent = context.upper - context.lower
    color_derivative = torch.stack(
        (
            0.72 / extent[0] * dp[:, 0],
            0.70 / extent[1] * dp[:, 1],
            -0.62 / extent[2] * dp[:, 2],
        ),
        1,
    )
    point_colors = meshfree_base_color(
        state.points[point_ids], context.lower, context.upper
    )
    raw_cosine = state.normals[point_ids] @ cell.direction
    cosine = raw_cosine.clamp_min(0.0)
    lobe = context.config.ambient + (1.0 - context.config.ambient) * cosine
    dcosine = torch.where(
        raw_cosine > 0.0,
        normal_derivative @ cell.direction,
        torch.zeros_like(raw_cosine),
    )
    dradiance = (
        color_derivative * lobe[:, None]
        + point_colors
        * ((1.0 - context.config.ambient) * dcosine)[:, None]
    )

    rows, columns = cell.resolution
    drow = -rows / cell.extent * (dp @ cell.up)
    dcolumn = columns / cell.extent * (dp @ cell.right)
    dweights = (
        render.weight_row_derivatives[local_events] * drow[:, None]
        + render.weight_column_derivatives[local_events] * dcolumn[:, None]
    )
    weights = render.weights[local_events]
    pixels = render.pixels[local_events]
    radiance = render.radiance[local_events]
    dnumerator = (
        dweights[..., None] * radiance[:, None, :]
        + weights[..., None] * dradiance[:, None, :]
    )
    dmass = dweights
    pixel_mass = render.mass[pixels].clamp_min(1e-30)
    pixel_numerator = render.numerator[pixels]
    derivative = context.config.sensor_gain * (
        dnumerator * pixel_mass[..., None]
        - pixel_numerator * dmass[..., None]
    ) / pixel_mass[..., None].square()
    derivative *= render.unclipped[pixels]
    channels = torch.arange(3, device=state.points.device)
    output_rows = (
        pixels[..., None] * 3 + channels[None, None, :]
    ).expand_as(derivative)
    parameter_columns = basis_ids[:, None, None].expand_as(output_rows)
    touched = render.valid[local_events, :, None] & (derivative.abs() > threshold)
    return torch.sparse_coo_tensor(
        torch.stack((output_rows[touched], parameter_columns[touched])),
        derivative[touched],
        size=(3 * rows * columns, layout.count),
        dtype=state.points.dtype,
        device=state.points.device,
    ).coalesce()


def _jacobians(
    context: CorrectedContext,
    state: CorrectedState,
    layout: BasisLayout,
    support: SupportPairs,
) -> list[Tensor]:
    model = CorrectedLocalZeroSet(
        context.base,
        layout,
        state.coefficients,
        context.reference_points,
        context.reference_normals,
        support,
    )
    directional_hessian = _directional_gradient_derivative(
        model, state.points, context.reference_normals
    )
    return [
        _sparse_jacobian(
            context,
            state,
            layout,
            support,
            cell,
            render,
            directional_hessian,
        )
        for cell, render in zip(context.cells, state.renders)
    ]


def _score_candidates(
    context: CorrectedContext, state: CorrectedState
) -> tuple[dict[str, Tensor], dict[str, object], float]:
    _sync(state.points.device)
    started = time.perf_counter()
    # Hessian belongs to the current active field, while candidate direct
    # gradients come from the full master layout.
    active_layout, active_support = _active_components(context, state.active_ids)
    active_model = CorrectedLocalZeroSet(
        context.base,
        active_layout,
        state.coefficients,
        context.reference_points,
        context.reference_normals,
        active_support,
    )
    directional_hessian = _directional_gradient_derivative(
        active_model, state.points, context.reference_normals
    )
    count = context.config.dictionary_count
    alignment = torch.zeros(count, dtype=state.points.dtype, device=state.points.device)
    norm_squared = torch.zeros_like(alignment)
    affected_pixels = torch.zeros(count, dtype=torch.long, device=state.points.device)
    affected_views = torch.zeros(count, dtype=torch.long, device=state.points.device)
    nnz = 0
    for cell, render, image, target in zip(
        context.cells, state.renders, state.images, context.target_images
    ):
        matrix = _sparse_jacobian(
            context,
            state,
            context.master_layout,
            context.master_support,
            cell,
            render,
            directional_hessian,
        )
        indices = matrix.indices()
        values = matrix.values()
        residual = image - target
        alignment.scatter_add_(0, indices[1], values * residual[indices[0]])
        norm_squared.scatter_add_(0, indices[1], values.square())
        encoded = torch.unique(
            torch.div(indices[0], 3, rounding_mode="floor") * count
            + indices[1]
        )
        columns = encoded % count
        affected_pixels += torch.bincount(columns, minlength=count)
        responsive = torch.bincount(indices[1], minlength=count) > 0
        affected_views += responsive.long()
        nnz += matrix._nnz()
        del matrix
    scores = {
        "raw": alignment.abs(),
        "quadratic": 0.5 * alignment.square() / (
            norm_squared + context.config.damping
        ),
        "jacobian_norm": torch.sqrt(norm_squared),
        "alignment": alignment,
        "norm_squared": norm_squared,
    }
    inactive = torch.ones(count, dtype=torch.bool, device=state.points.device)
    inactive[state.active_ids] = False
    responsive = inactive & (scores["jacobian_norm"] > 1e-12)
    maximum_norm = float(scores["jacobian_norm"][inactive].max())
    _sync(state.points.device)
    seconds = time.perf_counter() - started
    return scores, {
        "candidate_jacobian_nnz": nnz,
        "responsive_candidate_fraction": float(
            responsive.sum() / max(int(inactive.sum()), 1)
        ),
        "near_null_candidate_fraction": float(
            (
                inactive
                & (scores["jacobian_norm"] <= 1e-8 * maximum_norm)
            ).sum()
            / max(int(inactive.sum()), 1)
        ),
        "nonfinite_candidate_score_count": int(
            (~torch.isfinite(scores["quadratic"][inactive])).sum()
        ),
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
        "best_remaining_quadratic": float(scores["quadratic"][inactive].max()),
    }, seconds


def _normal_matvec(matrices: list[Tensor], vector: Tensor, damping: float) -> Tensor:
    result = damping * vector
    for matrix in matrices:
        projected = torch.sparse.mm(matrix, vector[:, None])
        result += torch.sparse.mm(matrix.transpose(0, 1), projected).flatten()
    return result


def _cg(
    matrices: list[Tensor], right: Tensor, config: CorrectedBirthConfig
) -> tuple[Tensor, int]:
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    residual_squared = residual @ residual
    iterations = 0
    for _ in range(config.cg_iterations):
        iterations += 1
        product = _normal_matvec(matrices, direction, config.damping)
        denominator = direction @ product
        if abs(float(denominator)) < 1e-30:
            break
        alpha = residual_squared / denominator
        solution += alpha * direction
        next_residual = residual - alpha * product
        next_squared = next_residual @ next_residual
        if float(torch.sqrt(next_squared)) < 1e-10:
            break
        direction = next_residual + next_squared / residual_squared * direction
        residual = next_residual
        residual_squared = next_squared
    return solution, iterations


def _optimize(
    context: CorrectedContext,
    state: CorrectedState,
    steps: int,
) -> tuple[CorrectedState, float]:
    started = time.perf_counter()
    coefficients = state.coefficients.clone()
    components = _active_components(context, state.active_ids)
    result = state
    completed = 0
    for _ in range(steps):
        result = _evaluate(
            context, state.active_ids, coefficients, components
        )
        matrices = _jacobians(context, result, *components)
        gradient = torch.zeros_like(coefficients)
        for matrix, image, target in zip(
            matrices, result.images, context.target_images
        ):
            gradient += torch.sparse.mm(
                matrix.transpose(0, 1), (image - target)[:, None]
            ).flatten()
        step, _ = _cg(matrices, -gradient, context.config)
        del matrices
        if not bool(torch.isfinite(step).all()):
            result.cg_failures += 1
            break
        accepted = result
        accepted_coefficients = coefficients
        for trial in range(context.config.line_evaluations):
            scale = 0.5**trial
            proposal = (coefficients + scale * step).clamp(
                -context.config.coefficient_limit,
                context.config.coefficient_limit,
            )
            candidate = _evaluate(
                context, state.active_ids, proposal, components
            )
            if candidate.root_failures == 0 and candidate.loss < accepted.loss:
                accepted = candidate
                accepted_coefficients = proposal
                break
        completed += 1
        if accepted.loss >= result.loss:
            result.line_search_failures += 1
            break
        relative = (result.loss - accepted.loss) / max(result.loss, 1e-30)
        coefficients = accepted_coefficients
        result = accepted
        if relative < 1e-7:
            break
    result.optimizer_steps = state.optimizer_steps + completed
    result.line_search_failures += state.line_search_failures
    result.cg_failures += state.cg_failures
    _sync(state.points.device)
    return result, time.perf_counter() - started


def _batch_size(active_count: int) -> int:
    if active_count < 256:
        return 32
    if active_count < 512:
        return 64
    return 128


def _select_observation_batch(
    context: CorrectedContext,
    state: CorrectedState,
    scores: dict[str, Tensor],
) -> Tensor:
    inactive = torch.ones(
        context.config.dictionary_count,
        dtype=torch.bool,
        device=state.points.device,
    )
    inactive[state.active_ids] = False
    values = torch.where(
        inactive,
        scores["quadratic"],
        torch.full_like(scores["quadratic"], -math.inf),
    )
    order = torch.argsort(values, descending=True, stable=True)
    desired = min(_batch_size(state.active_ids.numel()), int(inactive.sum()))
    best = float(values[order[0]])
    selected: list[int] = []
    for item in order:
        candidate = int(item)
        score = float(values[candidate])
        if not math.isfinite(score) or score < 0.05 * best:
            break
        if selected:
            old = torch.tensor(selected, device=item.device)
            distances = torch.linalg.vector_norm(
                context.master_layout.centers[old]
                - context.master_layout.centers[candidate],
                dim=1,
            )
            separation = 0.20 * torch.minimum(
                context.master_layout.radii[old],
                context.master_layout.radii[candidate],
            )
            if bool((distances < separation).any()):
                continue
        selected.append(candidate)
        if len(selected) == desired:
            break
    if not selected:
        selected.append(int(order[0]))
    return torch.tensor(selected, dtype=torch.long, device=state.points.device)


def _select_dynamic_observation_batch(
    context: CorrectedContext,
    state: CorrectedState,
    scores: dict[str, Tensor],
    score_key: str = "quadratic",
) -> tuple[Tensor, dict[str, object]]:
    inactive = torch.ones(
        context.config.dictionary_count,
        dtype=torch.bool,
        device=state.points.device,
    )
    inactive[state.active_ids] = False
    values = torch.where(
        inactive,
        scores[score_key],
        torch.full_like(scores[score_key], -math.inf),
    )
    order = torch.argsort(values, descending=True, stable=True)
    best = float(values[order[0]])
    floor = context.config.dynamic_score_floor_fraction * best
    eligible = order[torch.isfinite(values[order]) & (values[order] >= floor)]
    if eligible.numel() == 0:
        eligible = order[:1]
    eligible_scores = values[eligible].clamp_min(0.0)
    total_gain = eligible_scores.sum()
    if float(total_gain) > 0.0:
        cumulative = torch.cumsum(eligible_scores, 0)
        saturation = context.config.dynamic_predicted_gain_fraction * total_gain
        desired = int(torch.searchsorted(cumulative, saturation).item()) + 1
    else:
        desired = 1
    p_max_raw = 100 * int(state.active_ids.numel())
    desired = min(desired, p_max_raw, int(inactive.sum()))
    selected: list[int] = []
    spatial_rejections = 0
    for item in eligible:
        candidate = int(item)
        if selected:
            previous = torch.tensor(selected, device=item.device)
            distances = torch.linalg.vector_norm(
                context.master_layout.centers[previous]
                - context.master_layout.centers[candidate],
                dim=1,
            )
            separation = 0.20 * torch.minimum(
                context.master_layout.radii[previous],
                context.master_layout.radii[candidate],
            )
            if bool((distances < separation).any()):
                spatial_rejections += 1
                continue
        selected.append(candidate)
        if len(selected) == desired:
            break
    if not selected:
        selected.append(int(order[0]))
    selected_tensor = torch.tensor(
        selected, dtype=torch.long, device=state.points.device
    )
    selected_gain = float(values[selected_tensor].clamp_min(0.0).sum())
    return selected_tensor, {
        "dynamic_p_max_raw": p_max_raw,
        "dynamic_eligible_candidates": int(eligible.numel()),
        "dynamic_saturation_target_size": desired,
        "dynamic_score_floor": floor,
        "dynamic_score_floor_fraction": (
            context.config.dynamic_score_floor_fraction
        ),
        "dynamic_score_policy": score_key,
        "dynamic_predicted_gain_fraction": (
            context.config.dynamic_predicted_gain_fraction
        ),
        "dynamic_selected_gain_fraction_of_eligible": selected_gain
        / max(float(total_gain), 1e-30),
        "dynamic_spatial_rejections": spatial_rejections,
        "dynamic_selection_limiter": (
            "p_max"
            if desired == p_max_raw
            else (
                "dictionary"
                if desired == int(inactive.sum())
                else "predicted_gain_saturation"
            )
        ),
    }


def _coupling_statistics(gram: Tensor) -> dict[str, float]:
    if gram.shape[0] < 2:
        return {
            "rho_off": 0.0,
            "pairwise_mean": 0.0,
            "pairwise_max": 0.0,
        }
    diagonal = torch.diagonal(gram).clamp_min(0.0)
    off_diagonal = gram - torch.diag(diagonal)
    norms = torch.sqrt(diagonal)
    cosine = off_diagonal.abs() / (
        norms[:, None] * norms[None, :]
    ).clamp_min(1e-30)
    upper_indices = torch.triu_indices(
        gram.shape[0], gram.shape[0], offset=1, device=gram.device
    )
    upper = cosine[upper_indices[0], upper_indices[1]]
    return {
        "rho_off": float(
            torch.linalg.vector_norm(off_diagonal)
            / torch.linalg.vector_norm(diagonal).clamp_min(1e-30)
        ),
        "pairwise_mean": float(upper.mean()),
        "pairwise_max": float(upper.max()),
    }


def _batch_gram(
    context: CorrectedContext,
    state: CorrectedState,
    selected: Tensor,
) -> tuple[Tensor, float]:
    _sync(state.points.device)
    started = time.perf_counter()
    layout, support = _active_components(context, selected)
    active_layout, active_support = _active_components(context, state.active_ids)
    active_model = CorrectedLocalZeroSet(
        context.base,
        active_layout,
        state.coefficients,
        context.reference_points,
        context.reference_normals,
        active_support,
    )
    directional_hessian = _directional_gradient_derivative(
        active_model, state.points, context.reference_normals
    )
    gram = torch.zeros(
        (selected.numel(), selected.numel()),
        dtype=state.points.dtype,
        device=state.points.device,
    )
    for cell, render in zip(context.cells, state.renders):
        matrix = _sparse_jacobian(
            context,
            state,
            layout,
            support,
            cell,
            render,
            directional_hessian,
        )
        product = torch.sparse.mm(matrix.transpose(0, 1), matrix)
        gram += product.to_dense() if product.is_sparse else product
    _sync(state.points.device)
    return gram, time.perf_counter() - started


def _coupling_checked_batch(
    context: CorrectedContext,
    state: CorrectedState,
    selected: Tensor,
) -> tuple[Tensor, Tensor, dict[str, object], float]:
    initial_count = int(selected.numel())
    gram, seconds = _batch_gram(context, state, selected)
    diagonal = torch.diagonal(gram).clamp_min(0.0)
    keep_list: list[int] = []
    pairwise_rejections = 0
    rho_rejections = 0
    diagonal_norm_squared = 0.0
    off_diagonal_norm_squared = 0.0
    for candidate in range(initial_count):
        if keep_list:
            previous = torch.tensor(keep_list, device=selected.device)
            correlations = gram[candidate, previous].abs() / torch.sqrt(
                (diagonal[candidate] * diagonal[previous]).clamp_min(1e-30)
            )
            if (
                float(correlations.max())
                > context.config.maximum_pairwise_coupling
            ):
                pairwise_rejections += 1
                continue
            proposed_off = off_diagonal_norm_squared + 2.0 * float(
                gram[candidate, previous].square().sum()
            )
        else:
            proposed_off = 0.0
        proposed_diagonal = diagonal_norm_squared + float(
            diagonal[candidate].square()
        )
        proposed_rho = math.sqrt(proposed_off) / max(
            math.sqrt(proposed_diagonal), 1e-30
        )
        if proposed_rho > context.config.maximum_batch_rho_off:
            rho_rejections += 1
            continue
        keep_list.append(candidate)
        diagonal_norm_squared = proposed_diagonal
        off_diagonal_norm_squared = proposed_off
    if not keep_list:
        keep_list.append(0)
    keep = torch.tensor(keep_list, dtype=torch.long, device=selected.device)
    selected = selected[keep]
    gram = gram[keep][:, keep]
    return selected, gram, {
        "coupling_candidates_before_pruning": initial_count,
        "coupling_rejections": pairwise_rejections + rho_rejections,
        "pairwise_coupling_rejections": pairwise_rejections,
        "rho_off_coupling_rejections": rho_rejections,
        **_coupling_statistics(gram),
    }, seconds


def _joint_prediction(
    scores: dict[str, Tensor], selected: Tensor, gram: Tensor, damping: float
) -> float:
    gradient = scores["alignment"][selected]
    hessian = gram + damping * torch.eye(
        selected.numel(), dtype=gram.dtype, device=gram.device
    )
    try:
        solution = torch.linalg.solve(hessian, gradient)
    except RuntimeError:
        solution = torch.linalg.pinv(hessian) @ gradient
    return 0.5 * float(gradient @ solution)


def _select_baseline_batch(
    context: CorrectedContext,
    active_ids: Tensor,
    count: int,
    method: str,
    generator: torch.Generator,
) -> Tensor:
    inactive = torch.ones(
        context.config.dictionary_count,
        dtype=torch.bool,
        device=active_ids.device,
    )
    inactive[active_ids] = False
    ids = torch.nonzero(inactive, as_tuple=False).flatten()
    if method == "uniform_matched":
        return ids[:count]
    permutation = torch.randperm(ids.numel(), generator=generator)[:count]
    return ids[permutation.to(ids.device)]


def _metrics(
    context: CorrectedContext,
    state: CorrectedState,
    method: str,
    round_index: int,
    scoring_seconds: float,
    optimization_seconds: float,
    total_seconds: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    config = context.config
    rows, columns = config.resolution_shape
    squared_error = 0.0
    absolute_error = 0.0
    intersection_count = 0
    union_count = 0
    image_support_count = 0
    target_support_count = 0
    for image, target in zip(state.images, context.target_images):
        difference = image - target
        squared_error += float(difference.square().sum())
        absolute_error += float(difference.abs().sum())
        image_mask = image.reshape(-1, 3).abs().amax(1) > 0.0
        target_mask = target.reshape(-1, 3).abs().amax(1) > 0.0
        intersection_count += int((image_mask & target_mask).sum())
        union_count += int((image_mask | target_mask).sum())
        image_support_count += int(image_mask.sum())
        target_support_count += int(target_mask.sum())
    scalar_count = config.views * rows * columns * 3
    rgb_mse = squared_error / scalar_count
    precision = intersection_count / max(image_support_count, 1)
    recall = intersection_count / max(target_support_count, 1)
    result: dict[str, object] = {
        "method": method,
        "birth_round": round_index,
        "active_dofs": int(state.active_ids.numel()),
        "image_loss": state.loss,
        "normalized_rgb_mse": 2.0
        * state.loss
        / (config.views * rows * columns * 3),
        "rgb_mae": absolute_error / scalar_count,
        "rgb_psnr": -10.0 * math.log10(max(rgb_mse, 1e-30)),
        "silhouette_iou": intersection_count / max(union_count, 1),
        "silhouette_f1": 2.0
        * precision
        * recall
        / max(precision + recall, 1e-30),
        "target_point_rms": float(
            torch.sqrt(
                (state.points - context.target_points).square().sum(1).mean()
            )
        ),
        "root_failures": state.root_failures,
        "line_search_failures": state.line_search_failures,
        "cg_failures": state.cg_failures,
        "optimizer_steps": state.optimizer_steps,
        "scoring_seconds": scoring_seconds,
        "optimization_seconds": optimization_seconds,
        "total_seconds": total_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    result.update(context.geometry_evaluator(state.points))
    if extra:
        result.update(extra)
    return result


def _run_observation_trajectory(
    context: CorrectedContext,
    score_policy: str = "quadratic",
) -> tuple[dict[str, object], CorrectedState]:
    if score_policy not in {"quadratic", "raw"}:
        raise ValueError("score_policy must be quadratic or raw")
    method = (
        "observation_driven"
        if score_policy == "quadratic"
        else "raw_alignment_dynamic"
    )
    device = context.reference_points.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    active = torch.arange(context.config.initial_count, device=device)
    state = _evaluate(
        context,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device=device),
    )
    state, optimization = _optimize(
        context, state, context.config.initial_optimization_steps
    )
    rows = [
        _metrics(
            context,
            state,
            method,
            0,
            0.0,
            optimization,
            time.perf_counter() - started,
        )
    ]
    batches: list[dict[str, object]] = []
    scoring = 0.0
    gram_seconds_total = 0.0
    relative_gains: list[float] = []
    stop_reason = "DICTIONARY_EXHAUSTED"
    while state.active_ids.numel() < context.config.dictionary_count:
        scores, diagnostics, score_seconds = _score_candidates(context, state)
        if context.config.dynamic_batch:
            selected, selection_diagnostics = (
                _select_dynamic_observation_batch(
                    context, state, scores, score_policy
                )
            )
        else:
            if score_policy != "quadratic":
                raise ValueError("raw policy requires dynamic_batch=True")
            selected = _select_observation_batch(context, state, scores)
            selection_diagnostics = {
                "dynamic_selection_limiter": "legacy_schedule",
                "dynamic_saturation_target_size": int(selected.numel()),
            }
        selected, gram, coupling, gram_seconds = _coupling_checked_batch(
            context, state, selected
        )
        score_seconds += gram_seconds
        scoring += score_seconds
        gram_seconds_total += gram_seconds
        predicted_independent = float(scores["quadratic"][selected].sum())
        predicted = _joint_prediction(
            scores, selected, gram, context.config.damping
        )
        raw = float(scores["raw"][selected].sum())
        predicted_policy_score = predicted if score_policy == "quadratic" else raw
        previous_state = state
        before = state.loss
        previous_chamfer = float(rows[-1]["symmetric_chamfer"])
        born = CorrectedState(
            torch.cat((state.active_ids, selected)),
            torch.cat(
                (
                    state.coefficients,
                    state.coefficients.new_zeros(selected.numel()),
                )
            ),
            state.points,
            state.normals,
            state.gradients,
            state.denominator,
            state.images,
            state.renders,
            state.loss,
            state.root_failures,
            state.optimizer_steps,
            state.line_search_failures,
            state.cg_failures,
        )
        at_birth = _evaluate(context, born.active_ids, born.coefficients)
        at_birth.optimizer_steps = state.optimizer_steps
        at_birth.line_search_failures = state.line_search_failures
        at_birth.cg_failures = state.cg_failures
        birth_only_geometry_jump = float(
            torch.linalg.vector_norm(
                at_birth.points - previous_state.points, dim=1
            ).max()
        )
        # Full-HD states retain dense RGB/numerator/mass buffers per view.
        # Once the exact zero-coefficient birth jump and counters are recorded,
        # the pre-birth state must be released before line-search proposals.
        del previous_state, born, state
        torch.cuda.empty_cache()
        state, optimize_seconds = _optimize(
            context, at_birth, context.config.post_birth_steps
        )
        del at_birth
        optimization += optimize_seconds
        realized = max(0.0, before - state.loss)
        relative_gain = realized / max(before, 1e-30)
        relative_gains.append(relative_gain)
        current_metrics = _metrics(
            context,
            state,
            method,
            len(batches) + 1,
            scoring,
            optimization,
            time.perf_counter() - started,
        )
        geometry_gain = previous_chamfer - float(
            current_metrics["symmetric_chamfer"]
        )
        batch = {
            "birth_round": len(batches) + 1,
            "active_before": int(state.active_ids.numel() - selected.numel()),
            "active_after": int(state.active_ids.numel()),
            "batch_size": int(selected.numel()),
            "selected_ids": selected.detach().cpu().tolist(),
            "predicted_quadratic_gain": predicted_independent,
            "predicted_joint_gain": predicted,
            "joint_to_independent_ratio": predicted
            / max(predicted_independent, 1e-30),
            "predicted_raw_alignment": raw,
            "realized_gain": realized,
            "relative_realized_gain": relative_gain,
            "realized_geometry_gain": geometry_gain,
            "birth_only_geometry_jump": birth_only_geometry_jump,
            "top25_error_region_fraction": float(
                torch.isin(selected, context.detail_ids).double().mean()
            ),
            "top10_error_region_fraction": float(
                torch.isin(selected, context.top_detail_ids).double().mean()
            ),
            **selection_diagnostics,
            **coupling,
            **diagnostics,
            "round_scoring_seconds": score_seconds,
            "round_optimization_seconds": optimize_seconds,
        }
        batches.append(batch)
        current_metrics.update(batch)
        rows.append(current_metrics)
        print(
            json.dumps(
                {
                    "phase": "corrected_birth",
                    "score_policy": score_policy,
                    "round": len(batches),
                    "active": int(state.active_ids.numel()),
                    "loss": state.loss,
                    "relative_gain": relative_gain,
                    "score_seconds": score_seconds,
                }
            ),
            flush=True,
        )
        enough = (
            state.active_ids.numel()
            >= context.config.minimum_active_before_stopping
        )
        window = relative_gains[-context.config.flattening_window :]
        geometry_window = [
            abs(float(item["realized_geometry_gain"]))
            for item in batches[-context.config.flattening_window :]
        ]
        geometry_flat = (
            not context.config.require_geometry_flattening
            or (
                len(geometry_window) == context.config.flattening_window
                and max(geometry_window)
                < context.config.geometry_flattening_gain
            )
        )
        if (
            enough
            and len(window) == context.config.flattening_window
            and max(window) < context.config.flattening_relative_gain
            and geometry_flat
        ):
            stop_reason = (
                "MARGINAL_IMAGE_AND_GEOMETRY_GAIN_FLATTENED"
                if context.config.require_geometry_flattening
                else "MARGINAL_IMAGE_GAIN_FLATTENED"
            )
            break
        if enough and score_seconds > context.config.maximum_scoring_seconds_per_round:
            stop_reason = "CANDIDATE_SCORING_RUNTIME_PROHIBITIVE"
            break
        if (
            enough
            and torch.cuda.max_memory_allocated() / 2**20
            > context.config.maximum_peak_allocated_mib
        ):
            stop_reason = "VRAM_PROHIBITIVE"
            break
        if state.root_failures or state.cg_failures:
            stop_reason = "NUMERICAL_FAILURE"
            break
        if int(selected.numel()) == 1 and predicted_policy_score <= 1e-12:
            stop_reason = "NO_RESPONSIVE_CANDIDATE"
            break
    predicted = torch.tensor(
        [
            float(
                item[
                    "predicted_joint_gain"
                    if score_policy == "quadratic"
                    else "predicted_raw_alignment"
                ]
            )
            for item in batches
        ]
    )
    realized = torch.tensor([float(item["realized_gain"]) for item in batches])
    all_selected = torch.tensor(
        [candidate for item in batches for candidate in item["selected_ids"]],
        dtype=torch.long,
        device=device,
    )
    report = {
        "method": method,
        "score_policy": score_policy,
        "rows": rows,
        "batches": batches,
        "batch_schedule": [int(item["batch_size"]) for item in batches],
        "birth_rounds": len(batches),
        "mean_batch_size": statistics.mean(
            int(item["batch_size"]) for item in batches
        ),
        "median_batch_size": statistics.median(
            int(item["batch_size"]) for item in batches
        ),
        "maximum_batch_size": max(int(item["batch_size"]) for item in batches),
        "top25_error_region_birth_fraction": float(
            torch.isin(all_selected, context.detail_ids).double().mean()
        ),
        "top10_error_region_birth_fraction": float(
            torch.isin(all_selected, context.top_detail_ids).double().mean()
        ),
        "maximum_pairwise_coupling": max(
            float(item["pairwise_max"]) for item in batches
        ),
        "maximum_batch_rho_off": max(
            float(item["rho_off"]) for item in batches
        ),
        "maximum_birth_only_geometry_jump": max(
            float(item["birth_only_geometry_jump"]) for item in batches
        ),
        "predicted_realized_spearman": _correlation(predicted, realized, True),
        "predicted_realized_pearson": _correlation(predicted, realized, False),
        "cumulative_scoring_seconds": scoring,
        "cumulative_batch_gram_seconds": gram_seconds_total,
        "cumulative_optimization_seconds": optimization,
        "total_seconds": time.perf_counter() - started,
        "stopping_reason": stop_reason,
        "maximum_validated_active_dofs": int(state.active_ids.numel()),
        "final_selected_ids": state.active_ids.detach().cpu().tolist(),
    }
    return report, state


def _run_matched_baseline(
    context: CorrectedContext,
    schedule: list[int],
    method: str,
) -> tuple[dict[str, object], CorrectedState]:
    device = context.reference_points.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(1707)
    active = torch.arange(context.config.initial_count, device=device)
    state = _evaluate(
        context,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device=device),
    )
    state, optimization = _optimize(
        context, state, context.config.initial_optimization_steps
    )
    rows = [
        _metrics(
            context,
            state,
            method,
            0,
            0.0,
            optimization,
            time.perf_counter() - started,
        )
    ]
    for round_index, count in enumerate(schedule, start=1):
        selected = _select_baseline_batch(
            context, state.active_ids, count, method, generator
        )
        before = state.loss
        state = _evaluate(
            context,
            torch.cat((state.active_ids, selected)),
            torch.cat((state.coefficients, state.coefficients.new_zeros(count))),
        )
        state, seconds = _optimize(
            context, state, context.config.post_birth_steps
        )
        optimization += seconds
        rows.append(
            _metrics(
                context,
                state,
                method,
                round_index,
                0.0,
                optimization,
                time.perf_counter() - started,
                {
                    "batch_size": count,
                    "realized_gain": max(0.0, before - state.loss),
                },
            )
        )
    return {
        "method": method,
        "rows": rows,
        "birth_rounds": len(schedule),
        "batch_schedule": schedule,
        "total_seconds": time.perf_counter() - started,
        "cumulative_optimization_seconds": optimization,
        "maximum_validated_active_dofs": int(state.active_ids.numel()),
    }, state


def _run_raw_baseline(
    context: CorrectedContext,
    schedule: list[int],
) -> tuple[dict[str, object], CorrectedState]:
    device = context.reference_points.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    active = torch.arange(context.config.initial_count, device=device)
    state = _evaluate(
        context,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device=device),
    )
    state, optimization = _optimize(
        context, state, context.config.initial_optimization_steps
    )
    scoring = 0.0
    batches: list[dict[str, object]] = []
    rows = [
        _metrics(
            context,
            state,
            "raw_alignment_matched",
            0,
            0.0,
            optimization,
            time.perf_counter() - started,
        )
    ]
    for round_index, count in enumerate(schedule, start=1):
        scores, diagnostics, seconds = _score_candidates(context, state)
        scoring += seconds
        inactive = torch.ones(
            context.config.dictionary_count, dtype=torch.bool, device=device
        )
        inactive[state.active_ids] = False
        values = torch.where(
            inactive,
            scores["raw"],
            torch.full_like(scores["raw"], -math.inf),
        )
        selected = torch.argsort(values, descending=True, stable=True)[:count]
        predicted_raw = float(scores["raw"][selected].sum())
        before = state.loss
        state = _evaluate(
            context,
            torch.cat((state.active_ids, selected)),
            torch.cat(
                (state.coefficients, state.coefficients.new_zeros(count))
            ),
        )
        state, optimize_seconds = _optimize(
            context, state, context.config.post_birth_steps
        )
        optimization += optimize_seconds
        realized = max(0.0, before - state.loss)
        batch = {
            "birth_round": round_index,
            "batch_size": count,
            "selected_ids": selected.detach().cpu().tolist(),
            "predicted_raw_alignment": predicted_raw,
            "realized_gain": realized,
            **diagnostics,
        }
        batches.append(batch)
        rows.append(
            _metrics(
                context,
                state,
                "raw_alignment_matched",
                round_index,
                scoring,
                optimization,
                time.perf_counter() - started,
                {
                    **batch,
                },
            )
        )
    predicted = torch.tensor(
        [float(item["predicted_raw_alignment"]) for item in batches]
    )
    realized = torch.tensor(
        [float(item["realized_gain"]) for item in batches]
    )
    return {
        "method": "raw_alignment_matched",
        "rows": rows,
        "batches": batches,
        "birth_rounds": len(schedule),
        "batch_schedule": schedule,
        "total_seconds": time.perf_counter() - started,
        "cumulative_scoring_seconds": scoring,
        "cumulative_optimization_seconds": optimization,
        "predicted_realized_spearman": _correlation(
            predicted, realized, True
        ),
        "predicted_realized_pearson": _correlation(
            predicted, realized, False
        ),
        "maximum_validated_active_dofs": int(state.active_ids.numel()),
    }, state


def _run_fixed_references(
    context: CorrectedContext,
    maximum_active: int,
) -> tuple[dict[str, object], CorrectedState]:
    requested = (256, 512, 1024, 2048, 4096)
    levels = [
        level
        for level in requested
        if level <= maximum_active and level <= context.config.dictionary_count
    ]
    if maximum_active not in levels:
        levels.append(maximum_active)
    rows: list[dict[str, object]] = []
    final_state: CorrectedState | None = None
    started = time.perf_counter()
    for level in sorted(set(levels)):
        # A Full-HD state owns dense RGB, numerator, and mass buffers for every
        # view.  Retaining the previous fixed-K state while optimizing the next
        # level makes line search hold three dense states at once.
        if final_state is not None:
            del final_state
            final_state = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        active = torch.arange(level, device=context.reference_points.device)
        final_state = _evaluate(
            context,
            active,
            torch.zeros(level, dtype=torch.float64, device=active.device),
        )
        final_state, seconds = _optimize(
            context, final_state, context.config.fixed_optimization_steps
        )
        rows.append(
            _metrics(
                context,
                final_state,
                "fixed_space_cold",
                0,
                0.0,
                seconds,
                time.perf_counter() - started,
                {"fixed_k": level},
            )
        )
    assert final_state is not None
    return {
        "method": "fixed_space_cold",
        "rows": rows,
        "levels": sorted(set(levels)),
        "optimization_steps_per_level": context.config.fixed_optimization_steps,
        "total_seconds": time.perf_counter() - started,
        "maximum_validated_active_dofs": int(final_state.active_ids.numel()),
    }, final_state


def _build_context(
    prepared: object, config: CorrectedBirthConfig
) -> CorrectedContext:
    device = torch.device("cuda")
    base = prepared.base_field.to(device)
    target_field = prepared.gt_field.to(device)
    surface = sample_meshfree_zero_set(
        base,
        config.surface_samples,
        sobol_scramble_seed=config.surface_scramble_seed,
    )
    reference_points = surface.points
    reference_normals = surface.normals
    layout, levels = _layout(reference_points, config)
    support = _multiscale_support(
        reference_points, layout, levels, config.support_margin
    )
    target_points, target_success, target_displacement = _normal_line_roots(
        target_field,
        reference_points,
        reference_normals,
        maximum_offset=config.deformation_max_offset,
    )
    if not bool(target_success.all()):
        raise RuntimeError(
            f"TARGET_NORMAL_LINE_FAILURES:{int((~target_success).sum())}"
        )
    target_gradients = target_field.gradient(target_points)
    target_normals = target_gradients / torch.linalg.vector_norm(
        target_gradients, dim=1, keepdim=True
    ).clamp_min(1e-30)
    atlas = nested_fibonacci_atlas(device, (config.views,))
    boundary = enclosing_observation_sphere(reference_points)
    cells, base_visibility = _visibility_cells(
        base,
        reference_points,
        reference_normals,
        atlas,
        boundary,
        config,
    )
    target_cells, target_visibility = _visibility_cells(
        target_field,
        target_points,
        target_normals,
        atlas,
        boundary,
        config,
    )
    center_normals = base.interpolant_gradient(layout.centers)
    center_normals /= torch.linalg.vector_norm(
        center_normals, dim=1, keepdim=True
    ).clamp_min(1e-30)
    _, center_success, center_displacement = _normal_line_roots(
        target_field,
        layout.centers,
        center_normals,
        maximum_offset=config.deformation_max_offset,
    )
    absolute = torch.where(
        center_success,
        center_displacement.abs(),
        torch.zeros_like(center_displacement),
    )
    detail_ids = torch.nonzero(
        absolute >= torch.quantile(absolute, 0.75), as_tuple=False
    ).flatten()
    top_detail_ids = torch.nonzero(
        absolute >= torch.quantile(absolute, 0.90), as_tuple=False
    ).flatten()
    evaluator = BunnyGeometryEvaluator(
        prepared, reference_points, reference_normals, 16_384
    )
    context = CorrectedContext(
        config,
        base,
        target_field,
        layout,
        support,
        levels,
        reference_points,
        reference_normals,
        base.lower,
        base.upper,
        cells,
        target_points,
        target_normals,
        [],
        target_displacement,
        detail_ids,
        top_detail_ids,
        evaluator,
        {
            "base": base_visibility,
            "target": target_visibility,
            "camera_atlas": {
                "construction": "deterministic Fibonacci sphere",
                "views": config.views,
                "digest": atlas.digest,
                "directions": atlas.directions.detach().cpu().tolist(),
            },
            "attempted_packets": config.surface_samples * config.views,
            "packets_per_emitter": config.views,
            "emitters": config.surface_samples,
        },
    )
    _, base_renders = _render(
        reference_points, reference_normals, context, cells
    )
    context.cells = [
        replace(
            cell,
            support_mask=render.mass
            >= config.support_threshold
            / math.prod(config.footprint_scale),
        )
        for cell, render in zip(cells, base_renders)
    ]
    del base_renders
    torch.cuda.empty_cache()
    _, target_renders = _render(
        target_points, target_normals, context, target_cells
    )
    target_cells = [
        replace(
            cell,
            support_mask=render.mass
            >= config.support_threshold
            / math.prod(config.footprint_scale),
        )
        for cell, render in zip(target_cells, target_renders)
    ]
    del target_renders
    torch.cuda.empty_cache()
    target_images, _ = _render(
        target_points, target_normals, context, target_cells
    )
    context.target_images = target_images
    return context


def _finite_difference_gate(
    context: CorrectedContext,
) -> dict[str, object]:
    active = torch.arange(
        context.config.initial_count, device=context.reference_points.device
    )
    state = _evaluate(
        context,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device=active.device),
    )
    layout, support = _active_components(context, active)
    matrices = _jacobians(context, state, layout, support)
    tested = (0, 7, 19, 31)
    errors = []
    epsilon = 2e-5
    for parameter in tested:
        plus_coefficients = state.coefficients.clone()
        minus_coefficients = state.coefficients.clone()
        plus_coefficients[parameter] += epsilon
        minus_coefficients[parameter] -= epsilon
        plus = _evaluate(
            context, active, plus_coefficients, (layout, support)
        )
        minus = _evaluate(
            context, active, minus_coefficients, (layout, support)
        )
        analytic = torch.cat(
            [
                torch.sparse.mm(
                    matrix,
                    torch.nn.functional.one_hot(
                        torch.tensor(parameter, device=active.device),
                        active.numel(),
                    ).to(torch.float64)[:, None],
                ).flatten()
                for matrix in matrices
            ]
        )
        finite = torch.cat(
            [
                (right - left) / (2.0 * epsilon)
                for right, left in zip(plus.images, minus.images)
            ]
        )
        errors.append(
            float(
                torch.linalg.vector_norm(analytic - finite)
                / torch.linalg.vector_norm(finite).clamp_min(1e-30)
            )
        )
    return {
        "parameters": list(tested),
        "scheme": "central",
        "epsilon": epsilon,
        "relative_errors": errors,
        "maximum_relative_error": max(errors),
        "passed": max(errors) < 0.08,
    }


def _save_images(
    path: Path,
    context: CorrectedContext,
    states: list[tuple[str, CorrectedState]],
) -> None:
    import matplotlib.pyplot as plt

    panels: list[tuple[str, np.ndarray]] = []
    rows, columns = context.config.resolution_shape
    target = context.target_images[3].reshape(rows, columns, 3)
    panels.append(("target corrected forward RGB", target.detach().cpu().numpy()))
    for label, state in states:
        image = state.images[3].reshape(rows, columns, 3)
        panels.append((label, image.detach().cpu().numpy()))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.1 * len(panels), 3.2),
        facecolor="black",
    )
    for axis, (label, image) in zip(np.asarray(axes).reshape(-1), panels):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(label, color="white", fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_curves(path: Path, trajectories: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for trajectory in trajectories:
        rows = trajectory["rows"]
        x = [int(row["active_dofs"]) for row in rows]
        label = str(trajectory["method"])
        axes[0].plot(
            x, [float(row["image_loss"]) for row in rows], "o-", label=label
        )
        axes[1].plot(
            x,
            [float(row["symmetric_chamfer"]) for row in rows],
            "o-",
            label=label,
        )
        axes[2].plot(
            x, [float(row["total_seconds"]) for row in rows], "o-", label=label
        )
    axes[0].set_ylabel("corrected RGB loss")
    axes[1].set_ylabel("symmetric Chamfer")
    axes[2].set_ylabel("cumulative seconds")
    for axis in axes:
        axis.set_xlabel("active DoFs")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_yscale("log")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _trajectory_auc(rows: list[dict[str, object]], key: str) -> float:
    x = np.asarray([float(row["active_dofs"]) for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    if len(rows) == 1 or x[-1] <= x[0]:
        return float(y[-1])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def run_corrected_birth_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
    config: CorrectedBirthConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    config = config or CorrectedBirthConfig()
    experiment_started = time.perf_counter()
    prepared = prepare_stanford_bunny(
        mesh_path, build_surface_scaffold=False
    )
    context = _build_context(prepared, config)
    finite_difference = _finite_difference_gate(context)
    if not finite_difference["passed"]:
        raise RuntimeError(f"CORRECTED_RGB_JACOBIAN_FAILED:{finite_difference}")
    observation, observation_state = _run_observation_trajectory(context)
    schedule = list(observation["batch_schedule"])
    uniform, uniform_state = _run_matched_baseline(
        context, schedule, "uniform_matched"
    )
    random, random_state = _run_matched_baseline(
        context, schedule, "random_matched"
    )
    raw, raw_state = _run_raw_baseline(context, schedule)
    fixed, fixed_state = _run_fixed_references(
        context, int(observation["maximum_validated_active_dofs"])
    )
    trajectories = [observation, uniform, random, raw, fixed]
    efficiency_auc = {
        item["method"]: {
            "image_loss": _trajectory_auc(item["rows"], "image_loss"),
            "symmetric_chamfer": _trajectory_auc(
                item["rows"], "symmetric_chamfer"
            ),
        }
        for item in (observation, uniform, random, raw)
    }
    final_rows = {item["method"]: item["rows"][-1] for item in trajectories}
    observation_final = final_rows["observation_driven"]
    uniform_final = final_rows["uniform_matched"]
    random_final = final_rows["random_matched"]
    raw_final = final_rows["raw_alignment_matched"]
    geometry_wins = (
        float(observation_final["symmetric_chamfer"])
        < float(uniform_final["symmetric_chamfer"])
        and float(observation_final["symmetric_chamfer"])
        < float(random_final["symmetric_chamfer"])
        and float(observation_final["symmetric_chamfer"])
        < float(raw_final["symmetric_chamfer"])
    )
    image_wins = (
        float(observation_final["image_loss"])
        < float(uniform_final["image_loss"])
        and float(observation_final["image_loss"])
        < float(random_final["image_loss"])
        and float(observation_final["image_loss"])
        < float(raw_final["image_loss"])
    )
    image_parameter_efficient = all(
        efficiency_auc["observation_driven"]["image_loss"]
        < efficiency_auc[method]["image_loss"]
        for method in (
            "uniform_matched",
            "random_matched",
            "raw_alignment_matched",
        )
    )
    geometry_parameter_efficient = all(
        efficiency_auc["observation_driven"]["symmetric_chamfer"]
        < efficiency_auc[method]["symmetric_chamfer"]
        for method in (
            "uniform_matched",
            "random_matched",
            "raw_alignment_matched",
        )
    )
    image_parameter_efficient_vs_uniform = (
        efficiency_auc["observation_driven"]["image_loss"]
        < efficiency_auc["uniform_matched"]["image_loss"]
    )
    geometry_parameter_efficient_vs_uniform = (
        efficiency_auc["observation_driven"]["symmetric_chamfer"]
        < efficiency_auc["uniform_matched"]["symmetric_chamfer"]
    )
    observation_beats_uniform_image = (
        float(observation_final["image_loss"])
        < float(uniform_final["image_loss"])
    )
    observation_beats_uniform_geometry = (
        float(observation_final["symmetric_chamfer"])
        < float(uniform_final["symmetric_chamfer"])
    )
    comparable_methods = (observation, uniform, random, raw)
    best_image_method = min(
        comparable_methods,
        key=lambda item: float(item["rows"][-1]["image_loss"]),
    )["method"]
    best_geometry_method = min(
        comparable_methods,
        key=lambda item: float(item["rows"][-1]["symmetric_chamfer"]),
    )["method"]
    stable = (
        int(observation_final["root_failures"]) == 0
        and int(observation_final["cg_failures"]) == 0
        and finite_difference["passed"]
    )
    if (
        stable
        and observation_beats_uniform_image
        and observation_beats_uniform_geometry
        and image_parameter_efficient_vs_uniform
        and geometry_parameter_efficient_vs_uniform
    ):
        verdict = (
            "CORRECTED_FORWARD_RGB_BIRTH_REVALIDATED"
            if image_wins and geometry_wins
            else "CORRECTED_FORWARD_RGB_BIRTH_PARTIALLY_REVALIDATED_RAW_WINS"
        )
    else:
        verdict = "CORRECTED_FORWARD_RGB_BIRTH_WEAKENS"
    _save_images(
        render_directory / "v07_corrected_birth_reconstruction.png",
        context,
        [
            ("observation-driven", observation_state),
            ("uniform matched", uniform_state),
            ("random matched", random_state),
            ("raw matched", raw_state),
            ("fixed-space", fixed_state),
        ],
    )
    _save_curves(
        figure_directory / "v07_corrected_birth_scaling.png", trajectories
    )
    all_rows = [row for trajectory in trajectories for row in trajectory["rows"]]
    report: dict[str, object] = {
        "version": "0.7.0",
        "phase": "corrected_forward_rgb_birth",
        "environment": cuda_environment(),
        "config": config.__dict__,
        "renderer": {
            "appearance_variant": "C",
            "distance_sigma": 0.0,
            "base_appearance": "camera-independent position RGB diagnostic",
            "normal_lighting": "0.35 + 0.65 max(0,n dot omega)",
            "forward_factorization": (
                "zero-set samples -> outgoing visibility -> local RGB slab -> "
                "detector pixels"
            ),
            "spatial_readout": (
                "continuous tensor-product cubic B-spline footprint in "
                "boundary-light-slab coordinates"
            ),
            "source_boundary_transverse_equivalence_checked": True,
            "primary_camera_rays": False,
            "mesh_surface_scaffold": False,
            "fixed_visibility_during_local_optimization": True,
        },
        "finite_difference_validation": finite_difference,
        "visibility": context.visibility_report,
        "target": {
            "normal_line_root_failures": 0,
            "rms_displacement": float(
                torch.sqrt(context.target_displacement.square().mean())
            ),
            "maximum_abs_displacement": float(
                context.target_displacement.abs().max()
            ),
        },
        "observation_driven": observation,
        "uniform_matched": uniform,
        "random_matched": random,
        "raw_alignment_matched": raw,
        "fixed_space_references": fixed,
        "final_comparison": final_rows,
        "parameter_efficiency_auc": efficiency_auc,
        "judgment": {
            "birth_signal_exists": bool(
                max(float(item["realized_gain"]) for item in observation["batches"])
                > 0.0
            ),
            "observation_driven_image_winner": image_wins,
            "observation_driven_geometry_winner": geometry_wins,
            "observation_driven_beats_uniform_image": (
                observation_beats_uniform_image
            ),
            "observation_driven_beats_uniform_geometry": (
                observation_beats_uniform_geometry
            ),
            "best_image_method": best_image_method,
            "best_geometry_method": best_geometry_method,
            "image_parameter_efficient": image_parameter_efficient,
            "geometry_parameter_efficient": geometry_parameter_efficient,
            "image_parameter_efficient_vs_uniform": (
                image_parameter_efficient_vs_uniform
            ),
            "geometry_parameter_efficient_vs_uniform": (
                geometry_parameter_efficient_vs_uniform
            ),
            "parameter_efficient": image_parameter_efficient
            and geometry_parameter_efficient,
            "compute_efficient": float(observation["total_seconds"])
            <= float(uniform["total_seconds"]),
            "maximum_validated_active_dofs": observation[
                "maximum_validated_active_dofs"
            ],
            "stopping_reason": observation["stopping_reason"],
            "dominant_bottleneck": (
                "candidate scoring"
                if float(observation["cumulative_scoring_seconds"])
                > float(observation["cumulative_optimization_seconds"])
                else "optimization"
            ),
            "effect_on_previous_birth_story": (
                "partially revalidated: observation-driven selection beats "
                "uniform/random, but raw alignment beats quadratic scoring "
                "and the scoring overhead is not compute-efficient"
                if verdict
                == "CORRECTED_FORWARD_RGB_BIRTH_PARTIALLY_REVALIDATED_RAW_WINS"
                else (
                    "strengthens"
                    if verdict == "CORRECTED_FORWARD_RGB_BIRTH_REVALIDATED"
                    else "weakens"
                )
            ),
            "verdict": verdict,
        },
        "limitations": [
            (
                "Visibility is generated by corrected forward zero-set "
                "transport for base and target, then frozen inside each local "
                "optimization cell."
            ),
            (
                "The local Jacobian differentiates surface position, position "
                "color, normal lighting, cubic footprint, and normalized "
                "accumulation, but not visibility topology changes."
            ),
            (
                "Birth views use the eight direct coarse-atlas directions; "
                "unlike the qualitative v0.6 readout they do not blend "
                "auxiliary angular neighbors."
            ),
            (
                "The Stanford mesh is used only to acquire the scalar fields "
                "and for final geometry evaluation."
            ),
            (
                "The finite 4096-element master dictionary is a safety "
                "envelope, not a preset active-DoF stopping cap."
            ),
        ],
        "artifacts": {
            "json": str(artifact_directory / "v07_corrected_birth.json"),
            "csv": str(artifact_directory / "v07_corrected_birth.csv"),
            "scaling_figure": str(
                figure_directory / "v07_corrected_birth_scaling.png"
            ),
            "render_figure": str(
                render_directory / "v07_corrected_birth_reconstruction.png"
            ),
        },
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v07_corrected_birth.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(artifact_directory / "v07_corrected_birth.csv", all_rows)
    return report
