"""Sparse field evaluation and edge-major projected-geodesic tangents.

Production routines never allocate ``points x total_basis`` tensors. A
chart's conservative path envelope is queried once against compact Wendland
supports; emitter derivatives then propagate only over that ragged local
union. Dense ``LocalField`` remains confined to small diagnostic comparisons.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .locality import wendland_gradients, wendland_values


Tensor = torch.Tensor


@dataclass(frozen=True)
class SparsePointBasisSupport:
    """CSR point/basis incidence; edges are point-major and never dense."""

    row_ptr: Tensor
    point_ids: Tensor
    basis_ids: Tensor
    point_count: int
    basis_count: int

    @property
    def nnz(self) -> int:
        return int(self.basis_ids.numel())

    def counts(self) -> Tensor:
        return self.row_ptr[1:].to(torch.int64) - self.row_ptr[:-1].to(torch.int64)


@dataclass(frozen=True)
class GeodesicTemplate:
    reference_centers: Tensor
    reference_normals: Tensor
    reference_tangent_1: Tensor
    reference_tangent_2: Tensor
    chart_radii: Tensor
    eta: float
    steps: int

    @property
    def chart_count(self) -> int:
        return int(self.reference_centers.shape[0])


@dataclass
class PreparedGeodesicCenters:
    positions: Tensor
    normals: Tensor
    tangent_1: Tensor
    tangent_2: Tensor
    support: SparsePointBasisSupport | None = None
    dpositions: Tensor | None = None
    dnormals: Tensor | None = None
    dtangent_1: Tensor | None = None
    dtangent_2: Tensor | None = None


class CompactSupportField:
    """Dense only over one chart's queried local union, never total K."""

    def __init__(
        self,
        base: Any,
        basis_centers: Tensor,
        basis_radii: Tensor,
        local_basis_ids: Tensor,
        local_coefficients: Tensor,
    ) -> None:
        self.base = base
        self.centers = basis_centers[local_basis_ids]
        self.radii = basis_radii[local_basis_ids]
        self.coefficients = local_coefficients
        self.lower = base.lower
        self.upper = base.upper

    def _terms(self, points: Tensor) -> tuple[Tensor, Tensor]:
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        offsets = flat[:, None, :] - self.centers[None, :, :]
        radii = self.radii[None, :].expand(flat.shape[0], -1)
        values = wendland_values(
            offsets.reshape(-1, 3), radii.reshape(-1)
        ).reshape(*shape, -1)
        gradients = wendland_gradients(
            offsets.reshape(-1, 3), radii.reshape(-1)
        ).reshape(*shape, -1, 3)
        return values, gradients

    def value(self, points: Tensor) -> Tensor:
        values, _ = self._terms(points)
        return self.base.value(points) + values @ self.coefficients

    def gradient(self, points: Tensor) -> Tensor:
        _, gradients = self._terms(points)
        return self.base.gradient(points) + (
            gradients * self.coefficients[..., None]
        ).sum(-2)


def build_chart_envelope_support(
    template: GeodesicTemplate,
    basis_centers: Tensor,
    basis_radii: Tensor,
) -> SparsePointBasisSupport:
    """Query the compact-support union of every possible chart trajectory."""
    chart_np = template.reference_centers.detach().cpu().numpy()
    chart_radius_np = template.chart_radii.detach().cpu().numpy()
    basis_np = basis_centers.detach().cpu().numpy()
    basis_radius_np = basis_radii.detach().cpu().numpy()
    tree = cKDTree(basis_np)
    maximum_basis_radius = float(basis_radius_np.max())
    point_ids: list[int] = []
    basis_ids: list[int] = []
    counts = np.zeros(template.chart_count, dtype=np.int64)
    # Projection can move the chord endpoint slightly beyond its nominal disk
    # radius.  A conservative 25% + 1e-2 envelope still has local fanout and
    # guarantees that no compact support touched by the four-step walk is lost.
    for chart, (center, radius) in enumerate(zip(chart_np, chart_radius_np)):
        path_envelope = 1.25 * radius + 1e-2
        candidates = tree.query_ball_point(
            center, float(path_envelope + maximum_basis_radius), workers=1
        )
        candidates = sorted(
            int(index) for index in candidates
            if np.linalg.norm(center - basis_np[index])
            <= path_envelope + basis_radius_np[index]
        )
        counts[chart] = len(candidates)
        point_ids.extend([chart] * len(candidates))
        basis_ids.extend(candidates)
    row_ptr = np.empty(template.chart_count + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(counts, out=row_ptr[1:])
    device = template.reference_centers.device
    return SparsePointBasisSupport(
        torch.as_tensor(row_ptr, dtype=torch.long, device=device),
        torch.as_tensor(point_ids, dtype=torch.long, device=device),
        torch.as_tensor(basis_ids, dtype=torch.long, device=device),
        template.chart_count,
        int(basis_centers.shape[0]),
    )


def emitter_support_from_owners(
    chart_support: SparsePointBasisSupport,
    owners: Tensor,
) -> tuple[SparsePointBasisSupport, Tensor]:
    """Expand chart dependency rows to emitters without a dense mask."""
    device = owners.device
    if owners.numel() > 1 and not bool(torch.all(owners[1:] >= owners[:-1])):
        raise ValueError("production emitter owners must be chart-major")
    counts = chart_support.counts()[owners]
    row_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.long)
    point_ids = torch.repeat_interleave(
        torch.arange(owners.numel(), device=device), counts
    )
    columns: list[Tensor] = []
    parent_edges: list[Tensor] = []
    for chart in torch.unique(owners, sorted=True).tolist():
        emitter_count = int((owners == chart).sum())
        begin = int(chart_support.row_ptr[chart])
        end = int(chart_support.row_ptr[chart + 1])
        columns.append(chart_support.basis_ids[begin:end].repeat(emitter_count))
        parent_edges.append(
            torch.arange(begin, end, device=device).repeat(emitter_count)
        )
    basis_ids = (
        torch.cat(columns)
        if columns
        else torch.empty(0, dtype=torch.long, device=device)
    )
    parents = (
        torch.cat(parent_edges)
        if parent_edges
        else torch.empty(0, dtype=torch.long, device=device)
    )
    return SparsePointBasisSupport(
        row_ptr, point_ids, basis_ids, int(owners.numel()),
        chart_support.basis_count,
    ), parents


def sparse_field_evaluate(
    field: Any,
    points: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate F/grad F and basis terms using sparse point-basis pairs."""
    values = field.value(points)
    gradients = field.gradient(points)
    if support.nnz == 0:
        empty = torch.empty(0, dtype=points.dtype, device=points.device)
        return values, gradients, empty, torch.empty(
            (0, 3), dtype=points.dtype, device=points.device
        )
    offsets = points[support.point_ids] - basis_centers[support.basis_ids]
    radii = basis_radii[support.basis_ids]
    edge_values = wendland_values(offsets, radii)
    edge_gradients = wendland_gradients(offsets, radii)
    if coefficients is not None:
        values = values.scatter_add(
            0, support.point_ids,
            edge_values * coefficients[support.basis_ids],
        )
        gradients = gradients.index_add(
            0, support.point_ids,
            edge_gradients * coefficients[support.basis_ids, None],
        )
    return values, gradients, edge_values, edge_gradients


def _project_sparse_forward(
    field: Any,
    points: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor,
    eta: float,
) -> Tensor:
    values, gradients, _, _ = sparse_field_evaluate(
        field, points, support, basis_centers, basis_radii, coefficients
    )
    denominator = gradients.square().sum(1, keepdim=True) + eta * eta
    return points - values[:, None] * gradients / denominator


def prepare_centers_sparse_field(
    field: Any,
    template: GeodesicTemplate,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor,
    chart_support: SparsePointBasisSupport | None = None,
) -> PreparedGeodesicCenters:
    """Prepare deformed centers with sparse field evaluation only."""
    chart_support = chart_support or build_chart_envelope_support(
        template, basis_centers, basis_radii
    )
    centers = _project_sparse_forward(
        field, template.reference_centers, chart_support,
        basis_centers, basis_radii, coefficients, template.eta,
    )
    _, gradients, _, _ = sparse_field_evaluate(
        field, centers, chart_support, basis_centers, basis_radii, coefficients
    )
    normals = gradients / torch.linalg.vector_norm(
        gradients, dim=1, keepdim=True
    ).clamp_min(1e-30)
    first = template.reference_tangent_1 - (
        template.reference_tangent_1 * normals
    ).sum(1, keepdim=True) * normals
    first = first / torch.linalg.vector_norm(
        first, dim=1, keepdim=True
    ).clamp_min(1e-30)
    second = torch.linalg.cross(normals, first, dim=1)
    second = second / torch.linalg.vector_norm(
        second, dim=1, keepdim=True
    ).clamp_min(1e-30)
    return PreparedGeodesicCenters(
        centers, normals, first, second, chart_support
    )


def map_emitter_chunk_sparse_field(
    field: Any,
    template: GeodesicTemplate,
    prepared: PreparedGeodesicCenters,
    owners: Tensor,
    rho: Tensor,
    theta: Tensor,
    basis_centers: Tensor,
    basis_radii: Tensor,
    coefficients: Tensor,
) -> tuple[Tensor, Tensor, Tensor, SparsePointBasisSupport]:
    """Nonlinear geodesic map using sparse field evaluation at every step."""
    support, _ = emitter_support_from_owners(prepared.support, owners)
    direction = (
        torch.cos(theta)[:, None] * prepared.tangent_1[owners]
        + torch.sin(theta)[:, None] * prepared.tangent_2[owners]
    )
    positions = prepared.positions[owners]
    velocity = direction
    step_length = rho * template.chart_radii[owners] / template.steps
    for _ in range(template.steps):
        candidate = positions + step_length[:, None] * velocity
        positions = _project_sparse_forward(
            field, candidate, support, basis_centers, basis_radii,
            coefficients, template.eta,
        )
        _, gradients, _, _ = sparse_field_evaluate(
            field, positions, support, basis_centers, basis_radii, coefficients
        )
        normals = gradients / torch.linalg.vector_norm(
            gradients, dim=1, keepdim=True
        ).clamp_min(1e-30)
        velocity = velocity - (
            velocity * normals
        ).sum(1, keepdim=True) * normals
        velocity = velocity / torch.linalg.vector_norm(
            velocity, dim=1, keepdim=True
        ).clamp_min(1e-30)
    values, gradients, _, _ = sparse_field_evaluate(
        field, positions, support, basis_centers, basis_radii, coefficients
    )
    normals = gradients / torch.linalg.vector_norm(
        gradients, dim=1, keepdim=True
    ).clamp_min(1e-30)
    return positions, normals, values.abs(), support


def _base_gradient_jacobian(field: Any, points: Tensor) -> Tensor:
    """Exact local 3x3 Jacobian of the interpolated base gradient.

    This graph contains only point coordinates, never lambda/basis axes, and
    is destroyed before the caller returns.  Centered differences are not
    valid at trilinear voxel boundaries, hence this small exact point VJP.
    """
    with torch.enable_grad():
        local = points.detach().requires_grad_(True)
        gradient = field.gradient(local)
        rows = []
        for component in range(3):
            rows.append(torch.autograd.grad(
                gradient[:, component].sum(), local,
                retain_graph=component < 2, create_graph=False,
            )[0])
    return torch.stack(rows, dim=1).detach()


def _unit_sparse(
    value: Tensor, dvalue: Tensor, point_ids: Tensor
) -> tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-30)
    unit = value / norm
    edge_unit = unit[point_ids]
    tangent = dvalue - edge_unit * (edge_unit * dvalue).sum(1, keepdim=True)
    return unit, tangent / norm[point_ids]


def _field_sparse_linearization(
    field: Any,
    points: Tensor,
    dpoints: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values, gradients, basis_values, basis_gradients = sparse_field_evaluate(
        field, points, support, basis_centers, basis_radii
    )
    rows = support.point_ids
    dvalues = basis_values + (gradients[rows] * dpoints).sum(1)
    hessian = _base_gradient_jacobian(field, points)
    dgradients = basis_gradients + torch.einsum(
        "eab,eb->ea", hessian[rows], dpoints
    )
    return values, gradients, dvalues, dgradients


def _project_sparse(
    field: Any,
    points: Tensor,
    dpoints: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor]:
    values, gradients, dvalues, dgradients = _field_sparse_linearization(
        field, points, dpoints, support, basis_centers, basis_radii
    )
    denominator = gradients.square().sum(1) + eta * eta
    projected = points - values[:, None] * gradients / denominator[:, None]
    rows = support.point_ids
    ddenominator = 2.0 * (gradients[rows] * dgradients).sum(1)
    derivative = dpoints - (
        dvalues[:, None] * gradients[rows]
        + values[rows, None] * dgradients
    ) / denominator[rows, None]
    derivative += (
        values[rows, None] * gradients[rows] * ddenominator[:, None]
        / denominator[rows, None].square()
    )
    return projected, derivative


def _normal_sparse(
    field: Any,
    points: Tensor,
    dpoints: Tensor,
    support: SparsePointBasisSupport,
    basis_centers: Tensor,
    basis_radii: Tensor,
) -> tuple[Tensor, Tensor]:
    _, gradients, _, dgradients = _field_sparse_linearization(
        field, points, dpoints, support, basis_centers, basis_radii
    )
    return _unit_sparse(gradients, dgradients, support.point_ids)


def prepare_centers(
    field: Any,
    template: GeodesicTemplate,
    basis_centers: Tensor | None = None,
    basis_radii: Tensor | None = None,
    chart_support: SparsePointBasisSupport | None = None,
) -> PreparedGeodesicCenters:
    linearized = basis_centers is not None and basis_radii is not None
    if linearized and chart_support is None:
        chart_support = build_chart_envelope_support(
            template, basis_centers, basis_radii
        )
    if linearized:
        zero = torch.zeros(
            (chart_support.nnz, 3), dtype=template.reference_centers.dtype,
            device=template.reference_centers.device,
        )
        centers, dcenters = _project_sparse(
            field, template.reference_centers, zero, chart_support,
            basis_centers, basis_radii, template.eta,
        )
        normals, dnormals = _normal_sparse(
            field, centers, dcenters, chart_support,
            basis_centers, basis_radii,
        )
    else:
        values = field.value(template.reference_centers)
        gradient = field.gradient(template.reference_centers)
        denominator = gradient.square().sum(1, keepdim=True) + template.eta**2
        centers = (
            template.reference_centers
            - values[:, None] * gradient / denominator
        )
        normals = field.gradient(centers)
        normals = normals / torch.linalg.vector_norm(
            normals, dim=1, keepdim=True
        ).clamp_min(1e-30)
        dcenters = dnormals = None
    raw = template.reference_tangent_1 - (
        template.reference_tangent_1 * normals
    ).sum(1, keepdim=True) * normals
    if linearized:
        rows = chart_support.point_ids
        dot = (template.reference_tangent_1 * normals).sum(1)
        ddot = (template.reference_tangent_1[rows] * dnormals).sum(1)
        draw = -(
            ddot[:, None] * normals[rows]
            + dot[rows, None] * dnormals
        )
        first, dfirst = _unit_sparse(raw, draw, rows)
        second_raw = torch.linalg.cross(normals, first, dim=1)
        dsecond_raw = (
            torch.linalg.cross(dnormals, first[rows], dim=1)
            + torch.linalg.cross(normals[rows], dfirst, dim=1)
        )
        second, dsecond = _unit_sparse(second_raw, dsecond_raw, rows)
    else:
        first = raw / torch.linalg.vector_norm(
            raw, dim=1, keepdim=True
        ).clamp_min(1e-30)
        second = torch.linalg.cross(normals, first, dim=1)
        second = second / torch.linalg.vector_norm(
            second, dim=1, keepdim=True
        ).clamp_min(1e-30)
        dfirst = dsecond = None
    return PreparedGeodesicCenters(
        centers, normals, first, second, chart_support,
        dcenters, dnormals, dfirst, dsecond,
    )


def map_emitter_chunk(
    field: Any,
    template: GeodesicTemplate,
    prepared: PreparedGeodesicCenters,
    owners: Tensor,
    rho: Tensor,
    theta: Tensor,
    basis_centers: Tensor | None = None,
    basis_radii: Tensor | None = None,
) -> tuple[
    Tensor, Tensor, Tensor | None, Tensor | None, Tensor,
    SparsePointBasisSupport | None,
]:
    linearized = basis_centers is not None and basis_radii is not None
    cosine, sine = torch.cos(theta), torch.sin(theta)
    direction = (
        cosine[:, None] * prepared.tangent_1[owners]
        + sine[:, None] * prepared.tangent_2[owners]
    )
    length = rho * template.chart_radii[owners]
    positions = prepared.positions[owners]
    velocity = direction
    if linearized:
        support, parents = emitter_support_from_owners(prepared.support, owners)
        dpositions = prepared.dpositions[parents]
        rows = support.point_ids
        dvelocity = (
            cosine[rows, None] * prepared.dtangent_1[parents]
            + sine[rows, None] * prepared.dtangent_2[parents]
        )
    else:
        support = None
        dpositions = dvelocity = None
    step_length = length / template.steps
    for _ in range(template.steps):
        candidate = positions + step_length[:, None] * velocity
        if linearized:
            rows = support.point_ids
            dcandidate = dpositions + step_length[rows, None] * dvelocity
            positions, dpositions = _project_sparse(
                field, candidate, dcandidate, support,
                basis_centers, basis_radii, template.eta,
            )
            normals, dnormals = _normal_sparse(
                field, positions, dpositions, support,
                basis_centers, basis_radii,
            )
            scalar = (velocity * normals).sum(1)
            dscalar = (
                (dvelocity * normals[rows]).sum(1)
                + (velocity[rows] * dnormals).sum(1)
            )
            raw_velocity = velocity - scalar[:, None] * normals
            draw_velocity = (
                dvelocity
                - dscalar[:, None] * normals[rows]
                - scalar[rows, None] * dnormals
            )
            velocity, dvelocity = _unit_sparse(
                raw_velocity, draw_velocity, rows
            )
        else:
            values = field.value(candidate)
            gradients = field.gradient(candidate)
            denominator = gradients.square().sum(1, keepdim=True) + template.eta**2
            positions = candidate - values[:, None] * gradients / denominator
            normals = field.gradient(positions)
            normals = normals / torch.linalg.vector_norm(
                normals, dim=1, keepdim=True
            ).clamp_min(1e-30)
            velocity = velocity - (
                velocity * normals
            ).sum(1, keepdim=True) * normals
            velocity = velocity / torch.linalg.vector_norm(
                velocity, dim=1, keepdim=True
            ).clamp_min(1e-30)
    if linearized:
        normals, dnormals = _normal_sparse(
            field, positions, dpositions, support,
            basis_centers, basis_radii,
        )
    else:
        normals = field.gradient(positions)
        normals = normals / torch.linalg.vector_norm(
            normals, dim=1, keepdim=True
        ).clamp_min(1e-30)
        dnormals = None
    return (
        positions, normals, dpositions, dnormals,
        field.value(positions).abs(), support,
    )


def exact_chart_local_linearization(
    field: Any,
    template: GeodesicTemplate,
    chart_support: SparsePointBasisSupport,
    chart: int,
    rho: Tensor,
    theta: Tensor,
    basis_centers: Tensor,
    basis_radii: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SparsePointBasisSupport]:
    """Exact AD linearization over one chart's compact dependency union.

    The only basis axis in this function is ``local_basis_ids``.  Consequently
    all temporary basis and Jacobian tensors scale as M times local fanout,
    never M times total parameter count.
    """
    begin = int(chart_support.row_ptr[chart])
    end = int(chart_support.row_ptr[chart + 1])
    local_basis_ids = chart_support.basis_ids[begin:end]
    local_count = int(local_basis_ids.numel())
    m = int(rho.numel())
    owners = torch.zeros(m, dtype=torch.long, device=rho.device)
    reference_center = template.reference_centers[chart:chart + 1]
    reference_normal = template.reference_normals[chart:chart + 1]
    reference_first = template.reference_tangent_1[chart:chart + 1]
    reference_second = template.reference_tangent_2[chart:chart + 1]
    radius = template.chart_radii[chart:chart + 1]

    def mapped(local_coefficients: Tensor) -> Tensor:
        local_field = CompactSupportField(
            field, basis_centers, basis_radii,
            local_basis_ids, local_coefficients,
        )
        gradient = local_field.gradient(reference_center)
        value = local_field.value(reference_center)
        denominator = gradient.square().sum(1, keepdim=True) + template.eta**2
        center = reference_center - value[:, None] * gradient / denominator
        normal = local_field.gradient(center)
        normal = normal / torch.linalg.vector_norm(
            normal, dim=1, keepdim=True
        ).clamp_min(1e-30)
        first = reference_first - (
            reference_first * normal
        ).sum(1, keepdim=True) * normal
        first = first / torch.linalg.vector_norm(
            first, dim=1, keepdim=True
        ).clamp_min(1e-30)
        second = torch.linalg.cross(normal, first, dim=1)
        second = second / torch.linalg.vector_norm(
            second, dim=1, keepdim=True
        ).clamp_min(1e-30)
        direction = (
            torch.cos(theta)[:, None] * first[owners]
            + torch.sin(theta)[:, None] * second[owners]
        )
        length = rho * radius[owners]
        position = center[owners]
        velocity = direction
        step_length = length / template.steps
        for _ in range(template.steps):
            candidate = position + step_length[:, None] * velocity
            value = local_field.value(candidate)
            gradient = local_field.gradient(candidate)
            denominator = gradient.square().sum(1, keepdim=True) + template.eta**2
            position = candidate - value[:, None] * gradient / denominator
            normal = local_field.gradient(position)
            normal = normal / torch.linalg.vector_norm(
                normal, dim=1, keepdim=True
            ).clamp_min(1e-30)
            velocity = velocity - (
                velocity * normal
            ).sum(1, keepdim=True) * normal
            velocity = velocity / torch.linalg.vector_norm(
                velocity, dim=1, keepdim=True
            ).clamp_min(1e-30)
        normal = local_field.gradient(position)
        normal = normal / torch.linalg.vector_norm(
            normal, dim=1, keepdim=True
        ).clamp_min(1e-30)
        return torch.cat((position.reshape(-1), normal.reshape(-1)))

    zero = torch.zeros(
        local_count, dtype=rho.dtype, device=rho.device,
        requires_grad=True,
    )
    value = mapped(zero)
    # local_count is the forward-mode width. It is the queried fanout, not K.
    jacobian = (
        torch.autograd.functional.jacobian(
            mapped, zero, vectorize=True, strategy="forward-mode"
        )
        if local_count
        else value.new_empty((value.numel(), 0))
    )
    positions = value[:3 * m].reshape(m, 3)
    normals = value[3 * m:].reshape(m, 3)
    dx = jacobian[:3 * m].reshape(m, 3, local_count).permute(0, 2, 1).reshape(-1, 3)
    dn = jacobian[3 * m:].reshape(m, 3, local_count).permute(0, 2, 1).reshape(-1, 3)
    counts = torch.full((m,), local_count, dtype=torch.long, device=rho.device)
    row_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    point_ids = torch.repeat_interleave(torch.arange(m, device=rho.device), counts)
    support = SparsePointBasisSupport(
        row_ptr, point_ids, local_basis_ids.repeat(m), m,
        chart_support.basis_count,
    )
    residual = field.value(positions).abs()
    return positions, normals, dx, dn, residual, support


@dataclass(frozen=True)
class SparseGeodesicGraph:
    row_ptr: Tensor
    column_indices: Tensor
    dx_dlambda: Tensor
    dn_dlambda: Tensor
    emitter_count: int
    parameter_count: int
    weight_derivative_policy: str = "FROZEN_ZERO"

    @property
    def nnz(self) -> int:
        return int(self.column_indices.numel())

    @property
    def bytes(self) -> int:
        return sum(
            item.numel() * item.element_size()
            for item in (
                self.row_ptr, self.column_indices,
                self.dx_dlambda, self.dn_dlambda,
            )
        )

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for item in (
            self.row_ptr, self.column_indices,
            self.dx_dlambda, self.dn_dlambda,
        ):
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def edge_counts(self) -> Tensor:
        return (
            self.row_ptr[1:].to(torch.int64)
            - self.row_ptr[:-1].to(torch.int64)
        )

    def jvp(self, delta_lambda: Tensor) -> tuple[Tensor, Tensor]:
        device = delta_lambda.device
        counts = self.edge_counts().to(device)
        rows = torch.repeat_interleave(
            torch.arange(self.emitter_count, device=device), counts
        )
        columns = self.column_indices.to(device=device, dtype=torch.long)
        out_x = torch.zeros(
            (self.emitter_count, 3),
            dtype=delta_lambda.dtype, device=device,
        )
        out_n = torch.zeros_like(out_x)
        out_x.index_add_(
            0, rows,
            self.dx_dlambda.to(device=device, dtype=delta_lambda.dtype)
            * delta_lambda[columns, None],
        )
        out_n.index_add_(
            0, rows,
            self.dn_dlambda.to(device=device, dtype=delta_lambda.dtype)
            * delta_lambda[columns, None],
        )
        return out_x, out_n

    def vjp(self, g_x: Tensor, g_n: Tensor | None = None) -> Tensor:
        device = g_x.device
        counts = self.edge_counts().to(device)
        rows = torch.repeat_interleave(
            torch.arange(self.emitter_count, device=device), counts
        )
        columns = self.column_indices.to(device=device, dtype=torch.long)
        values = (
            self.dx_dlambda.to(device=device, dtype=g_x.dtype) * g_x[rows]
        ).sum(1)
        if g_n is not None:
            values += (
                self.dn_dlambda.to(device=device, dtype=g_n.dtype) * g_n[rows]
            ).sum(1)
        output = torch.zeros(
            self.parameter_count, dtype=values.dtype, device=device
        )
        output.index_add_(0, columns, values)
        return output


class SparseGraphBuilder:
    def __init__(
        self, emitter_count: int, parameter_count: int,
        threshold: float = 1e-10,
    ) -> None:
        self.emitter_count = emitter_count
        self.parameter_count = parameter_count
        self.threshold = threshold
        self._counts: list[np.ndarray] = []
        self._columns: list[np.ndarray] = []
        self._dx: list[np.ndarray] = []
        self._dn: list[np.ndarray] = []
        self._rows = 0

    def append(
        self, dx: Tensor, dn: Tensor,
        support: SparsePointBasisSupport,
    ) -> None:
        keep = torch.maximum(
            torch.linalg.vector_norm(dx, dim=1),
            torch.linalg.vector_norm(dn, dim=1),
        ) > self.threshold
        counts = torch.bincount(
            support.point_ids[keep], minlength=support.point_count
        )
        self._counts.append(counts.to(torch.int32).cpu().numpy())
        self._columns.append(
            support.basis_ids[keep].to(torch.int32).cpu().numpy()
        )
        self._dx.append(
            dx[keep].detach().to(torch.float32).cpu().numpy()
        )
        self._dn.append(
            dn[keep].detach().to(torch.float32).cpu().numpy()
        )
        self._rows += support.point_count

    def finish(self, *, pin_memory: bool = False) -> SparseGeodesicGraph:
        if self._rows != self.emitter_count:
            raise ValueError(
                f"expected {self.emitter_count} rows, received {self._rows}"
            )
        counts = (
            np.concatenate(self._counts)
            if self._counts else np.zeros(0, np.int32)
        )
        row_ptr = np.empty(self.emitter_count + 1, dtype=np.int32)
        row_ptr[0] = 0
        np.cumsum(counts, out=row_ptr[1:])
        columns = (
            np.concatenate(self._columns)
            if self._columns else np.zeros(0, np.int32)
        )
        dx = (
            np.concatenate(self._dx)
            if self._dx else np.zeros((0, 3), np.float32)
        )
        dn = (
            np.concatenate(self._dn)
            if self._dn else np.zeros((0, 3), np.float32)
        )
        tensors = [
            torch.from_numpy(item) for item in (row_ptr, columns, dx, dn)
        ]
        if pin_memory and torch.cuda.is_available():
            tensors = [item.pin_memory() for item in tensors]
        return SparseGeodesicGraph(
            *tensors, self.emitter_count, self.parameter_count
        )
