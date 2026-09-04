"""Deterministic hierarchical surface bases and compact-support lookup."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from .fields import LocalBasisField, SphereField, TorusField, ZeroSetField, unit_normals


Tensor = torch.Tensor


def hierarchical_surface_points(
    field: ZeroSetField,
    count: int,
    device: torch.device,
) -> Tensor:
    """Map a nested Sobol prefix to a deterministic analytic surface."""
    if count <= 0:
        raise ValueError("surface point count must be positive")
    sampler = getattr(field, "hierarchical_surface_points", None)
    if sampler is not None:
        return sampler(count, device)
    parameters = torch.quasirandom.SobolEngine(2, scramble=False).draw(count)
    parameters = parameters.to(dtype=torch.float64, device=device)
    first, second = parameters.unbind(dim=-1)
    azimuth = 2.0 * torch.pi * second
    if isinstance(field, SphereField):
        z = field.radius * (1.0 - 2.0 * first)
        radial = torch.sqrt((field.radius**2 - z * z).clamp_min(0.0))
        return torch.stack(
            (radial * torch.cos(azimuth), radial * torch.sin(azimuth), z),
            dim=-1,
        )
    if isinstance(field, TorusField):
        minor_angle = 2.0 * torch.pi * first
        ring = field.major_radius + field.minor_radius * torch.cos(minor_angle)
        return torch.stack(
            (
                ring * torch.cos(azimuth),
                ring * torch.sin(azimuth),
                field.minor_radius * torch.sin(minor_angle),
            ),
            dim=-1,
        )
    raise TypeError("hierarchical layout supports sphere and torus fields")


def wendland_values(offsets: Tensor, radii: Tensor) -> Tensor:
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    q = distances / radii
    one_minus_q = (1.0 - q).clamp_min(0.0)
    return one_minus_q**4 * (4.0 * q + 1.0)


def wendland_gradients(offsets: Tensor, radii: Tensor) -> Tensor:
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    q = distances / radii
    derivative = torch.where(
        q < 1.0,
        -20.0 * q * (1.0 - q).clamp_min(0.0) ** 3,
        torch.zeros_like(q),
    )
    return (
        derivative[:, None]
        / radii[:, None]
        * offsets
        / distances[:, None].clamp_min(1e-15)
    )


@dataclass(frozen=True)
class BasisLayout:
    centers: Tensor
    radii: Tensor

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError("basis centers must have shape (K, 3)")
        if self.radii.shape != (self.centers.shape[0],):
            raise ValueError("basis radii must have shape (K,)")
        if bool((self.radii <= 0.0).any()):
            raise ValueError("basis radii must be positive")

    @property
    def count(self) -> int:
        return self.centers.shape[0]


@dataclass
class SupportPairs:
    """Sparse point/basis incidence with both point- and basis-major order."""

    point_ids: Tensor
    basis_ids: Tensor
    point_order: Tensor
    point_offsets: Tensor
    basis_order: Tensor
    basis_offsets: Tensor
    point_count: int
    basis_count: int

    @classmethod
    def from_pairs(
        cls,
        point_ids: Tensor,
        basis_ids: Tensor,
        point_count: int,
        basis_count: int,
    ) -> "SupportPairs":
        point_order = torch.argsort(point_ids, stable=True)
        basis_order = torch.argsort(basis_ids, stable=True)
        point_counts = torch.bincount(point_ids, minlength=point_count)
        basis_counts = torch.bincount(basis_ids, minlength=basis_count)
        point_offsets = torch.cat(
            (point_counts.new_zeros(1), torch.cumsum(point_counts, dim=0))
        )
        basis_offsets = torch.cat(
            (basis_counts.new_zeros(1), torch.cumsum(basis_counts, dim=0))
        )
        return cls(
            point_ids,
            basis_ids,
            point_order,
            point_offsets,
            basis_order,
            basis_offsets,
            point_count,
            basis_count,
        )

    @property
    def count(self) -> int:
        return self.point_ids.numel()

    def points_for_basis(self, basis: int) -> Tensor:
        start = int(self.basis_offsets[basis])
        stop = int(self.basis_offsets[basis + 1])
        selected = self.basis_order[start:stop]
        return self.point_ids[selected]

    def subset_points(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Return local-point and basis IDs for sorted unique global point IDs."""
        if points.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.point_ids.device)
            return empty, empty
        starts = self.point_offsets[points]
        stops = self.point_offsets[points + 1]
        counts = stops - starts
        repeated = torch.repeat_interleave(
            torch.arange(points.numel(), device=points.device), counts
        )
        flat_starts = torch.cumsum(counts, 0) - counts
        offsets = torch.arange(
            int(counts.sum()), dtype=torch.long, device=points.device
        ) - torch.repeat_interleave(flat_starts, counts)
        pair_positions = torch.repeat_interleave(starts, counts) + offsets
        selected = self.point_order[pair_positions]
        return repeated, self.basis_ids[selected]


class UniformGridIndex:
    """Simple GPU cell list for deterministic compact-support radius queries."""

    def __init__(self, centers: Tensor, cell_size: float) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell size must be positive")
        self.centers = centers
        self.cell_size = float(cell_size)
        padding = 2.0 * self.cell_size
        self.lower = centers.amin(dim=0) - padding
        upper = centers.amax(dim=0) + padding
        self.dimensions = (
            torch.ceil((upper - self.lower) / self.cell_size).to(torch.long) + 1
        )
        cells = torch.floor((centers - self.lower) / self.cell_size).to(torch.long)
        keys = self._keys(cells)
        self.order = torch.argsort(keys, stable=True)
        self.sorted_keys = keys[self.order]
        self.neighbor_offsets = torch.tensor(
            list(itertools.product((-1, 0, 1), repeat=3)),
            dtype=torch.long,
            device=centers.device,
        )

    def _keys(self, cells: Tensor) -> Tensor:
        return (
            (cells[..., 0] * self.dimensions[1] + cells[..., 1])
            * self.dimensions[2]
            + cells[..., 2]
        )

    def query(
        self,
        points: Tensor,
        radius: float,
        *,
        chunk_size: int = 65536,
    ) -> SupportPairs:
        if radius > self.cell_size + 1e-12:
            raise ValueError("query radius must not exceed grid cell size")
        point_parts: list[Tensor] = []
        basis_parts: list[Tensor] = []
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            local = points[start:stop]
            cells = torch.floor((local - self.lower) / self.cell_size).to(torch.long)
            neighbor_cells = cells[:, None, :] + self.neighbor_offsets[None, :, :]
            valid = (
                (neighbor_cells >= 0)
                & (neighbor_cells < self.dimensions[None, None, :])
            ).all(dim=-1)
            keys = self._keys(neighbor_cells)
            keys = torch.where(valid, keys, torch.full_like(keys, -1)).reshape(-1)
            lower = torch.searchsorted(self.sorted_keys, keys, right=False)
            upper = torch.searchsorted(self.sorted_keys, keys, right=True)
            counts = upper - lower
            slots = torch.repeat_interleave(
                torch.arange(keys.numel(), device=points.device), counts
            )
            flat_starts = torch.cumsum(counts, 0) - counts
            offsets = torch.arange(
                int(counts.sum()), dtype=torch.long, device=points.device
            ) - torch.repeat_interleave(flat_starts, counts)
            center_positions = torch.repeat_interleave(lower, counts) + offsets
            candidate_basis = self.order[center_positions]
            candidate_points = torch.div(slots, 27, rounding_mode="floor")
            distances = torch.linalg.vector_norm(
                local[candidate_points] - self.centers[candidate_basis], dim=-1
            )
            keep = distances <= radius
            point_parts.append(candidate_points[keep] + start)
            basis_parts.append(candidate_basis[keep])
        point_ids = torch.cat(point_parts)
        basis_ids = torch.cat(basis_parts)
        return SupportPairs.from_pairs(
            point_ids, basis_ids, points.shape[0], self.centers.shape[0]
        )


@dataclass
class LocalZeroSet:
    """Zero-set coefficients evaluated through a fixed local support table."""

    base: ZeroSetField
    layout: BasisLayout
    coefficients: Tensor
    reference_points: Tensor
    reference_normals: Tensor
    support: SupportPairs

    @property
    def parameter_count(self) -> int:
        return self.layout.count

    def with_coefficients(self, coefficients: Tensor) -> "LocalZeroSet":
        return LocalZeroSet(
            self.base,
            self.layout,
            coefficients,
            self.reference_points,
            self.reference_normals,
            self.support,
        )

    def values_and_gradients(self, points: Tensor) -> tuple[Tensor, Tensor]:
        if points.shape != self.reference_points.shape:
            raise ValueError("local zero-set queries must preserve reference identities")
        values = self.base.value(points)
        gradients = self.base.gradient(points)
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

    def deform(
        self,
        *,
        iterations: int = 10,
        max_offset: float = 0.015,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        displacement = torch.zeros(
            self.reference_points.shape[0],
            dtype=self.reference_points.dtype,
            device=self.reference_points.device,
        )
        for _ in range(iterations):
            points = (
                self.reference_points
                + displacement[:, None] * self.reference_normals
            )
            values, gradients = self.values_and_gradients(points)
            denominator = (gradients * self.reference_normals).sum(dim=-1)
            safe = denominator.abs() > 1e-8
            step = torch.where(safe, values / denominator, torch.zeros_like(values))
            displacement = (displacement - step).clamp(-max_offset, max_offset)
        points = self.reference_points + displacement[:, None] * self.reference_normals
        values, gradients = self.values_and_gradients(points)
        denominator = (gradients * self.reference_normals).sum(dim=-1)
        success = (values.abs() < 1e-8) & (denominator.abs() > 1e-8)
        return points, success, displacement, denominator


def make_support(
    points: Tensor,
    layout: BasisLayout,
    *,
    margin: float,
) -> tuple[UniformGridIndex, SupportPairs]:
    radius = float(layout.radii.max()) + margin
    index = UniformGridIndex(layout.centers, radius)
    return index, index.query(points, radius)


def support_radius(base_radius: float, parameter_count: int) -> float:
    return base_radius * math.sqrt(32.0 / parameter_count)


def locality_cpu_verification() -> dict[str, object]:
    """Compare sparse local values, gradients, and incidence against dense K=32."""
    device = torch.device("cpu")
    base = SphereField()
    centers = hierarchical_surface_points(base, 32, device)
    points = hierarchical_surface_points(base, 256, device)
    normals = unit_normals(base, points)
    radii = torch.full((32,), 0.60, dtype=torch.float64)
    coefficients = 0.002 * torch.sin(torch.arange(32, dtype=torch.float64))
    layout = BasisLayout(centers, radii)
    _, support = make_support(points, layout, margin=0.005)
    dense_pairs = torch.nonzero(torch.cdist(points, centers) <= 0.605)
    sparse_encoded = torch.sort(support.point_ids * 32 + support.basis_ids).values
    dense_encoded = torch.sort(dense_pairs[:, 0] * 32 + dense_pairs[:, 1]).values
    local = LocalZeroSet(base, layout, coefficients, points, normals, support)
    dense = LocalBasisField(base, centers, radii, coefficients)
    values, gradients = local.values_and_gradients(points)
    deformed, success, _, _ = local.deform()
    residual, _ = local.values_and_gradients(deformed)
    report = {
        "support_indices_exact": torch.equal(sparse_encoded, dense_encoded),
        "value_max_absolute_error": float((values - dense.value(points)).abs().max()),
        "gradient_max_absolute_error": float(
            (gradients - dense.gradient(points)).abs().max()
        ),
        "root_success": bool(success.all()),
        "root_max_absolute_residual": float(residual.abs().max()),
        "active_bases_per_query": support.count / points.shape[0],
    }
    if not (
        report["support_indices_exact"]
        and report["value_max_absolute_error"] < 1e-12
        and report["gradient_max_absolute_error"] < 1e-12
        and report["root_success"]
        and report["root_max_absolute_residual"] < 1e-8
    ):
        raise AssertionError(f"sparse locality verification failed: {report}")
    return report
