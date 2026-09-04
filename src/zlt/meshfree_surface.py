"""Deterministic mesh-free sampling of trilinear zero sets.

The sampler sees only a scalar grid, its domain, and the exact gradient of the
trilinear interpolant.  It never reads marching-cubes vertices, faces, or area
weights.  Sign-changing cells provide coverage; safeguarded Newton projection
provides interior samples; an exact zero-crossing cell edge is the fallback.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import torch


Tensor = torch.Tensor


class TrilinearGridField(Protocol):
    grid: Tensor
    lower: Tensor
    upper: Tensor

    def value(self, points: Tensor) -> Tensor: ...

    def interpolant_gradient(self, points: Tensor) -> Tensor: ...


CELL_CORNERS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)
CELL_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _digest(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sign_changing_cells(grid: Tensor) -> Tensor:
    """Return every grid cell whose eight corner values bracket zero."""
    minimum = grid[:-1, :-1, :-1].clone()
    maximum = minimum.clone()
    for dx, dy, dz in CELL_CORNERS[1:]:
        values = grid[
            dx : grid.shape[0] - 1 + dx,
            dy : grid.shape[1] - 1 + dy,
            dz : grid.shape[2] - 1 + dz,
        ]
        minimum = torch.minimum(minimum, values)
        maximum = torch.maximum(maximum, values)
    changing = (minimum <= 0.0) & (maximum >= 0.0) & (maximum > minimum)
    return torch.nonzero(changing, as_tuple=False)


def zero_crossing_edge_count(grid: Tensor) -> int:
    count = 0
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        a = grid[tuple(first)]
        b = grid[tuple(second)]
        count += int(((a * b <= 0.0) & (a != b)).sum())
    return count


def meshfree_base_color(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Smooth interpretable RGB albedo tied to scene position, not a camera."""
    normalized = ((points - lower) / (upper - lower)).clamp(0.0, 1.0)
    red = 0.20 + 0.72 * normalized[:, 0]
    green = 0.18 + 0.70 * normalized[:, 1]
    blue = 0.28 + 0.62 * (1.0 - normalized[:, 2])
    return torch.stack((red, green, blue), 1).clamp(0.0, 1.0)


@dataclass(frozen=True)
class MeshFreeSurfaceState:
    points: Tensor
    normals: Tensor
    base_colors: Tensor
    emission_strengths: Tensor
    source_cell_ids: Tensor
    sign_changing_cell_count: int
    zero_crossing_edge_count: int
    requested_count: int
    valid_count: int
    newton_converged_count: int
    edge_fallback_count: int
    root_failure_count: int
    degenerate_normal_count: int
    nonfinite_count: int
    residual_max: float
    residual_mean: float
    residual_median: float
    residual_p95: float
    sampled_cell_fraction: float
    build_seconds: float
    peak_allocated_mib: float
    digest: str


def _selected_cell_corners(
    field: TrilinearGridField, cells: Tensor
) -> tuple[Tensor, Tensor]:
    offsets = torch.tensor(
        CELL_CORNERS, dtype=torch.long, device=cells.device
    )
    indices = cells[:, None, :] + offsets[None, :, :]
    nx, ny, nz = field.grid.shape
    linear = (indices[..., 0] * ny + indices[..., 1]) * nz + indices[..., 2]
    values = field.grid.reshape(-1)[linear]
    return offsets, values


def _edge_fallback_points(
    field: TrilinearGridField, cells: Tensor, corner_values: Tensor
) -> Tensor:
    edges = torch.tensor(CELL_EDGES, dtype=torch.long, device=cells.device)
    first_values = corner_values[:, edges[:, 0]]
    second_values = corner_values[:, edges[:, 1]]
    crossings = (first_values * second_values <= 0.0) & (
        first_values != second_values
    )
    if not bool(crossings.any(1).all()):
        raise RuntimeError("sign-changing cell without a crossing edge")
    edge_ids = crossings.to(torch.int64).argmax(1)
    chosen = edges[edge_ids]
    corner_offsets = torch.tensor(
        CELL_CORNERS, dtype=field.grid.dtype, device=cells.device
    )
    first_offset = corner_offsets[chosen[:, 0]]
    second_offset = corner_offsets[chosen[:, 1]]
    row = torch.arange(cells.shape[0], device=cells.device)
    first = first_values[row, edge_ids]
    second = second_values[row, edge_ids]
    fraction = first / (first - second)
    grid_coordinate = (
        cells.to(field.grid.dtype)
        + first_offset
        + fraction[:, None] * (second_offset - first_offset)
    )
    dimensions = torch.tensor(
        field.grid.shape, dtype=field.grid.dtype, device=cells.device
    )
    return field.lower + grid_coordinate / (dimensions - 1.0) * (
        field.upper - field.lower
    )


def sample_meshfree_zero_set(
    field: TrilinearGridField,
    count: int,
    *,
    newton_steps: int = 32,
    residual_tolerance: float = 1e-9,
    gradient_tolerance: float = 1e-10,
    sobol_scramble_seed: int | None = None,
) -> MeshFreeSurfaceState:
    """Sample sign-changing cells and project without any mesh scaffold."""
    if count < 1 or newton_steps < 1:
        raise ValueError("count and Newton steps must be positive")
    device = field.grid.device
    dtype = field.grid.dtype
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    cells = sign_changing_cells(field.grid)
    if cells.numel() == 0:
        raise RuntimeError("implicit grid contains no sign-changing cells")
    sequence = torch.quasirandom.SobolEngine(
        4,
        scramble=sobol_scramble_seed is not None,
        seed=sobol_scramble_seed,
    ).draw(count)
    sequence = sequence.to(dtype=dtype, device=device)
    choice = torch.floor(sequence[:, 0] * cells.shape[0]).to(torch.long)
    choice.clamp_max_(cells.shape[0] - 1)
    selected_cells = cells[choice]
    source_cell_ids = (
        (selected_cells[:, 0] * (field.grid.shape[1] - 1) + selected_cells[:, 1])
        * (field.grid.shape[2] - 1)
        + selected_cells[:, 2]
    )
    _, corner_values = _selected_cell_corners(field, selected_cells)
    fallback = _edge_fallback_points(field, selected_cells, corner_values)
    dimensions = torch.tensor(field.grid.shape, dtype=dtype, device=device)
    voxel = (field.upper - field.lower) / (dimensions - 1.0)
    cell_lower = field.lower + selected_cells.to(dtype) * voxel
    cell_upper = cell_lower + voxel
    points = cell_lower + sequence[:, 1:] * voxel
    for _ in range(newton_steps):
        values = field.value(points)
        gradients = field.interpolant_gradient(points)
        denominator = gradients.square().sum(1, keepdim=True)
        step = values[:, None] * gradients / denominator.clamp_min(1e-24)
        step_norm = torch.linalg.vector_norm(step, dim=1, keepdim=True)
        step *= (0.45 * voxel.min() / step_norm.clamp_min(1e-30)).clamp_max(1.0)
        proposal = points - step
        points = torch.minimum(torch.maximum(proposal, cell_lower), cell_upper)
    residual = field.value(points).abs()
    gradient = field.interpolant_gradient(points)
    gradient_norm = torch.linalg.vector_norm(gradient, dim=1)
    finite_newton = torch.isfinite(points).all(1) & torch.isfinite(residual)
    converged = (
        finite_newton
        & (residual <= residual_tolerance)
        & (gradient_norm > gradient_tolerance)
    )
    points = torch.where(converged[:, None], points, fallback)
    residual = field.value(points).abs()
    gradient = field.interpolant_gradient(points)
    gradient_norm = torch.linalg.vector_norm(gradient, dim=1)
    finite = (
        torch.isfinite(points).all(1)
        & torch.isfinite(residual)
        & torch.isfinite(gradient).all(1)
    )
    valid = finite & (residual <= residual_tolerance) & (
        gradient_norm > gradient_tolerance
    )
    normals = gradient / gradient_norm[:, None].clamp_min(1e-30)
    points = points[valid]
    normals = normals[valid]
    source_cell_ids = source_cell_ids[valid]
    residual = residual[valid]
    colors = meshfree_base_color(points, field.lower, field.upper)
    emission_strengths = torch.ones(
        points.shape[0], dtype=dtype, device=device
    )
    _sync(device)
    seconds = time.perf_counter() - started
    peak = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    digest = _digest(
        points, normals, colors, emission_strengths, source_cell_ids
    )
    return MeshFreeSurfaceState(
        points,
        normals,
        colors,
        emission_strengths,
        source_cell_ids,
        int(cells.shape[0]),
        zero_crossing_edge_count(field.grid),
        count,
        int(valid.sum()),
        int(converged.sum()),
        int((~converged).sum()),
        count - int(valid.sum()),
        int((gradient_norm <= gradient_tolerance).sum()),
        int((~finite).sum()),
        float(residual.max()) if residual.numel() else torch.inf,
        float(residual.mean()) if residual.numel() else torch.inf,
        float(residual.median()) if residual.numel() else torch.inf,
        float(torch.quantile(residual, 0.95)) if residual.numel() else torch.inf,
        int(torch.unique(source_cell_ids).numel()) / int(cells.shape[0]),
        seconds,
        peak,
        digest,
    )
