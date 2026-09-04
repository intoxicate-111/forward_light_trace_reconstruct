"""Sparse spatial-angular representation of outgoing boundary transport."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import torch

from .boundary_transport import (
    BoundaryTransportEvents,
    DirectionAtlas,
    ObservationSphere,
)


Tensor = torch.Tensor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class SparseBoundaryLightField:
    """Occupied (direction, transverse spatial cell) values of H_Sigma.

    ``keys`` are sorted flattened indices into a conceptual D x H x W tensor.
    Only occupied cells exist.  The original boundary events remain separately
    auditable and are not replaced by a dense camera-dependent image stack.
    """

    keys: Tensor
    radiance: Tensor
    mass: Tensor
    atlas: DirectionAtlas
    boundary: ObservationSphere
    spatial_extent: float
    spatial_resolution: tuple[int, int]
    source_event_digest: str
    digest: str
    build_seconds: float
    peak_allocated_mib: float
    source_event_count: int
    boundary_spatial_occupied_cells: int
    boundary_spatial_total_cells: int

    @property
    def occupied_bins(self) -> int:
        return int(self.keys.numel())

    @property
    def possible_bins(self) -> int:
        rows, columns = self.spatial_resolution
        return self.atlas.count * rows * columns

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.keys,
            self.radiance,
            self.mass,
            self.atlas.directions,
            self.atlas.right,
            self.atlas.up,
        )
        return sum(item.numel() * item.element_size() for item in tensors)


def _field_digest(
    keys: Tensor,
    radiance: Tensor,
    mass: Tensor,
    events: BoundaryTransportEvents,
    extent: float,
    resolution: tuple[int, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(events.digest.encode())
    for tensor in (keys, radiance, mass):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr((extent, resolution)).encode())
    return digest.hexdigest()


def _bilinear_entries(
    horizontal: Tensor,
    vertical: Tensor,
    extent: float,
    resolution: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    rows, columns = resolution
    column = (horizontal / extent + 0.5) * columns - 0.5
    row = (0.5 - vertical / extent) * rows - 0.5
    base_row = torch.floor(row).to(torch.long)
    base_column = torch.floor(column).to(torch.long)
    row_fraction = row - base_row.to(row.dtype)
    column_fraction = column - base_column.to(column.dtype)
    pixel_rows = torch.stack((base_row, base_row, base_row + 1, base_row + 1), 1)
    pixel_columns = torch.stack(
        (base_column, base_column + 1, base_column, base_column + 1), 1
    )
    weights = torch.stack(
        (
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        ),
        1,
    )
    valid = (
        (pixel_rows >= 0)
        & (pixel_rows < rows)
        & (pixel_columns >= 0)
        & (pixel_columns < columns)
        & (weights > 1e-12)
    )
    pixels = pixel_rows * columns + pixel_columns
    return pixels[valid], weights[valid]


def _boundary_occupancy(
    positions: Tensor, boundary: ObservationSphere, resolution: tuple[int, int]
) -> tuple[int, int]:
    relative = positions - boundary.center
    relative /= torch.linalg.vector_norm(relative, dim=1, keepdim=True).clamp_min(1e-30)
    latitude = torch.asin(relative[:, 1].clamp(-1.0, 1.0))
    longitude = torch.atan2(relative[:, 2], relative[:, 0])
    rows, columns = resolution
    row = (
        torch.floor((0.5 - latitude / math.pi) * rows)
        .to(torch.long)
        .clamp(0, rows - 1)
    )
    column = torch.floor(
        (longitude / (2.0 * math.pi) + 0.5) * columns
    ).to(torch.long)
    column = column.remainder(columns)
    occupied = torch.unique(row * columns + column).numel()
    return int(occupied), rows * columns


def build_sparse_boundary_light_field(
    events: BoundaryTransportEvents,
    *,
    spatial_extent: float = 2.8,
    spatial_resolution: tuple[int, int] = (256, 256),
    boundary_coverage_resolution: tuple[int, int] = (128, 256),
) -> SparseBoundaryLightField:
    """Compress boundary events into local sparse position-direction bins."""
    if spatial_extent <= 0.0 or min(spatial_resolution) < 1:
        raise ValueError("invalid light-field spatial discretization")
    device = events.boundary_positions.device
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    rows, columns = spatial_resolution
    pixels = rows * columns
    key_parts: list[Tensor] = []
    radiance_parts: list[Tensor] = []
    mass_parts: list[Tensor] = []
    # Process one angular slice at a time, so construction memory is O(HW)
    # rather than O(number_of_events x footprint).
    for direction_id in range(events.atlas.count):
        bounds = torch.searchsorted(
            events.direction_ids,
            torch.tensor(
                [direction_id, direction_id + 1],
                dtype=events.direction_ids.dtype,
                device=device,
            ),
        )
        start, stop = int(bounds[0]), int(bounds[1])
        if start == stop:
            continue
        positions = events.boundary_positions[start:stop] - events.boundary.center
        horizontal = positions @ events.atlas.right[direction_id]
        vertical = positions @ events.atlas.up[direction_id]
        entry_pixels, entry_weights = _bilinear_entries(
            horizontal, vertical, spatial_extent, spatial_resolution
        )
        event_colors = events.radiance[start:stop].repeat_interleave(4, 0)
        # _bilinear_entries removes only out-of-frame/tiny footprint entries.
        coordinate_column = (horizontal / spatial_extent + 0.5) * columns - 0.5
        coordinate_row = (0.5 - vertical / spatial_extent) * rows - 0.5
        base_row = torch.floor(coordinate_row).to(torch.long)
        base_column = torch.floor(coordinate_column).to(torch.long)
        candidate_rows = torch.stack((base_row, base_row, base_row + 1, base_row + 1), 1)
        candidate_columns = torch.stack(
            (base_column, base_column + 1, base_column, base_column + 1), 1
        )
        fractions_row = coordinate_row - base_row.to(coordinate_row.dtype)
        fractions_column = coordinate_column - base_column.to(coordinate_column.dtype)
        candidate_weights = torch.stack(
            (
                (1.0 - fractions_row) * (1.0 - fractions_column),
                (1.0 - fractions_row) * fractions_column,
                fractions_row * (1.0 - fractions_column),
                fractions_row * fractions_column,
            ),
            1,
        )
        valid = (
            (candidate_rows >= 0)
            & (candidate_rows < rows)
            & (candidate_columns >= 0)
            & (candidate_columns < columns)
            & (candidate_weights > 1e-12)
        ).reshape(-1)
        event_colors = event_colors[valid]
        event_weights = events.weights[start:stop].repeat_interleave(4)[valid]
        entry_weights = entry_weights * event_weights
        mass = torch.zeros(pixels, dtype=entry_weights.dtype, device=device)
        color_mass = torch.zeros((pixels, 3), dtype=entry_weights.dtype, device=device)
        mass.scatter_add_(0, entry_pixels, entry_weights)
        color_mass.index_add_(0, entry_pixels, entry_weights[:, None] * event_colors)
        occupied = mass > 0.0
        local_pixels = torch.nonzero(occupied, as_tuple=False).flatten()
        key_parts.append(direction_id * pixels + local_pixels)
        mass_parts.append(mass[occupied])
        radiance_parts.append(color_mass[occupied] / mass[occupied, None])
    keys = (
        torch.cat(key_parts)
        if key_parts
        else torch.empty(0, dtype=torch.long, device=device)
    )
    radiance = (
        torch.cat(radiance_parts)
        if radiance_parts
        else torch.empty((0, 3), dtype=events.radiance.dtype, device=device)
    )
    mass = (
        torch.cat(mass_parts)
        if mass_parts
        else torch.empty(0, dtype=events.weights.dtype, device=device)
    )
    boundary_occupied, boundary_total = _boundary_occupancy(
        events.boundary_positions, events.boundary, boundary_coverage_resolution
    )
    _sync(device)
    seconds = time.perf_counter() - started
    digest = _field_digest(
        keys, radiance, mass, events, spatial_extent, spatial_resolution
    )
    peak = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    return SparseBoundaryLightField(
        keys,
        radiance,
        mass,
        events.atlas,
        events.boundary,
        spatial_extent,
        spatial_resolution,
        events.digest,
        digest,
        seconds,
        peak,
        events.count,
        boundary_occupied,
        boundary_total,
    )
