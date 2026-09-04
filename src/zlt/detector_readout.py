"""Passive detector readout of a frozen outgoing boundary light field.

This module intentionally has no zero-set, mesh, surface-point, projection, or
scene-intersection dependency.  A pixel defines an outgoing boundary ray state;
the detector locally interpolates already-existing H_Sigma samples.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .boundary_transport import ObservationSphere, transverse_basis
from .camera import PlanarCamera
from .light_field import SparseBoundaryLightField


Tensor = torch.Tensor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def detector_from_outgoing_direction(
    boundary: ObservationSphere,
    outgoing_direction: Tensor,
    *,
    resolution: tuple[int, int] = (256, 256),
    extent: float = 2.8,
    distance: float = 3.0,
) -> PlanarCamera:
    """Instantiate a detector after H exists, using only boundary coordinates."""
    outgoing = outgoing_direction / torch.linalg.vector_norm(outgoing_direction)
    right, up = transverse_basis(outgoing[None, :])
    center = boundary.center + distance * outgoing
    return PlanarCamera(center, -outgoing, right[0], up[0], extent, extent, resolution)


@dataclass(frozen=True)
class BoundaryRayQueries:
    boundary_positions: Tensor
    outgoing_direction: Tensor
    valid: Tensor


@dataclass(frozen=True)
class LightFieldReadout:
    image: Tensor
    mask: Tensor
    confidence: Tensor
    rays: BoundaryRayQueries
    angular_direction_ids: Tensor
    angular_distances_radians: Tensor
    readout_seconds: float
    peak_allocated_mib: float
    direct_support_fraction: float
    interpolated_support_fraction: float
    no_support_fraction: float
    field_digest_before: str
    field_digest_after: str


def detector_boundary_rays(
    camera: PlanarCamera, boundary: ObservationSphere
) -> BoundaryRayQueries:
    """Convert pixels to boundary phase-space queries without touching a scene."""
    device = camera.center.device
    dtype = camera.center.dtype
    rows, columns = camera.resolution
    horizontal = (
        (torch.arange(columns, dtype=dtype, device=device) + 0.5) / columns - 0.5
    ) * camera.width
    vertical = (
        0.5 - (torch.arange(rows, dtype=dtype, device=device) + 0.5) / rows
    ) * camera.height
    yy, xx = torch.meshgrid(vertical, horizontal, indexing="ij")
    origins = (
        camera.center
        + xx.reshape(-1, 1) * camera.right
        + yy.reshape(-1, 1) * camera.up
    )
    direction = camera.normal.expand_as(origins)
    relative = origins - boundary.center
    linear = (relative * direction).sum(1)
    constant = relative.square().sum(1) - boundary.radius * boundary.radius
    discriminant = linear.square() - constant
    valid = discriminant >= 0.0
    entry_time = -linear - torch.sqrt(discriminant.clamp_min(0.0))
    valid &= entry_time > 0.0
    positions = origins + entry_time[:, None] * direction
    return BoundaryRayQueries(positions, -camera.normal, valid)


def _lookup(field: SparseBoundaryLightField, keys: Tensor) -> tuple[Tensor, Tensor]:
    if field.keys.numel() == 0:
        return (
            torch.zeros((*keys.shape, 3), dtype=field.radiance.dtype, device=keys.device),
            torch.zeros(keys.shape, dtype=torch.bool, device=keys.device),
        )
    location = torch.searchsorted(field.keys, keys)
    safe = location.clamp_max(field.keys.numel() - 1)
    found = (location < field.keys.numel()) & (field.keys[safe] == keys)
    values = field.radiance[safe]
    values = torch.where(found[..., None], values, torch.zeros_like(values))
    return values, found


def _spatial_query_entries(
    row: Tensor,
    column: Tensor,
    resolution: tuple[int, int],
    bandwidth_pixels: float,
) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = resolution
    if bandwidth_pixels <= 0.0:
        base_row = torch.floor(row).to(torch.long)
        base_column = torch.floor(column).to(torch.long)
        fr = row - base_row.to(row.dtype)
        fc = column - base_column.to(column.dtype)
        query_rows = torch.stack((base_row, base_row, base_row + 1, base_row + 1), 1)
        query_columns = torch.stack(
            (base_column, base_column + 1, base_column, base_column + 1), 1
        )
        weights = torch.stack(
            ((1.0 - fr) * (1.0 - fc), (1.0 - fr) * fc, fr * (1.0 - fc), fr * fc),
            1,
        )
    else:
        radius = max(1, int(math.ceil(2.0 * bandwidth_pixels)))
        offsets = torch.arange(-radius, radius + 1, dtype=torch.long, device=row.device)
        grid_row, grid_column = torch.meshgrid(offsets, offsets, indexing="ij")
        offsets_2d = torch.stack((grid_row.reshape(-1), grid_column.reshape(-1)), 1)
        center_row = torch.round(row).to(torch.long)
        center_column = torch.round(column).to(torch.long)
        query_rows = center_row[:, None] + offsets_2d[None, :, 0]
        query_columns = center_column[:, None] + offsets_2d[None, :, 1]
        delta_row = query_rows.to(row.dtype) - row[:, None]
        delta_column = query_columns.to(column.dtype) - column[:, None]
        weights = torch.exp(
            -0.5 * (delta_row.square() + delta_column.square()) / bandwidth_pixels**2
        )
        weights /= weights.sum(1, keepdim=True).clamp_min(1e-30)
    valid = (
        (query_rows >= 0)
        & (query_rows < rows)
        & (query_columns >= 0)
        & (query_columns < columns)
    )
    pixels = query_rows.clamp(0, rows - 1) * columns + query_columns.clamp(0, columns - 1)
    return pixels, weights, valid


def read_light_field(
    field: SparseBoundaryLightField,
    camera: PlanarCamera,
    *,
    angular_neighbors: int = 4,
    angular_bandwidth_radians: float = 0.08,
    spatial_bandwidth_pixels: float = 0.0,
    support_threshold: float = 0.5,
) -> LightFieldReadout:
    """Apply M_c to H_Sigma; only boundary ray states and H are inspected."""
    if angular_neighbors < 1 or angular_neighbors > field.atlas.count:
        raise ValueError("invalid angular neighbor count")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support threshold must lie in [0, 1]")
    device = field.keys.device
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    digest_before = field.digest
    _sync(device)
    started = time.perf_counter()
    rays = detector_boundary_rays(camera, field.boundary)
    cosine = field.atlas.directions @ rays.outgoing_direction
    best_cosine, direction_ids = torch.topk(cosine, angular_neighbors)
    angular_distance = torch.acos(best_cosine.clamp(-1.0, 1.0))
    if angular_neighbors == 1 or angular_bandwidth_radians <= 0.0:
        angular_weight = torch.zeros_like(angular_distance)
        angular_weight[0] = 1.0
    else:
        angular_weight = torch.exp(
            -0.5 * (angular_distance / angular_bandwidth_radians).square()
        )
        angular_weight /= angular_weight.sum().clamp_min(1e-30)
    query_count = rays.boundary_positions.shape[0]
    color_sum = torch.zeros((query_count, 3), dtype=field.radiance.dtype, device=device)
    support_sum = torch.zeros(query_count, dtype=field.radiance.dtype, device=device)
    direct_found = torch.zeros(query_count, dtype=torch.bool, device=device)
    rows, columns = field.spatial_resolution
    pixels_per_direction = rows * columns
    relative = rays.boundary_positions - field.boundary.center
    # Parameterize the requested ray by its transverse detector coordinates.
    # Keeping (s,t) fixed while sampling neighboring angular slices is the
    # local two-plane/light-slab interpolation; it does not inspect a scene.
    closest = relative - (
        relative * rays.outgoing_direction
    ).sum(1, keepdim=True) * rays.outgoing_direction
    detector_horizontal = closest @ camera.right
    detector_vertical = closest @ camera.up
    for neighbor in range(direction_ids.numel()):
        direction_id = direction_ids[neighbor]
        column = (detector_horizontal / field.spatial_extent + 0.5) * columns - 0.5
        row = (0.5 - detector_vertical / field.spatial_extent) * rows - 0.5
        pixels, spatial_weight, spatial_valid = _spatial_query_entries(
            row, column, field.spatial_resolution, spatial_bandwidth_pixels
        )
        keys = direction_id * pixels_per_direction + pixels
        values, found = _lookup(field, keys)
        found &= spatial_valid
        weighted_support = (spatial_weight * found.to(spatial_weight.dtype)).sum(1)
        weighted_color = (
            spatial_weight[..., None] * values * found[..., None]
        ).sum(1)
        color_sum += angular_weight[neighbor] * weighted_color
        support_sum += angular_weight[neighbor] * weighted_support
        if neighbor == 0 and float(angular_distance[neighbor]) < 1e-7:
            direct_found = weighted_support >= support_threshold
    confidence = torch.where(rays.valid, support_sum, torch.zeros_like(support_sum))
    mask = confidence >= support_threshold
    image = color_sum / support_sum[:, None].clamp_min(1e-30)
    image = torch.where(mask[:, None], image, torch.zeros_like(image))
    _sync(device)
    seconds = time.perf_counter() - started
    supported = mask
    direct = supported & direct_found
    interpolated = supported & ~direct_found
    total = max(query_count, 1)
    peak = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    image = image.reshape(*camera.resolution, 3)
    mask = mask.reshape(*camera.resolution)
    confidence = confidence.reshape(*camera.resolution)
    return LightFieldReadout(
        image,
        mask,
        confidence,
        rays,
        direction_ids,
        angular_distance,
        seconds,
        peak,
        float(direct.sum()) / total,
        float(interpolated.sum()) / total,
        float((~supported).sum()) / total,
        digest_before,
        field.digest,
    )
