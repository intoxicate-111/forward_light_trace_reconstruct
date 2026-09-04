"""Camera-independent forward transport to an enclosing observation sphere.

This module owns the expensive scene operation in v0.5.  It accepts a zero-set
field, surface samples, and a global direction quadrature.  It never accepts a
camera.  Every retained event has survived a forward zero-set re-intersection
test before reaching the observation boundary.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Protocol

import torch

from .tracer import first_zero_set_intersections


Tensor = torch.Tensor


class ZeroSetEvaluator(Protocol):
    def value(self, points: Tensor) -> Tensor: ...


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def transverse_basis(directions: Tensor) -> tuple[Tensor, Tensor]:
    """Return deterministic right/up axes perpendicular to outgoing directions."""
    world_up = torch.tensor(
        [0.0, 1.0, 0.0], dtype=directions.dtype, device=directions.device
    ).expand_as(directions)
    alternate = torch.tensor(
        [0.0, 0.0, 1.0], dtype=directions.dtype, device=directions.device
    ).expand_as(directions)
    inward = -directions
    helper = torch.where(
        (inward * world_up).sum(1).abs()[:, None] > 0.9, alternate, world_up
    )
    right = torch.linalg.cross(inward, helper, dim=1)
    right /= torch.linalg.vector_norm(right, dim=1, keepdim=True).clamp_min(1e-30)
    up = torch.linalg.cross(right, inward, dim=1)
    return right, up


@dataclass(frozen=True)
class DirectionAtlas:
    """A global outgoing-direction quadrature created before any detector."""

    directions: Tensor
    right: Tensor
    up: Tensor
    levels: tuple[int, ...]
    digest: str

    @property
    def count(self) -> int:
        return int(self.directions.shape[0])


def nested_fibonacci_atlas(
    device: torch.device,
    levels: tuple[int, ...] = (8, 128),
) -> DirectionAtlas:
    """Build a deterministic nested spherical quadrature with no camera input."""
    if not levels or any(level < 1 for level in levels):
        raise ValueError("direction levels must be positive")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    blocks: list[Tensor] = []
    for count in levels:
        index = torch.arange(count, dtype=torch.float64, device=device)
        vertical = 1.0 - 2.0 * (index + 0.5) / count
        radial = torch.sqrt((1.0 - vertical.square()).clamp_min(0.0))
        angle = index * golden_angle
        blocks.append(
            torch.stack(
                (radial * torch.cos(angle), vertical, radial * torch.sin(angle)), 1
            )
        )
    directions = torch.cat(blocks, 0)
    # Different Fibonacci levels can only coincide accidentally.  Deterministic
    # first-occurrence filtering keeps the atlas definition transparent.
    keep: list[int] = []
    for index in range(directions.shape[0]):
        if (
            not keep
            or float((directions[keep] @ directions[index]).max()) < 1.0 - 1e-13
        ):
            keep.append(index)
    directions = directions[torch.tensor(keep, device=device)]
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    right, up = transverse_basis(directions)
    digest = hashlib.sha256(
        directions.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    return DirectionAtlas(directions, right, up, levels, digest)


@dataclass(frozen=True)
class ObservationSphere:
    center: Tensor
    radius: float

    def exit_times(self, origins: Tensor, directions: Tensor) -> Tensor:
        relative = origins - self.center
        linear = (relative * directions).sum(1)
        constant = relative.square().sum(1) - self.radius * self.radius
        discriminant = (linear.square() - constant).clamp_min(0.0)
        return -linear + torch.sqrt(discriminant)


def enclosing_observation_sphere(
    points: Tensor, margin: float = 1.25
) -> ObservationSphere:
    if margin <= 1.0:
        raise ValueError("observation-boundary margin must exceed one")
    lower = points.amin(0)
    upper = points.amax(0)
    center = 0.5 * (lower + upper)
    radius = margin * float(torch.linalg.vector_norm(points - center, dim=1).max())
    return ObservationSphere(center, radius)


@dataclass(frozen=True)
class BoundaryTransportEvents:
    """Sparse outgoing events on Sigma, generated with no cameras present."""

    boundary_positions: Tensor
    direction_ids: Tensor
    radiance: Tensor
    weights: Tensor
    owner_ids: Tensor
    path_lengths: Tensor
    atlas: DirectionAtlas
    boundary: ObservationSphere
    surface_sample_count: int
    emitted_count: int
    absorbed_count: int
    scene_seconds: float
    peak_allocated_mib: float
    digest: str

    @property
    def count(self) -> int:
        return int(self.owner_ids.numel())

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.boundary_positions,
            self.direction_ids,
            self.radiance,
            self.weights,
            self.owner_ids,
            self.path_lengths,
            self.atlas.directions,
            self.atlas.right,
            self.atlas.up,
        )
        return sum(item.numel() * item.element_size() for item in tensors)


def _event_digest(
    positions: Tensor,
    direction_ids: Tensor,
    radiance: Tensor,
    weights: Tensor,
    owner_ids: Tensor,
    path_lengths: Tensor,
    atlas: DirectionAtlas,
    boundary: ObservationSphere,
) -> str:
    digest = hashlib.sha256()
    for tensor in (
        positions,
        direction_ids,
        radiance,
        weights,
        owner_ids,
        path_lengths,
        atlas.directions,
        boundary.center,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(str(boundary.radius).encode())
    return digest.hexdigest()


def build_boundary_transport(
    field: ZeroSetEvaluator,
    surface_points: Tensor,
    surface_normals: Tensor,
    atlas: DirectionAtlas,
    boundary: ObservationSphere,
    *,
    root_samples: int = 16,
    bisection_steps: int = 18,
    collision_chunk_size: int = 8192,
    epsilon: float = 2e-4,
    constant_radiance: float = 0.72,
) -> BoundaryTransportEvents:
    """Trace global outgoing directions to Sigma without detector knowledge."""
    if (
        surface_points.shape != surface_normals.shape
        or surface_points.shape[1:] != (3,)
    ):
        raise ValueError("surface points/normals must be matching Nx3 tensors")
    if surface_points.device != atlas.directions.device:
        raise ValueError("surface state and direction atlas must share a device")
    device = surface_points.device
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    positions: list[Tensor] = []
    direction_ids: list[Tensor] = []
    radiance: list[Tensor] = []
    weights: list[Tensor] = []
    owners: list[Tensor] = []
    lengths: list[Tensor] = []
    emitted = 0
    absorbed = 0
    for direction_id in range(atlas.count):
        direction = atlas.directions[direction_id]
        outward = (surface_normals @ direction) > 1e-8
        owner = torch.nonzero(outward, as_tuple=False).flatten()
        if owner.numel() == 0:
            continue
        origins = surface_points[owner]
        directions = direction.expand_as(origins)
        maximum = boundary.exit_times(origins, directions)
        hit, _ = first_zero_set_intersections(
            field,
            origins,
            directions,
            maximum,
            epsilon=epsilon,
            samples=root_samples,
            bisection_steps=bisection_steps,
            chunk_size=collision_chunk_size,
        )
        survive = ~hit
        survivor_owner = owner[survive]
        survivor_length = maximum[survive]
        survivor_position = origins[survive] + survivor_length[:, None] * direction
        count = int(survivor_owner.numel())
        emitted += int(owner.numel())
        absorbed += int(hit.sum())
        positions.append(survivor_position)
        direction_ids.append(
            torch.full((count,), direction_id, dtype=torch.long, device=device)
        )
        radiance.append(
            torch.full(
                (count, 3), constant_radiance, dtype=surface_points.dtype, device=device
            )
        )
        weights.append(torch.ones(count, dtype=surface_points.dtype, device=device))
        owners.append(survivor_owner)
        lengths.append(survivor_length)
    empty_vector = torch.empty((0, 3), dtype=surface_points.dtype, device=device)
    empty_scalar = torch.empty(0, dtype=surface_points.dtype, device=device)
    event_positions = torch.cat(positions) if positions else empty_vector
    event_directions = (
        torch.cat(direction_ids)
        if direction_ids
        else torch.empty(0, dtype=torch.long, device=device)
    )
    event_radiance = torch.cat(radiance) if radiance else empty_vector
    event_weights = torch.cat(weights) if weights else empty_scalar
    event_owners = (
        torch.cat(owners)
        if owners
        else torch.empty(0, dtype=torch.long, device=device)
    )
    event_lengths = torch.cat(lengths) if lengths else empty_scalar
    _sync(device)
    seconds = time.perf_counter() - started
    event_digest = _event_digest(
        event_positions,
        event_directions,
        event_radiance,
        event_weights,
        event_owners,
        event_lengths,
        atlas,
        boundary,
    )
    peak = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    return BoundaryTransportEvents(
        event_positions,
        event_directions,
        event_radiance,
        event_weights,
        event_owners,
        event_lengths,
        atlas,
        boundary,
        int(surface_points.shape[0]),
        emitted,
        absorbed,
        seconds,
        peak,
        event_digest,
    )


def restrict_events_by_owner(
    events: BoundaryTransportEvents,
    surface_sample_count: int,
    surface_normals: Tensor,
) -> BoundaryTransportEvents:
    """Extract an exact nested-prefix field without repeating scene traversal."""
    if not 0 < surface_sample_count <= events.surface_sample_count:
        raise ValueError("invalid nested surface-sample count")
    keep = events.owner_ids < surface_sample_count
    positions = events.boundary_positions[keep]
    direction_ids = events.direction_ids[keep]
    radiance = events.radiance[keep]
    weights = events.weights[keep]
    owner_ids = events.owner_ids[keep]
    path_lengths = events.path_lengths[keep]
    digest = _event_digest(
        positions,
        direction_ids,
        radiance,
        weights,
        owner_ids,
        path_lengths,
        events.atlas,
        events.boundary,
    )
    if surface_normals.shape[0] < surface_sample_count:
        raise ValueError("insufficient normals for the requested prefix")
    emitted = int(
        (
            surface_normals[:surface_sample_count] @ events.atlas.directions.T
            > 1e-8
        ).sum()
    )
    absorbed = max(emitted - int(keep.sum()), 0)
    return BoundaryTransportEvents(
        positions,
        direction_ids,
        radiance,
        weights,
        owner_ids,
        path_lengths,
        events.atlas,
        events.boundary,
        surface_sample_count,
        emitted,
        absorbed,
        0.0,
        0.0,
        digest,
    )
