"""Forward photon emission, zero-set absorption, and sparse image formation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import PlanarCamera
from .fields import ZeroSetField, unit_normals


Tensor = torch.Tensor


@dataclass
class PhotonBatch:
    origins: Tensor
    directions: Tensor
    colors: Tensor
    energies: Tensor
    emit_times: Tensor
    emitter_ids: Tensor

    @property
    def count(self) -> int:
        return self.origins.shape[0]


@dataclass
class TraceResult:
    emitted_count: int
    camera_candidate_count: int
    absorbed_count: int
    pixels: Tensor
    arrival_times: Tensor
    colors: Tensor
    energies: Tensor
    emitter_ids: Tensor
    photon_ids: Tensor

    @property
    def camera_hit_count(self) -> int:
        return self.pixels.numel()


@dataclass
class RenderResult:
    direct_image: Tensor
    sparse_image: Tensor
    transport: Tensor
    max_difference: float


def _tangent_basis(normals: Tensor) -> tuple[Tensor, Tensor]:
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=normals.dtype, device=normals.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    use_y = normals[:, 2].abs() > 0.9
    helper = torch.where(use_y[:, None], y_axis, z_axis)
    tangent = torch.linalg.cross(helper, normals, dim=-1)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-15)
    bitangent = torch.linalg.cross(normals, tangent, dim=-1)
    return tangent, bitangent


def emit_photons(
    points: Tensor,
    normals: Tensor,
    colors: Tensor,
    packets_per_emitter: int,
    cone_power: float,
    emission_interval: float,
    generator: torch.Generator,
) -> PhotonBatch:
    """Sample unit-speed directions with density proportional to max(0, n dot omega)^k."""
    if packets_per_emitter <= 0 or cone_power < 0.0 or emission_interval < 0.0:
        raise ValueError("invalid emission parameters")
    emitter_count = points.shape[0]
    origins = points.repeat_interleave(packets_per_emitter, dim=0)
    primary = normals.repeat_interleave(packets_per_emitter, dim=0)
    packet_colors = colors.repeat_interleave(packets_per_emitter, dim=0)
    emitter_ids = torch.arange(
        emitter_count, device=points.device
    ).repeat_interleave(packets_per_emitter)
    count = origins.shape[0]

    uniform = torch.rand(
        (count, 2), generator=generator, dtype=points.dtype, device=points.device
    )
    cos_theta = uniform[:, 0].pow(1.0 / (cone_power + 1.0))
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))
    azimuth = 2.0 * torch.pi * uniform[:, 1]
    tangent, bitangent = _tangent_basis(primary)
    directions = (
        cos_theta[:, None] * primary
        + (sin_theta * torch.cos(azimuth))[:, None] * tangent
        + (sin_theta * torch.sin(azimuth))[:, None] * bitangent
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    energies = torch.full(
        (count,), 1.0 / packets_per_emitter, dtype=points.dtype, device=points.device
    )
    if emission_interval == 0.0:
        emit_times = torch.zeros(count, dtype=points.dtype, device=points.device)
    else:
        emit_times = emission_interval * torch.rand(
            count, generator=generator, dtype=points.dtype, device=points.device
        )
    return PhotonBatch(origins, directions, packet_colors, energies, emit_times, emitter_ids)


def first_zero_set_intersections(
    field: ZeroSetField,
    origins: Tensor,
    directions: Tensor,
    maximum_times: Tensor,
    *,
    epsilon: float = 1e-4,
    samples: int = 64,
    bisection_steps: int = 28,
    chunk_size: int = 4096,
) -> tuple[Tensor, Tensor]:
    """Find earliest sign-changing roots after the source using samples and bisection.

    Tangent roots that do not change sign are intentionally undetected in v0.1.
    """
    if samples < 2:
        raise ValueError("at least two root samples are required")
    found_parts: list[Tensor] = []
    time_parts: list[Tensor] = []
    fractions = torch.linspace(0.0, 1.0, samples, dtype=origins.dtype, device=origins.device)

    for start in range(0, origins.shape[0], chunk_size):
        stop = min(start + chunk_size, origins.shape[0])
        local_origins = origins[start:stop]
        local_directions = directions[start:stop]
        local_maximum = maximum_times[start:stop]
        usable = local_maximum > 2.0 * epsilon
        end = (local_maximum - epsilon).clamp_min(epsilon)
        times = epsilon + fractions[None, :] * (end - epsilon)[:, None]
        points = local_origins[:, None, :] + local_directions[:, None, :] * times[:, :, None]
        values = field.value(points)
        changes = (values[:, :-1] * values[:, 1:] <= 0.0) & usable[:, None]
        found = changes.any(dim=1)
        first = changes.to(torch.int64).argmax(dim=1)
        row = torch.arange(stop - start, device=origins.device)
        low = times[row, first]
        high = times[row, first + 1]
        low_value = values[row, first]

        # Refinement is harmless for misses; their returned time is masked by `found`.
        for _ in range(bisection_steps):
            middle = 0.5 * (low + high)
            middle_value = field.value(local_origins + local_directions * middle[:, None])
            left_contains_root = low_value * middle_value <= 0.0
            high = torch.where(left_contains_root, middle, high)
            low = torch.where(left_contains_root, low, middle)
            low_value = torch.where(left_contains_root, low_value, middle_value)
        roots = 0.5 * (low + high)
        roots = torch.where(found, roots, torch.full_like(roots, torch.inf))
        found_parts.append(found)
        time_parts.append(roots)

    return torch.cat(found_parts), torch.cat(time_parts)


def trace_photons(
    field: ZeroSetField,
    camera: PlanarCamera,
    photons: PhotonBatch,
    *,
    epsilon: float = 1e-4,
    root_samples: int = 64,
) -> TraceResult:
    camera_valid, camera_times, pixels, _ = camera.intersect(photons.origins, photons.directions)
    candidate_indices = torch.nonzero(camera_valid, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=photons.origins.device)
        empty_float = torch.empty(
            0, dtype=photons.origins.dtype, device=photons.origins.device
        )
        empty_color = torch.empty(
            (0, 3), dtype=photons.origins.dtype, device=photons.origins.device
        )
        return TraceResult(
            photons.count,
            0,
            0,
            empty_long,
            empty_float,
            empty_color,
            empty_float,
            empty_long,
            empty_long,
        )

    absorbed, _ = first_zero_set_intersections(
        field,
        photons.origins[candidate_indices],
        photons.directions[candidate_indices],
        camera_times[candidate_indices],
        epsilon=epsilon,
        samples=root_samples,
    )
    survivors = candidate_indices[~absorbed]
    arrivals = photons.emit_times[survivors] + camera_times[survivors]
    return TraceResult(
        photons.count,
        candidate_indices.numel(),
        int(absorbed.sum()),
        pixels[survivors],
        arrivals,
        photons.colors[survivors],
        photons.energies[survivors],
        photons.emitter_ids[survivors],
        survivors,
    )


def render_first_arrival(
    trace: TraceResult,
    pixel_count: int,
    emitter_colors: Tensor,
    *,
    mode: str = "hard",
    beta: float = 50.0,
) -> RenderResult:
    """Build both direct and COO-sparse images from identical arrival weights."""
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    chosen_packets: list[Tensor] = []
    chosen_weights: list[Tensor] = []
    for pixel in torch.unique(trace.pixels):
        packet_indices = torch.nonzero(trace.pixels == pixel, as_tuple=False).flatten()
        local_times = trace.arrival_times[packet_indices]
        if mode == "hard":
            local_choice = torch.argmin(local_times)
            chosen_packets.append(packet_indices[local_choice : local_choice + 1])
            chosen_weights.append(
                torch.ones(1, dtype=emitter_colors.dtype, device=emitter_colors.device)
            )
        else:
            weights = torch.exp(-beta * (local_times - local_times.min()))
            weights = weights / weights.sum()
            chosen_packets.append(packet_indices)
            chosen_weights.append(weights)

    direct = torch.zeros(
        (pixel_count, 3),
        dtype=emitter_colors.dtype,
        device=emitter_colors.device,
    )
    if chosen_packets:
        packets = torch.cat(chosen_packets)
        weights = torch.cat(chosen_weights)
        selected_pixels = trace.pixels[packets]
        selected_emitters = trace.emitter_ids[packets]
        direct.index_add_(0, selected_pixels, weights[:, None] * trace.colors[packets])
        indices = torch.stack((selected_pixels, selected_emitters))
        transport = torch.sparse_coo_tensor(
            indices,
            weights,
            size=(pixel_count, emitter_colors.shape[0]),
            dtype=emitter_colors.dtype,
        ).coalesce()
    else:
        transport = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=emitter_colors.device),
            torch.empty(
                0, dtype=emitter_colors.dtype, device=emitter_colors.device
            ),
            size=(pixel_count, emitter_colors.shape[0]),
        ).coalesce()
    sparse_image = torch.sparse.mm(transport, emitter_colors)
    difference = float((direct - sparse_image).abs().max()) if direct.numel() else 0.0
    return RenderResult(direct, sparse_image, transport, difference)


def make_scene(
    field: ZeroSetField,
    camera: PlanarCamera,
    *,
    emitter_count: int,
    packets_per_emitter: int,
    cone_power: float,
    emission_interval: float,
    seed: int,
    root_samples: int,
    mode: str,
    beta: float,
) -> tuple[Tensor, Tensor, PhotonBatch, TraceResult, RenderResult]:
    """Run the complete deterministic forward pipeline."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    points = field.sample_surface(emitter_count, generator)
    normals = unit_normals(field, points)
    colors = field.color(points)
    photons = emit_photons(
        points, normals, colors, packets_per_emitter, cone_power, emission_interval, generator
    )
    trace = trace_photons(field, camera, photons, root_samples=root_samples)
    render = render_first_arrival(trace, camera.pixel_count, colors, mode=mode, beta=beta)
    return points, normals, photons, trace, render
