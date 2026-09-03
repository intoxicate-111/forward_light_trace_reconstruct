"""Analytic scalar fields and temporary zero-set sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


Tensor = torch.Tensor


class ZeroSetField(Protocol):
    """Interface used by the tracer; values are not assumed to be distances."""

    def value(self, points: Tensor) -> Tensor: ...

    def gradient(self, points: Tensor) -> Tensor: ...

    def sample_surface(self, count: int, generator: torch.Generator) -> Tensor: ...

    def color(self, points: Tensor) -> Tensor: ...


def unit_normals(field: ZeroSetField, points: Tensor) -> Tensor:
    gradients = field.gradient(points)
    return gradients / torch.linalg.vector_norm(gradients, dim=-1, keepdim=True).clamp_min(1e-15)


@dataclass(frozen=True)
class SphereField:
    radius: float = 1.0

    def value(self, points: Tensor) -> Tensor:
        return (points * points).sum(dim=-1) - self.radius**2

    def gradient(self, points: Tensor) -> Tensor:
        return 2.0 * points

    def sample_surface(self, count: int, generator: torch.Generator) -> Tensor:
        # Random-area sampling is deterministic through the caller-owned generator.
        z = 2.0 * torch.rand(count, generator=generator, dtype=torch.float64) - 1.0
        phi = 2.0 * torch.pi * torch.rand(count, generator=generator, dtype=torch.float64)
        radial = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        return self.radius * torch.stack(
            (radial * torch.cos(phi), radial * torch.sin(phi), z), dim=-1
        )

    def color(self, points: Tensor) -> Tensor:
        return (0.5 * (points / self.radius + 1.0)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class TorusField:
    major_radius: float = 1.2
    minor_radius: float = 0.45

    def value(self, points: Tensor) -> Tensor:
        radial = torch.linalg.vector_norm(points[..., :2], dim=-1)
        return (radial - self.major_radius) ** 2 + points[..., 2] ** 2 - self.minor_radius**2

    def gradient(self, points: Tensor) -> Tensor:
        x, y, z = points.unbind(dim=-1)
        radial = torch.sqrt(x * x + y * y).clamp_min(1e-15)
        scale = 2.0 * (radial - self.major_radius) / radial
        return torch.stack((scale * x, scale * y, 2.0 * z), dim=-1)

    def sample_surface(self, count: int, generator: torch.Generator) -> Tensor:
        # Uniform parameters are sufficient here: samples are temporary emitters,
        # not the persistent representation of the surface.
        u = 2.0 * torch.pi * torch.rand(count, generator=generator, dtype=torch.float64)
        v = 2.0 * torch.pi * torch.rand(count, generator=generator, dtype=torch.float64)
        ring = self.major_radius + self.minor_radius * torch.cos(v)
        return torch.stack(
            (ring * torch.cos(u), ring * torch.sin(u), self.minor_radius * torch.sin(v)),
            dim=-1,
        )

    def color(self, points: Tensor) -> Tensor:
        scale = torch.tensor(
            [
                self.major_radius + self.minor_radius,
                self.major_radius + self.minor_radius,
                self.minor_radius,
            ],
            dtype=points.dtype,
            device=points.device,
        )
        return (0.5 * (points / scale + 1.0)).clamp(0.0, 1.0)
