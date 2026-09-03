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


@dataclass(frozen=True)
class LocalBasisField:
    """Analytic reference field plus compact Wendland C2 perturbations.

    The coefficients are ordinary tensors rather than an optimizer-owned model:
    v0.2 studies the geometry Jacobian, not inverse reconstruction.
    """

    base: ZeroSetField
    centers: Tensor
    radii: Tensor
    coefficients: Tensor

    def __post_init__(self) -> None:
        centers = torch.as_tensor(self.centers, dtype=torch.float64)
        radii = torch.as_tensor(self.radii, dtype=torch.float64)
        coefficients = torch.as_tensor(self.coefficients, dtype=torch.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("basis centers must have shape (K, 3)")
        if radii.shape != (centers.shape[0],) or torch.any(radii <= 0.0):
            raise ValueError("basis radii must be positive with shape (K,)")
        if coefficients.shape != (centers.shape[0],):
            raise ValueError("basis coefficients must have shape (K,)")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def parameter_count(self) -> int:
        return self.centers.shape[0]

    def with_coefficients(self, coefficients: Tensor) -> "LocalBasisField":
        return LocalBasisField(self.base, self.centers, self.radii, coefficients)

    def basis_values(self, points: Tensor) -> Tensor:
        distances = torch.linalg.vector_norm(
            points[..., None, :] - self.centers, dim=-1
        )
        q = distances / self.radii
        one_minus_q = (1.0 - q).clamp_min(0.0)
        return one_minus_q**4 * (4.0 * q + 1.0)

    def basis_gradients(self, points: Tensor) -> Tensor:
        offsets = points[..., None, :] - self.centers
        distances = torch.linalg.vector_norm(offsets, dim=-1)
        q = distances / self.radii
        # d/dq [(1-q)^4(4q+1)] = -20 q (1-q)^3 within support.
        radial_derivative = torch.where(
            q < 1.0,
            -20.0 * q * (1.0 - q).clamp_min(0.0) ** 3,
            torch.zeros_like(q),
        )
        unit_offsets = offsets / distances[..., None].clamp_min(1e-15)
        return radial_derivative[..., None] / self.radii[:, None] * unit_offsets

    def value(self, points: Tensor) -> Tensor:
        return self.base.value(points) + self.basis_values(points) @ self.coefficients

    def gradient(self, points: Tensor) -> Tensor:
        perturbation = torch.einsum(
            "...kd,k->...d", self.basis_gradients(points), self.coefficients
        )
        return self.base.gradient(points) + perturbation

    def sample_surface(self, count: int, generator: torch.Generator) -> Tensor:
        reference = self.base.sample_surface(count, generator)
        reference_normals = unit_normals(self.base, reference)
        points, success, _ = deform_reference_surface(
            self, reference, reference_normals
        )
        if not bool(success.all()):
            raise RuntimeError("local normal-line root failed while sampling surface")
        return points

    def color(self, points: Tensor) -> Tensor:
        return self.base.color(points)


def deform_reference_surface(
    field: LocalBasisField,
    reference_points: Tensor,
    reference_normals: Tensor,
    *,
    max_offset: float = 0.15,
    bracket_samples: int = 33,
    bisection_steps: int = 45,
) -> tuple[Tensor, Tensor, Tensor]:
    """Track the nearest sign-changing root on each reference normal line."""
    offsets = torch.linspace(
        -max_offset,
        max_offset,
        bracket_samples,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    candidates = (
        reference_points[:, None, :]
        + offsets[None, :, None] * reference_normals[:, None, :]
    )
    values = field.value(candidates)
    changes = values[:, :-1] * values[:, 1:] <= 0.0
    midpoints = 0.5 * (offsets[:-1] + offsets[1:])
    scores = torch.where(
        changes,
        midpoints.abs()[None, :],
        torch.full_like(values[:, :-1], torch.inf),
    )
    bracket_index = scores.argmin(dim=1)
    success = changes.any(dim=1)
    rows = torch.arange(reference_points.shape[0], device=reference_points.device)
    low = offsets[bracket_index]
    high = offsets[bracket_index + 1]
    low_value = values[rows, bracket_index]

    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        middle_points = reference_points + middle[:, None] * reference_normals
        middle_value = field.value(middle_points)
        left_contains_root = low_value * middle_value <= 0.0
        high = torch.where(left_contains_root, middle, high)
        low = torch.where(left_contains_root, low, middle)
        low_value = torch.where(left_contains_root, low_value, middle_value)

    roots = 0.5 * (low + high)
    points = reference_points + roots[:, None] * reference_normals
    residuals = field.value(points).abs()
    success = success & (residuals < 1e-8)
    return points, success, roots


def implicit_position_jacobian(
    field: LocalBasisField,
    points: Tensor,
    reference_normals: Tensor,
    *,
    denominator_threshold: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate dp_j/dlambda_k from the implicit-function relation."""
    denominator = (field.gradient(points) * reference_normals).sum(dim=-1)
    stable = denominator.abs() >= denominator_threshold
    ds_dlambda = -field.basis_values(points) / denominator[:, None]
    ds_dlambda = torch.where(stable[:, None], ds_dlambda, torch.zeros_like(ds_dlambda))
    dp_dlambda = reference_normals[:, None, :] * ds_dlambda[:, :, None]
    return dp_dlambda, stable, denominator
