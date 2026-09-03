"""Finite planar photon detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass
class PlanarCamera:
    center: Tensor
    normal: Tensor
    right: Tensor
    up: Tensor
    width: float
    height: float
    resolution: tuple[int, int]

    def __post_init__(self) -> None:
        vectors = [self.center, self.normal, self.right, self.up]
        self.center, self.normal, self.right, self.up = [
            torch.as_tensor(v, dtype=torch.float64) for v in vectors
        ]
        self.normal = self.normal / torch.linalg.vector_norm(self.normal)
        self.right = self.right / torch.linalg.vector_norm(self.right)
        self.up = self.up / torch.linalg.vector_norm(self.up)
        dot_products = (self.normal @ self.right, self.normal @ self.up, self.right @ self.up)
        if any(abs(float(value)) > 1e-10 for value in dot_products):
            raise ValueError("camera normal, right, and up must be orthogonal")
        if self.width <= 0 or self.height <= 0 or min(self.resolution) <= 0:
            raise ValueError("camera dimensions and resolution must be positive")

    @property
    def pixel_count(self) -> int:
        return self.resolution[0] * self.resolution[1]

    def intersect(
        self, origins: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return valid mask, travel time, flattened pixel, and local (right, up) coordinates."""
        denominator = directions @ self.normal
        parallel = denominator.abs() < 1e-12
        safe_denominator = torch.where(parallel, torch.ones_like(denominator), denominator)
        times = ((self.center - origins) * self.normal).sum(dim=-1) / safe_denominator
        hits = origins + directions * times[:, None]
        relative = hits - self.center
        local = torch.stack((relative @ self.right, relative @ self.up), dim=-1)
        inside = (local[:, 0].abs() <= self.width / 2.0) & (local[:, 1].abs() <= self.height / 2.0)
        valid = (~parallel) & (times > 0.0) & inside

        rows, columns = self.resolution
        column = torch.floor((local[:, 0] / self.width + 0.5) * columns).to(torch.long)
        row = torch.floor((0.5 - local[:, 1] / self.height) * rows).to(torch.long)
        column = column.clamp(0, columns - 1)
        row = row.clamp(0, rows - 1)
        pixels = row * columns + column
        pixels = torch.where(valid, pixels, torch.full_like(pixels, -1))
        return valid, times, pixels, local
