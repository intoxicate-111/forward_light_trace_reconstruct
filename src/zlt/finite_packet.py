"""v0.8.5 finite-support, root-free zero-set energy transport.

This module deliberately makes no novelty or physical-radiometry claim.  It
tests a continuous algorithmic transport operator which is linear in packet
energy for fixed geometry and nonlinear in the implicit geometry.
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from .benchmark import cuda_environment
from .boundary_transport import enclosing_observation_sphere, nested_fibonacci_atlas
from .corrected_birth import (
    CorrectedBirthConfig,
    _active_components,
    _build_context,
    _evaluate,
    _render,
    _visibility_cells,
)
from .high_sample import _json_ready, _prepare_surface, _release, _write_csv
from .locality import wendland_gradients, wendland_values
from .mesh_field import GridZeroSetField, prepare_stanford_bunny
from .meshfree_surface import meshfree_base_color, sample_meshfree_zero_set, sign_changing_cells
from .pixel_diagnostic import _silhouette_regions
from .sampling_diagnostic import _copy_layout


Tensor = torch.Tensor


def _progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


class Field(Protocol):
    lower: Tensor
    upper: Tensor

    def value(self, points: Tensor) -> Tensor: ...
    def gradient(self, points: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class FinitePacketConfig:
    surface_samples: int = 4_096
    views: int = 4
    resolutions: tuple[tuple[int, int], ...] = ((256, 256), (512, 512))
    seed: int = 101
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601, 701, 809)
    emitter_chunk_size: int = 256
    root_samples: int = 16
    bisection_steps: int = 18
    detector_extent: float = 2.8
    sensor_gain: float = 1.5
    ambient: float = 0.35
    levelset_eta_relative: float = 1e-6
    packet_radius_over_h: float = 1.0
    shell_width_over_r: float = 1.0
    micro_samples: int = 8
    path_step_over_epsilon: float = 0.5
    target_crossing_transmission: float = 1e-2
    launch_exclusion_over_r_plus_epsilon: float = 1.05
    radius_ratios: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    epsilon_over_r: tuple[float, ...] = (0.25, 0.5, 1.0)
    opacity_targets: tuple[float, ...] = (1e-1, 1e-2, 1e-3)
    path_step_ratios: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    micro_counts: tuple[int, ...] = (1, 8, 32, 128)
    perturbation_deltas: tuple[float, ...] = (1e-3, 3e-4)
    fd_epsilons: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    screen_samples: int = 512
    multiseed_samples: int = 1_024
    diagnostic_samples: int = 512
    diagnostic_views: int = 1
    diagnostic_resolution: int = 64
    scale_invariance_tolerance: float = 2e-10
    sphere_cap_tolerance: float = 2e-8
    energy_linearity_tolerance: float = 1e-14
    opacity_tolerance: float = 2e-4
    path_transmission_rmse_tolerance: float = 2e-2
    path_derivative_relative_tolerance: float = 0.15
    analytic_fd_relative_tolerance: float = 0.15
    topology_reduction_minimum: float = 0.20
    multiseed_response_cosine_minimum: float = 0.35


@dataclass
class SoftRender:
    images: dict[tuple[int, int], list[np.ndarray]]
    transmissions: list[np.ndarray]
    owners: list[np.ndarray]
    runtime_seconds: float
    interaction_evaluations: int
    attempted_packets: int
    outward_packets: int
    peak_allocated_mib: float
    peak_reserved_mib: float


class ScaledField:
    def __init__(self, field: Field, scale: float) -> None:
        self.field = field
        self.scale = scale
        self.lower = field.lower
        self.upper = field.upper

    def value(self, points: Tensor) -> Tensor:
        return self.scale * self.field.value(points)

    def gradient(self, points: Tensor) -> Tensor:
        return self.scale * self.field.gradient(points)


class ShiftedField:
    def __init__(self, field: Field, shift: float | Tensor) -> None:
        self.field = field
        self.shift = shift
        self.lower = field.lower
        self.upper = field.upper

    def value(self, points: Tensor) -> Tensor:
        return self.field.value(points) + self.shift

    def gradient(self, points: Tensor) -> Tensor:
        return self.field.gradient(points)


class LocalField:
    """Dense-at-query local basis field used only for small FD diagnostics."""

    def __init__(self, context: Any, coefficients: Tensor) -> None:
        self.context = context
        self.coefficients = coefficients
        self.lower = context.lower
        self.upper = context.upper
        self.layout, _ = _active_components(
            context,
            torch.arange(coefficients.numel(), device=coefficients.device),
        )

    def _basis(self, points: Tensor) -> tuple[Tensor, Tensor]:
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        values: list[Tensor] = []
        gradients: list[Tensor] = []
        for start in range(0, flat.shape[0], 32_768):
            stop = min(start + 32_768, flat.shape[0])
            offsets = flat[start:stop, None, :] - self.layout.centers[None, :, :]
            radii = self.layout.radii[None, :].expand(stop - start, -1)
            values.append(wendland_values(
                offsets.reshape(-1, 3), radii.reshape(-1)
            ).reshape(stop - start, -1))
            gradients.append(wendland_gradients(
                offsets.reshape(-1, 3), radii.reshape(-1)
            ).reshape(stop - start, -1, 3))
        return torch.cat(values).reshape(*shape, -1), torch.cat(gradients).reshape(*shape, -1, 3)

    def value(self, points: Tensor) -> Tensor:
        basis, _ = self._basis(points)
        return self.context.base.value(points) + basis @ self.coefficients

    def gradient(self, points: Tensor) -> Tensor:
        _, gradients = self._basis(points)
        return self.context.base.gradient(points) + (
            gradients * self.coefficients[..., None]
        ).sum(-2)


def _compact_shell(q: Tensor) -> Tensor:
    """Normalized compact C1 profile: integral psi(q)dq = 1."""
    absolute = q.abs()
    return torch.where(
        absolute < 1.0,
        (15.0 / 16.0) * (1.0 - absolute.square()).square(),
        torch.zeros_like(q),
    )


def _compact_shell_numpy(q: np.ndarray) -> np.ndarray:
    absolute = np.abs(q)
    return np.where(absolute < 1.0, 15.0 / 16.0 * (1.0 - absolute**2) ** 2, 0.0)


def _micro_offsets(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if count <= 1:
        return torch.zeros((1, 3), dtype=dtype, device=device)
    unit = torch.quasirandom.SobolEngine(3, scramble=False).draw(count).to(
        device=device, dtype=dtype
    )
    radius = unit[:, 0].clamp_min(1e-12).pow(1.0 / 3.0)
    cosine = 1.0 - 2.0 * unit[:, 1]
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    angle = 2.0 * math.pi * unit[:, 2]
    return radius[:, None] * torch.stack(
        (sine * torch.cos(angle), cosine, sine * torch.sin(angle)), 1
    )


def _eta(field: Field, reference_points: Tensor, relative: float) -> float:
    norms = torch.linalg.vector_norm(field.gradient(reference_points), dim=1)
    return relative * float(norms.median())


def _normalized_levelset(field: Field, points: Tensor, eta: float) -> tuple[Tensor, Tensor]:
    value = field.value(points)
    gradient = field.gradient(points)
    norm = torch.sqrt(gradient.square().sum(-1) + eta * eta)
    return value / norm.clamp_min(1e-30), gradient / norm[..., None].clamp_min(1e-30)


def _transmission(
    field: Field,
    origins: Tensor,
    directions: Tensor,
    maximum_times: Tensor,
    *,
    radius: float,
    epsilon: float,
    path_step: float,
    offsets: Tensor,
    eta: float,
    kappa: float,
    surface_barrier: bool,
    launch_exclusion_factor: float,
) -> tuple[Tensor, Tensor, int]:
    """Root-free path quadrature; no root identity or first hit is selected."""
    if origins.numel() == 0:
        empty = torch.empty(0, dtype=origins.dtype, device=origins.device)
        return empty, empty, 0
    start = launch_exclusion_factor * (radius + epsilon)
    available = (maximum_times - start).clamp_min(0.0)
    steps = max(1, int(math.ceil(float(available.max().detach()) / path_step)))
    t = start + (torch.arange(
        steps, dtype=origins.dtype, device=origins.device
    ) + 0.5) * path_step
    valid = t[None, :] < maximum_times[:, None]
    centers = origins[:, None, :] + t[None, :, None] * directions[:, None, :]
    samples = centers[:, :, None, :] + radius * offsets[None, None, :, :]
    flat = samples.reshape(-1, 3)
    distance, unit_gradient = _normalized_levelset(field, flat, eta)
    influence = _compact_shell(distance / epsilon) / epsilon
    if surface_barrier:
        repeated_direction = directions[:, None, None, :].expand(
            -1, steps, offsets.shape[0], -1
        ).reshape(-1, 3)
        influence = influence * (unit_gradient * repeated_direction).sum(1).abs()
    influence = influence.reshape(origins.shape[0], steps, offsets.shape[0]).mean(2)
    tau = kappa * (influence * valid).sum(1) * path_step
    return torch.exp(-tau), tau, flat.shape[0]


def _cubic(distance: Tensor) -> Tensor:
    absolute = distance.abs()
    inner = 2.0 / 3.0 - absolute.square() + 0.5 * absolute**3
    outer = (2.0 - absolute).clamp_min(0.0) ** 3 / 6.0
    return torch.where(absolute < 1.0, inner, outer)


def _accumulate_continuous(
    image: Tensor,
    points: Tensor,
    energies: Tensor,
    right: Tensor,
    up: Tensor,
    center: Tensor,
    extent: float,
    resolution: tuple[int, int],
) -> None:
    rows, columns = resolution
    relative = points - center
    row = (0.5 - relative @ up / extent) * rows - 0.5
    column = (relative @ right / extent + 0.5) * columns - 0.5
    base_row = torch.floor(row).long()
    base_column = torch.floor(column).long()
    offsets = torch.arange(-1, 3, device=points.device)
    row_ids = base_row[:, None] + offsets[None]
    column_ids = base_column[:, None] + offsets[None]
    row_weights = _cubic(row[:, None] - row_ids)
    column_weights = _cubic(column[:, None] - column_ids)
    pixel_rows = row_ids[:, :, None].expand(-1, 4, 4).reshape(-1, 16)
    pixel_columns = column_ids[:, None, :].expand(-1, 4, 4).reshape(-1, 16)
    weights = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(-1, 16)
    valid = ((pixel_rows >= 0) & (pixel_rows < rows)
             & (pixel_columns >= 0) & (pixel_columns < columns))
    pixels = pixel_rows.clamp(0, rows - 1) * columns + pixel_columns.clamp(0, columns - 1)
    image.index_add_(
        0,
        pixels.reshape(-1),
        (weights.mul(valid)[..., None] * energies[:, None, :]).reshape(-1, 3),
    )


def _soft_render(
    field: Field,
    points: Tensor,
    normals: Tensor,
    atlas: Any,
    boundary: Any,
    resolutions: tuple[tuple[int, int], ...],
    *,
    radius: float,
    epsilon: float,
    path_step: float,
    micro_samples: int,
    eta_relative: float,
    kappa: float,
    surface_barrier: bool,
    sensor_gain: float,
    ambient: float,
    detector_extent: float,
    chunk_size: int,
    launch_exclusion_factor: float,
    frozen_owners: list[Tensor] | None = None,
) -> SoftRender:
    device = points.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    offsets = _micro_offsets(micro_samples, device, points.dtype)
    eta = _eta(field, points.detach(), eta_relative)
    accumulators = {
        resolution: [
            torch.zeros((math.prod(resolution), 3), dtype=points.dtype, device=device)
            for _ in range(atlas.count)
        ]
        for resolution in resolutions
    }
    transmissions: list[np.ndarray] = []
    owners_cpu: list[np.ndarray] = []
    interactions = outward_total = 0
    for view_id in range(atlas.count):
        direction = atlas.directions[view_id]
        owners = (
            frozen_owners[view_id]
            if frozen_owners is not None
            else torch.nonzero(normals @ direction > 1e-8, as_tuple=False).flatten()
        )
        outward_total += int(owners.numel())
        view_transmission: list[Tensor] = []
        for start in range(0, owners.numel(), chunk_size):
            selected = owners[start : start + chunk_size]
            origins = points[selected]
            directions = direction.expand_as(origins)
            maximum = boundary.exit_times(origins, directions)
            transmission, _, evaluations = _transmission(
                field,
                origins,
                directions,
                maximum,
                radius=radius,
                epsilon=epsilon,
                path_step=path_step,
                offsets=offsets,
                eta=eta,
                kappa=kappa,
                surface_barrier=surface_barrier,
                launch_exclusion_factor=launch_exclusion_factor,
            )
            interactions += evaluations
            colors = meshfree_base_color(origins, field.lower, field.upper)
            cosine = (normals[selected] @ direction).clamp_min(0.0)
            radiance = colors * (ambient + (1.0 - ambient) * cosine)[:, None]
            energy = transmission[:, None] * radiance
            for resolution in resolutions:
                _accumulate_continuous(
                    accumulators[resolution][view_id], origins, energy,
                    atlas.right[view_id], atlas.up[view_id], boundary.center,
                    detector_extent, resolution,
                )
            view_transmission.append(transmission.detach())
        transmissions.append(torch.cat(view_transmission).cpu().numpy())
        owners_cpu.append(owners.detach().cpu().numpy())
    images: dict[tuple[int, int], list[np.ndarray]] = {}
    for resolution in resolutions:
        scale = sensor_gain * math.prod(resolution) / points.shape[0]
        images[resolution] = [
            (scale * image).reshape(*resolution, 3).detach().cpu().to(torch.float32).numpy()
            for image in accumulators[resolution]
        ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return SoftRender(
        images,
        transmissions,
        owners_cpu,
        time.perf_counter() - started,
        interactions,
        points.shape[0] * atlas.count,
        outward_total,
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
    )


def _cap_fraction(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    middle = ((1.0 - s) ** 2 * (2.0 + s)) / 4.0
    return np.where(s >= 1.0, 0.0, np.where(s <= -1.0, 1.0, middle))


def _sphere_cap_test() -> dict[str, Any]:
    s = np.linspace(-1.25, 1.25, 10_001)
    analytic = _cap_fraction(s)
    # Independent cross-section integration: volume density along z/r is
    # 3/4(1-z^2); blocked means z >= s for this sign convention.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    numerical = []
    for value in s[::50]:
        lower = np.clip(value, -1.0, 1.0)
        if value >= 1.0:
            numerical.append(0.0)
        elif value <= -1.0:
            numerical.append(1.0)
        else:
            mapped = 0.5 * (nodes + 1.0) * (1.0 - lower) + lower
            numerical.append(float(
                0.5 * (1.0 - lower) * np.sum(weights * 0.75 * (1.0 - mapped**2))
            ))
    numerical_array = np.asarray(numerical)
    analytic_selected = _cap_fraction(s[::50])
    derivative = np.gradient(analytic, s)
    second = np.gradient(derivative, s)
    epsilon = 1e-5
    fd = (_cap_fraction(s + epsilon) - _cap_fraction(s - epsilon)) / (2 * epsilon)
    values32 = _cap_fraction(s.astype(np.float32)).astype(np.float32)
    boundary = np.argmin(np.abs(s - 1.0))
    return {
        "formula": "phi(s)=0 for s>=1; 1 for s<=-1; ((1-s)^2(2+s))/4 otherwise",
        "independent_gauss_legendre_max_absolute_error": float(
            np.max(np.abs(numerical_array - analytic_selected))
        ),
        "finite_difference_derivative_max_absolute_error": float(
            np.max(np.abs(fd - derivative))
        ),
        "float32_float64_max_absolute_error": float(
            np.max(np.abs(values32.astype(np.float64) - analytic))
        ),
        "value_jump_at_support_boundary": float(abs(analytic[boundary + 1] - analytic[boundary - 1])),
        "first_derivative_jump_at_support_boundary": float(abs(derivative[boundary + 1] - derivative[boundary - 1])),
        "maximum_second_derivative_magnitude": float(np.max(np.abs(second))),
        "samples": int(s.size),
        "plot": {"s": s.tolist(), "phi": analytic.tolist(), "transmission": (1.0 - analytic).tolist()},
    }


def _integrate_1d_shell(
    distance: np.ndarray,
    coordinate: np.ndarray,
    epsilon: float,
    kappa: float,
    angle: np.ndarray | float = 1.0,
) -> float:
    influence = _compact_shell_numpy(distance / epsilon) / epsilon
    return float(np.exp(-kappa * np.trapz(influence * angle, coordinate)))


def _toy_transitions(config: FinitePacketConfig) -> dict[str, Any]:
    kappa = -math.log(config.target_crossing_transmission)
    epsilon = 0.04
    x = np.linspace(0.0, 1.0, 4_001)

    plane_lambda = np.linspace(-0.15, 1.15, 1_301)
    plane_soft = np.asarray([
        _integrate_1d_shell(x - value, x, epsilon, kappa)
        for value in plane_lambda
    ])
    plane_hard = ((plane_lambda > 0.0) & (plane_lambda < 1.0)).astype(np.float64)
    plane_hard = 1.0 - plane_hard

    packet_radius = 0.08
    packet_offsets = _micro_offsets(32, torch.device("cpu"), torch.float64).numpy()
    sphere_lambda = np.linspace(0.25, 0.75, 1_001)
    sphere_radius = 0.50
    centered_x = np.linspace(-1.2, 1.2, 2_001)
    sphere_soft = []
    for offset in sphere_lambda:
        x3 = centered_x[:, None] + packet_radius * packet_offsets[None, :, 0]
        y3 = offset + packet_radius * packet_offsets[None, :, 1]
        z3 = packet_radius * packet_offsets[None, :, 2]
        radial = np.sqrt(x3**2 + y3**2 + z3**2)
        distance = radial - sphere_radius
        gradient_dot = np.abs(x3) / np.maximum(radial, 1e-30)
        density = (_compact_shell_numpy(distance / epsilon) / epsilon * gradient_dot).mean(1)
        sphere_soft.append(float(np.exp(-kappa * np.trapz(density, centered_x))))
    sphere_soft = np.asarray(sphere_soft)
    sphere_hard = (sphere_lambda >= sphere_radius).astype(np.float64)

    birth_lambda = np.linspace(-0.10, 0.10, 1_001)
    birth_x = np.linspace(-0.8, 0.8, 2_001)
    birth_soft = []
    root_counts = []
    eta = 1e-4
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(96)
    packet_rho = 0.5 * packet_radius * (radial_nodes + 1.0)
    packet_rho_weights = (
        0.5 * packet_radius * radial_weights
        * 3.0 * packet_rho * np.sqrt(np.maximum(packet_radius**2 - packet_rho**2, 0.0))
        / packet_radius**3
    )
    for value in birth_lambda:
        radial2 = birth_x[:, None] ** 2 + packet_rho[None, :] ** 2
        field_value = radial2 + value
        gradient_norm = np.sqrt(4.0 * radial2 + eta**2)
        distance = field_value / gradient_norm
        angle = np.abs(2.0 * birth_x[:, None]) / gradient_norm
        density = (
            _compact_shell_numpy(distance / epsilon) / epsilon * angle
            * packet_rho_weights[None, :]
        ).sum(1)
        birth_soft.append(float(np.exp(-kappa * np.trapz(density, birth_x))))
        root_counts.append(2 if value < 0.0 else (1 if value == 0.0 else 0))
    birth_soft = np.asarray(birth_soft)
    root_counts = np.asarray(root_counts)
    birth_hard = (root_counts == 0).astype(np.float64)

    separation = np.linspace(0.08, 0.45, 256)
    two_surface = []
    single_surface = []
    for gap in separation:
        f = (centered_x + gap / 2.0) * (centered_x - gap / 2.0)
        g = 2.0 * centered_x
        d = f / np.sqrt(g**2 + eta**2)
        angle = np.abs(g) / np.sqrt(g**2 + eta**2)
        two_surface.append(_integrate_1d_shell(d, centered_x, epsilon, kappa, angle))
        single_surface.append(config.target_crossing_transmission)
    two_surface = np.asarray(two_surface)

    def metrics(parameter: np.ndarray, hard: np.ndarray, soft: np.ndarray) -> dict[str, float]:
        derivative = np.gradient(soft, parameter)
        fd_derivatives = []
        for stride in (1, 2, 4):
            coarse = np.gradient(soft[::stride], parameter[::stride])
            fd_derivatives.append(np.interp(parameter, parameter[::stride], coarse))
        reference_fd = fd_derivatives[0]
        fd_spread = max(
            np.linalg.norm(item - reference_fd) / max(np.linalg.norm(reference_fd), 1e-30)
            for item in fd_derivatives[1:]
        )
        return {
            "hard_maximum_value_jump": float(np.max(np.abs(np.diff(hard)))),
            "soft_maximum_adjacent_value_jump": float(np.max(np.abs(np.diff(soft)))),
            "soft_maximum_adjacent_derivative_jump": float(np.max(np.abs(np.diff(derivative)))),
            "soft_transmission_minimum": float(soft.min()),
            "soft_transmission_maximum": float(soft.max()),
            "hard_vs_soft_mean_absolute_bias": float(np.mean(np.abs(hard - soft))),
            "finite_difference_relative_spread_across_1x_2x_4x_steps": float(fd_spread),
            "grid_spacing": float(parameter[1] - parameter[0]),
        }

    return {
        "moving_plane": {
            **metrics(plane_lambda, plane_hard, plane_soft),
            "plot": {"lambda": plane_lambda.tolist(), "hard": plane_hard.tolist(), "soft": plane_soft.tolist(), "derivative": np.gradient(plane_soft, plane_lambda).tolist()},
        },
        "grazing_sphere": {
            **metrics(sphere_lambda, sphere_hard, sphere_soft),
            "tangent_parameter": sphere_radius,
            "plot": {"lambda": sphere_lambda.tolist(), "hard": sphere_hard.tolist(), "soft": sphere_soft.tolist(), "derivative": np.gradient(sphere_soft, sphere_lambda).tolist()},
        },
        "root_birth_death": {
            **metrics(birth_lambda, birth_hard, birth_soft),
            "critical_parameter": 0.0,
            "root_counts": {"before": 2, "critical": 1, "after": 0},
            "packet_radius": packet_radius,
            "plot": {"lambda": birth_lambda.tolist(), "roots": root_counts.tolist(), "hard": birth_hard.tolist(), "soft": birth_soft.tolist(), "derivative": np.gradient(birth_soft, birth_lambda).tolist()},
        },
        "two_surface_path": {
            "expected_well_separated_transmission": config.target_crossing_transmission**2,
            "measured_well_separated_transmission": float(two_surface[-1]),
            "relative_error": abs(float(two_surface[-1]) - config.target_crossing_transmission**2) / config.target_crossing_transmission**2,
            "first_root_selected": False,
            "plot": {"separation": separation.tolist(), "transmission": two_surface.tolist(), "single_surface": single_surface},
        },
    }


def _local_plane_overlap_test(config: FinitePacketConfig) -> dict[str, Any]:
    """Compare semi-analytic plane convolution with fixed ball quadrature."""
    radius = 1.0
    epsilon = 0.5
    centers = np.linspace(-2.0, 2.0, 401)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    analytic = []
    for center in centers:
        density = 0.75 * (1.0 - nodes**2)
        shell = _compact_shell_numpy((center + radius * nodes) / epsilon) / epsilon
        analytic.append(float(np.sum(weights * density * shell)))
    analytic = np.asarray(analytic)
    quadrature_rows = []
    for count in config.micro_counts:
        offsets = _micro_offsets(count, torch.device("cpu"), torch.float64).numpy()
        estimate = np.asarray([
            np.mean(_compact_shell_numpy((center + radius * offsets[:, 0]) / epsilon) / epsilon)
            for center in centers
        ])
        quadrature_rows.append({
            "micro_samples": count,
            "rmse_vs_local_plane_analytic": float(np.sqrt(np.mean((estimate - analytic) ** 2))),
            "maximum_absolute_error": float(np.max(np.abs(estimate - analytic))),
        })
    # A large-radius sphere is a low-curvature control for the tangent-plane
    # approximation.  The comparison is in normalized packet coordinates.
    sphere_radius = 10.0 * radius
    sphere_distance = np.sqrt((sphere_radius + centers) ** 2) - sphere_radius
    sphere_influence = np.asarray([
        np.mean(_compact_shell_numpy(
            (np.sqrt((sphere_radius + center + radius * _micro_offsets(
                128, torch.device("cpu"), torch.float64
            ).numpy()[:, 0]) ** 2
            + (radius * _micro_offsets(128, torch.device("cpu"), torch.float64).numpy()[:, 1]) ** 2
            + (radius * _micro_offsets(128, torch.device("cpu"), torch.float64).numpy()[:, 2]) ** 2)
             - sphere_radius) / epsilon
        ) / epsilon)
        for center in centers
    ])
    return {
        "local_plane_method": "Gauss-Legendre integral against uniform-ball marginal 3/4(1-z^2)",
        "packet_quadrature_method": "fixed unscrambled Sobol uniform-ball offsets",
        "radius": radius,
        "epsilon": epsilon,
        "rows": quadrature_rows,
        "low_curvature_sphere": {
            "sphere_radius_over_packet_radius": sphere_radius / radius,
            "rmse_vs_local_plane_analytic": float(np.sqrt(np.mean((sphere_influence - analytic) ** 2))),
            "maximum_absolute_error": float(np.max(np.abs(sphere_influence - analytic))),
            "signed_distance_axis_consistency_max_error": float(np.max(np.abs(sphere_distance - centers))),
        },
        "plot": {"center_distance": centers.tolist(), "analytic_influence": analytic.tolist()},
    }


def _analytic_levelset_scale_invariance(config: FinitePacketConfig) -> dict[str, Any]:
    coordinate = torch.linspace(-1.0, 1.0, 20_001, dtype=torch.float64)
    step = float(coordinate[1] - coordinate[0])
    epsilon = 0.04
    kappa = -math.log(config.target_crossing_transmission)
    rows = []
    reference: dict[str, np.ndarray | float] | None = None
    for scale in (1.0, 0.5, 2.0, 10.0):
        eta_plane = config.levelset_eta_relative * scale
        plane_f = scale * coordinate
        plane_g = torch.full_like(coordinate, scale)
        plane_d = plane_f / torch.sqrt(plane_g.square() + eta_plane**2)
        plane_density = _compact_shell(plane_d / epsilon) / epsilon
        plane_tau = kappa * float((plane_density * plane_g.abs() /
                                   torch.sqrt(plane_g.square() + eta_plane**2)).sum()) * step
        radius = 0.55
        offset = 0.2
        radial = torch.sqrt(coordinate.square() + offset**2)
        sphere_f = scale * (radial - radius)
        sphere_grad_norm = torch.full_like(radial, scale)
        eta_sphere = config.levelset_eta_relative * scale
        sphere_d = sphere_f / torch.sqrt(sphere_grad_norm.square() + eta_sphere**2)
        sphere_angle = scale * coordinate.abs() / radial.clamp_min(1e-30)
        sphere_angle /= torch.sqrt(sphere_grad_norm.square() + eta_sphere**2)
        sphere_tau = kappa * float((
            _compact_shell(sphere_d / epsilon) / epsilon * sphere_angle
        ).sum()) * step
        gradient_probe = torch.autograd.functional.jacobian(
            lambda shift: torch.exp(-kappa * (
                _compact_shell((scale * (coordinate + shift)) /
                               math.sqrt(scale * scale + eta_plane * eta_plane) / epsilon)
                / epsilon * scale / math.sqrt(scale * scale + eta_plane * eta_plane)
            ).sum() * step),
            torch.tensor(0.013, dtype=torch.float64),
        )
        values = {
            "plane_tau": plane_tau,
            "plane_transmission": math.exp(-plane_tau),
            "sphere_tau": sphere_tau,
            "sphere_transmission": math.exp(-sphere_tau),
            "plane_translation_gradient": float(gradient_probe),
        }
        if reference is None:
            reference = values
        rows.append({
            "field_scale": scale,
            **values,
            "maximum_absolute_difference_vs_F": max(
                abs(float(values[key]) - float(reference[key])) for key in values
            ),
        })
    return {
        "rows": rows,
        "maximum_absolute_difference": max(row["maximum_absolute_difference_vs_F"] for row in rows),
    }


def _energy_linearity_test() -> dict[str, Any]:
    generator = np.random.default_rng(101)
    transmission = generator.uniform(0.0, 1.0, size=(4096, 1))
    first = generator.normal(size=(4096, 3))
    second = generator.normal(size=(4096, 3))
    scale_rows = []
    for scale in (0.25, 0.5, 2.0, 4.0):
        left = transmission * (scale * first)
        right = scale * (transmission * first)
        scale_rows.append({
            "scale": scale,
            "maximum_absolute_error": float(np.max(np.abs(left - right))),
            "maximum_relative_error": float(np.max(np.abs(left - right)) / max(np.max(np.abs(right)), 1e-30)),
        })
    a, b = 0.37, -1.21
    left = transmission * (a * first + b * second)
    right = a * transmission * first + b * transmission * second
    return {
        "claim_scope": "linear in transported packet energy for fixed geometry; nonlinear in geometry",
        "scale_tests": scale_rows,
        "superposition": {
            "a": a,
            "b": b,
            "maximum_absolute_error": float(np.max(np.abs(left - right))),
            "maximum_relative_error": float(np.max(np.abs(left - right)) / max(np.max(np.abs(right)), 1e-30)),
        },
    }


def _surface_spacing(points: Tensor) -> float:
    array = points.detach().cpu().numpy()
    distances = cKDTree(array).query(array, k=2, workers=1)[0][:, 1]
    return float(np.median(distances[distances > 0.0]))


def _hard_images(
    field: Any,
    points: Tensor,
    normals: Tensor,
    context: Any,
    atlas: Any,
    boundary: Any,
    resolution: tuple[int, int],
    *,
    recompute_visibility: bool,
    base_cells: list[Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    local_config = replace(
        context.config,
        views=atlas.count,
        resolution=resolution[0],
        resolution_width=resolution[1],
    )
    if recompute_visibility:
        cells, report = _visibility_cells(
            field, points, normals, atlas, boundary,
            local_config,
        )
    else:
        cells = base_cells or context.cells
        report = {"retained": sum(int(cell.owner_ids.numel()) for cell in cells)}
    cells = [replace(cell, resolution=resolution, support_mask=None) for cell in cells]
    _, renders = _render(points, normals, context, cells)
    scale = context.config.sensor_gain * math.prod(resolution) / points.shape[0]
    images = [
        (scale * render.numerator).reshape(*resolution, 3).detach().cpu().to(torch.float32).numpy()
        for render in renders
    ]
    return images, report


def _image_metrics(
    image: list[np.ndarray], reference: list[np.ndarray]
) -> dict[str, float]:
    totals = {name: [0.0, 0] for name in (
        "whole_image", "foreground", "interior_gt8px", "silhouette_0_8px"
    )}
    energy = brightness = 0.0
    foreground_pixels = 0
    edge = reference_edge = 0.0
    thin = reference_thin = 0.0
    for current, target in zip(image, reference):
        foreground, _, codes = _silhouette_regions(target)
        masks = {
            "whole_image": np.ones(foreground.shape, dtype=bool),
            "foreground": foreground,
            "interior_gt8px": codes == 5,
            "silhouette_0_8px": (codes >= 1) & (codes <= 4),
        }
        difference = current - target
        for name, mask in masks.items():
            totals[name][0] += float(np.square(difference[mask]).sum())
            totals[name][1] += int(mask.sum()) * 3
        energy += float(np.square(current).sum())
        brightness += float(current[foreground].sum())
        foreground_pixels += int(foreground.sum())
        gray = current.mean(2).astype(np.float64)
        target_gray = target.mean(2).astype(np.float64)
        dy, dx = np.gradient(gray)
        tdy, tdx = np.gradient(target_gray)
        band = masks["silhouette_0_8px"]
        edge += float(np.sqrt(dx * dx + dy * dy)[band].mean())
        reference_edge += float(np.sqrt(tdx * tdx + tdy * tdy)[band].mean())
        target_gradient = np.sqrt(tdx * tdx + tdy * tdy)
        thin_mask = target_gradient >= np.quantile(target_gradient[foreground], 0.95)
        thin += float(np.sqrt(dx * dx + dy * dy)[thin_mask].mean())
        reference_thin += float(target_gradient[thin_mask].mean())
    result = {
        f"{name}_mse": value / max(count, 1)
        for name, (value, count) in totals.items()
    }
    result.update({
        "total_image_energy": energy,
        "foreground_mean_brightness": brightness / max(foreground_pixels * 3, 1),
        "edge_sharpness": edge / len(image),
        "reference_edge_sharpness": reference_edge / len(image),
        "edge_sharpness_ratio": edge / max(reference_edge, 1e-30),
        "thin_feature_response_ratio": thin / max(reference_thin, 1e-30),
    })
    return result


def _soft_tensor_image(
    field: Field,
    points: Tensor,
    normals: Tensor,
    atlas: Any,
    boundary: Any,
    resolution: tuple[int, int],
    *,
    radius: float,
    epsilon: float,
    path_step: float,
    micro_samples: int,
    eta_relative: float,
    kappa: float,
    surface_barrier: bool,
    sensor_gain: float,
    ambient: float,
    detector_extent: float,
    chunk_size: int,
    launch_exclusion_factor: float,
    frozen_owners: list[Tensor] | None,
) -> Tensor:
    offsets = _micro_offsets(micro_samples, points.device, points.dtype)
    eta = _eta(field, points.detach(), eta_relative)
    rows, columns = resolution
    output: list[Tensor] = []
    for view_id in range(atlas.count):
        direction = atlas.directions[view_id]
        owners = frozen_owners[view_id] if frozen_owners is not None else torch.nonzero(
            normals @ direction > 1e-8, as_tuple=False
        ).flatten()
        chunks: list[tuple[Tensor, Tensor]] = []
        for start in range(0, owners.numel(), chunk_size):
            selected = owners[start : start + chunk_size]
            origins = points[selected]
            directions = direction.expand_as(origins)
            transmission, _, _ = _transmission(
                field, origins, directions, boundary.exit_times(origins, directions),
                radius=radius, epsilon=epsilon, path_step=path_step,
                offsets=offsets, eta=eta, kappa=kappa,
                surface_barrier=surface_barrier,
                launch_exclusion_factor=launch_exclusion_factor,
            )
            colors = meshfree_base_color(origins, field.lower, field.upper)
            cosine = (normals[selected] @ direction).clamp_min(0.0)
            energy = transmission[:, None] * colors * (
                ambient + (1.0 - ambient) * cosine
            )[:, None]
            chunks.append((origins, energy))
        accumulator = torch.zeros(
            (rows * columns, 3), dtype=points.dtype, device=points.device
        )
        for origins, energy in chunks:
            _accumulate_continuous(
                accumulator, origins, energy, atlas.right[view_id], atlas.up[view_id],
                boundary.center, detector_extent, resolution,
            )
        output.append(
            (sensor_gain * rows * columns / points.shape[0] * accumulator).reshape(-1)
        )
    return torch.cat(output)


def _render_parameters(
    config: FinitePacketConfig,
    h: float,
) -> dict[str, float]:
    radius = config.packet_radius_over_h * h
    epsilon = config.shell_width_over_r * radius
    return {
        "h": h,
        "radius": radius,
        "radius_over_h": radius / h,
        "epsilon": epsilon,
        "epsilon_over_radius": epsilon / radius,
        "path_step": config.path_step_over_epsilon * epsilon,
        "path_step_over_epsilon": config.path_step_over_epsilon,
        "kappa": -math.log(config.target_crossing_transmission),
        "target_crossing_transmission": config.target_crossing_transmission,
    }


def _path_and_micro_convergence(
    field: Field,
    points: Tensor,
    normals: Tensor,
    atlas: Any,
    boundary: Any,
    parameters: dict[str, float],
    config: FinitePacketConfig,
) -> dict[str, Any]:
    direction = atlas.directions[0]
    owners = torch.nonzero(normals @ direction > 1e-8, as_tuple=False).flatten()[:128]
    origins = points[owners]
    directions = direction.expand_as(origins)
    maximum = boundary.exit_times(origins, directions)
    eta = _eta(field, points, config.levelset_eta_relative)
    reference_step = parameters["epsilon"] * min(config.path_step_ratios)
    reference_offsets = _micro_offsets(max(config.micro_counts), points.device, points.dtype)
    reference_t, _, _ = _transmission(
        field, origins, directions, maximum,
        radius=parameters["radius"], epsilon=parameters["epsilon"],
        path_step=reference_step, offsets=reference_offsets, eta=eta,
        kappa=parameters["kappa"], surface_barrier=True,
        launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
    )
    shift = 1e-4
    shifted_plus = ShiftedField(field, shift)
    shifted_minus = ShiftedField(field, -shift)

    def derivative(step: float, offsets: Tensor) -> Tensor:
        eta_plus = _eta(shifted_plus, points, config.levelset_eta_relative)
        eta_minus = _eta(shifted_minus, points, config.levelset_eta_relative)
        plus, _, _ = _transmission(
            shifted_plus, origins, directions, maximum,
            radius=parameters["radius"], epsilon=parameters["epsilon"],
            path_step=step, offsets=offsets, eta=eta_plus,
            kappa=parameters["kappa"], surface_barrier=True,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        minus, _, _ = _transmission(
            shifted_minus, origins, directions, maximum,
            radius=parameters["radius"], epsilon=parameters["epsilon"],
            path_step=step, offsets=offsets, eta=eta_minus,
            kappa=parameters["kappa"], surface_barrier=True,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        return (plus - minus) / (2 * shift)

    reference_derivative = derivative(reference_step, reference_offsets)
    path_rows = []
    for ratio in config.path_step_ratios:
        step = parameters["epsilon"] * ratio
        started = time.perf_counter()
        transmission, _, evaluations = _transmission(
            field, origins, directions, maximum,
            radius=parameters["radius"], epsilon=parameters["epsilon"],
            path_step=step, offsets=reference_offsets, eta=eta,
            kappa=parameters["kappa"], surface_barrier=True,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        current_derivative = derivative(step, reference_offsets)
        path_rows.append({
            "path_step_over_epsilon": ratio,
            "path_step_world": step,
            "transmission_rmse_vs_finest": float(torch.sqrt((transmission - reference_t).square().mean())),
            "transmission_max_error_vs_finest": float((transmission - reference_t).abs().max()),
            "geometry_derivative_relative_error_vs_finest": float(
                torch.linalg.vector_norm(current_derivative - reference_derivative)
                / torch.linalg.vector_norm(reference_derivative).clamp_min(1e-30)
            ),
            "interaction_evaluations": evaluations,
            "runtime_seconds": time.perf_counter() - started,
        })
    micro_rows = []
    for count in config.micro_counts:
        offsets = _micro_offsets(count, points.device, points.dtype)
        transmission, _, evaluations = _transmission(
            field, origins, directions, maximum,
            radius=parameters["radius"], epsilon=parameters["epsilon"],
            path_step=reference_step, offsets=offsets, eta=eta,
            kappa=parameters["kappa"], surface_barrier=True,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        micro_rows.append({
            "micro_samples": count,
            "transmission_rmse_vs_128": float(torch.sqrt((transmission - reference_t).square().mean())),
            "transmission_max_error_vs_128": float((transmission - reference_t).abs().max()),
            "interaction_evaluations": evaluations,
        })
    return {
        "path_rows": path_rows,
        "micro_rows": micro_rows,
        "rays": int(origins.shape[0]),
        "geometry_derivative_shift": shift,
    }


def _opacity_and_angle_test(config: FinitePacketConfig) -> dict[str, Any]:
    epsilon = 0.02
    coordinate = np.linspace(-0.2, 0.2, 20_001)
    rows = []
    for target in config.opacity_targets:
        kappa = -math.log(target)
        for cosine in (1.0, 0.5, 0.25):
            # Parameter t follows the ray; signed normal distance is cosine*t.
            distance = cosine * coordinate
            volumetric = _integrate_1d_shell(distance, coordinate, epsilon, kappa)
            barrier = _integrate_1d_shell(
                distance, coordinate, epsilon, kappa, cosine
            )
            rows.append({
                "target_crossing_transmission": target,
                "kappa": kappa,
                "absolute_normal_direction_dot": cosine,
                "volumetric_transmission": volumetric,
                "surface_barrier_transmission": barrier,
                "surface_barrier_absolute_error": abs(barrier - target),
            })
    selected = [
        row for row in rows
        if row["target_crossing_transmission"] == config.target_crossing_transmission
    ]
    return {
        "rows": rows,
        "primary_semantics": "SURFACE_BARRIER_ATTENUATION",
        "reason": "the zero set represents a surface; one complete crossing is intended to have angle-independent opacity",
        "selected_kappa": -math.log(config.target_crossing_transmission),
        "selected_maximum_angle_dependent_error": max(
            row["surface_barrier_absolute_error"] for row in selected
        ),
        "volumetric_control_retained": True,
    }


def _scale_invariance_real(
    field: Field,
    points: Tensor,
    normals: Tensor,
    atlas: Any,
    boundary: Any,
    parameters: dict[str, float],
    config: FinitePacketConfig,
) -> dict[str, Any]:
    count = min(config.screen_samples, points.shape[0])
    local_points = points[:count]
    local_normals = normals[:count]
    local_atlas = nested_fibonacci_atlas(points.device, (1,))
    rows = []
    reference: SoftRender | None = None
    for scale in (1.0, 0.5, 2.0, 10.0):
        scaled = ScaledField(field, scale)
        render = _soft_render(
            scaled, local_points, local_normals, local_atlas, boundary,
            ((64, 64),), radius=parameters["radius"],
            epsilon=parameters["epsilon"], path_step=parameters["path_step"],
            micro_samples=config.micro_samples,
            eta_relative=config.levelset_eta_relative,
            kappa=parameters["kappa"], surface_barrier=True,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        if reference is None:
            reference = render
        image_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(render.images[(64, 64)], reference.images[(64, 64)])
        )
        transmission_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(render.transmissions, reference.transmissions)
        )
        rows.append({
            "field_scale": scale,
            "eta": _eta(scaled, local_points, config.levelset_eta_relative),
            "image_max_absolute_error": image_error,
            "transmission_max_absolute_error": transmission_error,
        })
    return {
        "normalization": "d_F=F/sqrt(||grad F||^2+eta^2)",
        "eta_rule": "eta=1e-6 median(||grad F||) for each positively scaled representation",
        "rows": rows,
        "maximum_image_absolute_error": max(row["image_max_absolute_error"] for row in rows),
        "maximum_transmission_absolute_error": max(row["transmission_max_absolute_error"] for row in rows),
    }


def _radius_epsilon_sweep(
    field: Field,
    points: Tensor,
    normals: Tensor,
    context: Any,
    atlas: Any,
    boundary: Any,
    h: float,
    config: FinitePacketConfig,
) -> list[dict[str, Any]]:
    count = min(config.screen_samples, points.shape[0])
    local_points = points[:count]
    local_normals = normals[:count]
    local_atlas = nested_fibonacci_atlas(points.device, (1,))
    hard, _ = _hard_images(
        field, local_points, local_normals, context, local_atlas, boundary,
        (256, 256), recompute_visibility=True,
    )
    rows = []
    configurations = [(0.0, 1.0)] + [
        (radius, epsilon)
        for radius in config.radius_ratios if radius > 0.0
        for epsilon in config.epsilon_over_r
    ]
    for radius_ratio, epsilon_ratio in configurations:
        radius = radius_ratio * h
        epsilon = h if radius_ratio == 0.0 else epsilon_ratio * radius
        render = _soft_render(
            field, local_points, local_normals, local_atlas, boundary,
            ((256, 256),), radius=radius, epsilon=epsilon,
            path_step=0.5 * epsilon, micro_samples=(1 if radius == 0.0 else config.micro_samples),
            eta_relative=config.levelset_eta_relative,
            kappa=-math.log(config.target_crossing_transmission), surface_barrier=True,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        metrics = _image_metrics(render.images[(256, 256)], hard)
        transmission = np.concatenate(render.transmissions)
        rows.append({
            "radius_over_h": radius_ratio,
            "radius_world": radius,
            "epsilon_over_radius": None if radius == 0.0 else epsilon_ratio,
            "epsilon_world": epsilon,
            "path_step_world": 0.5 * epsilon,
            "micro_samples": 1 if radius == 0.0 else config.micro_samples,
            "mean_transmission": float(transmission.mean()),
            "transmission_std": float(transmission.std()),
            "runtime_seconds": render.runtime_seconds,
            **metrics,
        })
    return rows


def _multiseed_test(
    base: Any,
    cells: Tensor,
    atlas: Any,
    boundary: Any,
    parameters: dict[str, float],
    config: FinitePacketConfig,
) -> dict[str, Any]:
    rows = []
    reference_image: list[np.ndarray] | None = None
    reference_response: np.ndarray | None = None
    for seed in config.seeds:
        surface = _prepare_surface(
            base, None, config.multiseed_samples, config.emitter_chunk_size,
            seed, cells,
        )
        local_atlas = nested_fibonacci_atlas(base.grid.device, (2,))
        local_boundary = enclosing_observation_sphere(surface.reference_points)
        local_h = _surface_spacing(surface.base_points)
        local_parameters = _render_parameters(config, local_h)
        render = _soft_render(
            base, surface.base_points, surface.base_normals, local_atlas,
            local_boundary, ((128, 128),),
            radius=local_parameters["radius"], epsilon=local_parameters["epsilon"],
            path_step=local_parameters["path_step"], micro_samples=config.micro_samples,
            eta_relative=config.levelset_eta_relative,
            kappa=local_parameters["kappa"], surface_barrier=True,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        image = render.images[(128, 128)]
        shifted = _soft_render(
            ShiftedField(base, 3e-4), surface.base_points, surface.base_normals,
            local_atlas, local_boundary, ((128, 128),),
            radius=local_parameters["radius"], epsilon=local_parameters["epsilon"],
            path_step=local_parameters["path_step"], micro_samples=config.micro_samples,
            eta_relative=config.levelset_eta_relative,
            kappa=local_parameters["kappa"], surface_barrier=True,
            sensor_gain=config.sensor_gain, ambient=config.ambient,
            detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
        )
        response = np.concatenate([
            (right - left).reshape(-1)
            for right, left in zip(shifted.images[(128, 128)], image)
        ])
        if reference_image is None:
            reference_image = image
            reference_response = response
        difference = np.concatenate([
            (left - right).reshape(-1)
            for left, right in zip(image, reference_image)
        ])
        cosine = float(np.dot(response, reference_response) / max(
            np.linalg.norm(response) * np.linalg.norm(reference_response), 1e-30
        ))
        rows.append({
            "seed": seed,
            "surface_spacing_h": local_h,
            "self_mse_vs_seed_101": float(np.mean(difference**2)),
            "geometry_response_norm": float(np.linalg.norm(response)),
            "geometry_response_cosine_vs_seed_101": cosine,
            "heldout_image_variance_proxy": float(np.var(difference)),
        })
        del surface, render, shifted
        _release()
    heldout = [row for row in rows if row["seed"] != config.seed]
    return {
        "seeds": list(config.seeds),
        "rows": rows,
        "mean_self_mse": statistics.mean(row["self_mse_vs_seed_101"] for row in heldout),
        "median_geometry_response_cosine": statistics.median(
            row["geometry_response_cosine_vs_seed_101"] for row in heldout
        ),
        "geometry_response_norm_cv": float(np.std([
            row["geometry_response_norm"] for row in heldout
        ]) / max(np.mean([row["geometry_response_norm"] for row in heldout]), 1e-30)),
    }


def _tensor_vector(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([image.reshape(-1).astype(np.float64) for image in images])


def _vector_metrics(left: np.ndarray | Tensor, right: np.ndarray | Tensor) -> dict[str, float]:
    a = left.detach().cpu().numpy() if isinstance(left, Tensor) else np.asarray(left)
    b = right.detach().cpu().numpy() if isinstance(right, Tensor) else np.asarray(right)
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return {
        "left_norm": float(np.linalg.norm(a)),
        "right_norm": float(np.linalg.norm(b)),
        "relative_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
        "cosine_similarity": float(np.dot(a, b) / denominator),
        "maximum_absolute_error": float(np.max(np.abs(a - b))) if a.size else 0.0,
        "changed_scalar_fraction": float(np.mean(np.abs(a) > 1e-12)) if a.size else 0.0,
    }


def _diagnostic_context(
    prepared: object,
    config: FinitePacketConfig,
    seed: int,
) -> Any:
    corrected = CorrectedBirthConfig(
        dictionary_count=64,
        initial_count=32,
        surface_samples=config.diagnostic_samples,
        views=config.diagnostic_views,
        resolution=config.diagnostic_resolution,
        surface_scramble_seed=seed,
    )
    return _build_context(prepared, corrected)


def _single_basis_grid_field(context: Any, parameter: int, coefficient: float) -> GridZeroSetField:
    """Rasterize one local perturbation for the genuinely resampled-source FD."""
    base = context.base
    grid = base.grid.clone()
    gradient_grid = base.gradient_grid.clone()
    center = context.master_layout.centers[parameter]
    radius = context.master_layout.radii[parameter]
    dimensions = torch.tensor(grid.shape, dtype=grid.dtype, device=grid.device)
    voxel = (base.upper - base.lower) / (dimensions - 1.0)
    low = torch.floor((center - radius - base.lower) / voxel).to(torch.long).clamp_min(0)
    high = torch.ceil((center + radius - base.lower) / voxel).to(torch.long)
    high = torch.minimum(high, torch.tensor(grid.shape, device=grid.device) - 1)
    ix = torch.arange(low[0], high[0] + 1, device=grid.device)
    iy = torch.arange(low[1], high[1] + 1, device=grid.device)
    iz = torch.arange(low[2], high[2] + 1, device=grid.device)
    mesh = torch.meshgrid(ix, iy, iz, indexing="ij")
    indices = torch.stack(mesh, -1)
    points = base.lower + indices.to(grid.dtype) * voxel
    offsets = points - center
    radii = radius.expand(offsets.shape[:-1])
    values = wendland_values(offsets.reshape(-1, 3), radii.reshape(-1)).reshape(offsets.shape[:-1])
    gradients = wendland_gradients(offsets.reshape(-1, 3), radii.reshape(-1)).reshape(*offsets.shape)
    selection = tuple(item.reshape(-1) for item in mesh)
    grid[selection] += coefficient * values.reshape(-1)
    gradient_grid[selection] += coefficient * gradients.reshape(-1, 3)
    return GridZeroSetField(
        grid, gradient_grid, base.lower, base.upper,
        base.surface_vertices, base.surface_faces, base.face_cdf,
    )


def _finite_packet_geometry_diagnostic(
    prepared: object,
    config: FinitePacketConfig,
    historical_v084: dict[str, Any],
) -> dict[str, Any]:
    """Separate detector, moving-source, and owner/topology response terms."""
    context = _diagnostic_context(prepared, config, config.seed)
    heldout = _diagnostic_context(prepared, config, config.seeds[1])
    _copy_layout(heldout, context)
    device = context.reference_points.device
    active = torch.arange(32, device=device)
    zero = torch.zeros(32, dtype=torch.float64, device=device)
    layout, support = _active_components(context, active)
    heldout_layout, heldout_support = _active_components(heldout, active)
    with torch.no_grad():
        state = _evaluate(context, active, zero, (layout, support))
        heldout_state = _evaluate(heldout, active, zero, (heldout_layout, heldout_support))
    atlas = nested_fibonacci_atlas(device, (config.diagnostic_views,))
    boundary = enclosing_observation_sphere(context.reference_points)
    heldout_boundary = enclosing_observation_sphere(heldout.reference_points)
    owners = [
        torch.nonzero(state.normals @ atlas.directions[index] > 1e-8, as_tuple=False).flatten()
        for index in range(atlas.count)
    ]
    heldout_owners = [
        torch.nonzero(heldout_state.normals @ atlas.directions[index] > 1e-8, as_tuple=False).flatten()
        for index in range(atlas.count)
    ]
    h = _surface_spacing(state.points)
    parameters = _render_parameters(config, h)
    resolution = (config.diagnostic_resolution, config.diagnostic_resolution)

    def soft_vector(
        local_context: Any,
        local_state: Any,
        coefficients: Tensor,
        local_atlas: Any,
        local_boundary: Any,
        *,
        radius: float,
        epsilon: float,
        frozen: list[Tensor] | None,
        fixed_sources: bool = False,
        base_state: Any | None = None,
    ) -> Tensor:
        field = LocalField(local_context, coefficients)
        source = base_state if fixed_sources and base_state is not None else local_state
        return _soft_tensor_image(
            field, source.points, source.normals, local_atlas, local_boundary, resolution,
            radius=radius, epsilon=epsilon, path_step=0.5 * epsilon,
            micro_samples=(1 if radius == 0.0 else config.micro_samples),
            eta_relative=config.levelset_eta_relative, kappa=parameters["kappa"],
            surface_barrier=True, sensor_gain=config.sensor_gain, ambient=config.ambient,
            detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
            launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
            frozen_owners=frozen,
        )

    categories = {
        str(name): int(index)
        for name, index in historical_v084["analytic_jacobian"]["categories"].items()
    }
    rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    finite_radius = parameters["radius"]
    finite_epsilon = parameters["epsilon"]

    def evaluated(local_context: Any, coeff: Tensor, components: Any) -> Any:
        with torch.no_grad():
            return _evaluate(local_context, active, coeff, components)

    for category, parameter in categories.items():
        _progress("geometry_category", category=category, parameter=parameter)
        for delta in config.perturbation_deltas:
            plus_c = zero.clone(); plus_c[parameter] += delta
            minus_c = zero.clone(); minus_c[parameter] -= delta
            plus = evaluated(context, plus_c, (layout, support))
            minus = evaluated(context, minus_c, (layout, support))
            held_plus = evaluated(heldout, plus_c, (heldout_layout, heldout_support))
            held_minus = evaluated(heldout, minus_c, (heldout_layout, heldout_support))
            with torch.no_grad():
                finite = (
                    soft_vector(context, plus, plus_c, atlas, boundary,
                                radius=finite_radius, epsilon=finite_epsilon, frozen=owners)
                    - soft_vector(context, minus, minus_c, atlas, boundary,
                                  radius=finite_radius, epsilon=finite_epsilon, frozen=owners)
                ) / (2.0 * delta)
                point = (
                    soft_vector(context, plus, plus_c, atlas, boundary,
                                radius=0.0, epsilon=h, frozen=owners)
                    - soft_vector(context, minus, minus_c, atlas, boundary,
                                  radius=0.0, epsilon=h, frozen=owners)
                ) / (2.0 * delta)
                held_finite = (
                    soft_vector(heldout, held_plus, plus_c, atlas, heldout_boundary,
                                radius=finite_radius, epsilon=finite_epsilon, frozen=heldout_owners)
                    - soft_vector(heldout, held_minus, minus_c, atlas, heldout_boundary,
                                  radius=finite_radius, epsilon=finite_epsilon, frozen=heldout_owners)
                ) / (2.0 * delta)
                hard_plus, _ = _hard_images(
                    LocalField(context, plus_c), plus.points, plus.normals, context,
                    atlas, boundary, resolution, recompute_visibility=True,
                )
                hard_minus, _ = _hard_images(
                    LocalField(context, minus_c), minus.points, minus.normals, context,
                    atlas, boundary, resolution, recompute_visibility=True,
                )
                hard = (_tensor_vector(hard_plus) - _tensor_vector(hard_minus)) / (2.0 * delta)
            response_rows.append({
                "category": category,
                "parameter": parameter,
                "delta": delta,
                "hard_response_norm": float(np.linalg.norm(hard)),
                "point_soft_response_norm": float(torch.linalg.vector_norm(point)),
                "finite_packet_response_norm": float(torch.linalg.vector_norm(finite)),
                "finite_vs_point": _vector_metrics(finite, point),
                "finite_same_vs_heldout": _vector_metrics(finite, held_finite),
            })

        for epsilon_fd in config.fd_epsilons:
            _progress("geometry_fd", category=category, epsilon=epsilon_fd)
            plus_c = zero.clone(); plus_c[parameter] += epsilon_fd
            minus_c = zero.clone(); minus_c[parameter] -= epsilon_fd
            plus = evaluated(context, plus_c, (layout, support))
            minus = evaluated(context, minus_c, (layout, support))
            with torch.no_grad():
                # B: only the implicit field in the path integral changes.
                b = (
                    soft_vector(context, state, plus_c, atlas, boundary,
                                radius=finite_radius, epsilon=finite_epsilon,
                                frozen=owners, fixed_sources=True, base_state=state)
                    - soft_vector(context, state, minus_c, atlas, boundary,
                                  radius=finite_radius, epsilon=finite_epsilon,
                                  frozen=owners, fixed_sources=True, base_state=state)
                ) / (2.0 * epsilon_fd)
                # C: source positions/normals move, but identities remain frozen.
                c = (
                    soft_vector(context, plus, plus_c, atlas, boundary,
                                radius=finite_radius, epsilon=finite_epsilon, frozen=owners)
                    - soft_vector(context, minus, minus_c, atlas, boundary,
                                  radius=finite_radius, epsilon=finite_epsilon, frozen=owners)
                ) / (2.0 * epsilon_fd)
                # D: recompute outward packet ownership; still no first-root selection.
                owner_recomputed = (
                    soft_vector(context, plus, plus_c, atlas, boundary,
                                radius=finite_radius, epsilon=finite_epsilon, frozen=None)
                    - soft_vector(context, minus, minus_c, atlas, boundary,
                                  radius=finite_radius, epsilon=finite_epsilon, frozen=None)
                ) / (2.0 * epsilon_fd)
                # D full rerender: rebuild the perturbed trilinear grid, its
                # sign-changing-cell set, and its projected source roots using
                # the same scrambled Sobol seed before running soft transport.
                plus_grid = _single_basis_grid_field(context, parameter, epsilon_fd)
                minus_grid = _single_basis_grid_field(context, parameter, -epsilon_fd)
                plus_surface = sample_meshfree_zero_set(
                    plus_grid, context.config.surface_samples,
                    sobol_scramble_seed=config.seed,
                )
                minus_surface = sample_meshfree_zero_set(
                    minus_grid, context.config.surface_samples,
                    sobol_scramble_seed=config.seed,
                )
                plus_full = _soft_tensor_image(
                    plus_grid, plus_surface.points, plus_surface.normals, atlas,
                    enclosing_observation_sphere(plus_surface.points), resolution,
                    radius=finite_radius, epsilon=finite_epsilon, path_step=0.5 * finite_epsilon,
                    micro_samples=config.micro_samples, eta_relative=config.levelset_eta_relative,
                    kappa=parameters["kappa"], surface_barrier=True,
                    sensor_gain=config.sensor_gain, ambient=config.ambient,
                    detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
                    launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
                    frozen_owners=None,
                )
                minus_full = _soft_tensor_image(
                    minus_grid, minus_surface.points, minus_surface.normals, atlas,
                    enclosing_observation_sphere(minus_surface.points), resolution,
                    radius=finite_radius, epsilon=finite_epsilon, path_step=0.5 * finite_epsilon,
                    micro_samples=config.micro_samples, eta_relative=config.levelset_eta_relative,
                    kappa=parameters["kappa"], surface_barrier=True,
                    sensor_gain=config.sensor_gain, ambient=config.ambient,
                    detector_extent=config.detector_extent, chunk_size=config.emitter_chunk_size,
                    launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
                    frozen_owners=None,
                )
                d = (plus_full - minus_full) / (2.0 * epsilon_fd)

            unit = torch.zeros_like(zero); unit[parameter] = 1.0
            def differentiable(coeff: Tensor) -> Tensor:
                local_state = _evaluate(context, active, coeff, (layout, support))
                return soft_vector(
                    context, local_state, coeff, atlas, boundary,
                    radius=finite_radius, epsilon=finite_epsilon, frozen=owners,
                )
            _, a = torch.autograd.functional.jvp(
                differentiable, (zero,), (unit,), create_graph=False, strict=False
            )
            visibility = owner_recomputed - c
            source = d - owner_recomputed
            topology = d - c
            moving_source = c - b
            other = a - c
            rows.append({
                "category": category,
                "parameter": parameter,
                "epsilon": epsilon_fd,
                "mode_A_autodiff_norm": float(torch.linalg.vector_norm(a)),
                "mode_B_frozen_source_fd_norm": float(torch.linalg.vector_norm(b)),
                "mode_C_moving_source_frozen_owner_fd_norm": float(torch.linalg.vector_norm(c)),
                "mode_D_full_rerender_fd_norm": float(torch.linalg.vector_norm(d)),
                "analytic_vs_mode_C": _vector_metrics(a, c),
                "mode_B_vs_mode_C": _vector_metrics(b, c),
                "mode_C_vs_mode_D": _vector_metrics(c, d),
                "continuous_source_motion_component_fraction_of_C": float(
                    torch.linalg.vector_norm(moving_source) / torch.linalg.vector_norm(c).clamp_min(1e-30)
                ),
                "visibility_owner_component_fraction_of_D": float(
                    torch.linalg.vector_norm(visibility) / torch.linalg.vector_norm(d).clamp_min(1e-30)
                ),
                "source_resampling_component_fraction_of_D": float(
                    torch.linalg.vector_norm(source) / torch.linalg.vector_norm(d).clamp_min(1e-30)
                ),
                "full_topology_component_fraction_of_D": float(
                    torch.linalg.vector_norm(topology) / torch.linalg.vector_norm(d).clamp_min(1e-30)
                ),
                "autodiff_numerical_residual_fraction_of_C": float(
                    torch.linalg.vector_norm(other) / torch.linalg.vector_norm(c).clamp_min(1e-30)
                ),
            })
            del plus, minus, a, b, c, d, owner_recomputed
            _release()

    best = [
        min(
            [row for row in rows if row["category"] == category],
            key=lambda row: row["analytic_vs_mode_C"]["relative_error"],
        )
        for category in categories
    ]
    old_best = historical_v084["analytic_jacobian"]["best_frozen_rows"]
    before = float(statistics.median(
        row["topology_component_fraction_of_full_norm"] for row in old_best
    ))
    after = float(statistics.median(
        row["full_topology_component_fraction_of_D"] for row in best
    ))
    result = {
        "categories": categories,
        "configuration": asdict(context.config),
        "surface_spacing_h": h,
        "transport_parameters": parameters,
        "response_rows": response_rows,
        "fd_rows": rows,
        "best_fd_rows": best,
        "decomposition": {
            "mode_A": "autodiff/JVP with moving sources and frozen packet-owner IDs",
            "mode_B": "central FD, field/path attenuation only, frozen source points and normals",
            "mode_C": "central FD, recomputed source positions/normals, frozen owner IDs",
            "mode_D": "central FD with rerasterized field, recomputed sign-changing cells, source roots, and outward packet owners",
            "detector_component": "included continuously in A/C/D via cubic detector integral",
            "continuous_source_motion_component": "C-B; normal-line source positions move with fixed identities",
            "visibility_owner_component": "owner-recomputed intermediate minus C; no first path root is ever selected",
            "source_sample_root_component": "D minus owner-recomputed intermediate; sign-changing-cell selection and source roots are rebuilt",
        },
        "topology_fraction_before_v084_median": before,
        "topology_fraction_after_v085_median": after,
        "topology_fraction_relative_reduction": (before - after) / max(before, 1e-30),
        "visibility_owner_component_fraction_after_median": float(statistics.median(
            row["visibility_owner_component_fraction_of_D"] for row in best
        )),
        "source_resampling_component_fraction_after_median": float(statistics.median(
            row["source_resampling_component_fraction_of_D"] for row in best
        )),
    }
    del context, heldout, state, heldout_state
    _release()
    return result


def _scalar_csv_rows(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_scalar_csv_rows(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(_scalar_csv_rows(item, f"{prefix}[{index}]"))
    elif isinstance(value, (str, bool, int, float)) or value is None:
        rows.append({"metric": prefix, "value": value})
    return rows


def _save_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_figures(
    directory: Path,
    report: dict[str, Any],
    captures: dict[str, Any],
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    path = directory / "v085_packet_zero_set_concept.png"
    figure, axis = plt.subplots(figsize=(10, 2.6))
    labels = ["finite-support\nenergy packet", "continuous overlap\nwith zero-set shell",
              "remaining energy\nEout=T(F)Ein", "continuous detector\narrival"]
    xs = np.arange(4)
    axis.scatter(xs, np.zeros(4), s=2200, c=["#4477aa", "#66ccee", "#228833", "#ccbb44"])
    for x, label in zip(xs, labels): axis.text(x, 0, label, ha="center", va="center", color="white", fontsize=9)
    for x in xs[:-1]: axis.annotate("", (x + 0.72, 0), (x + 0.28, 0), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.set_xlim(-0.55, 3.55); axis.set_ylim(-0.55, 0.55); axis.axis("off")
    _save_figure(path, figure); paths.append(path)

    cap = report["sphere_cap"]
    path = directory / "v085_sphere_cap_overlap.png"
    figure, axis = plt.subplots()
    axis.plot(cap["plot"]["s"], cap["plot"]["phi"], label="blocked cap fraction")
    axis.plot(cap["plot"]["s"], cap["plot"]["transmission"], label="transmitted fraction")
    axis.set(xlabel="signed plane distance / packet radius", ylabel="fraction")
    axis.grid(alpha=.25); axis.legend(); _save_figure(path, figure); paths.append(path)

    toy_specs = [
        ("moving_plane", "v085_plane_transition.png", "moving plane"),
        ("grazing_sphere", "v085_grazing_sphere_transition.png", "grazing sphere"),
        ("root_birth_death", "v085_root_birth_death_transition.png", "root birth/death"),
    ]
    for key, filename, title in toy_specs:
        data = report["toy_transitions"][key]["plot"]
        parameter = data["lambda"]
        path = directory / filename
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
        axes[0].plot(parameter, data["hard"], label="hard")
        axes[0].plot(parameter, data["soft"], label="finite packet")
        axes[1].plot(parameter, data["derivative"], color="#cc6677")
        axes[0].set_title(title); axes[0].set_ylabel("transmission"); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].set(xlabel="geometry parameter", ylabel="dT/dlambda"); axes[1].grid(alpha=.25)
        _save_figure(path, figure); paths.append(path)

    path = directory / "v085_two_surface_path.png"
    two = report["toy_transitions"]["two_surface_path"]["plot"]
    figure, axis = plt.subplots()
    axis.semilogy(two["separation"], two["transmission"], label="two sheets")
    axis.semilogy(two["separation"], two["single_surface"], "--", label="one-sheet target")
    axis.set(xlabel="sheet separation", ylabel="transmission", title="root-free two-surface path")
    axis.grid(alpha=.25); axis.legend(); _save_figure(path, figure); paths.append(path)

    path = directory / "v085_levelset_scale_invariance.png"
    rows = report["levelset_scale_invariance"]["real_scene"]["rows"]
    figure, axis = plt.subplots()
    axis.semilogy([row["field_scale"] for row in rows],
                  [max(row["image_max_absolute_error"], 1e-18) for row in rows], "o-", label="RGB max error")
    axis.semilogy([row["field_scale"] for row in rows],
                  [max(row["transmission_max_absolute_error"], 1e-18) for row in rows], "s-", label="T max error")
    axis.set(xlabel="positive F scale", ylabel="absolute error", title="normalized level-set invariance")
    axis.grid(alpha=.25); axis.legend(); _save_figure(path, figure); paths.append(path)

    sweep = report["radius_epsilon_sweep"]
    path = directory / "v085_radius_epsilon_tradeoff.png"
    figure, axis = plt.subplots()
    scatter = axis.scatter([row["edge_sharpness_ratio"] for row in sweep],
                           [row["foreground_mse"] for row in sweep],
                           c=[row["radius_over_h"] for row in sweep], s=55)
    axis.set(xlabel="edge sharpness / hard", ylabel="foreground MSE", title="radius/width Pareto screen")
    axis.grid(alpha=.25); figure.colorbar(scatter, ax=axis, label="r/h")
    _save_figure(path, figure); paths.append(path)

    path = directory / "v085_hard_vs_soft_transport_rgb.png"
    images = captures["comparison_images"]
    figure, axes = plt.subplots(3, len(images["hard"]), figsize=(3 * len(images["hard"]), 8))
    for row_id, name in enumerate(("hard", "point_soft", "finite_packet")):
        for view, image in enumerate(images[name]):
            axes[row_id, view].imshow(np.clip(image, 0, 1)); axes[row_id, view].axis("off")
            axes[row_id, view].set_title(f"{name}, view {view}")
    _save_figure(path, figure); paths.append(path)

    response = report["geometry_diagnostic"]["response_rows"]
    path = directory / "v085_geometry_perturbation_response.png"
    categories = list(report["geometry_diagnostic"]["categories"])
    selected = [next(row for row in response if row["category"] == category) for category in categories]
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(categories)); width = .25
    for offset, key, label in ((-width, "hard_response_norm", "hard"), (0, "point_soft_response_norm", "point soft"),
                               (width, "finite_packet_response_norm", "finite packet")):
        axis.bar(x + offset, [row[key] for row in selected], width, label=label)
    axis.set_xticks(x, categories, rotation=20); axis.set_yscale("log"); axis.set_ylabel("response norm"); axis.legend(); axis.grid(axis="y", alpha=.25)
    _save_figure(path, figure); paths.append(path)

    fd = report["geometry_diagnostic"]["best_fd_rows"]
    path = directory / "v085_frozen_vs_full_fd.png"
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fd)); width = .25
    for offset, key, label in ((-width, "mode_B_frozen_source_fd_norm", "B frozen source"),
                               (0, "mode_C_moving_source_frozen_owner_fd_norm", "C moving source"),
                               (width, "mode_D_full_rerender_fd_norm", "D full rerender")):
        axis.bar(x + offset, [row[key] for row in fd], width, label=label)
    axis.set_xticks(x, [row["category"] for row in fd], rotation=20); axis.set_yscale("log")
    axis.set_ylabel("FD norm"); axis.legend(); axis.grid(axis="y", alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v085_topology_fraction_before_after.png"
    geometry = report["geometry_diagnostic"]
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["v0.8.4 binary", "v0.8.5 finite packet"],
             [geometry["topology_fraction_before_v084_median"], geometry["topology_fraction_after_v085_median"]],
             color=["#cc6677", "#228833"])
    axis.set_ylabel("topology / full-FD norm"); axis.grid(axis="y", alpha=.25)
    _save_figure(path, figure); paths.append(path)

    multiseed = report["multi_seed_mc"]["rows"]
    path = directory / "v085_mc_multiseed.png"
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([row["seed"] for row in multiseed], [row["self_mse_vs_seed_101"] for row in multiseed], "o-")
    axes[1].plot([row["seed"] for row in multiseed], [row["geometry_response_cosine_vs_seed_101"] for row in multiseed], "o-")
    axes[0].set_ylabel("self-MSE vs seed 101"); axes[1].set_ylabel("response cosine vs seed 101")
    for axis in axes: axis.set_xlabel("Sobol scramble seed"); axis.grid(alpha=.25)
    _save_figure(path, figure); paths.append(path)

    path = directory / "v085_runtime_scaling.png"
    path_rows = report["path_micro_convergence"]["path_rows"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].loglog([row["path_step_over_epsilon"] for row in path_rows],
                   [row["runtime_seconds"] for row in path_rows], "o-")
    axes[1].loglog([row["interaction_evaluations"] for row in path_rows],
                   [row["runtime_seconds"] for row in path_rows], "o-")
    axes[0].set(xlabel="path step / epsilon", ylabel="seconds")
    axes[1].set(xlabel="interaction evaluations", ylabel="seconds")
    for axis in axes: axis.grid(alpha=.25)
    _save_figure(path, figure); paths.append(path)
    return [str(path) for path in paths]


def _verdicts(report: dict[str, Any], config: FinitePacketConfig) -> tuple[dict[str, bool], dict[str, Any]]:
    toys = report["toy_transitions"]
    scale = report["levelset_scale_invariance"]
    path_rows = report["path_micro_convergence"]["path_rows"]
    finest_comparison = next(row for row in path_rows if row["path_step_over_epsilon"] == 0.25)
    energy_error = max(
        [row["maximum_absolute_error"] for row in report["energy_linearity"]["scale_tests"]]
        + [report["energy_linearity"]["superposition"]["maximum_absolute_error"]]
    )
    geometry = report["geometry_diagnostic"]
    best = geometry["best_fd_rows"]
    response = [row for row in geometry["response_rows"] if row["delta"] == config.perturbation_deltas[0]]
    selected_metrics = report["image_fidelity"]["selected_256"]
    evidence = {
        "scale_max_error": max(scale["analytic"]["maximum_absolute_difference"],
                               scale["real_scene"]["maximum_image_absolute_error"],
                               scale["real_scene"]["maximum_transmission_absolute_error"]),
        "sphere_cap_max_error": report["sphere_cap"]["independent_gauss_legendre_max_absolute_error"],
        "toy_max_soft_value_jump": max(toys[name]["soft_maximum_adjacent_value_jump"] for name in
                                        ("moving_plane", "grazing_sphere", "root_birth_death")),
        "toy_max_fd_spread": max(toys[name]["finite_difference_relative_spread_across_1x_2x_4x_steps"] for name in
                                  ("moving_plane", "grazing_sphere", "root_birth_death")),
        "energy_max_error": energy_error,
        "path_quarter_step_transmission_rmse": finest_comparison["transmission_rmse_vs_finest"],
        "path_quarter_step_derivative_relative_error": finest_comparison["geometry_derivative_relative_error_vs_finest"],
        "opacity_max_error": report["surface_opacity"]["selected_maximum_angle_dependent_error"],
        "two_surface_relative_error": toys["two_surface_path"]["relative_error"],
        "multiseed_median_response_cosine": report["multi_seed_mc"]["median_geometry_response_cosine"],
        "multiseed_response_norm_cv": report["multi_seed_mc"]["geometry_response_norm_cv"],
        "median_finite_vs_point_response_norm_ratio": statistics.median(
            row["finite_packet_response_norm"] / max(row["point_soft_response_norm"], 1e-30) for row in response
        ),
        "median_finite_heldout_response_cosine": statistics.median(
            row["finite_same_vs_heldout"]["cosine_similarity"] for row in response
        ),
        "selected_edge_sharpness_ratio": selected_metrics["edge_sharpness_ratio"],
        "selected_thin_feature_response_ratio": selected_metrics["thin_feature_response_ratio"],
        "topology_before": geometry["topology_fraction_before_v084_median"],
        "topology_after": geometry["topology_fraction_after_v085_median"],
        "topology_relative_reduction": geometry["topology_fraction_relative_reduction"],
        "visibility_owner_component_after": geometry["visibility_owner_component_fraction_after_median"],
        "source_resampling_component_after": geometry["source_resampling_component_fraction_after_median"],
        "median_analytic_vs_C_relative_error": statistics.median(
            row["analytic_vs_mode_C"]["relative_error"] for row in best
        ),
    }
    verdicts = {
        "LEVELSET_SCALE_INVARIANT": evidence["scale_max_error"] <= config.scale_invariance_tolerance,
        "SPHERE_CAP_ANALYTIC_VALIDATED": evidence["sphere_cap_max_error"] <= config.sphere_cap_tolerance,
        "FINITE_PACKET_VALUE_CONTINUOUS": evidence["toy_max_soft_value_jump"] < 0.25,
        "FINITE_PACKET_GRADIENT_CONTINUOUS": evidence["toy_max_fd_spread"] < 0.50,
        "ROOT_FREE_TRANSPORT_IMPLEMENTED": True,
        "ENERGY_TRANSPORT_LINEAR": evidence["energy_max_error"] <= config.energy_linearity_tolerance,
        "PATH_QUADRATURE_CONVERGED": (
            evidence["path_quarter_step_transmission_rmse"] <= config.path_transmission_rmse_tolerance
            and evidence["path_quarter_step_derivative_relative_error"] <= config.path_derivative_relative_tolerance
        ),
        "SURFACE_OPACITY_CALIBRATED": evidence["opacity_max_error"] <= config.opacity_tolerance,
        "HARD_VISIBILITY_TOPOLOGY_JUMP_CONFIRMED": all(
            toys[name]["hard_maximum_value_jump"] >= 0.99 for name in
            ("moving_plane", "grazing_sphere", "root_birth_death")
        ),
        "ROOT_BIRTH_DEATH_CONTINUOUSIZED": toys["root_birth_death"]["soft_maximum_adjacent_value_jump"] < 0.25,
        "MULTI_SURFACE_PATH_ROOT_FREE": evidence["two_surface_relative_error"] < 0.01,
        "FINITE_PACKET_MC_VARIANCE_ACCEPTABLE": (
            evidence["multiseed_median_response_cosine"] >= config.multiseed_response_cosine_minimum
            and evidence["multiseed_response_norm_cv"] < 1.0
        ),
        "GEOMETRY_SIGNAL_PRESERVED": (
            evidence["median_finite_vs_point_response_norm_ratio"] >= 0.25
            and evidence["median_finite_heldout_response_cosine"] > 0.0
        ),
        "SILHOUETTE_BANDWIDTH_ACCEPTABLE": (
            evidence["selected_edge_sharpness_ratio"] >= 0.50
            and evidence["selected_thin_feature_response_ratio"] >= 0.40
        ),
        "VISIBILITY_TOPOLOGY_COMPONENT_REDUCED": (
            evidence["visibility_owner_component_after"]
            <= (1.0 - config.topology_reduction_minimum) * evidence["topology_before"]
        ),
        "SOURCE_ROOT_TOPOLOGY_REMAINS": True,
        "ANALYTIC_SOFT_TRANSPORT_MATCHES_FROZEN_FD": evidence["median_analytic_vs_C_relative_error"] <= config.analytic_fd_relative_tolerance,
        "FULL_RERENDER_MISMATCH_REDUCED": evidence["topology_after"] < evidence["topology_before"],
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED": evidence["topology_relative_reduction"] >= config.topology_reduction_minimum,
        "PACKET_KERNEL_GEOMETRY_ARTIFACT_DETECTED": False,
        "SMALL_GEOMETRY_OPTIMIZATION_READY": False,
        "HIGH_RES_BIRTH_READY_TO_RETEST": False,
    }
    prerequisites = [
        "FINITE_PACKET_VALUE_CONTINUOUS", "LEVELSET_SCALE_INVARIANT", "ENERGY_TRANSPORT_LINEAR",
        "PATH_QUADRATURE_CONVERGED", "GEOMETRY_SIGNAL_PRESERVED",
        "ANALYTIC_SOFT_TRANSPORT_MATCHES_FROZEN_FD", "SILHOUETTE_BANDWIDTH_ACCEPTABLE",
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED",
    ]
    verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"] = all(verdicts[name] for name in prerequisites)
    # Birth remains blocked in this version because source/root topology has not
    # been made continuous and no validated small optimization is yet available.
    verdicts["HIGH_RES_BIRTH_READY_TO_RETEST"] = False
    evidence["by_verdict"] = {
        "LEVELSET_SCALE_INVARIANT": {"observed_max_error": evidence["scale_max_error"], "required_max": config.scale_invariance_tolerance},
        "SPHERE_CAP_ANALYTIC_VALIDATED": {"quadrature_max_error": evidence["sphere_cap_max_error"], "float32_vs_float64_max_error": report["sphere_cap"]["float32_float64_max_absolute_error"], "required_quadrature_max": config.sphere_cap_tolerance},
        "FINITE_PACKET_VALUE_CONTINUOUS": {"maximum_grid_adjacent_soft_jump": evidence["toy_max_soft_value_jump"], "hard_jump": 1.0, "required_max": 0.25},
        "FINITE_PACKET_GRADIENT_CONTINUOUS": {"maximum_1x_2x_4x_fd_relative_spread": evidence["toy_max_fd_spread"], "required_max": 0.50},
        "ROOT_FREE_TRANSPORT_IMPLEMENTED": {"first_root_selected": False, "path_integral_used": True},
        "ENERGY_TRANSPORT_LINEAR": {"maximum_absolute_error": evidence["energy_max_error"], "required_max": config.energy_linearity_tolerance},
        "PATH_QUADRATURE_CONVERGED": {"quarter_step_transmission_rmse": evidence["path_quarter_step_transmission_rmse"], "required_rmse_max": config.path_transmission_rmse_tolerance, "quarter_step_derivative_relative_error": evidence["path_quarter_step_derivative_relative_error"], "required_derivative_max": config.path_derivative_relative_tolerance},
        "SURFACE_OPACITY_CALIBRATED": {"maximum_angle_dependent_error": evidence["opacity_max_error"], "required_max": config.opacity_tolerance},
        "HARD_VISIBILITY_TOPOLOGY_JUMP_CONFIRMED": {"plane_hard_jump": toys["moving_plane"]["hard_maximum_value_jump"], "grazing_hard_jump": toys["grazing_sphere"]["hard_maximum_value_jump"], "birth_hard_jump": toys["root_birth_death"]["hard_maximum_value_jump"], "required_min": 0.99},
        "ROOT_BIRTH_DEATH_CONTINUOUSIZED": {"maximum_adjacent_soft_jump": toys["root_birth_death"]["soft_maximum_adjacent_value_jump"], "hard_jump": toys["root_birth_death"]["hard_maximum_value_jump"], "required_max": 0.25},
        "MULTI_SURFACE_PATH_ROOT_FREE": {"relative_product_error": evidence["two_surface_relative_error"], "required_max": 0.01, "first_root_selected": False},
        "FINITE_PACKET_MC_VARIANCE_ACCEPTABLE": {"median_response_cosine": evidence["multiseed_median_response_cosine"], "required_min": config.multiseed_response_cosine_minimum, "response_norm_cv": evidence["multiseed_response_norm_cv"], "required_cv_max": 1.0},
        "GEOMETRY_SIGNAL_PRESERVED": {"median_finite_vs_point_norm_ratio": evidence["median_finite_vs_point_response_norm_ratio"], "required_min": 0.25, "median_heldout_cosine": evidence["median_finite_heldout_response_cosine"], "required_cosine_min": 0.0},
        "SILHOUETTE_BANDWIDTH_ACCEPTABLE": {"edge_ratio": evidence["selected_edge_sharpness_ratio"], "required_edge_min": 0.50, "thin_feature_ratio": evidence["selected_thin_feature_response_ratio"], "required_thin_min": 0.40},
        "VISIBILITY_TOPOLOGY_COMPONENT_REDUCED": {"v084_topology_fraction": evidence["topology_before"], "v085_visibility_owner_fraction": evidence["visibility_owner_component_after"], "required_relative_reduction": config.topology_reduction_minimum},
        "SOURCE_ROOT_TOPOLOGY_REMAINS": {"source_resampling_fraction": evidence["source_resampling_component_after"], "source_cells_recomputed": True},
        "ANALYTIC_SOFT_TRANSPORT_MATCHES_FROZEN_FD": {"median_relative_error": evidence["median_analytic_vs_C_relative_error"], "required_max": config.analytic_fd_relative_tolerance},
        "FULL_RERENDER_MISMATCH_REDUCED": {"before_topology_fraction": evidence["topology_before"], "after_topology_fraction": evidence["topology_after"], "required_after_less_than_before": True},
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED": {"relative_reduction": evidence["topology_relative_reduction"], "required_min": config.topology_reduction_minimum},
        "PACKET_KERNEL_GEOMETRY_ARTIFACT_DETECTED": {"small_optimization_run": False, "detection_available": False},
        "SMALL_GEOMETRY_OPTIMIZATION_READY": {"required_prerequisites": prerequisites, "failed_prerequisites": [name for name in prerequisites if not verdicts[name]]},
        "HIGH_RES_BIRTH_READY_TO_RETEST": {"source_root_topology_remains": verdicts["SOURCE_ROOT_TOPOLOGY_REMAINS"], "small_optimization_validated": False, "birth_run": False},
    }
    return verdicts, evidence


def run_finite_packet_experiment(
    mesh_path: Path,
    artifact_directory: Path = Path("artifacts"),
    figure_directory: Path = Path("figures"),
    render_directory: Path = Path("render_res"),
    config: FinitePacketConfig | None = None,
) -> dict[str, Any]:
    config = config or FinitePacketConfig()
    started = time.perf_counter()
    _progress("v085_start", surface_samples=config.surface_samples, views=config.views)
    historical_path = artifact_directory / "v084_mc_continuous_detector.json"
    with historical_path.open() as stream:
        historical_v084 = json.load(stream)
    prepared = prepare_stanford_bunny(mesh_path, build_surface_scaffold=False)
    corrected = CorrectedBirthConfig(
        dictionary_count=64, initial_count=32,
        surface_samples=config.surface_samples, views=config.views,
        resolution=config.resolutions[0][0], resolution_width=config.resolutions[0][1],
        surface_scramble_seed=config.seed,
    )
    context = _build_context(prepared, corrected)
    _progress("main_context_ready")
    field = context.base
    points, normals = context.reference_points, context.reference_normals
    atlas = nested_fibonacci_atlas(points.device, (config.views,))
    boundary = enclosing_observation_sphere(points)
    h = _surface_spacing(points)
    parameters = _render_parameters(config, h)
    hard_images: dict[tuple[int, int], list[np.ndarray]] = {}
    base_cells = None
    for resolution in config.resolutions:
        images, _ = _hard_images(
            field, points, normals, context, atlas, boundary, resolution,
            recompute_visibility=base_cells is None, base_cells=base_cells,
        )
        hard_images[resolution] = images
        if base_cells is None:
            base_cells, _ = _visibility_cells(field, points, normals, atlas, boundary, corrected)
        _progress("hard_reference", resolution=list(resolution))
    point = _soft_render(
        field, points, normals, atlas, boundary, config.resolutions,
        radius=0.0, epsilon=h, path_step=0.5 * h, micro_samples=1,
        eta_relative=config.levelset_eta_relative, kappa=parameters["kappa"], surface_barrier=True,
        sensor_gain=config.sensor_gain, ambient=config.ambient, detector_extent=config.detector_extent,
        chunk_size=config.emitter_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
    )
    finite = _soft_render(
        field, points, normals, atlas, boundary, config.resolutions,
        radius=parameters["radius"], epsilon=parameters["epsilon"],
        path_step=parameters["path_step"], micro_samples=config.micro_samples,
        eta_relative=config.levelset_eta_relative, kappa=parameters["kappa"], surface_barrier=True,
        sensor_gain=config.sensor_gain, ambient=config.ambient, detector_extent=config.detector_extent,
        chunk_size=config.emitter_chunk_size,
        launch_exclusion_factor=config.launch_exclusion_over_r_plus_epsilon,
    )
    _progress("main_soft_renders", point_seconds=point.runtime_seconds, finite_seconds=finite.runtime_seconds)
    fidelity: dict[str, Any] = {"rows": []}
    for resolution in config.resolutions:
        point_metrics = _image_metrics(point.images[resolution], hard_images[resolution])
        finite_metrics = _image_metrics(finite.images[resolution], hard_images[resolution])
        fidelity["rows"].extend([
            {"resolution": list(resolution), "variant": "point_soft", **point_metrics},
            {"resolution": list(resolution), "variant": "finite_packet_soft", **finite_metrics},
        ])
    fidelity["selected_256"] = next(
        row for row in fidelity["rows"]
        if row["resolution"] == list(config.resolutions[0]) and row["variant"] == "finite_packet_soft"
    )
    sphere_cap = _sphere_cap_test()
    toys = _toy_transitions(config)
    local_overlap = _local_plane_overlap_test(config)
    analytic_scale = _analytic_levelset_scale_invariance(config)
    real_scale = _scale_invariance_real(field, points, normals, atlas, boundary, parameters, config)
    convergence = _path_and_micro_convergence(field, points, normals, atlas, boundary, parameters, config)
    opacity = _opacity_and_angle_test(config)
    _progress("analytic_and_toy_controls_ready")
    sweep = _radius_epsilon_sweep(field, points, normals, context, atlas, boundary, h, config)
    _progress("radius_sweep_ready", rows=len(sweep))
    cells = sign_changing_cells(field.grid)
    multiseed = _multiseed_test(field, cells, atlas, boundary, parameters, config)
    _progress("multiseed_ready", seeds=len(config.seeds))
    geometry = _finite_packet_geometry_diagnostic(prepared, config, historical_v084)
    _progress("geometry_fd_ready", rows=len(geometry["fd_rows"]))
    report: dict[str, Any] = {
        "version": "0.8.5",
        "scope": "continuous zero-set-native forward energy transport operator; no novelty or physical-exactness claim",
        "configuration": asdict(config),
        "environment": cuda_environment(),
        "transport_definition": {
            "root_free": True,
            "first_root_selected": False,
            "normalized_coordinate": "d_F=F/sqrt(||grad F||^2+eta^2)",
            "compact_shell": "psi(q)=15/16(1-q^2)^2 for |q|<1, zero otherwise",
            "packet_support": "fixed Sobol quadrature in a uniform 3D ball",
            "optical_depth": "tau=kappa integral mean_u psi_epsilon(d_F(x(t)+r u))*|n_F dot omega| dt",
            "transmission": "T=exp(-tau)",
            "energy": "E_out=T(F) E_in",
            "source_topology_frozen_or_remaining": "sign-changing-cell selection and normal-line root identity remain discrete",
        },
        "required_variants": {
            "A": "hard visibility plus continuous cubic detector",
            "B": "point soft zero-set attenuation (r=0)",
            "C": "finite-support packet soft zero-set attenuation",
            "D": "analytic half-space sphere-cap control",
        },
        "transport_derivative": {
            "optical_depth": "tau(lambda)=kappa integral sigma(F_lambda,x(t)) dt",
            "transmission": "dT/dlambda=-T dtau/dlambda",
            "energy": "dE_out/dlambda=E_in dT/dlambda + T dE_in/dlambda",
            "image": "continuous cubic detector projection and kernel derivatives are differentiated through autodiff",
            "not_differentiated": "discrete sign-changing-cell count/order and source root identity",
        },
        "surface_spacing_and_selected_parameters": parameters,
        "sphere_cap": sphere_cap,
        "local_plane_and_packet_quadrature": local_overlap,
        "toy_transitions": toys,
        "energy_linearity": _energy_linearity_test(),
        "surface_opacity": opacity,
        "levelset_scale_invariance": {"analytic": analytic_scale, "real_scene": real_scale},
        "path_micro_convergence": convergence,
        "radius_epsilon_sweep": sweep,
        "image_fidelity": fidelity,
        "multi_seed_mc": multiseed,
        "geometry_diagnostic": geometry,
        "small_geometry_optimization": {"run": False, "reason": "filled after prerequisite verdict audit"},
        "direct_photon_evaluation_transfer": {"run": False, "reason": "only allowed after a validated small optimization"},
        "birth_experiment_run": False,
        "performance": {
            "point_soft": {
                "runtime_seconds": point.runtime_seconds,
                "interaction_evaluations": point.interaction_evaluations,
                "attempted_packets": point.attempted_packets,
                "outward_packets": point.outward_packets,
                "peak_allocated_mib": point.peak_allocated_mib,
                "peak_reserved_mib": point.peak_reserved_mib,
            },
            "finite_packet_soft": {
                "runtime_seconds": finite.runtime_seconds,
                "interaction_evaluations": finite.interaction_evaluations,
                "attempted_packets": finite.attempted_packets,
                "outward_packets": finite.outward_packets,
                "peak_allocated_mib": finite.peak_allocated_mib,
                "peak_reserved_mib": finite.peak_reserved_mib,
            },
            "packets_per_second": finite.attempted_packets / max(finite.runtime_seconds, 1e-30),
            "effective_interaction_evaluations_per_second": finite.interaction_evaluations / max(finite.runtime_seconds, 1e-30),
            "mean_path_samples_per_outward_packet": finite.interaction_evaluations /
                max(finite.outward_packets * config.micro_samples, 1),
            "micro_quadrature_points_per_packet": config.micro_samples,
            "bounded_streaming_chunk_size": config.emitter_chunk_size,
            "full_packet_graph_materialized": False,
            "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "commands": {
            "formal": "PYTHONPATH=src python demo.py --finite-packet-transport --bunny-mesh data/stanford_bunny/cache/bun_zipper.ply --bunny-artifacts artifacts --bunny-figures figures --render-output render_res",
            "compile": "python -m py_compile demo.py src/zlt/finite_packet.py",
            "historical_regression": "python demo.py --verify",
        },
        "historical_v084_topology_reference": {
            "path": str(historical_path),
            "reported_median_approximately": 0.9924,
            "computed_from_best_rows": geometry["topology_fraction_before_v084_median"],
        },
        "limitations": [
            "finite packet attenuation is an algorithmic surface-barrier model, not a claim of exact radiometry",
            "source sign-changing-cell selection and source root identities remain discrete",
            "fixed finite packet quadrature can retain integration variance",
            "soft-vs-hard image differences are measurement-model bias, not geometry error",
        ],
    }
    verdicts, evidence = _verdicts(report, config)
    report["verdicts"] = verdicts
    report["verdict_evidence"] = evidence
    if verdicts["SMALL_GEOMETRY_OPTIMIZATION_READY"]:
        report["small_geometry_optimization"] = {
            "run": False,
            "reason": "all numerical gates passed, but source-root topology remains and this run reserves optimization for a separately reviewed follow-up",
        }
        # Readiness means the experiment is permitted, not that a result exists;
        # consequently birth remains false and no optimization claim is made.
    core = [
        "LEVELSET_SCALE_INVARIANT", "SPHERE_CAP_ANALYTIC_VALIDATED",
        "FINITE_PACKET_VALUE_CONTINUOUS", "FINITE_PACKET_GRADIENT_CONTINUOUS",
        "ROOT_FREE_TRANSPORT_IMPLEMENTED", "ENERGY_TRANSPORT_LINEAR",
        "PATH_QUADRATURE_CONVERGED", "SURFACE_OPACITY_CALIBRATED",
        "FINITE_PACKET_MC_VARIANCE_ACCEPTABLE", "GEOMETRY_SIGNAL_PRESERVED",
        "SILHOUETTE_BANDWIDTH_ACCEPTABLE", "ANALYTIC_SOFT_TRANSPORT_MATCHES_FROZEN_FD",
        "FULL_RERENDER_TOPOLOGY_FRACTION_REDUCED",
    ]
    report["PRIMARY_TRANSPORT"] = (
        "FINITE_SUPPORT_ZEROSET_ENERGY_PACKETS" if all(verdicts[name] for name in core)
        else "UNRESOLVED"
    )
    report["runtime_seconds"] = time.perf_counter() - started
    render_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = render_directory / "v085_hard_vs_soft_transport_rgb.png"
    comparison = np.concatenate([
        np.concatenate([np.clip(image, 0, 1) for image in hard_images[config.resolutions[0]]], axis=1),
        np.concatenate([np.clip(image, 0, 1) for image in point.images[config.resolutions[0]]], axis=1),
        np.concatenate([np.clip(image, 0, 1) for image in finite.images[config.resolutions[0]]], axis=1),
    ], axis=0)
    plt.imsave(comparison_path, comparison)
    captures = {"comparison_images": {
        "hard": hard_images[config.resolutions[0]],
        "point_soft": point.images[config.resolutions[0]],
        "finite_packet": finite.images[config.resolutions[0]],
    }}
    report["figures"] = _save_figures(figure_directory, report, captures)
    report["render_comparison"] = str(comparison_path)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    json_path = artifact_directory / "v085_finite_packet_zero_set_transport.json"
    csv_path = artifact_directory / "v085_finite_packet_zero_set_transport.csv"
    report["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    with json_path.open("w") as stream:
        json.dump(_json_ready(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_csv(csv_path, _scalar_csv_rows(_json_ready(report)))
    _progress("v085_complete", runtime_seconds=report["runtime_seconds"], primary=report["PRIMARY_TRANSPORT"])
    return report
