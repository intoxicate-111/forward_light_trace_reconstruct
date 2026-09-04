"""Falsification experiment for utility prediction of nonexistent geometry DoFs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .camera import PlanarCamera
from .fields import (
    LocalBasisField,
    SphereField,
    deform_reference_surface,
    unit_normals,
)
from .jacobian import (
    FixedTransportCell,
    _continuous_sensor,
    _photon_batch,
    build_fixed_transport_cell,
    deterministic_directions,
    make_local_basis_field,
    render_fixed_transport_cell,
    sparse_geometry_image_jacobian,
    topology_event_pixels,
)
from .locality import (
    BasisLayout,
    LocalZeroSet,
    SupportPairs,
    hierarchical_surface_points,
    make_support,
    support_radius,
    wendland_gradients,
    wendland_values,
)
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    multiview_cameras as shared_multiview_cameras,
    project_camera,
)
from .benchmark import BenchmarkGeometry, cuda_environment
from .tracer import PhotonBatch, first_zero_set_intersections


Tensor = torch.Tensor
SCORE_NAMES = (
    "random",
    "visibility",
    "local_residual",
    "jacobian_norm",
    "raw_alignment",
    "quadratic_gain",
    "orthogonalized_gain",
)


@dataclass(frozen=True)
class BirthConfig:
    current_dofs: int = 32
    target_dofs: int = 64
    candidates: int = 64
    emitters: int = 256
    views: int = 8
    resolution: int = 256
    packets_per_emitter: int = 8
    cone_power: float = 128.0
    current_radius: float = 0.60
    fine_radius: float = 0.34
    target_amplitude: float = 0.004
    damping: float = 1e-8
    baseline_iterations: int = 64
    oracle_iterations: int = 5
    coefficient_limit: float = 0.03


@dataclass
class OptimizationResult:
    coefficients: Tensor
    points: Tensor
    loss: float
    initial_loss: float
    iterations: int
    gradient_norm: float
    relative_improvement: float
    last_step_relative_improvement: float
    termination_reason: str


def _field_to(field: LocalBasisField, device: torch.device) -> LocalBasisField:
    return LocalBasisField(
        field.base,
        field.centers.to(device),
        field.radii.to(device),
        field.coefficients.to(device),
    )


def _camera(
    direction: tuple[float, float, float], resolution: int, device: torch.device
) -> PlanarCamera:
    center = 3.0 * torch.tensor(direction, dtype=torch.float64, device=device)
    normal = -center / torch.linalg.vector_norm(center)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    if abs(float(normal @ world_up)) > 0.9:
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
    right = torch.linalg.cross(normal, world_up)
    right = right / torch.linalg.vector_norm(right)
    up = torch.linalg.cross(right, normal)
    return PlanarCamera(
        center, normal, right, up, 4.0, 4.0, (resolution, resolution)
    )


def multiview_cameras(
    resolution: int, views: int, device: torch.device
) -> list[PlanarCamera]:
    directions = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (2**-0.5, 0.0, 2**-0.5),
        (-2**-0.5, 0.0, 2**-0.5),
        (0.0, 2**-0.5, -(2**-0.5)),
    )
    if not 1 <= views <= len(directions):
        raise ValueError("views must be between 1 and 8")
    return [_camera(direction, resolution, device) for direction in directions[:views]]


def _build_cells(
    field: LocalBasisField,
    points: Tensor,
    reference_normals: Tensor,
    colors: Tensor,
    cameras: list[PlanarCamera],
    config: BirthConfig,
) -> list[FixedTransportCell]:
    return [
        build_fixed_transport_cell(
            field,
            camera,
            points,
            reference_normals,
            colors,
            packets_per_emitter=config.packets_per_emitter,
            cone_power=config.cone_power,
            normal_mode="reference",
        )
        for camera in cameras
    ]


def _deform(
    field: LocalBasisField, reference_points: Tensor, reference_normals: Tensor
) -> tuple[Tensor, Tensor]:
    points, success, _ = deform_reference_surface(
        field, reference_points, reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError("candidate experiment left the normal-line root bracket")
    return points, success


def _render(cells: list[FixedTransportCell], points: Tensor) -> list[Tensor]:
    return [render_fixed_transport_cell(cell, points).reshape(-1) for cell in cells]


def _loss(images: list[Tensor], targets: list[Tensor]) -> Tensor:
    return 0.5 * sum(((image - target) ** 2).sum() for image, target in zip(images, targets))


def _sparse_product(left: Tensor, right: Tensor) -> Tensor:
    product = torch.sparse.mm(left.transpose(0, 1), right)
    return product.to_dense() if product.layout != torch.strided else product


def _normal_equations(
    field: LocalBasisField,
    points: Tensor,
    reference_normals: Tensor,
    cells: list[FixedTransportCell],
    images: list[Tensor],
    targets: list[Tensor],
) -> tuple[Tensor, Tensor, float]:
    count = field.parameter_count
    gram = torch.zeros((count, count), dtype=points.dtype, device=points.device)
    gradient = torch.zeros(count, dtype=points.dtype, device=points.device)
    for cell, image, target in zip(cells, images, targets):
        jacobian = sparse_geometry_image_jacobian(
            field, points, reference_normals, cell
        )
        residual = image - target
        gram += _sparse_product(jacobian, jacobian)
        gradient += torch.sparse.mm(
            jacobian.transpose(0, 1), residual[:, None]
        ).flatten()
    return gram, gradient, float(torch.linalg.vector_norm(gradient))


def _optimize(
    template: LocalBasisField,
    initial_coefficients: Tensor,
    reference_points: Tensor,
    reference_normals: Tensor,
    cells: list[FixedTransportCell],
    targets: list[Tensor],
    active_parameters: Tensor,
    *,
    iterations: int,
    damping: float,
    coefficient_limit: float,
) -> OptimizationResult:
    coefficients = initial_coefficients.clone()
    initial_loss = math.nan
    gradient_norm = math.inf
    completed = 0
    last_relative = math.inf
    termination_reason = "iteration_budget"
    for iteration in range(iterations):
        field = template.with_coefficients(coefficients)
        points, _ = _deform(field, reference_points, reference_normals)
        images = _render(cells, points)
        loss_before = float(_loss(images, targets))
        if iteration == 0:
            initial_loss = loss_before
        gram, gradient, gradient_norm = _normal_equations(
            field, points, reference_normals, cells, images, targets
        )
        local_gram = gram[active_parameters[:, None], active_parameters[None, :]]
        local_gradient = gradient[active_parameters]
        system = local_gram + damping * torch.eye(
            active_parameters.numel(), dtype=points.dtype, device=points.device
        )
        gauss_newton_step = torch.linalg.solve(system, -local_gradient)
        steepest_step = -local_gradient / local_gram.diagonal().abs().max().clamp_min(
            damping
        )
        accepted = False
        accepted_loss = loss_before
        for direction in (gauss_newton_step, steepest_step):
            for scale in (1.0, 0.25, 0.0625, 0.015625, 0.00390625):
                proposal = coefficients.clone()
                proposal[active_parameters] += scale * direction
                proposal = proposal.clamp(-coefficient_limit, coefficient_limit)
                proposal_field = template.with_coefficients(proposal)
                proposal_points, _ = _deform(
                    proposal_field, reference_points, reference_normals
                )
                proposal_loss = float(_loss(_render(cells, proposal_points), targets))
                if proposal_loss < loss_before:
                    coefficients = proposal
                    accepted_loss = proposal_loss
                    accepted = True
                    break
            if accepted:
                break
        completed = iteration + 1
        relative = (loss_before - accepted_loss) / max(loss_before, 1e-30)
        last_relative = relative
        if not accepted:
            termination_reason = "no_descent_step"
            break
        if relative < 1e-7:
            termination_reason = "relative_improvement"
            break
        if gradient_norm < 1e-10:
            termination_reason = "gradient_norm"
            break
    final_field = template.with_coefficients(coefficients)
    final_points, _ = _deform(
        final_field, reference_points, reference_normals
    )
    final_loss = float(_loss(_render(cells, final_points), targets))
    return OptimizationResult(
        coefficients=coefficients,
        points=final_points,
        loss=final_loss,
        initial_loss=initial_loss,
        iterations=completed,
        gradient_norm=gradient_norm,
        relative_improvement=(initial_loss - final_loss) / max(initial_loss, 1e-30),
        last_step_relative_improvement=last_relative,
        termination_reason=termination_reason,
    )


def _candidate_pool(
    base: SphereField,
    target_field: LocalBasisField,
    target_coefficients: Tensor,
    current_field: LocalBasisField,
    reference_points: Tensor,
    count: int,
) -> tuple[Tensor, Tensor, list[str], Tensor, int, int]:
    if count < 8:
        raise ValueError("at least eight candidates are required")
    target_support = torch.linalg.vector_norm(
        target_field.basis_values(reference_points), dim=0
    )
    order = torch.argsort(
        target_coefficients.abs() * target_support, descending=True
    )
    repeated = order.repeat(math.ceil((count - 2) / order.numel()))[: count - 2]
    centers = target_field.centers[repeated].clone()
    radii = target_field.radii[repeated].clone()
    labels = ["ordinary" for _ in range(count)]
    source_ids = torch.cat(
        (
            torch.full((1,), -1, dtype=torch.long, device=order.device),
            repeated,
            torch.full((1,), -1, dtype=torch.long, device=order.device),
        )
    )

    redundant_id = 0
    invisible_id = count - 1
    current_support = torch.linalg.vector_norm(
        current_field.basis_values(reference_points), dim=0
    )
    redundant_source = int(current_support.argmax())
    centers = torch.cat(
        (
            current_field.centers[redundant_source : redundant_source + 1],
            centers,
            target_field.centers[:1],
        ),
        dim=0,
    )
    radii = torch.cat(
        (
            current_field.radii[redundant_source : redundant_source + 1],
            radii,
            target_field.radii[:1],
        ),
        dim=0,
    )
    labels[redundant_id] = "redundant_control"

    generator = torch.Generator().manual_seed(907)
    probes = base.sample_surface(4096, generator).to(reference_points.device)
    nearest = torch.cdist(probes, reference_points).min(dim=1).values
    farthest = int(nearest.argmax())
    centers[invisible_id] = probes[farthest]
    radii[invisible_id] = 0.45 * nearest[farthest]
    labels[invisible_id] = "invisible_control"
    return centers, radii, labels, source_ids, redundant_id, invisible_id


def _slice_columns(matrix: Tensor, start: int, stop: int) -> Tensor:
    matrix = matrix.coalesce()
    indices = matrix.indices()
    keep = (indices[1] >= start) & (indices[1] < stop)
    sliced_indices = indices[:, keep].clone()
    sliced_indices[1] -= start
    return torch.sparse_coo_tensor(
        sliced_indices,
        matrix.values()[keep],
        size=(matrix.shape[0], stop - start),
        dtype=matrix.dtype,
        device=matrix.device,
    ).coalesce()


def _column_dense(matrix: Tensor, column: int) -> Tensor:
    matrix = matrix.coalesce()
    keep = matrix.indices()[1] == column
    result = torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    result.index_add_(0, matrix.indices()[0, keep], matrix.values()[keep])
    return result


def _candidate_statistics(
    analysis_field: LocalBasisField,
    current_dofs: int,
    points: Tensor,
    reference_normals: Tensor,
    cells: list[FixedTransportCell],
    baseline_images: list[Tensor],
    targets: list[Tensor],
    damping: float,
) -> tuple[dict[str, Tensor], dict[str, object], list[Tensor]]:
    candidate_count = analysis_field.parameter_count - current_dofs
    gram = torch.zeros(
        (current_dofs, current_dofs), dtype=points.dtype, device=points.device
    )
    cross = torch.zeros(
        (current_dofs, candidate_count), dtype=points.dtype, device=points.device
    )
    candidate_norm_squared = torch.zeros(
        candidate_count, dtype=points.dtype, device=points.device
    )
    current_residual = torch.zeros(
        current_dofs, dtype=points.dtype, device=points.device
    )
    candidate_residual = torch.zeros(
        candidate_count, dtype=points.dtype, device=points.device
    )
    local_residual_squared = torch.zeros_like(candidate_residual)
    visibility = torch.zeros_like(candidate_residual)
    nnz = torch.zeros(candidate_count, dtype=torch.long, device=points.device)
    affected_pixels = torch.zeros_like(nnz)
    candidate_matrices: list[Tensor] = []

    for cell, image, target in zip(cells, baseline_images, targets):
        combined = sparse_geometry_image_jacobian(
            analysis_field, points, reference_normals, cell
        )
        current = _slice_columns(combined, 0, current_dofs)
        candidates = _slice_columns(combined, current_dofs, combined.shape[1])
        candidate_matrices.append(candidates)
        residual = image - target
        gram += _sparse_product(current, current)
        cross += _sparse_product(current, candidates)
        current_residual += torch.sparse.mm(
            current.transpose(0, 1), residual[:, None]
        ).flatten()
        candidate_residual += torch.sparse.mm(
            candidates.transpose(0, 1), residual[:, None]
        ).flatten()
        coalesced = candidates.coalesce()
        columns = coalesced.indices()[1]
        candidate_norm_squared.scatter_add_(0, columns, coalesced.values() ** 2)
        nnz += torch.bincount(columns, minlength=candidate_count)
        for candidate in range(candidate_count):
            rows = coalesced.indices()[0, columns == candidate]
            pixels = torch.unique(torch.div(rows, 3, rounding_mode="floor"))
            if pixels.numel():
                visibility[candidate] += 1.0
                affected_pixels[candidate] += pixels.numel()
                rgb_rows = (
                    pixels[:, None] * 3
                    + torch.arange(3, device=points.device)[None, :]
                ).reshape(-1)
                local_residual_squared[candidate] += (residual[rgb_rows] ** 2).sum()

    system = gram + damping * torch.eye(
        current_dofs, dtype=points.dtype, device=points.device
    )
    candidate_projection = torch.linalg.solve(system, cross)
    residual_projection = torch.linalg.solve(system, current_residual)
    projected_norm_squared = (
        candidate_norm_squared
        - 2.0 * (candidate_projection * cross).sum(dim=0)
        + (candidate_projection * (gram @ candidate_projection)).sum(dim=0)
    ).clamp_min(0.0)
    projected_residual_dot = (
        candidate_residual
        - cross.T @ residual_projection
        - candidate_projection.T @ current_residual
        + candidate_projection.T @ (gram @ residual_projection)
    )
    jacobian_norm = torch.sqrt(candidate_norm_squared)
    novelty_norm = torch.sqrt(projected_norm_squared)

    generator = torch.Generator().manual_seed(1701)
    scores = {
        "random": torch.rand(candidate_count, generator=generator).to(points.device),
        "visibility": visibility,
        "local_residual": torch.sqrt(local_residual_squared),
        "jacobian_norm": jacobian_norm,
        "raw_alignment": candidate_residual.abs(),
        "quadratic_gain": candidate_residual**2 / (candidate_norm_squared + damping),
        "orthogonalized_gain": projected_residual_dot**2
        / (projected_norm_squared + damping),
        "novelty_ratio": novelty_norm / (jacobian_norm + 1e-15),
        "novelty_norm": novelty_norm,
    }

    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular_values = torch.sqrt(eigenvalues.flip(0))
    rank_threshold = float(singular_values[0]) * 1e-6
    rank = int((singular_values > rank_threshold).sum())
    rank_increase = torch.zeros(candidate_count, dtype=torch.bool, device=points.device)
    for candidate in range(candidate_count):
        augmented = torch.zeros(
            (current_dofs + 1, current_dofs + 1),
            dtype=points.dtype,
            device=points.device,
        )
        augmented[:current_dofs, :current_dofs] = gram
        augmented[:current_dofs, current_dofs] = cross[:, candidate]
        augmented[current_dofs, :current_dofs] = cross[:, candidate]
        augmented[current_dofs, current_dofs] = candidate_norm_squared[candidate]
        augmented_values = torch.sqrt(
            torch.linalg.eigvalsh(augmented).clamp_min(0.0)
        )
        rank_increase[candidate] = int(
            (augmented_values > rank_threshold).sum()
        ) > rank
    diagnostics: dict[str, object] = {
        "current_rank": rank,
        "rank_threshold": rank_threshold,
        "rank_increase": rank_increase,
        "visibility": visibility,
        "nnz": nnz,
        "affected_pixels": affected_pixels,
    }
    return scores, diagnostics, candidate_matrices


def _finite_difference_validation(
    current_field: LocalBasisField,
    checkpoint_coefficients: Tensor,
    candidate_centers: Tensor,
    candidate_radii: Tensor,
    candidate_matrices: list[Tensor],
    candidate_ids: list[int],
    reference_points: Tensor,
    reference_normals: Tensor,
    colors: Tensor,
    cells: list[FixedTransportCell],
) -> dict[str, object]:
    errors: list[float] = []
    event_fractions: list[float] = []
    best_steps: list[float] = []
    for candidate in candidate_ids:
        template = LocalBasisField(
            current_field.base,
            torch.cat((current_field.centers, candidate_centers[candidate : candidate + 1])),
            torch.cat((current_field.radii, candidate_radii[candidate : candidate + 1])),
            torch.cat((checkpoint_coefficients, checkpoint_coefficients.new_zeros(1))),
        )
        best_error = math.inf
        best_events = 1.0
        best_step = 0.0
        for step in (1e-4, 3e-5, 1e-5):
            plus = template.coefficients.clone()
            minus = template.coefficients.clone()
            plus[-1] = step
            minus[-1] = -step
            plus_field = template.with_coefficients(plus)
            minus_field = template.with_coefficients(minus)
            plus_points, _ = _deform(
                plus_field, reference_points, reference_normals
            )
            minus_points, _ = _deform(
                minus_field, reference_points, reference_normals
            )
            numerator_squared = torch.zeros((), dtype=torch.float64, device=plus.device)
            denominator_squared = torch.zeros_like(numerator_squared)
            event_count = 0
            for view, cell in enumerate(cells):
                finite = (
                    render_fixed_transport_cell(cell, plus_points).reshape(-1)
                    - render_fixed_transport_cell(cell, minus_points).reshape(-1)
                ) / (2.0 * step)
                analytic = _column_dense(candidate_matrices[view], candidate)
                events = topology_event_pixels(
                    template,
                    reference_points,
                    reference_normals,
                    colors,
                    cell,
                    plus,
                ) | topology_event_pixels(
                    template,
                    reference_points,
                    reference_normals,
                    colors,
                    cell,
                    minus,
                )
                stable_rgb = (~events)[:, None].expand(-1, 3).reshape(-1)
                numerator_squared += ((finite - analytic)[stable_rgb] ** 2).sum()
                denominator_squared += (analytic[stable_rgb] ** 2).sum()
                event_count += int(events.sum())
            error = float(
                torch.sqrt(numerator_squared / denominator_squared.clamp_min(1e-30))
            )
            event_fraction = event_count / (len(cells) * cells[0].camera.pixel_count)
            if error < best_error:
                best_error = error
                best_events = event_fraction
                best_step = step
        errors.append(best_error)
        event_fractions.append(best_events)
        best_steps.append(best_step)
    return {
        "candidate_ids": candidate_ids,
        "best_steps": best_steps,
        "median_relative_error": statistics.median(errors),
        "maximum_relative_error": max(errors),
        "mean_transport_event_fraction": statistics.mean(event_fractions),
        "maximum_transport_event_fraction": max(event_fractions),
    }


def _rankdata(values: Tensor) -> Tensor:
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left: Tensor, right: Tensor, *, ranked: bool) -> float:
    x = _rankdata(left) if ranked else left.detach().cpu().to(torch.float64)
    y = _rankdata(right) if ranked else right.detach().cpu().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y) / denominator) if denominator > 0.0 else 0.0


def _selection_report(scores: dict[str, Tensor], actual_gain: Tensor) -> dict[str, object]:
    best = float(actual_gain.max())
    report: dict[str, object] = {}
    for name in SCORE_NAMES:
        order = torch.argsort(scores[name], descending=True, stable=True)
        selections: dict[str, object] = {}
        for count in (1, 3, 5, 10):
            selected = actual_gain[order[: min(count, order.numel())]]
            selections[f"top_{count}"] = {
                "candidate_ids": order[: min(count, order.numel())].tolist(),
                "mean_actual_gain": float(selected.mean()),
                "best_actual_gain": float(selected.max()),
                "oracle_regret": best - float(selected.max()),
            }
        report[name] = selections
    return report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    directory: Path,
    scores: dict[str, Tensor],
    actual_gain: Tensor,
    correlations: dict[str, dict[str, float]],
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    proposed = scores["orthogonalized_gain"].detach().cpu()
    gains = actual_gain.detach().cpu()

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(proposed, gains, s=18)
    axis.set_xlabel("orthogonalized predicted gain")
    axis.set_ylabel("actual joint-reoptimization gain")
    figure.tight_layout()
    path = directory / "v03a_score_vs_gain.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    order = torch.argsort(proposed, descending=True)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.bar(torch.arange(order.numel()), gains[order], width=0.8)
    axis.set_xlabel("candidate rank by proposed score")
    axis.set_ylabel("actual gain")
    figure.tight_layout()
    path = directory / "v03a_gain_by_predicted_rank.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    figure, axis = plt.subplots(figsize=(7, 3.5))
    values = [correlations[name]["spearman"] for name in SCORE_NAMES]
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(values)), SCORE_NAMES, rotation=30, ha="right")
    axis.set_ylabel("Spearman correlation")
    figure.tight_layout()
    path = directory / "v03a_spearman_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))
    return written


def run_birth_experiment(
    config: BirthConfig = BirthConfig(),
    *,
    csv_path: Path | None = None,
    figure_directory: Path | None = None,
) -> dict[str, object]:
    """Run one deterministic candidate-score versus independent-birth oracle study."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    if config.current_dofs != 32 or config.target_dofs <= config.current_dofs:
        raise ValueError("v0.3a requires K=32 and K_GT>K")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base = SphereField()
    current = _field_to(
        make_local_basis_field(
            base, config.current_dofs, config.current_radius, seed=101
        ),
        device,
    )
    target_zero = _field_to(
        make_local_basis_field(
            base, config.target_dofs, config.fine_radius, seed=401
        ),
        device,
    )
    indices = torch.arange(config.target_dofs, dtype=torch.float64, device=device)
    raw_coefficients = torch.sin(1.71 * indices + 0.3) + 0.55 * torch.cos(
        0.73 * indices - 0.2
    )
    target_coefficients = (
        config.target_amplitude
        * raw_coefficients
        / raw_coefficients.abs().max()
    )
    target = target_zero.with_coefficients(target_coefficients)

    generator = torch.Generator().manual_seed(211)
    reference_points = base.sample_surface(config.emitters, generator).to(device)
    reference_normals = unit_normals(base, reference_points)
    colors = base.color(reference_points)
    cameras = multiview_cameras(config.resolution, config.views, device)
    base_points, success, _ = deform_reference_surface(
        current, reference_points, reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError("baseline surface sampling failed")
    cells = _build_cells(
        current, base_points, reference_normals, colors, cameras, config
    )
    target_points, success, _ = deform_reference_surface(
        target, reference_points, reference_normals
    )
    if not bool(success.all()):
        raise RuntimeError("target deformation left the safe root bracket")
    targets = _render(cells, target_points)
    target_event_pixels = sum(
        int(
            topology_event_pixels(
                target_zero,
                reference_points,
                reference_normals,
                colors,
                cell,
                target_coefficients,
            ).sum()
        )
        for cell in cells
    )
    target_event_fraction = target_event_pixels / (
        config.views * config.resolution**2
    )

    all_current = torch.arange(config.current_dofs, device=device)
    baseline_started = time.perf_counter()
    baseline = _optimize(
        current,
        current.coefficients,
        reference_points,
        reference_normals,
        cells,
        targets,
        all_current,
        iterations=config.baseline_iterations,
        damping=config.damping,
        coefficient_limit=config.coefficient_limit,
    )
    torch.cuda.synchronize()
    baseline_runtime = time.perf_counter() - baseline_started
    checkpoint = current.with_coefficients(baseline.coefficients)
    baseline_images = _render(cells, baseline.points)
    residual_norm = math.sqrt(2.0 * baseline.loss)

    (
        candidate_centers,
        candidate_radii,
        labels,
        candidate_source_ids,
        redundant_id,
        invisible_id,
    ) = _candidate_pool(
        base,
        target_zero,
        target_coefficients,
        current,
        reference_points,
        config.candidates,
    )
    analysis_field = LocalBasisField(
        base,
        torch.cat((current.centers, candidate_centers)),
        torch.cat((current.radii, candidate_radii)),
        torch.cat((baseline.coefficients, baseline.coefficients.new_zeros(config.candidates))),
    )

    scoring_started = time.perf_counter()
    scores, diagnostics, candidate_matrices = _candidate_statistics(
        analysis_field,
        config.current_dofs,
        baseline.points,
        reference_normals,
        cells,
        baseline_images,
        targets,
        config.damping,
    )
    source_coefficients = torch.where(
        candidate_source_ids >= 0,
        target_coefficients[candidate_source_ids.clamp_min(0)].abs(),
        torch.zeros_like(scores["jacobian_norm"]),
    )
    useful_id = int((source_coefficients * scores["jacobian_norm"]).argmax())
    labels[useful_id] = "known_missing_detail_control"
    torch.cuda.synchronize()
    scoring_runtime = time.perf_counter() - scoring_started
    responsive_order = torch.argsort(scores["jacobian_norm"], descending=True).tolist()
    validation_ids = [redundant_id, useful_id]
    validation_ids.extend(
        candidate
        for candidate in responsive_order
        if candidate not in validation_ids
    )
    validation_ids = validation_ids[:4]
    derivative = _finite_difference_validation(
        checkpoint,
        baseline.coefficients,
        candidate_centers,
        candidate_radii,
        candidate_matrices,
        validation_ids,
        reference_points,
        reference_normals,
        colors,
        cells,
    )
    if derivative["maximum_relative_error"] >= 1e-3:
        raise RuntimeError(f"candidate derivative validation failed: {derivative}")

    new_only_gains = torch.zeros(
        config.candidates, dtype=torch.float64, device=device
    )
    joint_gains = torch.zeros_like(new_only_gains)
    geometry_improvements = torch.zeros_like(new_only_gains)
    baseline_geometry_error = float(
        torch.sqrt(((baseline.points - target_points) ** 2).sum(dim=1).mean())
    )
    oracle_started = time.perf_counter()
    for candidate in range(config.candidates):
        template = LocalBasisField(
            base,
            torch.cat((current.centers, candidate_centers[candidate : candidate + 1])),
            torch.cat((current.radii, candidate_radii[candidate : candidate + 1])),
            torch.cat((baseline.coefficients, baseline.coefficients.new_zeros(1))),
        )
        initial = template.coefficients
        new_only = _optimize(
            template,
            initial,
            reference_points,
            reference_normals,
            cells,
            targets,
            torch.tensor([config.current_dofs], device=device),
            iterations=config.oracle_iterations,
            damping=config.damping,
            coefficient_limit=config.coefficient_limit,
        )
        joint = _optimize(
            template,
            initial,
            reference_points,
            reference_normals,
            cells,
            targets,
            torch.arange(config.current_dofs + 1, device=device),
            iterations=config.oracle_iterations,
            damping=config.damping,
            coefficient_limit=config.coefficient_limit,
        )
        new_only_gains[candidate] = max(0.0, baseline.loss - new_only.loss)
        joint_gains[candidate] = max(0.0, baseline.loss - joint.loss)
        joint_geometry_error = torch.sqrt(
            ((joint.points - target_points) ** 2).sum(dim=1).mean()
        )
        geometry_improvements[candidate] = baseline_geometry_error - joint_geometry_error
    torch.cuda.synchronize()
    oracle_runtime = time.perf_counter() - oracle_started

    correlations = {
        name: {
            "spearman": _correlation(scores[name], joint_gains, ranked=True),
            "pearson": _correlation(scores[name], joint_gains, ranked=False),
        }
        for name in SCORE_NAMES
    }
    proposed_beats_falsification_baselines = all(
        correlations["orthogonalized_gain"]["spearman"]
        > correlations[name]["spearman"]
        for name in ("random", "local_residual", "raw_alignment")
    )
    scientific_verdict = (
        "NONEXISTENT_DOF_UTILITY_PREDICTION_SUPPORTED"
        if proposed_beats_falsification_baselines
        else "NONEXISTENT_DOF_UTILITY_PREDICTION_NOT_SUPPORTED"
    )
    selection = _selection_report(scores, joint_gains)
    proposed_best = int(torch.argmax(scores["orthogonalized_gain"]))
    actual_best = int(torch.argmax(joint_gains))
    rank_increase = diagnostics["rank_increase"]

    rows: list[dict[str, object]] = []
    for candidate in range(config.candidates):
        center = candidate_centers[candidate]
        rows.append(
            {
                "candidate_id": candidate,
                "control_label": labels[candidate],
                "target_source_id": int(candidate_source_ids[candidate]),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "support_radius": float(candidate_radii[candidate]),
                "visibility": int(diagnostics["visibility"][candidate]),
                "jacobian_nnz": int(diagnostics["nnz"][candidate]),
                "affected_pixels": int(diagnostics["affected_pixels"][candidate]),
                "jacobian_norm": float(scores["jacobian_norm"][candidate]),
                "local_residual": float(scores["local_residual"][candidate]),
                "raw_alignment": float(scores["raw_alignment"][candidate]),
                "quadratic_score": float(scores["quadratic_gain"][candidate]),
                "novelty_ratio": float(scores["novelty_ratio"][candidate]),
                "orthogonalized_score": float(
                    scores["orthogonalized_gain"][candidate]
                ),
                "rank_increase": bool(rank_increase[candidate]),
                "actual_gain_new_only": float(new_only_gains[candidate]),
                "actual_gain_joint": float(joint_gains[candidate]),
                "relative_gain_joint": float(
                    joint_gains[candidate] / max(baseline.loss, 1e-30)
                ),
                "geometry_error_improvement": float(geometry_improvements[candidate]),
            }
        )
    if csv_path is not None:
        _write_csv(csv_path, rows)
    figures = (
        _write_figures(figure_directory, scores, joint_gains, correlations)
        if figure_directory is not None
        else []
    )

    def control(candidate: int) -> dict[str, object]:
        return {
            "candidate_id": candidate,
            "jacobian_norm": float(scores["jacobian_norm"][candidate]),
            "novelty_norm": float(scores["novelty_norm"][candidate]),
            "novelty_ratio": float(scores["novelty_ratio"][candidate]),
            "proposed_score": float(scores["orthogonalized_gain"][candidate]),
            "actual_gain_new_only": float(new_only_gains[candidate]),
            "actual_gain_joint": float(joint_gains[candidate]),
            "visibility": int(diagnostics["visibility"][candidate]),
            "rank_increase": bool(rank_increase[candidate]),
        }

    return {
        "configuration": {
            "current_dofs": config.current_dofs,
            "target_dofs": config.target_dofs,
            "candidates": config.candidates,
            "emitters": config.emitters,
            "views": config.views,
            "resolution": [config.resolution, config.resolution],
            "packets_per_emitter": config.packets_per_emitter,
            "damping": config.damping,
            "oracle_iterations": config.oracle_iterations,
            "device": torch.cuda.get_device_name(0),
        },
        "baseline": {
            "initial_loss": baseline.initial_loss,
            "converged_loss": baseline.loss,
            "relative_improvement": baseline.relative_improvement,
            "iterations": baseline.iterations,
            "gradient_norm": baseline.gradient_norm,
            "last_step_relative_improvement": baseline.last_step_relative_improvement,
            "termination_reason": baseline.termination_reason,
            "residual_norm": residual_norm,
            "geometry_rms_error": baseline_geometry_error,
            "runtime_seconds": baseline_runtime,
        },
        "candidate_derivative_validation": derivative,
        "target_transport_event_fraction": target_event_fraction,
        "correlations": correlations,
        "top_k_selection": selection,
        "best_predicted_candidate": {
            "candidate_id": proposed_best,
            "actual_gain": float(joint_gains[proposed_best]),
            "relative_gain": float(
                joint_gains[proposed_best] / max(baseline.loss, 1e-30)
            ),
        },
        "true_best_candidate": {
            "candidate_id": actual_best,
            "actual_gain": float(joint_gains[actual_best]),
            "relative_gain": float(
                joint_gains[actual_best] / max(baseline.loss, 1e-30)
            ),
        },
        "rank": {
            "current": diagnostics["current_rank"],
            "threshold": diagnostics["rank_threshold"],
            "candidates_increasing_rank": int(rank_increase.sum()),
        },
        "controls": {
            "redundant": control(redundant_id),
            "known_missing_detail": control(useful_id),
            "invisible": control(invisible_id),
        },
        "runtime": {
            "candidate_scoring_seconds": scoring_runtime,
            "oracle_seconds": oracle_runtime,
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "candidate_csv": str(csv_path) if csv_path is not None else None,
        "figures": figures,
        "scientific_verdict": scientific_verdict,
        "verdict_basis": {
            "proposed_beats_random_local_residual_and_raw_alignment_spearman": (
                proposed_beats_falsification_baselines
            ),
            "criterion": "the task's explicit falsification comparison",
        },
        "status": "completed",
    }


# v0.3b deliberately uses a separate sparse/local path.  The v0.3a functions
# above remain unchanged so its committed artifact stays reproducible.


@dataclass(frozen=True)
class KScalingConfig:
    k_values: tuple[int, ...] = (
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    )
    views: int = 8
    resolution: int = 256
    emitters_per_basis: int = 8
    packets_per_emitter: int = 8
    maximum_photons: int = 262144
    base_support_radius: float = 0.60
    target_rms_displacement: float = 0.003
    coefficient_limit: float = 0.015
    support_margin: float = 0.005
    baseline_iterations: int = 6
    cg_iterations: int = 12
    damping: float = 1e-8
    oracle_evaluations: int = 17
    unbiased_oracle_size: int = 128
    exhaustive_candidate_limit: int = 1024
    root_samples: int = 32
    root_bisection_steps: int = 20
    known_failed_k: int | None = None
    known_failure_reason: str | None = None


@dataclass
class _ScalingContext:
    k: int
    base: SphereField
    current_layout: BasisLayout
    target_layout: BasisLayout
    candidate_layout: BasisLayout
    current_support: SupportPairs
    target_support: SupportPairs
    candidate_support: SupportPairs
    reference_points: Tensor
    reference_normals: Tensor
    colors: Tensor
    cells: list[FixedTransportCell]
    cameras: list[PlanarCamera]
    state: SceneTransportState
    geometry: BenchmarkGeometry
    packets_per_emitter: int
    candidate_labels: list[str]
    candidate_target_ids: Tensor
    redundant_id: int
    invisible_id: int


@dataclass
class _ScalingOptimization:
    coefficients: Tensor
    points: Tensor
    denominator: Tensor
    images: list[Tensor]
    loss: float
    initial_loss: float
    iterations: int
    last_step_relative_improvement: float
    gradient_norm: float
    root_failures: int


def _scaling_packets(k: int, config: KScalingConfig) -> int:
    emitters = config.emitters_per_basis * k
    return max(
        1,
        min(config.packets_per_emitter, config.maximum_photons // emitters),
    )


def _scaling_layouts(
    k: int, config: KScalingConfig, device: torch.device
) -> tuple[BasisLayout, BasisLayout, BasisLayout, list[str], Tensor, int, int]:
    base = SphereField()
    # Emitters use the first 8K Sobol points.  The invisible control is taken
    # beyond that prefix and given a tiny radius, rather than accidentally
    # placing it exactly on a sampled emitter.
    master = hierarchical_surface_points(base, 10 * k + 1, device)
    current_radius = support_radius(config.base_support_radius, k)
    target_radius = support_radius(config.base_support_radius, 2 * k)
    current = BasisLayout(
        master[:k],
        torch.full((k,), current_radius, dtype=torch.float64, device=device),
    )
    target = BasisLayout(
        master[: 2 * k],
        torch.full((2 * k,), target_radius, dtype=torch.float64, device=device),
    )
    candidate_centers = torch.cat((master[k : 2 * k], master[2 * k : 3 * k]))
    candidate_radii = torch.full(
        (2 * k,), target_radius, dtype=torch.float64, device=device
    )
    labels = ["missing_detail" for _ in range(k)] + [
        "distractor" for _ in range(k)
    ]
    target_ids = torch.cat(
        (
            torch.arange(k, 2 * k, device=device),
            torch.full((k,), -1, dtype=torch.long, device=device),
        )
    )
    redundant_id = k
    invisible_id = 2 * k - 1
    candidate_centers[redundant_id] = current.centers[0]
    candidate_radii[redundant_id] = current.radii[0]
    labels[redundant_id] = "redundant_control"
    candidate_centers[invisible_id] = master[10 * k]
    candidate_radii[invisible_id] = 0.03 * target_radius
    labels[invisible_id] = "invisible_control"
    candidates = BasisLayout(candidate_centers, candidate_radii)
    return current, target, candidates, labels, target_ids, redundant_id, invisible_id


def _build_scaling_context(
    k: int, config: KScalingConfig, device: torch.device
) -> _ScalingContext:
    base = SphereField()
    (
        current_layout,
        target_layout,
        candidate_layout,
        labels,
        target_ids,
        redundant_id,
        invisible_id,
    ) = _scaling_layouts(k, config, device)
    emitter_count = config.emitters_per_basis * k
    reference_points = hierarchical_surface_points(base, emitter_count, device)
    reference_normals = unit_normals(base, reference_points)
    colors = base.color(reference_points)
    _, current_support = make_support(
        reference_points, current_layout, margin=config.support_margin
    )
    _, target_support = make_support(
        reference_points, target_layout, margin=config.support_margin
    )
    _, candidate_support = make_support(
        reference_points, candidate_layout, margin=config.support_margin
    )
    packets = _scaling_packets(k, config)
    directions = deterministic_directions(reference_normals, packets, 128.0)
    photons = _photon_batch(reference_points, directions, colors, packets)
    maximum_times = torch.full(
        (photons.count,), 4.0, dtype=torch.float64, device=device
    )
    surface_hits, surface_times = first_zero_set_intersections(
        base,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
        chunk_size=8192,
    )
    state = SceneTransportState(
        reference_points,
        reference_normals,
        photons,
        directions,
        surface_hits,
        surface_times,
        torch.ones(emitter_count, dtype=torch.bool, device=device),
    )
    cameras = shared_multiview_cameras(
        "sphere", (config.resolution, config.resolution), device, count=config.views
    )
    geometry = BenchmarkGeometry(
        base, reference_points, reference_normals, colors  # type: ignore[arg-type]
    )
    cells = [project_camera(geometry, camera, state, MultiviewConfig(
        resolution=(config.resolution, config.resolution),
        emitters=emitter_count,
        packets_per_emitter=packets,
        parameter_count=k,
        cone_power=128.0,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )).cell for camera in cameras]
    return _ScalingContext(
        k,
        base,
        current_layout,
        target_layout,
        candidate_layout,
        current_support,
        target_support,
        candidate_support,
        reference_points,
        reference_normals,
        colors,
        cells,
        cameras,
        state,
        geometry,
        packets,
        labels,
        target_ids,
        redundant_id,
        invisible_id,
    )


def _target_coefficients(
    context: _ScalingContext, config: KScalingConfig, seed: int
) -> tuple[Tensor, Tensor, float]:
    indices = torch.arange(
        2 * context.k, dtype=torch.float64, device=context.reference_points.device
    )
    phase = 0.37 * seed
    raw = torch.sin(1.71 * indices + 0.3 + phase) + 0.55 * torch.cos(
        0.73 * indices - 0.2 - 0.5 * phase
    )
    raw = raw / torch.sqrt((raw * raw).mean())
    template = LocalZeroSet(
        context.base,
        context.target_layout,
        raw,
        context.reference_points,
        context.reference_normals,
        context.target_support,
    )

    def evaluate(scale: float) -> tuple[Tensor, Tensor, float, bool]:
        coefficients = scale * raw
        target = template.with_coefficients(coefficients)
        points, success, displacement, _ = target.deform()
        rms = float(torch.sqrt((displacement * displacement).mean()))
        return coefficients, points, rms, bool(success.all())

    desired = config.target_rms_displacement
    low_scale = 0.0
    high_scale = 0.003
    best: tuple[Tensor, Tensor, float, bool] | None = None
    high = evaluate(high_scale)
    while high[3] and high[2] < desired:
        low_scale = high_scale
        best = high
        high_scale *= 2.0
        high = evaluate(high_scale)
        if high_scale > 0.192:
            raise RuntimeError("target RMS normalization failed to find a bracket")

    # A failed upper endpoint is still useful: bisection locates the largest
    # valid scale and distinguishes an unreachable target from a poor first guess.
    for _ in range(24):
        middle_scale = 0.5 * (low_scale + high_scale)
        middle = evaluate(middle_scale)
        if middle[3]:
            if best is None or abs(middle[2] - desired) < abs(best[2] - desired):
                best = middle
            if middle[2] < desired:
                low_scale = middle_scale
            else:
                high_scale = middle_scale
        else:
            high_scale = middle_scale
    if best is None or abs(best[2] - desired) > 1e-6 * desired:
        achieved = best[2] if best is not None else 0.0
        raise RuntimeError(
            "fixed-RMS target is unreachable before the normal-line solve fails "
            f"(requested={desired:.9g}, largest_valid={achieved:.9g})"
        )
    return best[0], best[1], best[2]


def _images(cells: list[FixedTransportCell], points: Tensor) -> list[Tensor]:
    return [render_fixed_transport_cell(cell, points).reshape(-1) for cell in cells]


def _image_loss(images: list[Tensor], targets: list[Tensor]) -> float:
    return 0.5 * sum(float(((image - target) ** 2).sum()) for image, target in zip(images, targets))


def _local_sparse_jacobian(
    layout: BasisLayout,
    points: Tensor,
    reference_normals: Tensor,
    denominator: Tensor,
    support: SupportPairs,
    cell: FixedTransportCell,
    *,
    threshold: float = 1e-12,
) -> Tensor:
    local_points, basis_ids = support.subset_points(cell.emitter_ids)
    emitter_ids = cell.emitter_ids[local_points]
    offsets = points[emitter_ids] - layout.centers[basis_ids]
    basis_values = wendland_values(offsets, layout.radii[basis_ids])
    in_support = basis_values > 0.0
    local_points = local_points[in_support]
    basis_ids = basis_ids[in_support]
    emitter_ids = emitter_ids[in_support]
    basis_values = basis_values[in_support]

    directions = cell.fixed_directions
    camera_denominator = directions @ cell.camera.normal
    right_gradient = (
        cell.camera.right
        - (directions @ cell.camera.right)[:, None]
        * cell.camera.normal
        / camera_denominator[:, None]
    )
    up_gradient = (
        cell.camera.up
        - (directions @ cell.camera.up)[:, None]
        * cell.camera.normal
        / camera_denominator[:, None]
    )
    rows, columns = cell.camera.resolution
    sensor_gradient = torch.stack(
        (
            columns / cell.camera.width * right_gradient,
            -rows / cell.camera.height * up_gradient,
        ),
        dim=1,
    )
    normal_motion = torch.einsum(
        "qad,qd->qa", sensor_gradient, reference_normals[cell.emitter_ids]
    )
    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], directions
    )
    fractional = sensor_xy - cell.footprint_base_xy.to(sensor_xy.dtype)
    dx, dy = fractional.unbind(dim=-1)
    weight_gradient = torch.stack(
        (
            torch.stack((-(1.0 - dy), -(1.0 - dx)), dim=-1),
            torch.stack((1.0 - dy, -dx), dim=-1),
            torch.stack((-dy, 1.0 - dx), dim=-1),
            torch.stack((dy, dx), dim=-1),
        ),
        dim=1,
    )
    ds = -basis_values / denominator[emitter_ids]
    dweights = torch.einsum(
        "pfa,pa->pf",
        weight_gradient[local_points],
        normal_motion[local_points] * ds[:, None],
    )
    values = (
        dweights[..., None]
        * cell.emitter_colors[emitter_ids, None, :]
    )
    channels = torch.arange(3, device=points.device)
    output_rows = (
        cell.footprint_pixels[local_points, :, None] * 3
        + channels[None, None, :]
    ).expand(-1, -1, 3)
    parameter_columns = basis_ids[:, None, None].expand_as(output_rows)
    touched = (
        cell.footprint_valid[local_points, :, None]
        & (values.abs() > threshold)
    )
    return torch.sparse_coo_tensor(
        torch.stack((output_rows[touched], parameter_columns[touched])),
        values[touched],
        size=(3 * cell.camera.pixel_count, layout.count),
        dtype=points.dtype,
        device=points.device,
    ).coalesce()


def _normal_matvec(matrices: list[Tensor], vector: Tensor, damping: float) -> Tensor:
    result = damping * vector
    for matrix in matrices:
        projected = torch.sparse.mm(matrix, vector[:, None])
        result = result + torch.sparse.mm(
            matrix.transpose(0, 1), projected
        ).flatten()
    return result


def _conjugate_gradient(
    matrices: list[Tensor], right: Tensor, config: KScalingConfig
) -> Tensor:
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    residual_squared = residual @ residual
    for _ in range(config.cg_iterations):
        product = _normal_matvec(matrices, direction, config.damping)
        alpha = residual_squared / (direction @ product).clamp_min(1e-30)
        solution = solution + alpha * direction
        next_residual = residual - alpha * product
        next_squared = next_residual @ next_residual
        if float(torch.sqrt(next_squared)) < 1e-10:
            break
        direction = next_residual + next_squared / residual_squared * direction
        residual = next_residual
        residual_squared = next_squared
    return solution


def _active_jacobians(
    context: _ScalingContext,
    field: LocalZeroSet,
    points: Tensor,
    denominator: Tensor,
) -> list[Tensor]:
    return [
        _local_sparse_jacobian(
            field.layout,
            points,
            context.reference_normals,
            denominator,
            field.support,
            cell,
        )
        for cell in context.cells
    ]


def _gradient(
    matrices: list[Tensor], images: list[Tensor], targets: list[Tensor], k: int
) -> Tensor:
    gradient = torch.zeros(k, dtype=torch.float64, device=images[0].device)
    for matrix, image, target in zip(matrices, images, targets):
        gradient += torch.sparse.mm(
            matrix.transpose(0, 1), (image - target)[:, None]
        ).flatten()
    return gradient


def _optimize_scaling_baseline(
    context: _ScalingContext,
    targets: list[Tensor],
    config: KScalingConfig,
    *,
    initial: Tensor | None = None,
    iterations: int | None = None,
    line_evaluations: int = 8,
) -> _ScalingOptimization:
    coefficients = (
        torch.zeros(context.k, dtype=torch.float64, device=context.reference_points.device)
        if initial is None
        else initial.clone()
    )
    field = LocalZeroSet(
        context.base,
        context.current_layout,
        coefficients,
        context.reference_points,
        context.reference_normals,
        context.current_support,
    )
    points, success, _, denominator = field.deform()
    images = _images(context.cells, points)
    initial_loss = _image_loss(images, targets)
    loss = initial_loss
    last_relative = 0.0
    gradient_norm = math.inf
    completed = 0
    for iteration in range(iterations or config.baseline_iterations):
        field = field.with_coefficients(coefficients)
        points, success, _, denominator = field.deform()
        images = _images(context.cells, points)
        loss = _image_loss(images, targets)
        matrices = _active_jacobians(context, field, points, denominator)
        gradient = _gradient(matrices, images, targets, context.k)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        step = _conjugate_gradient(matrices, -gradient, config)
        accepted_loss = loss
        accepted = coefficients
        for trial in range(line_evaluations):
            scale = 0.5**trial
            proposal = (coefficients + scale * step).clamp(
                -config.coefficient_limit, config.coefficient_limit
            )
            proposal_field = field.with_coefficients(proposal)
            proposal_points, proposal_success, _, _ = proposal_field.deform()
            if not bool(proposal_success.all()):
                continue
            proposal_images = _images(context.cells, proposal_points)
            proposal_loss = _image_loss(proposal_images, targets)
            if proposal_loss < accepted_loss:
                accepted_loss = proposal_loss
                accepted = proposal
        completed = iteration + 1
        last_relative = (loss - accepted_loss) / max(loss, 1e-30)
        coefficients = accepted
        if accepted_loss >= loss or last_relative < 1e-7 or gradient_norm < 1e-10:
            break
    field = field.with_coefficients(coefficients)
    points, success, _, denominator = field.deform()
    images = _images(context.cells, points)
    loss = _image_loss(images, targets)
    return _ScalingOptimization(
        coefficients,
        points,
        denominator,
        images,
        loss,
        initial_loss,
        completed,
        last_relative,
        gradient_norm,
        int((~success).sum()),
    )


def _sparse_column_statistics(
    matrix: Tensor,
    residual: Tensor,
    parameter_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    columns = indices[1]
    alignment = torch.sparse.mm(
        matrix.transpose(0, 1), residual[:, None]
    ).flatten()
    norm_squared = torch.zeros(
        parameter_count, dtype=values.dtype, device=values.device
    )
    norm_squared.scatter_add_(0, columns, values * values)
    pixels = torch.div(indices[0], 3, rounding_mode="floor")
    pixel_parameter = torch.unique(pixels * parameter_count + columns)
    unique_pixels = torch.div(pixel_parameter, parameter_count, rounding_mode="floor")
    unique_columns = pixel_parameter % parameter_count
    affected = torch.bincount(unique_columns, minlength=parameter_count)
    local_residual = torch.zeros_like(norm_squared)
    pixel_rgb = (
        unique_pixels[:, None] * 3
        + torch.arange(3, device=values.device)[None, :]
    )
    local_residual.scatter_add_(
        0, unique_columns, (residual[pixel_rgb] ** 2).sum(dim=1)
    )
    responsive = torch.bincount(columns, minlength=parameter_count) > 0
    return alignment, norm_squared, affected, local_residual, responsive


def _candidate_scores(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
    targets: list[Tensor],
    config: KScalingConfig,
) -> tuple[dict[str, Tensor], dict[str, object], list[Tensor]]:
    count = context.candidate_layout.count
    field = LocalZeroSet(
        context.base,
        context.current_layout,
        baseline.coefficients,
        context.reference_points,
        context.reference_normals,
        context.current_support,
    )
    matrices: list[Tensor] = []
    alignment = torch.zeros(count, dtype=torch.float64, device=baseline.points.device)
    norm_squared = torch.zeros_like(alignment)
    affected_pixels = torch.zeros(count, dtype=torch.long, device=baseline.points.device)
    local_residual_squared = torch.zeros_like(alignment)
    affected_views = torch.zeros(count, dtype=torch.long, device=baseline.points.device)
    jacobian_nnz = 0
    for cell, image, target in zip(context.cells, baseline.images, targets):
        matrix = _local_sparse_jacobian(
            context.candidate_layout,
            baseline.points,
            context.reference_normals,
            baseline.denominator,
            context.candidate_support,
            cell,
        )
        matrices.append(matrix)
        local = _sparse_column_statistics(matrix, image - target, count)
        alignment += local[0]
        norm_squared += local[1]
        affected_pixels += local[2]
        local_residual_squared += local[3]
        affected_views += local[4].to(torch.long)
        jacobian_nnz += matrix._nnz()
    generator = torch.Generator().manual_seed(1701)
    scores = {
        "random": torch.rand(count, generator=generator).to(baseline.points.device),
        "visibility": affected_views.to(torch.float64),
        "local_residual": torch.sqrt(local_residual_squared),
        "jacobian_norm": torch.sqrt(norm_squared),
        "raw_alignment": alignment.abs(),
        "quadratic_gain": alignment**2 / (norm_squared + config.damping),
    }
    responsive = scores["jacobian_norm"] > 1e-12
    responsive_pixels = affected_pixels[responsive]
    diagnostics: dict[str, object] = {
        "affected_views": affected_views,
        "affected_pixels": affected_pixels,
        "responsive": responsive,
        "responsive_fraction": float(responsive.double().mean()),
        "median_affected_views": float(affected_views[responsive].double().median())
        if bool(responsive.any())
        else 0.0,
        "median_affected_pixels": float(responsive_pixels.double().median())
        if responsive_pixels.numel()
        else 0.0,
        "zero_jacobian_fraction": float((~responsive).double().mean()),
        "jacobian_nnz": jacobian_nnz,
    }
    # Touch the current field explicitly so the denominator provenance is clear.
    diagnostics["current_parameter_count"] = field.parameter_count
    return scores, diagnostics, matrices


def _cell_weights(cell: FixedTransportCell, points: Tensor) -> Tensor:
    sensor_xy, _ = _continuous_sensor(
        cell.camera, points[cell.emitter_ids], cell.fixed_directions
    )
    fractional = sensor_xy - cell.footprint_base_xy.to(sensor_xy.dtype)
    dx, dy = fractional.unbind(dim=-1)
    return torch.stack(
        ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy),
        dim=-1,
    ) * cell.footprint_valid


def _candidate_deformation(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
    candidate: int,
    coefficient: float,
) -> tuple[Tensor, Tensor, bool]:
    affected = torch.unique(context.candidate_support.points_for_basis(candidate))
    if affected.numel() == 0:
        return affected, baseline.points.new_empty((0, 3)), True
    local_reference = context.reference_points[affected]
    local_normals = context.reference_normals[affected]
    local_point_ids, active_basis = context.current_support.subset_points(affected)
    displacement = (
        (baseline.points[affected] - local_reference) * local_normals
    ).sum(dim=-1)
    center = context.candidate_layout.centers[candidate]
    radius = context.candidate_layout.radii[candidate]
    for _ in range(10):
        points = local_reference + displacement[:, None] * local_normals
        values = context.base.value(points)
        gradients = context.base.gradient(points)
        active_offsets = points[local_point_ids] - context.current_layout.centers[active_basis]
        active_radii = context.current_layout.radii[active_basis]
        active_values = wendland_values(active_offsets, active_radii)
        active_gradients = wendland_gradients(active_offsets, active_radii)
        values.scatter_add_(
            0,
            local_point_ids,
            active_values * baseline.coefficients[active_basis],
        )
        gradients.scatter_add_(
            0,
            local_point_ids[:, None].expand(-1, 3),
            active_gradients * baseline.coefficients[active_basis, None],
        )
        candidate_offsets = points - center
        candidate_radii = radius.expand(points.shape[0])
        values = values + coefficient * wendland_values(
            candidate_offsets, candidate_radii
        )
        gradients = gradients + coefficient * wendland_gradients(
            candidate_offsets, candidate_radii
        )
        denominator = (gradients * local_normals).sum(dim=-1)
        safe = denominator.abs() > 1e-8
        displacement = (
            displacement
            - torch.where(safe, values / denominator, torch.zeros_like(values))
        ).clamp(-0.015, 0.015)
    points = local_reference + displacement[:, None] * local_normals
    return affected, points, bool(torch.isfinite(points).all())


def _candidate_image_deltas(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
    candidate: int,
    coefficient: float,
    baseline_weights: list[Tensor],
) -> tuple[list[tuple[Tensor, Tensor]], bool]:
    affected, changed_points, success = _candidate_deformation(
        context, baseline, candidate, coefficient
    )
    if affected.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=baseline.points.device)
        return [(empty, baseline.points.new_empty(0)) for _ in context.cells], success
    deltas: list[tuple[Tensor, Tensor]] = []
    for cell, old_weights in zip(context.cells, baseline_weights):
        positions = torch.searchsorted(affected, cell.emitter_ids)
        safe_positions = positions.clamp_max(affected.numel() - 1)
        selected = affected[safe_positions] == cell.emitter_ids
        photon_rows = torch.nonzero(selected, as_tuple=False).flatten()
        if photon_rows.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=baseline.points.device)
            deltas.append((empty, baseline.points.new_empty(0)))
            continue
        local_ids = safe_positions[photon_rows]
        sensor_xy, _ = _continuous_sensor(
            cell.camera,
            changed_points[local_ids],
            cell.fixed_directions[photon_rows],
        )
        fractional = sensor_xy - cell.footprint_base_xy[photon_rows].to(sensor_xy.dtype)
        dx, dy = fractional.unbind(dim=-1)
        new_weights = torch.stack(
            ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy),
            dim=-1,
        ) * cell.footprint_valid[photon_rows]
        weight_delta = new_weights - old_weights[photon_rows]
        values = (
            weight_delta[..., None]
            * cell.emitter_colors[cell.emitter_ids[photon_rows], None, :]
        )
        rows = (
            cell.footprint_pixels[photon_rows, :, None] * 3
            + torch.arange(3, device=baseline.points.device)[None, None, :]
        ).expand_as(values)
        keep = values != 0.0
        sparse = torch.sparse_coo_tensor(
            rows[keep][None, :],
            values[keep],
            size=(3 * cell.camera.pixel_count,),
            dtype=values.dtype,
            device=values.device,
        ).coalesce()
        deltas.append((sparse.indices()[0], sparse.values()))
    return deltas, success


def _candidate_loss(
    baseline_loss: float,
    residuals: list[Tensor],
    deltas: list[tuple[Tensor, Tensor]],
) -> float:
    change = 0.0
    for residual, (rows, values) in zip(residuals, deltas):
        change += float((residual[rows] * values).sum() + 0.5 * (values * values).sum())
    return baseline_loss + change


def _new_only_oracle(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
    targets: list[Tensor],
    candidate_ids: Tensor,
    config: KScalingConfig,
) -> tuple[Tensor, Tensor, int]:
    gains = torch.full(
        (context.candidate_layout.count,),
        torch.nan,
        dtype=torch.float64,
        device=baseline.points.device,
    )
    best_coefficients = torch.zeros_like(gains)
    residuals = [image - target for image, target in zip(baseline.images, targets)]
    baseline_weights = [
        _cell_weights(cell, baseline.points) for cell in context.cells
    ]
    failures = 0
    samples = torch.linspace(
        -config.coefficient_limit,
        config.coefficient_limit,
        config.oracle_evaluations,
        dtype=torch.float64,
    ).tolist()
    for candidate in candidate_ids.tolist():
        best_loss = baseline.loss
        best_coefficient = 0.0
        for coefficient in samples:
            deltas, success = _candidate_image_deltas(
                context,
                baseline,
                candidate,
                coefficient,
                baseline_weights,
            )
            if not success:
                failures += 1
                continue
            loss = _candidate_loss(baseline.loss, residuals, deltas)
            if loss < best_loss:
                best_loss = loss
                best_coefficient = coefficient
        gains[candidate] = max(0.0, baseline.loss - best_loss)
        best_coefficients[candidate] = best_coefficient
    return gains, best_coefficients, failures


def _oracle_candidate_ids(
    context: _ScalingContext,
    scores: dict[str, Tensor],
    config: KScalingConfig,
) -> tuple[Tensor, Tensor, str]:
    count = context.candidate_layout.count
    device = context.reference_points.device
    if count <= config.exhaustive_candidate_limit:
        ids = torch.arange(count, device=device)
        return ids, ids, "exhaustive"
    generator = torch.Generator().manual_seed(8123 + context.k)
    unbiased = torch.randperm(count, generator=generator)[: min(
        config.unbiased_oracle_size, count
    )].to(device)
    selected = [unbiased]
    for name in ("random", "local_residual", "jacobian_norm", "raw_alignment", "quadratic_gain"):
        selected.append(torch.argsort(scores[name], descending=True, stable=True)[:32])
    selected.extend(
        (
            torch.tensor([context.redundant_id, context.invisible_id], device=device),
        )
    )
    ids = torch.unique(torch.cat(selected))
    return ids, unbiased, "sampled"


def _correlations_on(
    scores: dict[str, Tensor], gains: Tensor, ids: Tensor
) -> dict[str, dict[str, float]]:
    valid = ids[torch.isfinite(gains[ids])]
    return {
        name: {
            "spearman": _correlation(scores[name][valid], gains[valid], ranked=True),
            "pearson": _correlation(scores[name][valid], gains[valid], ranked=False),
        }
        for name in ("random", "visibility", "local_residual", "jacobian_norm", "raw_alignment", "quadratic_gain")
    }


def _top_k_sampled(
    scores: dict[str, Tensor], gains: Tensor
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    finite_gains = gains[torch.isfinite(gains)]
    sampled_best = float(finite_gains.max()) if finite_gains.numel() else 0.0
    for name in ("random", "local_residual", "jacobian_norm", "raw_alignment", "quadratic_gain"):
        order = torch.argsort(scores[name], descending=True, stable=True)
        summary: dict[str, object] = {}
        for count in (1, 3, 5, 10, 32):
            ids = order[:count]
            selected = gains[ids]
            selected = selected[torch.isfinite(selected)]
            summary[f"top_{count}"] = {
                "candidate_ids": ids.tolist(),
                "mean_actual_gain": float(selected.mean()) if selected.numel() else math.nan,
                "best_actual_gain": float(selected.max()) if selected.numel() else math.nan,
                "sample_relative_regret": (
                    sampled_best - float(selected.max()) if selected.numel() else math.nan
                ),
            }
        result[name] = summary
    return result


def _transport_event_fraction(
    context: _ScalingContext, points: Tensor
) -> float:
    if bool(context.state.surface_hits.any()):
        raise RuntimeError("sphere scaling path unexpectedly contains absorption")
    emitter_ids = context.state.photons.emitter_ids
    photons = PhotonBatch(
        points[emitter_ids],
        context.state.directions,
        context.state.photons.colors,
        context.state.photons.energies,
        context.state.photons.emit_times,
        emitter_ids,
    )
    state = SceneTransportState(
        points,
        context.reference_normals,
        photons,
        context.state.directions,
        context.state.surface_hits,
        context.state.surface_times,
        context.state.emitter_root_success,
    )
    changed = 0
    for camera, baseline_cell in zip(context.cameras, context.cells):
        perturbed = project_camera(
            context.geometry,
            camera,
            state,
            MultiviewConfig(
                resolution=camera.resolution,
                emitters=points.shape[0],
                packets_per_emitter=context.packets_per_emitter,
                parameter_count=context.k,
                cone_power=128.0,
            ),
        ).cell
        changed += int((perturbed.owner_map != baseline_cell.owner_map).sum())
    return changed / (len(context.cells) * context.cells[0].camera.pixel_count)


def _candidate_derivative_validation_scaling(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
    matrices: list[Tensor],
    scores: dict[str, Tensor],
    config: KScalingConfig,
) -> dict[str, object]:
    responsive = torch.nonzero(
        scores["jacobian_norm"] > 1e-12, as_tuple=False
    ).flatten()
    order = torch.argsort(scores["raw_alignment"], descending=True, stable=True)
    ids = [int(order[0])]
    if responsive.numel():
        responsive_order = responsive[
            torch.argsort(scores["raw_alignment"][responsive], stable=True)
        ]
        ids.extend(
            [
                int(responsive_order[responsive_order.numel() // 2]),
                int(responsive_order[0]),
            ]
        )
    ids.append(context.invisible_id)
    ids = list(dict.fromkeys(ids))[:4]
    while len(ids) < 4:
        candidate = int(order[len(ids)])
        if candidate not in ids:
            ids.append(candidate)
    baseline_weights = [_cell_weights(cell, baseline.points) for cell in context.cells]
    errors: list[float] = []
    event_fractions: list[float] = []
    step = 1e-5
    for candidate in ids:
        plus, plus_success = _candidate_image_deltas(
            context, baseline, candidate, step, baseline_weights
        )
        minus, minus_success = _candidate_image_deltas(
            context, baseline, candidate, -step, baseline_weights
        )
        if not (plus_success and minus_success):
            errors.append(math.inf)
            event_fractions.append(1.0)
            continue
        numerator = torch.zeros((), dtype=torch.float64, device=baseline.points.device)
        denominator = torch.zeros_like(numerator)
        for matrix, plus_view, minus_view in zip(matrices, plus, minus):
            finite = torch.zeros(
                matrix.shape[0], dtype=torch.float64, device=baseline.points.device
            )
            finite.index_add_(0, plus_view[0], plus_view[1] / (2.0 * step))
            finite.index_add_(0, minus_view[0], -minus_view[1] / (2.0 * step))
            analytic = _column_dense(matrix, candidate)
            numerator += ((finite - analytic) ** 2).sum()
            denominator += (analytic * analytic).sum()
        errors.append(float(torch.sqrt(numerator / denominator.clamp_min(1e-30))))
        affected, plus_points, _ = _candidate_deformation(
            context, baseline, candidate, step
        )
        full_plus = baseline.points.clone()
        full_plus[affected] = plus_points
        affected, minus_points, _ = _candidate_deformation(
            context, baseline, candidate, -step
        )
        full_minus = baseline.points.clone()
        full_minus[affected] = minus_points
        event_fractions.append(
            max(
                _transport_event_fraction(context, full_plus),
                _transport_event_fraction(context, full_minus),
            )
        )
    finite_errors = [value for value in errors if math.isfinite(value)]
    return {
        "candidate_ids": ids,
        "step": step,
        "median_relative_error": statistics.median(finite_errors),
        "maximum_relative_error": max(finite_errors),
        "maximum_transport_event_fraction": max(event_fractions),
        "numerical_failures": len(errors) - len(finite_errors),
    }


def _active_locality_metrics(
    context: _ScalingContext,
    baseline: _ScalingOptimization,
) -> dict[str, object]:
    field = LocalZeroSet(
        context.base,
        context.current_layout,
        baseline.coefficients,
        context.reference_points,
        context.reference_normals,
        context.current_support,
    )
    pair_offsets = (
        baseline.points[context.current_support.point_ids]
        - context.current_layout.centers[context.current_support.basis_ids]
    )
    active_pairs = wendland_values(
        pair_offsets,
        context.current_layout.radii[context.current_support.basis_ids],
    ) > 0.0
    active_bases_per_query = float(active_pairs.sum() / baseline.points.shape[0])
    matrices = _active_jacobians(
        context, field, baseline.points, baseline.denominator
    )
    pixel_parameter_count = 0
    affected_observations = 0
    total_nnz = 0
    for matrix in matrices:
        indices = matrix.indices()
        pixels = torch.div(indices[0], 3, rounding_mode="floor")
        encoded = torch.unique(pixels * context.k + indices[1])
        pixel_parameter_count += encoded.numel()
        affected_observations += torch.unique(pixels).numel()
        total_nnz += matrix._nnz()
    return {
        "active_bases_per_query": active_bases_per_query,
        "active_parameters_per_affected_pixel": (
            pixel_parameter_count / max(affected_observations, 1)
        ),
        "affected_image_fraction_per_parameter": pixel_parameter_count
        / (
            context.k
            * len(context.cells)
            * context.cells[0].camera.pixel_count
        ),
        "jacobian_nnz": total_nnz,
        "jacobian_nnz_per_parameter": total_nnz / context.k,
        "jacobian_density": total_nnz
        / (
            3
            * len(context.cells)
            * context.cells[0].camera.pixel_count
            * context.k
        ),
    }


def _run_k_level(
    k: int,
    config: KScalingConfig,
    *,
    target_seed: int,
    secondary: bool,
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    level_started = time.perf_counter()
    context = _build_scaling_context(k, config, device)
    target_coefficients, target_points, target_rms = _target_coefficients(
        context, config, target_seed
    )
    targets = _images(context.cells, target_points)
    initial_images = _images(context.cells, context.reference_points)
    initial_loss = _image_loss(initial_images, targets)

    baseline_started = time.perf_counter()
    baseline = _optimize_scaling_baseline(context, targets, config)
    torch.cuda.synchronize()
    baseline_runtime = time.perf_counter() - baseline_started
    baseline_geometry_error = float(
        torch.sqrt(((baseline.points - target_points) ** 2).sum(dim=1).mean())
    )

    continuation_started = time.perf_counter()
    continuation = _optimize_scaling_baseline(
        context,
        targets,
        config,
        initial=baseline.coefficients,
        iterations=1,
        line_evaluations=config.oracle_evaluations,
    )
    torch.cuda.synchronize()
    continuation_runtime = time.perf_counter() - continuation_started
    continuation_gain = max(0.0, baseline.loss - continuation.loss)

    scoring_started = time.perf_counter()
    scores, diagnostics, candidate_matrices = _candidate_scores(
        context, baseline, targets, config
    )
    torch.cuda.synchronize()
    scoring_runtime = time.perf_counter() - scoring_started
    source = context.candidate_target_ids
    source_strength = torch.where(
        source >= 0,
        target_coefficients[source.clamp_min(0)].abs()
        * scores["jacobian_norm"],
        torch.zeros_like(scores["jacobian_norm"]),
    )
    useful_id = int(source_strength.argmax())
    context.candidate_labels[useful_id] = "known_missing_detail_control"

    derivative = _candidate_derivative_validation_scaling(
        context, baseline, candidate_matrices, scores, config
    )
    oracle_ids, unbiased_ids, oracle_mode = _oracle_candidate_ids(
        context, scores, config
    )
    oracle_ids = torch.unique(
        torch.cat(
            (
                oracle_ids,
                torch.tensor([useful_id], device=oracle_ids.device),
            )
        )
    )
    oracle_started = time.perf_counter()
    gains, best_coefficients, oracle_failures = _new_only_oracle(
        context, baseline, targets, oracle_ids, config
    )
    torch.cuda.synchronize()
    oracle_runtime = time.perf_counter() - oracle_started
    correlations = _correlations_on(scores, gains, unbiased_ids)
    top_k = _top_k_sampled(scores, gains)
    locality = _active_locality_metrics(context, baseline)
    target_event_fraction = _transport_event_fraction(context, target_points)
    torch.cuda.synchronize()
    total_runtime = time.perf_counter() - level_started

    def control(candidate: int) -> dict[str, object]:
        return {
            "candidate_id": candidate,
            "jacobian_norm": float(scores["jacobian_norm"][candidate]),
            "raw_alignment": float(scores["raw_alignment"][candidate]),
            "actual_gain_new_only": float(gains[candidate])
            if torch.isfinite(gains[candidate])
            else None,
            "best_coefficient": float(best_coefficients[candidate])
            if torch.isfinite(gains[candidate])
            else None,
            "affected_views": int(diagnostics["affected_views"][candidate]),
            "affected_pixels": int(diagnostics["affected_pixels"][candidate]),
        }

    row: dict[str, object] = {
        "target_seed": target_seed,
        "secondary_target": secondary,
        "k": k,
        "k_gt": 2 * k,
        "candidate_count": 2 * k,
        "emitters": config.emitters_per_basis * k,
        "packets_per_emitter": context.packets_per_emitter,
        "photons": context.state.photons.count,
        "support_radius": float(context.current_layout.radii[0]),
        "target_support_radius": float(context.target_layout.radii[0]),
        "emitters_per_basis": config.emitters_per_basis,
        "photons_per_basis": context.state.photons.count / k,
        "target_geometry_rms_displacement": target_rms,
        "initial_loss": initial_loss,
        "baseline_final_loss": baseline.loss,
        "baseline_relative_reduction": (initial_loss - baseline.loss)
        / max(initial_loss, 1e-30),
        "baseline_iterations": baseline.iterations,
        "baseline_last_step_relative_improvement": baseline.last_step_relative_improvement,
        "baseline_gradient_norm": baseline.gradient_norm,
        "initial_residual_norm": math.sqrt(2.0 * initial_loss),
        "residual_norm": math.sqrt(2.0 * baseline.loss),
        "geometry_rms_error": baseline_geometry_error,
        "no_birth_continuation_gain": continuation_gain,
        "no_birth_continuation_runtime_seconds": continuation_runtime,
        "responsive_candidate_fraction": diagnostics["responsive_fraction"],
        "responsive_candidates": int(diagnostics["responsive"].sum()),
        "median_affected_views_per_responsive_candidate": diagnostics[
            "median_affected_views"
        ],
        "median_affected_pixels_per_responsive_candidate": diagnostics[
            "median_affected_pixels"
        ],
        "zero_jacobian_candidate_fraction": diagnostics["zero_jacobian_fraction"],
        "correlation_sample_size": int(unbiased_ids.numel()),
        "oracle_evaluated_candidates": int(oracle_ids.numel()),
        "oracle_mode": oracle_mode,
        "regret_scope": (
            "global exhaustive" if oracle_mode == "exhaustive" else "sample-relative"
        ),
        "correlations": correlations,
        "top_k": top_k,
        "raw_top_1_actual_gain": top_k["raw_alignment"]["top_1"][
            "mean_actual_gain"
        ],
        "raw_top_5_mean_actual_gain": top_k["raw_alignment"]["top_5"][
            "mean_actual_gain"
        ],
        "finite_difference": derivative,
        "target_transport_event_fraction": target_event_fraction,
        **locality,
        "candidate_score_runtime_seconds": scoring_runtime,
        "baseline_optimization_runtime_seconds": baseline_runtime,
        "oracle_runtime_seconds": oracle_runtime,
        "total_runtime_seconds": total_runtime,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "numerical_failures": baseline.root_failures
        + derivative["numerical_failures"]
        + oracle_failures,
        "controls": {
            "redundant": control(context.redundant_id),
            "known_missing_detail": control(useful_id),
            "invisible": control(context.invisible_id),
        },
        "scene_surface_hits": int(context.state.surface_hits.sum()),
    }
    print(
        json.dumps(
            {
                "k": k,
                "secondary": secondary,
                "runtime_seconds": total_runtime,
                "peak_allocated_mib": row["peak_allocated_mib"],
                "peak_reserved_mib": row["peak_reserved_mib"],
                "candidate_score_seconds": scoring_runtime,
                "baseline_seconds": baseline_runtime,
                "responsive_fraction": row["responsive_candidate_fraction"],
                "transport_event_fraction": target_event_fraction,
                "numerical_failures": row["numerical_failures"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return row


def _power_law(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    x = torch.log(torch.tensor([row["k"] for row in rows], dtype=torch.float64))
    y = torch.log(
        torch.tensor([max(float(row[key]), 1e-30) for row in rows], dtype=torch.float64)
    )
    design = torch.stack((torch.ones_like(x), x), dim=1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution.flatten()
    predicted = design @ solution
    total = ((y - y.mean()) ** 2).sum()
    residual = ((y - predicted) ** 2).sum()
    return {
        "exponent": float(solution[1]),
        "coefficient": float(torch.exp(solution[0])),
        "r_squared": float(1.0 - residual / total) if total > 0 else 1.0,
    }


def _write_k_scaling_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = (
        "k",
        "k_gt",
        "candidate_count",
        "emitters",
        "packets_per_emitter",
        "photons",
        "emitters_per_basis",
        "photons_per_basis",
        "support_radius",
        "target_geometry_rms_displacement",
        "initial_loss",
        "baseline_final_loss",
        "baseline_relative_reduction",
        "baseline_iterations",
        "baseline_last_step_relative_improvement",
        "baseline_gradient_norm",
        "initial_residual_norm",
        "residual_norm",
        "geometry_rms_error",
        "no_birth_continuation_gain",
        "responsive_candidate_fraction",
        "median_affected_views_per_responsive_candidate",
        "median_affected_pixels_per_responsive_candidate",
        "zero_jacobian_candidate_fraction",
        "correlation_sample_size",
        "oracle_evaluated_candidates",
        "oracle_mode",
        "regret_scope",
        "raw_alignment_spearman",
        "raw_alignment_pearson",
        "quadratic_spearman",
        "quadratic_pearson",
        "local_residual_spearman",
        "raw_top_1_actual_gain",
        "raw_top_5_mean_actual_gain",
        "fd_median_error",
        "fd_maximum_error",
        "transport_event_fraction",
        "active_bases_per_query",
        "active_parameters_per_affected_pixel",
        "affected_image_fraction_per_parameter",
        "jacobian_nnz",
        "jacobian_nnz_per_parameter",
        "jacobian_density",
        "candidate_score_runtime_seconds",
        "baseline_optimization_runtime_seconds",
        "oracle_runtime_seconds",
        "total_runtime_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "numerical_failures",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.update(
                {
                    "raw_alignment_spearman": row["correlations"]["raw_alignment"]["spearman"],
                    "raw_alignment_pearson": row["correlations"]["raw_alignment"]["pearson"],
                    "quadratic_spearman": row["correlations"]["quadratic_gain"]["spearman"],
                    "quadratic_pearson": row["correlations"]["quadratic_gain"]["pearson"],
                    "local_residual_spearman": row["correlations"]["local_residual"]["spearman"],
                    "fd_median_error": row["finite_difference"]["median_relative_error"],
                    "fd_maximum_error": row["finite_difference"]["maximum_relative_error"],
                    "transport_event_fraction": row["target_transport_event_fraction"],
                }
            )
            writer.writerow({name: flat[name] for name in columns})


def _write_k_scaling_figures(
    directory: Path, rows: list[dict[str, object]]
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    k = [row["k"] for row in rows]
    definitions = (
        (
            "v03b_spearman_vs_k.png",
            "Pre-birth ranking across scale",
            "Spearman correlation",
            (
                ("Raw alignment", [row["correlations"]["raw_alignment"]["spearman"] for row in rows]),
                ("Quadratic", [row["correlations"]["quadratic_gain"]["spearman"] for row in rows]),
                ("Local residual", [row["correlations"]["local_residual"]["spearman"] for row in rows]),
            ),
            "semilogx",
        ),
        (
            "v03b_scoring_runtime_vs_k.png",
            "Streamed candidate-scoring cost",
            "seconds",
            (("Candidate scoring", [row["candidate_score_runtime_seconds"] for row in rows]),),
            "loglog",
        ),
        (
            "v03b_vram_vs_k.png",
            "Peak CUDA memory",
            "MiB",
            (
                ("Allocated", [row["peak_allocated_mib"] for row in rows]),
                ("Reserved", [row["peak_reserved_mib"] for row in rows]),
            ),
            "loglog",
        ),
        (
            "v03b_local_connectivity_vs_k.png",
            "Local geometry connectivity",
            "mean count",
            (
                ("Active bases/query", [row["active_bases_per_query"] for row in rows]),
                ("Active parameters/affected pixel", [row["active_parameters_per_affected_pixel"] for row in rows]),
            ),
            "semilogx",
        ),
        (
            "v03b_affected_fraction_vs_k.png",
            "Affected image fraction per basis",
            "fraction",
            (("Affected fraction", [row["affected_image_fraction_per_parameter"] for row in rows]),),
            "loglog",
        ),
    )
    paths: list[str] = []
    for filename, title, ylabel, series, scale in definitions:
        figure, axis = plt.subplots(figsize=(5.4, 3.8))
        for label, values in series:
            getattr(axis, scale)(k, values, "o-", label=label)
        axis.set_xlabel("active geometry parameters K")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if len(series) > 1:
            axis.legend()
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def run_k_scaling_experiment(
    config: KScalingConfig = KScalingConfig(),
    *,
    csv_path: Path | None = None,
    json_path: Path | None = None,
    figure_directory: Path | None = None,
) -> dict[str, object]:
    """Run the v0.3b sparse pre-birth evidence scaling experiment."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    primary: list[dict[str, object]] = []
    failed_attempt: dict[str, object] | None = None
    for k in config.k_values:
        if config.known_failed_k == k:
            failed_attempt = {
                "k": k,
                "reason": config.known_failure_reason
                or "known failure retained from the first attempted run",
                "reused_from_prior_single_attempt": True,
            }
            break
        try:
            row = _run_k_level(
                k, config, target_seed=1, secondary=False
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            failed_attempt = {"k": k, "reason": f"{type(error).__name__}: {error}"}
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            break
        primary.append(row)
        if float(row["peak_reserved_mib"]) > 0.8 * 15904.875:
            failed_attempt = {
                "k": 2 * k,
                "reason": "projected CUDA memory would exceed 80% policy",
            }
            break
    if not primary:
        raise RuntimeError(f"no K level completed: {failed_attempt}")
    maximum_k = int(primary[-1]["k"])
    sentinel = sorted(set((32, 256, 1024, maximum_k)).intersection(config.k_values))
    secondary = [
        _run_k_level(k, config, target_seed=19, secondary=True)
        for k in sentinel
        if k <= maximum_k
    ]
    fits = {
        "candidate_scoring": _power_law(primary, "candidate_score_runtime_seconds"),
        "peak_allocated_vram": _power_law(primary, "peak_allocated_mib"),
        "affected_fraction": _power_law(
            primary, "affected_image_fraction_per_parameter"
        ),
    }
    raw_correlations = [
        float(row["correlations"]["raw_alignment"]["spearman"])
        for row in primary
    ]
    random_correlations = [
        float(row["correlations"]["random"]["spearman"])
        for row in primary
    ]
    primary_supported = (
        statistics.median(raw_correlations) > 0.5
        and raw_correlations[-1] > 0.5
        and statistics.median(raw_correlations)
        > statistics.median(random_correlations)
    )
    local_counts = [float(row["active_bases_per_query"]) for row in primary]
    pixel_counts = [
        float(row["active_parameters_per_affected_pixel"]) for row in primary
    ]
    connectivity_supported = (
        max(local_counts) / max(min(local_counts), 1e-15) < 3.0
        and max(pixel_counts) / max(min(pixel_counts), 1e-15) < 3.0
    )
    report: dict[str, object] = {
        "experiment": "v0.3b aggressive K-scaling of pre-birth geometry evidence",
        "environment": cuda_environment(),
        "configuration": {
            "k_ladder_requested": list(config.k_values),
            "views": config.views,
            "resolution": [config.resolution, config.resolution],
            "emitters_per_basis": config.emitters_per_basis,
            "default_packets_per_emitter": config.packets_per_emitter,
            "maximum_photons": config.maximum_photons,
            "support_radius_rule": "0.60 * sqrt(32 / K)",
            "hierarchy": "nested unscrambled Sobol surface prefixes",
            "target": "K_GT=2K, coefficient-normalized to fixed geometry RMS",
            "candidate_pool": "K missing target centers + K finer distractors; coefficients remain nonexistent during scoring",
            "candidate_scoring": "sparse local contributions accumulated directly; no dense observation Jacobian",
            "oracle": f"nonlinear {config.oracle_evaluations}-sample one-dimensional line search",
        },
        "primary_levels": primary,
        "secondary_target_levels": secondary,
        "scaling_fits": fits,
        "maximum_validated_k": maximum_k,
        "maximum_attempted_k": (
            int(failed_attempt["k"]) if failed_attempt is not None else maximum_k
        ),
        "scaling_stop": failed_attempt or {
            "reason": "completed natural power-of-two endpoint K=32768"
        },
        "primary_verdict": (
            "PREBIRTH_SIGNAL_K_SCALING_SUPPORTED"
            if primary_supported
            else "PREBIRTH_SIGNAL_K_SCALING_NOT_SUPPORTED"
        ),
        "local_connectivity_verdict": (
            "LOCAL_CONNECTIVITY_SCALING_SUPPORTED"
            if connectivity_supported
            else "LOCAL_CONNECTIVITY_SCALING_NOT_SUPPORTED"
        ),
        "verdict_basis": {
            "raw_spearman_median": statistics.median(raw_correlations),
            "raw_spearman_largest_k": raw_correlations[-1],
            "random_spearman_median": statistics.median(random_correlations),
            "local_connectivity_max_to_min_ratio": max(local_counts)
            / max(min(local_counts), 1e-15),
            "pixel_connectivity_max_to_min_ratio": max(pixel_counts)
            / max(min(pixel_counts), 1e-15),
            "affected_fraction_exponent_is_diagnostic_not_connectivity_gate": True,
        },
    }
    if csv_path is not None:
        _write_k_scaling_csv(csv_path, primary)
    if figure_directory is not None:
        report["figures"] = _write_k_scaling_figures(figure_directory, primary)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = {
            "csv": str(csv_path) if csv_path is not None else None,
            "json": str(json_path),
            "figures": report.get("figures", []),
        }
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
