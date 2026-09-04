"""Falsification experiment for utility prediction of nonexistent geometry DoFs."""

from __future__ import annotations

import csv
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
    build_fixed_transport_cell,
    make_local_basis_field,
    render_fixed_transport_cell,
    sparse_geometry_image_jacobian,
    topology_event_pixels,
)


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
