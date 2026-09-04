"""True simultaneous geometry-DoF birth and Bunny headroom experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .benchmark import BenchmarkGeometry, cuda_environment
from .birth import _image_loss, _images, _local_sparse_jacobian
from .bunny import (
    BunnyExperimentConfig,
    BunnyGeometryEvaluator,
    _build_bunny_context,
    _normal_line_roots,
)
from .fields import LocalBasisField, unit_normals
from .jacobian import _photon_batch
from .locality import hierarchical_surface_points
from .mesh_field import PreparedBunny, prepare_stanford_bunny, resmooth_stanford_bunny
from .multiview import MultiviewConfig, SceneTransportState, project_camera
from .sequential import (
    ActiveState,
    RepeatedBirthConfig,
    SequentialContext,
    _candidate_scores,
    _checkpoint_row,
    _correlation,
    _evaluate_active,
    _hierarchical_layout,
    _multiscale_support,
    _normalized_auc,
    _optimize,
    _run_fixed,
)
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor
REPORT_LEVELS = (32, 64, 96, 128, 160, 192, 224, 256)


@dataclass(frozen=True)
class BatchPolicy:
    name: str
    nominal_size: int | None = None
    dynamic: bool = False
    uniform_schedule: tuple[int, ...] = ()
    alpha: float = 0.10
    tau: float = 0.10
    rho_max: float = 0.15
    maximum_size: int = 32


@dataclass
class CandidateColumns:
    matrices: list[Tensor]
    alignment: Tensor
    seconds: float


def _image_digest(images: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _candidate_columns(
    context: SequentialContext,
    state: ActiveState,
    targets: list[Tensor],
) -> CandidateColumns:
    """Rebuild exact sparse candidate columns retained only for one selection."""
    started = time.perf_counter()
    alignment = torch.zeros(
        context.config.master_count,
        dtype=torch.float64,
        device=state.points.device,
    )
    matrices: list[Tensor] = []
    for cell, image, target in zip(context.cells, state.images, targets):
        matrix = _local_sparse_jacobian(
            context.master_layout,
            state.points,
            context.reference_normals,
            state.denominator,
            context.master_support,
            cell,
        ).coalesce()
        indices = matrix.indices()
        alignment.scatter_add_(
            0,
            indices[1],
            matrix.values() * (image - target)[indices[0]],
        )
        matrices.append(matrix)
    if state.points.is_cuda:
        torch.cuda.synchronize()
    return CandidateColumns(matrices, alignment, time.perf_counter() - started)


def _dot_with_all(columns: CandidateColumns, candidate: int, count: int) -> Tensor:
    result = torch.zeros(
        count,
        dtype=columns.alignment.dtype,
        device=columns.alignment.device,
    )
    for matrix in columns.matrices:
        indices = matrix.indices()
        values = matrix.values()
        selected = indices[1] == candidate
        probe = torch.zeros(
            matrix.shape[0], dtype=values.dtype, device=values.device
        )
        probe.scatter_add_(0, indices[0, selected], values[selected])
        result.scatter_add_(0, indices[1], values * probe[indices[0]])
    return result


def _coupling_statistics(gram: Tensor) -> dict[str, float]:
    if gram.shape[0] < 2:
        return {"rho_off": 0.0, "pairwise_mean": 0.0, "pairwise_max": 0.0}
    diagonal = torch.diagonal(gram)
    off = gram - torch.diag(diagonal)
    rho = float(
        torch.linalg.vector_norm(off)
        / torch.linalg.vector_norm(diagonal).clamp_min(1e-30)
    )
    norm = torch.sqrt(diagonal.clamp_min(0.0))
    cosine = (gram.abs() / (norm[:, None] * norm[None, :]).clamp_min(1e-30))
    upper = cosine[torch.triu_indices(gram.shape[0], gram.shape[0], offset=1).unbind()]
    return {
        "rho_off": rho,
        "pairwise_mean": float(upper.mean()),
        "pairwise_max": float(upper.max()),
    }


def _select_compatible_batch(
    context: SequentialContext,
    state: ActiveState,
    scores: dict[str, Tensor],
    columns: CandidateColumns | None,
    policy: BatchPolicy,
    round_index: int,
) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    remaining_budget = context.config.budget - int(state.active_ids.numel())
    inactive = torch.ones(
        context.config.master_count,
        dtype=torch.bool,
        device=state.active_ids.device,
    )
    inactive[state.active_ids] = False
    if policy.uniform_schedule:
        nominal = min(policy.uniform_schedule[round_index], remaining_budget)
        selected = torch.nonzero(inactive, as_tuple=False).flatten()[:nominal]
        assert columns is not None
        dot_vectors = [
            _dot_with_all(columns, int(candidate), context.config.master_count)
            for candidate in selected
        ]
        gram = torch.empty(
            (selected.numel(), selected.numel()),
            dtype=torch.float64,
            device=state.points.device,
        )
        for row, candidate in enumerate(selected):
            gram[row, row] = scores["jacobian_norm"][candidate].square()
            for column in range(row + 1, selected.numel()):
                gram[row, column] = dot_vectors[row][selected[column]]
                gram[column, row] = gram[row, column]
        coupling = _coupling_statistics(gram)
        return selected, gram, {
            "nominal_batch_size": nominal,
            "actual_batch_size": int(selected.numel()),
            **coupling,
        }

    nominal = min(
        policy.maximum_size if policy.dynamic else int(policy.nominal_size or 1),
        remaining_budget,
    )
    values = torch.where(
        inactive,
        scores["quadratic"],
        torch.full_like(scores["quadratic"], -math.inf),
    )
    order = torch.argsort(values, descending=True, stable=True)
    best_score = float(values[order[0]])
    selected_ids: list[int] = []
    dot_vectors: list[Tensor] = []
    for candidate_tensor in order:
        candidate = int(candidate_tensor)
        score = float(values[candidate])
        if not math.isfinite(score):
            break
        if policy.dynamic and selected_ids and score < policy.alpha * best_score:
            break
        if selected_ids:
            assert columns is not None
            norm = float(scores["jacobian_norm"][candidate])
            correlations = torch.stack(
                [
                    vector[candidate].abs()
                    / (
                        scores["jacobian_norm"][previous] * norm
                    ).clamp_min(1e-30)
                    for previous, vector in zip(selected_ids, dot_vectors)
                ]
            )
            if float(correlations.max()) > policy.tau:
                continue
            if policy.dynamic:
                ids = selected_ids + [candidate]
                trial = torch.empty(
                    (len(ids), len(ids)),
                    dtype=torch.float64,
                    device=state.points.device,
                )
                for row, item in enumerate(ids):
                    trial[row, row] = scores["jacobian_norm"][item].square()
                for row in range(len(selected_ids)):
                    for column in range(row + 1, len(selected_ids)):
                        trial[row, column] = dot_vectors[row][selected_ids[column]]
                        trial[column, row] = trial[row, column]
                    trial[row, -1] = dot_vectors[row][candidate]
                    trial[-1, row] = trial[row, -1]
                if _coupling_statistics(trial)["rho_off"] > policy.rho_max:
                    continue
        selected_ids.append(candidate)
        if columns is not None:
            dot_vectors.append(
                _dot_with_all(columns, candidate, context.config.master_count)
            )
        if len(selected_ids) >= nominal:
            break
    selected = torch.tensor(
        selected_ids, dtype=torch.long, device=state.active_ids.device
    )
    size = len(selected_ids)
    gram = torch.zeros(
        (size, size), dtype=torch.float64, device=state.points.device
    )
    for row, candidate in enumerate(selected_ids):
        gram[row, row] = scores["jacobian_norm"][candidate].square()
        for column in range(row + 1, size):
            gram[row, column] = dot_vectors[row][selected_ids[column]]
            gram[column, row] = gram[row, column]
    coupling = _coupling_statistics(gram)
    return selected, gram, {
        "nominal_batch_size": nominal,
        "actual_batch_size": size,
        **coupling,
    }


def _batch_prediction(
    context: SequentialContext,
    scores: dict[str, Tensor],
    columns: CandidateColumns | None,
    selected: Tensor,
    gram: Tensor,
) -> tuple[float, float, float]:
    independent = float(scores["quadratic"][selected].sum())
    if selected.numel() == 1 or columns is None:
        return independent, independent, 1.0
    gradient = columns.alignment[selected]
    hessian = gram + context.config.damping * torch.eye(
        selected.numel(), dtype=gram.dtype, device=gram.device
    )
    try:
        joint = float(gradient @ torch.linalg.solve(hessian, gradient))
    except RuntimeError:
        joint = float(gradient @ torch.linalg.pinv(hessian) @ gradient)
    return joint, independent, joint / max(independent, 1e-30)


def _target_membership(target: dict[str, object], selected: Tensor, key: str) -> float:
    target_ids = target[key]
    return float(torch.isin(selected, target_ids).double().mean())  # type: ignore[arg-type]


def _run_batch_trajectory(
    context: SequentialContext,
    target: dict[str, object],
    policy: BatchPolicy,
) -> dict[str, object]:
    device = context.reference_points.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    active_ids = torch.arange(context.config.initial_count, device=device)
    initial = _evaluate_active(
        context,
        active_ids,
        torch.zeros(context.config.initial_count, dtype=torch.float64, device=device),
        target["images"],  # type: ignore[arg-type]
    )
    state, optimization_time = _optimize(
        context,
        initial,
        target["images"],  # type: ignore[arg-type]
        context.config.initial_optimization_steps,
    )
    scoring_time = 0.0
    scores, diagnostics, seconds = _candidate_scores(
        context, state, target["images"]  # type: ignore[arg-type]
    )
    scoring_time += seconds
    initial_row = _checkpoint_row(
        context,
        target,
        policy.name,
        0,
        state,
        scoring_time,
        optimization_time,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
        None,
        diagnostics,
        0.0,
    )
    initial_row.update(
        {
            "birth_round": 0,
            "nominal_batch_size": 0,
            "actual_batch_size": 0,
            "rho_off": 0.0,
            "pairwise_correlation_mean": 0.0,
            "pairwise_correlation_max": 0.0,
            "predicted_batch_gain": None,
            "independent_sum_gain": None,
            "joint_to_independent_ratio": None,
            "birth_top25_fraction": None,
            "birth_top10_fraction": None,
        }
    )
    rows = [initial_row]
    batches: list[dict[str, object]] = []
    selected_ids: list[int] = []
    while int(state.active_ids.numel()) < context.config.budget:
        columns = None
        if policy.uniform_schedule or policy.dynamic or (policy.nominal_size or 1) > 1:
            columns = _candidate_columns(
                context, state, target["images"]  # type: ignore[arg-type]
            )
            scoring_time += columns.seconds
        selected, gram, coupling = _select_compatible_batch(
            context, state, scores, columns, policy, len(batches)
        )
        if selected.numel() == 0:
            raise RuntimeError(f"{policy.name} could not select a nonempty batch")
        predicted_joint, predicted_sum, prediction_ratio = _batch_prediction(
            context, scores, columns, selected, gram
        )
        selected_scores = scores["quadratic"][selected]
        predicted_raw = float(scores["raw"][selected].sum())
        before_loss = state.loss
        born_ids = torch.cat((state.active_ids, selected))
        born_coefficients = torch.cat(
            (state.coefficients, state.coefficients.new_zeros(selected.numel()))
        )
        at_birth = _evaluate_active(
            context,
            born_ids,
            born_coefficients,
            target["images"],  # type: ignore[arg-type]
        )
        at_birth.optimizer_steps = state.optimizer_steps
        at_birth.line_search_failures = state.line_search_failures
        at_birth.cg_failures = state.cg_failures
        jump = float(torch.linalg.vector_norm(at_birth.points - state.points))
        state, optimize_seconds = _optimize(
            context,
            at_birth,
            target["images"],  # type: ignore[arg-type]
            context.config.post_birth_steps,
        )
        optimization_time += optimize_seconds
        realized_gain = max(0.0, before_loss - state.loss)
        batch = {
            "birth_round": len(batches) + 1,
            "active_k_before": int(state.active_ids.numel() - selected.numel()),
            "active_k_after": int(state.active_ids.numel()),
            "candidate_ids": selected.detach().cpu().tolist(),
            "centers": context.master_layout.centers[selected].detach().cpu().tolist(),
            **coupling,
            "predicted_batch_gain": predicted_joint,
            "independent_sum_gain": predicted_sum,
            "joint_to_independent_ratio": prediction_ratio,
            "selected_score_max": float(selected_scores.max()),
            "selected_score_median": float(selected_scores.median()),
            "selected_score_min": float(selected_scores.min()),
            "selected_score_min_to_max": float(
                selected_scores.min() / selected_scores.max().clamp_min(1e-30)
            ),
            "realized_gain": realized_gain,
            "birth_geometry_jump": jump,
            "top25_fraction": _target_membership(target, selected, "detail_ids"),
            "top10_fraction": _target_membership(
                target, selected, "detail_ids_top_ten"
            ),
        }
        batches.append(batch)
        selected_ids.extend(int(item) for item in selected.detach().cpu())
        del columns
        scores, diagnostics, seconds = _candidate_scores(
            context, state, target["images"]  # type: ignore[arg-type]
        )
        scoring_time += seconds
        elapsed = time.perf_counter() - started
        pseudo_birth = {
            "predicted_quadratic": predicted_joint,
            "predicted_raw": predicted_raw,
            "realized_gain": realized_gain,
            "new_only_gain": None,
        }
        row = _checkpoint_row(
            context,
            target,
            policy.name,
            0,
            state,
            scoring_time,
            optimization_time,
            elapsed,
            torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20,
            pseudo_birth,
            diagnostics,
            0.0,
        )
        row.update(
            {
                "birth_round": batch["birth_round"],
                "nominal_batch_size": batch["nominal_batch_size"],
                "actual_batch_size": batch["actual_batch_size"],
                "rho_off": batch["rho_off"],
                "pairwise_correlation_mean": batch["pairwise_mean"],
                "pairwise_correlation_max": batch["pairwise_max"],
                "predicted_batch_gain": predicted_joint,
                "independent_sum_gain": predicted_sum,
                "joint_to_independent_ratio": prediction_ratio,
                "selected_score_min_to_max": batch[
                    "selected_score_min_to_max"
                ],
                "birth_top25_fraction": batch["top25_fraction"],
                "birth_top10_fraction": batch["top10_fraction"],
            }
        )
        rows.append(row)
        if state.root_failures:
            break
    predicted = torch.tensor(
        [float(item["predicted_batch_gain"]) for item in batches],
        dtype=torch.float64,
    )
    predicted_sum = torch.tensor(
        [float(item["independent_sum_gain"]) for item in batches],
        dtype=torch.float64,
    )
    realized = torch.tensor(
        [float(item["realized_gain"]) for item in batches], dtype=torch.float64
    )
    sizes = [int(item["actual_batch_size"]) for item in batches]
    total = time.perf_counter() - started
    return {
        "method": policy.name,
        "rows": rows,
        "batches": batches,
        "selected_ids": selected_ids,
        "batch_schedule": sizes,
        "birth_rounds": len(batches),
        "mean_batch_size": statistics.mean(sizes),
        "median_batch_size": statistics.median(sizes),
        "maximum_batch_size": max(sizes),
        "minimum_batch_size": min(sizes),
        "cumulative_scoring_seconds": scoring_time,
        "cumulative_optimization_seconds": optimization_time,
        "render_evaluation_overhead_seconds": max(
            0.0, total - scoring_time - optimization_time
        ),
        "total_runtime_seconds": total,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "maximum_birth_geometry_jump": max(
            float(item["birth_geometry_jump"]) for item in batches
        ),
        "mean_rho_off": statistics.mean(float(item["rho_off"]) for item in batches),
        "maximum_rho_off": max(float(item["rho_off"]) for item in batches),
        "mean_pairwise_correlation": statistics.mean(
            float(item["pairwise_mean"]) for item in batches
        ),
        "maximum_pairwise_correlation": max(
            float(item["pairwise_max"]) for item in batches
        ),
        "predicted_joint_calibration": {
            "pearson": _correlation(predicted, realized, False),
            "spearman": _correlation(predicted, realized, True),
        },
        "predicted_sum_calibration": {
            "pearson": _correlation(predicted_sum, realized, False),
            "spearman": _correlation(predicted_sum, realized, True),
        },
        "chamfer_dof_auc": _normalized_auc(rows, "active_k", "symmetric_chamfer"),
        "p2s_p95_dof_auc": _normalized_auc(rows, "active_k", "point_to_surface_p95"),
        "image_dof_auc": _normalized_auc(rows, "active_k", "image_loss"),
        "duplicate_births": len(selected_ids) - len(set(selected_ids)),
        "top25_birth_fraction": statistics.mean(
            float(item["top25_fraction"])
            for item in batches
            for _ in range(int(item["actual_batch_size"]))
        ),
        "top10_birth_fraction": statistics.mean(
            float(item["top10_fraction"])
            for item in batches
            for _ in range(int(item["actual_batch_size"]))
        ),
    }


def _report_rows(trajectory: dict[str, object]) -> list[dict[str, object]]:
    rows = trajectory["rows"]
    selected: list[dict[str, object]] = []
    for requested in REPORT_LEVELS:
        row = min(rows, key=lambda item: abs(int(item["active_k"]) - requested))
        copy = dict(row)
        copy["requested_k"] = requested
        selected.append(copy)
    return selected


def _trajectory_summary(trajectory: dict[str, object]) -> dict[str, object]:
    final = trajectory["rows"][-1]
    return {
        key: trajectory[key]
        for key in (
            "method",
            "birth_rounds",
            "mean_batch_size",
            "median_batch_size",
            "maximum_batch_size",
            "minimum_batch_size",
            "cumulative_scoring_seconds",
            "cumulative_optimization_seconds",
            "render_evaluation_overhead_seconds",
            "total_runtime_seconds",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "maximum_birth_geometry_jump",
            "mean_rho_off",
            "maximum_rho_off",
            "mean_pairwise_correlation",
            "maximum_pairwise_correlation",
            "predicted_joint_calibration",
            "predicted_sum_calibration",
            "chamfer_dof_auc",
            "p2s_p95_dof_auc",
            "image_dof_auc",
            "top25_birth_fraction",
            "top10_birth_fraction",
        )
    } | {
        "final": final,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase_a_figures(
    directory: Path, trajectories: list[dict[str, object]]
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix, x_key, y_key, xlabel, ylabel in (
        ("chamfer_vs_dofs", "active_k", "symmetric_chamfer", "active DoFs", "Chamfer"),
        ("p2s_p95_vs_dofs", "active_k", "point_to_surface_p95", "active DoFs", "P2S p95"),
        (
            "geometry_vs_time",
            "total_runtime_seconds",
            "symmetric_chamfer",
            "wall-clock time (s)",
            "Chamfer",
        ),
        (
            "runtime_vs_dofs",
            "active_k",
            "total_runtime_seconds",
            "active DoFs",
            "cumulative time (s)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7.0, 4.6))
        for trajectory in trajectories:
            rows = trajectory["rows"]
            axis.plot(
                [row[x_key] for row in rows],
                [row[y_key] for row in rows],
                marker="o",
                markersize=2,
                label=trajectory["method"],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if y_key != "total_runtime_seconds":
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / f"v03e_batch_{suffix}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    dynamic = next(item for item in trajectories if item["method"] == "dynamic")
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.step(
        [item["birth_round"] for item in dynamic["batches"]],
        dynamic["batch_schedule"],
        where="mid",
    )
    axis.set_xlabel("birth round")
    axis.set_ylabel("dynamic batch size")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = directory / "v03e_batch_dynamic_size.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for trajectory in trajectories:
        axis.plot(
            [item["birth_round"] for item in trajectory["batches"]],
            [item["rho_off"] for item in trajectory["batches"]],
            marker="o",
            markersize=2,
            label=trajectory["method"],
        )
    axis.axhline(0.15, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("birth round")
    axis.set_ylabel(r"$\rho_{off}$")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "v03e_batch_coupling.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _base_row(
    context: SequentialContext,
    target: dict[str, object],
    sigma: float,
) -> dict[str, object]:
    state = _evaluate_active(
        context,
        torch.arange(context.config.initial_count, device=context.reference_points.device),
        torch.zeros(
            context.config.initial_count,
            dtype=torch.float64,
            device=context.reference_points.device,
        ),
        target["images"],  # type: ignore[arg-type]
    )
    row: dict[str, object] = {
        "sigma": sigma,
        "method": "base",
        "active_k": 0,
        "image_loss": state.loss,
        "root_failures": state.root_failures,
        "total_runtime_seconds": 0.0,
        "peak_allocated_mib": 0.0,
    }
    row.update(context.geometry_evaluator(state.points))  # type: ignore[misc]
    return row


def _headroom(
    base: dict[str, object], fixed_full: dict[str, object], key: str
) -> float:
    return float(base[key]) - float(fixed_full[key])


def _recovered(
    base: dict[str, object],
    fixed_full: dict[str, object],
    row: dict[str, object],
    key: str,
) -> float:
    denominator = _headroom(base, fixed_full, key)
    if abs(denominator) <= 1e-15:
        return math.nan
    return 100.0 * (float(base[key]) - float(row[key])) / denominator


def _annotate_headroom(
    row: dict[str, object],
    sigma: float,
    base: dict[str, object],
    full: dict[str, object],
    transport_event_fraction: float,
) -> dict[str, object]:
    result = dict(row)
    result["sigma"] = sigma
    result["chamfer_recovered_headroom_percent"] = _recovered(
        base, full, row, "symmetric_chamfer"
    )
    result["p2s_p95_recovered_headroom_percent"] = _recovered(
        base, full, row, "point_to_surface_p95"
    )
    result["transport_event_fraction"] = transport_event_fraction
    result.setdefault("birth_rounds", 0)
    result.setdefault("mean_batch_size", 0.0)
    result.setdefault("median_batch_size", 0.0)
    result.setdefault("maximum_batch_size", 0)
    result.setdefault("cumulative_scoring_seconds", 0.0)
    result.setdefault("cumulative_optimization_seconds", 0.0)
    for region in ("ear", "head", "torso", "leg"):
        key = f"regional_{region}_p2s_mean"
        result[f"regional_{region}_recovered_headroom_percent"] = _recovered(
            base, full, row, key
        )
    return result


def _shared_observation_context(
    prepared: PreparedBunny,
    canonical: SequentialContext,
    canonical_target: dict[str, object],
    config: BunnyExperimentConfig,
) -> tuple[SequentialContext, dict[str, object], dict[str, object]]:
    """Change only base/dictionary while retaining v0.3d observations exactly."""
    device = canonical.reference_points.device
    base = prepared.base_field.to(device)
    gt = prepared.gt_field.to(device)
    import open3d as o3d

    vertices = prepared.base_field.surface_vertices.detach().cpu().numpy()
    faces = prepared.base_field.surface_faces.detach().cpu().numpy().astype(np.int32)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    projection_scene = o3d.t.geometry.RaycastingScene()
    projection_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    canonical_points = canonical.reference_points.detach().cpu().numpy().astype(np.float32)
    closest = projection_scene.compute_closest_points(
        o3d.core.Tensor(canonical_points)
    )["points"].numpy()
    closest_tensor = torch.from_numpy(closest.astype(np.float64)).to(device)
    reference_points = closest_tensor
    for _ in range(20):
        values = base.value(reference_points)
        gradients = base.interpolant_gradient(reference_points)
        reference_points = reference_points - values[:, None] * gradients / (
            (gradients * gradients).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )
    reference_residual = base.value(reference_points).abs()
    reference_success = reference_residual < 1e-8
    projection_distance = torch.linalg.vector_norm(
        reference_points - canonical.reference_points, dim=1
    )
    repeated = RepeatedBirthConfig(
        master_count=config.master_count,
        initial_count=config.initial_count,
        budget=config.budget,
        checkpoints=REPORT_LEVELS,
        target_regimes=("bunny",),
        target_seeds=(0,),
        random_seeds=(),
        views=config.views,
        resolution=config.resolution,
        emitters_per_master_basis=config.emitters // config.master_count,
        packets_per_emitter=config.packets_per_emitter,
        base_support_radius=config.base_support_radius,
        coefficient_limit=config.coefficient_limit,
        root_samples=config.root_samples,
        root_bisection_steps=config.root_bisection_steps,
        deformation_iterations=20,
        deformation_max_offset=0.03,
        initial_optimization_steps=config.initial_optimization_steps,
        post_birth_steps=config.post_birth_steps,
        fixed_optimization_steps=config.fixed_optimization_steps,
        run_oracle=False,
    )
    layout, levels = _hierarchical_layout(
        base,
        config.master_count,
        config.initial_count,
        config.base_support_radius,
        device,
    )
    reference_normals = unit_normals(base, reference_points)
    support = _multiscale_support(
        reference_points, layout, levels, repeated.support_margin
    )
    initial_field = LocalBasisField(
        base,
        layout.centers[: config.initial_count],
        layout.radii[: config.initial_count],
        torch.zeros(config.initial_count, dtype=torch.float64, device=device),
    )
    geometry = BenchmarkGeometry(
        initial_field,
        reference_points,
        reference_normals,
        canonical.colors,
    )
    evaluator = BunnyGeometryEvaluator(
        prepared,
        reference_points,
        reference_normals,
        config.evaluation_samples,
    )
    context = SequentialContext(
        repeated,
        base,
        layout,
        support,
        levels,
        reference_points,
        reference_normals,
        canonical.colors,
        canonical.cells,
        canonical.state,
        geometry,
        evaluator,
    )
    center_normals = unit_normals(base, layout.centers)
    _, success, displacement = _normal_line_roots(
        gt, layout.centers, center_normals
    )
    absolute = displacement.abs()
    target = {
        "regime": "bunny",
        "seed": 0,
        "points": canonical_target["points"],
        "images": canonical_target["images"],
        "transport_cells": canonical_target["transport_cells"],
        "detail_ids": torch.nonzero(
            absolute >= torch.quantile(absolute, 0.75), as_tuple=False
        ).flatten(),
        "detail_ids_top_ten": torch.nonzero(
            absolute >= torch.quantile(absolute, 0.90), as_tuple=False
        ).flatten(),
    }
    state = _evaluate_active(
        context,
        torch.arange(config.initial_count, device=device),
        torch.zeros(config.initial_count, dtype=torch.float64, device=device),
    )
    photons = _photon_batch(
        state.points,
        canonical.state.directions,
        canonical.colors,
        config.packets_per_emitter,
    )
    maximum_times = torch.full(
        (photons.count,), 3.0, dtype=torch.float64, device=device
    )
    scene_started = time.perf_counter()
    hits, times = first_zero_set_intersections(
        base,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
        chunk_size=8192,
    )
    current_state = SceneTransportState(
        state.points,
        unit_normals(base, state.points),
        photons,
        canonical.state.directions,
        hits,
        times,
        torch.ones(config.emitters, dtype=torch.bool, device=device),
    )
    view_config = MultiviewConfig(
        resolution=(config.resolution, config.resolution),
        emitters=config.emitters,
        packets_per_emitter=config.packets_per_emitter,
        parameter_count=config.master_count,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )
    current_transports = [
        project_camera(geometry, cell.camera, current_state, view_config)
        for cell in canonical.cells
    ]
    torch.cuda.synchronize()
    target_transports = canonical_target["transport_cells"]
    state_events = []
    owner_events = []
    for current, reference in zip(current_transports, target_transports):
        state_events.append(
            float((current.cell.photon_state != reference.cell.photon_state).double().mean())
        )
        occupied = (current.cell.owner_map >= 0) | (reference.cell.owner_map >= 0)
        owner_events.append(
            float(
                (current.cell.owner_map[occupied] != reference.cell.owner_map[occupied])
                .double()
                .mean()
            )
        )
    diagnostics = {
        "candidate_normal_line_failures": int((~success).sum()),
        "reference_projection_failures": int((~reference_success).sum()),
        "reference_projection_distance_mean": float(projection_distance.mean()),
        "reference_projection_distance_p95": float(
            torch.quantile(projection_distance, 0.95)
        ),
        "reference_projection_distance_max": float(projection_distance.max()),
        "reference_projection_residual_max": float(reference_residual.max()),
        "initial_reference_root_failures": state.root_failures,
        "photon_state_event_fraction_mean": statistics.mean(state_events),
        "owner_event_fraction_mean_on_occupied_union": statistics.mean(owner_events),
        "transport_diagnostic_seconds": time.perf_counter() - scene_started,
        "surface_hits": int(hits.sum()),
    }
    return context, target, diagnostics


def _phase_b_figures(
    directory: Path,
    sweep: list[dict[str, object]],
    trajectories: dict[str, dict[str, dict[str, object]]],
    contexts: dict[str, SequentialContext],
    targets: dict[str, dict[str, object]],
) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    sigmas = [float(item["sigma"]) for item in sweep]
    base_values = [float(item["base"]["symmetric_chamfer"]) for item in sweep]
    full_values = [float(item["fixed_1024"]["symmetric_chamfer"]) for item in sweep]
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.plot(sigmas, base_values, marker="o", label="base")
    axis.plot(sigmas, full_values, marker="o", label="fixed K=1024")
    axis.fill_between(sigmas, full_values, base_values, alpha=0.2, label="recoverable")
    axis.set_xlabel("Gaussian sigma (voxels)")
    axis.set_ylabel("symmetric Chamfer")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = directory / "v03e_headroom_vs_sigma.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    for suffix, key, ylabel in (
        (
            "chamfer_recovered",
            "chamfer_recovered_headroom_percent",
            "Chamfer recovered headroom (%)",
        ),
        (
            "p2s_p95_recovered",
            "p2s_p95_recovered_headroom_percent",
            "P2S p95 recovered headroom (%)",
        ),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
        for axis, sigma in zip(axes, sigmas):
            group = trajectories[f"{sigma:.1f}"]
            sweep_item = next(item for item in sweep if float(item["sigma"]) == sigma)
            for method, trajectory in group.items():
                metric = (
                    "symmetric_chamfer"
                    if "chamfer" in suffix
                    else "point_to_surface_p95"
                )
                values = [
                    _recovered(
                        sweep_item["base"],
                        sweep_item["fixed_1024"],
                        row,
                        metric,
                    )
                    for row in trajectory["rows"]
                ]
                axis.plot(
                    [row["active_k"] for row in trajectory["rows"]],
                    values,
                    marker="o",
                    markersize=2,
                    label=method,
                )
            axis.set_title(f"sigma={sigma:.1f}")
            axis.set_xlabel("active DoFs")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        figure.tight_layout()
        path = directory / f"v03e_{suffix}_vs_dofs.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    adaptive_advantage = []
    for item in sweep:
        sigma_key = f"{float(item['sigma']):.1f}"
        group = trajectories[sigma_key]
        adaptive = group["adaptive"]["rows"][-1]
        uniform = group["uniform_matched"]["rows"][-1]
        adaptive_advantage.append(
            _recovered(item["base"], item["fixed_1024"], adaptive, "symmetric_chamfer")
            - _recovered(item["base"], item["fixed_1024"], uniform, "symmetric_chamfer")
        )
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.plot(sigmas, adaptive_advantage, marker="o")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Gaussian sigma (voxels)")
    axis.set_ylabel("adaptive - uniform recovered Chamfer (pp)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = directory / "v03e_adaptive_advantage_vs_sigma.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for sigma in sigmas:
        for method, trajectory in trajectories[f"{sigma:.1f}"].items():
            axis.plot(
                [row["total_runtime_seconds"] for row in trajectory["rows"]],
                [row["symmetric_chamfer"] for row in trajectory["rows"]],
                marker="o",
                markersize=2,
                label=f"sigma={sigma:.1f} {method}",
            )
    axis.set_xlabel("wall-clock time (s)")
    axis.set_ylabel("symmetric Chamfer")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = directory / "v03e_smoothing_geometry_vs_time.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 9.8))
    for row_index, sigma in enumerate(sigmas):
        context = contexts[f"{sigma:.1f}"]
        target = targets[f"{sigma:.1f}"]
        base_state = _evaluate_active(
            context,
            torch.arange(
                context.config.initial_count,
                device=context.reference_points.device,
            ),
            torch.zeros(
                context.config.initial_count,
                dtype=torch.float64,
                device=context.reference_points.device,
            ),
        )
        background = base_state.points.detach().cpu().numpy()[::8]
        error = torch.linalg.vector_norm(
            base_state.points - target["points"], dim=1  # type: ignore[operator]
        ).detach().cpu().numpy()[::8]
        batches = trajectories[f"{sigma:.1f}"]["adaptive"]["batches"]
        for active_k, axis in zip((64, 128, 256), axes[row_index]):
            endpoints = [context.config.initial_count] + [
                int(batch["active_k_after"]) for batch in batches
            ]
            actual_k = min(endpoints, key=lambda value: abs(value - active_k))
            needed = actual_k - context.config.initial_count
            flat_ids: list[int] = []
            for batch in batches:
                flat_ids.extend(int(item) for item in batch["candidate_ids"])
                if len(flat_ids) >= needed:
                    break
            axis.scatter(background[:, 0], background[:, 1], c=error, s=1, cmap="magma", alpha=0.4)
            ids = flat_ids
            centers = context.master_layout.centers[ids].detach().cpu().numpy()
            axis.scatter(centers[:, 0], centers[:, 1], s=5, c="cyan")
            axis.set_title(f"σ={sigma:.1f}, K≈{active_k} ({actual_k})")
            axis.set_aspect("equal")
            axis.set_axis_off()
    figure.tight_layout()
    path = directory / "v03e_smoothing_birth_geography.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    return paths


def run_batch_birth_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    config = BunnyExperimentConfig(
        checkpoints=REPORT_LEVELS,
        oracle_subset_size=0,
        random_seeds=(),
    )
    prepared = prepare_stanford_bunny(mesh_path)
    context, target, transport = _build_bunny_context(prepared, config)
    policies = (
        BatchPolicy("p1", nominal_size=1),
        BatchPolicy("p4", nominal_size=4),
        BatchPolicy("p8", nominal_size=8),
        BatchPolicy("p16", nominal_size=16),
        BatchPolicy("dynamic", dynamic=True),
    )
    trajectories = []
    for policy in policies:
        trajectory = _run_batch_trajectory(context, target, policy)
        trajectories.append(trajectory)
        print(
            json.dumps(
                {
                    "phase": "A",
                    "method": policy.name,
                    "final_chamfer": trajectory["rows"][-1]["symmetric_chamfer"],
                    "runtime_seconds": trajectory["total_runtime_seconds"],
                    "birth_rounds": trajectory["birth_rounds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    base = _base_row(context, target, 1.5)
    serial = trajectories[0]
    serial_final = serial["rows"][-1]
    eligible = []
    eligibility: dict[str, object] = {}
    for trajectory in trajectories[1:]:
        final = trajectory["rows"][-1]
        chamfer_recovery = float(base["symmetric_chamfer"]) - float(
            final["symmetric_chamfer"]
        )
        serial_chamfer_recovery = float(base["symmetric_chamfer"]) - float(
            serial_final["symmetric_chamfer"]
        )
        p95_recovery = float(base["point_to_surface_p95"]) - float(
            final["point_to_surface_p95"]
        )
        serial_p95_recovery = float(base["point_to_surface_p95"]) - float(
            serial_final["point_to_surface_p95"]
        )
        numerical = (
            int(final["root_failures"]) == 0
            and int(final["cg_failures"]) == 0
            and int(final["line_search_failures"]) == 0
            and int(final["nonfinite_candidate_score_count"]) == 0
            and trajectory["maximum_birth_geometry_jump"] < 1e-12
        )
        checks = {
            "chamfer_recovered_headroom_ratio_vs_serial": chamfer_recovery
            / max(serial_chamfer_recovery, 1e-30),
            "p2s_p95_recovered_headroom_ratio_vs_serial": p95_recovery
            / max(serial_p95_recovery, 1e-30),
            "numerically_valid": numerical,
            "runtime_lower_than_serial": trajectory["total_runtime_seconds"]
            < serial["total_runtime_seconds"],
            "birth_geometry_unchanged": trajectory[
                "maximum_birth_geometry_jump"
            ]
            < 1e-12,
        }
        passed = (
            checks["chamfer_recovered_headroom_ratio_vs_serial"] >= 0.95
            and checks["p2s_p95_recovered_headroom_ratio_vs_serial"] >= 0.95
            and checks["numerically_valid"]
            and checks["runtime_lower_than_serial"]
            and checks["birth_geometry_unchanged"]
        )
        checks["eligible"] = passed
        eligibility[trajectory["method"]] = checks
        if passed:
            eligible.append(trajectory)
    winner = min(
        eligible,
        key=lambda item: (
            float(item["total_runtime_seconds"]),
            float(item["rows"][-1]["symmetric_chamfer"]),
            float(item["rows"][-1]["point_to_surface_p95"]),
        ),
        default=None,
    )
    phase_a_report = {
        "verdict": "BATCH_BIRTH_SUPPORTED" if winner else "BATCH_BIRTH_NOT_SUPPORTED",
        "best_multi_birth": winner["method"] if winner else None,
        "configuration": {
            **config.__dict__,
            "checkpoints": list(config.checkpoints),
            "compatibility_tau": 0.10,
            "dynamic_alpha": 0.10,
            "dynamic_rho_max": 0.15,
            "dynamic_maximum_batch_size": 32,
            "true_simultaneous_zero_initialization": True,
        },
        "environment": cuda_environment(),
        "bunny": prepared.metadata,
        "target_image_sha256": _image_digest(target["images"]),
        "transport": transport,
        "base": base,
        "trajectory_summaries": [_trajectory_summary(item) for item in trajectories],
        "trajectories": trajectories,
        "eligibility": eligibility,
        "failed_gates": [] if winner else ["no_eligible_multi_birth_strategy"],
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    phase_a_csv = artifact_directory / "v03e_batch_birth.csv"
    phase_a_json = artifact_directory / "v03e_batch_birth.json"
    _write_rows(
        phase_a_csv,
        [
            {"phase": "A", **row}
            for trajectory in trajectories
            for row in _report_rows(trajectory)
        ],
    )
    phase_a_report["figures"] = _phase_a_figures(figure_directory, trajectories)
    phase_a_json.write_text(json.dumps(phase_a_report, indent=2, sort_keys=True) + "\n")
    if winner is None:
        return {
            "phase_a": phase_a_report,
            "phase_b": "BUNNY_HEADROOM_PHASE_SKIPPED",
        }

    contexts: dict[str, SequentialContext] = {"1.5": context}
    targets: dict[str, dict[str, object]] = {"1.5": target}
    transports: dict[str, dict[str, object]] = {"1.5": transport}
    for sigma in (2.5, 4.0):
        key = f"{sigma:.1f}"
        current = resmooth_stanford_bunny(prepared, sigma)
        contexts[key], targets[key], transports[key] = _shared_observation_context(
            current, context, target, config
        )
    sweep = []
    for sigma in (1.5, 2.5, 4.0):
        key = f"{sigma:.1f}"
        current_context = contexts[key]
        current_target = targets[key]
        base_row = _base_row(current_context, current_target, sigma)
        fixed_32 = _run_fixed(current_context, current_target, 32)
        fixed_256 = _run_fixed(current_context, current_target, 256)
        fixed_1024 = _run_fixed(current_context, current_target, 1024)
        sweep.append(
            {
                "sigma": sigma,
                "base": base_row,
                "fixed_32": fixed_32,
                "fixed_256": fixed_256,
                "fixed_1024": fixed_1024,
                "recoverable_headroom": {
                    metric: _headroom(base_row, fixed_1024, metric)
                    for metric in (
                        "symmetric_chamfer",
                        "point_to_surface_mean",
                        "point_to_surface_p95",
                        "surface_rms",
                        "normal_error",
                    )
                },
            }
        )
        print(
            json.dumps(
                {
                    "phase": "B-headroom",
                    "sigma": sigma,
                    "base_chamfer": base_row["symmetric_chamfer"],
                    "fixed_1024_chamfer": fixed_1024["symmetric_chamfer"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    phase_b_trajectories: dict[str, dict[str, dict[str, object]]] = {}
    for sigma in (1.5, 2.5, 4.0):
        key = f"{sigma:.1f}"
        if sigma == 1.5:
            adaptive = winner
        else:
            adaptive_policy = next(
                policy for policy in policies if policy.name == winner["method"]
            )
            adaptive = _run_batch_trajectory(
                contexts[key], targets[key], adaptive_policy
            )
        uniform = _run_batch_trajectory(
            contexts[key],
            targets[key],
            BatchPolicy(
                "uniform_matched",
                uniform_schedule=tuple(adaptive["batch_schedule"]),
            ),
        )
        phase_b_trajectories[key] = {
            "adaptive": adaptive,
            "uniform_matched": uniform,
        }
        print(
            json.dumps(
                {
                    "phase": "B",
                    "sigma": sigma,
                    "adaptive_chamfer": adaptive["rows"][-1]["symmetric_chamfer"],
                    "uniform_chamfer": uniform["rows"][-1]["symmetric_chamfer"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    phase_b_rows = []
    phase_b_summaries = []
    for item in sweep:
        sigma = float(item["sigma"])
        key = f"{sigma:.1f}"
        event = float(transports[key]["photon_state_event_fraction_mean"])
        for fixed_name in ("base", "fixed_32", "fixed_256", "fixed_1024"):
            phase_b_rows.append(
                _annotate_headroom(
                    item[fixed_name], sigma, item["base"], item["fixed_1024"], event
                )
            )
        for method, trajectory in phase_b_trajectories[key].items():
            for row in _report_rows(trajectory):
                annotated = _annotate_headroom(
                    {"method": method, **row},
                    sigma,
                    item["base"],
                    item["fixed_1024"],
                    event,
                )
                annotated.update(
                    {
                        "birth_rounds": trajectory["birth_rounds"],
                        "mean_batch_size": trajectory["mean_batch_size"],
                        "median_batch_size": trajectory["median_batch_size"],
                        "maximum_batch_size": trajectory["maximum_batch_size"],
                    }
                )
                phase_b_rows.append(annotated)
            phase_b_summaries.append(
                {
                    "sigma": sigma,
                    "role": method,
                    **_trajectory_summary(trajectory),
                    "final_chamfer_recovered_headroom_percent": _recovered(
                        item["base"],
                        item["fixed_1024"],
                        trajectory["rows"][-1],
                        "symmetric_chamfer",
                    ),
                    "final_p2s_p95_recovered_headroom_percent": _recovered(
                        item["base"],
                        item["fixed_1024"],
                        trajectory["rows"][-1],
                        "point_to_surface_p95",
                    ),
                }
            )
    phase_b_csv = artifact_directory / "v03e_bunny_smoothing.csv"
    phase_b_json = artifact_directory / "v03e_bunny_smoothing.json"
    _write_rows(phase_b_csv, phase_b_rows)
    figures = _phase_b_figures(
        figure_directory, sweep, phase_b_trajectories, contexts, targets
    )
    advantage = []
    for item in sweep:
        key = f"{float(item['sigma']):.1f}"
        adaptive = phase_b_trajectories[key]["adaptive"]["rows"][-1]
        uniform = phase_b_trajectories[key]["uniform_matched"]["rows"][-1]
        advantage.append(
            _recovered(item["base"], item["fixed_1024"], adaptive, "symmetric_chamfer")
            - _recovered(item["base"], item["fixed_1024"], uniform, "symmetric_chamfer")
        )
    headrooms = [
        float(item["recoverable_headroom"]["symmetric_chamfer"]) for item in sweep
    ]
    sequential_rows = [
        trajectory["rows"][-1]
        for group in phase_b_trajectories.values()
        for trajectory in group.values()
    ]
    no_root_cg_or_nonfinite_failure = all(
        int(row["root_failures"]) == 0
        and int(row["cg_failures"]) == 0
        and int(row["nonfinite_candidate_score_count"]) == 0
        for row in sequential_rows
    )
    no_systematic_line_search_failure = all(
        int(trajectory["rows"][-1]["line_search_failures"])
        < int(trajectory["birth_rounds"])
        for group in phase_b_trajectories.values()
        for trajectory in group.values()
    )
    fixed_reference_monotone = all(
        float(item["fixed_1024"]["symmetric_chamfer"])
        <= float(item["fixed_256"]["symmetric_chamfer"])
        for item in sweep
    )
    phase_b_supported = (
        headrooms[-1] > headrooms[0]
        and statistics.mean(advantage[1:]) > advantage[0]
        and any(value > 0.0 for value in advantage)
        and no_root_cg_or_nonfinite_failure
        and no_systematic_line_search_failure
        and fixed_reference_monotone
    )
    phase_b_report = {
        "verdict": (
            "BUNNY_HEADROOM_ADAPTIVE_BIRTH_SUPPORTED"
            if phase_b_supported
            else "BUNNY_HEADROOM_ADAPTIVE_BIRTH_NOT_SUPPORTED"
        ),
        "best_multi_birth": winner["method"],
        "bunny": prepared.metadata,
        "target_image_sha256": _image_digest(target["images"]),
        "observation_contract": (
            "all sigmas reuse the exact sigma=1.5 emitter identities/colors, "
            "photon directions, cameras, fixed transport cells, and GT target "
            "images; identities are deterministically closest-point transferred "
            "onto each changed base surface"
        ),
        "sweep": sweep,
        "transports": transports,
        "trajectory_summaries": phase_b_summaries,
        "trajectories": phase_b_trajectories,
        "adaptive_minus_uniform_recovered_chamfer_percentage_points": advantage,
        "support_gate": {
            "headroom_increases": headrooms[-1] > headrooms[0],
            "adaptive_advantage_grows": statistics.mean(advantage[1:]) > advantage[0],
            "adaptive_beats_uniform_somewhere": any(value > 0.0 for value in advantage),
            "no_root_cg_or_nonfinite_failure": no_root_cg_or_nonfinite_failure,
            "no_systematic_line_search_failure": no_systematic_line_search_failure,
            "fixed_reference_monotone": fixed_reference_monotone,
        },
        "failed_gates": [
            key
            for key, passed in {
                "headroom_does_not_increase": headrooms[-1] > headrooms[0],
                "adaptive_advantage_does_not_grow": statistics.mean(advantage[1:])
                > advantage[0],
                "adaptive_never_beats_uniform": any(value > 0.0 for value in advantage),
                "root_cg_or_nonfinite_failure": no_root_cg_or_nonfinite_failure,
                "systematic_line_search_failure": no_systematic_line_search_failure,
                "nonmonotone_fixed_reference": fixed_reference_monotone,
            }.items()
            if not passed
        ],
        "figures": figures,
    }
    phase_b_json.write_text(json.dumps(phase_b_report, indent=2, sort_keys=True) + "\n")
    return {"phase_a": phase_a_report, "phase_b": phase_b_report}
