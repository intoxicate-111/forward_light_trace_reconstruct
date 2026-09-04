"""Full-HD, 20-view, million-packet corrected-RGB birth experiment."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import cuda_environment
from .corrected_birth import (
    CorrectedBirthConfig,
    CorrectedContext,
    CorrectedState,
    _build_context,
    _evaluate,
    _finite_difference_gate,
    _json_ready,
    _metrics,
    _optimize,
    _run_fixed_references,
    _run_matched_baseline,
    _run_observation_trajectory,
    _run_raw_baseline,
    _score_candidates,
    _trajectory_auc,
    _write_csv,
)
from .mesh_field import prepare_stanford_bunny


Tensor = torch.Tensor


@dataclass(frozen=True)
class HighBandwidthConfig:
    low_emitters: int = 16_384
    medium_emitters: int = 32_768
    high_emitters: int = 65_536
    views: int = 20
    low_resolution: tuple[int, int] = (256, 256)
    high_resolution: tuple[int, int] = (1080, 1920)
    main_dictionary_count: int = 8192
    ablation_dictionary_count: int = 2048
    initial_count: int = 32
    ablation_target_k: int = 1024
    fixed_optimization_steps: int = 5
    ablation_optimization_steps: int = 3


def _corrected_config(
    experiment: HighBandwidthConfig,
    emitters: int,
    resolution: tuple[int, int],
    *,
    main: bool,
) -> CorrectedBirthConfig:
    return CorrectedBirthConfig(
        dictionary_count=(
            experiment.main_dictionary_count
            if main
            else experiment.ablation_dictionary_count
        ),
        initial_count=experiment.initial_count,
        surface_samples=emitters,
        views=experiment.views,
        resolution=resolution[0],
        resolution_width=resolution[1],
        initial_optimization_steps=5 if main else 3,
        post_birth_steps=1,
        fixed_optimization_steps=(
            experiment.fixed_optimization_steps
            if main
            else experiment.ablation_optimization_steps
        ),
        minimum_active_before_stopping=1024,
        flattening_window=3,
        flattening_relative_gain=2e-4,
        dynamic_batch=main,
        dynamic_score_floor_fraction=0.05,
        dynamic_predicted_gain_fraction=0.95,
        require_geometry_flattening=main,
        geometry_flattening_gain=1e-7,
    )


def _profile_context(
    prepared: object,
    config: CorrectedBirthConfig,
    label: str,
) -> tuple[CorrectedContext, dict[str, object]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = _build_context(prepared, config)
    active = torch.arange(config.initial_count, device="cuda")
    state = _evaluate(
        context,
        active,
        torch.zeros(active.numel(), dtype=torch.float64, device="cuda"),
    )
    row = _metrics(
        context,
        state,
        f"profile_{label}",
        0,
        0.0,
        0.0,
        time.perf_counter() - started,
    )
    result = {
        "condition": label,
        "emitters": config.surface_samples,
        "packets_per_emitter": config.views,
        "total_attempted_packets": config.surface_samples * config.views,
        "views": config.views,
        "resolution": list(config.resolution_shape),
        "context_build_and_initial_render_seconds": time.perf_counter()
        - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "initial_image_loss": state.loss,
        "initial_normalized_rgb_mse": row["normalized_rgb_mse"],
        "initial_rgb_psnr": row["rgb_psnr"],
        "retained_forward_events": context.visibility_report["base"][
            "retained"
        ],
        "absorbed_forward_events": context.visibility_report["base"][
            "absorbed"
        ],
        "camera_digest": context.visibility_report["camera_atlas"][
            "digest"
        ],
    }
    del state
    torch.cuda.empty_cache()
    return context, result


def _snapshot(
    context: CorrectedContext,
    state: CorrectedState,
    view_ids: tuple[int, ...],
) -> list[np.ndarray]:
    rows, columns = context.config.resolution_shape
    return [
        state.images[index]
        .reshape(rows, columns, 3)
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        for index in view_ids
    ]


def _state_checkpoint(state: CorrectedState) -> dict[str, Tensor]:
    """Keep only the sparse model state needed to reconstruct dense renders."""
    return {
        "active_ids": state.active_ids.detach().cpu(),
        "coefficients": state.coefficients.detach().cpu(),
    }


def _restore_checkpoint_state(
    context: CorrectedContext, payload: dict[str, Tensor]
) -> CorrectedState:
    device = context.reference_points.device
    return _evaluate(
        context,
        payload["active_ids"].to(device),
        payload["coefficients"].to(device),
    )


def _write_main_progress(
    report_path: Path,
    state_path: Path,
    signature: dict[str, object],
    completed: dict[str, object],
    states: dict[str, dict[str, Tensor]],
    stage: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "0.8.0",
        "stage": stage,
        "signature": signature,
        "completed": completed,
    }
    report_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
    )
    torch.save(states, state_path)


def _target_snapshot(
    context: CorrectedContext, view_ids: tuple[int, ...]
) -> list[np.ndarray]:
    rows, columns = context.config.resolution_shape
    return [
        context.target_images[index]
        .reshape(rows, columns, 3)
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        for index in view_ids
    ]


def _run_warm_fixed_references(
    context: CorrectedContext,
    maximum_active: int,
) -> dict[str, object]:
    levels = [
        level
        for level in (256, 512, 1024, 2048, 4096, 8192)
        if level <= maximum_active
        and level <= context.config.dictionary_count
    ]
    if maximum_active not in levels:
        levels.append(maximum_active)
    levels = sorted(set(levels))
    started = time.perf_counter()
    state: CorrectedState | None = None
    rows: list[dict[str, object]] = []
    for level in levels:
        active = torch.arange(level, device=context.reference_points.device)
        if state is None:
            coefficients = torch.zeros(
                level, dtype=torch.float64, device=active.device
            )
        else:
            coefficients = torch.cat(
                (
                    state.coefficients,
                    state.coefficients.new_zeros(
                        level - state.active_ids.numel()
                    ),
                )
            )
            del state
        state = _evaluate(context, active, coefficients)
        state, seconds = _optimize(
            context, state, context.config.fixed_optimization_steps
        )
        rows.append(
            _metrics(
                context,
                state,
                "fixed_space_warm",
                len(rows),
                0.0,
                seconds,
                time.perf_counter() - started,
                {"fixed_k": level},
            )
        )
    del state
    torch.cuda.empty_cache()
    return {
        "method": "fixed_space_warm",
        "levels": levels,
        "rows": rows,
        "total_seconds": time.perf_counter() - started,
    }


def _one_shot_factor_cell(
    context: CorrectedContext,
    condition: str,
    target_k: int,
) -> dict[str, object]:
    device = context.reference_points.device
    started = time.perf_counter()
    initial_ids = torch.arange(context.config.initial_count, device=device)
    initial = _evaluate(
        context,
        initial_ids,
        torch.zeros(initial_ids.numel(), dtype=torch.float64, device=device),
    )
    initial, _ = _optimize(
        context, initial, context.config.initial_optimization_steps
    )
    initial_coefficients = initial.coefficients.clone()
    scores, score_diagnostics, score_seconds = _score_candidates(
        context, initial
    )
    inactive = torch.ones(
        context.config.dictionary_count, dtype=torch.bool, device=device
    )
    inactive[initial_ids] = False
    raw_values = torch.where(
        inactive,
        scores["raw"],
        torch.full_like(scores["raw"], -math.inf),
    )
    selected = torch.argsort(
        raw_values, descending=True, stable=True
    )[: target_k - context.config.initial_count]
    del scores, initial
    raw_ids = torch.cat((initial_ids, selected))
    raw = _evaluate(
        context,
        raw_ids,
        torch.cat(
            (
                initial_coefficients,
                initial_coefficients.new_zeros(selected.numel()),
            )
        ),
    )
    raw, raw_seconds = _optimize(
        context, raw, context.config.fixed_optimization_steps
    )
    raw_row = _metrics(
        context,
        raw,
        "raw_one_shot",
        1,
        score_seconds,
        raw_seconds,
        time.perf_counter() - started,
        {
            "condition": condition,
            "selected_count": int(selected.numel()),
            "predicted_raw_alignment": float(raw_values[selected].sum()),
            **score_diagnostics,
        },
    )
    del raw
    uniform_ids = torch.arange(target_k, device=device)
    uniform = _evaluate(
        context,
        uniform_ids,
        torch.cat(
            (
                initial_coefficients,
                initial_coefficients.new_zeros(
                    target_k - context.config.initial_count
                ),
            )
        ),
    )
    uniform, uniform_seconds = _optimize(
        context,
        uniform,
        context.config.fixed_optimization_steps,
    )
    uniform_row = _metrics(
        context,
        uniform,
        "uniform_one_shot",
        1,
        0.0,
        uniform_seconds,
        time.perf_counter() - started,
        {"condition": condition},
    )
    del uniform
    fixed, fixed_state = _run_fixed_references(context, target_k)
    del fixed_state
    torch.cuda.empty_cache()
    fixed_rows = [
        {**row, "condition": condition} for row in fixed["rows"]
    ]
    return {
        "condition": condition,
        "emitters": context.config.surface_samples,
        "packets_per_emitter": context.config.views,
        "total_attempted_packets": (
            context.config.surface_samples * context.config.views
        ),
        "views": context.config.views,
        "resolution": list(context.config.resolution_shape),
        "raw": raw_row,
        "uniform": uniform_row,
        "fixed_rows": fixed_rows,
        "total_seconds": time.perf_counter() - started,
    }


def _save_main_curves(
    path: Path, trajectories: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for trajectory in trajectories:
        rows = trajectory["rows"]
        label = str(trajectory["method"])
        active = [int(row["active_dofs"]) for row in rows]
        seconds = [float(row["total_seconds"]) for row in rows]
        loss = [float(row["image_loss"]) for row in rows]
        chamfer = [float(row["symmetric_chamfer"]) for row in rows]
        axes[0, 0].plot(active, loss, "o-", label=label)
        axes[0, 1].plot(active, chamfer, "o-", label=label)
        axes[1, 0].plot(seconds, loss, "o-", label=label)
        axes[1, 1].plot(seconds, chamfer, "o-", label=label)
    labels = (
        ("active DoFs", "RGB loss"),
        ("active DoFs", "symmetric Chamfer"),
        ("seconds", "RGB loss"),
        ("seconds", "symmetric Chamfer"),
    )
    for axis, (x_label, y_label) in zip(axes.reshape(-1), labels):
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_predictor_plot(
    path: Path,
    quadratic: dict[str, object],
    raw: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, report, key, title in (
        (
            axes[0],
            quadratic,
            "predicted_joint_gain",
            "quadratic joint prediction",
        ),
        (
            axes[1],
            raw,
            "predicted_raw_alignment",
            "raw alignment prediction",
        ),
    ):
        predicted = [float(row[key]) for row in report["batches"]]
        realized = [float(row["realized_gain"]) for row in report["batches"]]
        axis.scatter(predicted, realized, s=28, alpha=0.8)
        axis.set_xlabel(title)
        axis.set_ylabel("realized RGB gain")
        axis.grid(alpha=0.25)
        axis.set_title(
            f"Pearson {report['predicted_realized_pearson']:.3f}, "
            f"Spearman {report['predicted_realized_spearman']:.3f}"
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_batch_plot(path: Path, report: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    batches = report["batches"]
    rounds = [int(row["birth_round"]) for row in batches]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(rounds, [int(row["batch_size"]) for row in batches], "o-")
    axes[0].set_ylabel("accepted batch size")
    axes[1].plot(
        rounds,
        [float(row["pairwise_max"]) for row in batches],
        "o-",
        label="max pair cosine",
    )
    axes[1].plot(
        rounds,
        [float(row["rho_off"]) for row in batches],
        "o-",
        label="rho off",
    )
    axes[1].legend(fontsize=8)
    axes[2].plot(
        rounds,
        [float(row["relative_realized_gain"]) for row in batches],
        "o-",
    )
    axes[2].set_ylabel("relative realized RGB gain")
    for axis in axes:
        axis.set_xlabel("birth round")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_birth_geography(
    path: Path, context: CorrectedContext, report: dict[str, object]
) -> None:
    import matplotlib.pyplot as plt

    centers = context.master_layout.centers.detach().cpu().numpy()
    displacement = (
        context.target_displacement[: context.config.dictionary_count]
        .abs()
        .detach()
        .cpu()
        .numpy()
    )
    rounds = np.full(context.config.dictionary_count, np.nan)
    for batch in report["batches"]:
        rounds[np.asarray(batch["selected_ids"], dtype=np.int64)] = float(
            batch["birth_round"]
        )
    selected = np.isfinite(rounds)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    background = axes[0].scatter(
        centers[:, 0], centers[:, 2], c=displacement, s=4, cmap="magma"
    )
    figure.colorbar(background, ax=axes[0], label="independent |target offset|")
    born = axes[1].scatter(
        centers[selected, 0],
        centers[selected, 2],
        c=rounds[selected],
        s=8,
        cmap="viridis",
    )
    figure.colorbar(born, ax=axes[1], label="birth round")
    axes[0].set_title("independent geometry-error geography")
    axes[1].set_title("quadratic birth geography")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        axis.set_aspect("equal")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_multiview(
    path: Path,
    view_ids: tuple[int, ...],
    snapshots: dict[str, list[np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    labels = ("target", "quadratic", "raw", "uniform")
    figure, axes = plt.subplots(
        len(labels), len(view_ids), figsize=(3.4 * len(view_ids), 9.5)
    )
    figure.patch.set_facecolor("black")
    for row, label in enumerate(labels):
        for column, view_id in enumerate(view_ids):
            axis = axes[row, column]
            axis.imshow(np.clip(snapshots[label][column], 0.0, 1.0))
            axis.set_title(
                f"{label}, view {view_id}", color="white", fontsize=8
            )
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor="black")
    plt.close(figure)


def _save_factor_plot(
    path: Path, factors: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(item["condition"]) for item in factors]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label, marker in (
        ("raw", "raw", "o"),
        ("uniform", "uniform", "s"),
    ):
        axes[0].plot(
            x,
            [float(item[key]["normalized_rgb_mse"]) for item in factors],
            marker=marker,
            label=label,
        )
        axes[1].plot(
            x,
            [float(item[key]["symmetric_chamfer"]) for item in factors],
            marker=marker,
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("normalized RGB MSE at K=1024")
    axes[1].set_ylabel("symmetric Chamfer at K=1024")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _factorial_effects(
    factors: list[dict[str, object]],
) -> dict[str, object]:
    """Report signed 2x2 effects; negative deltas mean lower error."""
    cells = {str(item["condition"]): item for item in factors}

    def row(condition: str, method: str, fixed_k: int | None = None) -> dict:
        cell = cells[condition]
        if fixed_k is None:
            return cell[method]
        return next(
            item for item in cell["fixed_rows"] if item["fixed_k"] == fixed_k
        )

    methods: dict[str, tuple[str, int | None]] = {
        "raw": ("raw", None),
        "uniform": ("uniform", None),
    }
    for fixed_k in (256, 1024):
        if all(
            any(row["fixed_k"] == fixed_k for row in cell["fixed_rows"])
            for cell in cells.values()
        ):
            methods[f"fixed_k{fixed_k}"] = ("fixed", fixed_k)
    report: dict[str, object] = {
        "delta_convention": "second condition minus first; negative is better",
        "methods": {},
    }
    for label, (method, fixed_k) in methods.items():
        method_report: dict[str, object] = {}
        for metric in ("normalized_rgb_mse", "symmetric_chamfer"):
            low_low = float(row("256_low_photons", method, fixed_k)[metric])
            low_high = float(
                row("256_m_level_photons", method, fixed_k)[metric]
            )
            high_low = float(
                row("1080p_low_photons", method, fixed_k)[metric]
            )
            high_high = float(
                row("1080p_m_level_photons", method, fixed_k)[metric]
            )
            photon_at_256 = low_high - low_low
            photon_at_1080 = high_high - high_low
            resolution_at_low = high_low - low_low
            resolution_at_high = high_high - low_high
            method_report[metric] = {
                "photon_effect_at_256": photon_at_256,
                "photon_effect_at_1080p": photon_at_1080,
                "resolution_effect_at_low_photons": resolution_at_low,
                "resolution_effect_at_m_level_photons": resolution_at_high,
                "interaction": photon_at_1080 - photon_at_256,
            }
        report["methods"][label] = method_report
    return report


def _main_comparison(
    context: CorrectedContext,
    figure_directory: Path,
    render_directory: Path,
    progress_path: Path,
    state_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    view_ids = (0, 5, 10, 15)
    snapshots = {"target": _target_snapshot(context, view_ids)}
    signature = {
        "config": _json_ready(asdict(context.config)),
        "camera_digest": context.visibility_report["camera_atlas"]["digest"],
        "transport_digest": context.visibility_report["base"]["digest"],
    }
    completed: dict[str, object] = {}
    checkpoint_states: dict[str, dict[str, Tensor]] = {}
    if progress_path.exists() and state_path.exists():
        saved = json.loads(progress_path.read_text())
        if saved.get("signature") == signature:
            completed = dict(saved.get("completed", {}))
            checkpoint_states = torch.load(
                state_path, map_location="cpu", weights_only=True
            )

    finite_difference = completed.get("finite_difference")
    if finite_difference is None:
        finite_difference = _finite_difference_gate(context)
        completed["finite_difference"] = finite_difference
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "finite_difference",
        )
    if not finite_difference["passed"]:
        raise RuntimeError(
            f"HIGH_BANDWIDTH_JACOBIAN_FAILED:{finite_difference}"
        )

    quadratic = completed.get("quadratic")
    if quadratic is None or "quadratic" not in checkpoint_states:
        quadratic, state = _run_observation_trajectory(context)
        completed["quadratic"] = quadratic
        checkpoint_states["quadratic"] = _state_checkpoint(state)
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "quadratic",
        )
    else:
        state = _restore_checkpoint_state(
            context, checkpoint_states["quadratic"]
        )
    snapshots["quadratic"] = _snapshot(context, state, view_ids)
    del state
    torch.cuda.empty_cache()
    schedule = list(quadratic["batch_schedule"])

    uniform = completed.get("uniform")
    if uniform is None or "uniform" not in checkpoint_states:
        uniform, state = _run_matched_baseline(
            context, schedule, "uniform_matched"
        )
        completed["uniform"] = uniform
        checkpoint_states["uniform"] = _state_checkpoint(state)
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "uniform",
        )
    else:
        state = _restore_checkpoint_state(
            context, checkpoint_states["uniform"]
        )
    snapshots["uniform"] = _snapshot(context, state, view_ids)
    del state
    torch.cuda.empty_cache()

    random = completed.get("random")
    if random is None:
        random, state = _run_matched_baseline(
            context, schedule, "random_matched"
        )
        del state
        completed["random"] = random
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "random",
        )
        torch.cuda.empty_cache()

    raw = completed.get("raw")
    if raw is None or "raw" not in checkpoint_states:
        raw, state = _run_raw_baseline(context, schedule)
        completed["raw"] = raw
        checkpoint_states["raw"] = _state_checkpoint(state)
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "raw",
        )
    else:
        state = _restore_checkpoint_state(context, checkpoint_states["raw"])
    snapshots["raw"] = _snapshot(context, state, view_ids)
    del state
    torch.cuda.empty_cache()

    maximum_active = int(quadratic["maximum_validated_active_dofs"])
    cold = completed.get("cold")
    if cold is None:
        cold, state = _run_fixed_references(context, maximum_active)
        del state
        completed["cold"] = cold
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "cold_fixed",
        )
        torch.cuda.empty_cache()

    warm = completed.get("warm")
    if warm is None:
        warm = _run_warm_fixed_references(context, maximum_active)
        completed["warm"] = warm
        _write_main_progress(
            progress_path,
            state_path,
            signature,
            completed,
            checkpoint_states,
            "main_complete",
        )
    trajectories = [quadratic, raw, uniform, random]
    _save_main_curves(
        figure_directory / "v08_high_bandwidth_quality_curves.png",
        trajectories,
    )
    _save_predictor_plot(
        figure_directory / "v08_high_bandwidth_predictor_calibration.png",
        quadratic,
        raw,
    )
    _save_batch_plot(
        figure_directory / "v08_high_bandwidth_batch_trajectory.png",
        quadratic,
    )
    _save_birth_geography(
        figure_directory / "v08_high_bandwidth_birth_geography.png",
        context,
        quadratic,
    )
    _save_multiview(
        render_directory / "v08_high_bandwidth_multiview.png",
        view_ids,
        snapshots,
    )
    final = {item["method"]: item["rows"][-1] for item in trajectories}
    raw_final = final["raw_alignment_matched"]
    quadratic_final = final["observation_driven"]
    uniform_final = final["uniform_matched"]
    random_final = final["random_matched"]
    birth_supported = all(
        float(raw_final[metric]) < float(other[metric])
        for metric in ("image_loss", "symmetric_chamfer")
        for other in (uniform_final, random_final)
    )
    raw_best = all(
        float(raw_final[metric]) < float(other[metric])
        for metric in ("image_loss", "symmetric_chamfer")
        for other in (quadratic_final, uniform_final, random_final)
    )
    quadratic_best = all(
        float(quadratic_final[metric]) < float(other[metric])
        for metric in ("image_loss", "symmetric_chamfer")
        for other in (raw_final, uniform_final, random_final)
    )
    compute_efficient = float(raw["total_seconds"]) <= float(
        uniform["total_seconds"]
    )
    auc = {
        item["method"]: {
            "image_loss": _trajectory_auc(item["rows"], "image_loss"),
            "symmetric_chamfer": _trajectory_auc(
                item["rows"], "symmetric_chamfer"
            ),
        }
        for item in trajectories
    }
    report = {
        "finite_difference_validation": finite_difference,
        "quadratic": quadratic,
        "raw": raw,
        "uniform": uniform,
        "random": random,
        "cold_fixed": cold,
        "warm_fixed": warm,
        "final_comparison": final,
        "trajectory_auc": auc,
        "verdicts": {
            "BIRTH_SUPPORTED": birth_supported,
            "OBSERVATION_DRIVEN_BIRTH_SUPPORTED": birth_supported,
            "RAW_ALIGNMENT_BEST_SUPPORTED": raw_best,
            "QUADRATIC_BEST_SUPPORTED": quadratic_best,
            "QUADRATIC_BEST_NOT_SUPPORTED": not quadratic_best,
            "COMPUTE_EFFICIENCY_SUPPORTED": compute_efficient,
            "COMPUTE_EFFICIENCY_NOT_SUPPORTED": not compute_efficient,
        },
        "view_ids_in_montage": list(view_ids),
    }
    rows = [
        row for trajectory in trajectories for row in trajectory["rows"]
    ]
    rows.extend(cold["rows"])
    rows.extend(warm["rows"])
    return report, rows


def run_high_bandwidth_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
    config: HighBandwidthConfig | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    config = config or HighBandwidthConfig()
    started = time.perf_counter()
    prepared = prepare_stanford_bunny(
        mesh_path, build_surface_scaffold=False
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    factor_progress_path = (
        artifact_directory / ".v08_high_bandwidth_factor_progress.json"
    )
    progress_signature = _json_ready(asdict(config))
    cached_profiles: dict[str, dict[str, object]] = {}
    cached_factors: dict[str, dict[str, object]] = {}
    if factor_progress_path.exists():
        saved = json.loads(factor_progress_path.read_text())
        if saved.get("signature") == progress_signature:
            cached_profiles = dict(saved.get("profiles", {}))
            cached_factors = dict(saved.get("factors", {}))

    def save_factor_progress(stage: str) -> None:
        factor_progress_path.write_text(
            json.dumps(
                _json_ready(
                    {
                        "version": "0.8.0",
                        "stage": stage,
                        "signature": progress_signature,
                        "profiles": cached_profiles,
                        "factors": cached_factors,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    main_config = _corrected_config(
        config, config.high_emitters, config.high_resolution, main=True
    )
    context, profile = _profile_context(
        prepared, main_config, "m_level_photons_full_hd"
    )
    cached_profiles["m_level_photons_full_hd"] = profile
    save_factor_progress("main_profile")
    progress_path = artifact_directory / ".v08_high_bandwidth_progress.json"
    state_path = artifact_directory / ".v08_high_bandwidth_states.pt"
    main, main_rows = _main_comparison(
        context,
        figure_directory,
        render_directory,
        progress_path,
        state_path,
    )
    checkpoint = {
        "version": "0.8.0",
        "status": "MAIN_COMPLETE_FACTOR_ABLATION_PENDING",
        "environment": cuda_environment(),
        "configuration": asdict(config),
        "main_profile": profile,
        "main": main,
    }
    (artifact_directory / "v08_high_bandwidth_main_checkpoint.json").write_text(
        json.dumps(_json_ready(checkpoint), indent=2, sort_keys=True) + "\n"
    )
    condition = "1080p_m_level_photons"
    if condition not in cached_factors:
        cached_factors[condition] = _one_shot_factor_cell(
            context, condition, config.ablation_target_k
        )
        save_factor_progress(condition)
    camera_atlas = context.visibility_report["camera_atlas"]
    visibility = context.visibility_report
    del context
    torch.cuda.empty_cache()

    condition = "1080p_low_photons"
    if (
        condition not in cached_factors
        or "low_photons_full_hd" not in cached_profiles
    ):
        low_high_config = _corrected_config(
            config, config.low_emitters, config.high_resolution, main=False
        )
        context, profile = _profile_context(
            prepared, low_high_config, "low_photons_full_hd"
        )
        cached_profiles["low_photons_full_hd"] = profile
        if condition not in cached_factors:
            cached_factors[condition] = _one_shot_factor_cell(
                context, condition, config.ablation_target_k
            )
        save_factor_progress(condition)
        del context
        torch.cuda.empty_cache()

    if "medium_photons_full_hd" not in cached_profiles:
        medium_config = _corrected_config(
            config, config.medium_emitters, config.high_resolution, main=False
        )
        context, profile = _profile_context(
            prepared, medium_config, "medium_photons_full_hd"
        )
        cached_profiles["medium_photons_full_hd"] = profile
        save_factor_progress("medium_photons_full_hd")
        del context
        torch.cuda.empty_cache()

    for emitters, photon_label in (
        (config.low_emitters, "low_photons"),
        (config.high_emitters, "m_level_photons"),
    ):
        condition = f"256_{photon_label}"
        if condition in cached_factors:
            continue
        factor_config = _corrected_config(
            config, emitters, config.low_resolution, main=False
        )
        context, _ = _profile_context(
            prepared, factor_config, f"256_{photon_label}"
        )
        cached_factors[condition] = _one_shot_factor_cell(
            context,
            condition,
            config.ablation_target_k,
        )
        save_factor_progress(condition)
        del context
        torch.cuda.empty_cache()

    factor_order = {
        "256_low_photons": 0,
        "256_m_level_photons": 1,
        "1080p_low_photons": 2,
        "1080p_m_level_photons": 3,
    }
    factors = sorted(
        cached_factors.values(),
        key=lambda item: factor_order[str(item["condition"])],
    )
    profiles = [
        cached_profiles[label]
        for label in (
            "low_photons_full_hd",
            "medium_photons_full_hd",
            "m_level_photons_full_hd",
        )
    ]
    _save_factor_plot(
        figure_directory / "v08_bandwidth_factor_ablation.png", factors
    )
    factor_rows: list[dict[str, object]] = []
    for factor in factors:
        factor_rows.extend((factor["raw"], factor["uniform"]))
        factor_rows.extend(factor["fixed_rows"])

    raw_final = main["final_comparison"]["raw_alignment_matched"]
    quadratic_final = main["final_comparison"]["observation_driven"]
    high_birth_supported = bool(
        main["verdicts"]["BIRTH_SUPPORTED"]
        and main["finite_difference_validation"]["passed"]
        and int(raw_final["active_dofs"]) >= 1024
    )
    factorial_effects = _factorial_effects(factors)
    raw_geometry_effects = factorial_effects["methods"]["raw"][
        "symmetric_chamfer"
    ]
    photon_geometry_improved = (
        float(raw_geometry_effects["photon_effect_at_256"]) < 0.0
        and float(raw_geometry_effects["photon_effect_at_1080p"]) < 0.0
    )
    resolution_geometry_improved = (
        float(raw_geometry_effects["resolution_effect_at_low_photons"])
        < 0.0
        and float(
            raw_geometry_effects["resolution_effect_at_m_level_photons"]
        )
        < 0.0
    )
    report: dict[str, Any] = {
        "version": "0.8.0",
        "phase": "high_bandwidth_corrected_forward_rgb_birth",
        "environment": cuda_environment(),
        "configuration": asdict(config),
        "exact_main_setup": {
            "resolution": list(config.high_resolution),
            "views": config.views,
            "emitters": config.high_emitters,
            "packets_per_emitter": config.views,
            "total_attempted_packets": config.high_emitters
            * config.views,
            "higher_photon_level_decision": (
                "131072 emitters was not attempted: the 65536-emitter main "
                "optimization already reached roughly 12.3 GiB allocated "
                "and 14.0 GiB process-resident on a 16 GiB GPU"
            ),
            "appearance": "C: no attenuation, normal/cosine lighting",
            "candidate_dictionary": config.main_dictionary_count,
            "candidate_capacity_rule": "p_max = 100 * current residents",
            "batch_rule": (
                "5% score tail, 95% cumulative predicted-gain saturation, "
                "spatial exclusion, then explicit Gram coupling gates"
            ),
            "stopping_rule": (
                "three consecutive rounds jointly below RGB relative-gain "
                "and geometry-gain thresholds, after at least K=1024; or "
                "numerical/runtime/VRAM/dictionary stop"
            ),
            "camera_construction": camera_atlas,
        },
        "bandwidth_ladder_profiles": profiles,
        "main": main,
        "bandwidth_factor_ablation": factors,
        "bandwidth_factor_effects": factorial_effects,
        "visibility": visibility,
        "final_verdicts": {
            "HIGH_BANDWIDTH_BIRTH_SUPPORTED": high_birth_supported,
            **main["verdicts"],
            "OBSERVATION_BANDWIDTH_LIMIT_PARTIALLY_REDUCED": (
                photon_geometry_improved and not resolution_geometry_improved
            ),
            "PHOTON_DENSITY_GEOMETRY_EFFECT_SUPPORTED": (
                photon_geometry_improved
            ),
            "RESOLUTION_GEOMETRY_EFFECT_SUPPORTED": (
                resolution_geometry_improved
            ),
            "RESOLUTION_GEOMETRY_EFFECT_NOT_SUPPORTED": (
                not resolution_geometry_improved
            ),
            "RAW_ALIGNMENT_REMAINS_BEST": main["verdicts"][
                "RAW_ALIGNMENT_BEST_SUPPORTED"
            ],
        },
        "scientific_interpretation": {
            "raw_minus_quadratic_final_rgb_loss": float(
                raw_final["image_loss"]
            )
            - float(quadratic_final["image_loss"]),
            "raw_minus_quadratic_final_chamfer": float(
                raw_final["symmetric_chamfer"]
            )
            - float(quadratic_final["symmetric_chamfer"]),
            "photon_density_geometry_improved_at_both_resolutions": (
                photon_geometry_improved
            ),
            "resolution_geometry_improved_at_both_photon_budgets": (
                resolution_geometry_improved
            ),
            "retained_events_per_view_pixel": (
                float(visibility["base"]["retained"])
                / (
                    config.views
                    * config.high_resolution[0]
                    * config.high_resolution[1]
                )
            ),
            "dominant_bottleneck": (
                "optimization"
                if float(main["quadratic"]["cumulative_optimization_seconds"])
                >= float(main["quadratic"]["cumulative_scoring_seconds"])
                else "candidate scoring"
            ),
            "quality_advantage": bool(main["verdicts"]["BIRTH_SUPPORTED"]),
            "dof_efficiency": (
                float(main["trajectory_auc"]["raw_alignment_matched"][
                    "symmetric_chamfer"
                ])
                < float(main["trajectory_auc"]["uniform_matched"][
                    "symmetric_chamfer"
                ])
            ),
            "compute_efficiency": bool(
                main["verdicts"]["COMPUTE_EFFICIENCY_SUPPORTED"]
            ),
        },
        "limitations": [
            (
                "The observations remain controlled synthetic calibrated "
                "Bunny images, not real photographs."
            ),
            (
                "The 20 direct Fibonacci packet directions are also the "
                "detector directions; this preserves shared scene-centric "
                "transport but does not test arbitrary off-atlas cameras."
            ),
            (
                "The finite 8192-candidate dictionary is a safety envelope "
                "and may become the stop if scientific flattening is not "
                "reached."
            ),
            (
                "The optional 2.62M-attempt 131072-emitter level was not run: "
                "the 1.31M-attempt optimization already approached the local "
                "16 GiB GPU's safe memory headroom."
            ),
            "Visibility and support topology are frozen inside local Jacobian cells.",
            (
                "The 2x2 factor study uses a simplified one-shot K=1024 "
                "raw/uniform comparison rather than rerunning every dynamic "
                "policy."
            ),
        ],
        "artifacts": {
            "json": str(artifact_directory / "v08_high_bandwidth.json"),
            "csv": str(artifact_directory / "v08_high_bandwidth.csv"),
            "quality_curves": str(
                figure_directory / "v08_high_bandwidth_quality_curves.png"
            ),
            "predictor": str(
                figure_directory
                / "v08_high_bandwidth_predictor_calibration.png"
            ),
            "batch": str(
                figure_directory / "v08_high_bandwidth_batch_trajectory.png"
            ),
            "geography": str(
                figure_directory / "v08_high_bandwidth_birth_geography.png"
            ),
            "factor": str(
                figure_directory / "v08_bandwidth_factor_ablation.png"
            ),
            "multiview": str(
                render_directory / "v08_high_bandwidth_multiview.png"
            ),
        },
        "total_experiment_seconds": time.perf_counter() - started,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v08_high_bandwidth.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(
        artifact_directory / "v08_high_bandwidth.csv",
        main_rows + factor_rows + profiles,
    )
    # Successful completion makes recovery-only files redundant.
    progress_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    factor_progress_path.unlink(missing_ok=True)
    (
        artifact_directory / "v08_high_bandwidth_main_checkpoint.json"
    ).unlink(missing_ok=True)
    return report
