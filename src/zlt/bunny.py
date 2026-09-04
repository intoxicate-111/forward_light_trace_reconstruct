"""Controlled Stanford Bunny transfer experiment for sequential DoF birth."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .benchmark import BenchmarkGeometry, cuda_environment
from .birth import _image_loss, _images
from .fields import LocalBasisField, unit_normals
from .jacobian import _photon_batch, deterministic_directions
from .locality import hierarchical_surface_points
from .mesh_field import PreparedBunny, mesh_surface_samples, prepare_stanford_bunny
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    camera_pose,
    multiview_cameras,
    project_camera,
)
from .sequential import (
    RepeatedBirthConfig,
    SequentialContext,
    _correlation,
    _hierarchical_layout,
    _multiscale_support,
    _normalized_auc,
    _run_fixed,
    _run_trajectory,
)
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor


@dataclass(frozen=True)
class BunnyExperimentConfig:
    phase: str = "phase1"
    master_count: int = 1024
    initial_count: int = 32
    budget: int = 256
    checkpoints: tuple[int, ...] = (32, 64, 128, 192, 256)
    fixed_levels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    views: int = 8
    resolution: int = 256
    emitters: int = 8192
    packets_per_emitter: int = 8
    random_seeds: tuple[int, ...] = (101, 211, 307)
    base_support_radius: float = 0.45
    coefficient_limit: float = 0.03
    root_samples: int = 48
    root_bisection_steps: int = 24
    initial_optimization_steps: int = 6
    post_birth_steps: int = 1
    fixed_optimization_steps: int = 16
    oracle_subset_size: int = 8
    oracle_evaluations: int = 5
    evaluation_samples: int = 16384


def phase2_config() -> BunnyExperimentConfig:
    return BunnyExperimentConfig(
        phase="phase2",
        master_count=2048,
        budget=512,
        checkpoints=(32, 64, 128, 256, 384, 512),
        fixed_levels=(64, 128, 256, 512, 1024, 2048),
        views=20,
        resolution=512,
        emitters=16384,
        random_seeds=(101,),
        base_support_radius=0.45,
        oracle_subset_size=0,
    )


class BunnyGeometryEvaluator:
    """Fixed original-mesh and deterministic sampled-surface metrics."""

    def __init__(
        self,
        prepared: PreparedBunny,
        reference_points: Tensor,
        reference_normals: Tensor,
        sample_count: int,
    ) -> None:
        import open3d as o3d
        from scipy.spatial import cKDTree

        self.o3d = o3d
        self.cKDTree = cKDTree
        vertices = prepared.original_vertices.astype(np.float64)
        faces = prepared.original_faces.astype(np.int32)
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        self.face_normals = cross / np.maximum(
            np.linalg.norm(cross, axis=1, keepdims=True), 1e-30
        )
        self.gt_samples, _ = mesh_surface_samples(vertices, faces, sample_count)
        self.gt_tree = cKDTree(self.gt_samples)
        self.reference_normals = reference_normals.detach().cpu().numpy()
        points = reference_points.detach().cpu().numpy()
        self.regions = {
            "ear": points[:, 1] > 0.45,
            "head": (points[:, 1] <= 0.45)
            & (points[:, 1] > -0.05)
            & (points[:, 0] < -0.35),
            "leg": points[:, 1] < -0.70,
        }
        self.regions["torso"] = ~(
            self.regions["ear"] | self.regions["head"] | self.regions["leg"]
        )

    def __call__(self, points: Tensor) -> dict[str, float]:
        current = points.detach().cpu().numpy().astype(np.float32)
        closest = self.scene.compute_closest_points(self.o3d.core.Tensor(current))
        target_points = closest["points"].numpy().astype(np.float64)
        primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
        current64 = current.astype(np.float64)
        point_to_surface = np.linalg.norm(current64 - target_points, axis=1)
        reverse = self.cKDTree(current64).query(self.gt_samples, workers=1)[0]
        target_normals = self.face_normals[primitive_ids]
        consistency = np.abs((self.reference_normals * target_normals).sum(axis=1))
        result = {
            "symmetric_chamfer": 0.5
            * (float(point_to_surface.mean()) + float(reverse.mean())),
            "point_to_surface_mean": float(point_to_surface.mean()),
            "point_to_surface_p95": float(np.quantile(point_to_surface, 0.95)),
            "surface_rms": float(np.sqrt(np.mean(point_to_surface**2))),
            "normal_consistency": float(consistency.mean()),
            "normal_error": float(1.0 - consistency.mean()),
        }
        for name, mask in self.regions.items():
            result[f"regional_{name}_p2s_mean"] = float(
                point_to_surface[mask].mean()
            )
        return result


def _normal_line_roots(
    field: object,
    reference_points: Tensor,
    reference_normals: Tensor,
    *,
    maximum_offset: float = 0.08,
) -> tuple[Tensor, Tensor, Tensor]:
    offsets = torch.linspace(
        -maximum_offset,
        maximum_offset,
        65,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    candidates = (
        reference_points[:, None]
        + offsets[None, :, None] * reference_normals[:, None]
    )
    values = field.value(candidates)
    changes = values[:, :-1] * values[:, 1:] <= 0.0
    midpoints = 0.5 * (offsets[:-1] + offsets[1:])
    score = torch.where(
        changes,
        midpoints.abs()[None],
        torch.full_like(changes, torch.inf, dtype=reference_points.dtype),
    )
    bracket = score.argmin(dim=1)
    success = changes.any(dim=1)
    rows = torch.arange(reference_points.shape[0], device=reference_points.device)
    low = offsets[bracket]
    high = offsets[bracket + 1]
    low_value = values[rows, bracket]
    for _ in range(30):
        middle = 0.5 * (low + high)
        middle_value = field.value(
            reference_points + middle[:, None] * reference_normals
        )
        left = low_value * middle_value <= 0.0
        high = torch.where(left, middle, high)
        low = torch.where(left, low, middle)
        low_value = torch.where(left, low_value, middle_value)
    displacement = 0.5 * (low + high)
    points = reference_points + displacement[:, None] * reference_normals
    residual = field.value(points).abs()
    success &= residual < 1e-7
    return points, success, displacement


def _build_bunny_context(
    prepared: PreparedBunny, config: BunnyExperimentConfig
) -> tuple[SequentialContext, dict[str, object], dict[str, object]]:
    device = torch.device("cuda")
    base = prepared.base_field.to(device)
    gt = prepared.gt_field.to(device)
    repeated = RepeatedBirthConfig(
        master_count=config.master_count,
        initial_count=config.initial_count,
        budget=config.budget,
        checkpoints=config.checkpoints,
        target_regimes=("bunny",),
        target_seeds=(0,),
        random_seeds=config.random_seeds,
        views=config.views,
        resolution=config.resolution,
        emitters_per_master_basis=max(1, config.emitters // config.master_count),
        packets_per_emitter=config.packets_per_emitter,
        base_support_radius=config.base_support_radius,
        coefficient_limit=config.coefficient_limit,
        root_samples=config.root_samples,
        root_bisection_steps=config.root_bisection_steps,
        deformation_iterations=20,
        initial_optimization_steps=config.initial_optimization_steps,
        post_birth_steps=config.post_birth_steps,
        fixed_optimization_steps=config.fixed_optimization_steps,
        run_oracle=False,
        oracle_subset_size=config.oracle_subset_size,
        oracle_checkpoints=tuple(
            level for level in (32, 64, 128, 256) if level <= config.budget
        ),
        oracle_evaluations=config.oracle_evaluations,
    )
    layout, levels = _hierarchical_layout(
        base,
        config.master_count,
        config.initial_count,
        config.base_support_radius,
        device,
    )
    reference_points = hierarchical_surface_points(base, config.emitters, device)
    reference_normals = unit_normals(base, reference_points)
    colors = base.color(reference_points)
    support = _multiscale_support(
        reference_points, layout, levels, repeated.support_margin
    )
    directions = deterministic_directions(
        reference_normals, config.packets_per_emitter, 128.0
    )
    photons = _photon_batch(
        reference_points, directions, colors, config.packets_per_emitter
    )
    maximum_times = torch.full(
        (photons.count,), 3.0, dtype=torch.float64, device=device
    )
    scene_started = time.perf_counter()
    surface_hits, surface_times = first_zero_set_intersections(
        base,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
        chunk_size=8192,
    )
    torch.cuda.synchronize()
    base_scene_seconds = time.perf_counter() - scene_started
    state = SceneTransportState(
        reference_points,
        reference_normals,
        photons,
        directions,
        surface_hits,
        surface_times,
        torch.ones(config.emitters, dtype=torch.bool, device=device),
    )
    cameras = multiview_cameras(
        "bunny", (config.resolution, config.resolution), device, count=config.views
    )
    initial_field = LocalBasisField(
        base,
        layout.centers[: config.initial_count],
        layout.radii[: config.initial_count],
        torch.zeros(config.initial_count, dtype=torch.float64, device=device),
    )
    geometry = BenchmarkGeometry(
        initial_field, reference_points, reference_normals, colors
    )
    view_config = MultiviewConfig(
        resolution=(config.resolution, config.resolution),
        emitters=config.emitters,
        packets_per_emitter=config.packets_per_emitter,
        parameter_count=config.master_count,
        cone_power=128.0,
        root_samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
    )
    detector_started = time.perf_counter()
    base_transports = [
        project_camera(geometry, camera, state, view_config) for camera in cameras
    ]
    torch.cuda.synchronize()
    base_detector_seconds = time.perf_counter() - detector_started

    target_points, target_success, target_displacement = _normal_line_roots(
        gt, reference_points, reference_normals
    )
    target_normals = unit_normals(gt, target_points)
    target_photons = _photon_batch(
        target_points, directions, colors, config.packets_per_emitter
    )
    target_started = time.perf_counter()
    target_hits, target_times = first_zero_set_intersections(
        gt,
        target_photons.origins,
        target_photons.directions,
        maximum_times,
        samples=config.root_samples,
        bisection_steps=config.root_bisection_steps,
        chunk_size=8192,
    )
    target_state = SceneTransportState(
        target_points,
        target_normals,
        target_photons,
        directions,
        target_hits,
        target_times,
        target_success,
    )
    target_transports = [
        project_camera(geometry, camera, target_state, view_config)
        for camera in cameras
    ]
    torch.cuda.synchronize()
    target_transport_seconds = time.perf_counter() - target_started
    target_images = [transport.image.reshape(-1) for transport in target_transports]

    center_normals = unit_normals(base, layout.centers)
    _, center_success, center_displacement = _normal_line_roots(
        gt, layout.centers, center_normals
    )
    detail_threshold = torch.quantile(center_displacement.abs(), 0.75)
    detail_ids = torch.nonzero(
        center_displacement.abs() >= detail_threshold, as_tuple=False
    ).flatten()
    top_ten_threshold = torch.quantile(center_displacement.abs(), 0.90)
    detail_ids_top_ten = torch.nonzero(
        center_displacement.abs() >= top_ten_threshold, as_tuple=False
    ).flatten()
    evaluator = BunnyGeometryEvaluator(
        prepared, reference_points, reference_normals, config.evaluation_samples
    )
    context = SequentialContext(
        repeated,
        base,
        layout,
        support,
        levels,
        reference_points,
        reference_normals,
        colors,
        [transport.cell for transport in base_transports],
        state,
        geometry,
        evaluator,
    )
    target = {
        "regime": "bunny",
        "seed": 0,
        "points": target_points,
        "images": target_images,
        "transport_cells": target_transports,
        "detail_ids": detail_ids,
        "detail_ids_top_ten": detail_ids_top_ten,
        "rms": float(torch.sqrt((target_displacement**2).mean())),
        "initial_loss": _image_loss(_images(context.cells, reference_points), target_images),
    }
    changed_states = []
    changed_owners = []
    per_view = []
    for base_transport, target_transport in zip(base_transports, target_transports):
        changed_states.append(
            float(
                (base_transport.cell.photon_state != target_transport.cell.photon_state)
                .double()
                .mean()
            )
        )
        occupied = (base_transport.cell.owner_map >= 0) | (
            target_transport.cell.owner_map >= 0
        )
        changed_owners.append(
            float(
                (
                    base_transport.cell.owner_map[occupied]
                    != target_transport.cell.owner_map[occupied]
                )
                .double()
                .mean()
            )
            if bool(occupied.any())
            else 0.0
        )
        per_view.append(
            {
                "base_camera_hits": int(base_transport.intersection.valid.sum()),
                "target_camera_hits": int(target_transport.intersection.valid.sum()),
                "base_absorbed": int(base_transport.absorbed_detector_mask.sum()),
                "target_absorbed": int(target_transport.absorbed_detector_mask.sum()),
                "base_transport_nnz": base_transport.transport._nnz(),
                "target_transport_nnz": target_transport.transport._nnz(),
            }
        )
    diagnostics = {
        "camera_poses": [camera_pose(camera) for camera in cameras],
        "base_scene_seconds": base_scene_seconds,
        "base_detector_seconds": base_detector_seconds,
        "target_transport_seconds": target_transport_seconds,
        "base_surface_hits": int(surface_hits.sum()),
        "target_surface_hits": int(target_hits.sum()),
        "target_normal_line_failures": int((~target_success).sum()),
        "candidate_normal_line_failures": int((~center_success).sum()),
        "target_displacement_rms": target["rms"],
        "target_displacement_p95": float(
            torch.quantile(target_displacement.abs(), 0.95)
        ),
        "target_displacement_max": float(target_displacement.abs().max()),
        "photon_state_event_fraction_mean": statistics.mean(changed_states),
        "owner_event_fraction_mean_on_occupied_union": statistics.mean(
            changed_owners
        ),
        "views": per_view,
    }
    return context, target, diagnostics


def _window_correlations(trajectory: dict[str, object]) -> dict[str, object]:
    births = trajectory["births"]
    result: dict[str, object] = {}
    for name, low, high in (
        ("early", 0.0, 1 / 3),
        ("middle", 1 / 3, 2 / 3),
        ("late", 2 / 3, 1.0),
    ):
        start = int(low * len(births))
        stop = len(births) if high == 1.0 else int(high * len(births))
        rows = births[start:stop]
        realized = torch.tensor(
            [float(row["realized_gain"]) for row in rows], dtype=torch.float64
        )
        result[name] = {
            "birth_count": len(rows),
            "quadratic_spearman": _correlation(
                torch.tensor(
                    [float(row["predicted_quadratic"]) for row in rows],
                    dtype=torch.float64,
                ),
                realized,
                True,
            ),
            "quadratic_pearson": _correlation(
                torch.tensor(
                    [float(row["predicted_quadratic"]) for row in rows],
                    dtype=torch.float64,
                ),
                realized,
                False,
            ),
            "raw_spearman": _correlation(
                torch.tensor(
                    [float(row["predicted_raw"]) for row in rows],
                    dtype=torch.float64,
                ),
                realized,
                True,
            ),
        }
    return result


def _quality_thresholds(
    trajectories: list[dict[str, object]], fixed_rows: list[dict[str, object]]
) -> dict[str, object]:
    fixed = sorted(fixed_rows, key=lambda row: int(row["active_k"]))
    start = float(fixed[0]["symmetric_chamfer"])
    full = float(fixed[-1]["symmetric_chamfer"])
    result: dict[str, object] = {}
    for percent in (90, 95, 99):
        threshold = start - percent / 100.0 * (start - full)
        methods: dict[str, list[int | None]] = {}
        for trajectory in trajectories:
            reached = next(
                (
                    int(row["active_k"])
                    for row in trajectory["checkpoints"]
                    if float(row["symmetric_chamfer"]) <= threshold
                ),
                None,
            )
            methods.setdefault(str(trajectory["method"]), []).append(reached)
        fixed_k = next(
            (
                int(row["active_k"])
                for row in fixed
                if float(row["symmetric_chamfer"]) <= threshold
            ),
            None,
        )
        result[str(percent)] = {
            "chamfer_threshold": threshold,
            "fixed_k": fixed_k,
            "method_k": methods,
        }
    return {
        "initial_fixed_chamfer": start,
        "full_fixed_chamfer": full,
        "thresholds": result,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean_curve(
    rows: list[dict[str, object]], method: str, key: str
) -> tuple[list[float], list[float]]:
    levels = sorted({int(row["active_k"]) for row in rows if row["method"] == method})
    return (
        [float(level) for level in levels],
        [
            statistics.mean(
                float(row[key])
                for row in rows
                if row["method"] == method and int(row["active_k"]) == level
            )
            for level in levels
        ],
    )


def _write_figures(
    directory: Path,
    phase: str,
    prepared: PreparedBunny,
    context: SequentialContext,
    trajectories: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
) -> list[str]:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"v03d_bunny_{phase}"
    paths: list[str] = []
    all_rows = [row for item in trajectories for row in item["checkpoints"]]
    combined = all_rows + fixed_rows
    methods = ("quadratic", "raw", "uniform", "random", "fixed_space")
    if phase == "phase1":
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
        gt = prepared.original_vertices[::8]
        base = context.base.surface_vertices.detach().cpu().numpy()[::8]
        for axis, points, title in zip(axes, (gt, base), ("GT", "coarse base")):
            axis.scatter(points[:, 0], points[:, 1], s=0.3)
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_axis_off()
        figure.tight_layout()
        path = directory / f"{prefix}_gt_vs_base.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    for suffix, key, ylabel in (
        ("geometry_vs_dofs", "symmetric_chamfer", "symmetric Chamfer"),
        ("image_vs_dofs", "image_loss", "image loss"),
    ):
        figure, axis = plt.subplots(figsize=(7.0, 4.6))
        for method in methods:
            x, y = _mean_curve(combined, method, key)
            if x:
                axis.plot(x, y, marker="o", label=method)
        axis.set_xlabel("active geometry DoFs")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / f"{prefix}_{suffix}.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for method in methods:
        method_rows = [row for row in combined if row["method"] == method]
        levels = sorted({int(row["active_k"]) for row in method_rows})
        x = [
            statistics.mean(
                float(row["total_runtime_seconds"])
                for row in method_rows
                if int(row["active_k"]) == level
            )
            for level in levels
        ]
        y = [
            statistics.mean(
                float(row["symmetric_chamfer"])
                for row in method_rows
                if int(row["active_k"]) == level
            )
            for level in levels
        ]
        if x:
            axis.plot(x, y, marker="o", label=method)
    axis.set_xlabel("cumulative wall-clock time (s)")
    axis.set_ylabel("symmetric Chamfer")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / f"{prefix}_geometry_vs_time.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))

    representative = next(
        item for item in trajectories if item["method"] == "quadratic"
    )
    levels = [
        level
        for level in (32, 64, 128, 192, 256, 384, 512)
        if level <= context.config.budget
    ]
    figure, axes = plt.subplots(1, len(levels), figsize=(3.0 * len(levels), 3.4))
    base_points = context.reference_points.detach().cpu().numpy()[::16]
    births = representative["births"]
    for axis, active_k in zip(np.atleast_1d(axes), levels):
        count = max(0, active_k - context.config.initial_count)
        ids = [int(row["candidate_id"]) for row in births[:count]]
        axis.scatter(base_points[:, 0], base_points[:, 1], s=0.4, alpha=0.25)
        if ids:
            selected = context.master_layout.centers[ids].detach().cpu().numpy()
            axis.scatter(selected[:, 0], selected[:, 1], s=5, c="tab:orange")
        axis.set_title(f"K={active_k}")
        axis.set_aspect("equal")
        axis.set_axis_off()
    figure.tight_layout()
    path = directory / f"{prefix}_birth_geography.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(str(path))
    if phase == "phase1":
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
        for axis, key, title in (
            (axes[0], "predicted_quadratic", "quadratic score"),
            (axes[1], "predicted_raw", "raw alignment"),
        ):
            for method in ("quadratic", "raw"):
                births = next(
                    item["births"]
                    for item in trajectories
                    if item["method"] == method
                )
                axis.scatter(
                    [row[key] for row in births],
                    [row["realized_gain"] for row in births],
                    s=8,
                    alpha=0.4,
                    label=method,
                )
            axis.set_xscale("symlog", linthresh=1e-10)
            axis.set_yscale("symlog", linthresh=1e-10)
            axis.set_xlabel(title)
            axis.set_ylabel("realized gain")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / f"{prefix}_predicted_vs_realized.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    return paths


def run_bunny_phase(
    prepared: PreparedBunny,
    config: BunnyExperimentConfig,
    *,
    csv_path: Path,
    json_path: Path,
    figure_directory: Path,
) -> dict[str, object]:
    context, target, transport = _build_bunny_context(prepared, config)
    initial_geometry = context.geometry_evaluator(context.reference_points)  # type: ignore[misc]
    trajectories: list[dict[str, object]] = []
    for method in ("quadratic", "raw", "uniform"):
        trajectory = _run_trajectory(context, target, method, 0)
        trajectory["window_correlations"] = _window_correlations(trajectory)
        trajectories.append(trajectory)
        print(
            json.dumps(
                {
                    "phase": config.phase,
                    "method": method,
                    "final_chamfer": trajectory["checkpoints"][-1][
                        "symmetric_chamfer"
                    ],
                    "runtime_seconds": trajectory["total_runtime_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for seed in config.random_seeds:
        trajectory = _run_trajectory(context, target, "random", seed)
        trajectory["window_correlations"] = _window_correlations(trajectory)
        trajectories.append(trajectory)
        print(
            json.dumps(
                {
                    "phase": config.phase,
                    "method": "random",
                    "selection_seed": seed,
                    "final_chamfer": trajectory["checkpoints"][-1][
                        "symmetric_chamfer"
                    ],
                    "runtime_seconds": trajectory["total_runtime_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    fixed_rows = [
        _run_fixed(context, target, level)
        for level in config.fixed_levels
        if level <= config.master_count
    ]
    summaries = []
    for trajectory in trajectories:
        rows = trajectory["checkpoints"]
        summaries.append(
            {
                "method": trajectory["method"],
                "selection_seed": trajectory["selection_seed"],
                "chamfer_dof_auc": _normalized_auc(
                    rows, "active_k", "symmetric_chamfer"
                ),
                "chamfer_time_auc": _normalized_auc(
                    rows, "total_runtime_seconds", "symmetric_chamfer"
                ),
                "image_dof_auc": _normalized_auc(rows, "active_k", "image_loss"),
                "final_chamfer": rows[-1]["symmetric_chamfer"],
                "final_p2s_mean": rows[-1]["point_to_surface_mean"],
                "final_p2s_p95": rows[-1]["point_to_surface_p95"],
                "final_normal_error": rows[-1]["normal_error"],
                "final_image_loss": rows[-1]["image_loss"],
            }
        )
    quadratic = next(item for item in trajectories if item["method"] == "quadratic")
    uniform = next(item for item in trajectories if item["method"] == "uniform")
    random = [item for item in trajectories if item["method"] == "random"]
    quadratic_final = float(quadratic["checkpoints"][-1]["symmetric_chamfer"])
    uniform_final = float(uniform["checkpoints"][-1]["symmetric_chamfer"])
    random_final = statistics.mean(
        float(item["checkpoints"][-1]["symmetric_chamfer"]) for item in random
    )
    quadratic_auc = next(
        float(item["chamfer_dof_auc"])
        for item in summaries
        if item["method"] == "quadratic"
    )
    uniform_auc = next(
        float(item["chamfer_dof_auc"])
        for item in summaries
        if item["method"] == "uniform"
    )
    random_auc = statistics.mean(
        float(item["chamfer_dof_auc"])
        for item in summaries
        if item["method"] == "random"
    )
    final_checkpoint = quadratic["checkpoints"][-1]
    numerical_rows = [
        row for trajectory in trajectories for row in trajectory["checkpoints"]
    ] + fixed_rows
    maximum_root_failures = max(int(row["root_failures"]) for row in numerical_rows)
    maximum_cg_failures = max(int(row["cg_failures"]) for row in numerical_rows)
    maximum_line_search_failures = max(
        int(row["line_search_failures"]) for row in numerical_rows
    )
    maximum_nonfinite_scores = max(
        int(row["nonfinite_candidate_score_count"] or 0)
        for row in numerical_rows
    )
    minimum_denominator = min(
        float(row["minimum_denominator_magnitude"])
        for row in numerical_rows
    )
    maximum_birth_jump = max(
        float(trajectory["maximum_birth_geometry_jump"])
        for trajectory in trajectories
    )
    normal_line_failures = int(transport["target_normal_line_failures"]) + int(
        transport["candidate_normal_line_failures"]
    )
    invalid_surfaces = int(not prepared.gt_field.surface.is_watertight) + int(
        not prepared.base_field.surface.is_watertight
    )
    degenerate_gradients = int(minimum_denominator <= 1e-8)
    numerical_valid = (
        maximum_root_failures == 0
        and maximum_birth_jump < 1e-12
        and maximum_cg_failures == 0
        and maximum_nonfinite_scores == 0
        and normal_line_failures == 0
        and invalid_surfaces == 0
        and degenerate_gradients == 0
    )
    geometry_improved = (
        float(final_checkpoint["symmetric_chamfer"])
        < float(quadratic["checkpoints"][0]["symmetric_chamfer"])
    )
    predictor_positive = float(
        quadratic["predictor_correlations"]["quadratic_spearman"]
    ) > 0.2
    checkpoint_spearman = [
        float(item["quadratic_spearman"])
        for item in quadratic["checkpoint_candidate_oracles"]
    ]
    visibility_adequate = (
        min(
            float(row["responsive_candidate_fraction"])
            for row in quadratic["checkpoints"]
        )
        >= 0.8
    )
    supported = (
        numerical_valid
        and quadratic_final < uniform_final
        and quadratic_final < random_final
        and quadratic_auc < uniform_auc
        and quadratic_auc < random_auc
        and geometry_improved
        and predictor_positive
        and visibility_adequate
    )
    continuation_gate = supported and quadratic_final <= 0.9 * uniform_final
    rows = [row for item in trajectories for row in item["checkpoints"]] + fixed_rows
    _write_csv(csv_path, rows)
    figures = _write_figures(
        figure_directory,
        config.phase,
        prepared,
        context,
        trajectories,
        fixed_rows,
    )
    report: dict[str, object] = {
        "experiment": f"v0.3d Stanford Bunny {config.phase}",
        "verdict": (
            "RABBIT_CONTROLLED_BIRTH_SUPPORTED"
            if config.phase == "phase1" and supported
            else "RABBIT_CONTROLLED_BIRTH_NOT_SUPPORTED"
            if config.phase == "phase1"
            else "RABBIT_SCALED_BIRTH_SUPPORTED"
            if supported
            else "RABBIT_SCALED_BIRTH_NOT_SUPPORTED"
        ),
        "phase2_continuation_gate": continuation_gate,
        "failed_gates": {
            "phase1": [] if supported else ["one_or_more_phase1_conditions"],
            "phase2_continuation": []
            if continuation_gate
            else ["quadratic_vs_uniform_improvement_at_least_0.10"],
        },
        "gate_values": {
            "numerical_valid": numerical_valid,
            "quadratic_final_chamfer": quadratic_final,
            "uniform_final_chamfer": uniform_final,
            "random_final_chamfer_mean": random_final,
            "quadratic_vs_uniform_improvement_fraction": 1.0
            - quadratic_final / uniform_final,
            "quadratic_chamfer_dof_auc": quadratic_auc,
            "uniform_chamfer_dof_auc": uniform_auc,
            "random_chamfer_dof_auc_mean": random_auc,
            "geometry_improved": geometry_improved,
            "predictor_positive": predictor_positive,
            "selected_quadratic_spearman": float(
                quadratic["predictor_correlations"]["quadratic_spearman"]
            ),
            "checkpoint_subset_quadratic_spearman": checkpoint_spearman,
            "visibility_adequate": visibility_adequate,
            "maximum_root_failures": maximum_root_failures,
            "maximum_birth_geometry_jump": maximum_birth_jump,
            "maximum_cg_failures": maximum_cg_failures,
            "maximum_line_search_failures": maximum_line_search_failures,
            "maximum_nonfinite_candidate_scores": maximum_nonfinite_scores,
            "minimum_denominator_magnitude": minimum_denominator,
            "normal_line_failures": normal_line_failures,
            "invalid_surface_count": invalid_surfaces,
            "degenerate_gradient_count": degenerate_gradients,
        },
        "environment": cuda_environment(),
        "configuration": {
            **config.__dict__,
            "checkpoints": list(config.checkpoints),
            "fixed_levels": list(config.fixed_levels),
            "random_seeds": list(config.random_seeds),
            "optimizer": (
                "stateless damped Gauss-Newton/CG with shared line search; "
                "one zero coefficient appended per birth"
            ),
            "selection_uses_target_geometry": False,
        },
        "bunny": prepared.metadata,
        "base_initial_geometry": initial_geometry,
        "target": {
            "normal_line_rms": target["rms"],
            "initial_image_loss": target["initial_loss"],
            "high_error_candidate_definition": (
                "top quartile of absolute base-to-GT normal-line displacement; "
                "evaluation only"
            ),
            "high_error_candidate_count": int(target["detail_ids"].numel()),
        },
        "transport": transport,
        "trajectories": trajectories,
        "fixed_space_rows": fixed_rows,
        "trajectory_summaries": summaries,
        "random_statistics": {
            "runs": len(random),
            "final_chamfer_mean": random_final,
            "final_chamfer_std": statistics.pstdev(
                float(item["checkpoints"][-1]["symmetric_chamfer"])
                for item in random
            ),
        },
        "quality_thresholds": _quality_thresholds(trajectories, fixed_rows),
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "figures": figures,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_bunny_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    prepared = prepare_stanford_bunny(mesh_path)
    phase1 = run_bunny_phase(
        prepared,
        BunnyExperimentConfig(),
        csv_path=artifact_directory / "v03d_bunny_phase1.csv",
        json_path=artifact_directory / "v03d_bunny_phase1.json",
        figure_directory=figure_directory,
    )
    result: dict[str, object] = {"phase1": phase1, "phase2": "PHASE2_SKIPPED"}
    if phase1["phase2_continuation_gate"]:
        result["phase2"] = run_bunny_phase(
            prepared,
            phase2_config(),
            csv_path=artifact_directory / "v03d_bunny_phase2.csv",
            json_path=artifact_directory / "v03d_bunny_phase2.json",
            figure_directory=figure_directory,
        )
    return result
