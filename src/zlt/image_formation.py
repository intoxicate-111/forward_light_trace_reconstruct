"""Visibility-aware zero-set surface image formation.

The historical renderer maps outward photon packets to detector pixels.  That
operator is useful for transport diagnostics, but it is not a camera model: a
packet may land anywhere on the detector and its emitter color visualizes the
arrival event.  This module instead builds one camera-independent zero-set
surface state and applies camera-side projection, depth ownership, and compact
local splatting to form recognizable object images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .bandwidth import _canonical_emitters, _geometry_bundle, _sync
from .benchmark import BenchmarkGeometry, cuda_environment
from .bunny import BunnyExperimentConfig
from .fields import unit_normals
from .jacobian import _photon_batch, nested_deterministic_directions
from .mesh_field import GridZeroSetField, prepare_stanford_bunny
from .multiview import (
    MultiviewConfig,
    SceneTransportState,
    camera_pose,
    project_camera,
)
from .natural import nested_bunny_cameras
from .tracer import first_zero_set_intersections


Tensor = torch.Tensor
Appearance = Literal["constant", "normal", "position", "shaded"]
Footprint = Literal["point", "bilinear", "gaussian"]
Reduction = Literal[
    "accumulate", "dominant_weight", "nearest_depth", "hybrid_depth"
]


@dataclass(frozen=True)
class SharedSurfaceState:
    """Camera-independent samples of a zero set and their local attributes."""

    points: Tensor
    normals: Tensor
    field_lower: Tensor
    field_upper: Tensor
    build_seconds: float
    residual_max: float
    peak_allocated_mib: float


@dataclass
class FormedImage:
    """One image and its fixed visibility/splat cell."""

    image: Tensor
    mask: Tensor
    depth: Tensor
    owner: Tensor
    transport: Tensor
    projected_points: int
    visible_entries: int
    runtime_seconds: float
    peak_allocated_mib: float


@dataclass(frozen=True)
class ReferenceImage:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    normals: np.ndarray


def _tensor_digest(tensor: Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def build_shared_surface_state(
    field: GridZeroSetField,
    count: int,
    device: torch.device,
) -> SharedSurfaceState:
    """Sample and refine one deterministic surface state for every camera."""
    if count < 1:
        raise ValueError("surface sample count must be positive")
    baseline_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    samples = torch.quasirandom.SobolEngine(3, scramble=False).draw(count)
    samples = samples.to(dtype=torch.float64, device=device)
    face_ids = torch.searchsorted(
        field.face_cdf, samples[:, 0].contiguous()
    ).clamp_max(field.surface_faces.shape[0] - 1)
    triangles = field.surface_vertices[field.surface_faces[face_ids]]
    radial = torch.sqrt(samples[:, 1])
    points = (
        (1.0 - radial)[:, None] * triangles[:, 0]
        + (radial * (1.0 - samples[:, 2]))[:, None] * triangles[:, 1]
        + (radial * samples[:, 2])[:, None] * triangles[:, 2]
    )
    voxel = float(
        ((field.upper - field.lower) / (torch.tensor(field.grid.shape, device=device) - 1)).min()
    )
    for _ in range(96):
        values = field.value(points)
        gradients = field.interpolant_gradient(points)
        step = values[:, None] * gradients / (
            gradients.square().sum(dim=1, keepdim=True).clamp_min(1e-18)
        )
        step_norm = torch.linalg.vector_norm(step, dim=1, keepdim=True)
        step *= (0.5 * voxel / step_norm.clamp_min(1e-30)).clamp_max(1.0)
        points = points - step
    normals = unit_normals(field, points)
    _sync(device)
    seconds = time.perf_counter() - started
    return SharedSurfaceState(
        points,
        normals,
        field.lower,
        field.upper,
        seconds,
        float(field.value(points).abs().max()),
        (torch.cuda.max_memory_allocated(device) - baseline_allocated) / 2**20
        if device.type == "cuda"
        else 0.0,
    )


def _appearance(
    state: SharedSurfaceState,
    camera: object,
    mode: Appearance,
) -> Tensor:
    points = state.points
    normals = state.normals
    if mode == "constant":
        return torch.full_like(points, 0.72)
    if mode == "normal":
        return (0.5 * (normals + 1.0)).clamp(0.0, 1.0)
    if mode == "position":
        return (
            (points - state.field_lower)
            / (state.field_upper - state.field_lower)
        ).clamp(0.0, 1.0)
    if mode != "shaded":
        raise ValueError(f"unknown appearance: {mode}")
    light = torch.tensor(
        [0.35, 0.80, 0.48], dtype=points.dtype, device=points.device
    )
    light = light / torch.linalg.vector_norm(light)
    view = camera.center - points  # type: ignore[attr-defined]
    view = view / torch.linalg.vector_norm(view, dim=1, keepdim=True)
    diffuse = (normals @ light).abs()
    facing = (normals * view).sum(dim=1).abs()
    intensity = (0.20 + 0.58 * diffuse + 0.22 * facing).clamp(0.0, 1.0)
    warm_gray = torch.tensor(
        [0.86, 0.82, 0.74], dtype=points.dtype, device=points.device
    )
    return intensity[:, None] * warm_gray[None, :]


def _project_orthographic(
    state: SharedSurfaceState, camera: object
) -> tuple[Tensor, Tensor, Tensor]:
    points = state.points
    relative = points - camera.center  # type: ignore[attr-defined]
    depth = relative @ camera.normal  # type: ignore[attr-defined]
    horizontal = relative @ camera.right  # type: ignore[attr-defined]
    vertical = relative @ camera.up  # type: ignore[attr-defined]
    rows, columns = camera.resolution  # type: ignore[attr-defined]
    column = (horizontal / camera.width + 0.5) * columns - 0.5  # type: ignore[attr-defined]
    row = (0.5 - vertical / camera.height) * rows - 0.5  # type: ignore[attr-defined]
    sensor = torch.stack((row, column), dim=1)
    valid = (
        (depth > 0.0)
        & (row >= -1.5)
        & (row <= rows + 0.5)
        & (column >= -1.5)
        & (column <= columns + 0.5)
    )
    return sensor, depth, valid


def _footprint_entries(
    sensor: Tensor,
    valid_points: Tensor,
    resolution: tuple[int, int],
    footprint: Footprint,
) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = resolution
    point_ids = torch.nonzero(valid_points, as_tuple=False).flatten()
    selected = sensor[point_ids]
    if footprint == "point":
        pixel_row = torch.round(selected[:, 0]).to(torch.long)
        pixel_column = torch.round(selected[:, 1]).to(torch.long)
        weights = torch.ones_like(selected[:, 0])
        repeated = point_ids
    elif footprint == "bilinear":
        base = torch.floor(selected).to(torch.long)
        fraction = selected - base.to(selected.dtype)
        offsets = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            dtype=torch.long,
            device=sensor.device,
        )
        coordinates = base[:, None, :] + offsets[None, :, :]
        pixel_row = coordinates[:, :, 0].reshape(-1)
        pixel_column = coordinates[:, :, 1].reshape(-1)
        weights = torch.stack(
            (
                (1.0 - fraction[:, 0]) * (1.0 - fraction[:, 1]),
                (1.0 - fraction[:, 0]) * fraction[:, 1],
                fraction[:, 0] * (1.0 - fraction[:, 1]),
                fraction[:, 0] * fraction[:, 1],
            ),
            dim=1,
        ).reshape(-1)
        repeated = point_ids.repeat_interleave(4)
    elif footprint == "gaussian":
        center = torch.round(selected).to(torch.long)
        axis = torch.arange(-1, 2, dtype=torch.long, device=sensor.device)
        grid_row, grid_column = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((grid_row.reshape(-1), grid_column.reshape(-1)), 1)
        coordinates = center[:, None, :] + offsets[None, :, :]
        pixel_row = coordinates[:, :, 0].reshape(-1)
        pixel_column = coordinates[:, :, 1].reshape(-1)
        delta = coordinates.to(sensor.dtype) - selected[:, None, :]
        weights = torch.exp(-0.5 * (delta / 0.72).square().sum(dim=2)).reshape(-1)
        repeated = point_ids.repeat_interleave(9)
    else:
        raise ValueError(f"unknown footprint: {footprint}")
    inside = (
        (pixel_row >= 0)
        & (pixel_row < rows)
        & (pixel_column >= 0)
        & (pixel_column < columns)
        & (weights > 1e-12)
    )
    pixels = pixel_row[inside] * columns + pixel_column[inside]
    return repeated[inside], pixels, weights[inside]


def form_camera_image(
    state: SharedSurfaceState,
    camera: object,
    *,
    appearance: Appearance = "shaded",
    footprint: Footprint = "gaussian",
    reduction: Reduction = "hybrid_depth",
    depth_tolerance: float = 0.018,
) -> FormedImage:
    """Project a shared zero-set state through one visibility-aware detector."""
    device = state.points.device
    baseline_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    sensor, point_depth, valid = _project_orthographic(state, camera)
    point_ids, pixels, weights = _footprint_entries(
        sensor, valid, camera.resolution, footprint  # type: ignore[attr-defined]
    )
    depths = point_depth[point_ids]
    colors = _appearance(state, camera, appearance)
    pixel_count = camera.pixel_count  # type: ignore[attr-defined]
    owner = torch.full(
        (pixel_count,), -1, dtype=torch.long, device=device
    )
    depth = torch.full(
        (pixel_count,), torch.inf, dtype=state.points.dtype, device=device
    )

    if reduction == "accumulate":
        selected = torch.ones_like(weights, dtype=torch.bool)
    elif reduction == "dominant_weight":
        maximum = torch.full_like(depth, -torch.inf)
        maximum.scatter_reduce_(0, pixels, weights, reduce="amax")
        selected = weights == maximum[pixels]
    else:
        depth.scatter_reduce_(0, pixels, depths, reduce="amin")
        if reduction == "nearest_depth":
            selected = depths == depth[pixels]
        elif reduction == "hybrid_depth":
            selected = depths <= depth[pixels] + depth_tolerance
        else:
            raise ValueError(f"unknown reduction: {reduction}")

    selected_points = point_ids[selected]
    selected_pixels = pixels[selected]
    selected_weights = weights[selected]
    selected_depths = depths[selected]
    normalizer = torch.zeros(
        pixel_count, dtype=state.points.dtype, device=device
    )
    normalizer.scatter_add_(0, selected_pixels, selected_weights)
    normalized = selected_weights / normalizer[selected_pixels].clamp_min(1e-30)
    indices = torch.stack((selected_pixels, selected_points), dim=0)
    transport = torch.sparse_coo_tensor(
        indices,
        normalized,
        (pixel_count, state.points.shape[0]),
        dtype=state.points.dtype,
        device=device,
    ).coalesce()
    image = torch.sparse.mm(transport, colors)
    weighted_depth = torch.zeros_like(depth)
    weighted_depth.scatter_add_(
        0, selected_pixels, normalized * selected_depths
    )
    mask = normalizer > 0.0
    depth = torch.where(mask, weighted_depth, torch.full_like(depth, torch.inf))
    missing = torch.iinfo(torch.long).max
    owner_work = torch.full_like(owner, missing)
    owner_work.scatter_reduce_(0, selected_pixels, selected_points, reduce="amin")
    owner[mask] = owner_work[mask]
    _sync(device)
    seconds = time.perf_counter() - started
    return FormedImage(
        image.reshape(*camera.resolution, 3),  # type: ignore[attr-defined]
        mask.reshape(*camera.resolution),  # type: ignore[attr-defined]
        depth.reshape(*camera.resolution),  # type: ignore[attr-defined]
        owner.reshape(*camera.resolution),  # type: ignore[attr-defined]
        transport,
        int(valid.sum()),
        int(transport._nnz()),
        seconds,
        (torch.cuda.max_memory_allocated(device) - baseline_allocated) / 2**20
        if device.type == "cuda"
        else 0.0,
    )


def _reference_appearance(
    points: np.ndarray,
    normals: np.ndarray,
    camera: object,
    mode: Appearance,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if mode == "constant":
        return np.full_like(points, 0.72)
    if mode == "normal":
        return np.clip(0.5 * (normals + 1.0), 0.0, 1.0)
    if mode == "position":
        return np.clip((points - lower) / (upper - lower), 0.0, 1.0)
    light = np.asarray([0.35, 0.80, 0.48], dtype=np.float64)
    light /= np.linalg.norm(light)
    center = camera.center.detach().cpu().numpy()  # type: ignore[attr-defined]
    view = center[None, :] - points
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-30)
    diffuse = np.abs(normals @ light)
    facing = np.abs(np.sum(normals * view, axis=1))
    intensity = np.clip(0.20 + 0.58 * diffuse + 0.22 * facing, 0.0, 1.0)
    return intensity[:, None] * np.asarray([0.86, 0.82, 0.74])[None, :]


def reference_zero_set_image(
    field: GridZeroSetField,
    camera: object,
    *,
    appearance: Appearance = "shaded",
) -> ReferenceImage:
    """Orthographic ray reference against the extracted zero-set surface."""
    import open3d as o3d

    vertices = field.surface_vertices.detach().cpu().numpy().astype(np.float32)
    faces = field.surface_faces.detach().cpu().numpy().astype(np.int32)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    rows, columns = camera.resolution  # type: ignore[attr-defined]
    horizontal = (
        (np.arange(columns, dtype=np.float64) + 0.5) / columns - 0.5
    ) * camera.width  # type: ignore[attr-defined]
    vertical = (
        0.5 - (np.arange(rows, dtype=np.float64) + 0.5) / rows
    ) * camera.height  # type: ignore[attr-defined]
    yy, xx = np.meshgrid(vertical, horizontal, indexing="ij")
    center = camera.center.detach().cpu().numpy()  # type: ignore[attr-defined]
    right = camera.right.detach().cpu().numpy()  # type: ignore[attr-defined]
    up = camera.up.detach().cpu().numpy()  # type: ignore[attr-defined]
    direction = camera.normal.detach().cpu().numpy()  # type: ignore[attr-defined]
    origins = center + xx[..., None] * right + yy[..., None] * up
    directions = np.broadcast_to(direction, origins.shape)
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    depth = result["t_hit"].numpy().astype(np.float64)
    mask = np.isfinite(depth)
    normals = result["primitive_normals"].numpy().astype(np.float64)
    points = origins + np.where(mask, depth, 0.0)[..., None] * directions
    flat_colors = _reference_appearance(
        points.reshape(-1, 3),
        normals.reshape(-1, 3),
        camera,
        appearance,
        field.lower.detach().cpu().numpy(),
        field.upper.detach().cpu().numpy(),
    )
    image = flat_colors.reshape(rows, columns, 3)
    image[~mask] = 0.0
    normals[~mask] = 0.0
    return ReferenceImage(image, mask, depth, normals)


def image_metrics(
    formed: FormedImage | np.ndarray,
    reference: ReferenceImage,
) -> dict[str, float]:
    """Silhouette, pixel, edge, support, and depth agreement metrics."""
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.metrics import structural_similarity

    if isinstance(formed, FormedImage):
        image = formed.image.detach().cpu().numpy()
        mask = formed.mask.detach().cpu().numpy()
        depth = formed.depth.detach().cpu().numpy()
    else:
        image = np.asarray(formed)
        mask = np.any(image > 1e-10, axis=-1)
        depth = np.full(mask.shape, np.inf)
    truth = reference.mask
    intersection = int(np.logical_and(mask, truth).sum())
    union = int(np.logical_or(mask, truth).sum())
    predicted = int(mask.sum())
    actual = int(truth.sum())
    precision = intersection / max(predicted, 1)
    recall = intersection / max(actual, 1)
    mse = float(np.mean((image - reference.image) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-30))
    ssim = float(
        structural_similarity(
            reference.image,
            image,
            channel_axis=-1,
            data_range=1.0,
        )
    )
    edge = mask ^ binary_erosion(mask)
    truth_edge = truth ^ binary_erosion(truth)
    edge_precision = int(
        np.logical_and(edge, binary_dilation(truth_edge, iterations=2)).sum()
    ) / max(int(edge.sum()), 1)
    edge_recall = int(
        np.logical_and(truth_edge, binary_dilation(edge, iterations=2)).sum()
    ) / max(int(truth_edge.sum()), 1)
    if predicted:
        predicted_center = np.asarray(np.nonzero(mask)).mean(axis=1)
    else:
        predicted_center = np.zeros(2)
    actual_center = np.asarray(np.nonzero(truth)).mean(axis=1)
    diagonal = math.hypot(*mask.shape)
    depth_overlap = mask & truth & np.isfinite(depth)
    return {
        "silhouette_iou": intersection / max(union, 1),
        "silhouette_precision": precision,
        "silhouette_recall": recall,
        "silhouette_f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
        "foreground_fraction": predicted / mask.size,
        "reference_foreground_fraction": actual / mask.size,
        "pixel_mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "edge_f1": 2.0 * edge_precision * edge_recall
        / max(edge_precision + edge_recall, 1e-30),
        "center_of_mass_error_normalized": float(
            np.linalg.norm(predicted_center - actual_center) / diagonal
        ),
        "depth_rmse_on_overlap": float(
            np.sqrt(np.mean((depth[depth_overlap] - reference.depth[depth_overlap]) ** 2))
        )
        if depth_overlap.any()
        else math.inf,
    }


def image_formation_cpu_verification() -> dict[str, object]:
    """Small deterministic fixed-cell and shared projection checks."""
    from .camera import PlanarCamera

    device = torch.device("cpu")
    count = 2048
    sequence = torch.quasirandom.SobolEngine(2, scramble=False).draw(count)
    vertical = 1.0 - 2.0 * sequence[:, 0]
    angle = 2.0 * torch.pi * sequence[:, 1]
    radial = torch.sqrt((1.0 - vertical.square()).clamp_min(0.0))
    points = torch.stack(
        (radial * torch.cos(angle), vertical, radial * torch.sin(angle)), 1
    ).to(torch.float64)
    camera = PlanarCamera(
        torch.tensor([0.0, 0.0, 3.0]),
        torch.tensor([0.0, 0.0, -1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        2.4,
        2.4,
        (32, 32),
    )
    state = SharedSurfaceState(
        points,
        points.clone(),
        torch.full((3,), -1.2, dtype=torch.float64),
        torch.full((3,), 1.2, dtype=torch.float64),
        0.0,
        0.0,
        0.0,
    )
    first = form_camera_image(state, camera)
    second = form_camera_image(state, camera)
    dense = first.transport.to_dense()
    row_sums = dense.sum(dim=1)
    assert torch.equal(first.owner, second.owner)
    assert torch.equal(first.mask, second.mask)
    assert torch.equal(first.transport.coalesce().indices(), second.transport.coalesce().indices())
    assert float((first.image - second.image).abs().max()) == 0.0
    assert float((row_sums[first.mask.reshape(-1)] - 1.0).abs().max()) < 1e-12
    return {
        "deterministic_owner_map": True,
        "deterministic_mask": True,
        "image_max_error": 0.0,
        "occupied_pixels": int(first.mask.sum()),
        "transport_nnz": int(first.transport._nnz()),
        "occupied_transport_row_sum_max_error": float(
            (row_sums[first.mask.reshape(-1)] - 1.0).abs().max()
        ),
        "device": str(device),
    }


@dataclass(frozen=True)
class LegacyPacketScene:
    geometry: BenchmarkGeometry
    state: SceneTransportState
    config: MultiviewConfig
    build_seconds: float
    emitted_photons: int
    surface_hits: int
    peak_allocated_mib: float


def build_legacy_packet_scene(
    field: GridZeroSetField,
    points: Tensor,
    surface_normals: Tensor,
    direction_normals: Tensor,
    colors: Tensor,
    *,
    packets_per_emitter: int,
    cone_power: float,
    resolution: int,
) -> LegacyPacketScene:
    """Build the historical outward-packet state for controlled comparisons."""
    device = points.device
    baseline_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    directions = nested_deterministic_directions(
        direction_normals, packets_per_emitter, cone_power
    )
    photons = _photon_batch(points, directions, colors, packets_per_emitter)
    maximum_times = torch.full(
        (photons.count,), 3.0, dtype=points.dtype, device=device
    )
    hits, hit_times = first_zero_set_intersections(
        field,
        photons.origins,
        photons.directions,
        maximum_times,
        samples=24,
        bisection_steps=20,
        chunk_size=8192,
    )
    _sync(device)
    config = MultiviewConfig(
        resolution=(resolution, resolution),
        emitters=points.shape[0],
        packets_per_emitter=packets_per_emitter,
        parameter_count=1,
        cone_power=cone_power,
        root_samples=24,
        bisection_steps=20,
    )
    return LegacyPacketScene(
        BenchmarkGeometry(field, points, surface_normals, colors),
        SceneTransportState(
            points,
            surface_normals,
            photons,
            directions,
            hits,
            hit_times,
            torch.ones(points.shape[0], dtype=torch.bool, device=device),
        ),
        config,
        time.perf_counter() - started,
        photons.count,
        int(hits.sum()),
        (torch.cuda.max_memory_allocated(device) - baseline_allocated) / 2**20
        if device.type == "cuda"
        else 0.0,
    )


def form_legacy_packet_image(
    scene: LegacyPacketScene, camera: object
) -> tuple[np.ndarray, dict[str, object]]:
    device = scene.state.points.device
    _sync(device)
    started = time.perf_counter()
    transport = project_camera(scene.geometry, camera, scene.state, scene.config)
    _sync(device)
    seconds = time.perf_counter() - started
    image = (
        transport.image.reshape(*camera.resolution, 3)  # type: ignore[attr-defined]
        .detach()
        .cpu()
        .numpy()
    )
    return image, {
        "detector_seconds": seconds,
        "camera_hits": int(transport.intersection.valid.sum()),
        "surviving_packets": int(transport.survivor_ids.numel()),
        "occupied_pixels": int((transport.cell.owner_map >= 0).sum()),
        "transport_nnz": int(transport.transport._nnz()),
    }


def _flatten_csv_value(value: object) -> object:
    return (
        json.dumps(value, separators=(",", ":"))
        if isinstance(value, (dict, list, tuple))
        else value
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    flattened = [
        {key: _flatten_csv_value(value) for key, value in row.items()}
        for row in rows
    ]
    keys = sorted({key for row in flattened for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _save_rgb(path: Path, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def _plot_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    rows: int,
    columns: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.25 * rows), facecolor="black"
    )
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (label, image) in zip(axes_array, panels):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(label, color="white", fontsize=10)
        axis.axis("off")
    for axis in axes_array[len(panels) :]:
        axis.axis("off")
    figure.suptitle(title, color="white", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _formed_summary(
    name: str,
    formed: FormedImage,
    reference: ReferenceImage,
    **metadata: object,
) -> dict[str, object]:
    pixels = formed.mask.numel()
    indices = formed.transport.coalesce().indices()
    column_counts = torch.bincount(
        indices[1], minlength=formed.transport.shape[1]
    )
    row_counts = torch.bincount(indices[0], minlength=pixels)
    responsive = column_counts > 0
    return {
        "name": name,
        **metadata,
        **image_metrics(formed, reference),
        "runtime_seconds": formed.runtime_seconds,
        "projected_surface_points": formed.projected_points,
        "operator_nnz": formed.visible_entries,
        "operator_density": formed.visible_entries
        / max(pixels * formed.transport.shape[1], 1),
        "maximum_pixels_per_surface_sample": int(column_counts.max()),
        "mean_pixels_per_responsive_surface_sample": float(
            column_counts[responsive].to(torch.float64).mean()
        )
        if bool(responsive.any())
        else 0.0,
        "maximum_contributors_per_pixel": int(row_counts.max()),
        "peak_allocated_mib": formed.peak_allocated_mib,
    }


def _legacy_summary(
    name: str,
    image: np.ndarray,
    reference: ReferenceImage,
    scene: LegacyPacketScene,
    detector: dict[str, object],
    **metadata: object,
) -> dict[str, object]:
    return {
        "name": name,
        **metadata,
        **image_metrics(image, reference),
        "scene_seconds": scene.build_seconds,
        "runtime_seconds": scene.build_seconds
        + float(detector["detector_seconds"]),
        "emitted_photons": scene.emitted_photons,
        "surface_hits": scene.surface_hits,
        "peak_allocated_mib": scene.peak_allocated_mib,
        **detector,
    }


def _legacy_for_state(
    field: GridZeroSetField,
    state: SharedSurfaceState,
    camera: object,
    reference: ReferenceImage,
    *,
    emitters: int,
    packets: int,
    cone_power: float,
    appearance: Appearance,
    resolution: int,
    name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    subset = SharedSurfaceState(
        state.points[:emitters],
        state.normals[:emitters],
        state.field_lower,
        state.field_upper,
        state.build_seconds,
        state.residual_max,
        state.peak_allocated_mib,
    )
    colors = _appearance(subset, camera, appearance)
    scene = build_legacy_packet_scene(
        field,
        subset.points,
        subset.normals,
        subset.normals,
        colors,
        packets_per_emitter=packets,
        cone_power=cone_power,
        resolution=resolution,
    )
    image, detector = form_legacy_packet_image(scene, camera)
    return image, _legacy_summary(
        name,
        image,
        reference,
        scene,
        detector,
        emitters=emitters,
        packets_per_emitter=packets,
        cone_power=cone_power,
        appearance=appearance,
        resolution=resolution,
    )


def run_image_formation_experiment(
    mesh_path: Path,
    artifact_directory: Path,
    figure_directory: Path,
    render_directory: Path,
) -> dict[str, object]:
    """Run the controlled Bunny image-formation correction experiment."""
    if not torch.cuda.is_available():
        raise RuntimeError("LOCAL_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    experiment_started = time.perf_counter()
    prepared = prepare_stanford_bunny(mesh_path)
    field = prepared.gt_field.to(device)
    cameras = nested_bunny_cameras(256, device, 8)
    reference_shaded = [
        reference_zero_set_image(field, camera, appearance="shaded")
        for camera in cameras
    ]
    main_state = build_shared_surface_state(field, 131_072, device)
    comparison_view = 3
    camera = cameras[comparison_view]
    reference = reference_shaded[comparison_view]

    reduction_specs: tuple[tuple[str, Footprint, Reduction, float], ...] = (
        ("surface_accumulate_gaussian", "gaussian", "accumulate", 0.0),
        ("surface_dominant_gaussian", "gaussian", "dominant_weight", 0.0),
        ("surface_hard_point_depth", "point", "nearest_depth", 0.0),
        ("surface_hard_bilinear_depth", "bilinear", "nearest_depth", 0.0),
        ("surface_hybrid_gaussian_depth", "gaussian", "hybrid_depth", 0.018),
    )
    reduction_rows: list[dict[str, object]] = []
    reduction_images: dict[str, np.ndarray] = {}
    for name, footprint, reduction, tolerance in reduction_specs:
        formed = form_camera_image(
            main_state,
            camera,
            appearance="shaded",
            footprint=footprint,
            reduction=reduction,
            depth_tolerance=tolerance,
        )
        reduction_rows.append(
            _formed_summary(
                name,
                formed,
                reference,
                phase="reduction",
                view=comparison_view,
                resolution=256,
                surface_samples=main_state.points.shape[0],
                appearance="shaded",
                footprint=footprint,
                reduction=reduction,
                depth_tolerance=tolerance,
            )
        )
        reduction_images[name] = formed.image.detach().cpu().numpy()
    for row in reduction_rows:
        row["selection_score"] = (
            0.45 * float(row["silhouette_iou"])
            + 0.35 * float(row["ssim"])
            + 0.20 * float(row["edge_f1"])
        )
    best_reduction = max(
        reduction_rows,
        key=lambda row: float(row["selection_score"]),
    )
    best_name = str(best_reduction["name"])
    best_footprint = str(best_reduction["footprint"])
    best_rule = str(best_reduction["reduction"])
    best_tolerance = float(best_reduction["depth_tolerance"])

    appearance_rows: list[dict[str, object]] = []
    appearance_images: dict[str, np.ndarray] = {}
    for appearance in ("constant", "normal", "position", "shaded"):
        appearance_reference = reference_zero_set_image(
            field, camera, appearance=appearance
        )
        formed = form_camera_image(
            main_state,
            camera,
            appearance=appearance,
            footprint=best_footprint,  # type: ignore[arg-type]
            reduction=best_rule,  # type: ignore[arg-type]
            depth_tolerance=best_tolerance,
        )
        appearance_rows.append(
            _formed_summary(
                f"appearance_{appearance}",
                formed,
                appearance_reference,
                phase="appearance",
                view=comparison_view,
                resolution=256,
                surface_samples=main_state.points.shape[0],
                appearance=appearance,
                footprint=best_footprint,
                reduction=best_rule,
            )
        )
        appearance_images[appearance] = formed.image.detach().cpu().numpy()

    best_images: list[np.ndarray] = []
    best_rows: list[dict[str, object]] = []
    formed_multiview: list[FormedImage] = []
    for view, (current_camera, current_reference) in enumerate(
        zip(cameras, reference_shaded)
    ):
        formed = form_camera_image(
            main_state,
            current_camera,
            appearance="shaded",
            footprint=best_footprint,  # type: ignore[arg-type]
            reduction=best_rule,  # type: ignore[arg-type]
            depth_tolerance=best_tolerance,
        )
        formed_multiview.append(formed)
        best_images.append(formed.image.detach().cpu().numpy())
        best_rows.append(
            _formed_summary(
                f"best_view_{view}",
                formed,
                current_reference,
                phase="multiview_best",
                view=view,
                resolution=256,
                surface_samples=main_state.points.shape[0],
                appearance="shaded",
                footprint=best_footprint,
                reduction=best_rule,
            )
        )

    # Reproduce the historical v0.3g target hit-map exactly for the comparison.
    legacy_config = BunnyExperimentConfig(
        views=8,
        resolution=256,
        emitters=8192,
        packets_per_emitter=32,
        random_seeds=(),
        oracle_subset_size=0,
    )
    canonical = _canonical_emitters(prepared, legacy_config.emitters, device)
    geometry = _geometry_bundle(prepared, canonical, 2.5, legacy_config)
    legacy_colors = geometry.colors
    legacy_scene = build_legacy_packet_scene(
        geometry.gt,
        geometry.target_points,
        geometry.target_normals,
        geometry.reference_normals,
        legacy_colors,
        packets_per_emitter=32,
        cone_power=128.0,
        resolution=256,
    )
    legacy_current_images = []
    legacy_rows = []
    for view, (current_camera, current_reference) in enumerate(
        zip(cameras, reference_shaded)
    ):
        image, detector = form_legacy_packet_image(legacy_scene, current_camera)
        legacy_current_images.append(image)
        legacy_rows.append(
            _legacy_summary(
                f"legacy_current_view_{view}",
                image,
                current_reference,
                legacy_scene,
                detector,
                phase="legacy_current",
                view=view,
                resolution=256,
                emitters=8192,
                packets_per_emitter=32,
                cone_power=128.0,
                appearance="position",
            )
        )

    # Appearance-isolated legacy control: constant color cannot repair its support.
    constant_colors = torch.full_like(geometry.colors, 0.72)
    legacy_constant_scene = build_legacy_packet_scene(
        geometry.gt,
        geometry.target_points,
        geometry.target_normals,
        geometry.reference_normals,
        constant_colors,
        packets_per_emitter=32,
        cone_power=128.0,
        resolution=256,
    )
    constant_image, constant_detector = form_legacy_packet_image(
        legacy_constant_scene, camera
    )
    constant_reference = reference_zero_set_image(
        field, camera, appearance="constant"
    )
    legacy_constant_row = _legacy_summary(
        "legacy_constant_appearance_control",
        constant_image,
        constant_reference,
        legacy_constant_scene,
        constant_detector,
        phase="appearance_control",
        view=comparison_view,
        resolution=256,
        emitters=8192,
        packets_per_emitter=32,
        cone_power=128.0,
        appearance="constant",
    )

    # Directional spread comparison for the historical packet operator.
    spread_rows: list[dict[str, object]] = []
    spread_images: dict[str, np.ndarray] = {}
    legacy_sample_state = SharedSurfaceState(
        geometry.target_points,
        geometry.target_normals,
        geometry.gt.lower,
        geometry.gt.upper,
        0.0,
        float(geometry.gt.value(geometry.target_points).abs().max()),
        0.0,
    )
    for label, cone_power in (("wide", 16.0), ("medium", 128.0), ("narrow", 1024.0)):
        image, row = _legacy_for_state(
            geometry.gt,
            legacy_sample_state,
            camera,
            reference_zero_set_image(field, camera, appearance="position"),
            emitters=8192,
            packets=32,
            cone_power=cone_power,
            appearance="position",
            resolution=256,
            name=f"legacy_spread_{label}",
        )
        row["phase"] = "directional_spread"
        spread_rows.append(row)
        spread_images[label] = image
    spread_rows.append(
        {
            **best_reduction,
            "name": "deterministic_projection_limit",
            "phase": "directional_spread",
            "cone_power": "projection_limit",
        }
    )
    spread_images["projection"] = reduction_images[best_name]

    # Shared versus independent camera execution with identical deterministic state.
    shared_total = main_state.build_seconds + sum(
        formed.runtime_seconds for formed in formed_multiview
    )
    independent_images = []
    independent_build_seconds = 0.0
    independent_detector_seconds = 0.0
    independent_peak = 0.0
    owner_exact = []
    mask_exact = []
    operator_indices_exact = []
    operator_values_max_error = []
    for view, current_camera in enumerate(cameras):
        rebuilt = build_shared_surface_state(field, 131_072, device)
        independent_build_seconds += rebuilt.build_seconds
        independent = form_camera_image(
            rebuilt,
            current_camera,
            appearance="shaded",
            footprint=best_footprint,  # type: ignore[arg-type]
            reduction=best_rule,  # type: ignore[arg-type]
            depth_tolerance=best_tolerance,
        )
        independent_detector_seconds += independent.runtime_seconds
        independent_peak = max(
            independent_peak,
            rebuilt.peak_allocated_mib,
            independent.peak_allocated_mib,
        )
        independent_images.append(independent.image.detach().cpu())
        shared = formed_multiview[view]
        owner_exact.append(torch.equal(shared.owner, independent.owner))
        mask_exact.append(torch.equal(shared.mask, independent.mask))
        shared_operator = shared.transport.coalesce()
        independent_operator = independent.transport.coalesce()
        same_indices = torch.equal(
            shared_operator.indices(), independent_operator.indices()
        )
        operator_indices_exact.append(same_indices)
        operator_values_max_error.append(
            float(
                (
                    shared_operator.values() - independent_operator.values()
                ).abs().max()
            )
            if same_indices and shared_operator._nnz()
            else (0.0 if same_indices else math.inf)
        )
    shared_images_cpu = [item.image.detach().cpu() for item in formed_multiview]
    shared_independent_error = max(
        float((left - right).abs().max())
        for left, right in zip(shared_images_cpu, independent_images)
    )
    independent_total = independent_build_seconds + independent_detector_seconds
    sharing = {
        "views": 8,
        "surface_samples": 131_072,
        "shared_surface_build_seconds": main_state.build_seconds,
        "shared_detector_seconds": sum(
            item.runtime_seconds for item in formed_multiview
        ),
        "shared_total_seconds": shared_total,
        "independent_surface_build_seconds": independent_build_seconds,
        "independent_detector_seconds": independent_detector_seconds,
        "independent_total_seconds": independent_total,
        "speedup": independent_total / max(shared_total, 1e-30),
        "image_max_abs_error": shared_independent_error,
        "owner_maps_exact": all(owner_exact),
        "masks_exact": all(mask_exact),
        "operator_indices_exact": all(operator_indices_exact),
        "operator_values_max_abs_error": max(operator_values_max_error),
        "exact": shared_independent_error == 0.0
        and all(owner_exact)
        and all(mask_exact)
        and all(operator_indices_exact)
        and max(operator_values_max_error) == 0.0,
        "shared_peak_allocated_mib": max(
            main_state.peak_allocated_mib,
            max(item.peak_allocated_mib for item in formed_multiview),
        ),
        "independent_peak_allocated_mib": independent_peak,
    }

    # Post-correction sample-density and resolution sweep.
    density_rows: list[dict[str, object]] = []
    density_states: dict[int, SharedSurfaceState] = {131_072: main_state}
    for count in (32_768, 131_072, 524_288):
        current_state = density_states.get(count)
        if current_state is None:
            current_state = build_shared_surface_state(field, count, device)
            density_states[count] = current_state
        formed = form_camera_image(
            current_state,
            camera,
            appearance="shaded",
            footprint=best_footprint,  # type: ignore[arg-type]
            reduction=best_rule,  # type: ignore[arg-type]
            depth_tolerance=best_tolerance,
        )
        density_rows.append(
            _formed_summary(
                f"surface_samples_{count}",
                formed,
                reference,
                phase="surface_sample_scaling",
                view=comparison_view,
                resolution=256,
                surface_samples=count,
                shared_surface_build_seconds=current_state.build_seconds,
                appearance="shaded",
                footprint=best_footprint,
                reduction=best_rule,
            )
        )

    resolution_rows: list[dict[str, object]] = []
    for resolution in (256, 512):
        current_camera = nested_bunny_cameras(resolution, device, 8)[comparison_view]
        current_reference = reference_zero_set_image(
            field, current_camera, appearance="shaded"
        )
        formed = form_camera_image(
            main_state,
            current_camera,
            appearance="shaded",
            footprint=best_footprint,  # type: ignore[arg-type]
            reduction=best_rule,  # type: ignore[arg-type]
            depth_tolerance=best_tolerance,
        )
        resolution_rows.append(
            _formed_summary(
                f"resolution_{resolution}",
                formed,
                current_reference,
                phase="resolution_scaling",
                view=comparison_view,
                resolution=resolution,
                surface_samples=main_state.points.shape[0],
                appearance="shaded",
                footprint=best_footprint,
                reduction=best_rule,
            )
        )

    photon_rows: list[dict[str, object]] = []
    photon_reference = reference_zero_set_image(
        field, camera, appearance="position"
    )
    for packets in (8, 32, 128):
        _, row = _legacy_for_state(
            geometry.gt,
            legacy_sample_state,
            camera,
            photon_reference,
            emitters=8192,
            packets=packets,
            cone_power=128.0,
            appearance="position",
            resolution=256,
            name=f"legacy_photons_{packets}",
        )
        row["phase"] = "photon_scaling"
        photon_rows.append(row)

    emitter_state = density_states[32_768]
    emitter_rows: list[dict[str, object]] = []
    for emitters in (2048, 8192, 32768):
        _, row = _legacy_for_state(
            field,
            emitter_state,
            camera,
            photon_reference,
            emitters=emitters,
            packets=8,
            cone_power=128.0,
            appearance="position",
            resolution=256,
            name=f"legacy_emitters_{emitters}",
        )
        row["phase"] = "emitter_scaling"
        emitter_rows.append(row)

    # Qualitative outputs.
    render_directory.mkdir(parents=True, exist_ok=True)
    for view, (image, current_reference) in enumerate(
        zip(best_images, reference_shaded)
    ):
        _save_rgb(
            render_directory / f"v04_bunny_image_view{view:02d}_256.png", image
        )
        _save_rgb(
            render_directory / f"v04_bunny_reference_view{view:02d}_256.png",
            current_reference.image,
        )
    _plot_grid(
        render_directory / "v04_old_vs_new.png",
        [
            ("old packet hit-map", legacy_current_images[comparison_view]),
            ("new shared image", best_images[comparison_view]),
            ("zero-set ray reference", reference.image),
        ],
        1,
        3,
        "Renderer semantics correction | Stanford Bunny | view 3",
    )
    _plot_grid(
        render_directory / "v04_multiview_bunny.png",
        [(f"view {view}", image) for view, image in enumerate(best_images)],
        2,
        4,
        "Shared zero-set surface image formation | 256x256",
    )
    _plot_grid(
        render_directory / "v04_reduction_comparison.png",
        [(name.replace("surface_", ""), image) for name, image in reduction_images.items()]
        + [("ray reference", reference.image)],
        2,
        3,
        "Detector reduction comparison",
    )
    _plot_grid(
        render_directory / "v04_appearance_comparison.png",
        [(name, image) for name, image in appearance_images.items()],
        1,
        4,
        "Surface appearance comparison",
    )
    _plot_grid(
        render_directory / "v04_directional_spread.png",
        [(name, image) for name, image in spread_images.items()],
        1,
        4,
        "Packet angular spread versus deterministic projection",
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].plot(
        [row["surface_samples"] for row in density_rows],
        [row["silhouette_iou"] for row in density_rows],
        marker="o",
    )
    axes[0].set_xscale("log", base=4)
    axes[0].set_xlabel("shared surface samples")
    axes[0].set_ylabel("silhouette IoU")
    axes[1].plot(
        [row["emitted_photons"] for row in photon_rows],
        [row["silhouette_iou"] for row in photon_rows],
        marker="o",
    )
    axes[1].set_xscale("log", base=4)
    axes[1].set_xlabel("legacy emitted photons")
    axes[1].set_ylabel("silhouette IoU")
    axes[2].bar(
        ["old hit-map", "new image"],
        [
            statistics.mean(float(row["silhouette_iou"]) for row in legacy_rows),
            statistics.mean(float(row["silhouette_iou"]) for row in best_rows),
        ],
    )
    axes[2].set_ylabel("mean V8 silhouette IoU")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure_directory.mkdir(parents=True, exist_ok=True)
    metric_figure = figure_directory / "v04_image_formation_metrics.png"
    figure.savefig(metric_figure, dpi=170)
    plt.close(figure)

    legacy_mean_iou = statistics.mean(
        float(row["silhouette_iou"]) for row in legacy_rows
    )
    new_mean_iou = statistics.mean(
        float(row["silhouette_iou"]) for row in best_rows
    )
    new_mean_f1 = statistics.mean(
        float(row["silhouette_f1"]) for row in best_rows
    )
    new_mean_ssim = statistics.mean(float(row["ssim"]) for row in best_rows)
    legacy_best_photon_iou = max(
        float(row["silhouette_iou"]) for row in photon_rows
    )
    verdict_checks = {
        "current_hitmap_insufficient": {
            "mean_V8_silhouette_iou_below_0p50": legacy_mean_iou < 0.50,
            "constant_appearance_does_not_reach_0p50_iou": float(
                legacy_constant_row["silhouette_iou"]
            )
            < 0.50,
        },
        "bunny_like_image_formation": {
            "mean_V8_silhouette_iou_at_least_0p85": new_mean_iou >= 0.85,
            "mean_V8_silhouette_f1_at_least_0p90": new_mean_f1 >= 0.90,
            "mean_V8_ssim_at_least_0p75": new_mean_ssim >= 0.75,
        },
        "shared_multiview": {
            "shared_independent_max_error_below_1e_12": shared_independent_error
            < 1e-12,
            "owners_masks_and_sparse_operator_exact": bool(sharing["exact"]),
            "shared_surface_build_is_amortized": sharing["speedup"] > 1.0,
        },
        "photon_limit_secondary": {
            "best_legacy_photon_iou_below_0p50": legacy_best_photon_iou < 0.50,
            "new_mean_iou_exceeds_best_legacy_by_0p35": new_mean_iou
            >= legacy_best_photon_iou + 0.35,
        },
    }
    verdicts = {
        "old_operator": (
            "CURRENT_HITMAP_RENDERING_INSUFFICIENT"
            if all(verdict_checks["current_hitmap_insufficient"].values())
            else "CURRENT_HITMAP_RENDERING_SUFFICIENT"
        ),
        "image_formation": (
            "BUNNY_LIKE_IMAGE_FORMATION_SUPPORTED"
            if all(verdict_checks["bunny_like_image_formation"].values())
            else "BUNNY_LIKE_IMAGE_FORMATION_NOT_SUPPORTED"
        ),
        "shared_multiview": (
            "SHARED_MULTIVIEW_IMAGE_FORMATION_SUPPORTED"
            if all(verdict_checks["shared_multiview"].values())
            else "SHARED_MULTIVIEW_IMAGE_FORMATION_NOT_SUPPORTED"
        ),
        "photon_limit": (
            "PHOTON_LIMIT_SECONDARY"
            if all(verdict_checks["photon_limit_secondary"].values())
            else "PHOTON_LIMIT_NOT_SECONDARY"
        ),
    }
    diagnosis = {
        "verdict": "CURRENT_RENDER_IS_TRANSPORT_VISUALIZATION_NOT_IMAGE_FORMATION",
        "old_operator": (
            "I[u] = C[e(q*)], where q* is the earliest surviving outward "
            "packet landing at detector pixel u; bilinear footprint weights "
            "visualize the selected packet/emitter event."
        ),
        "missing_or_insufficient": {
            "visibility": "surface self-occlusion exists along packet paths, but not camera-ray visibility",
            "ownership": "earliest packet arrival owns a detector cell, not nearest visible surface",
            "depth_ordering": "packet flight time is not camera-space surface depth",
            "compositing": "one event color is splatted; no visible-surface reconstruction",
            "appearance": "coordinate-coded emitter colors expose packet identity/location",
            "detector_reduction": "reduces stochastic arrivals rather than projected surface support",
            "projection": "outward cone intersection with a plane is not a camera projection",
        },
        "new_operator": (
            "One deterministic shared zero-set surface state is orthographically "
            "projected into each camera. The selected "
            f"{best_rule}/{best_footprint} detector rule establishes camera-space "
            "visibility and local pixel support. The resulting normalized COO "
            "operator maps local surface appearance to pixels."
        ),
    }
    all_rows = (
        reduction_rows
        + appearance_rows
        + best_rows
        + legacy_rows
        + [legacy_constant_row]
        + spread_rows
        + density_rows
        + resolution_rows
        + photon_rows
        + emitter_rows
    )
    report = {
        "environment": cuda_environment(),
        "bunny": prepared.metadata,
        "diagnosis": diagnosis,
        "configuration": {
            "views": 8,
            "resolution": 256,
            "surface_samples": 131_072,
            "projection": "orthographic using existing physical camera frames/extents",
            "best_reduction": best_rule,
            "best_footprint": best_footprint,
            "best_depth_tolerance": best_tolerance,
            "best_appearance": "shaded deterministic surface gray",
            "reduction_selection_score": (
                "0.45 * silhouette IoU + 0.35 * SSIM + 0.20 * "
                "two-pixel-tolerant edge F1"
            ),
            "comparison_view": comparison_view,
            "camera_poses": [camera_pose(item) for item in cameras],
            "shared_surface_point_sha256": _tensor_digest(main_state.points),
            "shared_surface_normal_sha256": _tensor_digest(main_state.normals),
            "surface_residual_max": main_state.residual_max,
            "fixed_cell_locality": (
                "bilinear projection proposes at most four pixels per surface "
                "sample; nearest-depth ownership retains a sparse normalized COO "
                "appearance operator"
            ),
        },
        "candidate_reductions": reduction_rows,
        "appearance_comparison": appearance_rows,
        "legacy_current_multiview": legacy_rows,
        "legacy_constant_control": legacy_constant_row,
        "directional_spread": spread_rows,
        "best_multiview": best_rows,
        "shared_vs_independent": sharing,
        "surface_sample_scaling": density_rows,
        "resolution_scaling": resolution_rows,
        "photon_scaling": photon_rows,
        "emitter_scaling": emitter_rows,
        "summary": {
            "legacy_mean_V8_silhouette_iou": legacy_mean_iou,
            "new_mean_V8_silhouette_iou": new_mean_iou,
            "new_mean_V8_silhouette_f1": new_mean_f1,
            "new_mean_V8_ssim": new_mean_ssim,
            "new_mean_V8_psnr": statistics.mean(
                float(row["psnr"]) for row in best_rows
            ),
            "new_mean_V8_edge_f1_2px": statistics.mean(
                float(row["edge_f1"]) for row in best_rows
            ),
            "new_mean_V8_center_error": statistics.mean(
                float(row["center_of_mass_error_normalized"])
                for row in best_rows
            ),
        },
        "verdict_checks": verdict_checks,
        "verdicts": verdicts,
        "artifacts": {
            "metrics_csv": str(artifact_directory / "v04_image_formation.csv"),
            "metrics_json": str(artifact_directory / "v04_image_formation.json"),
            "metric_figure": str(metric_figure),
            "render_directory": str(render_directory),
        },
        "total_experiment_seconds": time.perf_counter() - experiment_started,
    }
    _write_csv(artifact_directory / "v04_image_formation.csv", all_rows)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "v04_image_formation.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n"
    )
    return report
