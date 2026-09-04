"""Deterministic Stanford Bunny acquisition and fixed grid zero-set fields."""

from __future__ import annotations

import hashlib
import io
import math
import tarfile
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


Tensor = torch.Tensor
STANFORD_BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
STANFORD_ARCHIVE_SHA256 = (
    "a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84"
)
STANFORD_MESH_SHA256 = (
    "b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe"
)
STANFORD_MESH_MEMBER = "bunny/reconstruction/bun_zipper.ply"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_stanford_bunny(cache_directory: Path) -> Path:
    """Download and verify only the official high-resolution reconstruction."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = cache_directory / "bun_zipper.ply"
    if mesh_path.is_file() and sha256_file(mesh_path) == STANFORD_MESH_SHA256:
        return mesh_path
    with urllib.request.urlopen(STANFORD_BUNNY_URL, timeout=120) as response:
        archive_bytes = response.read()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    if archive_hash != STANFORD_ARCHIVE_SHA256:
        raise RuntimeError(f"unexpected Stanford Bunny archive hash: {archive_hash}")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        member = archive.getmember(STANFORD_MESH_MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("official archive does not contain bun_zipper.ply")
        mesh_bytes = stream.read()
    mesh_hash = hashlib.sha256(mesh_bytes).hexdigest()
    if mesh_hash != STANFORD_MESH_SHA256:
        raise RuntimeError(f"unexpected Stanford Bunny mesh hash: {mesh_hash}")
    mesh_path.write_bytes(mesh_bytes)
    return mesh_path


def _trilinear(data: Tensor, points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    shape = points.shape[:-1]
    flat_points = points.reshape(-1, 3)
    dimensions = torch.tensor(
        data.shape[:3], dtype=points.dtype, device=points.device
    )
    coordinate = (flat_points - lower) / (upper - lower) * (dimensions - 1.0)
    maximum = dimensions.to(torch.long) - 2
    index = torch.floor(coordinate).to(torch.long)
    index = torch.minimum(torch.maximum(index, torch.zeros_like(index)), maximum)
    fraction = (coordinate - index.to(points.dtype)).clamp(0.0, 1.0)
    nx, ny, nz = data.shape[:3]
    flattened = data.reshape(nx * ny * nz, *data.shape[3:])
    output = torch.zeros(
        (flat_points.shape[0], *data.shape[3:]),
        dtype=data.dtype,
        device=data.device,
    )
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = index[:, 0] + dx
                iy = index[:, 1] + dy
                iz = index[:, 2] + dz
                linear = (ix * ny + iy) * nz + iz
                weight = (
                    (fraction[:, 0] if dx else 1.0 - fraction[:, 0])
                    * (fraction[:, 1] if dy else 1.0 - fraction[:, 1])
                    * (fraction[:, 2] if dz else 1.0 - fraction[:, 2])
                )
                while weight.ndim < output.ndim:
                    weight = weight.unsqueeze(-1)
                output += weight * flattened[linear]
    return output.reshape(*shape, *data.shape[3:])


def _trilinear_scalar_gradient(
    data: Tensor, points: Tensor, lower: Tensor, upper: Tensor
) -> Tensor:
    """Evaluate the exact spatial derivative of a trilinear scalar interpolant."""
    shape = points.shape[:-1]
    flat_points = points.reshape(-1, 3)
    dimensions = torch.tensor(
        data.shape, dtype=points.dtype, device=points.device
    )
    coordinate = (flat_points - lower) / (upper - lower) * (dimensions - 1.0)
    maximum = dimensions.to(torch.long) - 2
    index = torch.floor(coordinate).to(torch.long)
    index = torch.minimum(torch.maximum(index, torch.zeros_like(index)), maximum)
    fraction = (coordinate - index.to(points.dtype)).clamp(0.0, 1.0)
    scale = (dimensions - 1.0) / (upper - lower)
    nx, ny, nz = data.shape
    flattened = data.reshape(-1)
    output = torch.zeros(
        (flat_points.shape[0], 3), dtype=data.dtype, device=data.device
    )
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                linear = (
                    (index[:, 0] + dx) * ny + index[:, 1] + dy
                ) * nz + index[:, 2] + dz
                value = flattened[linear]
                output[:, 0] += value * (1.0 if dx else -1.0) * wy * wz * scale[0]
                output[:, 1] += value * wx * (1.0 if dy else -1.0) * wz * scale[1]
                output[:, 2] += value * wx * wy * (1.0 if dz else -1.0) * scale[2]
    return output.reshape(*shape, 3)


@dataclass(frozen=True)
class GridZeroSetField:
    """A fixed trilinear scalar field; learned distance semantics are not used."""

    grid: Tensor
    gradient_grid: Tensor
    lower: Tensor
    upper: Tensor
    surface_vertices: Tensor
    surface_faces: Tensor
    face_cdf: Tensor

    def to(self, device: torch.device) -> "GridZeroSetField":
        return GridZeroSetField(
            self.grid.to(device),
            self.gradient_grid.to(device),
            self.lower.to(device),
            self.upper.to(device),
            self.surface_vertices.to(device),
            self.surface_faces.to(device),
            self.face_cdf.to(device),
        )

    def value(self, points: Tensor) -> Tensor:
        return _trilinear(self.grid, points, self.lower, self.upper)

    def gradient(self, points: Tensor) -> Tensor:
        return _trilinear(self.gradient_grid, points, self.lower, self.upper)

    def interpolant_gradient(self, points: Tensor) -> Tensor:
        return _trilinear_scalar_gradient(
            self.grid, points, self.lower, self.upper
        )

    def hierarchical_surface_points(
        self, count: int, device: torch.device
    ) -> Tensor:
        if count <= 0:
            raise ValueError("surface point count must be positive")
        samples = torch.quasirandom.SobolEngine(3, scramble=False).draw(count)
        samples = samples.to(dtype=torch.float64, device=device)
        face_ids = torch.searchsorted(
            self.face_cdf, samples[:, 0].contiguous()
        ).clamp_max(
            self.surface_faces.shape[0] - 1
        )
        triangles = self.surface_vertices[self.surface_faces[face_ids]]
        radial = torch.sqrt(samples[:, 1])
        first = 1.0 - radial
        second = radial * (1.0 - samples[:, 2])
        third = radial * samples[:, 2]
        points = (
            first[:, None] * triangles[:, 0]
            + second[:, None] * triangles[:, 1]
            + third[:, None] * triangles[:, 2]
        )
        for _ in range(10):
            values = self.value(points)
            gradients = self.gradient(points)
            points = points - values[:, None] * gradients / (
                (gradients * gradients).sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )
        return points

    def sample_surface(self, count: int, generator: torch.Generator) -> Tensor:
        return self.hierarchical_surface_points(count, generator.device)

    def color(self, points: Tensor) -> Tensor:
        return ((points - self.lower) / (self.upper - self.lower)).clamp(0.0, 1.0)


@dataclass
class PreparedBunny:
    mesh_path: Path
    metadata: dict[str, object]
    original_vertices: np.ndarray
    original_faces: np.ndarray
    repaired_vertices: np.ndarray
    repaired_faces: np.ndarray
    gt_field: GridZeroSetField
    base_field: GridZeroSetField


def _boundary_loops(mesh: object) -> list[list[int]]:
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("Bunny boundary is not a collection of simple loops")
    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        loop = [start]
        previous: int | None = None
        current = start
        while True:
            visited.add(current)
            following = [item for item in adjacency[current] if item != previous][0]
            if following == start:
                break
            loop.append(following)
            previous, current = current, following
        loops.append(loop)
    return loops


def _repair_with_centroid_caps(mesh: object) -> tuple[object, list[int]]:
    import trimesh

    loops = _boundary_loops(mesh)
    vertices = np.asarray(mesh.vertices).tolist()
    faces = np.asarray(mesh.faces).tolist()
    for loop in loops:
        center_id = len(vertices)
        vertices.append(np.asarray(mesh.vertices)[loop].mean(axis=0).tolist())
        for first, second in zip(loop, loop[1:] + loop[:1]):
            faces.append([first, second, center_id])
    repaired = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(repaired, multibody=True)
    if not repaired.is_watertight or not repaired.is_winding_consistent:
        raise RuntimeError("deterministic Bunny boundary capping did not become watertight")
    return repaired, [len(loop) for loop in loops]


def _signed_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    axis = np.linspace(lower, upper, resolution, dtype=np.float32)
    total = resolution**3
    values = np.empty(total, dtype=np.float32)
    chunk = 262144
    for start in range(0, total, chunk):
        ids = np.arange(start, min(start + chunk, total))
        first = ids // (resolution * resolution)
        second = (ids // resolution) % resolution
        third = ids % resolution
        points = np.stack((axis[first], axis[second], axis[third]), axis=1)
        values[start : start + len(ids)] = scene.compute_signed_distance(
            o3d.core.Tensor(points)
        ).numpy()
    return values.reshape((resolution,) * 3)


def _grid_field(
    values: np.ndarray,
    lower: float,
    upper: float,
    *,
    build_surface_scaffold: bool = True,
) -> tuple[GridZeroSetField, dict[str, object]]:
    spacing = (upper - lower) / (values.shape[0] - 1)
    if build_surface_scaffold:
        import trimesh
        from skimage.measure import marching_cubes

        vertices, faces, _, _ = marching_cubes(
            values, level=0.0, spacing=(spacing, spacing, spacing)
        )
        vertices += lower
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        cdf = np.cumsum(areas)
        cdf /= cdf[-1]
        surface_metadata: dict[str, object] = {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "watertight": bool(mesh.is_watertight),
            "surface_area": float(mesh.area),
            "bounds": np.asarray(mesh.bounds).tolist(),
            "scaffold": "marching_cubes",
        }
    else:
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
        cdf = np.empty(0, dtype=np.float64)
        inside = np.argwhere(values <= 0.0)
        approximate_bounds = (
            [
                (lower + inside.min(axis=0) * spacing).tolist(),
                (lower + inside.max(axis=0) * spacing).tolist(),
            ]
            if len(inside)
            else None
        )
        surface_metadata = {
            "vertices": 0,
            "faces": 0,
            "watertight": None,
            "surface_area": None,
            "bounds": approximate_bounds,
            "scaffold": "omitted_mesh_free",
        }
    gradients = np.stack(
        np.gradient(values.astype(np.float64), spacing, edge_order=2), axis=-1
    )
    field = GridZeroSetField(
        torch.from_numpy(values.astype(np.float64)),
        torch.from_numpy(gradients),
        torch.full((3,), lower, dtype=torch.float64),
        torch.full((3,), upper, dtype=torch.float64),
        torch.from_numpy(np.asarray(vertices, dtype=np.float64)),
        torch.from_numpy(np.asarray(faces, dtype=np.int64)),
        torch.from_numpy(cdf),
    )
    metadata = {
        **surface_metadata,
        "grid_minimum": float(values.min()),
        "grid_maximum": float(values.max()),
    }
    return field, metadata


def prepare_stanford_bunny(
    mesh_path: Path,
    *,
    grid_resolution: int = 160,
    base_smoothing_sigma: float = 1.5,
    grid_bound: float = 1.2,
    build_surface_scaffold: bool = True,
) -> PreparedBunny:
    """Create original-reference, watertight-proxy, GT-grid, and coarse-base data."""
    import trimesh
    from scipy.ndimage import gaussian_filter

    mesh_hash = sha256_file(mesh_path)
    if mesh_hash != STANFORD_MESH_SHA256:
        raise RuntimeError(f"mesh is not the expected official bun_zipper.ply: {mesh_hash}")
    original = trimesh.load(mesh_path, process=False, force="mesh")
    original_bounds = np.asarray(original.bounds, dtype=np.float64)
    center = original_bounds.mean(axis=0)
    scale = 2.0 / float(np.asarray(original.extents).max())
    normalized = trimesh.Trimesh(
        (np.asarray(original.vertices) - center) * scale,
        np.asarray(original.faces),
        process=False,
    )
    repaired, loop_sizes = _repair_with_centroid_caps(normalized)
    gt_values = _signed_grid(
        np.asarray(repaired.vertices),
        np.asarray(repaired.faces),
        grid_resolution,
        -grid_bound,
        grid_bound,
    )
    base_values = gaussian_filter(gt_values, sigma=base_smoothing_sigma)
    gt_field, gt_stats = _grid_field(
        gt_values,
        -grid_bound,
        grid_bound,
        build_surface_scaffold=build_surface_scaffold,
    )
    base_field, base_stats = _grid_field(
        base_values,
        -grid_bound,
        grid_bound,
        build_surface_scaffold=build_surface_scaffold,
    )
    metadata: dict[str, object] = {
        "source": (
            "Stanford University Computer Graphics Laboratory, "
            "Stanford 3D Scanning Repository"
        ),
        "source_url": STANFORD_BUNNY_URL,
        "original_filename": "bun_zipper.ply",
        "archive_sha256": STANFORD_ARCHIVE_SHA256,
        "mesh_sha256": mesh_hash,
        "original": {
            "vertices": int(len(original.vertices)),
            "faces": int(len(original.faces)),
            "watertight": bool(original.is_watertight),
            "boundary_loops": len(loop_sizes),
            "boundary_loop_edges": loop_sizes,
            "bounds": original_bounds.tolist(),
        },
        "normalization": {
            "translation_before_scale": (-center).tolist(),
            "uniform_scale": scale,
            "normalized_bounds": np.asarray(normalized.bounds).tolist(),
        },
        "sign_proxy_repair": {
            "method": (
                "one centroid fan cap per original boundary loop; "
                "visible triangles unchanged"
            ),
            "vertices": int(len(repaired.vertices)),
            "faces": int(len(repaired.faces)),
            "watertight": bool(repaired.is_watertight),
            "surface_area_added": float(repaired.area - normalized.area),
        },
        "implicit_grid": {
            "resolution": [grid_resolution] * 3,
            "bounds": [[-grid_bound] * 3, [grid_bound] * 3],
            "construction": (
                "Open3D closest-triangle signed field sampled once; "
                "trilinear generic zero-set evaluation"
            ),
            "render_surface_scaffold": (
                "marching_cubes" if build_surface_scaffold else "omitted_mesh_free"
            ),
            "gt_surface": gt_stats,
        },
        "base": {
            "construction": "fixed Gaussian smoothing of the GT proxy grid",
            "gaussian_sigma_voxels": base_smoothing_sigma,
            "effective_sigma_world": base_smoothing_sigma
            * (2.0 * grid_bound / (grid_resolution - 1)),
            "surface": base_stats,
        },
    }
    return PreparedBunny(
        mesh_path,
        metadata,
        np.asarray(normalized.vertices),
        np.asarray(normalized.faces),
        np.asarray(repaired.vertices),
        np.asarray(repaired.faces),
        gt_field,
        base_field,
    )


def resmooth_stanford_bunny(
    prepared: PreparedBunny,
    sigma_voxels: float,
) -> PreparedBunny:
    """Reuse a prepared GT grid while changing only the fixed coarse base."""
    from scipy.ndimage import gaussian_filter

    if sigma_voxels <= 0.0:
        raise ValueError("Bunny smoothing sigma must be positive")
    gt_values = prepared.gt_field.grid.detach().cpu().numpy()
    base_values = gaussian_filter(gt_values, sigma=sigma_voxels)
    lower = float(prepared.gt_field.lower[0])
    upper = float(prepared.gt_field.upper[0])
    base_field, base_stats = _grid_field(base_values, lower, upper)
    metadata = deepcopy(prepared.metadata)
    resolution = int(gt_values.shape[0])
    metadata["base"] = {
        "construction": "fixed Gaussian smoothing of the GT proxy grid",
        "gaussian_sigma_voxels": sigma_voxels,
        "effective_sigma_world": sigma_voxels
        * ((upper - lower) / (resolution - 1)),
        "surface": base_stats,
    }
    return PreparedBunny(
        prepared.mesh_path,
        metadata,
        prepared.original_vertices,
        prepared.original_faces,
        prepared.repaired_vertices,
        prepared.repaired_faces,
        prepared.gt_field,
        base_field,
    )


def mesh_surface_samples(
    vertices: np.ndarray, faces: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed mesh by a nested unscrambled Sobol prefix."""
    if count <= 0 or count & (count - 1):
        raise ValueError("mesh evaluation sample count must be a positive power of two")
    from scipy.stats import qmc

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-30)
    cdf = np.cumsum(areas)
    cdf /= cdf[-1]
    samples = qmc.Sobol(3, scramble=False).random_base2(int(math.log2(count)))
    face_ids = np.searchsorted(cdf, samples[:, 0]).clip(max=len(faces) - 1)
    radial = np.sqrt(samples[:, 1])
    weights = np.stack(
        (
            1.0 - radial,
            radial * (1.0 - samples[:, 2]),
            radial * samples[:, 2],
        ),
        axis=1,
    )
    points = (weights[:, :, None] * triangles[face_ids]).sum(axis=1)
    return points, normals[face_ids]
