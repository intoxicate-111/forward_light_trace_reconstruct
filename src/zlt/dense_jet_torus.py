"""v0.9.5 dense manifold-jet sphere-to-torus inverse-rendering experiment.

The optimizer sees target RGB images only.  Marching cubes is used to refresh
the real surface quadrature and to diagnose topology; target scalar values are
never used by the optimizer.  The local image tangent is the existing v0.9.2
CURRENT tangent, rebuilt around every freshly sampled real state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .basis_initialization import directions
from .boundary_transport import ObservationSphere, transverse_basis
from .density_matrix import capture_splat
from .manifold_jets import JetLayout, ManifoldJetField
from .matched_jacobian import integrated_kernel, optical_tangent, source_tangent
from .matched_operator import frozen_config
from .measurement_bandwidth import gate, lobe
from .meshfree_surface import meshfree_base_color
from .transverse_packet import digest, transmission_prefixes, write_json


CACHE = Path("runs/v095_dense_jet_torus")
REPORT = Path("artifacts/v095_dense_jet_torus.json")
DOMAIN = 1.45
TARGET_MAJOR = .64
TARGET_MINOR = .36
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SphereReference:
    """Exact phase reference required by ``ManifoldJetField``."""

    def __init__(self, device: str | torch.device):
        self.lower = torch.full((3,), -DOMAIN, dtype=torch.float64, device=device)
        self.upper = -self.lower

    def value(self, p):
        return p.square().sum(1) - 1.

    def gradient(self, p):
        return 2*p

    def evaluate(self, p):
        return torch.cat((self.value(p)[:, None], self.gradient(p)), 1)

    def jet(self, p):
        return self.gradient(p), 2*torch.eye(3, dtype=p.dtype, device=p.device)[None].expand(len(p), -1, -1)


class TorusTarget(SphereReference):
    """Independent target generator; not passed to optimization code."""

    def value(self, p):
        rho = p[:, :2].norm(dim=1)
        return (rho-TARGET_MAJOR).square()+p[:, 2].square()-TARGET_MINOR**2

    def gradient(self, p):
        rho = p[:, :2].norm(dim=1).clamp_min(1e-15)
        scale = 2*(rho-TARGET_MAJOR)/rho
        return torch.stack((scale*p[:, 0], scale*p[:, 1], 2*p[:, 2]), 1)

    def evaluate(self, p):
        return torch.cat((self.value(p)[:, None], self.gradient(p)), 1)


class EllipsoidTarget(SphereReference):
    scale = (1.04, .96, 1.02)

    def __init__(self, device):
        super().__init__(device)
        self.s = self.lower.new_tensor(self.scale)

    def value(self, p):
        return (p/self.s).square().sum(1)-1

    def gradient(self, p):
        return 2*p/self.s.square()

    def evaluate(self, p):
        return torch.cat((self.value(p)[:, None], self.gradient(p)), 1)


@dataclass
class Surface:
    vertices: np.ndarray
    faces: np.ndarray
    points: torch.Tensor
    normals: torch.Tensor
    normalized_weights: torch.Tensor
    area: float
    topology: dict
    residual_max: float


def _snapshot():
    path = CACHE/"starting_state.json"
    if path.exists():
        return
    protected = []
    for folder in ("artifacts", "figures", "src/zlt", "scripts", "tests"):
        protected.extend(p for p in Path(folder).glob("*") if p.is_file() and not p.name.startswith(("v095", "dense_jet_torus")))
    write_json(path, {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protected_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
    })


def make_field(centers: int, order: int = 0, support_radius: float = 1.1):
    reference = SphereReference(DEVICE)
    c = directions(centers, 955).to(DEVICE)
    layout = JetLayout(c, reference.gradient(c), c.new_full((centers,), support_radius), order)
    return ManifoldJetField(reference, layout, c.new_zeros(layout.count))


def _grid(field, n: int):
    axis = torch.linspace(-DOMAIN, DOMAIN, n, dtype=torch.float64, device=DEVICE)
    result = []
    # Cartesian product ordering matches reshape(x,y,z).
    for x in axis:
        yz = torch.cartesian_prod(axis, axis)
        points = torch.cat((x.expand(len(yz), 1), yz), 1)
        result.append(field.value(points).detach().cpu())
    return torch.cat(result).numpy().reshape(n, n, n)


def _mesh_topology(vertices, faces, n):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    watertight = bool(mesh.is_watertight)
    euler = int(mesh.euler_number)
    genus = int(round((2*len(components)-euler)/2)) if watertight else None
    return {
        "grid": n, "vertices": int(len(vertices)), "faces": int(len(faces)),
        "components": int(len(components)), "watertight": watertight,
        "euler_number": euler, "inferred_genus": genus,
        "component_areas": [float(x.area) for x in components],
        "diagnostic_only": True,
    }


def topology_only(field, n=72):
    """Higher-resolution real mesh diagnostic without constructing emitters."""
    from skimage.measure import marching_cubes
    values = _grid(field, n); spacing = 2*DOMAIN/(n-1)
    vertices, faces, _, _ = marching_cubes(values, 0., spacing=(spacing,)*3)
    vertices -= DOMAIN
    return _mesh_topology(vertices, faces, n)


def extract_surface(field, count: int, *, n: int = 56, seed: int = 719, face_quadrature: bool = True) -> Surface:
    from skimage.measure import marching_cubes

    values = _grid(field, n)
    if not (values.min() < 0 < values.max()):
        raise RuntimeError("ZERO_SET_EXTRACTION_EMPTY")
    spacing = 2*DOMAIN/(n-1)
    vertices, faces, _, _ = marching_cubes(values, 0., spacing=(spacing,)*3)
    vertices -= DOMAIN
    if np.any(np.isclose(np.abs(vertices), DOMAIN, atol=2*spacing)):
        raise RuntimeError("ZERO_SET_TOUCHES_EXTRACTION_BOUNDARY")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise RuntimeError("INVALID_EXTRACTED_SURFACE")
    if face_quadrature:
        # One deterministic centroid per triangle is substantially smoother
        # across optimization steps than remapping a finite random point set.
        ids = np.arange(len(faces)); bary = np.full((len(ids), 3), 1/3)
        normalized_weights = areas/areas.sum()
    else:
        sobol = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
        u = sobol.draw(count, dtype=torch.float64).numpy()
        ids = np.searchsorted(np.cumsum(areas), u[:, 0]*areas.sum()).clip(max=len(areas)-1)
        root = np.sqrt(u[:, 1])
        bary = np.stack((1-root, root*(1-u[:, 2]), root*u[:, 2]), 1)
        normalized_weights = np.full(count, 1/count)
    p = torch.as_tensor(np.einsum("ni,nij->nj", bary, triangles[ids]), dtype=torch.float64, device=DEVICE)
    # Refine the piecewise-linear scaffold onto the actual analytic jet field.
    for _ in range(8):
        f = field.value(p); g = field.gradient(p)
        step = f[:, None]*g/g.square().sum(1, keepdim=True).clamp_min(1e-24)
        p = p-step*(.05/step.norm(dim=1, keepdim=True).clamp_min(1e-30)).clamp_max(1.)
    residual = field.value(p).abs()
    normal = field.gradient(p)
    normal /= normal.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return Surface(vertices, faces, p, normal, torch.as_tensor(normalized_weights, dtype=torch.float64), float(areas.sum()),
                   _mesh_topology(vertices, faces, n), float(residual.max()))


def camera_state():
    d = torch.tensor([[0., 0., 1.], [.55, .25, .7968688725], [1., 0., 0.]], dtype=torch.float64, device=DEVICE)
    d /= d.norm(dim=1, keepdim=True)
    right, up = transverse_basis(d)
    return {"center": torch.zeros(3, dtype=torch.float64), "directions": d.cpu(),
            "right": right.cpu(), "up": up.cpu()}


def config(resolution=32, sources=768):
    cfg = frozen_config()
    return dict(cfg, resolution=[resolution, resolution], views=3, count=sources,
                extent=2.55, surface_grid=36, face_quadrature=True, tangent_sources=sources,
                stage="SMALL_DENSE_JET_TOPOLOGY_DIAGNOSTIC",
                note="CURRENT/C3 constants retained; source count/resolution reduced for high-dimensional inverse-rendering diagnosis.")


def render(field, cfg, *, seed=719):
    started = time.perf_counter()
    surface = extract_surface(field, cfg["count"], n=cfg.get("surface_grid", 56), seed=seed,
                              face_quadrature=cfg.get("face_quadrature", False))
    state = camera_state()
    state.update(positions=surface.points.cpu(), normals=surface.normals.cpu(),
                 weights=surface.normalized_weights*cfg["mass"])
    state["colors"] = meshfree_base_color(surface.points, field.lower, field.upper).cpu()
    boundary = ObservationSphere(torch.zeros(3, dtype=torch.float64, device=DEVICE), 1.75)
    h, w = cfg["resolution"]
    transmissions, images = [], []
    for view in range(cfg["views"]):
        p = surface.points; normal = surface.normals
        d = state["directions"][view].to(DEVICE).expand_as(p)
        t, _, _ = transmission_prefixes(field, p, d, boundary.exit_times(p, d),
            radius=cfg["h"], epsilon=cfg["epsilon"], path_step=cfg["path_step"],
            offsets=p.new_zeros((1, 3)), counts=(1,), eta=cfg["eta"], kappa=cfg["kappa"],
            launch_exclusion_factor=cfg["launch"])
        transmissions.append(t[:, 0].cpu())
        cosine = (normal*d).sum(1)
        energy = state["weights"].to(DEVICE)[:, None]*t*state["colors"].to(DEVICE)*gate(cosine, "CURRENT_SOFT_W005")[:, None]*lobe(cosine, "CURRENT_LOBE")[:, None]
        acc = torch.zeros((h*w, 3), dtype=torch.float64, device=DEVICE)
        rel = p-state["center"].to(DEVICE)
        row = (.5-rel@state["up"][view].to(DEVICE)/cfg["extent"])*h-.5
        col = (rel@state["right"][view].to(DEVICE)/cfg["extent"]+.5)*w-.5
        capture_splat(acc, row, col, energy, cfg["capture"], resolution=(h, w))
        images.append((acc.reshape(h, w, 3)*(cfg["gain"]*h*w)).cpu().numpy())
    state["transmission"] = torch.stack(transmissions)
    array = np.stack(images)
    if not np.isfinite(array).all():
        raise RuntimeError("NONFINITE_RENDER")
    return array, state, surface, boundary, {"seconds": time.perf_counter()-started}


def _differential_block(field, reference, state, owners, view, supports, boundary, cfg):
    """Device-neutral form of the unchanged v0.9.2 differential block."""
    p = state["positions"][owners].to(DEVICE); normal = state["normals"][owners].to(DEVICE)
    nref = reference["normals"][owners].to(DEVICE)
    d = state["directions"][view].to(DEVICE).expand_as(p); cosine = (normal*d).sum(1)
    keep = cosine > 0; p, normal, nref, d, cosine = p[keep], normal[keep], nref[keep], d[keep], cosine[keep]
    if not len(p):
        return None
    dx, dn = source_tangent(field, p, nref, supports)
    tau, dtau = optical_tangent(field, p, d, boundary.exit_times(p, d), dx, supports, cfg)
    transmission = state["transmission"][view, owners].to(DEVICE)[keep]
    error = float((torch.exp(-tau)-transmission).abs().max())
    if error > 1e-8:
        raise RuntimeError(f"CURRENT_TANGENT_FORWARD_MISMATCH:{error}")
    color = state["colors"][owners].to(DEVICE)[keep]; weight = state["weights"][owners].to(DEVICE)[keep]
    gate_value = gate(cosine, "CURRENT_SOFT_W005"); ell = lobe(cosine, "CURRENT_LOBE")
    energy = weight[:, None]*transmission[:, None]*color*(gate_value*ell)[:, None]
    dc = torch.einsum("nka,na->nk", dn, d)
    dg = 2*cosine/cfg["gate_width"]**2*torch.exp(-cosine.square()/cfg["gate_width"]**2)
    grad_color = p.new_tensor([.72, .70, -.62])/(field.upper-field.lower)
    inside = ((p >= field.lower) & (p <= field.upper)).to(p.dtype)
    dcolor = dx*grad_color[None, None, :]*inside[:, None, :]
    de = (weight*transmission)[:, None, None]*(dcolor*(gate_value*ell)[:, None, None]
        + color[:, None, :]*(dc*(dg*ell+.65*gate_value)[:, None])[:, :, None])
    de -= energy[:, None, :]*dtau[:, :, None]
    right = state["right"][view].to(DEVICE); up = state["up"][view].to(DEVICE); center = state["center"].to(DEVICE)
    h, w = cfg["resolution"]; rel = p-center
    row = (.5-rel@up/cfg["extent"])*h-.5; col = (rel@right/cfg["extent"]+.5)*w-.5
    drow = -torch.einsum("nka,a->nk", dx, up)*h/cfg["extent"]
    dcol = torch.einsum("nka,a->nk", dx, right)*w/cfg["extent"]
    return row, col, energy, de, drow, dcol, error


def image_jacobian(field, state, boundary, cfg):
    """Existing sparse-support v0.9.2 CURRENT tangent at a refreshed state."""
    h, w = cfg["resolution"]; k = field.layout.count; views = []
    if len(state["weights"]) > cfg["tangent_sources"]:
        generator = torch.Generator().manual_seed(919)
        ids = torch.multinomial(state["weights"], cfg["tangent_sources"], replacement=True, generator=generator)
        state = {key: (value[:, ids] if key == "transmission" else value[ids]
                       if isinstance(value, torch.Tensor) and value.ndim > 1 and len(value) == len(state["weights"])
                       else value[ids] if isinstance(value, torch.Tensor) and value.ndim == 1 and len(value) == len(state["weights"])
                       else value) for key, value in state.items()}
        state["weights"] = torch.full((len(ids),), cfg["mass"]/len(ids), dtype=torch.float64)
    reference = {**state, "normals": state["normals"]}
    for view in range(cfg["views"]):
        acc = torch.zeros((h*w, k, 3), device=DEVICE, dtype=torch.float64)
        for start in range(0, len(state["weights"]), 96):
            block = _differential_block(field, reference, state, slice(start, start+96), view,
                                        field.supports, boundary, cfg)
            if block is None:
                continue
            row, col, energy, de, drow, dcol, _ = block
            ids, weights, dy, dx = integrated_kernel(row, col, cfg["capture"], cfg["resolution"])
            for tap in range(36):
                term = weights[:, tap, None, None]*de
                term += (dy[:, tap, None]*drow+dx[:, tap, None]*dcol)[:, :, None]*energy[:, None, :]
                acc.index_add_(0, ids[:, tap], term)
        views.append((acc*cfg["gain"]*h*w).permute(0, 2, 1).reshape(-1, k).cpu().numpy())
    return np.stack(views)


def field_diagnostics(field, surface):
    p = surface.points
    gradient = field.gradient(p).norm(dim=1)
    # The reference-radial denominator is deliberately distinct from |grad F|.
    nref = p/p.norm(dim=1, keepdim=True).clamp_min(1e-30)
    denominator = (field.gradient(p)*nref).sum(1).abs()
    q = torch.tensor([0., .01, .05, .1, .5], dtype=p.dtype, device=p.device)
    lowest = torch.topk(gradient, min(8, len(gradient)), largest=False).indices
    point = p[lowest[:1]]
    _, modes, _, _ = field.layout.query(point)
    centers = torch.unique(torch.div(modes, field.layout.modes, rounding_mode="floor"))
    block_xyz = field.layout.centers[centers]
    extent = float(torch.cdist(block_xyz, block_xyz).max()) if len(centers) > 1 else 0.
    coefficients = field.coefficients
    return {
        "spatial_gradient_quantiles": torch.quantile(gradient, q).cpu().tolist(),
        "chart_denominator_quantiles": torch.quantile(denominator, q).cpu().tolist(),
        "lowest_gradient_locations": p[lowest].cpu().tolist(),
        "critical_candidate": {"location": point[0].cpu().tolist(), "active_centers": int(len(centers)),
            "active_scalar_parameters": int(len(modes)), "center_ids": centers.cpu().tolist(), "block_extent": extent},
        "active_coefficients_gt_1e_8": int((coefficients.abs() > 1e-8).sum()),
        "coefficient_l2": float(coefficients.norm()), "coefficient_abs_max": float(coefficients.abs().max()),
        "surface_residual_max": surface.residual_max, "surface_area": surface.area,
        "topology": surface.topology,
    }


def coverage(field, points):
    pi, modes, _, _ = field.layout.query(points)
    counts = torch.bincount(pi, minlength=len(points)).double()/field.layout.modes
    return {
        "support_edges": int(len(modes)/field.layout.modes),
        "overlap_mean": float(counts.mean()), "overlap_median": float(counts.median()),
        "overlap_p05_p95_min_max": torch.quantile(counts, counts.new_tensor([.05, .95, 0., 1.])).cpu().tolist(),
        "uncovered_fraction": float((counts == 0).double().mean()),
    }


def optimize(name, target_image, *, cfg=None, centers=256, order=0, selected_views=(0, 1), steps=6, support_radius=1.1):
    cfg = config() if cfg is None else cfg
    field = make_field(centers, order, support_radius); trajectory = []
    image, state, surface, boundary, timing = render(field, cfg)
    initial_image = image.copy(); initial_surface = surface
    stop = "ITERATION_BUDGET"
    for iteration in range(steps+1):
        residual = image[list(selected_views)]-target_image[list(selected_views)]
        mse = float(np.mean(residual**2))
        row = {"iteration": iteration, "actual_retraced_mse": mse,
               "all_view_mse": float(np.mean((image-target_image)**2)),
               "render": timing, "geometry": field_diagnostics(field, surface)}
        trajectory.append(row)
        if iteration == steps:
            break
        started = time.perf_counter()
        J = image_jacobian(field, state, boundary, cfg)
        js = J[list(selected_views)].reshape(-1, J.shape[-1]); r = residual.reshape(-1)
        gram = js.T@js/len(r); alignment = js.T@r/len(r)
        damping = max(float(np.diag(gram).max())*.02, 1e-7)
        delta = -np.linalg.solve(gram+damping*np.eye(len(gram)), alignment)
        raw_max = float(np.abs(delta).max())
        if raw_max > .10:
            delta *= .10/raw_max
        row.update({"jacobian_seconds": time.perf_counter()-started, "jacobian_shape": list(js.shape),
                    "jacobian_nnz": int(np.count_nonzero(np.abs(js) > 1e-12)),
                    "gradient_l2": float(np.linalg.norm(alignment)), "damping": damping,
                    "raw_update_abs_max": raw_max, "clipped_update_abs_max": float(np.abs(delta).max()), "trials": []})
        accepted = None
        for alpha in (1., .5, .25, .125, .0625):
            candidate = field.with_coefficients(field.coefficients+alpha*torch.as_tensor(delta, device=DEVICE))
            trial = {"alpha": alpha, "predicted_local_mse": float(np.mean((r+alpha*js@delta)**2))}
            try:
                candidate_image, candidate_state, candidate_surface, candidate_boundary, candidate_timing = render(candidate, cfg)
                value = float(np.mean((candidate_image[list(selected_views)]-target_image[list(selected_views)])**2))
                trial.update(actual_fresh_retrace_mse=value, topology=candidate_surface.topology,
                             accepted=value < mse*(1-1e-7))
            except RuntimeError as error:
                trial.update(accepted=False, failure=str(error))
            row["trials"].append(trial)
            if trial["accepted"]:
                accepted = (candidate, candidate_image, candidate_state, candidate_surface, candidate_boundary, candidate_timing, alpha, delta, J)
                break
        row["accepted"] = accepted is not None
        if accepted is None:
            stop = "NO_FRESH_RETRACE_IMPROVEMENT"; break
        field, image, state, surface, boundary, timing, alpha, delta, J = accepted
        row["accepted_update_l2"] = float(np.linalg.norm(alpha*delta))
        row["accepted_update_abs_max"] = float(np.abs(alpha*delta).max())
        # A mode is discovered only from the accepted dense update around the
        # observed minimum-gradient point; no fixed topology coordinate exists.
        block = row["geometry"]["critical_candidate"]["center_ids"]
        ids = np.concatenate([np.arange(i*field.layout.modes, (i+1)*field.layout.modes) for i in block]) if block else np.empty(0, int)
        mode = np.zeros(field.layout.count)
        if len(ids) and np.linalg.norm(delta[ids]) > 0:
            mode[ids] = delta[ids]/np.linalg.norm(delta[ids])
        row["trajectory_local_mode"] = {"derived_from_accepted_update": bool(np.linalg.norm(mode)),
            "parameter_count": int(len(ids)), "Jv_norm_per_view": np.linalg.norm(J.reshape(cfg["views"], -1, len(mode))@mode, axis=1).tolist() if len(mode) else []}
        torch.save({"coefficients": field.coefficients.cpu()}, CACHE/f"{name}_step{iteration+1}.pt")
        np.save(CACHE/f"{name}_step{iteration+1}.npy", image)
        write_json(CACHE/f"{name}_progress.json", {"trajectory": trajectory, "stop": stop})
        print(f"v095 {name}: step={iteration+1} mse={row['trials'][-1]['actual_fresh_retrace_mse']:.6g} genus={surface.topology['inferred_genus']}", flush=True)
    np.save(CACHE/f"{name}_initial.npy", initial_image); np.save(CACHE/f"{name}_final.npy", image)
    result = {"name": name, "centers": centers, "order": order, "parameters_per_jet": field.layout.modes,
              "total_parameters": field.layout.count, "support_radius": support_radius,
              "selected_views": list(selected_views), "optimizer": "damped full-column image Gauss-Newton; max |delta|=.10; fixed backtracking",
              "loss": "mean squared RGB residual on selected rendered views only", "trajectory": trajectory,
              "initial_mse": trajectory[0]["actual_retraced_mse"], "final_mse": trajectory[-1]["actual_retraced_mse"],
              "accepted_steps": sum(x.get("accepted", False) for x in trajectory), "stop": stop,
              "topology_initial": initial_surface.topology, "topology_final": surface.topology,
              "coverage_initial": coverage(make_field(centers, order, support_radius), initial_surface.points),
              "final_coefficients": field.coefficients.cpu().tolist(),
              "target_geometry_accessed_by_optimizer": False, "fresh_real_trace_each_trial": True,
              "complex_continuation_activated": False}
    write_json(CACHE/f"{name}.json", result)
    return result


def run(quick=False):
    torch.set_num_threads(2); CACHE.mkdir(parents=True, exist_ok=True); _snapshot()
    started = time.perf_counter()
    cpu = DEVICE.type == "cpu"
    cfg = config(sources=256 if cpu else (384 if quick else 768))
    if cpu:
        cfg = dict(cfg, surface_grid=32)
    # Target scalar fields leave scope immediately after generating immutable RGB.
    target_image, _, target_surface, _, target_timing = render(TorusTarget(DEVICE), cfg, seed=719)
    sphere_image, _, sphere_surface, _, _ = render(SphereReference(DEVICE), cfg, seed=719)
    sphere_like_image, _, sphere_like_surface, _, _ = render(EllipsoidTarget(DEVICE), cfg, seed=719)
    np.save(CACHE/"target_torus.npy", target_image); np.save(CACHE/"target_sphere_like.npy", sphere_like_image)
    main_k = 64 if quick else (128 if cpu else 256)
    main_steps = 1 if quick else (4 if cpu else 6)
    runs = [
        optimize("A_dense_real_torus", target_image, cfg=cfg, centers=main_k, steps=main_steps),
        optimize("B_sphere_like", sphere_like_image, cfg=cfg, centers=main_k, steps=1 if quick else (2 if cpu else 4)),
        optimize("C_weak_torus", target_image, cfg=cfg, centers=main_k, selected_views=(2,), steps=1 if quick else (2 if cpu else 4)),
    ]
    if not quick:
        density = (32, 64) if cpu else (64, 128)
        runs.extend([optimize(f"D_density_{k}", target_image, cfg=cfg, centers=k,
                              steps=2 if cpu else 3) for k in density])
    main = runs[0]
    min_grad = min(row["geometry"]["spatial_gradient_quantiles"][0] for row in main["trajectory"])
    genus_changed = main["topology_initial"]["inferred_genus"] == 0 and main["topology_final"]["inferred_genus"] == 1
    loss_reduced = main["final_mse"] < .8*main["initial_mse"]
    report = {
        "version": "v095", "experiment": "dense high-dimensional local-manifold-jet sphere-to-torus image-only optimization",
        "initial_field": "F=||x||^2-1-||2x|| h_theta(x); theta=0 exactly", "jet_centers": "Fibonacci sphere directions, independent of target",
        "target": {"kind": "analytic real torus used only to render target/evaluate topology", "major_radius": TARGET_MAJOR,
                   "minor_radius": TARGET_MINOR, "topology": target_surface.topology, "render_seconds": target_timing["seconds"],
                   "image_digest": digest(torch.from_numpy(target_image))},
        "views": {"directions": camera_state()["directions"].tolist(), "informative": [0, 1], "weak_control": [2]},
        "renderer": "existing CURRENT finite-packet transmission, outward gate/current lobe, and C3 detector; real extracted surface quadrature refreshed per trial",
        "config": cfg, "execution_device": str(DEVICE), "runs": runs,
        "initial_validity": {"sphere_topology": sphere_surface.topology, "sphere_surface_residual_max": sphere_surface.residual_max,
                             "zero_coefficients_exact_field": True, "initial_image_digest": digest(torch.from_numpy(sphere_image))},
        "verdicts": {
            "INITIAL_DENSE_JET_IS_EXACT_SPHERE": sphere_surface.topology["inferred_genus"] == 0 and sphere_surface.residual_max < 1e-10,
            "TARGET_INFORMATION_IMAGE_ONLY": True, "MANY_LOCAL_PARAMETERS_OPTIMIZED": main["total_parameters"] >= 64,
            "FRESH_REAL_RETRACE_FOR_ACCEPTANCE": all(x["fresh_real_trace_each_trial"] for x in runs),
            "IMAGE_LOSS_SUBSTANTIALLY_REDUCED": loss_reduced, "REAL_TOPOLOGY_CHANGED_0_TO_1": genus_changed,
            "SPONTANEOUS_SPHERE_TO_TORUS_SUPPORTED": genus_changed and loss_reduced,
            "GENUINE_CRITICAL_EVENT_OBSERVED": min_grad < .05,
            "EMERGENT_ONE_DIMENSIONAL_DISCRIMINANT_IDENTIFIED": False,
            "COMPLEX_CONTINUATION_NEEDED": False,
        },
        "critical_policy": "No critical coordinate is predefined. Lowest-|grad F| locations and their compact support blocks are logged; a scalar discriminant is not asserted without trajectory evidence.",
        "density_adjustment": ("K={32,64,128}; CUDA was unavailable in the execution environment, so 256/512/1024 were not claimed. K=128 is the CPU main run and retains dense overlap."
                               if cpu else "K={64,128,256}; 512/1024 omitted because the existing exact full image-column Gram is cubic-solve/dense-image-column limited. K=256 retains many overlapping local DoFs."),
        "order_ablation": "Not run: density/required controls prioritized; p=0 is the requested clean scalar-normal test.",
        "runtime_seconds": time.perf_counter()-started,
    }
    write_json(REPORT if not quick else CACHE/"quick.json", report)
    print(json.dumps({"quick": quick, "main_initial": main["initial_mse"], "main_final": main["final_mse"],
                      "main_genus": main["topology_final"]["inferred_genus"], "seconds": report["runtime_seconds"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
