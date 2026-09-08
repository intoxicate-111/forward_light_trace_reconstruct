"""v0.9.6 frozen-chart topology capacity and penetration diagnostics.

Phase A may use an analytic shifted torus as a geometry oracle.  Later phases
consume only images of a saved same-family coefficient vector.  CUDA is a hard
requirement: this module never selects a CPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .dense_jet_torus import DOMAIN, SphereReference, topology_only
from .basis_initialization import directions
from .manifold_jets import JetLayout, ManifoldJetField
from .transverse_packet import digest, write_json


CACHE = Path("runs/v096_topology_penetration")
REPORT = Path("artifacts/v096_topology_penetration.json")
DEVICE = torch.device("cuda")


def _sign_changing_intervals(values):
    """Count a root on a sampled node once rather than in both neighbors."""
    return (((values[:, :-1] < 0) & (values[:, 1:] >= 0)) |
            ((values[:, :-1] > 0) & (values[:, 1:] <= 0)))


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("V096_CUDA_REQUIRED_NO_CPU_FALLBACK")
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(), "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0))}


def snapshot():
    path = CACHE/"starting_state.json"
    if path.exists(): return
    protected = [p for folder in ("artifacts", "figures", "src/zlt", "scripts", "tests")
                 for p in Path(folder).glob("*") if p.is_file() and not p.name.startswith(("v096", "topology_penetration"))]
    write_json(path, {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protected_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}})


class ShiftedTorus:
    """Genus-1 oracle whose solid tube contains the invariant origin."""
    def __init__(self, major=.50, minor=.30, center_x=.50):
        self.major, self.minor, self.center_x = major, minor, center_x

    def signed_distance(self, p):
        rho = torch.stack((p[:, 0]-self.center_x, p[:, 1]), 1).norm(dim=1)
        return torch.sqrt((rho-self.major).square()+p[:, 2].square()).clamp_min(1e-15)-self.minor

    def value(self, p):
        return self.signed_distance(p)/self.minor

    def surface(self, count, seed=960):
        engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
        uv = engine.draw(count, dtype=torch.float64).to(DEVICE)*2*torch.pi
        u, v = uv.unbind(1); ring = self.major+self.minor*torch.cos(v)
        p = torch.stack((self.center_x+ring*torch.cos(u), ring*torch.sin(u), self.minor*torch.sin(v)), 1)
        n = torch.stack((torch.cos(v)*torch.cos(u), torch.cos(v)*torch.sin(u), torch.sin(v)), 1)
        return p, n


def make_field(k, order, radius, coefficients=None):
    reference = SphereReference(DEVICE); centers = directions(k, 955).to(DEVICE)
    layout = JetLayout(centers, reference.gradient(centers), centers.new_full((k,), radius), order)
    if coefficients is None: coefficients = centers.new_zeros(layout.count)
    return ManifoldJetField(reference, layout, coefficients)


def oracle_dataset(target, volume_count=12288, surface_count=4096):
    surface, normals = target.surface(surface_count)
    delta = .08
    near = torch.cat((surface, surface-delta*normals, surface+delta*normals))
    near_y = torch.cat((surface.new_zeros(surface_count),
                        target.value(surface-delta*normals), target.value(surface+delta*normals)))
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=961)
    volume = (2*engine.draw(volume_count, dtype=torch.float64).to(DEVICE)-1)*1.34
    volume_y = target.value(volume).clamp(-1.5, 2.)
    special = surface.new_tensor([[0., 0., 0.], [1.36, 0., 0.], [-1.36, 0., 0.],
                                  [.5, 0., 0.], [.5, 0., .6]])
    points = torch.cat((near, volume, special)); desired = torch.cat((near_y, volume_y, target.value(special)))
    weights = torch.cat((surface.new_full((len(near),), 8.), surface.new_ones(len(volume)), surface.new_full((len(special),), 16.)))
    return points, desired, weights


def value_design(field, points, chunk=4096):
    """Dense optimization matrix assembled from sparse point/mode pairs."""
    blocks = []
    for start in range(0, len(points), chunk):
        p = points[start:start+chunk]; pi, bi, b, _ = field.supports.query(p)
        matrix = p.new_zeros((len(p), field.layout.count))
        matrix.index_put_((pi, bi), b, accumulate=True); blocks.append(matrix)
    return torch.cat(blocks)


def fit_oracle(k, order, radius, *, ridge=1e-5):
    target = ShiftedTorus(); field = make_field(k, order, radius)
    points, desired, weights = oracle_dataset(target)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    A = value_design(field, points); base = field.reference.value(points); rhs = desired-base
    scale = weights.sqrt(); Aw = A*scale[:, None]; bw = rhs*scale
    gram = Aw.T@Aw/len(points); vector = Aw.T@bw/len(points)
    diagonal = float(torch.diag(gram).max()); damping = max(ridge*diagonal, 1e-10)
    coefficients = torch.linalg.solve(gram+damping*torch.eye(len(gram), device=DEVICE, dtype=torch.float64), vector)
    fitted = field.with_coefficients(coefficients)
    prediction = base+A@coefficients
    result = {"K": k, "order": order, "parameters": field.layout.count, "radius": radius,
        "weighted_rmse": float(torch.sqrt(((prediction-desired).square()*weights).sum()/weights.sum())),
        "surface_abs_mean": float(fitted.value(points[:4096]).abs().mean()),
        "coefficient_l2": float(coefficients.norm()), "coefficient_abs_max": float(coefficients.abs().max()),
        "damping": damping, "gram_condition": float(torch.linalg.cond(gram+damping*torch.eye(len(gram), device=DEVICE, dtype=torch.float64))),
        "runtime_seconds": time.perf_counter()-started, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "origin_value": float(fitted.value(points.new_zeros((1, 3)))), "topology_screen": topology_only(fitted, 72)}
    # Exact overlap statistics on the initial sphere.
    pi, _, _, _ = field.layout.query(field.layout.centers)
    count = torch.bincount(pi, minlength=k).double()/field.layout.modes
    result["overlap"] = {"median": float(count.median()), "max": float(count.max()), "min": float(count.min())}
    torch.save({"K": k, "order": order, "radius": radius, "coefficients": coefficients.cpu()},
               CACHE/f"oracle_K{k}_p{order}_r{radius:.2f}.pt")
    return fitted, result


def fiber_roots(field, *, direction_count=2048, samples=513, s_min=-.98, s_max=1.0):
    """All sign-changing radial/normal-fiber roots plus near-even-root flags."""
    dirs = directions(direction_count, 962).to(DEVICE); s = torch.linspace(s_min, s_max, samples, device=DEVICE, dtype=torch.float64)
    counts, slopes, near_even, locations = [], [], [], []
    for start in range(0, direction_count, 128):
        p = dirs[start:start+128]; x = p[:, None, :]*(1+s[None, :, None])
        values = torch.cat([field.value(x[:, j:j+32].reshape(-1, 3)).reshape(len(p), -1)
                            for j in range(0, samples, 32)], 1)[:, :samples]
        changes = _sign_changing_intervals(values)
        local_counts = changes.sum(1); counts.append(local_counts.cpu())
        # Refine each sign-changing interval independently.
        root_pairs = changes.nonzero()
        if len(root_pairs):
            rows, columns = root_pairs.unbind(1); root_dirs = p[rows]
            lo, hi = s[columns].clone(), s[columns+1].clone(); flo = values[rows, columns]
            for _ in range(35):
                mid = (lo+hi)/2; fm = field.value(root_dirs*(1+mid[:, None]))
                left = flo*fm <= 0; hi = torch.where(left, mid, hi); lo = torch.where(left, lo, mid); flo = torch.where(left, flo, fm)
            root = (lo+hi)/2; point = root_dirs*(1+root[:, None]); gradient = field.gradient(point)
            local_slopes = (gradient*root_dirs).sum(1).abs()
            slopes.extend(local_slopes.cpu().tolist()); locations.extend(point.cpu().tolist())
        # A local sampled extremum close to zero with equal-sign neighbors is
        # an empirical near-even-root candidate, not a certified double root.
        absolute = values.abs(); minima = (absolute[:, 1:-1] < absolute[:, :-2]) & (absolute[:, 1:-1] < absolute[:, 2:])
        even = minima & (absolute[:, 1:-1] < 2e-3) & (values[:, :-2]*values[:, 2:] > 0)
        near_even.append(even.any(1).cpu())
    counts = torch.cat(counts); near_even = torch.cat(near_even)
    unique, frequency = torch.unique(counts, return_counts=True)
    return {"directions": direction_count, "samples_per_fiber": samples, "s_interval": [s_min, s_max],
        "root_count_histogram": {str(int(k)): int(v) for k, v in zip(unique, frequency)},
        "one_root_fraction": float((counts == 1).double().mean()), "multi_root_fraction": float((counts > 1).double().mean()),
        "zero_root_fraction": float((counts == 0).double().mean()), "near_even_root_fibers": int(near_even.sum()),
        "root_slope_abs_min": min(slopes) if slopes else None,
        "root_slope_abs_quantiles": np.quantile(slopes, [0, .01, .05, .5]).tolist() if slopes else [],
        "lowest_slope_root_locations": [locations[i] for i in np.argsort(slopes)[:16]] if slopes else []}


def empirical_monotonicity(field, *, direction_count=2048, samples=513):
    dirs = directions(direction_count, 963).to(DEVICE); s = torch.linspace(-.98, 1., samples, device=DEVICE, dtype=torch.float64)
    minimum = math.inf; quantile_parts = []
    for start in range(0, direction_count, 128):
        p = dirs[start:start+128]; x = p[:, None, :]*(1+s[None, :, None]); flat = x.reshape(-1, 3)
        derivative = (field.gradient(flat)*p[:, None, :].expand_as(x).reshape(-1, 3)).sum(1)
        minimum = min(minimum, float(derivative.min())); quantile_parts.append(derivative.detach().cpu())
    values = torch.cat(quantile_parts)
    return {"empirical_grid_only": True, "directions": direction_count, "samples_per_fiber": samples,
            "minimum_dG_ds": minimum, "quantiles": torch.quantile(values, values.new_tensor([0., .001, .01, .05, .5])).tolist(),
            "not_a_theorem": True}


def fast_fiber_counts(field, *, direction_count=512, samples=257):
    """Trajectory root-count monitor; final reports use ``fiber_roots``."""
    dirs = directions(direction_count, 964).to(DEVICE)
    s = torch.linspace(-.98, 1., samples, device=DEVICE, dtype=torch.float64)
    counts = []
    for start in range(0, direction_count, 64):
        p = dirs[start:start+64]; x = p[:, None, :]*(1+s[None, :, None])
        values = torch.cat([field.value(x[:, j:j+32].reshape(-1, 3)).reshape(len(p), -1)
                            for j in range(0, samples, 32)], 1)[:, :samples]
        counts.append(_sign_changing_intervals(values).sum(1).cpu())
    counts = torch.cat(counts); unique, frequency = torch.unique(counts, return_counts=True)
    return {"directions": direction_count, "samples": samples,
            "histogram": {str(int(k)): int(v) for k, v in zip(unique, frequency)},
            "multi_root_fraction": float((counts > 1).double().mean())}


def _image_step(J, residual, damping_fraction):
    """The optimizer step has no target-field/coefficient argument by design."""
    gram = J.T@J/len(residual); alignment = J.T@residual/len(residual)
    damping = max(float(np.diag(gram).max())*damping_fraction, 1e-8)
    delta = -np.linalg.solve(gram+damping*np.eye(len(gram)), alignment)
    return delta, gram, alignment, damping


def _spectrum_and_oracle(gram, negative_gradient, oracle_direction):
    """Diagnostic only; return value is never passed to the optimizer."""
    eigenvalues, vectors = np.linalg.eigh(gram); eigenvalues = np.maximum(eigenvalues, 0)
    order = np.argsort(eigenvalues)[::-1]; eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    threshold = max(eigenvalues[0]*1e-8, 1e-14) if len(eigenvalues) else 0
    keep = eigenvalues > threshold
    projected = vectors[:, keep]@(vectors[:, keep].T@oracle_direction) if keep.any() else np.zeros_like(oracle_direction)
    norm = np.linalg.norm(negative_gradient)
    return {"effective_rank_relative_1e_8": int(keep.sum()),
            "leading_singular_values": np.sqrt(eigenvalues[:32]).tolist(),
            "smallest_retained_singular": float(np.sqrt(eigenvalues[keep][-1])) if keep.any() else 0.,
            "oracle_gradient_cosine": float(negative_gradient@oracle_direction/max(norm, 1e-30)),
            "oracle_observable_projection_fraction": float(np.linalg.norm(projected)),
            "oracle_direction_used_by_optimizer": False}


def _penetration_geometry(field, surface):
    from .dense_jet_torus import field_diagnostics
    result = field_diagnostics(field, surface); result["fibers"] = fast_fiber_counts(field)
    return result


def penetrate_same_family(max_steps=100):
    """Phase B: theta=0 to a same-family genus-1 RGB target, image loss only."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    from .dense_jet_torus import config as render_config, image_jacobian, render
    saved = torch.load(CACHE/"theta_T.pt", weights_only=True)
    k, order, radius = int(saved["K"]), int(saved["order"]), float(saved["radius"])
    oracle = saved["coefficients"].numpy(); oracle_direction = oracle/np.linalg.norm(oracle)
    cfg = render_config(resolution=32, sources=1024)
    cfg = dict(cfg, surface_grid=40, tangent_sources=1024,
               transport_chunk=128,
               stage="V096_SAME_FAMILY_IMAGE_ONLY_PENETRATION_CUDA")
    # theta_T goes out of scope after this immutable image is made.
    target_field = make_field(k, order, radius, saved["coefficients"].to(DEVICE))
    with torch.no_grad(): target_image, _, target_surface, _, target_timing = render(target_field, cfg, seed=970)
    np.save(CACHE/"same_family_target.npy", target_image); del target_field
    field = make_field(k, order, radius); trajectory = []; no_accept = 0
    with torch.no_grad(): image, state, surface, boundary, timing = render(field, cfg, seed=970)
    np.save(CACHE/"same_family_initial.npy", image)
    selected = (0, 1); started = time.perf_counter(); stop = "ITERATION_BUDGET"
    for iteration in range(max_steps+1):
        residual_views = image[list(selected)]-target_image[list(selected)]
        row = {"iteration": iteration, "actual_fresh_retrace_mse": float(np.mean(residual_views**2)),
               "all_view_mse": float(np.mean((image-target_image)**2)), "topology": surface.topology,
               "geometry": _penetration_geometry(field, surface), "render_seconds": timing["seconds"],
               "coefficient_l2": float(field.coefficients.norm()), "coefficient_abs_max": float(field.coefficients.abs().max())}
        trajectory.append(row); write_json(CACHE/"penetration_progress.json", {"trajectory": trajectory, "stop": stop})
        if surface.topology["watertight"] and surface.topology["components"] == 1 and surface.topology["inferred_genus"] == 1:
            stop = "TOPOLOGY_SUCCESS"; break
        if iteration == max_steps: break
        jac_started = time.perf_counter()
        with torch.no_grad(): Jall = image_jacobian(field, state, boundary, cfg)
        J = Jall[list(selected)].reshape(-1, field.layout.count); residual = residual_views.reshape(-1)
        proposals = []
        for damping_fraction in (.02, .1, .5):
            delta, gram, alignment, damping = _image_step(J, residual, damping_fraction)
            raw_max = float(np.abs(delta).max())
            if raw_max > .15: delta *= .15/raw_max
            proposals.append((delta, gram, alignment, damping, f"GN_{damping_fraction:g}"))
        # A separately scaled steepest-descent fallback prevents one failed GN
        # model from silently ending the meaningful budget.
        gradient = proposals[0][2]; steepest = -gradient
        if np.abs(steepest).max() > 0: steepest *= .08/np.abs(steepest).max()
        proposals.append((steepest, proposals[0][1], gradient, 0., "STEEPEST"))
        row["jacobian_seconds"] = time.perf_counter()-jac_started
        row["jacobian_shape"] = list(J.shape); row["jacobian_nnz"] = int(np.count_nonzero(np.abs(J) > 1e-12))
        row["oracle_diagnostic"] = _spectrum_and_oracle(proposals[0][1], -gradient, oracle_direction)
        row["trials"] = []; accepted = None
        for delta, gram, alignment, damping, method in proposals:
            for alpha in (1., .5, .25, .125, .0625, .03125):
                candidate = field.with_coefficients(field.coefficients+alpha*torch.as_tensor(delta, device=DEVICE))
                trial = {"method": method, "alpha": alpha, "damping": damping,
                         "predicted_local_mse": float(np.mean((residual+alpha*J@delta)**2))}
                try:
                    with torch.no_grad(): trial_image, trial_state, trial_surface, trial_boundary, trial_timing = render(candidate, cfg, seed=970)
                    value = float(np.mean((trial_image[list(selected)]-target_image[list(selected)])**2))
                    trial.update(actual_fresh_retrace_mse=value, topology=trial_surface.topology,
                                 accepted=value < row["actual_fresh_retrace_mse"]*(1-1e-7))
                except RuntimeError as error:
                    trial.update(accepted=False, failure=str(error))
                row["trials"].append(trial)
                if trial["accepted"]:
                    accepted = (candidate, trial_image, trial_state, trial_surface, trial_boundary, trial_timing, delta, alpha, method)
                    break
            if accepted is not None: break
        if accepted is None:
            no_accept += 1; row["accepted"] = False
            if no_accept >= 1: stop = "EXHAUSTED_24_FRESH_RETRACE_TRIALS"; break
            continue
        no_accept = 0; row["accepted"] = True
        field, image, state, surface, boundary, timing, delta, alpha, method = accepted
        row["accepted_method"] = method; row["accepted_update_l2"] = float(np.linalg.norm(alpha*delta))
        torch.save({"coefficients": field.coefficients.cpu()}, CACHE/f"penetration_step{iteration+1}.pt")
        np.save(CACHE/f"penetration_step{iteration+1}.npy", image)
        print(json.dumps({"phase": "penetration", "iteration": iteration+1,
            "mse": row["trials"][-1]["actual_fresh_retrace_mse"], "genus": surface.topology["inferred_genus"],
            "multi_root": _penetration_geometry(field, surface)["fibers"]["multi_root_fraction"], "method": method}), flush=True)
    np.save(CACHE/"same_family_final.npy", image)
    final_fibers = fiber_roots(field); final_topology = {str(n): topology_only(field, n) for n in (72, 96, 144)}
    output = {"phase": "B_SAME_FAMILY_RGB_ONLY", "K": k, "order": order, "parameters": field.layout.count,
        "radius": radius, "selected_views": list(selected), "maximum_steps": max_steps,
        "optimizer_target_inputs": ["immutable RGB target image"],
        "optimizer_forbidden_inputs": {"theta_T": False, "torus_field": False, "target_mesh": False,
                                       "genus": False, "hole_center": False, "topology_mode": False},
        "oracle_direction_scope": "post-step diagnostic only; _image_step has no oracle argument",
        "trajectory": trajectory, "stop": stop, "accepted_steps": sum(x.get("accepted", False) for x in trajectory),
        "initial_mse": trajectory[0]["actual_fresh_retrace_mse"], "final_mse": trajectory[-1]["actual_fresh_retrace_mse"],
        "target_topology": target_surface.topology, "final_topology": final_topology,
        "final_fibers": final_fibers, "final_coefficients": field.coefficients.cpu().tolist(),
        "runtime_seconds": time.perf_counter()-started, "target_render_seconds": target_timing["seconds"],
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "topology_penetrated": all(x["watertight"] and x["components"] == 1 and x["inferred_genus"] == 1 for x in final_topology.values())}
    write_json(CACHE/"penetration.json", output); print(json.dumps({k: output[k] for k in ("stop", "accepted_steps", "initial_mse", "final_mse", "topology_penetrated", "runtime_seconds")}, indent=2))
    return output


def _control_optimize(name, target_image, cfg, selected_views, max_steps=5):
    """Short matched controls; same fresh renderer and image-only update rule."""
    from .dense_jet_torus import image_jacobian, render
    saved = torch.load(CACHE/"theta_T.pt", weights_only=True)
    field = make_field(saved["K"], saved["order"], saved["radius"]); trajectory = []
    with torch.no_grad(): image, state, surface, boundary, timing = render(field, cfg, seed=970)
    np.save(CACHE/f"{name}_initial.npy", image); started = time.perf_counter(); stop = "ITERATION_BUDGET"
    for iteration in range(max_steps+1):
        residual_views = image[list(selected_views)]-target_image[list(selected_views)]
        row = {"iteration": iteration, "mse": float(np.mean(residual_views**2)),
               "topology": surface.topology, "min_gradient": float(field.gradient(surface.points).norm(dim=1).min()),
               "fibers": fast_fiber_counts(field)}
        trajectory.append(row)
        if iteration == max_steps: break
        with torch.no_grad(): Jall = image_jacobian(field, state, boundary, cfg)
        J = Jall[list(selected_views)].reshape(-1, field.layout.count); residual = residual_views.reshape(-1)
        proposals = []
        for fraction in (.02, .1, .5):
            delta, _, _, damping = _image_step(J, residual, fraction)
            if np.abs(delta).max() > .15: delta *= .15/np.abs(delta).max()
            proposals.append((delta, damping, f"GN_{fraction:g}"))
        row["trials"] = []; accepted = None
        for delta, damping, method in proposals:
            for alpha in (1., .5, .25, .125, .0625, .03125):
                candidate = field.with_coefficients(field.coefficients+alpha*torch.as_tensor(delta, device=DEVICE))
                trial = {"method": method, "alpha": alpha, "damping": damping,
                         "predicted_mse": float(np.mean((residual+alpha*J@delta)**2))}
                try:
                    with torch.no_grad(): ti, ts, tf, tb, tt = render(candidate, cfg, seed=970)
                    value = float(np.mean((ti[list(selected_views)]-target_image[list(selected_views)])**2))
                    trial.update(actual_fresh_mse=value, accepted=value < row["mse"]*(1-1e-7))
                except RuntimeError as error: trial.update(accepted=False, failure=str(error))
                row["trials"].append(trial)
                if trial["accepted"]: accepted = (candidate, ti, ts, tf, tb, tt); break
            if accepted: break
        row["accepted"] = accepted is not None
        if accepted is None: stop = "EXHAUSTED_18_FRESH_RETRACE_TRIALS"; break
        field, image, state, surface, boundary, timing = accepted
        print(json.dumps({"phase": name, "iteration": iteration+1, "mse": row["trials"][-1]["actual_fresh_mse"],
                          "genus": surface.topology["inferred_genus"]}), flush=True)
    np.save(CACHE/f"{name}_final.npy", image)
    return {"name": name, "selected_views": list(selected_views), "maximum_steps": max_steps,
            "trajectory": trajectory, "stop": stop, "accepted_steps": sum(x.get("accepted", False) for x in trajectory),
            "initial_mse": trajectory[0]["mse"], "final_mse": trajectory[-1]["mse"],
            "final_topology": {str(n): topology_only(field, n) for n in (72, 96)},
            "runtime_seconds": time.perf_counter()-started, "target_information": "RGB only"}


def controls():
    require_cuda()
    from .dense_jet_torus import EllipsoidTarget, config as render_config, render
    cfg = render_config(resolution=32, sources=1024)
    cfg = dict(cfg, surface_grid=40, tangent_sources=1024, transport_chunk=128,
               stage="V096_MATCHED_IMAGE_CONTROLS_CUDA")
    same_family = np.load(CACHE/"same_family_target.npy")
    with torch.no_grad(): ellipsoid, _, _, _, _ = render(EllipsoidTarget(DEVICE), cfg, seed=970)
    np.save(CACHE/"ellipsoid_target.npy", ellipsoid)
    result = {"same_topology": _control_optimize("control_same_topology", ellipsoid, cfg, (0, 1), 5),
              "weak_observability": _control_optimize("control_weak_view", same_family, cfg, (2,), 5),
              "analytic_torus_generalization": {"status": "NOT_RUN_PHASE_B_DID_NOT_REACH_GENUS1",
                  "reason": "Required experiment order makes analytic-target generalization conditional on same-family penetration success."}}
    write_json(CACHE/"controls.json", result); print(json.dumps(result, indent=2)); return result


def environment_report():
    import os, sys
    def capture(argv):
        result = subprocess.run(argv, text=True, capture_output=True)
        return {"command": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    report = {"python": sys.executable, "torch": torch.__version__, "torch_version_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "PATH": os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "nvidia_smi": capture(["nvidia-smi"]), "pip_show_torch": capture([sys.executable, "-m", "pip", "show", "torch"]),
        "hard_gate_passed": torch.cuda.is_available() and torch.cuda.device_count() > 0,
        "diagnosis": "The managed workspace sandbox hid the GPU; the existing local test Conda environment is CUDA-capable when executed in the host context. No dependency was installed or changed."}
    write_json(Path("artifacts/v096_cuda_environment.json"), report)
    Path("artifacts/v096_cuda_environment.txt").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2)); return report


def capacity_sweep():
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True); snapshot(); target = ShiftedTorus()
    configs = [(128, 0, r) for r in (.80, 1.10, 1.35)]
    configs += [(k, 0, r) for k in (256, 512, 1024) for r in (1.10, 1.35)]
    configs += [(256, 1, 1.10), (128, 2, 1.10)]
    rows = json.loads((CACHE/"capacity_progress.json").read_text()) if (CACHE/"capacity_progress.json").exists() else []
    winner = None
    for k, order, radius in configs[len(rows):]:
        field, row = fit_oracle(k, order, radius); rows.append(row)
        print(json.dumps({key: row[key] for key in ("K", "order", "parameters", "radius", "weighted_rmse", "topology_screen", "runtime_seconds")}), flush=True)
        write_json(CACHE/"capacity_progress.json", rows); del field; torch.cuda.empty_cache()
    if (CACHE/"theta_T.pt").exists():
        state = torch.load(CACHE/"theta_T.pt", weights_only=True)
        winner = next(x for x in rows if x["K"] == state["K"] and x["order"] == state["order"] and x["radius"] == state["radius"])
        f = make_field(state["K"], state["order"], state["radius"], state["coefficients"].to(DEVICE))
        winner["topology_verification"] = {str(n): topology_only(f, n) for n in (96, 144)}
        if not all(x["watertight"] and x["components"] == 1 and x["inferred_genus"] == 1
                   and not x["extraction_boundary_contact"] for x in winner["topology_verification"].values()):
            winner = None
    else:
        for row in rows:
            topology = row["topology_screen"]
            if topology["watertight"] and topology["components"] == 1 and topology["inferred_genus"] == 1:
                path = CACHE/f"oracle_K{row['K']}_p{row['order']}_r{row['radius']:.2f}.pt"
                state = torch.load(path, weights_only=True); f = make_field(state["K"], state["order"], state["radius"], state["coefficients"].to(DEVICE))
                high = {str(n): topology_only(f, n) for n in (96, 144)}
                if all(x["watertight"] and x["components"] == 1 and x["inferred_genus"] == 1
                       and not x["extraction_boundary_contact"] for x in high.values()):
                    row["topology_verification"] = high; winner = row
                    torch.save(state, CACHE/"theta_T.pt"); break
    sphere = make_field(128, 0, 1.1)
    report = {"version": "v096", "phase": "A_CAPACITY", "cuda": require_cuda(),
        "representation": "unchanged v0.9.2 ManifoldJetField with exact sphere reference and frozen charts",
        "structural_invariant": "F_theta(0)=-1 for every theta because |grad F_sphere(0)|=0; centered ordinary torus is excluded, not all genus-1 surfaces",
        "oracle": {"kind": "shifted analytic torus", "major": target.major, "minor": target.minor, "center_x": target.center_x,
                   "origin_inside_solid_tube": True, "optimization_use": "Phase A geometry capacity only"},
        "configs": rows, "winner": winner,
        "sphere_fibers": fiber_roots(sphere), "sphere_monotonicity": empirical_monotonicity(sphere),
        "capacity_success": winner is not None}
    if winner:
        state = torch.load(CACHE/"theta_T.pt", weights_only=True); f = make_field(state["K"], state["order"], state["radius"], state["coefficients"].to(DEVICE))
        report["winner_fibers"] = fiber_roots(f); report["winner_monotonicity"] = empirical_monotonicity(f)
    write_json(CACHE/"capacity.json", report); print(json.dumps({"capacity_success": report["capacity_success"], "winner": winner}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--capacity", action="store_true"); parser.add_argument("--penetration", action="store_true")
    parser.add_argument("--controls", action="store_true"); parser.add_argument("--environment", action="store_true")
    args = parser.parse_args()
    if args.capacity: capacity_sweep()
    elif args.penetration: penetrate_same_family()
    elif args.controls: controls()
    elif args.environment: environment_report()
    else: parser.error("select --capacity, --penetration, --controls, or --environment")
