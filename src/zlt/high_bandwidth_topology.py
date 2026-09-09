"""v0.9.7 high-bandwidth topology observability and penetration diagnostics.

The renderer and CURRENT/C3 differential equations are unchanged.  Full-HD
derivatives are accumulated as emitter-streamed products: a dense
``[pixels, parameters]`` Jacobian is never formed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .basis_initialization import directions
from .boundary_transport import ObservationSphere
from .dense_jet_torus import (DOMAIN, SphereReference, camera_state,
                              config as base_config, extract_surface, render,
                              topology_only)
from .density_matrix import capture_splat
from .manifold_jets import JetLayout, ManifoldJetField
from .finite_packet import _compact_shell
from .matched_jacobian import integrated_kernel
from .measurement_bandwidth import gate, lobe
from .meshfree_surface import meshfree_base_color
from .transverse_packet import transmission_prefixes, write_json

DEVICE = torch.device("cuda")
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs/v097_high_bandwidth_topology"
TARGET_PATH = ROOT / "artifacts/v096_theta_T.pt"
TARGET_SHA256 = "e137b58e55d8184b67feecf537575435a6a5ce58b0541d667dc08311592e9d4c"
SELECTED_VIEWS = (0, 1)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("V097_CUDA_REQUIRED_NO_CPU_FALLBACK")
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(), "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0))}


def environment_report():
    require_cuda()
    def capture(argv):
        result = subprocess.run(argv, text=True, capture_output=True)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    result = {**require_cuda(), "python": sys.executable,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
              "PATH": os.environ.get("PATH", ""), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
              "nvidia_smi": capture(["nvidia-smi"]),
              "pip_show_torch": capture([sys.executable, "-m", "pip", "show", "torch"]),
              "hard_gate_passed": True, "cpu_fallback": False, "hpc_used": False}
    write_json(ROOT / "artifacts/v097_cuda_environment.json", result)
    (ROOT / "artifacts/v097_cuda_environment.txt").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return result


def load_target():
    digest = hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest()
    if digest != TARGET_SHA256:
        raise RuntimeError(f"V096_THETA_DIGEST_MISMATCH:{digest}")
    saved = torch.load(TARGET_PATH, weights_only=True)
    reference = SphereReference(DEVICE)
    centers = directions(int(saved["K"]), 955).to(DEVICE)
    layout = JetLayout(centers, reference.gradient(centers),
                       centers.new_full((len(centers),), float(saved["radius"])), int(saved["order"]))
    target = ManifoldJetField(reference, layout, saved["coefficients"].to(DEVICE))
    sphere = ManifoldJetField(reference, layout, torch.zeros_like(target.coefficients))
    return sphere, target, {"path": str(TARGET_PATH.relative_to(ROOT)), "sha256": digest,
                            "K": int(saved["K"]), "order": int(saved["order"]),
                            "radius": float(saved["radius"]), "parameters": layout.count}


def configuration(height, width, sources):
    cfg = base_config(resolution=height, sources=sources)
    cfg.update(resolution=[height, width], count=sources, tangent_sources=sources,
               face_quadrature=False, surface_grid=40, transport_chunk=512,
               stage="V097_HIGH_BANDWIDTH_CURRENT_C3_CUDA",
               note="Exact existing real CURRENT/C3 path; deterministic area-Sobol source quadrature.")
    return cfg


def _state_path(label, sources):
    return CACHE / f"scene_{label}_{sources}.pt"


def build_scene(field, label, sources):
    """Fresh geometry/transport once; detector resolutions replay this state."""
    path = _state_path(label, sources)
    if path.exists():
        saved = torch.load(path, weights_only=False)
        return saved["state"], saved["topology"], saved["runtime_seconds"], True
    cfg = configuration(512, 512, sources)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    image, state, surface, _, timing = render(field, cfg, seed=970)
    payload = {"state": state, "topology": surface.topology, "runtime_seconds": timing["seconds"],
               "source_count": len(state["weights"]), "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20}
    torch.save(payload, path)
    np.save(CACHE / f"measurement_{label}_{sources}_512x512.npy", image)
    return state, surface.topology, timing["seconds"], False


def measure_state(state, cfg):
    """Exact current C3 measurement replay from fixed transported source state."""
    h, w = cfg["resolution"]
    p = state["positions"].to(DEVICE); normal = state["normals"].to(DEVICE)
    images = []
    for view in range(len(state["directions"])):
        d = state["directions"][view].to(DEVICE).expand_as(p)
        cosine = (normal*d).sum(1)
        transmission = state["transmission"][view].to(DEVICE)
        energy = (state["weights"].to(DEVICE)[:, None]*transmission[:, None]*state["colors"].to(DEVICE)
                  * gate(cosine, "CURRENT_SOFT_W005")[:, None]*lobe(cosine, "CURRENT_LOBE")[:, None])
        relative = p-state["center"].to(DEVICE)
        row = (.5-relative@state["up"][view].to(DEVICE)/cfg["extent"])*h-.5
        col = (relative@state["right"][view].to(DEVICE)/cfg["extent"]+.5)*w-.5
        accumulator = torch.zeros((h*w, 3), dtype=torch.float64, device=DEVICE)
        capture_splat(accumulator, row, col, energy, cfg["capture"], resolution=(h, w))
        images.append((accumulator.reshape(h, w, 3)*(cfg["gain"]*h*w)).cpu().numpy())
    result = np.stack(images)
    if not np.isfinite(result).all():
        raise RuntimeError("NONFINITE_HIGH_BANDWIDTH_MEASUREMENT")
    return result


def _sparse_source_edges(field, p, nref):
    """Implicit source/normal tangent on compact-support COO edges only."""
    values = field.evaluate(p); spatial_gradient = values[:, 1:]
    gradient_norm = spatial_gradient.norm(dim=1)
    normal = spatial_gradient/gradient_norm[:, None].clamp_min(1e-30)
    true_gradient, hessian = field.jet(p)
    denominator = (true_gradient*nref).sum(1)
    point_ids, basis_ids, basis_value, basis_gradient = field.supports.query(p)
    if len(point_ids) and bool((denominator[point_ids].abs() < 1e-10).any()):
        raise RuntimeError("DEGENERATE_ATTACHMENT_DERIVATIVE")
    dx = -nref[point_ids]*(basis_value/denominator[point_ids])[:, None]
    dg = basis_gradient+torch.einsum("eab,eb->ea", hessian[point_ids], dx)
    dn = (dg-normal[point_ids]*(dg*normal[point_ids]).sum(1, keepdim=True))/gradient_norm[point_ids, None]
    return normal, point_ids, basis_ids, dx, dn


def _sparse_optical_edges(field, p, d, maximum, point_ids, basis_ids, dx, cfg, path_chunk=32):
    """CURRENT optical tangent accumulated directly on (emitter,basis) edges."""
    start = cfg["launch"]*(cfg["h"]+cfg["epsilon"]); step = cfg["path_step"]
    epsilon = cfg["epsilon"]; eta = cfg["eta"]
    steps = max(1, math.ceil(float((maximum-start).clamp_min(0).max())/step))
    count = field.layout.count; tau = p.new_zeros(len(p)); spatial = p.new_zeros((len(p), 3))
    direct_keys, direct_values = [], []
    for beginning in range(0, steps, path_chunk):
        times = start+(torch.arange(beginning, min(beginning+path_chunk, steps), device=DEVICE, dtype=p.dtype)+.5)*step
        valid = times[None] < maximum[:, None]
        points = p[:, None]+times[None, :, None]*d[:, None]
        flat = points.reshape(-1, 3); values = field.evaluate(flat); f, g = values[:, 0], values[:, 1:]
        norm = (g.square().sum(1)+eta*eta).sqrt().clamp_min(1e-30); q = f/(norm*epsilon)
        owner = torch.arange(len(p), device=DEVICE).repeat_interleave(len(times)); direction = d[owner]
        dot = (g*direction).sum(1)/norm
        influence = _compact_shell(q)/epsilon*dot.abs()*valid.flatten()
        tau += influence.reshape(len(p), len(times)).sum(1)*cfg["kappa"]*step
        hit = (q.abs() < 1)&valid.flatten(); ids = hit.nonzero().flatten()
        if not len(ids):
            continue
        x = flat[ids]; own = owner[ids]; gv = g[ids]; nv = norm[ids]
        fv = f[ids]; qv = q[ids]; av = dot[ids]; dv = direction[ids]
        psi = _compact_shell(qv)/epsilon; dpsi = (-15/4*qv*(1-qv*qv))/epsilon**2
        af = dpsi*av.abs()/nv
        ag = -dpsi[:, None]*av.abs()[:, None]*fv[:, None]*gv/nv[:, None]**3
        ag += psi[:, None]*av.sign()[:, None]*(dv/nv[:, None]-(gv*dv).sum(1)[:, None]*gv/nv[:, None]**3)
        factor = cfg["kappa"]*step; af *= factor; ag *= factor
        true_gradient, hessian = field.jet(x)
        spatial.index_add_(0, own, af[:, None]*true_gradient+torch.einsum("eab,ea->eb", hessian, ag))
        local_point, local_basis, b, db = field.supports.query(x)
        if len(local_point):
            direct_keys.append(own[local_point]*count+local_basis)
            direct_values.append(af[local_point]*b+(ag[local_point]*db).sum(1))
    keys = [point_ids*count+basis_ids]
    values = [(spatial[point_ids]*dx).sum(1)]
    if direct_keys:
        keys.extend(direct_keys); values.extend(direct_values)
    sparse = torch.sparse_coo_tensor(torch.cat(keys)[None], torch.cat(values),
                                     size=(len(p)*count,), dtype=p.dtype, device=DEVICE).coalesce()
    flat_key = sparse.indices()[0]
    return tau, torch.div(flat_key, count, rounding_mode="floor"), flat_key % count, sparse.values()


def _tangent_core(field, state, owners, view, boundary, cfg):
    """Existing CURRENT tangent represented only by sparse emitter/basis edges."""
    p = state["positions"][owners].to(DEVICE); normal = state["normals"][owners].to(DEVICE)
    nref = state["normals"][owners].to(DEVICE)
    d = state["directions"][view].to(DEVICE).expand_as(p); cosine = (normal*d).sum(1)
    keep = cosine > 0
    p, nref, d, cosine = p[keep], nref[keep], d[keep], cosine[keep]
    if not len(p):
        return None
    normal, source_point, source_basis, source_dx, source_dn = _sparse_source_edges(field, p, nref)
    tau, optical_point, optical_basis, optical_dtau = _sparse_optical_edges(
        field, p, d, boundary.exit_times(p, d), source_point, source_basis, source_dx, cfg)
    transmission = state["transmission"][view, owners].to(DEVICE)[keep]
    error = float((torch.exp(-tau)-transmission).abs().max())
    if error > 1e-8:
        raise RuntimeError(f"CURRENT_TANGENT_FORWARD_MISMATCH:{error}")
    color = state["colors"][owners].to(DEVICE)[keep]
    weight = state["weights"][owners].to(DEVICE)[keep]
    gate_value = gate(cosine, "CURRENT_SOFT_W005"); ell = lobe(cosine, "CURRENT_LOBE")
    energy = weight[:, None]*transmission[:, None]*color*(gate_value*ell)[:, None]
    count = field.layout.count
    combined_keys = torch.cat((source_point*count+source_basis, optical_point*count+optical_basis))
    unique_keys, inverse = torch.unique(combined_keys, sorted=True, return_inverse=True)
    point_ids = torch.div(unique_keys, count, rounding_mode="floor"); basis_ids = unique_keys % count
    source_inverse = inverse[:len(source_point)]; optical_inverse = inverse[len(source_point):]
    dx = p.new_zeros((len(unique_keys), 3)); dn = p.new_zeros((len(unique_keys), 3)); dtau = p.new_zeros(len(unique_keys))
    dx.index_add_(0, source_inverse, source_dx); dn.index_add_(0, source_inverse, source_dn)
    dtau.index_add_(0, optical_inverse, optical_dtau)
    dc = (dn*d[point_ids]).sum(1)
    dg = 2*cosine/cfg["gate_width"]**2*torch.exp(-cosine.square()/cfg["gate_width"]**2)
    grad_color = p.new_tensor([.72, .70, -.62])/(field.upper-field.lower)
    inside = ((p >= field.lower) & (p <= field.upper)).to(p.dtype)
    dcolor = dx*grad_color[None, :]*inside[point_ids]
    de = ((weight*transmission)[point_ids, None]
          *(dcolor*(gate_value*ell)[point_ids, None]
            +color[point_ids]*(dc*(dg[point_ids]*ell[point_ids]+.65*gate_value[point_ids]))[:, None]))
    de -= energy[point_ids]*dtau[:, None]
    return p, energy, point_ids, basis_ids, de, dx, error


def _sparse_detector_block(core, state, view, cfg, parameter_count, threshold=1e-12):
    """COO Jacobian for one emitter block with compressed touched detector rows."""
    p, energy, point_ids, basis_ids, de, dx, _ = core
    h, w = cfg["resolution"]; right = state["right"][view].to(DEVICE); up = state["up"][view].to(DEVICE)
    relative = p-state["center"].to(DEVICE)
    row = (.5-relative@up/cfg["extent"])*h-.5
    col = (relative@right/cfg["extent"]+.5)*w-.5
    drow = -(dx@up)*h/cfg["extent"]; dcol = (dx@right)*w/cfg["extent"]
    ids, weights, dy, dcapture = integrated_kernel(row, col, cfg["capture"], cfg["resolution"])
    unique_pixels, pixel_inverse = torch.unique(ids, sorted=True, return_inverse=True)
    pixel_inverse = pixel_inverse.reshape_as(ids)
    channels = torch.arange(3, device=DEVICE)
    rows_parts, columns_parts, values_parts = [], [], []
    scale = cfg["gain"]*h*w
    for tap in range(36):
        movement = dy[point_ids, tap]*drow+dcapture[point_ids, tap]*dcol
        derivative = (weights[point_ids, tap, None]*de+movement[:, None]*energy[point_ids])*scale
        local_rows = pixel_inverse[point_ids, tap, None]*3+channels[None]
        columns = basis_ids[:, None].expand_as(local_rows)
        touched = derivative.abs() > threshold
        rows_parts.append(local_rows[touched]); columns_parts.append(columns[touched]); values_parts.append(derivative[touched])
    local_rows = len(unique_pixels)*3
    matrix = torch.sparse_coo_tensor(torch.stack((torch.cat(rows_parts), torch.cat(columns_parts))),
                                     torch.cat(values_parts), size=(local_rows, parameter_count),
                                     dtype=p.dtype, device=DEVICE).coalesce()
    global_rows = (unique_pixels[:, None]*3+channels[None]).reshape(-1)
    return matrix, global_rows


def _parameter_probe_basis(parameters, oracle, probes=12):
    generator = torch.Generator(device="cpu").manual_seed(977)
    random = torch.randn((parameters, probes), generator=generator, dtype=torch.float64, device="cpu").to(DEVICE)
    v = oracle/oracle.norm()
    random -= v[:, None]*(v@random)[None, :]
    random, _ = torch.linalg.qr(random, mode="reduced")
    return torch.cat((v[:, None], random), 1)


def _output_rademacher(global_rows, view, probes):
    """Stateless output-space sketch; identical detector rows get identical signs."""
    row = global_rows.to(torch.int64)[:, None]
    probe = torch.arange(probes, dtype=torch.int64, device=DEVICE)[None]
    # Integer mixing avoids allocating [all pixels, probes].  Overflow is the
    # intended modular arithmetic; the low mixed bit gives a Rademacher sign.
    key = row*6364136223846793005 + probe*1442695040888963407 + (view+1)*3202034522624059733
    key = key ^ (key >> 30); key = key*2862933555777941757
    key = key ^ (key >> 27)
    return ((key & 1).to(torch.float64)*2-1)/math.sqrt(probes)


def observation_group(field, state, target_state, source_count, resolutions, oracle, *, probes=12,
                      range_probes=32, block=32):
    """Exact J^T r plus seeded randomized J products for several detectors."""
    started = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfgs = {name: configuration(h, w, source_count) for name, h, w in resolutions}
    current_images, target_images, residuals = {}, {}, {}
    for name, cfg in cfgs.items():
        current_images[name] = measure_state(state, cfg)
        target_images[name] = measure_state(target_state, cfg)
        residuals[name] = current_images[name]-target_images[name]
        np.savez_compressed(CACHE/f"images_{name}_{source_count}.npz",
                            initial=current_images[name], target=target_images[name])
    k = field.layout.count; probe_basis = _parameter_probe_basis(k, oracle, probes)
    q = probe_basis.shape[1]
    accum = {name: {"gradient": torch.zeros(k, dtype=torch.float64, device=DEVICE),
                    "subspace_gram": torch.zeros((q, q), dtype=torch.float64, device=DEVICE),
                    "range_sketch": torch.zeros((k, range_probes), dtype=torch.float64, device=DEVICE),
                    "residual_square": float(np.square(residuals[name][list(SELECTED_VIEWS)]).sum()),
                    "sparse_nnz_before_cross_block_coalescing": 0}
             for name in cfgs}
    boundary = ObservationSphere(torch.zeros(3, dtype=torch.float64, device=DEVICE), 1.75)
    maximum_forward_error = 0.
    for view in SELECTED_VIEWS:
        detector_accumulators = {name: torch.zeros((cfg["resolution"][0]*cfg["resolution"][1]*3, q),
                                                    dtype=torch.float32, device=DEVICE)
                                 for name, cfg in cfgs.items()}
        for start in range(0, len(state["weights"]), block):
            core = _tangent_core(field, state, slice(start, start+block), view, boundary,
                                 next(iter(cfgs.values())))
            if core is None:
                continue
            maximum_forward_error = max(maximum_forward_error, core[-1])
            for name, cfg in cfgs.items():
                matrix, global_rows = _sparse_detector_block(core, state, view, cfg, k)
                residual_flat = torch.as_tensor(residuals[name][view].reshape(-1), dtype=torch.float64, device=DEVICE)
                accum[name]["gradient"] += torch.sparse.mm(
                    matrix.transpose(0, 1), residual_flat[global_rows, None]).flatten()
                output_probe = _output_rademacher(global_rows, view, range_probes)
                accum[name]["range_sketch"] += torch.sparse.mm(matrix.transpose(0, 1), output_probe)
                projected = torch.sparse.mm(matrix, probe_basis)
                detector_accumulators[name].index_add_(0, global_rows, projected.float())
                accum[name]["sparse_nnz_before_cross_block_coalescing"] += matrix._nnz()
                del matrix, projected
            if start and start % 4096 == 0:
                print(json.dumps({"phase": "observation_tangent", "sources": source_count,
                                  "view": view, "processed": start}), flush=True)
        for name, cfg in cfgs.items():
            flat = detector_accumulators[name].double()
            accum[name]["subspace_gram"] += flat.T@flat
            del detector_accumulators[name]

    rows = []
    for name, cfg in cfgs.items():
        h, w = cfg["resolution"]; observations = len(SELECTED_VIEWS)*h*w*3
        gradient = accum[name]["gradient"]
        gram = accum[name]["subspace_gram"]
        range_basis, range_r = torch.linalg.qr(accum[name]["range_sketch"], mode="reduced")
        range_diagonal = torch.diagonal(range_r).abs()
        range_keep = range_diagonal > max(float(range_diagonal.max())*1e-8, 1e-12)
        range_projection = float((range_basis[:, range_keep].T@probe_basis[:, 0]).norm()) \
            if bool(range_keep.any()) else 0.
        eigenvalues, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues, vectors = eigenvalues[order].clamp_min(0), vectors[:, order]
        singular = eigenvalues.sqrt(); threshold = max(float(singular[0])*1e-6, 1e-12)
        retained = singular > threshold
        projection = float(vectors[0, retained].square().sum().sqrt()) if bool(retained.any()) else 0.
        jnorm = float(gram[0, 0].clamp_min(0).sqrt())
        g_t = float(gradient@probe_basis[:, 0]); h_t = float(gram[0, 0])
        negative = -gradient; cosine = float((negative@probe_basis[:, 0])/negative.norm().clamp_min(1e-30))
        residual_norm = math.sqrt(accum[name]["residual_square"])
        rows.append({"name": name, "height": h, "width": w, "sources": source_count,
            "views": list(SELECTED_VIEWS), "observations_rgb": observations,
            "mse": accum[name]["residual_square"]/observations,
            "residual_norm": residual_norm, "residual_norm_per_sqrt_observation": residual_norm/math.sqrt(observations),
            "Jv_oracle_norm": jnorm, "Jv_oracle_norm_per_sqrt_observation": jnorm/math.sqrt(observations),
            "gradient_norm": float(gradient.norm()),
            "gradient_norm_per_observation": float(gradient.norm())/observations,
            "negative_gradient_oracle_cosine": cosine,
            "oracle_g": g_t, "oracle_h": h_t, "oracle_alpha_star": -g_t/max(h_t, 1e-30),
            "oracle_g_crosscheck_from_Jv": "implicit in shared streamed detector accumulation",
            "randomized_probe_count": probes, "seeded_subspace_dimension": q,
            "seeded_randomized_projection_estimate": projection,
            "output_sketch_probe_count": range_probes,
            "output_sketch_effective_rank_relative_1e_8": int(range_keep.sum()),
            "oracle_projection_on_randomized_range_JT": range_projection,
            "range_projection_scope": "lower-bound projection on Q=orth(J^T Omega), with stateless seeded output-space Rademacher Omega",
            "seeded_randomized_effective_rank_relative_1e_6": int(retained.sum()),
            "seeded_randomized_singular_values": singular.cpu().tolist(),
            "spectrum_scope": "J restricted to span{v_T, 12 target-independent orthonormal Gaussian probes}; rank is a lower bound",
            "legacy_seeded_projection_interpretation": "projection inside the deliberately v_T-containing parameter probe space; not used as range(J^T) evidence",
            "sparse_nnz_before_cross_block_coalescing": accum[name]["sparse_nnz_before_cross_block_coalescing"],
            "jacobian_storage": "chunk-local coalesced COO; torch.sparse.mm for JQ and J^T r; no dense point-by-parameter or pixel-by-parameter tensor",
            "dense_jacobian_allocated": False})
    return rows, {"runtime_seconds": time.perf_counter()-started,
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                  "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                  "maximum_tangent_forward_error": maximum_forward_error,
                  "source_count": source_count, "parameter_count": k, "emitter_block": block}


def sparse_equivalence():
    """Small exact comparison with the historical dense v0.9.2 tangent."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    from .dense_jet_torus import _differential_block
    sphere, _, _ = load_target(); cfg = configuration(32, 32, 64)
    _, state, _, boundary, _ = render(sphere, cfg, seed=978)
    owners = slice(0, 8); view = 0; k = sphere.layout.count
    dense = _differential_block(sphere, {**state, "normals": state["normals"]}, state,
                                        owners, view, sphere.supports, boundary, cfg)
    if dense is None:
        raise RuntimeError("EMPTY_EQUIVALENCE_BLOCK")
    row, col, energy, de, drow, dcol, dense_forward_error = dense
    ids, weights, dy, dcapture = integrated_kernel(row, col, cfg["capture"], cfg["resolution"])
    h, w = cfg["resolution"]
    accumulator = torch.zeros((h*w, k, 3), dtype=torch.float64, device=DEVICE)
    for tap in range(36):
        contribution = (weights[:, tap, None, None]*de
                        +(dy[:, tap, None]*drow+dcapture[:, tap, None]*dcol)[:, :, None]*energy[:, None, :])
        accumulator.index_add_(0, ids[:, tap], contribution)
    dense_matrix = (accumulator*(cfg["gain"]*h*w)).permute(0, 2, 1).reshape(-1, k)
    core = _tangent_core(sphere, state, owners, view, boundary, cfg)
    sparse_local, global_rows = _sparse_detector_block(core, state, view, cfg, k, threshold=0.)
    local = sparse_local.coalesce(); global_indices = global_rows[local.indices()[0]]
    sparse_global = torch.sparse_coo_tensor(torch.stack((global_indices, local.indices()[1])), local.values(),
                                            size=dense_matrix.shape, dtype=torch.float64, device=DEVICE).coalesce()
    difference = sparse_global.to_dense()-dense_matrix
    maximum = float(difference.abs().max()); relative = float(difference.norm()/dense_matrix.norm().clamp_min(1e-30))
    result = {"version": "v0.9.7", "dense_shape": list(dense_matrix.shape),
              "dense_entries": dense_matrix.numel(), "sparse_nnz": sparse_global._nnz(),
              "maximum_absolute_error": maximum, "relative_l2_error": relative,
              "dense_forward_error": dense_forward_error, "sparse_forward_error": core[-1],
              "equivalent": maximum <= 1e-9 and relative <= 1e-10,
              "production_storage": "chunk-local COO only"}
    write_json(ROOT / "artifacts/v097_sparse_equivalence.json", result)
    print(json.dumps(result, indent=2))
    if not result["equivalent"]:
        raise RuntimeError("SPARSE_CURRENT_C3_TANGENT_NOT_EQUIVALENT")
    return result


def sparse_gradient(field, state, image, target_image, cfg, *, block=32):
    """Exact streamed J^T r and a block-local COO Jacobi preconditioner."""
    started = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    k = field.layout.count; gradient = torch.zeros(k, dtype=torch.float64, device=DEVICE)
    diagonal = torch.zeros_like(gradient); residual = image-target_image
    boundary = ObservationSphere(torch.zeros(3, dtype=torch.float64, device=DEVICE), 1.75)
    nnz = 0; maximum_forward_error = 0.
    for view in SELECTED_VIEWS:
        residual_flat = torch.as_tensor(residual[view].reshape(-1), dtype=torch.float64, device=DEVICE)
        for start in range(0, len(state["weights"]), block):
            core = _tangent_core(field, state, slice(start, start+block), view, boundary, cfg)
            if core is None:
                continue
            maximum_forward_error = max(maximum_forward_error, core[-1])
            matrix, global_rows = _sparse_detector_block(core, state, view, cfg, k)
            matrix = matrix.coalesce(); indices, values = matrix.indices(), matrix.values()
            gradient += torch.sparse.mm(matrix.transpose(0, 1), residual_flat[global_rows, None]).flatten()
            diagonal.scatter_add_(0, indices[1], values.square())
            nnz += matrix._nnz()
            if start and start % 4096 == 0:
                print(json.dumps({"phase": "optimization_gradient", "view": view, "processed": start,
                                  "sources": len(state["weights"])}), flush=True)
    return gradient, diagonal, {"runtime_seconds": time.perf_counter()-started,
        "sparse_nnz_before_cross_block_coalescing": nnz,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
        "maximum_tangent_forward_error": maximum_forward_error,
        "preconditioner": "sum of squared entries after within-emitter-block COO coalescing; cross-block pixel coupling omitted",
        "gradient": "exact J^T r over all sources and selected views"}


def _image_only_proposals(gradient, diagonal):
    """Target-agnostic trust directions; no oracle or target field argument."""
    nonzero = diagonal[diagonal > 0]
    scale = float(torch.quantile(nonzero, .5)) if len(nonzero) else 1.
    proposals = []
    for damping_fraction in (.02, .1, .5):
        direction = -gradient/(diagonal+damping_fraction*max(scale, 1e-30))
        if float(direction.norm()) > .65:
            direction *= .65/direction.norm()
        if float(direction.abs().max()) > .15:
            direction *= .15/direction.abs().max()
        proposals.append((f"SPARSE_JACOBI_{damping_fraction:g}", direction))
    steepest = -gradient/gradient.norm().clamp_min(1e-30)*.5
    proposals.append(("STEEPEST", steepest))
    return proposals


def _optimization_geometry(field, surface):
    from .dense_jet_torus import field_diagnostics
    from .topology_penetration import fast_fiber_counts
    result = field_diagnostics(field, surface)
    result["fibers"] = fast_fiber_counts(field)
    return result


def optimize_image_only(initial_field, target_image, cfg, *, label, maximum_steps=100):
    """Fresh real RGB optimization; only an initial field and target RGB enter."""
    field = initial_field.with_coefficients(initial_field.coefficients.clone())
    trajectory = []; started = time.perf_counter(); stop = "ACCEPTED_STEP_BUDGET"
    with torch.no_grad(): image, state, surface, _, timing = render(field, cfg, seed=970)
    np.save(CACHE/f"{label}_initial.npy", image)
    for iteration in range(maximum_steps+1):
        residual = image[list(SELECTED_VIEWS)]-target_image[list(SELECTED_VIEWS)]
        row = {"iteration": iteration, "mse": float(np.mean(residual**2)),
               "all_view_mse": float(np.mean((image-target_image)**2)),
               "coefficient_l2": float(field.coefficients.norm()),
               "coefficient_abs_max": float(field.coefficients.abs().max()),
               "topology": surface.topology, "geometry": _optimization_geometry(field, surface),
               "render_seconds": timing["seconds"]}
        trajectory.append(row)
        write_json(CACHE/f"{label}_progress.json", {"trajectory": trajectory, "stop": stop})
        if surface.topology["watertight"] and surface.topology["components"] == 1 and surface.topology["inferred_genus"] == 1:
            stop = "TOPOLOGY_SUCCESS"; break
        if iteration == maximum_steps:
            break
        gradient, diagonal, linearization = sparse_gradient(field, state, image, target_image, cfg)
        row["linearization"] = linearization; row["gradient_norm"] = float(gradient.norm())
        proposals = _image_only_proposals(gradient, diagonal); row["trials"] = []; accepted = None
        # Four target-independent directions and eight trust scales constitute
        # a substantial no-progress audit, not the v0.9.6 24-trial stop.
        for method, direction in proposals:
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
                candidate = field.with_coefficients(field.coefficients+alpha*direction)
                trial = {"method": method, "alpha": alpha,
                         "update_l2": float((alpha*direction).norm()), "accepted": False}
                try:
                    with torch.no_grad(): ti, ts, tf, _, tt = render(candidate, cfg, seed=970)
                    value = float(np.mean((ti[list(SELECTED_VIEWS)]-target_image[list(SELECTED_VIEWS)])**2))
                    trial["actual_fresh_retrace_mse"] = value
                    trial["topology"] = tf.topology
                    trial["accepted"] = value < row["mse"]*(1-1e-8)
                except RuntimeError as error:
                    trial["failure"] = str(error)
                row["trials"].append(trial)
                if trial["accepted"]:
                    accepted = candidate, ti, ts, tf, tt
                    break
            if accepted is not None:
                break
        if accepted is None:
            row["accepted"] = False; stop = "STABLE_NO_PROGRESS_AFTER_32_FRESH_RETRACES"; break
        row["accepted"] = True; row["accepted_method"] = row["trials"][-1]["method"]
        field, image, state, surface, timing = accepted
        torch.save({"coefficients": field.coefficients.cpu()}, CACHE/f"{label}_step{iteration+1}.pt")
        np.save(CACHE/f"{label}_step{iteration+1}.npy", image)
        print(json.dumps({"phase": label, "accepted": iteration+1,
                          "mse": row["trials"][-1]["actual_fresh_retrace_mse"],
                          "genus": surface.topology["inferred_genus"]}), flush=True)
    from .topology_penetration import fiber_roots
    np.save(CACHE/f"{label}_final.npy", image)
    output = {"label": label, "configuration": {"resolution": cfg["resolution"], "sources": cfg["count"],
              "views": list(SELECTED_VIEWS), "operator": "CURRENT/C3"},
        "maximum_accepted_steps": maximum_steps, "optimizer_inputs": ["initial coefficients", "immutable target RGB"],
        "optimizer_forbidden_inputs": {"theta_T": False, "v_T": False, "target_field": False,
             "target_mesh": False, "genus": False, "hole_location": False},
        "trajectory": trajectory, "accepted_steps": sum(bool(x.get("accepted")) for x in trajectory),
        "stop": stop, "initial_mse": trajectory[0]["mse"], "final_mse": trajectory[-1]["mse"],
        "final_coefficients": field.coefficients.cpu().tolist(), "final_fibers": fiber_roots(field),
        "final_topology": {str(n): topology_only(field, n) for n in (72, 96, 144)},
        "topology_penetrated": surface.topology["inferred_genus"] == 1,
        "runtime_seconds": time.perf_counter()-started}
    write_json(CACHE/f"{label}.json", output)
    return output


def finalize_accepted_budget(initial_field, target_image, cfg, *, label, maximum_steps):
    """Finalize a fully checkpointed accepted-step budget without recomputation."""
    progress = json.loads((CACHE/f"{label}_progress.json").read_text())
    trajectory = progress["trajectory"]
    if not trajectory or int(trajectory[-1]["iteration"]) != maximum_steps:
        raise RuntimeError(f"INCOMPLETE_CHECKPOINT_BUDGET:{label}")
    saved = torch.load(CACHE/f"{label}_step{maximum_steps}.pt", weights_only=True)
    field = initial_field.with_coefficients(saved["coefficients"].to(DEVICE))
    image = np.load(CACHE/f"{label}_step{maximum_steps}.npy", mmap_mode="r")
    from .topology_penetration import fiber_roots
    output = {"label": label, "configuration": {"resolution": cfg["resolution"], "sources": cfg["count"],
              "views": list(SELECTED_VIEWS), "operator": "CURRENT/C3"},
        "maximum_accepted_steps": maximum_steps, "optimizer_inputs": ["initial coefficients", "immutable target RGB"],
        "optimizer_forbidden_inputs": {"theta_T": False, "v_T": False, "target_field": False,
             "target_mesh": False, "genus": False, "hole_location": False},
        "trajectory": trajectory, "accepted_steps": maximum_steps, "stop": "ACCEPTED_STEP_BUDGET",
        "initial_mse": trajectory[0]["mse"], "final_mse": trajectory[-1]["mse"],
        "final_coefficients": field.coefficients.cpu().tolist(), "final_fibers": fiber_roots(field),
        "final_topology": {str(n): topology_only(field, n) for n in (72, 96, 144)},
        "topology_penetrated": all(x["inferred_genus"] == 1 for x in
                                    (topology_only(field, 72), topology_only(field, 96), topology_only(field, 144))),
        "runtime_seconds": float(sum(row.get("render_seconds", 0.)+
                                     (row.get("linearization") or {}).get("runtime_seconds", 0.)
                                     for row in trajectory)),
        "runtime_note": "lower bound from persisted render and sparse-linearization timings; line-search rejects excluded"}
    write_json(CACHE/f"{label}.json", output)
    return output


def optimize_primary():
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    sphere, _, _ = load_target(); cfg = configuration(1024, 1024, 16384)
    target_state = torch.load(_state_path("target", 16384), weights_only=False)["state"]
    target_image = measure_state(target_state, cfg)
    result = optimize_image_only(sphere, target_image, cfg,
                                 label="primary_1024x1024_16K", maximum_steps=100)
    write_json(CACHE/"optimization.json", result)
    print(json.dumps({key: result[key] for key in ("accepted_steps", "stop", "initial_mse", "final_mse", "topology_penetrated", "runtime_seconds")}, indent=2))
    return result


def oracle_path():
    """Fresh CURRENT/C3 tomography of the immutable real oracle segment."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    sphere, target, target_info = load_target(); cfg = configuration(1024, 1024, 16384)
    target_state = torch.load(_state_path("target", 16384), weights_only=False)["state"]
    target_image = measure_state(target_state, cfg)
    # Dense near the empirically bracketed real topology event (0.725, 0.75).
    samples = (0., .25, .5, .6, .65, .7, .725, .74, .75, .8, 1.)
    progress_path = CACHE/"oracle_path_progress.json"
    if progress_path.exists():
        rows = json.loads(progress_path.read_text()).get("samples", [])
    else:
        rows = []
    completed = {float(row["t"]) for row in rows}
    started = time.perf_counter()
    for t in samples:
        if t in completed:
            continue
        field = sphere.with_coefficients(t*target.coefficients)
        with torch.no_grad(): image, state, surface, _, timing = render(field, cfg, seed=970)
        residual = image[list(SELECTED_VIEWS)]-target_image[list(SELECTED_VIEWS)]
        geometry = _optimization_geometry(field, surface)
        row = {"t": t, "mse": float(np.mean(residual**2)),
               "all_view_mse": float(np.mean((image-target_image)**2)),
               "coefficient_l2": float(field.coefficients.norm()),
               "render_seconds": timing["seconds"], "topology_grid40": surface.topology,
               "topology_grid72": topology_only(field, 72), "geometry": geometry}
        remaining = target.coefficients-field.coefficients
        if float(remaining.norm()) > 1e-14:
            gradient, _, linearization = sparse_gradient(field, state, image, target_image, cfg)
            direction = remaining/remaining.norm()
            cosine = float((-gradient@direction)/gradient.norm().clamp_min(1e-30))
            row.update(gradient_norm=float(gradient.norm()),
                       negative_gradient_remaining_direction_cosine=cosine,
                       directional_loss_derivative=float(gradient@direction),
                       linearization=linearization)
            if t == 0.:
                torch.save({"v_G": (-gradient/gradient.norm().clamp_min(1e-30)).cpu(),
                            "gradient": gradient.cpu(), "definition": "v_G=-J^T r/||J^T r|| at sphere"},
                           CACHE/"sphere_vG.pt")
        else:
            row.update(gradient_norm=0., negative_gradient_remaining_direction_cosine=None,
                       directional_loss_derivative=0., linearization=None)
        np.save(CACHE/f"oracle_path_t{t:.3f}.npy", image)
        rows.append(row); rows.sort(key=lambda x: x["t"])
        write_json(progress_path, {"version": "v0.9.7", "configuration": cfg,
                   "target": target_info, "samples": rows})
        print(json.dumps({"phase": "oracle_path", "t": t, "mse": row["mse"],
                          "genus": row["topology_grid72"]["inferred_genus"],
                          "alignment": row["negative_gradient_remaining_direction_cosine"]}), flush=True)
    result = {"version": "v0.9.7", "phase": "C_ORACLE_PATH_TOMOGRAPHY",
              "configuration": {"resolution": cfg["resolution"], "sources": cfg["count"],
                                "views": list(SELECTED_VIEWS), "operator": "CURRENT/C3"},
              "target": target_info, "path_definition": "theta(t)=t theta_T; every state freshly rendered",
              "samples": rows, "runtime_seconds_this_invocation": time.perf_counter()-started}
    write_json(CACHE/"oracle_path.json", result)
    return result


def basin_runs():
    """Image-only recovery from three declared oracle-path initializations."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    sphere, target, _ = load_target(); cfg = configuration(1024, 1024, 16384)
    target_state = torch.load(_state_path("target", 16384), weights_only=False)["state"]
    target_image = measure_state(target_state, cfg)
    output = {"version": "v0.9.7", "phase": "BASIN_OF_ATTRACTION",
              "initialization_only_uses_theta_T": True, "optimizer_uses_theta_T_after_start": False,
              "runs": []}
    # Start with the near-target control so an interrupted long run still
    # distinguishes an already topological basin from the two genus-0 basins.
    for fraction in (.75, .5, .25):
        label = f"basin_{fraction:.2f}"
        existing = CACHE/f"{label}.json"
        if existing.exists():
            result = json.loads(existing.read_text())
        else:
            initial = sphere.with_coefficients(fraction*target.coefficients)
            progress = CACHE/f"{label}_progress.json"
            checkpoint = CACHE/f"{label}_step10.pt"
            if progress.exists() and checkpoint.exists() and \
                    json.loads(progress.read_text())["trajectory"][-1]["iteration"] == 10:
                result = finalize_accepted_budget(initial, target_image, cfg, label=label, maximum_steps=10)
            else:
                result = optimize_image_only(initial, target_image, cfg, label=label, maximum_steps=10)
        output["runs"].append({"initial_fraction": fraction, **result})
        write_json(CACHE/"basins.json", output)
    return output


def loss_slice():
    """Fresh 2-D loss slice in normalized oracle and negative-gradient axes."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    sphere, target, _ = load_target(); cfg = configuration(1024, 1024, 16384)
    target_state = torch.load(_state_path("target", 16384), weights_only=False)["state"]
    target_image = measure_state(target_state, cfg)
    gradient_path = CACHE/"sphere_vG.pt"
    if not gradient_path.exists():
        raise RuntimeError("RUN_ORACLE_PATH_FIRST_TO_DEFINE_IMAGE_GRADIENT_AXIS")
    v_g = torch.load(gradient_path, weights_only=True)["v_G"].to(DEVICE)
    v_t = target.coefficients/target.coefficients.norm()
    oracle_norm = float(target.coefficients.norm())
    alpha_fractions = (.65, .7, .725, .75, .8)
    beta_values = (-.5, -.25, 0., .25, .5)
    progress_path = CACHE/"loss_slice_progress.json"
    rows = json.loads(progress_path.read_text()).get("samples", []) if progress_path.exists() else []
    completed = {(float(x["alpha_fraction"]), float(x["beta"])) for x in rows}
    for alpha_fraction in alpha_fractions:
        for beta in beta_values:
            if (alpha_fraction, beta) in completed:
                continue
            coefficients = alpha_fraction*oracle_norm*v_t+beta*v_g
            field = sphere.with_coefficients(coefficients)
            row = {"alpha_fraction": alpha_fraction, "alpha": alpha_fraction*oracle_norm,
                   "beta": beta, "coefficient_l2": float(coefficients.norm())}
            try:
                with torch.no_grad(): image, _, surface, _, timing = render(field, cfg, seed=970)
                row.update(mse=float(np.mean((image[list(SELECTED_VIEWS)]-
                                             target_image[list(SELECTED_VIEWS)])**2)),
                           all_view_mse=float(np.mean((image-target_image)**2)),
                           topology_grid40=surface.topology, topology_grid72=topology_only(field, 72),
                           render_seconds=timing["seconds"], finite=bool(np.isfinite(image).all()))
            except RuntimeError as error:
                row.update(failure=str(error), finite=False)
            rows.append(row)
            write_json(progress_path, {"version": "v0.9.7", "samples": rows})
            print(json.dumps({"phase": "loss_slice", **row}), flush=True)
    result = {"version": "v0.9.7", "phase": "TWO_DIMENSIONAL_LOSS_SLICE",
              "configuration": {"resolution": cfg["resolution"], "sources": cfg["count"],
                                "views": list(SELECTED_VIEWS), "operator": "CURRENT/C3"},
              "axes": {"v_T": "theta_T/||theta_T||", "v_G": "-J^T r/||J^T r|| at theta=0",
                       "alpha_fractions": list(alpha_fractions), "beta": list(beta_values)},
              "every_sample_freshly_rendered": True, "samples": rows}
    write_json(CACHE/"loss_slice.json", result)
    return result


def audit_fibers_2048():
    """Fixed-direction high-resolution fiber audit for selected saved states."""
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    from .topology_penetration import fiber_roots
    sphere, target, _ = load_target()
    optimization = json.loads((CACHE/"optimization.json").read_text())
    selected_iterations = (0, 5, 10, 15, 20)
    primary = []
    for iteration in selected_iterations:
        if iteration == 0:
            coefficients = torch.zeros_like(target.coefficients)
        else:
            coefficients = torch.load(CACHE/f"primary_1024x1024_16K_step{iteration}.pt",
                                      weights_only=True)["coefficients"].to(DEVICE)
        field = sphere.with_coefficients(coefficients)
        fibers = fiber_roots(field)
        trajectory_row = next(row for row in optimization["trajectory"] if row["iteration"] == iteration)
        primary.append({"iteration": iteration, "mse": trajectory_row["mse"], "fibers": fibers})
        print(json.dumps({"phase": "fiber_audit_primary", "iteration": iteration,
                          "histogram": fibers["root_count_histogram"]}), flush=True)
    oracle = []
    for row in json.loads((CACHE/"oracle_path.json").read_text())["samples"]:
        t = float(row["t"]); field = sphere.with_coefficients(t*target.coefficients)
        fibers = fiber_roots(field)
        oracle.append({"t": t, "mse": row["mse"],
                       "genus_grid72": row["topology_grid72"]["inferred_genus"], "fibers": fibers})
        print(json.dumps({"phase": "fiber_audit_oracle", "t": t,
                          "histogram": fibers["root_count_histogram"]}), flush=True)
    result = {"version": "v0.9.7", "directions": 2048, "samples_per_fiber": 513,
              "direction_construction": "fixed deterministic v0.9.6 reference directions",
              "primary_selected_iterations": primary, "oracle_path": oracle}
    write_json(CACHE/"fiber_audit_2048.json", result)
    return result


def observe():
    require_cuda(); CACHE.mkdir(parents=True, exist_ok=True)
    sphere, target, target_info = load_target()
    topology = {str(n): topology_only(target, n) for n in (96, 144)}
    if not all(x["watertight"] and x["components"] == 1 and x["inferred_genus"] == 1
               and not x["extraction_boundary_contact"] for x in topology.values()):
        raise RuntimeError("V096_TARGET_TOPOLOGY_REPRODUCTION_FAILED")
    groups = {16384: [("512x512_16K", 512, 512), ("1024x1024_16K", 1024, 1024)],
              65536: [("512x512_64K", 512, 512), ("1024x1024_64K", 1024, 1024),
                      ("1920x1080_64K", 1080, 1920)]}
    progress_path = CACHE / "observation_progress.json"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        all_rows = previous.get("configurations", [])
        group_reports = previous.get("groups", [])
        scene_reports = previous.get("scenes", [])
    else:
        all_rows, group_reports, scene_reports = [], [], []
    oracle = target.coefficients.detach()
    for source_count, resolutions in groups.items():
        expected_names = {item[0] for item in resolutions}
        completed_rows = [row for row in all_rows if row["name"] in expected_names]
        if expected_names.issubset({row["name"] for row in completed_rows}) and all(
                "oracle_projection_on_randomized_range_JT" in row for row in completed_rows):
            print(json.dumps({"skipped_completed_sources": source_count}), flush=True)
            continue
        # Replace the obsolete parameter-probe-only range estimate atomically
        # after the corrected output-space sketch finishes.
        all_rows = [row for row in all_rows if row["name"] not in expected_names]
        group_reports = [row for row in group_reports if row.get("source_count") != source_count]
        scene_reports = [row for row in scene_reports if row.get("sources") != source_count]
        sphere_state, sphere_topology, sphere_seconds, sphere_cached = build_scene(sphere, "sphere", source_count)
        target_state, target_topology, target_seconds, target_cached = build_scene(target, "target", source_count)
        if len(sphere_state["weights"]) != source_count or len(target_state["weights"]) != source_count:
            raise RuntimeError("SOURCE_COUNT_NOT_REALIZED")
        scene_reports.append({"sources": source_count, "sphere_seconds": sphere_seconds, "target_seconds": target_seconds,
                              "sphere_cached": sphere_cached, "target_cached": target_cached,
                              "sphere_topology": sphere_topology, "target_topology": target_topology,
                              "transport_frozen_across_detector_resolutions": True})
        rows, timing = observation_group(sphere, sphere_state, target_state, source_count, resolutions, oracle)
        all_rows.extend(rows); group_reports.append(timing)
        progress = {"version": "v0.9.7", "target": {**target_info, "topology": topology},
                    "configurations": all_rows, "groups": group_reports, "scenes": scene_reports}
        write_json(progress_path, progress)
        print(json.dumps({"completed_sources": source_count, "rows": rows, "timing": timing}, indent=2), flush=True)
    cosines = np.asarray([x["negative_gradient_oracle_cosine"] for x in all_rows])
    result = {"version": "v0.9.7", "phase": "A_HIGH_BANDWIDTH_OBSERVATION",
        "cuda": require_cuda(), "target": {**target_info, "topology": topology},
        "configurations": all_rows, "groups": group_reports, "scenes": scene_reports,
        "matrix": {"detectors": ["512x512", "1024x1024", "1920x1080"],
                   "sources": [16384, 65536], "fullhd_attempted": True,
                   "sources_262144": "not selected: exact 768-parameter VJP at 64K is the highest practical complete diagnostic on a 16-GiB RTX 5000"},
        "measurement_bandwidth_hypothesis_supported_phase_A": bool(np.max(np.abs(cosines)) > .1),
        "bandwidth_verdict": "PENDING_NUMERIC_INTERPRETATION"}
    write_json(CACHE / "observation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", action="store_true")
    parser.add_argument("--equivalence", action="store_true")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--oracle-path", action="store_true")
    parser.add_argument("--basins", action="store_true")
    parser.add_argument("--loss-slice", action="store_true")
    parser.add_argument("--fiber-audit", action="store_true")
    args = parser.parse_args()
    if args.environment:
        environment_report()
    elif args.equivalence:
        sparse_equivalence()
    elif args.observe:
        observe()
    elif args.optimize:
        optimize_primary()
    elif args.oracle_path:
        oracle_path()
    elif args.basins:
        basin_runs()
    elif args.loss_slice:
        loss_slice()
    elif args.fiber_audit:
        audit_fibers_2048()
    else:
        parser.error("select an experiment phase")
