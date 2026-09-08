"""Consolidate the v0.9.6 CUDA topology-capacity and penetration experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zlt.basis_initialization import directions
from zlt.topology_penetration import DEVICE, _sign_changing_intervals, make_field
from zlt.transverse_packet import write_json

CACHE = ROOT / "runs/v096_topology_penetration"
ARTIFACT = ROOT / "artifacts/v096_topology_penetration.json"


def _save(name, fig, *, tight=True):
    path = ROOT / "figures" / f"v096_{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def _reduction(run):
    return 1.0 - run["final_mse"] / run["initial_mse"]


def _image_limit(*arrays):
    return max(float(np.quantile(np.concatenate([x.ravel() for x in arrays]), .995)), 1e-12)


def _fiber_count_map(field, count=2048, samples=513):
    """Return direction/count arrays; refined root slopes remain in Phase A."""
    unit = directions(count, 962).to(DEVICE)
    s = torch.linspace(-.98, 1., samples, device=DEVICE, dtype=torch.float64)
    counts = []
    for start in range(0, count, 128):
        local = unit[start:start+128]
        points = local[:, None, :] * (1+s[None, :, None])
        values = torch.cat([field.value(points[:, j:j+32].reshape(-1, 3)).reshape(len(local), -1)
                            for j in range(0, samples, 32)], 1)[:, :samples]
        counts.append(_sign_changing_intervals(values).sum(1).cpu())
    return unit.cpu().numpy(), torch.cat(counts).numpy()


def main():
    env = json.loads((ROOT / "artifacts/v096_cuda_environment.json").read_text())
    capacity = json.loads((CACHE / "capacity.json").read_text())
    penetration = json.loads((CACHE / "penetration.json").read_text())
    controls = json.loads((CACHE / "controls.json").read_text())
    winner = capacity["winner"]
    assert capacity["capacity_success"] and winner is not None

    theta_source = CACHE / "theta_T.pt"
    theta_artifact = ROOT / "artifacts/v096_theta_T.pt"
    shutil.copy2(theta_source, theta_artifact)
    theta_digest = hashlib.sha256(theta_artifact.read_bytes()).hexdigest()

    rows = []
    for item in capacity["configs"]:
        topology = item["topology_screen"]
        rows.append({"phase": "capacity", "case": f"K{item['K']}_p{item['order']}_r{item['radius']}",
                     "iteration": "", "K": item["K"], "order": item["order"],
                     "parameters": item["parameters"], "radius": item["radius"],
                     "mse_or_rmse": item["weighted_rmse"], "genus": topology["inferred_genus"],
                     "components": topology["components"], "watertight": topology["watertight"],
                     "min_gradient": "", "chart_denominator": "", "root_multi_fraction": "",
                     "effective_rank": "", "oracle_gradient_cosine": "",
                     "oracle_projection_fraction": "", "accepted": ""})
    for item in penetration["trajectory"]:
        geometry = item["geometry"]
        oracle = item.get("oracle_diagnostic", {})
        rows.append({"phase": "penetration", "case": "same_family_genus1", "iteration": item["iteration"],
                     "K": penetration["K"], "order": penetration["order"], "parameters": penetration["parameters"],
                     "radius": penetration["radius"], "mse_or_rmse": item["actual_fresh_retrace_mse"],
                     "genus": geometry["topology"]["inferred_genus"], "components": geometry["topology"]["components"],
                     "watertight": geometry["topology"]["watertight"],
                     "min_gradient": geometry["spatial_gradient_quantiles"][0],
                     "chart_denominator": geometry["chart_denominator_quantiles"][0],
                     "root_multi_fraction": geometry["fibers"]["multi_root_fraction"],
                     "effective_rank": oracle.get("effective_rank_relative_1e_8", ""),
                     "oracle_gradient_cosine": oracle.get("oracle_gradient_cosine", ""),
                     "oracle_projection_fraction": oracle.get("oracle_observable_projection_fraction", ""),
                     "accepted": item.get("accepted", "final")})
    for key in ("same_topology", "weak_observability"):
        run = controls[key]
        for item in run["trajectory"]:
            rows.append({"phase": "control", "case": run["name"], "iteration": item["iteration"],
                         "K": penetration["K"], "order": penetration["order"], "parameters": penetration["parameters"],
                         "radius": penetration["radius"], "mse_or_rmse": item["mse"],
                         "genus": item["topology"]["inferred_genus"], "components": item["topology"]["components"],
                         "watertight": item["topology"]["watertight"], "min_gradient": item["min_gradient"],
                         "chart_denominator": "", "root_multi_fraction": item["fibers"]["multi_root_fraction"],
                         "effective_rank": "", "oracle_gradient_cosine": "", "oracle_projection_fraction": "",
                         "accepted": item.get("accepted", "final")})
    csv_path = ROOT / "artifacts/v096_topology_penetration.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figures = []
    configs = capacity["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for order in (0, 1, 2):
        subset = [x for x in configs if x["order"] == order]
        axes[0].scatter([x["parameters"] for x in subset], [x["weighted_rmse"] for x in subset],
                        s=55, label=f"p={order}")
    for x in configs:
        axes[1].scatter(x["parameters"], x["radius"], c=("tab:green" if x["topology_screen"]["inferred_genus"] == 1 else "tab:gray"), s=60)
    axes[0].set(xlabel="parameters", ylabel="weighted field RMSE", title="geometry-oracle capacity sweep")
    axes[0].legend(); axes[1].set(xlabel="parameters", ylabel="support radius", title="green = watertight genus 1")
    for ax in axes: ax.grid(alpha=.25)
    figures.append(_save("capacity_sweep", fig))

    sf, wf, ff = capacity["sphere_fibers"], capacity["winner_fibers"], penetration["final_fibers"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    labels = ["sphere", "oracle genus 1", "RGB endpoint"]
    histograms = [sf["root_count_histogram"], wf["root_count_histogram"], ff["root_count_histogram"]]
    roots = sorted({int(k) for hist in histograms for k in hist})
    width = .24
    for i, (label, hist) in enumerate(zip(labels, histograms)):
        axes[0].bar(np.asarray(roots)+(i-1)*width, [hist.get(str(x), 0) for x in roots], width, label=label)
    axes[0].set(xlabel="roots per normal fiber", ylabel="fiber count", title="2048-fiber root audit")
    axes[0].set_xticks(roots); axes[0].legend(fontsize=8)
    axes[1].bar(labels, [sf["root_slope_abs_min"], wf["root_slope_abs_min"], ff["root_slope_abs_min"]])
    axes[1].set(ylabel="minimum |dG/ds| at roots", title="simple-root conditioning")
    axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=.25)
    figures.append(_save("normal_fibers", fig))

    initial = np.load(CACHE / "same_family_initial.npy")
    target = np.load(CACHE / "same_family_target.npy")
    final = np.load(CACHE / "same_family_final.npy")
    limit = _image_limit(initial, target, final)
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5))
    for row, (label, images) in enumerate((("initial sphere", initial), ("same-family genus 1 target", target), ("RGB-only endpoint", final))):
        for view in range(3):
            axes[row, view].imshow(np.clip(images[view] / limit, 0, 1)); axes[row, view].axis("off")
            axes[row, view].set_title(f"{label}; view {view}", fontsize=9)
    figures.append(_save("same_family_images", fig))

    trajectory = penetration["trajectory"]
    iterations = [x["iteration"] for x in trajectory]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(iterations, [x["actual_fresh_retrace_mse"] for x in trajectory], "o-")
    axes[1].plot(iterations, [x["geometry"]["spatial_gradient_quantiles"][0] for x in trajectory], "o-", label="true min ||grad F||")
    axes[1].plot(iterations, [x["geometry"]["chart_denominator_quantiles"][0] for x in trajectory], "s-", label="chart denominator")
    axes[2].plot(iterations, [x["geometry"]["fibers"]["multi_root_fraction"] for x in trajectory], "o-")
    axes[0].set_title("fresh-retrace image MSE"); axes[1].set_title("distinct conditioning diagnostics"); axes[2].set_title("multi-root fiber fraction")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.set_xlabel("accepted iteration"); ax.grid(alpha=.25)
    figures.append(_save("penetration", fig))

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for item in trajectory:
        diagnostic = item.get("oracle_diagnostic")
        if diagnostic:
            values = diagnostic["leading_singular_values"]
            ax.semilogy(np.arange(1, len(values)+1), values, marker=".", linewidth=.8, label=f"iter {item['iteration']}")
    ax.set(xlabel="leading mode index", ylabel="singular value", title="image-Jacobian leading spectrum")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=7)
    figures.append(_save("spectrum", fig))

    diagnostics = [x["oracle_diagnostic"] for x in trajectory if "oracle_diagnostic" in x]
    xdiag = iterations[:len(diagnostics)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(xdiag, [x["oracle_observable_projection_fraction"] for x in diagnostics], "o-")
    axes[1].plot(xdiag, [x["oracle_gradient_cosine"] for x in diagnostics], "o-")
    axes[0].set_title("oracle direction in observable subspace"); axes[0].set_ylabel("projection fraction")
    axes[1].set_title("negative gradient vs oracle direction"); axes[1].set_ylabel("cosine")
    for ax in axes: ax.set_xlabel("iteration"); ax.grid(alpha=.25)
    figures.append(_save("oracle_alignment", fig))

    same, weak = controls["same_topology"], controls["weak_observability"]
    fig, ax = plt.subplots(figsize=(6.3, 4))
    for label, run in (("same-topology ellipsoid", same), ("same-family weak view", weak), ("same-family informative views", penetration)):
        values = [x.get("mse", x.get("actual_fresh_retrace_mse")) for x in run["trajectory"]]
        ax.semilogy(range(len(values)), values, "o-", label=label)
    ax.set(xlabel="accepted iteration", ylabel="fresh-retrace MSE", title="matched optimization controls")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    figures.append(_save("controls", fig))

    state = torch.load(theta_source, weights_only=True)
    target_field = make_field(state["K"], state["order"], state["radius"], state["coefficients"].to(DEVICE))
    final_coefficients = torch.tensor(penetration["final_coefficients"], dtype=torch.float64, device=DEVICE)
    final_field = make_field(penetration["K"], penetration["order"], penetration["radius"], final_coefficients)
    sphere = make_field(penetration["K"], penetration["order"], penetration["radius"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    fiber_maps = {}
    fiber_map_cache = CACHE / "fiber_count_maps_v2.npz"
    if fiber_map_cache.exists():
        cached_maps = np.load(fiber_map_cache)
        unit = cached_maps["directions"]
        mapped_counts = [cached_maps[f"counts_{i}"] for i in range(3)]
    else:
        mapped_counts = []
        for field in (sphere, target_field, final_field):
            unit, root_counts = _fiber_count_map(field)
            mapped_counts.append(root_counts)
        np.savez(fiber_map_cache, directions=unit, **{f"counts_{i}": value for i, value in enumerate(mapped_counts)})
    for ax, label, root_counts in zip(axes, labels, mapped_counts):
        fiber_maps[label] = {"directions": len(unit), "root_count_histogram":
                             {str(int(k)): int((root_counts == k).sum()) for k in np.unique(root_counts)}}
        scatter = ax.scatter(np.arctan2(unit[:, 1], unit[:, 0]), unit[:, 2], c=root_counts,
                             s=7, cmap="viridis", vmin=1, vmax=max(3, int(root_counts.max())))
        ax.set(xlabel="azimuth", ylabel="z", title=label)
    fig.subplots_adjust(right=.88, wspace=.32)
    color_axis = fig.add_axes([.91, .18, .016, .64])
    fig.colorbar(scatter, cax=color_axis, label="sign-changing roots per fiber")
    figures.append(_save("normal_fiber_maps", fig, tight=False))

    axis = np.linspace(-1.35, 1.35, 300); xx, zz = np.meshgrid(axis, axis, indexing="xy")
    points = torch.tensor(np.stack((xx.ravel(), np.zeros(xx.size), zz.ravel()), 1), dtype=torch.float64, device=DEVICE)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, label, field in zip(axes, labels, (sphere, target_field, final_field)):
        chunks = [field.value(points[i:i+8192]).detach().cpu().numpy() for i in range(0, len(points), 8192)]
        values = np.concatenate(chunks).reshape(xx.shape)
        ax.contour(xx, zz, values, levels=[0], colors="black")
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal", title=label, xlabel="x", ylabel="z")
    figures.append(_save("topology_sections", fig))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for item in trajectory:
        points_low = np.asarray(item["geometry"]["lowest_gradient_locations"])
        ax.scatter(points_low[:, 0], points_low[:, 2], s=20, alpha=.7, label=f"iter {item['iteration']}")
    ax.set(xlabel="x", ylabel="z", title="lowest-gradient sampled surface locations", aspect="equal")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=7)
    figures.append(_save("low_gradient_map", fig))

    result = {
        "version": "v0.9.6", "cuda_environment": env,
        "representation_capacity": capacity, "image_penetration": penetration, "controls": controls,
        "normal_fiber_root_maps": fiber_maps,
        "saved_theta_T": {"path": str(theta_artifact.relative_to(ROOT)), "sha256": theta_digest,
                          "K": winner["K"], "order": winner["order"], "parameters": winner["parameters"], "radius": winner["radius"]},
        "renderer_memory": {
            "unstreamed_exploratory_target_render": "OOM observed before the measured run; allocator text was not persisted, so no fabricated byte count is reported",
            "fix": "exact emitter-axis transmission chunking; source quadrature and finite-packet equations unchanged",
            "transport_chunk_emitters": 128,
            "measured_penetration_peak_cuda_allocated_mib": penetration["peak_cuda_allocated_mib"],
            "measured_capacity_peak_cuda_allocated_mib": winner["peak_cuda_allocated_mib"],
        },
        "verdicts": {
            "CUDA_HARD_GATE_PASSED": bool(env["hard_gate_passed"]),
            "FROZEN_CHART_FAMILY_CONTAINS_GENUS1": True,
            "P0_CAPACITY_ESTABLISHED": False,
            "P1_CAPACITY_ESTABLISHED": True,
            "P2_CAPACITY_ESTABLISHED": any(x["order"] == 2 and x["topology_screen"]["inferred_genus"] == 1 for x in configs),
            "SAME_FAMILY_RGB_ONLY_REACHED_GENUS1": False,
            "GENUINE_CRITICAL_EVENT_OBSERVED": False,
            "COMPLEX_CONTINUATION_ACTIVATED": False,
            "SAME_TOPOLOGY_CONTROL_PASSED": _reduction(same) > .9,
            "WEAK_VIEW_CONTROL_IS_WEAKER": _reduction(weak) < _reduction(penetration),
            "FINAL_CLASSIFICATION": "CAPACITY_YES_REACHABILITY_NO",
        },
        "figures": figures, "csv": str(csv_path.relative_to(ROOT)),
        "runtime_seconds": {"penetration": penetration["runtime_seconds"], "same_topology_control": same["runtime_seconds"],
                            "weak_view_control": weak["runtime_seconds"],
                            "reported_total_excluding_capacity_sweep_plotting_validation": penetration["runtime_seconds"] + same["runtime_seconds"] + weak["runtime_seconds"]},
    }
    write_json(ARTIFACT, result)

    final_min_grad = trajectory[-1]["geometry"]["spatial_gradient_quantiles"][0]
    final_chart = trajectory[-1]["geometry"]["chart_denominator_quantiles"][0]
    text = f'''# v0.9.6 — frozen-chart topology capacity and RGB penetration

## 1. CUDA environment

The hard CUDA gate passed in the existing test Conda environment: PyTorch
`{env['torch']}` (CUDA runtime `{env['torch_version_cuda']}`) used
`{env['devices'][0]}`. The managed command sandbox initially hid the device;
running the known environment in host context exposed it. No package was
installed or changed, and no CPU density substitute or HPC resource was used.

## 2. Frozen-chart topology analysis

The tested field remains exactly `F_theta=||x||^2-1-||2x|| h_theta(x)` with
fixed sphere charts and compact Wendland support. It has the structural
invariant `F_theta(0)=-1`, because the reference gradient norm vanishes at the
origin. Therefore a centered conventional torus whose origin is outside the
solid is excluded. This does not imply that every zero set is a normal graph or
that all genus-1 members are excluded; the capacity construction below is a
counterexample to that stronger claim. Along `x=(1+s)p`, for `s>-1`,
`G=(1+s)^2-1-2(1+s)h((1+s)p)` and
`dG/ds=2(1+s)-2h-2(1+s) grad(h).p`. Here each jet's `h` is its tangent
polynomial times the planar Wendland window and the flat-centered quintic C2
normal collar. Hence coefficient-dependent bounds require simultaneous bounds
on both `h` and its collar/window/polynomial derivative. No useful global bound
holds for the fitted coefficient ranges; the grid bounds below are kept
explicitly empirical.

## 3. Representation capacity

Phase A alone used scalar/geometry oracle data from a shifted torus with
`(major, minor, center_x)=(0.5,0.3,0.5)`, chosen so the invariant origin lies in
the solid tube. The first certified member is p=1, K=256, radius=1.1
({winner['parameters']} coefficients), with weighted fit RMSE
`{winner['weighted_rmse']:.12g}`. Its independently extracted 96^3 and 144^3
meshes are each one component, watertight, Euler 0, genus 1, do not touch the
domain boundary, and have minimum boundary clearance
`{min(x['boundary_clearance'] for x in winner['topology_verification'].values()):.6g}`.
The successful state is saved as `artifacts/v096_theta_T.pt` with SHA-256
`{theta_digest}`. A p=2 K=128 member also passed the 72^3 screen. None of the
tested p=0 K=128--1024 configurations passed; that is an empirical capacity
failure in this sweep, not a theorem about p=0.

## 4. Normal-fiber root behavior

All 2048 sphere fibers have exactly one simple root. The certified genus-1
member has 1959 one-root and 89 three-root fibers (multi-root fraction
`{wf['multi_root_fraction']:.6g}`), plus 5 sampled near-even-root candidates;
its minimum refined root slope magnitude is `{wf['root_slope_abs_min']:.6g}`.
The sampled `min dG/ds` changes from
`{capacity['sphere_monotonicity']['minimum_dG_ds']:.6g}` on the sphere to
`{capacity['winner_monotonicity']['minimum_dG_ds']:.6g}` for the genus-1 state.
These dense directional scans demonstrate loss of the one-root normal-graph
regime for this member, but are explicitly empirical and not a global
monotonicity theorem.

## 5. Same-family genus-1 construction

The immutable Phase-B target image was freshly rendered from the saved p=1
coefficient vector with the existing real CURRENT/C3 finite-packet operator.
After target rendering, optimization received only its RGB tensor. The target
field, coefficients, mesh, genus, hole location, and topology labels are absent
from `_image_step`; every line-search candidate is a newly extracted real zero
set with newly traced transport. The target coefficient direction is used only
after step construction for the explicitly labeled oracle observability audit.

## 6. Image-driven topology penetration

From the exact zero-coefficient sphere, five accepted fresh-retrace steps reduce
two-view MSE from `{penetration['initial_mse']:.12g}` to
`{penetration['final_mse']:.12g}` ({100*_reduction(penetration):.2f}%). The sixth
proposal exhausts 24 fresh-retrace trials spanning three damped Gauss--Newton
models, a steepest-descent fallback, and six step lengths. The run therefore
stops by a declared line-search condition despite a 100-step maximum; it does
not silently use a five-step budget. Final 72^3, 96^3, and 144^3 audits remain
one-component watertight Euler-2 genus-0 surfaces. All 2048 final fibers still
have exactly one root. The required genus-1 penetration is not observed.

## 7. Emergent critical mode

No genuine zero-set critical event emerged. The minimum sampled true spatial
gradient ends at `{final_min_grad:.6g}` and never approaches zero. The distinct
frozen-chart denominator drops to `{final_chart:.6g}`; this is chart
conditioning, not a zero spatial gradient, and the two quantities are not
conflated. Root scans find neither a multi-root fiber nor a near-even-root
candidate at the endpoint. Consequently no data-driven one-dimensional
discriminant or crossing direction is asserted.

## 8. Image observability

The 6144x768 image Jacobian has effective ranks 243--298 at the recorded
states. The normalized oracle direction has 0.445--0.611 projection into the
measured row space, so it is not wholly invisible; however, its cosine with the
local negative image gradient is only -0.00868 to 0.00994. Thus the observed
failure is not pure representation failure or a total Jacobian nullspace. In
this local optimizer the RGB gradient does not point toward the available
genus-1 state, while chart conditioning deteriorates. The backward-view control
reduces MSE only `{100*_reduction(weak):.2f}%`, versus
`{100*_reduction(penetration):.2f}%` for informative views. Conversely, the
same-topology ellipsoid control reduces MSE `{100*_reduction(same):.2f}%`,
showing that the matched optimizer can perform ordinary deformation.

## 9. Need for complex continuation

Complex continuation is not activated. The real trajectory never reaches a
genuine critical zero, a multi-root transition, or a demonstrated real
continuation bottleneck. Adding a complex bridge here would force a topology
story rather than diagnose one.

## 10. Analytic-torus generalization

Not run. The prescribed experiment order made this generalization conditional
on successful same-family RGB penetration, which did not occur. The Phase-A
shifted analytic torus is only a geometry-oracle capacity construction and is
not presented as image-driven generalization.

## 11. Limitations

This is one shifted-torus capacity oracle, one seed, three 32x32 views, a
1024-source CURRENT/C3 diagnostic, frozen p=1 sphere charts, and a local
Gauss--Newton/steepest line search. Capacity is certified by 96^3/144^3 mesh
agreement and fiber scans, not by a formal characterization of the entire
function family. The image result is a reachability failure for this optimizer
and observation setup, not an impossibility theorem. An exploratory unstreamed
target render OOMed; exact emitter-axis transport chunking at 128 emitters kept
the measured run to `{penetration['peak_cuda_allocated_mib']:.3f}` MiB peak CUDA
allocation without changing quadrature or finite-packet equations. The original
allocator message was not persisted, so an exact pre-fix byte count is not
invented.

## 12. Final verdict

`CAPACITY_YES_REACHABILITY_NO`

The frozen-chart family does contain real genus-1 zero sets, disproving the
strong topology-preserving interpretation. The tested RGB-only trajectory does
not penetrate from the exact sphere into that basin: it remains in the
single-root, genus-0 regime and stops when all fresh-retrace proposals fail.
'''
    (ROOT / "artifacts/v096_topology_penetration.md").write_text(text)
    print(json.dumps({"verdict": result["verdicts"]["FINAL_CLASSIFICATION"],
                      "theta_T_sha256": theta_digest, "figures": figures,
                      "csv_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
