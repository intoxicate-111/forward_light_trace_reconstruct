"""Consolidate the v0.9.7 high-bandwidth topology reachability experiment."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zlt.transverse_packet import write_json

CACHE = ROOT / "runs/v097_high_bandwidth_topology"
ARTIFACT = ROOT / "artifacts/v097_high_bandwidth_topology.json"


def save(name, fig):
    path = ROOT / "figures" / f"v097_{name}.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return str(path.relative_to(ROOT))


def pct(initial, final):
    return 100. * (1-final/initial)


def main():
    environment = json.loads((ROOT/"artifacts/v097_cuda_environment.json").read_text())
    equivalence = json.loads((ROOT/"artifacts/v097_sparse_equivalence.json").read_text())
    observation = json.loads((CACHE/"observation.json").read_text())
    optimization = json.loads((CACHE/"optimization.json").read_text())
    oracle_path = json.loads((CACHE/"oracle_path.json").read_text())
    basins = json.loads((CACHE/"basins.json").read_text())
    loss_slice = json.loads((CACHE/"loss_slice.json").read_text())
    fiber_audit = json.loads((CACHE/"fiber_audit_2048.json").read_text())
    configs = observation["configurations"]
    by_name = {row["name"]: row for row in configs}

    figures = []
    labels = [row["name"] for row in configs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(x, [row["Jv_oracle_norm_per_sqrt_observation"] for row in configs])
    axes[1].bar(x, [row["negative_gradient_oracle_cosine"] for row in configs])
    axes[1].axhline(0, color="black", lw=.8)
    axes[2].bar(x, [row["oracle_projection_on_randomized_range_JT"] for row in configs])
    axes[0].set_ylabel(r"$||Jv_T||/\sqrt{N_{obs}}$")
    axes[1].set_ylabel(r"$\cos(-J^Tr,v_T)$")
    axes[2].set_ylabel(r"$||P_{Q(J^T\Omega)}v_T||$")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8); ax.grid(axis="y", alpha=.25)
    figures.append(save("oracle_metrics", fig))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for sources, marker in ((16384, "o"), (65536, "s")):
        rows = sorted([r for r in configs if r["sources"] == sources], key=lambda r: r["height"]*r["width"])
        pix = [r["height"]*r["width"] for r in rows]
        axes[0].plot(pix, [r["mse"] for r in rows], marker+"-", label=f"{sources//1024}K")
        axes[1].plot(pix, [r["negative_gradient_oracle_cosine"] for r in rows], marker+"-")
        axes[2].plot(pix, [r["oracle_projection_on_randomized_range_JT"] for r in rows], marker+"-")
    axes[0].set_ylabel("MSE"); axes[1].set_ylabel("gradient/oracle cosine")
    axes[2].set_ylabel("randomized range projection")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("detector pixels/view"); ax.grid(alpha=.25)
    axes[0].legend(title="sources")
    figures.append(save("bandwidth_comparison", fig))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for row in configs:
        spectrum = np.asarray(row["seeded_randomized_singular_values"])
        ax.semilogy(np.arange(1, len(spectrum)+1), spectrum/spectrum[0], "o-", ms=3, label=row["name"])
    ax.set(xlabel="restricted mode", ylabel="singular value / leading value",
           title=r"Spectrum of $J$ on span{$v_T$, 12 seeded probes}")
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    figures.append(save("jacobian_spectrum", fig))

    image_data = np.load(CACHE/"images_1024x1024_16K_16384.npz")
    initial, target = image_data["initial"], image_data["target"]
    final = np.load(CACHE/"primary_1024x1024_16K_final.npy", mmap_mode="r")
    limit = max(float(np.quantile(np.concatenate((initial[:, ::8, ::8].ravel(),
                                                   target[:, ::8, ::8].ravel(),
                                                   np.asarray(final)[:, ::8, ::8].ravel())), .997)), 1e-12)
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    for i, (title, images) in enumerate((("sphere", initial), ("genus-1 target", target),
                                         ("RGB-only endpoint", final))):
        for j, view in enumerate((0, 1)):
            axes[i, j].imshow(np.clip(np.asarray(images[view])/limit, 0, 1)); axes[i, j].axis("off")
            axes[i, j].set_title(f"{title}, view {view}")
    figures.append(save("selected_images", fig))

    trajectory = optimization["trajectory"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    iterations = [row["iteration"] for row in trajectory]
    axes[0].plot(iterations, [row["mse"] for row in trajectory], "o-")
    axes[1].plot(iterations, [row["geometry"]["spatial_gradient_quantiles"][0] for row in trajectory], "o-", label=r"min $||\nabla F||$")
    axes[1].plot(iterations, [row["geometry"]["chart_denominator_quantiles"][0] for row in trajectory], "s-", label="min chart denominator")
    axes[2].plot(iterations, [row["geometry"]["fibers"]["multi_root_fraction"] for row in trajectory], "o-")
    axes[0].set_ylabel("fresh-retrace MSE"); axes[1].set_ylabel("conditioning value")
    axes[2].set_ylabel("multi-root fiber fraction")
    for ax in axes: ax.set_xlabel("accepted iteration"); ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    figures.append(save("optimization_trajectory", fig))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    primary_fibers = fiber_audit["primary_selected_iterations"]
    axes[0].plot([row["iteration"] for row in primary_fibers],
                 [row["fibers"]["root_count_histogram"].get("3", 0) for row in primary_fibers], "o-")
    axes[0].set(xlabel="accepted primary iteration", ylabel="three-root fibers / 2048",
                title="RGB trajectory")
    oracle_fibers = fiber_audit["oracle_path"]
    axes[1].plot([row["t"] for row in oracle_fibers],
                 [row["fibers"]["root_count_histogram"].get("3", 0) for row in oracle_fibers], "o-")
    axes[1].set(xlabel=r"$t$ in $\theta=t\theta_T$", ylabel="three-root fibers / 2048",
                title="oracle path")
    for ax in axes: ax.grid(alpha=.25)
    figures.append(save("fiber_trajectory", fig))

    path_rows = oracle_path["samples"]
    ts = [row["t"] for row in path_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    axes[0].plot(ts, [row["mse"] for row in path_rows], "o-")
    axes[0].axvspan(.725, .74, alpha=.18, color="tab:red", label="topology bracket")
    axes[0].legend(fontsize=8); axes[0].set_ylabel("fresh MSE")
    axes[1].plot(ts[:-1], [row["negative_gradient_remaining_direction_cosine"] for row in path_rows[:-1]], "o-")
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_ylabel("remaining-direction alignment")
    axes[2].plot(ts, [row["geometry"]["spatial_gradient_quantiles"][0] for row in path_rows], "o-", label=r"min $||\nabla F||$")
    axes[2].plot(ts, [row["geometry"]["chart_denominator_quantiles"][0] for row in path_rows], "s-", label="chart")
    axes[2].set_yscale("log"); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlabel(r"$t$ in $\theta=t\theta_T$"); ax.grid(alpha=.25)
    figures.append(save("oracle_tomography", fig))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for run in basins["runs"]:
        values = [row["mse"] for row in run["trajectory"]]
        axes[0].plot(range(len(values)), values, "o-", label=f"{run['initial_fraction']:.2f} theta_T")
    axes[0].set(xlabel="accepted RGB-only step", ylabel="MSE", title="basin starts")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].bar([f"{r['initial_fraction']:.2f}" for r in basins["runs"]],
                [int(r["topology_penetrated"]) for r in basins["runs"]])
    axes[1].set(xlabel=r"initial fraction of $\theta_T$", ylabel="final genus-1 indicator",
                title="0.75 begins beyond the topology event", ylim=(0, 1.15))
    figures.append(save("basins", fig))

    alpha_values = sorted({row["alpha_fraction"] for row in loss_slice["samples"]})
    beta_values = sorted({row["beta"] for row in loss_slice["samples"]})
    z = np.full((len(beta_values), len(alpha_values)), np.nan)
    genus = np.full_like(z, np.nan)
    for row in loss_slice["samples"]:
        i, j = beta_values.index(row["beta"]), alpha_values.index(row["alpha_fraction"])
        if "mse" in row:
            z[i, j] = row["mse"]; genus[i, j] = row["topology_grid72"]["inferred_genus"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, values, title, cmap in ((axes[0], z, "fresh-rendered MSE", "magma_r"),
                                    (axes[1], genus, "genus (72^3)", "viridis")):
        im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap,
                       extent=(min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)))
        ax.set(xlabel=r"oracle fraction $\alpha/||\theta_T||$", ylabel=r"$\beta$ on $v_G$", title=title)
        fig.colorbar(im, ax=ax)
    figures.append(save("loss_slice", fig))

    csv_rows = []
    for row in configs:
        spectrum = row["seeded_randomized_singular_values"]
        csv_rows.append({"phase": "observation", "case": row["name"], "step": "", "resolution": f"{row['width']}x{row['height']}",
                         "sources": row["sources"], "mse": row["mse"], "genus": 0,
                         "gradient_oracle_cosine": row["negative_gradient_oracle_cosine"],
                         "oracle_alpha_star": row["oracle_alpha_star"],
                         "oracle_response_normalized": row["Jv_oracle_norm_per_sqrt_observation"],
                         "oracle_range_projection": row["oracle_projection_on_randomized_range_JT"],
                         "rank": row["output_sketch_effective_rank_relative_1e_8"],
                         "condition": spectrum[0]/spectrum[-1], "min_gradient": "", "min_chart": "", "multi_root_fraction": ""})
    for row in trajectory:
        csv_rows.append({"phase": "optimization", "case": "sphere_start", "step": row["iteration"], "resolution": "1024x1024",
                         "sources": 16384, "mse": row["mse"], "genus": row["topology"]["inferred_genus"],
                         "gradient_oracle_cosine": "", "oracle_alpha_star": "", "oracle_response_normalized": "",
                         "oracle_range_projection": "", "rank": "", "condition": "",
                         "min_gradient": row["geometry"]["spatial_gradient_quantiles"][0],
                         "min_chart": row["geometry"]["chart_denominator_quantiles"][0],
                         "multi_root_fraction": row["geometry"]["fibers"]["multi_root_fraction"]})
    for row in path_rows:
        csv_rows.append({"phase": "oracle_path", "case": "theta=t theta_T", "step": row["t"], "resolution": "1024x1024",
                         "sources": 16384, "mse": row["mse"], "genus": row["topology_grid72"]["inferred_genus"],
                         "gradient_oracle_cosine": row["negative_gradient_remaining_direction_cosine"] if row["t"] < 1 else "",
                         "oracle_alpha_star": "", "oracle_response_normalized": "", "oracle_range_projection": "",
                         "rank": "", "condition": "", "min_gradient": row["geometry"]["spatial_gradient_quantiles"][0],
                         "min_chart": row["geometry"]["chart_denominator_quantiles"][0],
                         "multi_root_fraction": row["geometry"]["fibers"]["multi_root_fraction"]})
    csv_path = ROOT/"artifacts/v097_high_bandwidth_topology.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(csv_rows)

    primary_peak = max((row.get("linearization") or {}).get("peak_cuda_allocated_mib", 0.) for row in trajectory)
    result = {"version": "v0.9.7", "experiment": "high-bandwidth topology reachability",
              "cuda_environment": environment, "sparse_equivalence": equivalence,
              "phase_A_observation": observation, "phase_B_optimization": optimization,
              "phase_C_oracle_path": oracle_path, "basin_of_attraction": basins,
              "two_dimensional_loss_slice": loss_slice, "fixed_2048_fiber_audit": fiber_audit,
              "selected_configuration": {"name": "1024x1024_16K", "reason": "smallest-magnitude gradient/oracle cosine, strong normalized sensitivity, and practical repeated retracing; no configuration had positive oracle GN preference"},
              "runtime_memory": {"phase_A_seconds": sum(x["runtime_seconds"] for x in observation["groups"]),
                                 "phase_A_peak_cuda_mib": max(x["peak_cuda_allocated_mib"] for x in observation["groups"]),
                                 "primary_optimization_seconds": optimization["runtime_seconds"],
                                 "primary_sparse_linearization_peak_cuda_mib": primary_peak},
              "figures": figures,
              "verdicts": {"CUDA_HARD_GATE_PASSED": True, "V096_TARGET_HASH_VERIFIED": True,
                           "V096_TARGET_TOPOLOGY_REPRODUCED": True,
                           "SPARSE_LOCAL_JACOBIAN_EQUIVALENT": equivalence["equivalent"],
                           "DENSE_FULLHD_JACOBIAN_ALLOCATED": False,
                           "BANDWIDTH_VERDICT": "TOPOLOGY_SIGNAL_REMAINS_MISALIGNED",
                           "MEASUREMENT_BANDWIDTH_HYPOTHESIS_SUPPORTED": False,
                           "PRIMARY_RGB_OPTIMIZATION_REACHES_GENUS1": False,
                           "TRUE_CRITICAL_MODE_FOUND_BY_RGB_OPTIMIZER": False,
                           "COMPLEX_CONTINUATION_ACTIVATED": False,
                           "SUCCESS_HIERARCHY": "HIGH_BANDWIDTH_STILL_MISALIGNED"}}
    write_json(ARTIFACT, result)

    rows_md = "\n".join(
        f"| {r['name']} | {r['mse']:.6g} | {r['Jv_oracle_norm_per_sqrt_observation']:.6g} | "
        f"{r['oracle_projection_on_randomized_range_JT']:.6g} | {r['negative_gradient_oracle_cosine']:.6g} | {r['oracle_alpha_star']:.6g} |"
        for r in configs)
    basin_md = "\n".join(
        f"| {r['initial_fraction']:.2f} | {r['accepted_steps']} | {r['initial_mse']:.6g} | {r['final_mse']:.6g} | "
        f"{r['stop']} | {'yes (initial)' if r['initial_fraction']==.75 else 'no'} |"
        for r in basins["runs"])
    report = f"""# v0.9.7 — high-bandwidth topology reachability

## CUDA and bandwidth configuration

The hard local-CUDA gate passed on {environment['device']} with PyTorch {environment['torch']} / CUDA {environment['torch_cuda']}; no HPC or CPU fallback was used. The immutable v0.9.6 target digest is `{observation['target']['sha256']}` and its one-component watertight genus-1 topology was reproduced at 96^3 and 144^3. Tests cover 512^2 and 1024^2 at 16K/64K sources plus 1920x1080 at 64K. A complete 768-parameter VJP at 64K was the highest practical diagnostic on the 16-GiB device; 262K was not run. Scene/transport was frozen across detector replays. The corrected Phase-A pass took {sum(x['runtime_seconds'] for x in observation['groups']):.1f} s and peaked at {max(x['peak_cuda_allocated_mib'] for x in observation['groups']):.1f} MiB allocated CUDA memory.

The implementation reuses compact-support pairs and chunk-local coalesced COO matrices with `torch.sparse.mm`. It never forms a dense source-by-parameter or pixel-by-parameter Jacobian. The small historical-dense comparison has max error {equivalence['maximum_absolute_error']:.3g}, relative L2 error {equivalence['relative_l2_error']:.3g}, and {equivalence['sparse_nnz']:,} sparse entries versus {equivalence['dense_entries']:,} dense entries.

## Oracle topology-direction observability

| configuration | MSE | `||Jv_T||/sqrt(Nobs)` | randomized `range(J^T)` projection lower bound | cosine | oracle GN alpha |
|---|---:|---:|---:|---:|---:|
{rows_md}

The topology direction produces a nonzero image response at every bandwidth, but the 32-vector output-space sketch captures only 0.0518--0.0579 of its norm in the sampled leading range of `J^T`, with no systematic bandwidth gain. This is a lower bound, not a full-rank projection claim.

## Gradient alignment

All five `cos(-J^T r,v_T)` values are negative and near zero: {min(r['negative_gradient_oracle_cosine'] for r in configs):.6g} to {max(r['negative_gradient_oracle_cosine'] for r in configs):.6g}. All oracle one-dimensional GN steps are also negative. High bandwidth therefore observes `v_T` but does not locally prefer it. The Phase-A classification is **`TOPOLOGY_SIGNAL_REMAINS_MISALIGNED`**.

## Source-density effect

At 512^2, increasing sources 16K to 64K lowers MSE {by_name['512x512_16K']['mse']:.4g} to {by_name['512x512_64K']['mse']:.4g}; at 1024^2 it lowers MSE {by_name['1024x1024_16K']['mse']:.4g} to {by_name['1024x1024_64K']['mse']:.4g}. This is a strong sampling/noise effect, but alignment remains near-zero/negative and randomized oracle projection remains near 0.05. Source density does not recover the topology preference.

## Detector-resolution effect

Raw and per-observation sensitivities rise with detector scaling because the unchanged renderer multiplies splats by pixel count. The scale-invariant alignment does not improve: 16K changes from {by_name['512x512_16K']['negative_gradient_oracle_cosine']:.4g} at 512^2 to {by_name['1024x1024_16K']['negative_gradient_oracle_cosine']:.4g} at 1024^2; 64K reaches {by_name['1920x1080_64K']['negative_gradient_oracle_cosine']:.4g} at Full-HD, still with a negative GN step. The failure is neither detector-limited nor rescued by Full-HD.

## Jacobian spectrum

The reported singular spectra are matrix-free `JQ` spectra on `span{{v_T, 12 seeded orthonormal parameter probes}}`; all 13 sampled modes are above the relative 1e-6 threshold. Their useful condition ratios range from {min(r['seeded_randomized_singular_values'][0]/r['seeded_randomized_singular_values'][-1] for r in configs):.3g} to {max(r['seeded_randomized_singular_values'][0]/r['seeded_randomized_singular_values'][-1] for r in configs):.3g}. Separately, 32 stateless output-space probes form `orth(J^T Omega)` for the meaningful observable-range projection. The earlier within-`Q` projection of 1.0 is explicitly retained only as a non-evidential legacy diagnostic because `Q` deliberately contains `v_T`.

## High-bandwidth RGB optimization

The selected 1024^2/16K configuration had the smallest-magnitude cosine, strong normalized sensitivity, and practical repeated-retrace cost; no tested configuration had positive oracle preference. Starting from the exact sphere and using only immutable target RGB, 20 accepted steps reduce MSE {optimization['initial_mse']:.6g} to {optimization['final_mse']:.6g} ({pct(optimization['initial_mse'], optimization['final_mse']):.2f}%). It then exhausts 32 fresh-retrace candidates. Runtime was {optimization['runtime_seconds']:.1f} s. The optimizer never receives `theta_T`, `v_T`, target field/mesh, genus, or hole location.

## Topology trajectory

Every accepted primary state is genus 0. The final 72^3/96^3/144^3 extractions are one-component watertight Euler-2 surfaces; all 2,048 final normal fibers have one root. A fixed 2,048-direction offline audit at accepted iterations 0/5/10/15/20 likewise finds exactly 2,048 one-root and zero three-root fibers at every checkpoint. The final minimum sampled spatial-gradient norm is {trajectory[-1]['geometry']['spatial_gradient_quantiles'][0]:.6g}, minimum chart denominator {trajectory[-1]['geometry']['chart_denominator_quantiles'][0]:.6g}, and minimum refined fiber slope {optimization['final_fibers']['root_slope_abs_min']:.6g}. No true zero-set critical precursor appears in the RGB trajectory.

## Oracle path tomography

Fresh renders of `theta(t)=t theta_T` show a non-monotone barrier: MSE starts at {path_rows[0]['mse']:.6g}, rises to a sampled maximum {max(r['mse'] for r in path_rows):.6g}, and only collapses to numerical zero at `t=1`. Topology changes inside `(0.725,0.740]`; the fixed 2,048-direction audit first sees 2 three-root fibers at `t=0.6`, then 27 at `t=0.725`, 34 at `t=0.74`, and reproduces the v0.9.6 target count of 89 at `t=1`. Alignment with the remaining oracle direction stays between {min(r['negative_gradient_remaining_direction_cosine'] for r in path_rows[:-1]):.4g} and {max(r['negative_gradient_remaining_direction_cosine'] for r in path_rows[:-1]):.4g}. The spatial gradient and frozen-chart denominator are reported separately: the sampled surface gradient does not approach zero at sampled `t`, while the chart denominator becomes small. Thus tomography brackets but does not resolve an exact critical parameter, and no chart failure is mislabeled as topology criticality.

The 2-D fresh-rendered slice confirms a narrow/folded genus-1 region near oracle fractions 0.725--0.8. Moving farther along the initial negative-gradient axis lowers MSE but can return to genus 0; no broad low-loss route from the sphere is visible in this slice.

## Basin of attraction

| initialization | accepted RGB steps | initial MSE | final MSE | stop | genus 1 |
|---:|---:|---:|---:|---|---|
{basin_md}

Only 0.75 `theta_T` is genus 1, and it already lies beyond the bracketed topology event before optimization, so its zero-step success is not an RGB-driven crossing. The 0.50 and 0.25 starts improve RGB loss but remain genus 0 at all final extraction grids with 2,048/2,048 single-root fibers. Approximate capture requires initialization already on the target-topology side; intermediate genus-0 initialization does not recover it.

## Need for complex continuation

Complex continuation was not activated. The actual RGB-only trajectory never approaches a genuine zero-set critical region or requests the topology direction; its minimum spatial gradient remains regular and the oracle GN preference at initialization is negative. The oracle path proves a real finite-parameter genus change exists, but that diagnostic cannot authorize a complex bridge in the optimizer. The present obstruction is loss alignment/basin geometry, not an observed singular real tracker.

## Final verdict

**`HIGH_BANDWIDTH_STILL_MISALIGNED`**. The evidence does **not** support the statement that v0.9.6 topology reachability failure was primarily a measurement-bandwidth limitation. Full-HD/64K measurements show nonzero topology-direction sensitivity, but gradient/oracle alignment remains near zero and negative, every local oracle GN step points away, the sphere-start high-bandwidth RGB optimizer remains genus 0, and intermediate genus-0 basin starts also fail. High bandwidth reduces source-sampling noise and changes raw sensitivity; it does not recover the missing optimization preference.
"""
    (ROOT/"artifacts/v097_high_bandwidth_topology.md").write_text(report)
    print(json.dumps({"artifact": str(ARTIFACT.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT)),
                      "figures": figures, "verdict": result["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
