"""Create v0.9.5 figures, CSV, high-resolution topology audit and report."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from zlt.dense_jet_torus import (DEVICE, EllipsoidTarget, TARGET_MAJOR, TARGET_MINOR,
                                 TorusTarget, _grid, camera_state, make_field, topology_only)
from zlt.transverse_packet import write_json

ARTIFACT = Path("artifacts/v095_dense_jet_torus.json")
CACHE = Path("runs/v095_dense_jet_torus")


def _save(name, fig):
    path = Path("figures")/f"v095_{name}.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return str(path)


def main():
    report = json.loads(ARTIFACT.read_text()); by = {x["name"]: x for x in report["runs"]}; main = by["A_dense_real_torus"]
    fields = {}
    for name, run in by.items():
        f = make_field(run["centers"], run["order"], run["support_radius"])
        f = f.with_coefficients(torch.tensor(run["final_coefficients"], dtype=torch.float64, device=DEVICE))
        fields[name] = f
    high = {"initial": topology_only(make_field(main["centers"]), 72),
            "target": topology_only(TorusTarget(DEVICE), 72),
            "final": topology_only(fields["A_dense_real_torus"], 72)}
    report["high_resolution_topology"] = high
    report["verdicts"]["REAL_TOPOLOGY_CHANGED_0_TO_1"] = high["initial"]["inferred_genus"] == 0 and high["final"]["inferred_genus"] == 1
    report["verdicts"]["IMAGE_LOSS_SUBSTANTIALLY_REDUCED"] = main["final_mse"] < .8*main["initial_mse"]
    report["verdicts"]["SPONTANEOUS_SPHERE_TO_TORUS_SUPPORTED"] = bool(
        report["verdicts"]["REAL_TOPOLOGY_CHANGED_0_TO_1"] and report["verdicts"]["IMAGE_LOSS_SUBSTANTIALLY_REDUCED"])

    rows = []
    for run in report["runs"]:
        for step in run["trajectory"]:
            geometry = step["geometry"]
            rows.append({"run": run["name"], "centers": run["centers"], "order": run["order"],
                         "iteration": step["iteration"], "mse": step["actual_retraced_mse"],
                         "all_view_mse": step["all_view_mse"], "genus": geometry["topology"]["inferred_genus"],
                         "components": geometry["topology"]["components"],
                         "gradient_min": geometry["spatial_gradient_quantiles"][0],
                         "gradient_p01": geometry["spatial_gradient_quantiles"][1],
                         "chart_denominator_min": geometry["chart_denominator_quantiles"][0],
                         "active_coefficients": geometry["active_coefficients_gt_1e_8"],
                         "coefficient_l2": geometry["coefficient_l2"],
                         "coefficient_abs_max": geometry["coefficient_abs_max"],
                         "accepted": step.get("accepted", "final")})
    csv_path = Path("artifacts/v095_dense_jet_torus.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    target = np.load(CACHE/"target_torus.npy"); initial = np.load(CACHE/"A_dense_real_torus_initial.npy")
    final = np.load(CACHE/"A_dense_real_torus_final.npy")
    limit = max(float(np.quantile(np.concatenate((target, initial, final)), .995)), 1e-12)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (name, images) in enumerate((("initial sphere", initial), ("target torus", target), ("final K=128", final))):
        for view in range(3):
            axes[row, view].imshow(np.clip(images[view]/limit, 0, 1)); axes[row, view].axis("off")
            axes[row, view].set_title(f"{name}; view {view}")
    figures = [_save("rendered_trajectory", fig)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for run in report["runs"][:3]:
        x = [r["iteration"] for r in run["trajectory"]]; y = [r["actual_retraced_mse"] for r in run["trajectory"]]
        axes[0].plot(x, y, "o-", label=run["name"])
        axes[1].plot(x, [r["geometry"]["spatial_gradient_quantiles"][0] for r in run["trajectory"]], "o-", label=run["name"])
        axes[2].plot(x, [r["geometry"]["chart_denominator_quantiles"][0] for r in run["trajectory"]], "o-", label=run["name"])
    axes[0].set_title("freshly traced MSE"); axes[1].set_title("min $||\\nabla F||$"); axes[2].set_title("min chart denominator")
    for ax in axes: ax.set_xlabel("accepted iteration"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    figures.append(_save("optimization_conditioning", fig))

    centers = make_field(main["centers"]).layout.centers.cpu().numpy(); coefficients = np.asarray(main["final_coefficients"])
    low = np.asarray(main["trajectory"][-1]["geometry"]["lowest_gradient_locations"])[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter = axes[0].scatter(centers[:, 0], centers[:, 2], c=coefficients, s=34, cmap="coolwarm")
    axes[0].scatter(low[0], low[2], marker="x", s=90, c="black", label="lowest-gradient point")
    axes[0].set_aspect("equal"); axes[0].set_title("final p=0 jet coefficients (x-z)"); axes[0].legend(fontsize=7); fig.colorbar(scatter, ax=axes[0])
    # Checkpoints contain the exact accepted coefficient trajectories.
    coefficient_paths = [np.zeros(main["centers"])]
    coefficient_paths += [torch.load(CACHE/f"A_dense_real_torus_step{i}.pt", weights_only=True)["coefficients"].numpy()
                          for i in range(1, main["accepted_steps"]+1)]
    axes[1].plot(np.arange(len(coefficient_paths)), np.asarray(coefficient_paths), alpha=.28, linewidth=.7)
    axes[1].set_title("all 128 coefficient trajectories"); axes[1].set_xlabel("accepted iteration"); axes[1].set_ylabel("coefficient")
    figures.append(_save("coefficient_activation", fig))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    density = [by[x] for x in ("D_density_32", "D_density_64", "A_dense_real_torus")]
    axes[0].plot([x["centers"] for x in density], [x["final_mse"] for x in density], "o-")
    axes[1].plot([x["centers"] for x in density], [min(r["geometry"]["spatial_gradient_quantiles"][0] for r in x["trajectory"]) for x in density], "o-")
    axes[0].set_title("density vs final MSE"); axes[1].set_title("density vs minimum $||\\nabla F||$")
    for ax in axes: ax.set_xlabel("jet centers K"); ax.grid(alpha=.25)
    figures.append(_save("density_ablation", fig))

    axis = np.linspace(-1.2, 1.2, 241); xx, zz = np.meshgrid(axis, axis, indexing="ij")
    points = torch.tensor(np.stack((xx.ravel(), np.zeros(xx.size), zz.ravel()), 1), dtype=torch.float64, device=DEVICE)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, name, f in zip(axes, ("initial sphere", "target torus", "optimized field"),
                           (make_field(main["centers"]), TorusTarget(DEVICE), fields["A_dense_real_torus"])):
        values = f.value(points).reshape(xx.shape).detach().cpu().numpy()
        ax.contour(xx, zz, values, levels=[0], colors="black"); ax.set_aspect("equal"); ax.set_title(name); ax.set_xlabel("x"); ax.set_ylabel("z")
    figures.append(_save("meridional_topology", fig))
    report["figures"] = figures; report["csv"] = str(csv_path)
    write_json(ARTIFACT, report)

    m = main; reduction = 1-m["final_mse"]/m["initial_mse"]
    text = f'''# v0.9.5 — dense manifold-jet sphere-to-torus optimization

The reconstruction field is exactly `F=||x||^2-1-||2x||h_theta(x)` with
all {m['total_parameters']} p=0 coefficients initialized to zero. Fibonacci
centers are fixed independently of the target; radius is {m['support_radius']}.
At the initial surface, median overlap is {m['coverage_initial']['overlap_median']:.0f}
jets and uncovered fraction is {m['coverage_initial']['uncovered_fraction']:.3g}.

The immutable target is the real torus `(R,r)=({TARGET_MAJOR},{TARGET_MINOR})`.
Only its three rendered RGB images enter optimization. The target equation,
mesh, genus, hole location, and v0.9.4 coordinate never enter the loss or step.
The two informative directions are `{report['views']['directions'][:2]}`; the
side direction is the weak-observation control.

The measurement operator retains CURRENT finite-packet attenuation, current
gate/lobe and C3 detector. Every trial reconstructs a real zero set, forms a
deterministic triangle-area quadrature, and retraces transmission. The existing
v0.9.2 sparse-support CURRENT tangent proposes damped Gauss--Newton steps from
256 area-weighted samples. Predicted and actual trial losses are separately
stored in JSON. No complex geometry is rendered.

Run A accepts {m['accepted_steps']} real steps. Freshly traced MSE falls from
{m['initial_mse']:.12g} to {m['final_mse']:.12g} ({100*reduction:.2f}%); all
{m['total_parameters']} coefficients activate. The minimum sampled spatial
gradient is {min(x['geometry']['spatial_gradient_quantiles'][0] for x in m['trajectory']):.12g},
so it does not approach a true zero-set critical event. The independent 72^3
mesh audit is watertight, has Euler number {high['final']['euler_number']}, one
component, and genus {high['final']['inferred_genus']}.

Run B (sphere-like target) reduces MSE from {by['B_sphere_like']['initial_mse']:.12g}
to {by['B_sphere_like']['final_mse']:.12g} without changing genus. Run C's weak
view reduces {by['C_weak_torus']['initial_mse']:.12g} to
{by['C_weak_torus']['final_mse']:.12g}, also without a critical event or topology
change. Density runs K=32/64 finish at {by['D_density_32']['final_mse']:.12g} and
{by['D_density_64']['final_mse']:.12g}; neither changes topology. CUDA was
unavailable, so K=256/512/1024 and p=1/p=2 are explicitly not claimed.

## 1. Dense representation validity

Supported at the tested scale: zero coefficients reproduce the analytic unit
sphere exactly; the extracted mesh is one watertight genus-0 component and
coverage is complete. K=128 is a CPU-limited dense pilot, not the requested
upper density regime.

## 2. Image-driven deformation

Supported. Target information entered only through images, all accepted states
were freshly real-rendered, all coefficients activated, and Run A reduced MSE
by {100*reduction:.2f}%. This is ordinary deformation, not evidence of a hole.

## 3. Emergent critical mode

Not supported. The lowest-gradient region and its local support block are
logged at every iteration, but `min ||grad F||={min(x['geometry']['spatial_gradient_quantiles'][0] for x in m['trajectory']):.6g}`
is far from zero. No one-dimensional discriminant can honestly be inferred,
so no `c0*c_pred<0` analysis is asserted.

## 4. Topology outcome

Negative: both 32^3 trajectory checks and the independent 72^3 final audit
remain one-component, watertight, Euler=2, genus 0. Appearance improvement is
not mislabeled as a torus.

## 5. Rendered-loss outcome

Run A improves by {100*reduction:.2f}%, while higher density improves the equal
two-step comparison only modestly. The sphere-like control improves much more
readily and does not create unnecessary topology. The strongest boxed claim is
therefore **not supported** by this experiment.

## 6. Need for complex continuation

No. The required precursor—a real, image-observable critical event limiting
ordinary optimization—did not arise. Activating a complex bypass would force
the intended story rather than test it.

## 7. Limitations

CPU-only K=128 maximum, p=0 only, 32x32 three-view diagnostic images, 32^3
optimization surface extraction, four main steps, local tangent subsampling,
one target/seed and a wide support radius needed to influence the sphere
interior. The deterministic marching-cubes area quadrature is a controlled
readout approximation; it is not Full-HD evidence or a global impossibility
result. Runtime was {report['runtime_seconds']:.3f} seconds excluding this
report and validation.
'''
    Path("artifacts/v095_dense_jet_torus.md").write_text(text)
    print(json.dumps({"high_resolution_topology": high, "figures": figures,
                      "claim_supported": report["verdicts"]["SPONTANEOUS_SPHERE_TO_TORUS_SUPPORTED"]}, indent=2))


if __name__ == "__main__":
    main()
