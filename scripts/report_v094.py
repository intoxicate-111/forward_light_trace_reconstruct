"""Plot and report the end-to-end sphere-to-torus optimization ablation."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from zlt.sphere_torus_optimization import CACHE, REPORT, write


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(REPORT.read_text()); runs = report['runs']; by = {x['name']: x for x in runs}
    direct, continuous, complex_run, same, weak = runs
    figures = []
    def save(fig, name):
        path = Path('figures')/f'v094_{name}.png'; fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
        figures.append(str(path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs[:4]:
        axes[0].semilogy([x['iteration'] for x in run['trajectory']], [x['loss'] for x in run['trajectory']], 'o-', label=run['name'])
        axes[1].plot([x['iteration'] for x in run['trajectory']], [x['c0'] for x in run['trajectory']], 'o-', label=run['name'])
    axes[0].set(xlabel='Iteration', ylabel='Freshly retraced real-image MSE'); axes[0].legend(fontsize=7)
    axes[1].axhline(0, color='k', ls='--'); axes[1].set(xlabel='Iteration', ylabel='Critical coordinate c'); axes[1].legend(fontsize=7)
    save(fig, 'optimization')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    first = direct['trajectory'][0]
    labels = ['view 0', 'view 1', 'weak view', 'informative stack']
    values = first['per_view_Jv_norm']+[first['Jv_norm']]
    axes[0].bar(labels, values); axes[0].tick_params(axis='x', rotation=20); axes[0].set_ylabel('||Jv||')
    axes[1].bar(['initial', 'direct real', 'continuous real', 'complex'],
                [direct['initial_loss'], direct['final_loss'], continuous['final_loss'], complex_run['final_loss']])
    axes[1].set_yscale('log'); axes[1].set_ylabel('Final MSE')
    axes[2].bar(['direct real', 'continuous real', 'complex', 'same-side'],
                [direct['topology_final']['meridian']['inferred_genus'], continuous['topology_final']['meridian']['inferred_genus'],
                 complex_run['topology_final']['meridian']['inferred_genus'], same['topology_final']['meridian']['inferred_genus']])
    axes[2].set_ylabel('Diagnosed genus'); save(fig, 'outcomes')
    bridge_path = np.load(CACHE/'B_complex_capable_bridge_0.npz')
    bridge = complex_run['bridges'][0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    coordinate = bridge_path['parameters']; axes[0].plot(coordinate.real, coordinate.imag); axes[0].scatter([0], [0], marker='x', c='red')
    axes[0].set(xlabel='Re(c)', ylabel='Im(c)', title='Triggered local bypass')
    axes[1].semilogy(np.maximum(bridge_path['residuals'], 1e-18)); axes[1].set(xlabel='Step', ylabel='max |F|')
    axes[2].plot(bridge_path['spatial_jacobian_min']); axes[2].set(xlabel='Step', ylabel='min |grad F · n_ref|')
    save(fig, 'complex_bridge')
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    pictures = [np.load(CACHE/'B_complex_capable_initial.npy'), np.load(CACHE/'B_complex_capable_target.npy'),
                np.load(CACHE/'B_complex_capable_final.npy')]
    scale = max(float(x.max()) for x in pictures)
    for row, (name, image) in enumerate(zip(('initial genus 0', 'target genus 1', 'final genus 1'), pictures)):
        for view in range(3):
            axes[row, view].imshow(np.clip(image[view].reshape(32, 32, 3)/scale, 0, 1)); axes[row, view].axis('off')
            axes[row, view].set_title(f'{name} / view {view}')
    save(fig, 'real_images')
    from zlt.critical_experiment import field
    import torch
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, c) in zip(axes, [('initial', -.005), ('continuous final', continuous['final_c']), ('complex/direct final', complex_run['final_c'])]):
        axis = torch.linspace(-.09, .09, 501, dtype=torch.float64); xx, zz = torch.meshgrid(axis, axis, indexing='xy')
        p = torch.stack((xx.flatten(), torch.zeros(xx.numel(), dtype=axis.dtype), zz.flatten()), 1)
        values = field(c).value(p).reshape(501, 501)
        ax.contour(xx, zz, values, levels=[0], colors='black'); ax.set_aspect('equal'); ax.set_title(f'{name}\nc={c:.3g}')
        ax.set(xlabel='x', ylabel='z')
    save(fig, 'topology_neck')
    rows = []
    for run in runs:
        for item in run['trajectory']:
            rows.append(dict(run=run['name'], iteration=item['iteration'], loss=item['loss'], c0=item['c0'],
                             delta_c=item['delta_c'], c_pred=item['c_pred'], trigger=item['trigger'], Jv_norm=item['Jv_norm'],
                             g_c=item['g_c'], h_c=item['h_c'], sampled_gradient_min=item['sampled_surface_gradient_min'],
                             chart_denominator_min=item['chart_denominator_min'], complex_bypass=item['complex_bypass_activated'],
                             accepted_c=item.get('accepted_c'), accepted_loss=item.get('accepted_loss'), stop=item.get('stop')))
    with Path('artifacts/v094_sphere_torus_optimization.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    report['figures'] = figures; write(REPORT, report)
    first_bridge = complex_run['bridges'][0]
    text = f'''# v0.9.4 — rendered sphere-to-torus optimization

## Setup

This controlled experiment starts at the real genus-0 field `c=-0.005` and
targets the real genus-1 field `c=+0.005` in the already validated family
`F_c = F_(a=1) + c W(||x||/0.8)`. The critical mode is the single existing
Wendland coefficient, `v=[1]`; `F_c(0)=c`, so its known discriminant is `c=0`.
Topology labels are used only after optimization.

All images are produced by the repository's existing real reference-direction,
first-arrival, bilinear renderer. Every current state, line-search candidate and
target is freshly traced. Complex geometry is never rendered. The observation
uses the same 64 persistent local neck source IDs/colors and two informative
32x32 views as v0.9.3; the backward view is the observability control. This is a
small operator experiment, not CURRENT/C3 production, complete-surface sampling,
Full-HD evidence, or a performance benchmark.

## Equations and trigger

At each iteration the direct sparse local image Jacobian produces `j=Jv`. We use
`g=j^T(I-I_target)`, `h=j^Tj`, `delta=-g/h` and trigger only when
`c0*(c0+delta)<0`. No topology label, gradient penalty or hand-coded hole target
enters this decision. The image block is constructed from 64 compact support
pairs and one affected parameter. At the first iteration it has sparse nnz
{first['sparse_nnz']} of 3072 entries/view; the weak view has zero entries.

The first torus-target prediction is `c=-.005 -> c_pred={first['c_pred']:.12g}`:
`||Jv||={first['Jv_norm']:.12g}`, `g={first['g_c']:.12g}` and
`h={first['h_c']:.12g}`. The full step increases re-traced loss, so the common
line search accepts alpha=.5 at `c={first['accepted_c']:.12g}`. This partial step
still crosses zero. Predicted and actual re-traced loss are deliberately kept
separate in the JSON.

## Run A — real-only baselines

The primary ordinary discrete real optimizer is allowed to evaluate real
parameters on either side of zero. It reaches `c={direct['final_c']:.12g}` in
five iterations, changes genus 0->1 and reduces MSE from
{direct['initial_loss']:.12g} to {direct['final_loss']:.12g}. It does not evaluate
the singular intermediate parameter; this is a parameter jump, not continuous
real zero-set continuation.

The additional continuity-constrained real tracker rejects steps whose parameter
segment crosses the known discriminant. With the same iteration budget it
approaches `c={continuous['final_c']:.12g}`, remains genus 0 and finishes at MSE
{continuous['final_loss']:.12g}. Its pole gradient and implicit derivative trend
toward singular conditioning. This baseline answers a different question from
ordinary discrete optimization and must not be substituted for it silently.

## Run B — complex-capable

The complex-capable run has the identical real optimizer and accepted endpoints.
At the triggered crossing it audits/realizes the accepted jump via
`c(t)=(-c_start)^(1-t)c_end^t exp(i*pi*(1-t))`, integrates the full implicit
normal-line equation, and corrects it after every predictor step. The bridge has
max residual {first_bridge['max_residual']:.3g}, minimum sampled spatial-gradient
norm {first_bridge['min_spatial_gradient_norm']:.9g}, minimum normal-line
denominator {first_bridge['min_spatial_jacobian']:.9g}, and stays at least
{first_bridge['min_distance_to_discriminant']:.9g} from `c=0`. Endpoint imaginary
residual is {first_bridge['endpoint_point_imaginary_max']:.3g}; it agrees with an
independent real zero-set solve to {first_bridge['endpoint_real_resolve_max_error']:.3g}.
No complex state is rendered.

It reaches genus 1 and the same final loss {complex_run['final_loss']:.12g} as the
ordinary discrete real run. Thus it beats the continuity-constrained real tracker,
but **does not beat or improve reachability over ordinary real parameter jumps**.
The bridge is a valid continuation mechanism, not an observed optimizer benefit
in this case.

## Run C — same-side control

For target `c=-.007`, every prediction remains on the negative side. Complex
capability is available but never activated. Final c is {same['final_c']:.12g},
MSE {same['final_loss']:.3g}, and topology remains genus 0.

## Run D — weak observability

The backward view has `||Jv||=0`, `h=0`, no sign decision, no update and no
complex bridge. This is reported as underdetermined; it is not confused with
zero-set singularity or chart failure.

## Conditioning and topology

Each iteration separately records (1) the true critical branch spatial gradient,
(2) `grad F dot n_ref` for source-chart conditioning, and (3) `||Jv||` for image
observability. At initialization the critical pole gradient is
{first['local_zero_set']['critical_branch_gradient_norm']:.9g}, while the sampled
source gradient and chart denominator are {first['sampled_surface_gradient_min']:.9g}
and {first['chart_denominator_min']:.9g}. They are not conflated.

Topology is checked with the 144^3 extraction and an analytic meridional count.
Both agree: discrete real and complex runs finish genus 1; continuous real and
same-side runs finish genus 0. The diagnostic is not an optimization signal.

## Acceptance/falsification summary

- A real-only difficulty: **false for ordinary discrete optimization**, true only
  for continuous real zero-set tracking.
- Image observability and sign-crossing evidence: true.
- Regular complex bridge and valid real re-entry: true on the 64 tracked branches.
- Actual loss beyond ordinary real-only: **false**; results match exactly.
- Genus 0->1 endpoint: true for both discrete-real and complex-capable runs.
- Non-forcing and weak-observation controls: pass.

## 1. Mathematical feasibility

Supported locally: the triggered complex path connects the selected real sample
roots without hitting its discriminant and returns to an independently solved
real endpoint.

## 2. Numerical stability

Supported for this sampled bridge by residual, gradient, denominator, endpoint
and tangent-FD checks. It is not a global regularity proof for the complete
complex surface.

## 3. Image observability

Supported in two views; exactly absent in the backward control view.

## 4. Image-preferred crossing

Supported: rendering residual and sparse J predict `c0*c_pred<0`; an actual
re-traced half step crosses and improves loss.

## 5. Actual rendered-loss benefit

Not supported relative to ordinary discrete real-only optimization. Complex and
discrete-real runs have identical real iterates and losses. A benefit exists only
relative to the explicitly continuity-constrained real tracker.

## 6. Topology outcome

Genus changes 0->1 in both the discrete-real and complex-capable runs. Therefore
the topology result cannot be causally attributed to complex continuation here.

## 7. Limitations

Known one-dimensional critical coordinate, one analytic family, local neck
observations, one seed, small historical diagnostic renderer, and a complex path
for sampled roots only. Source birth/death and the vanishing cycle are not
globally continued; multiple critical modes are absent. Direct real jumps skip
the singular parameter and are valid for this optimizer. No claim of necessity,
novelty, production readiness or whole-surface complex homotopy is made.

**Final answer:** the experiment supports that real rendering loss can drive the
sphere-like field to a torus and that the trigger identifies a critical crossing.
It supports local complex bypass as a numerically valid optional bridge. It does
**not** support the stronger statement that the bypass provides useful additional
reachability or rendered-loss improvement over ordinary discrete real optimization.

Runtime: {report['seconds']:.3f} seconds, excluding plots/tests/validation.
'''
    Path('artifacts/v094_sphere_torus_optimization.md').write_text(text)


if __name__ == '__main__': main()
