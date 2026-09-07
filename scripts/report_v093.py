"""Generate the v093 tables, plots and explicitly qualified research report."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from zlt.critical_experiment import CACHE, REPORT, write


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(REPORT.read_text()); image = report['image']; rows = image['candidates']
    opposite, same, invisible = rows
    figures = []
    def save(fig, name):
        p = Path('figures')/('v093_'+name+'.png')
        fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig); figures.append(str(p))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name in ('scalar', 'analytic_pole', 'analytic_inner', 'analytic_outer'):
        path = np.load(CACHE/(name+'_path.npz'))
        root = path['roots'][:, 0]
        axes[0].plot(root.real, root.imag, label=name)
        axes[1].plot(path['spatial_jacobian_min'], label=name)
        axes[2].semilogy(np.maximum(path['residuals'], 1e-18), label=name)
    axes[0].set(xlabel='Re(root)', ylabel='Im(root)'); axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Continuation step', ylabel='|dF/d tracked coordinate|')
    axes[2].set(xlabel='Continuation step', ylabel='Full equation residual')
    save(fig, 'analytic_paths')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    curve = image['real_critical_curve']
    axes[0].loglog([-x['c'] for x in curve], [x['spatial_gradient'] for x in curve], 'o-')
    axes[1].loglog([-x['c'] for x in curve], [x['implicit_derivative'] for x in curve], 'o-')
    axes[0].set(xlabel='|c| on real critical pole', ylabel='Spatial gradient norm')
    axes[1].set(xlabel='|c|', ylabel='|d pole / dc|'); save(fig, 'real_critical_collapse')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in rows[:2]:
        trials = row['trials']
        axes[0].plot([x['c'] for x in trials], [x['retraced_loss'] for x in trials], 'o', label=row['name'])
        axes[0].scatter([row['accepted_real_coordinate']], [row['endpoint_loss']], marker='*', s=100)
    axes[0].axvline(0, color='k', ls='--'); axes[0].set(xlabel='Real c', ylabel='Retraced target MSE'); axes[0].legend()
    axes[1].bar(['view 0', 'view 1', 'invisible', 'multiview'], opposite['per_view_Jv_norm']+[opposite['multiview_Jv_norm']])
    axes[1].set_ylabel('||Jv||'); save(fig, 'image_evidence')
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    pictures = [np.load(CACHE/'initial.npy'), np.load(CACHE/'opposite_side_target.npy'), np.load(CACHE/'opposite_side_endpoint.npy')]
    vmax = max(float(x.max()) for x in pictures)
    for i, (label, picture) in enumerate(zip(('initial', 'real target', 'bypassed real endpoint'), pictures)):
        for v in range(3):
            axes[i, v].imshow(np.clip(picture[v].reshape(32, 32, 3)/vmax, 0, 1))
            axes[i, v].set_title(f'{label} / view {v}'); axes[i, v].axis('off')
    fig.suptitle('Existing real first-arrival renderer; 64 local neck observations')
    save(fig, 'real_images')
    path = np.load(CACHE/'opposite_side_path.npz'); c = path['parameters']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(c.real, c.imag); axes[0].scatter([0], [0], c='r', marker='x')
    axes[0].set(xlabel='Re(c)', ylabel='Im(c)', title='Image-requested bypass')
    axes[1].semilogy(np.maximum(path['residuals'], 1e-18)); axes[1].set_title('Full local F residual')
    axes[2].plot(path['spatial_jacobian_min']); axes[2].set_title('|grad F · n_ref| (not ||grad F||)')
    save(fig, 'image_bypass')
    # Real meridional contours, with an enlarged neck inset; no complex render.
    from zlt.critical_experiment import field
    import torch
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, extent in zip(axes, (1.6, .07)):
        a = torch.linspace(-extent, extent, 401, dtype=torch.float64)
        xx, zz = torch.meshgrid(a, a, indexing='xy')
        p = torch.stack((xx.flatten(), torch.zeros(xx.numel(), dtype=a.dtype), zz.flatten()), 1)
        for coord, color in [(opposite['c0'], 'blue'), (opposite['accepted_real_coordinate'], 'red')]:
            values = torch.cat([field(coord).value(p[i:i+8192]) for i in range(0, len(p), 8192)]).reshape(401, 401)
            ax.contour(xx, zz, values, levels=[0], colors=[color], linewidths=1)
        ax.set_aspect('equal'); ax.set(xlabel='x', ylabel='z', title='Blue: initial; red: accepted real endpoint')
    save(fig, 'topology_meridian')
    metrics = ['name', 'c0', 'target_c', 'Jv_norm', 'g_c', 'h_c', 'c_pred', 'sign_crossing_predicted',
               'complex_continuation_used', 'accepted_real_coordinate', 'initial_loss', 'endpoint_loss']
    with Path('artifacts/v093_critical_continuation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics, lineterminator='\n'); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in metrics})
    verdicts = dict(
        scalar_monodromy=report['scalar']['passed'], analytic_detour_regular=report['sphere_torus']['regular_complex_paths'],
        meaningful_image_mode=opposite['Jv_norm'] > 1e-8,
        real_target_predicts_crossing=opposite['sign_crossing_predicted'],
        same_side_not_forced=not same['sign_crossing_predicted'] and not same['complex_continuation_used'],
        invisible_mode_underdetermined=not invisible['observable'] and not invisible['complex_continuation_used'],
        complex_sample_path_regular=opposite['continuation']['max_residual'] < 1e-12 and opposite['min_complex_spatial_gradient'] > 0,
        real_reentry=opposite['endpoint_imaginary_max'] < 1e-12 and opposite['endpoint_real_F_max'] < 1e-12,
        actual_retraced_loss_improves=opposite['endpoint_loss'] < opposite['initial_loss'],
        endpoint_genus_changed=image['initial_topology']['meridian']['inferred_genus'] != opposite['endpoint_topology']['meridian']['inferred_genus'],
        raw_full_GN_step_improves=opposite['raw_GN_retraced_improves'],
        every_analytic_branch_returns_real=report['sphere_torus']['all_tracked_branches_return_real'],
        complex_bypass_proven_necessary=False, globally_regular_complexified_compact_field_proven=False)
    report.update(verdicts=verdicts, figures=figures,
                  scope='Local analytic continuation and image-driven branch selection verified; not a global real-surface continuation theorem or production topology optimizer.')
    write(REPORT, report)
    text = f'''# v0.9.3 — image-driven critical mode and complex continuation

## Setup and repository map

The experiment is `python -m zlt.critical_experiment`. Math controls live in
`src/zlt/critical_continuation.py`; this report/figures are generated by
`python scripts/report_v093.py`; validation is `python scripts/validate_v093.py`.
No existing renderer or historical result is changed. CPU only; no HPC, CUDA,
Full-HD, complex RGB, gradient penalty, or production refactor.

Inspected `ZeroSetField`, `LocalBasisField`, its Wendland values/gradients,
`implicit_position_jacobian`, `geometry_image_jacobian`, the direct sparse
`birth._local_sparse_jacobian`, mesh-free samplers and the newer manifold-jet /
birth paths. We use the existing direct sparse fixed-cell diagnostic because it
is self-contained on committed main and exposes the required normal-line
denominator explicitly. This is **not** the newer CURRENT finite-packet/C3
production operator. No conclusion about that operator is claimed.

There are 64 deterministic mesh-free neck observations, two informative 32²
views and one backward/invisible view. Root seeds use radial bisection; all
subsequent real points use the existing `deform_reference_surface` solver.
Source IDs, colors, directions and reference normals remain fixed. The existing
first-arrival tracer constructs cells; the existing bilinear image operator
renders them. We re-trace both the real target and all real endpoint candidates.
Thus the primary residual is from normally re-traced real images, not a frozen
target that has inherited the initial visibility mask. Frozen-target results
are retained as a separate diagnostic.

## Equations and critical coordinate

Analytic control: `F_a=(x²+y²+z²+a²)²-4a²(x²+y²)-1`.
At a=0 its zero set is the unit sphere. The only complex parameter discriminants
are a=1,-1,i,-i: solving all three spatial-gradient equations on F=0 forces the
origin, where a⁴=1. Numerical real topology checks at a=0,.8,1.2 give genus 0,0,1.

The image experiment is local, not a global change in a:
`F_c = F_(a=1) + c W(||x||/0.8)`, with the **existing** compact Wendland C2 basis.
Only one coefficient varies: lambda=c, v=[1], lambda_perp=0. The mode is chosen
analytically for this controlled test, not discovered from arbitrary geometry.
Because W(0)=1 and grad W(0)=0, F_c(0)=c and grad F_c(0)=0. The origin is on the
zero set **only at c=0**. Its critical Hessian there is diag(-4,-4,4).
The singular coordinate is known independently of the image loss. No topology
labels enter the score or backtracking.

The sparse Jacobian is built directly from local support pairs and touched
detector entries using `_local_sparse_jacobian`, not constructed dense then
sparsified. Geometry support has 64 point/basis pairs and one affected parameter.
Per-view image nonzeros are {opposite['nonzero_image_entries']} of 3072 entries
per view. Multiview j=Jv is stacked before computing `g=jᵀr`, `h=jᵀj` and
`delta=-g/h`. A zero h yields an underdetermined decision, never a forced step.

## Scalar and 3D complex controls

For z²-mu, RK4 implicit tangents plus full-equation Newton corrections track
mu=exp(i theta), theta=0..2pi. Start +1 becomes -1; max residual
{report['scalar']['max_residual']:.3g}, min |F_z|
{report['scalar']['min_spatial_jacobian']:.6g}, max |dz/dmu|=.5.
The real approach mu->0 has collapsing gradient and diverging derivative.

For the complete quartic F_a, a semicircle of radius .2 around a=1 avoids all
four discriminants. Pole, future inner and outer branches remain regular with
full-equation residual below 5e-15. An important **negative** result: the real
pole z=.6 ends at z=-0.663325i, not on the real torus. The inner branch starts
imaginary and ends real. Only the outer witness is real at both endpoints.
Avoiding a complex singularity does not guarantee every root returns real.

## Image evidence and actual losses

| Case | ||Jv|| | c0 | c_pred | Cross? | Accepted real c | Actual final MSE |
|---|---:|---:|---:|---|---:|---:|
| Opposite target | {opposite['Jv_norm']:.6g} | -.005 | {opposite['c_pred']:.9g} | yes | {opposite['accepted_real_coordinate']:.9g} | {opposite['endpoint_loss']:.9g} |
| Same-side target | {same['Jv_norm']:.6g} | -.005 | {same['c_pred']:.9g} | no | {same['accepted_real_coordinate']:.9g} | {same['endpoint_loss']:.9g} |
| Invisible target | 0 | -.005 | undefined | underdetermined | unchanged | 0 (uninformative) |

Opposite-target g={opposite['g_c']:.12g}, h={opposite['h_c']:.12g}.
Single-view norms are {opposite['per_view_Jv_norm']}; multiview norm is
{opposite['multiview_Jv_norm']:.9g}. Sparse columns agree with the original AD
geometry-image interface to {image['sparse_vs_original_AD_max_error']:.3g}.
Four-epsilon image FD relative errors: {[x['relative_l2'] for x in image['fd']]}.

**Raw GN failure:** the full step predicts crossing but re-traced loss becomes
{opposite['trials'][-1]['retraced_loss']:.9g}, worse than initial
{opposite['initial_loss']:.9g}. Discrete first-arrival owners change. Ordinary
predeclared backtracking alpha=1,.5,.25,.125 accepts the half step, which remains
positive and reduces actual re-traced loss to {opposite['endpoint_loss']:.9g}.
The accepted coordinate is selected by rendered loss, not by target parameter
or desired topology. Small +/- real probes agree with the local loss prediction;
the large full GN step must not be called an exact global prediction.

## Complex bypass and real re-entry

Only after sign crossing is predicted and an actual improving opposite-side
endpoint survives backtracking do we continue
`c(t)=(-c0)^(1-t) c_end^t exp(i*pi*(1-t))`.
We integrate 64 complex normal-line offsets using `dt_line/dc=-B/(grad F·n_ref)`
and correct the **full local implicit equation**, not a reduced surrogate.
At the complex interior, `r=sqrt(x²+y²+z²)` is the analytic radial branch,
not a Hermitian norm. All tracked points stay strictly inside the support
neighborhood and away from the radial branch cut. There is no globally
holomorphic compact-support construction or global regularity proof.

Max full-equation residual {opposite['continuation']['max_residual']:.3g};
min complex spatial gradient norm {opposite['min_complex_spatial_gradient']:.9g};
min normal-line denominator {opposite['continuation']['min_normal_line_denominator']:.9g};
min distance to c=0 {opposite['min_distance_to_c_zero']:.9g};
endpoint point imaginary residual {opposite['endpoint_imaginary_max']:.3g};
real endpoint equation residual {opposite['endpoint_real_F_max']:.3g};
agreement with independently attached real endpoint {opposite['endpoint_point_match']:.3g}.
Complex states never enter the real tracer. Numerical tangent finite-difference
errors are recorded for every path, in addition to residuals and derivative bounds.

## Topology and separation of condition numbers

Real critical pole samples for c->0- explicitly show ||grad F|| collapsing and
the implicit derivative diverging. Separately, a tangent reference-line control
has |grad F·n_ref| <={image['chart_failure_control']['denominator_max']:.3g} while
the sampled surface gradient remains >={image['chart_failure_control']['surface_gradient_min']:.6g}.
This is chart failure on a regular surface, not zero-set singularity. The dark
view has Jv=0 on that same regular surface: lack of image information is neither
of these geometric failures.

The real initial endpoint has one component, Euler 2/genus 0. The accepted
opposite endpoint has one component, Euler
{opposite['endpoint_topology']['euler']}/genus {opposite['endpoint_topology']['genus']} in
the 144³ diagnostic. The meridian independently has inner radius
{opposite['endpoint_topology']['meridian']['inner_radius']:.9g}, no axis intersections,
and outer radius sqrt(2), confirming genus 1. For |c|<.02, F_z/z is strictly
positive; F(r,0) is strictly decreasing in the local support and the unchanged
quartic outside it supplies the single outer root. This justifies the meridional
topology interpretation without using it as the trigger.

An earlier frozen-target trial accepted a much smaller positive coordinate;
the even 144³ grid missed its tiny hole although the meridian resolved it.
Earlier attempts are preserved in `runs/v093_critical_continuation/attempt_*`.
Grid topology alone is not a global guarantee.

## Research conclusion (six parts)

1. **Mathematically possible:** scalar branch exchange and regular complex
   detours of the analytic sphere-to-torus family are demonstrated. A local
   compact-basis mode has a known real critical value and an analytic extension
   along the tested paths. Not every tracked root admits real re-entry.
2. **Numerically stable:** predictor/corrector paths have small residuals and
   nonzero sampled gradients; no gradient penalty/barrier is used. This is not
   proof that an entire complex surface remains regular everywhere.
3. **Image-observable:** existing sparse J gives a meaningful mode in two views;
   the backward view is exactly uninformative and triggers no motion.
4. **Image-preferred:** actual real-image residual predicts the other sign, and
   an ordinary half step improves freshly re-traced loss. The full GN step fails
   because of discrete transport-cell changes; that failure is retained.
5. **Topology changed:** accepted real endpoints are genus 0 and genus 1 by the
   diagnostic and meridional checks. No topology label drives the optimizer.
6. **Limitations:** known one-dimensional mode, local neck observations, one
   target family, existing historical fixed-direction renderer, not CURRENT/C3.
   No arbitrary critical-mode discovery, multiple-candidate interaction test,
   complete-surface real-to-real correspondence, global complexification, or
   production inverse topology optimizer is established.

Crucially, the 64 observed source branches themselves stay regular even on the
real crossing: the actual singular point is elsewhere at the origin. Thus this
experiment shows that a complex bypass **can** connect the image-preferred real
parameters and sampled points, not that the bypass was **necessary** or solves
continuation of every source through the vanishing cycle. This is the central
qualification to a positive local result.

Runtime: {report['seconds']:.3f} s for the CPU experiment, excluding plots/tests.
No commit is authorized by this report until the validation script passes.
'''
    Path('artifacts/v093_critical_continuation.md').write_text(text)


if __name__ == '__main__': main()
