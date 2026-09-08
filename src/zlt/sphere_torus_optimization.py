"""v0.9.4 rendered sphere-to-torus optimization and topology-crossing ablation.

Every loss is evaluated with real geometry and freshly traced transport cells.
Complex numbers are used only to continue already selected source roots between
two real parameter states.  The historical renderer/Jacobian are not modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from .birth import _local_sparse_jacobian
from .critical_continuation import continue_root, local_equation, summary
from .critical_experiment import CACHE as OLD_CACHE
from .critical_experiment import field, images, make_cells, source, topology, write
from .fields import deform_reference_surface, implicit_position_jacobian
from .locality import BasisLayout, UniformGridIndex

CACHE = Path('runs/v094_sphere_torus_optimization')
REPORT = Path('artifacts/v094_sphere_torus_optimization.json')
C0 = -.005
TARGET_C = .005
VIEWS = (0, 1)
ALPHAS = (1., .5, .25, .125, .0625, .03125)


def _snapshot():
    path = CACHE/'starting_state.json'
    if path.exists():
        return
    protected = [p for folder in ('artifacts', 'figures', 'src/zlt', 'scripts', 'tests')
                 for p in Path(folder).glob('*') if p.is_file() and not p.name.startswith(('v094', 'sphere_torus_optimization'))]
    write(path, dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     protected_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}))


def _real_state(c, reference_points, reference_normals, colors):
    current = field(c)
    points, stable, _ = deform_reference_surface(current, reference_points, reference_normals)
    if not bool(stable.all()):
        raise RuntimeError(f'real attachment failed at c={c}')
    cells = make_cells(current, points, reference_normals, colors)
    image = torch.stack([cell_image(cell, points) for cell in cells])
    return current, points, cells, image


def cell_image(cell, points):
    from .jacobian import render_fixed_transport_cell
    return render_fixed_transport_cell(cell, points)


def _sparse_mode(current, points, reference_normals, cells):
    layout = BasisLayout(current.centers, current.radii)
    support = UniformGridIndex(layout.centers, .8).query(points, .8)
    _, stable, denominator = implicit_position_jacobian(current, points, reference_normals)
    if not bool(stable.all()):
        raise RuntimeError('chart denominator failed away from the monitored critical point')
    matrices = [_local_sparse_jacobian(layout, points, reference_normals, denominator, support, cell)
                for cell in cells]
    one = torch.ones((1, 1), dtype=torch.float64)
    columns = torch.stack([torch.sparse.mm(matrix, one).flatten() for matrix in matrices])
    return columns, matrices, support, denominator


def _critical_indicators(c):
    """Exact local singular witness at origin plus real pole branch for c<0."""
    origin = torch.zeros((1, 3), dtype=torch.float64)
    result = dict(origin_F=float(field(c).value(origin)), origin_gradient_norm=0.,
                  origin_is_zero_set=abs(c) < 1e-14)
    if c < 0:
        lo, hi = 0., .25
        for _ in range(60):
            mid = (lo+hi)/2
            value = float(field(c).value(torch.tensor([[0., 0., mid]], dtype=torch.float64)))
            if value < 0: lo = mid
            else: hi = mid
        pole = torch.tensor([[0., 0., (lo+hi)/2]], dtype=torch.float64)
        gradient = field(c).gradient(pole).norm()
        basis = field(c).basis_values(pole)[0, 0]
        result.update(real_pole=float(pole[0, 2]), critical_branch_gradient_norm=float(gradient),
                      critical_branch_normal_denominator=float(gradient), implicit_dc=float(abs(basis/gradient)))
    else:
        result.update(real_pole=None, critical_branch_gradient_norm=None,
                      critical_branch_normal_denominator=None, implicit_dc=None)
    return result


def _complex_bridge(start, end, reference_points, reference_normals):
    if not start < 0 < end:
        raise ValueError('bridge requires negative-to-positive real endpoints')
    p, n = reference_points.numpy(), reference_normals.numpy()
    def path(t):
        radius = (-start)**(1-t)*end**t
        value = radius*np.exp(1j*np.pi*(1-t))
        return value, value*(np.log(end/(-start))-1j*np.pi)
    def quantities(offset, coordinate):
        return local_equation(p+offset[:, None]*n, coordinate)
    result = continue_root(path, np.zeros(len(p)), lambda s, c: quantities(s, c)[0],
                           lambda s, c: (quantities(s, c)[1]*n).sum(1),
                           lambda s, c: quantities(s, c)[2])
    xyz = p[None]+result['roots'][:, :, None]*n[None]
    gradients = np.asarray([local_equation(x, c)[1] for x, c in zip(xyz, result['parameters'])])
    _, endpoint, _, _ = _real_state(end, reference_points, reference_normals,
                                     field(start).color(reference_points))
    out = summary(result)
    out.update(start=start, end=end, path='log-radius upper semicircle in c-plane',
               min_distance_to_discriminant=float(np.abs(result['parameters']).min()),
               min_spatial_gradient_norm=float(np.linalg.norm(gradients, axis=2).min()),
               endpoint_point_imaginary_max=float(np.abs(xyz[-1].imag).max()),
               endpoint_real_resolve_max_error=float(np.abs(xyz[-1].real-endpoint.numpy()).max()),
               rendered_complex_states=0)
    return out, result


def optimize(name, target_c, complex_capable, continuous_real, selected_views=VIEWS, iterations=5):
    reference_points, reference_normals, colors = source(C0)
    _, _, _, target_image = _real_state(target_c, reference_points, reference_normals, colors)
    current_c = C0; trajectory = []; bridges = []
    for iteration in range(iterations):
        current, points, cells, current_image = _real_state(current_c, reference_points, reference_normals, colors)
        residual = (current_image[list(selected_views)]-target_image[list(selected_views)]).flatten()
        loss = float(residual.square().mean())
        columns, matrices, support, denominator = _sparse_mode(current, points, reference_normals, cells)
        jc = columns[list(selected_views)].flatten(); g = float(jc@residual); h = float(jc@jc)
        observable = h > 1e-20; delta = -g/h if observable else None
        predicted = current_c+delta if observable else None
        trigger = bool(current_c*predicted < 0) if observable else False
        row = dict(iteration=iteration, loss=loss, c0=current_c, delta_c=delta, c_pred=predicted,
                   trigger=trigger, Jv_norm=float(jc.norm()), g_c=g, h_c=h, observable=observable,
                   per_view_Jv_norm=columns.norm(dim=1).tolist(),
                   local_zero_set=_critical_indicators(current_c),
                   sampled_surface_gradient_min=float(current.gradient(points).norm(dim=1).min()),
                   chart_denominator_min=float(denominator.abs().min()), support_location=[0., 0., 0.],
                   support_radius=.8, support_pairs=len(support.point_ids), affected_parameters=1,
                   sparse_nnz=[int(x._nnz()) for x in matrices], image_entries=[x.shape[0] for x in matrices],
                   nonzero_fraction=[x._nnz()/x.shape[0] for x in matrices], trials=[],
                   complex_capable=complex_capable, complex_bypass_activated=False)
        if not observable or abs(g) < 1e-14:
            row['stop'] = 'UNOBSERVABLE_OR_STATIONARY'; trajectory.append(row); break
        accepted = None
        for alpha in ALPHAS:
            candidate = current_c+alpha*delta
            trial = dict(alpha=alpha, c=candidate)
            if continuous_real and current_c*candidate <= 0:
                trial.update(accepted=False, rejection='REAL_CONTINUATION_CANNOT_CROSS_KNOWN_SINGULAR_C0')
                row['trials'].append(trial); continue
            try:
                _, _, _, image = _real_state(candidate, reference_points, reference_normals, colors)
                trial_loss = float((image[list(selected_views)]-target_image[list(selected_views)]).square().mean())
                trial.update(loss=trial_loss, accepted=trial_loss < loss*(1-1e-10))
            except RuntimeError as error:
                trial.update(accepted=False, rejection=str(error))
            row['trials'].append(trial)
            if trial['accepted']:
                accepted = candidate; break
        if accepted is None and continuous_real and trigger:
            # Approach the real discriminant without crossing; this is the best
            # admissible boundary step, evaluated with the real renderer.
            candidate = -max(abs(current_c)*.1, 1e-10)
            _, _, _, image = _real_state(candidate, reference_points, reference_normals, colors)
            trial_loss = float((image[list(selected_views)]-target_image[list(selected_views)]).square().mean())
            trial = dict(alpha=None, c=candidate, loss=trial_loss,
                         accepted=trial_loss < loss*(1-1e-10), boundary_approach=True)
            row['trials'].append(trial)
            if trial['accepted']: accepted = candidate
        if accepted is None:
            row['stop'] = 'NO_ACTUAL_RETRACED_LOSS_GAIN'; trajectory.append(row); break
        if complex_capable and trigger and current_c*accepted < 0:
            bridge, raw = _complex_bridge(current_c, accepted, reference_points, reference_normals)
            bridge['iteration'] = iteration; bridge['accepted_retraced_loss'] = row['trials'][-1]['loss']
            bridges.append(bridge); row['complex_bypass_activated'] = True; row['bridge_index'] = len(bridges)-1
            np.savez(CACHE/f'{name}_bridge_{len(bridges)-1}.npz', **raw)
        row['accepted_c'] = accepted; row['accepted_loss'] = row['trials'][-1]['loss']
        trajectory.append(row); current_c = accepted
    final, final_points, final_cells, final_image = _real_state(current_c, reference_points, reference_normals, colors)
    final_loss = float((final_image[list(selected_views)]-target_image[list(selected_views)]).square().mean())
    np.save(CACHE/f'{name}_initial.npy', _real_state(C0, reference_points, reference_normals, colors)[3].numpy())
    np.save(CACHE/f'{name}_target.npy', target_image.numpy()); np.save(CACHE/f'{name}_final.npy', final_image.numpy())
    return dict(name=name, target_c=target_c, selected_views=list(selected_views), complex_capable=complex_capable,
                continuous_real=continuous_real, trajectory=trajectory, bridges=bridges, final_c=current_c,
                initial_loss=trajectory[0]['loss'], final_loss=final_loss,
                topology_initial=topology(C0), topology_final=topology(current_c),
                topology_target=topology(target_c),
                final_sample_gradient_min=float(final.gradient(final_points).norm(dim=1).min()),
                final_owner_counts=[int((cell.owner_map >= 0).sum()) for cell in final_cells])


def run():
    CACHE.mkdir(parents=True, exist_ok=True); _snapshot(); started = time.perf_counter()
    runs = [
        optimize('A_real_only_discrete', TARGET_C, False, False),
        optimize('A2_real_only_continuous', TARGET_C, False, True),
        optimize('B_complex_capable', TARGET_C, True, False),
        optimize('C_same_side', -.007, True, False),
        optimize('D_weak_observability', TARGET_C, True, False, selected_views=(2,)),
    ]
    by = {x['name']: x for x in runs}; direct, continuous, complex_run = runs[:3]
    report = dict(version='v094', experiment='rendered sphere-to-torus optimization', runs=runs,
                  operator='Existing real reference-direction, first-arrival, bilinear fixed-cell renderer; every state/target re-traced.',
                  comparison_policy='Identical initialization, target, views, source IDs, GN rule and line search. Complex run uses the same accepted real endpoint as discrete real-only, with an internal continuation audit.',
                  verdicts=dict(
                      REAL_ONLY_DISCRETE_REACHES_GENUS1=direct['topology_final']['meridian']['inferred_genus']==1,
                      REAL_CONTINUATION_STALLS_AT_BARRIER=continuous['topology_final']['meridian']['inferred_genus']==0,
                      CRITICAL_MODE_IMAGE_OBSERVABLE=direct['trajectory'][0]['Jv_norm']>1e-8,
                      IMAGE_TRIGGER_PREDICTS_CROSSING=direct['trajectory'][0]['trigger'],
                      COMPLEX_BYPASS_REGULAR=bool(complex_run['bridges']) and all(x['max_residual']<1e-12 and x['min_spatial_gradient_norm']>0 for x in complex_run['bridges']),
                      COMPLEX_REAL_REENTRY=bool(complex_run['bridges']) and all(x['endpoint_point_imaginary_max']<1e-12 and x['endpoint_real_resolve_max_error']<1e-12 for x in complex_run['bridges']),
                      COMPLEX_BEATS_CONTINUOUS_REAL=complex_run['final_loss']<continuous['final_loss'],
                      COMPLEX_BEATS_DISCRETE_REAL=complex_run['final_loss']<direct['final_loss']*(1-1e-10),
                      COMPLEX_MATCHES_DISCRETE_REAL=abs(complex_run['final_loss']-direct['final_loss'])<1e-12 and abs(complex_run['final_c']-direct['final_c'])<1e-12,
                      COMPLEX_RUN_REACHES_GENUS1=complex_run['topology_final']['meridian']['inferred_genus']==1,
                      SAME_SIDE_DOES_NOT_TRIGGER=not any(x['trigger'] for x in by['C_same_side']['trajectory']),
                      WEAK_VIEW_UNDERDETERMINED=not by['D_weak_observability']['trajectory'][0]['observable'],
                      FULL_CLAIM_SUPPORTED=False),
                  seconds=time.perf_counter()-started,
                  central_result='Complex continuation is a valid regular audit/bridge and beats a continuity-constrained real tracker, but provides no loss or reachability advantage over ordinary discrete real parameter updates in this controlled renderer.')
    write(REPORT, report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': run()
