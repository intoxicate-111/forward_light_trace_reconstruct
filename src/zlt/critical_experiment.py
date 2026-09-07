"""v093: local critical-mode image evidence and complex continuation.

Uses the existing real fixed-transport-cell diagnostic and its sparse Jacobian.
Also re-traces real endpoints to expose visibility/ownership cell changes.
No changes to historical operators; no complex rendering or optimization.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .fields import LocalBasisField, unit_normals, deform_reference_surface, implicit_position_jacobian
from .locality import BasisLayout, UniformGridIndex
from .camera import PlanarCamera
from .jacobian import build_fixed_transport_cell, render_fixed_transport_cell, geometry_image_jacobian
from .birth import _local_sparse_jacobian
from .critical_continuation import scalar_control, sphere_torus_control, continue_root, summary, local_equation

CACHE = Path('runs/v093_critical_continuation')
REPORT = Path('artifacts/v093_critical_continuation.json')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


class CriticalBase:
    """Analytic a=1 family; critical origin, with no sphere sampling assumption."""
    def __init__(self, a=1.):
        self.a = a

    def value(self, p):
        s = (p*p).sum(-1)
        return (s+self.a**2)**2-4*self.a**2*(p[..., :2]**2).sum(-1)-1

    def gradient(self, p):
        s = (p*p).sum(-1)
        scale = torch.ones_like(p)*4*(s+self.a**2)[..., None]
        scale[..., :2] -= 8*self.a**2
        return p*scale

    def color(self, p):
        return (.5+.3*p).clamp(0, 1)


def field(c):
    return LocalBasisField(CriticalBase(), torch.zeros((1, 3), dtype=torch.float64),
                           torch.tensor([.8], dtype=torch.float64), torch.tensor([c], dtype=torch.float64))


def source(c, count=64):
    """Controlled local surface observations, not a full-surface area estimator.

    Mesh-free bisection of the neck at persistent z/azimuth seeds. Subsequent
    motion uses the existing normal-line solver with unchanged source IDs.
    """
    seq = torch.quasirandom.SobolEngine(2, scramble=True, seed=193).draw(count, dtype=torch.float64)
    z = .08+.22*seq[:, 0]; phi = 2*torch.pi*seq[:, 1]
    lo = torch.zeros_like(z); hi = torch.full_like(z, .5)
    def points(r):
        return torch.stack((r*phi.cos(), r*phi.sin(), z), 1)
    f = field(c)
    if not bool((f.value(points(lo)) > 0).all() and (f.value(points(hi)) < 0).all()):
        raise RuntimeError('neck source bracket failed')
    for _ in range(55):
        middle = (lo+hi)/2; positive = f.value(points(middle)) > 0
        lo = torch.where(positive, middle, lo); hi = torch.where(positive, hi, middle)
    p = points((lo+hi)/2)
    return p, unit_normals(f, p), f.color(p)


def cameras():
    cams = []
    for center in ([0, 0, 1.2], [.35, 0, 1.2], [0, 0, -1.2]):
        center = torch.tensor(center, dtype=torch.float64)
        normal = -center/center.norm()
        up = center.new_tensor([0., 1, 0]); right = torch.linalg.cross(normal, up)
        cams.append(PlanarCamera(center, normal, right, up, 3., 3., (32, 32)))
    return cams


def make_cells(f, p, normals, colors):
    return [build_fixed_transport_cell(f, camera, p, normals, colors, packets_per_emitter=2,
                                       cone_power=128., normal_mode='reference', root_samples=128) for camera in cameras()]


def images(c, p0, n0, colors, cells, retrace=False):
    f = field(c)
    p, stable, _ = deform_reference_surface(f, p0, n0)
    if not bool(stable.all()):
        raise RuntimeError('normal-line attachment failed')
    use = make_cells(f, p, n0, colors) if retrace else cells
    im = torch.stack([render_fixed_transport_cell(cell, p) for cell in use])
    return im, p, use


def topology(c=None, a=None):
    from skimage.measure import marching_cubes
    import trimesh
    n = 144; axis = torch.linspace(-1.65, 1.65, n, dtype=torch.float64)
    p = torch.cartesian_prod(axis, axis, axis)
    f = field(c) if a is None else CriticalBase(a)
    vals = torch.cat([f.value(p[i:i+32768]) for i in range(0, len(p), 32768)]).numpy().reshape(n, n, n)
    v, faces, _, _ = marching_cubes(vals, 0., spacing=(3.3/(n-1),)*3)
    mesh = trimesh.Trimesh(vertices=v-1.65, faces=faces, process=False)
    parts = mesh.split(only_watertight=False)
    result = dict(c=float(c) if c is not None else None, a=a, grid=n, components=len(parts), watertight=bool(mesh.is_watertight),
                euler=int(mesh.euler_number), genus=(2*len(parts)-int(mesh.euler_number))/2 if mesh.is_watertight else None,
                interpretation='Finite extraction diagnostic only; not a renderer input or topology trigger.')
    if c is not None and abs(c) < .02 and c != 0:
        # Rotational meridian diagnostic with resolved 1D root, independent of
        # Cartesian grid alignment. F increases strictly with z>0; F(r,0) has
        # one positive outer root, plus an inner root iff c>0.
        lo, hi = 0., .8
        for _ in range(60):
            mid = (lo+hi)/2
            p = torch.tensor([[mid, 0., 0.] if c > 0 else [0., 0., mid]], dtype=torch.float64)
            value = float(f.value(p))
            if (value > 0) == (c > 0): lo = mid
            else: hi = mid
        root = (lo+hi)/2
        result['meridian'] = dict(inner_radius=root if c > 0 else 0., pole=root if c < 0 else None,
                                  outer_radius=2**.5, axis_intersections=0 if c > 0 else 2,
                                  inferred_genus=1 if c > 0 else 0,
                                  positive_z_derivative_over_z_lower_bound=4-20*abs(c)/.8**2,
                                  argument='For |c|<.02: F_z/z>0, and F_r(r,0)<0 for 0<r<.8; outside support the unchanged quartic gives the unique outer radius sqrt(2). Rotating the one meridian gives sphere/torus.',
                                  grid_agrees=result['genus'] == (1 if c > 0 else 0))
    return result


def local_real_critical_curve():
    rows = []
    for c in -np.logspace(-3, -12, 10):
        lo, hi = 0., .25
        def values(z):
            q = z/.8; b = (1-q)**4*(4*q+1)
            f = z**4+2*z*z+c*b
            g = 4*z**3+4*z+c*(-20*(1-q)**3*z/.8**2)
            return f, g, b
        for _ in range(60):
            mid = (lo+hi)/2
            if values(mid)[0] < 0: lo = mid
            else: hi = mid
        z = (lo+hi)/2; f, g, b = values(z)
        rows.append(dict(c=float(c), pole=z, equation_residual=abs(f), spatial_gradient=abs(g),
                         normal_line_denominator=abs(g), implicit_derivative=abs(b/g)))
    return rows


def image_experiment():
    c0 = -.005
    p0, n0, colors = source(c0)
    initial = field(c0); cells = make_cells(initial, p0, n0, colors)
    base_image = torch.stack([render_fixed_transport_cell(cell, p0) for cell in cells])
    layout = BasisLayout(initial.centers, initial.radii)
    support = UniformGridIndex(layout.centers, .8).query(p0, .8)
    _, stable, denominator = implicit_position_jacobian(initial, p0, n0)
    matrices = [_local_sparse_jacobian(layout, p0, n0, denominator, support, cell) for cell in cells]
    mode = torch.ones((1, 1), dtype=torch.float64)
    j = torch.stack([torch.sparse.mm(J, mode).flatten() for J in matrices])
    # Check the sparse construction against the original geometry/image AD API.
    ad = geometry_image_jacobian(initial, p0, n0, cells[0]).matrix[:, 0]
    sparse_ad_error = float((j[0]-ad).abs().max())
    fd = []
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        plus, _, _ = images(c0+eps, p0, n0, colors, cells)
        minus, _, _ = images(c0-eps, p0, n0, colors, cells)
        diff = ((plus-minus)/(2*eps)).reshape(3, -1)
        fd.append(dict(epsilon=eps, relative_l2=float((diff-j).norm()/j.norm())))
    rows = []; paths = {}
    for name, target_c, views in [('opposite_side', .005, [0, 1]), ('same_side', -.007, [0, 1]),
                                 ('invisible_opposite', .005, [2])]:
        target, _, _ = images(target_c, p0, n0, colors, cells)
        target_live, _, _ = images(target_c, p0, n0, colors, cells, True)
        # Primary residual uses normally retraced REAL target images. The
        # derivative is the existing sparse tangent at the initial stable cell.
        # Keep frozen-target evidence as a separate diagnostic, not a substitute.
        residual = (base_image[views]-target_live[views]).flatten()
        jc = j[views].flatten(); g = float(jc@residual); h = float(jc@jc)
        meaningful = h > 1e-20
        delta = -g/h if meaningful else None; predicted = c0+delta if meaningful else None
        crossing = bool(c0*predicted < 0) if meaningful else None
        row = dict(name=name, c0=c0, target_c=target_c, views=views, mode=[1.], critical_location=[0., 0., 0.],
                   critical_coordinate='c=lambda[0]=F_c(origin); singular event at c=0',
                   critical_point_F=c0, critical_point_gradient=0.,
                   critical_point_is_on_surface=False,
                   source_gradient_min=float(initial.gradient(p0).norm(dim=1).min()),
                   normal_line_denominator_min=float(denominator.abs().min()),
                   representative_surface_location=p0[0].tolist(),
                   support_center=[0., 0., 0.], support_radius=.8, affected_geometry_parameters=1,
                   geometry_support_pairs=len(support.point_ids), source_count=len(p0),
                   nonzero_image_entries=[int(J._nnz()) for J in matrices], image_entries=[J.shape[0] for J in matrices],
                   image_nonzero_fraction=[J._nnz()/J.shape[0] for J in matrices],
                   Jv_norm=float(jc.norm()), per_view_Jv_norm=j.norm(dim=1).tolist(),
                   multiview_Jv_norm=float(j[:2].norm()), g_c=g, h_c=h, delta_c=delta, c_pred=predicted,
                   sign_crossing_predicted=crossing, observable=meaningful, complex_continuation_used=False,
                   initial_loss=float(residual.square().mean()),
                   initial_retraced_target_loss=float((base_image[views]-target_live[views]).square().mean()), trials=[])
        live_residual = (base_image[views]-target_live[views]).flatten()
        live_g = float(jc@live_residual)
        row['retraced_target_evidence'] = dict(g_c=live_g, h_c=h,
                                              c_pred=c0-live_g/h if meaningful else None,
                                              sign_crossing=bool(c0*(c0-live_g/h) < 0) if meaningful else None)
        frozen_g = float(jc@(base_image[views]-target[views]).flatten())
        row['frozen_target_evidence'] = dict(g_c=frozen_g, c_pred=c0-frozen_g/h if meaningful else None)
        if meaningful:
            for label, c in [('small_minus', c0-1e-5), ('small_plus', c0+1e-5),
                             ('same_side_update', c0*.5), ('near_singularity', -1e-8), ('GN_endpoint', predicted)]:
                im, pos, _ = images(c, p0, n0, colors, cells)
                live, _, live_cells = images(c, p0, n0, colors, cells, True)
                loss = float((im[views]-target_live[views]).square().mean())
                pred_loss = float((residual+(c-c0)*jc).square().mean())
                row['trials'].append(dict(label=label, c=c, loss=loss, linear_prediction_loss=pred_loss,
                                         retraced_loss=float((live[views]-target_live[views]).square().mean()),
                                         min_surface_gradient=float(field(c).gradient(pos).norm(dim=1).min()),
                                         changed_owner_entries=[int((old.owner_map != new.owner_map).sum()) for old, new in zip(cells, live_cells)]))
            # Ordinary real-endpoint backtracking, judged ONLY by freshly
            # retraced loss. Do not tune the transport or force a positive end.
            accepted = c0
            row['retraced_backtracking'] = []
            for alpha in (1., .5, .25, .125):
                candidate = c0+alpha*delta
                live, _, _ = images(candidate, p0, n0, colors, cells, True)
                live_loss = float((live[views]-target_live[views]).square().mean())
                row['retraced_backtracking'].append(dict(alpha=alpha, c=candidate, loss=live_loss))
                if live_loss < row['initial_retraced_target_loss']:
                    accepted = candidate
                    break
            row['accepted_real_coordinate'] = accepted
            row['accepted_retraced_loss'] = row['retraced_backtracking'][-1]['loss'] if accepted != c0 else row['initial_retraced_target_loss']
            row['raw_GN_retraced_improves'] = row['trials'][-1]['retraced_loss'] < row['initial_retraced_target_loss']
            if crossing and c0*accepted < 0:
                # c begins negative, ends at the image-predicted positive value.
                def path(t):
                    radius = (-c0)**(1-t)*accepted**t
                    c = radius*np.exp(1j*np.pi*(1-t))
                    return c, c*(np.log(accepted/(-c0))-1j*np.pi)
                p, n = p0.numpy(), n0.numpy()
                def quantities(offset, c):
                    return local_equation(p+offset[:, None]*n, c)
                result = continue_root(path, np.zeros(len(p)), lambda t, c: quantities(t, c)[0],
                                       lambda t, c: (quantities(t, c)[1]*n).sum(1), lambda t, c: quantities(t, c)[2])
                xyz = p[None]+result['roots'][:, :, None]*n[None]
                grads = np.array([local_equation(x, c)[1] for x, c in zip(xyz, result['parameters'])])
                reentry = xyz[-1]; endpoint, real_points, _ = images(accepted, p0, n0, colors, cells)
                row.update(complex_continuation_used=True, continuation=summary(result),
                           min_complex_spatial_gradient=float(np.linalg.norm(grads, axis=2).min()),
                           min_distance_to_c_zero=float(np.abs(result['parameters']).min()),
                           min_complex_radial_modulus=float(np.abs(np.sqrt((xyz*xyz).sum(2))).min()),
                           max_complex_radial_modulus=float(np.abs(np.sqrt((xyz*xyz).sum(2))).max()),
                           min_real_radial_squared=float(np.real((xyz*xyz).sum(2)).min()),
                           endpoint_imaginary_max=float(np.abs(reentry.imag).max()),
                           endpoint_point_match=float(np.abs(reentry.real-real_points.numpy()).max()),
                           endpoint_real_F_max=float(field(accepted).value(torch.from_numpy(reentry.real)).abs().max()),
                           endpoint_frozen_cell_loss=float((endpoint[views]-target_live[views]).square().mean()),
                           endpoint_loss=row['accepted_retraced_loss'],
                           endpoint_topology=topology(accepted))
                row['continuation']['min_normal_line_denominator'] = row['continuation'].pop('min_spatial_jacobian')
                paths[name] = result
            else:
                endpoint, _, _ = images(accepted, p0, n0, colors, cells)
                row['endpoint_loss'] = row['accepted_retraced_loss']
            row['final_real_side'] = 'positive' if accepted > 0 else 'negative'
        else:
            row.update(final_real_side='negative_unchanged', endpoint_loss=row['initial_loss'],
                       decision='UNDERDETERMINED: image null mode; no step and no complex motion')
        rows.append(row)
        np.save(CACHE/(name+'_target.npy'), target_live.numpy())
    np.save(CACHE/'initial.npy', base_image.numpy())
    for row in rows:
        if row['observable']:
            endpoint, _, _ = images(row['accepted_real_coordinate'], p0, n0, colors, cells, True)
            np.save(CACHE/(row['name']+'_endpoint.npy'), endpoint.numpy())
    # A deliberately tangent reference line has d=0 on a regular surface.
    tangent = torch.linalg.cross(n0, n0.new_tensor([0., 0., 1.]).expand_as(n0))
    tangent /= tangent.norm(dim=1, keepdim=True)
    _, chart_stable, tangent_d = implicit_position_jacobian(initial, p0, tangent)
    return dict(candidates=rows, fd=fd, sparse_vs_original_AD_max_error=sparse_ad_error,
                real_critical_curve=local_real_critical_curve(),
                initial_topology=topology(c0), target_topology=topology(.005),
                chart_failure_control=dict(surface_gradient_min=float(initial.gradient(p0).norm(dim=1).min()),
                                           denominator_max=float(tangent_d.abs().max()), stable_fraction=float(chart_stable.double().mean())),
                operator='Existing reference-direction fixed first-arrival transport cell, bilinear detector, frozen color. Real endpoint retracing reported separately; NOT the v0817 CURRENT soft/C3 operator.',
                source_measure='64 deterministic local neck observations; no complete-surface coverage or surface-area claim.'), paths


def run():
    CACHE.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        index = len(list(CACHE.glob('attempt_*_report.json')))+1
        write(CACHE/f'attempt_{index:02d}_report.json', json.loads(REPORT.read_text()))
    snapshot = CACHE/'starting_state.json'
    if not snapshot.exists():
        protected = [p for folder in ('artifacts', 'figures', 'src/zlt') for p in Path(folder).glob('*')
                     if p.is_file() and not p.name.startswith(('v093', 'critical_'))]
        write(snapshot, dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                             protected_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
                             git_status=subprocess.check_output(['git', 'status', '--porcelain=v1'], text=True)))
    started = time.perf_counter()
    scalar, scalar_path = scalar_control(); analytic, analytic_paths = sphere_torus_control()
    analytic['real_topology'] = [topology(a=a) for a in (0., .8, 1.2)]
    write(CACHE/'analytic_controls.json', dict(scalar=scalar, sphere_torus=analytic))
    images_result, image_paths = image_experiment()
    report = dict(version='v093', scalar=scalar, sphere_torus=analytic, image=images_result,
                  seconds=time.perf_counter()-started)
    write(REPORT, report)
    curves = dict(scalar=scalar_path, **{'analytic_'+k: v for k, v in analytic_paths.items()}, **image_paths)
    for name, result in curves.items():
        np.savez(CACHE/(name+'_path.npz'), **result)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': run()
