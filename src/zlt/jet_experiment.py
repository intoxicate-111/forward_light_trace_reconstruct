"""v092 small matched-operator manifold-jet validation (no historical writes)."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .basis_only import BasisOnlyZeroSetField
from .basis_initialization import directions, radial_roots
from .basis_renderer import make_context, render, radial_attachment
from .locality import BasisLayout, wendland_values
from .manifold_jets import JetLayout, ManifoldJetField
from .matched_jacobian import CandidateSupports, differential_block, integrated_kernel

CACHE = Path('runs/v092_manifold_jets')


def initial_field(device='cpu'):
    # Exact spherical initialization of the EXISTING basis-only representation,
    # F=1+lambda W(|x|/2); no analytic sphere is stored or evaluated at runtime.
    c = torch.zeros((1, 3), dtype=torch.float64, device=device)
    r = c.new_tensor([2.])
    b = wendland_values(c.new_tensor([[1., 0, 0]]), r)
    return BasisOnlyZeroSetField(BasisLayout(c, r), -1/b)


class ScalarField(ManifoldJetField):
    def __init__(self, reference, layout, coefficients, enabled=True):
        super().__init__(reference, layout, coefficients, enabled)
        self.supports = CandidateSupports(layout.centers, layout.radii)


class Ellipsoid(BasisOnlyZeroSetField):
    """Independent analytic target only, never a reconstruction base."""
    def __init__(self, device):
        self.scale = torch.tensor([1.025, .985, 1.015], device=device, dtype=torch.float64)
        self.lower = self.scale.new_full((3,), -1.2)
        self.upper = -self.lower
        self.radial = initial_field(device)

    def evaluate(self, p):
        values = self.radial.evaluate(p/self.scale)
        return torch.cat((values[:, :1], values[:, 1:]/self.scale), 1)


def snapshot():
    import subprocess
    path = CACHE/'starting_state.json'
    if path.exists():
        return
    historical = list(Path('artifacts').glob('*'))+list(Path('figures').glob('*'))
    code = list(Path('src/zlt').glob('*.py'))+[Path('demo.py')]
    protected = [p for p in historical+code if p.is_file() and not p.name.startswith(('v092', 'jet_', 'manifold_jets'))]
    write(path, dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     protected_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}))


def geometry(field, target):
    d = directions(512, 591).cuda()
    x, _ = radial_attachment(field, d)
    truth = 1/(d/target.scale).norm(dim=1)
    error = x.norm(dim=1)-truth
    curvature_proxy = d[:, 2].abs() > .7
    return dict(radial_mae=float(error.abs().mean()), radial_p95=float(error.abs().quantile(.95)),
                radial_max=float(error.abs().max()), min_gradient=float(field.gradient(x).norm(dim=1).min()),
                pole_mae=float(error[curvature_proxy].abs().mean()), equator_mae=float(error[~curvature_proxy].abs().mean()))


def optimize(field, target_image, reference, boundary, cfg, target_field, steps=4):
    started = time.perf_counter(); records = []
    image, state, _ = render(field, reference, boundary, cfg)
    for iteration in range(steps+1):
        mse = float(np.mean((image-target_image)**2))
        row = dict(iteration=iteration, mse=mse, geometry=geometry(field, target_field))
        records.append(row)
        if iteration == steps:
            break
        J = jacobian(field, field.supports, reference, state, boundary, cfg)
        r = (image-target_image).flatten(); gram = J.T@J/len(r); alignment = J.T@r/len(r)
        mu = max(float(np.diag(gram).max())*1e-3, 1e-8)
        delta = -np.linalg.solve(gram+mu*np.eye(len(gram)), alignment)
        row.update(gradient_percentiles=np.quantile(np.abs(alignment), [0, .5, .9, 1]).tolist(),
                   image_jacobian_nnz=int(np.count_nonzero(np.abs(J) > 1e-12)), image_jacobian_entries=int(J.size),
                   regularized_condition=float(np.linalg.cond(gram+mu*np.eye(len(gram)))), damping=mu, trials=[])
        accepted = False
        for alpha in (1., .5, .25, .125, .0625):
            candidate = field.with_coefficients(field.coefficients+alpha*torch.from_numpy(delta).to(field.coefficients))
            try:
                im, st, _ = render(candidate, reference, boundary, cfg)
                loss = float(np.mean((im-target_image)**2))
                row['trials'].append(dict(alpha=alpha, mse=loss))
                if loss < mse*(1-1e-8):
                    field, image, state = candidate, im, st; accepted = True; break
            except RuntimeError as exc:
                row['trials'].append(dict(alpha=alpha, failure=str(exc)))
        row['accepted'] = accepted
        print('opt', len(field.coefficients), iteration, mse, accepted, flush=True)
        if not accepted:
            break
    final_mse = float(np.mean((image-target_image)**2))
    return field, image, state, dict(trajectory=records, initial_mse=records[0]['mse'], final_mse=final_mse,
                                   geometry=geometry(field, target_field), seconds=time.perf_counter()-started)


def topology(field, label):
    from skimage.measure import marching_cubes
    import trimesh
    n = 64; a = torch.linspace(-1.5, 1.5, n, device='cuda', dtype=torch.float64)
    points = torch.cartesian_prod(a, a, a)
    values = torch.cat([field.value(points[i:i+4096]) for i in range(0, len(points), 4096)]).cpu().numpy().reshape(n, n, n)
    vertices, faces, _, _ = marching_cubes(values, 0, spacing=(3/(n-1),)*3)
    mesh = trimesh.Trimesh(vertices=vertices-1.5, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    mesh.export(CACHE/(label+'.ply'))
    _, _, radial = radial_roots(field, directions(512, 691).cuda())
    return dict(grid=64, components=len(components), watertight=bool(mesh.is_watertight),
                component_areas=[float(x.area) for x in components], radial_scan=radial,
                caveat='Finite grid and 512 rays, not a global topology proof; MC is diagnostic only.')


def save_field(field, name):
    torch.save(dict(reference=field.reference.state_dict(), centers=field.layout.centers.cpu(),
                    frames=field.layout.frames.cpu(), radii=field.layout.radii.cpu(), order=field.layout.order,
                    coefficients=field.coefficients.cpu(), scalar=isinstance(field, ScalarField)), CACHE/(name+'.pt'))


def load_field(name, device='cuda'):
    state = torch.load(CACHE/(name+'.pt'), weights_only=True)
    base = BasisOnlyZeroSetField.from_state_dict(state['reference'], device)
    layout = JetLayout(state['centers'].to(device), state['frames'][:, 2].to(device),
                       state['radii'].to(device), state['order'])
    layout.frames = state['frames'].to(device)
    cls = ScalarField if state['scalar'] else ManifoldJetField
    return cls(base, layout, state['coefficients'].to(device))


def audit():
    audit_path = Path('artifacts/v092_jet_audit.json')
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if not previous['passed'] and not (CACHE/'audit_original_bitwise_gate.json').exists():
            write(CACHE/'audit_original_bitwise_gate.json', previous)
    cfg, reference, boundary = make_context()
    cfg = dict(cfg, count=256)
    reference = dict(reference, positions=directions(256, 391), normals=directions(256, 391),
                     weights=torch.full((256,), cfg['mass']/256, dtype=torch.float64))
    with torch.no_grad():
        base = initial_field('cuda')
        field = load_field('fixed_p2_24')
        disabled = ManifoldJetField(base, field.layout, field.coefficients, enabled=False)
        baseline_image, _, _ = render(base, reference, boundary, cfg)
        disabled_image, _, _ = render(disabled, reference, boundary, cfg)
        replay, _, _ = render(field, reference, boundary, cfg)
        checks = dict(disabled_image_max_abs_error=float(np.abs(disabled_image-baseline_image).max()),
                      saved_image_replay_max_abs_error=float(np.abs(replay-np.load(CACHE/'fixed_p2_24.npy')).max()),
                      disabled_field_bitwise_equal=bool(torch.equal(disabled.evaluate(directions(512).cuda()), base.evaluate(directions(512).cuda()))),
                      numerical_tolerance=1e-12,
                      gate_note='Field outputs are bitwise identical. Separate CUDA scatter image renders need floating-point tolerance; original exact-zero image gate is retained if it failed.',
                      fields=[])
        for path in sorted(CACHE.glob('*.pt')):
            field = load_field(path.stem)
            x, _ = radial_attachment(field, directions(512, 591).cuda())
            pi, _, _, _ = field.supports.query(x)
            counts = torch.bincount(pi, minlength=len(x)).double()
            gamma_ratio = field.reference.gradient(x).norm(dim=1)/field.gradient(x).norm(dim=1)
            checks['fields'].append(dict(name=path.stem, sparse_mode_edges=len(pi),
                                         mode_fanout_quantiles=torch.quantile(counts, counts.new_tensor([.5, .9, .95, 1])).cpu().tolist(),
                                         max_radial_departure_from_phase_reference=float((x.norm(dim=1)-1).abs().max()),
                                         frozen_vs_current_gradient_norm_ratio=[float(gamma_ratio.min()), float(gamma_ratio.max())]))
        checks['passed'] = (checks['disabled_field_bitwise_equal'] and checks['disabled_image_max_abs_error'] < 1e-12
                            and checks['saved_image_replay_max_abs_error'] < 1e-12)
    write(Path('artifacts/v092_jet_audit.json'), checks)
    print(json.dumps(checks, indent=2))


def fixed_experiments(base, target, target_image, reference, boundary, cfg):
    progress = CACHE/'fixed_progress.json'
    results = json.loads(progress.read_text()) if progress.exists() else []
    for budget in (12, 24):
        for name, order, cls in [('scalar', 0, ScalarField), ('p0', 0, ManifoldJetField),
                                 ('p1', 1, ManifoldJetField), ('p2', 2, ManifoldJetField)]:
            if any(x['name'] == name and x['budget'] == budget for x in results):
                continue
            count = budget//(1, 3, 6)[order]
            centers = directions(count, 391).cuda()
            layout = JetLayout(centers, base.gradient(centers), centers.new_full((count,), 1.1), order)
            field = cls(base, layout, centers.new_zeros(budget))
            field, image, _, row = optimize(field, target_image, reference, boundary, cfg, target)
            label = f'fixed_{name}_{budget}'
            np.save(CACHE/(label+'.npy'), image); save_field(field, label)
            row.update(name=name, budget=budget, centers=count, scalar_dofs=budget, label=label,
                       topology=topology(field, label), coefficients=field.coefficients.cpu().tolist())
            results.append(row); write(CACHE/'fixed_progress.json', results)
    return results


def birth_experiments(base, target, target_image, reference, boundary, cfg):
    progress = CACHE/'birth_progress.json'
    results = json.loads(progress.read_text()) if progress.exists() else []
    for policy in ('loss_driven', 'random', 'uniform'):
        if any(x['policy'] == policy for x in results):
            continue
        rng = np.random.default_rng(391); chosen = [0]; centers = directions(8, 391).cuda()
        initial_layout = JetLayout(centers[:1], base.gradient(centers[:1]), centers.new_tensor([1.1]), 2)
        field = ManifoldJetField(base, initial_layout, centers.new_zeros(6)); sequence = []
        for stage in range(3):
            field, image, state, row = optimize(field, target_image, reference, boundary, cfg, target, steps=3)
            label = f'birth_{policy}_{stage}'
            np.save(CACHE/(label+'.npy'), image); save_field(field, label)
            row.update(stage=stage, centers=len(chosen), dofs=len(chosen)*6, chosen=chosen.copy(),
                       topology=topology(field, label), label=label)
            sequence.append(row)
            if stage == 2:
                break
            available = [i for i in range(len(centers)) if i not in chosen]
            # Candidate centers/frames come from CURRENT surface, then freeze.
            points, normals = radial_attachment(field, centers[available])
            block_layout = JetLayout(points, normals, points.new_full((len(points),), 1.1), 2)
            # Same frozen phase reference for candidate scoring AND insertion;
            # existing coefficients remain jointly optimizable after birth.
            from .manifold_jets import JetSupports
            supports = JetSupports(base, block_layout)
            J = jacobian(field, supports, reference, state, boundary, cfg)
            residual = (image-target_image).flatten(); scores = []; mus = []
            for j in range(len(available)):
                block = J[:, j*6:(j+1)*6]; gram = block.T@block/len(residual); align = block.T@residual/len(residual)
                mu = max(float(np.diag(gram).max())*1e-3, 1e-8); mus.append(mu)
                step = -np.linalg.solve(gram+mu*np.eye(6), align)
                scores.append(float(-align@step-.5*step@gram@step))
            if policy == 'loss_driven':
                selected = int(np.argmax(scores))
            elif policy == 'random':
                selected = int(rng.integers(len(available)))
            else:
                # Greedy farthest angular coverage, no image information.
                distance = torch.cdist(centers[available], centers[chosen]).amin(1)
                selected = int(distance.argmax())
            chosen.append(available[selected])
            row.update(candidate_ids=available, predicted_reduction=scores, damping=mus,
                       selected=available[selected], score_definition='Full 6x6 block GN prediction for 0.5 MSE')
            # Keep previous active coefficients optimizable: append a new chart
            # to the original phase. Its gamma is the frozen phase reference;
            # score must therefore use that SAME gamma (see below).
            new_centers = torch.cat((field.layout.centers, points[selected:selected+1]))
            new_normals = torch.cat((field.layout.frames[:, 2], normals[selected:selected+1]))
            layout = JetLayout(new_centers, new_normals, new_centers.new_full((len(chosen),), 1.1), 2)
            field = ManifoldJetField(base, layout, torch.cat((field.coefficients, centers.new_zeros(6))))
            write(CACHE/f'birth_{policy}_progress.json', sequence)
        results.append(dict(policy=policy, sequence=sequence, final_mse=sequence[-1]['final_mse']))
        write(CACHE/'birth_progress.json', results)
    return results


def jacobian(field, supports, reference, state, boundary, cfg):
    """Existing transport tangent + existing detector tangent; small image J."""
    h, w = cfg['resolution']; k = len(supports.radii); views = []
    for v in range(4):
        acc = torch.zeros((h*w, k, 3), device='cuda', dtype=torch.float64)
        for a in range(0, len(reference['weights']), 128):
            result = differential_block(field, reference, state, slice(a, a+128), v,
                                        supports, boundary, cfg, diagnostic=True)
            if result is None:
                continue
            row, col, energy, de, drow, dcol, _ = result
            ids, weights, dy, dx = integrated_kernel(row, col, cfg['capture'], cfg['resolution'])
            for tap in range(36):
                term = weights[:, tap, None, None]*de
                term = term+(dy[:, tap, None]*drow+dx[:, tap, None]*dcol)[:, :, None]*energy[:, None, :]
                acc.index_add_(0, ids[:, tap], term)
        views.append((acc*cfg['gain']*h*w).permute(0, 2, 1).reshape(-1, k).cpu().numpy())
    return np.concatenate(views)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fd_validation(base, reference, boundary, cfg):
    center = directions(1, 43).cuda()
    layout = JetLayout(center, base.gradient(center), center.new_tensor([.9]), 2)
    field = ManifoldJetField(base, layout, center.new_tensor([.008, -.003, .002, .004, -.002, .003]))
    image, state, _ = render(field, reference, boundary, cfg)
    j = jacobian(field, field.supports, reference, state, boundary, cfg)
    rows = []
    for k, name in enumerate(('constant', 'u', 'v', 'uu', 'uv', 'vv')):
        col = j[:, k]; norm = np.linalg.norm(col); keep = np.abs(col) > norm*1e-6; epsrows = []
        for eps in (1e-3, 1e-4, 1e-5, 1e-6):
            step = torch.zeros_like(field.coefficients); step[k] = eps
            plus, _, _ = render(field.with_coefficients(field.coefficients+step), reference, boundary, cfg)
            minus, _, _ = render(field.with_coefficients(field.coefficients-step), reference, boundary, cfg)
            fd = ((plus-minus)/(2*eps)).flatten(); fn = np.linalg.norm(fd)
            epsrows.append(dict(epsilon=eps, relative_l2=float(np.linalg.norm(fd-col)/max(norm, 1e-30)),
                                cosine=float(fd@col/max(norm*fn, 1e-30)), norm_ratio=float(fn/max(norm, 1e-30)),
                                sign_agreement=float(np.mean(np.sign(fd[keep]) == np.sign(col[keep]))) if keep.any() else 0.))
        good = [x['relative_l2'] < 1e-4 and x['cosine'] > .9999 and .999 < x['norm_ratio'] < 1.001
                and x['sign_agreement'] > .99 for x in epsrows]
        pairs = [[epsrows[i]['epsilon'], epsrows[i+1]['epsilon']] for i in range(3) if good[i] and good[i+1]]
        rows.append(dict(mode=name, norm=float(norm), steps=epsrows, resolved_intervals=pairs, passed=bool(pairs)))
        print('FD', name, rows[-1]['passed'], [x['relative_l2'] for x in epsrows], flush=True)
        write(CACHE/'fd_progress.json', rows)
    return dict(passed=all(x['passed'] for x in rows), coefficients=rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fd-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        audit(); return
    if not args.fd_only and Path('artifacts/v092_manifold_jets.json').exists():
        print('Completed report exists; preserving its results and runtime. Use --audit-only for replay checks.', flush=True)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    snapshot()
    cfg, reference, boundary = make_context()
    base = initial_field('cuda')
    fd_path = Path('artifacts/v092_manifold_jet_fd.json')
    with torch.no_grad():
        fd = (json.loads(fd_path.read_text()) if fd_path.exists() and not args.fd_only
              else fd_validation(base, reference, boundary, cfg))
    write(Path('artifacts/v092_manifold_jet_fd.json'), fd)
    if args.fd_only:
        return
    if not fd['passed']:
        raise RuntimeError('IMAGE_FD_GATE_FAILED; fixed/birth runs not started')
    started = time.perf_counter()
    # All reconstruction arms use exactly the same reduced source IDs/weights;
    # the more expensive FD preflight above uses the original 1024-source set.
    cfg = dict(cfg, count=256, note=cfg['note']+' Fixed/birth pilot: 256 shared source IDs, not sampling-convergence evidence.')
    reference = dict(reference)
    reference['positions'] = directions(256, 391)
    reference['normals'] = reference['positions'].clone()
    reference['weights'] = torch.full((256,), cfg['mass']/256, dtype=torch.float64)
    with torch.no_grad():
        target = Ellipsoid('cuda')
        target_image, _, _ = render(target, reference, boundary, cfg)
        initial_image, _, _ = render(base, reference, boundary, cfg)
        np.save(CACHE/'target.npy', target_image); np.save(CACHE/'initial.npy', initial_image)
        fixed = fixed_experiments(base, target, target_image, reference, boundary, cfg)
        birth = birth_experiments(base, target, target_image, reference, boundary, cfg)
    report = dict(version='v092_manifold_jets', config=cfg, fd=fd, fixed=fixed, birth=birth,
                  seconds=time.perf_counter()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
    write(Path('artifacts/v092_manifold_jets.json'), report)


if __name__ == '__main__':
    main()
