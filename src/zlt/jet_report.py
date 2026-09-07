"""Report only v092 outputs; never rewrite historical experiments."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .jet_experiment import CACHE, write


def finalize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    path = Path('artifacts/v092_manifold_jets.json')
    report = json.loads(path.read_text()); fixed = report['fixed']; birth = report['birth']
    target = np.load(CACHE/'target.npy'); initial = np.load(CACHE/'initial.npy')
    exposure = float(np.quantile(target, .999)); files = []
    def save(fig, name):
        name = 'figures/v092_'+name+'.png'
        fig.savefig(name, dpi=130, bbox_inches='tight'); plt.close(fig); files.append(name)
    selected = [x for x in fixed if x['budget'] == 24]
    images = [target, initial]+[np.load(CACHE/(x['label']+'.npy')) for x in selected]
    names = ['target', 'initial']+[x['name'] for x in selected]
    fig, axes = plt.subplots(6, 4, figsize=(12, 16), constrained_layout=True)
    for row, (name, image) in enumerate(zip(names, images)):
        for v in range(4):
            axes[row, v].imshow(np.clip(image[v]/exposure, 0, 1)); axes[row, v].set_title(f'{name}: view {v}'); axes[row, v].axis('off')
    fig.suptitle('Matched CURRENT operator; 64² / 256 sources, common exposure; 24 incremental DoFs')
    save(fig, 'reconstructions')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    vmax = max(float(np.abs(im-target).max()) for im in images[2:])
    for ax, row, image in zip(axes, selected, images[2:]):
        ax.imshow(np.abs(image[0]-target[0]).mean(2), vmin=0, vmax=vmax, cmap='magma')
        ax.set_title(row['name']+' absolute RGB error'); ax.axis('off')
    save(fig, 'differences')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ('scalar', 'p0', 'p1', 'p2'):
        rows = [x for x in fixed if x['name'] == name]
        axes[0].plot([x['scalar_dofs'] for x in rows], [x['final_mse'] for x in rows], 'o-', label=name)
        axes[1].plot([x['centers'] for x in rows], [x['final_mse'] for x in rows], 'o-', label=name)
    for ax in axes: ax.set_yscale('log'); ax.set_ylabel('final image MSE'); ax.legend()
    axes[0].set_xlabel('Incremental scalar DoFs (common frozen reference has 1)'); axes[1].set_xlabel('Active deformation centers')
    save(fig, 'efficiency')
    fig = plt.figure(figsize=(14, 4))
    for i, row in enumerate(selected):
        state = torch.load(CACHE/(row['label']+'.pt'), weights_only=True)
        ax = fig.add_subplot(1, 4, i+1, projection='3d'); c = state['centers'].numpy(); f = state['frames'].numpy(); r = state['radii'].numpy()
        ax.scatter(*c.T, s=25)
        angle = np.linspace(0, 2*np.pi, 80)
        for center, frame, radius in zip(c, f, r):
            ring = center+radius*(np.cos(angle)[:, None]*frame[0]+np.sin(angle)[:, None]*frame[1])
            ax.plot(*ring.T, alpha=.3, linewidth=.5)
        ax.set_title(f"{row['name']}: {len(c)} centers, r=1.1"); ax.set_box_aspect((1, 1, 1))
    save(fig, 'centers_support_orders')
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, result in enumerate(birth):
        for stage, row in enumerate(result['sequence']):
            im = np.load(CACHE/(row['label']+'.npy'))
            axes[i, stage].imshow(np.clip(im[0]/exposure, 0, 1)); axes[i, stage].axis('off')
            axes[i, stage].set_title(f"{result['policy']} / {row['centers']} jets\nMSE {row['final_mse']:.5g}")
    save(fig, 'birth_sequence')
    fig, ax = plt.subplots(figsize=(6, 4))
    for result in birth:
        ax.plot([x['centers'] for x in result['sequence']], [x['final_mse'] for x in result['sequence']], 'o-', label=result['policy'])
    ax.set(xlabel='Active p2 jets', ylabel='MSE'); ax.legend(); save(fig, 'birth_loss')
    from .jet_experiment import load_field
    from .basis_initialization import directions
    from .basis_renderer import radial_attachment
    d = directions(1024, 591)
    axes_scale = d.new_tensor([1.025, .985, 1.015])
    target_r = 1/(d/axes_scale).norm(dim=1)
    gt_gradient = 2*d*target_r[:, None]/axes_scale**2
    gt_norm = gt_gradient.norm(dim=1)
    gt_normal = gt_gradient/gt_norm[:, None]
    diagonal = 2/axes_scale**2
    curvature = .5*(diagonal.sum()-(gt_normal.square()*diagonal).sum(1))/gt_norm
    low = curvature <= curvature.quantile(.25)
    high = curvature >= curvature.quantile(.75)
    regional = []
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    for i, row in enumerate(selected):
        field = load_field(row['label'], 'cpu')
        with torch.no_grad():
            x, _ = radial_attachment(field, d)
        error = x.norm(dim=1)-target_r
        regional.append(dict(name=row['name'], scalar_dofs=24,
                             lower_curvature_radial_mae=float(error[low].abs().mean()),
                             higher_curvature_radial_mae=float(error[high].abs().mean())))
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.scatter(*x.numpy().T, c=error.numpy(), vmin=-.025, vmax=.025, cmap='coolwarm', s=3)
        ax.set_box_aspect((1, 1, 1)); ax.set_title(row['name']+' signed radial error')
    save(fig, 'geometry')
    fig, ax = plt.subplots(figsize=(7, 4))
    index = np.arange(len(regional))
    ax.bar(index-.18, [x['lower_curvature_radial_mae'] for x in regional], width=.36, label='lowest curvature quartile')
    ax.bar(index+.18, [x['higher_curvature_radial_mae'] for x in regional], width=.36, label='highest curvature quartile')
    ax.set_xticks(index, [x['name'] for x in regional]); ax.set_ylabel('Radial MAE'); ax.legend()
    ax.set_title('24 incremental DoFs; target mean curvature bins, not flat patches')
    save(fig, 'curvature_regions')
    initial_loss = float(np.mean((initial-target)**2))
    thresholds = [initial_loss*fraction for fraction in (.9, .75, .5)]
    efficiency = []
    for threshold in thresholds:
        for name in ('scalar', 'p0', 'p1', 'p2'):
            passing = [x for x in fixed if x['name'] == name and x['final_mse'] <= threshold]
            best = min(passing, key=lambda x: x['scalar_dofs']) if passing else None
            efficiency.append(dict(threshold=threshold, name=name, reached=best is not None,
                                   centers=best['centers'] if best else None, scalar_dofs=best['scalar_dofs'] if best else None))
    all_topologies = [x['topology'] for x in fixed]+[s['topology'] for x in birth for s in x['sequence']]
    report.update(figures=files, efficiency=efficiency, initial_mse=initial_loss,
                  curvature_regions=dict(target_mean_curvature_range=[float(curvature.min()), float(curvature.max())],
                                         quartiles=[float(curvature.quantile(.25)), float(curvature.quantile(.75))],
                                         results=regional, note='Actual ellipsoid mean curvature, bottom/top quartile of 1024 common directions. No truly flat region; location/coverage confounding remains.'),
                  audit=json.loads(Path('artifacts/v092_jet_audit.json').read_text()) if Path('artifacts/v092_jet_audit.json').exists() else None,
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [CACHE/'target.npy', CACHE/'initial.npy']},
                  runtime_scope='Comparison run including target, fixed, birth and topology checks; excludes FD preflight, final replay audit and plots.',
                  target='Independent ellipsoidal warp of the same radial basis profile; axes [1.025,.985,1.015]. No target jet dictionary.',
                  formulation='F_theta=F_ref-|grad F_ref| sum_k W(sqrt(u²+v²)) collar(w) sum_ij a_ij u^i v^j',
                  initialization='F_ref=1-(16/3) W(|x|/2), exact unit-sphere zero set in existing basis-only representation, NOT an analytic sphere base.',
                  frame_policy='One small-deformation frozen reference phase; born centers and frames sampled on current surface, gamma remains the same frozen phase reference for both scoring and insertion.',
                  reference_scalar_dofs=1, fd_source_count=1024,
                  single_closed_surface_in_sampled_checks=all(x['components'] == 1 and x['watertight'] for x in all_topologies),
                  birth_loss_driven_beats_both=all(birth[0]['final_mse'] < x['final_mse'] for x in birth[1:]),
                  limitations=[
                      'Single smooth near-spherical target, two budgets, one seed; no general superiority claim. Threshold counts are minimum within the tested grid, not global minimum budgets.',
                      'All arms use a common frozen current basis-only reference; counts are incremental optimized DoFs, not total unrestricted field capacity.',
                      'Tangent modes describe first-order normal displacement; finite steps include nonlinear root effects.',
                      'No chart rebase or orientation derivative in this small phase; not validated for large deformations.',
                      'Normal displacement requires regular surface points. Exact reference gradient norm may be nonsmooth at off-surface critical points; C2 window continuity is not a global smoothness guarantee for that norm.',
                      'Normal collar is an explicit compact off-surface extension; it may affect transport away from the surface.',
                      '64² / 256 sources are formulation diagnostics, not image or source convergence evidence.',
                      'Ellipsoid has no flat region; pole/equator errors do not establish low-vs-high-curvature order preference.',
                      'Finite-grid/ray topology checks are not proof of global absence of accidental components.',
                      'Small dense image Jacobian blocks only; field supports remain sparse. No large-scale performance claim.',
                      'Birth compares full block GN scores with random and farthest-coverage policies, but joint post-birth optimization is not a pure single-block gain measurement.',
                      'No p refinement, radius learning, death, merge, global UV, Eikonal, Bunny, Full-HD or HPC.'])
    write(path, report)
    with Path('artifacts/v092_manifold_jets.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['experiment', 'name', 'centers', 'scalar_dofs', 'initial_mse', 'final_mse', 'radial_mae', 'seconds'])
        writer.writeheader()
        for x in fixed:
            writer.writerow(dict(experiment='fixed', name=x['name'], centers=x['centers'], scalar_dofs=x['scalar_dofs'],
                                 initial_mse=x['initial_mse'], final_mse=x['final_mse'], radial_mae=x['geometry']['radial_mae'], seconds=x['seconds']))
        for x in birth:
            for row in x['sequence']:
                writer.writerow(dict(experiment='birth', name=x['policy'], centers=row['centers'], scalar_dofs=row['dofs'],
                                     initial_mse=row['initial_mse'], final_mse=row['final_mse'], radial_mae=row['geometry']['radial_mae'], seconds=row['seconds']))
    print(json.dumps(dict(efficiency=efficiency, birth=[(x['policy'], x['final_mse']) for x in birth]), indent=2))


if __name__ == '__main__': finalize()
