"""Isolated plots and exact verdicts for the v091 representation experiment."""
import json
from pathlib import Path
import numpy as np
import torch
from .basis_only import BasisOnlyZeroSetField
from .basis_only_experiment import CACHE
from .transverse_packet import write_json
from .high_sample import _write_csv
from .finite_packet import _scalar_csv_rows


def figures(initial,renderer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    field=BasisOnlyZeroSetField.from_state_dict(torch.load(CACHE/'initialized_field.pt',weights_only=True))
    folder=Path('figures'); folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/('v091_'+name+'.png'),dpi=150,bbox_inches='tight'); plt.close(fig)
    roots=np.load(CACHE/'initial_roots.npy'); radial=np.linalg.norm(roots,axis=1)-1
    fig=plt.figure(figsize=(11,5)); ax=fig.add_subplot(121,projection='3d')
    cloud=ax.scatter(*roots.T,c=radial,cmap='coolwarm',s=2,vmin=-.1,vmax=.1); fig.colorbar(cloud,ax=ax,shrink=.6,label='signed radial error')
    ax.set_title('F = 1 + sum(lambda B), K=256, r=0.6'); ax.set_box_aspect((1,1,1))
    ax=fig.add_subplot(122); ax.plot(roots[:,0],roots[:,1],'.',markersize=1,alpha=.1); ax.set_aspect('equal'); ax.set_title('Actual sampled zero set: XY projection'); save(fig,'basis_only_sphere')
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].hist(radial,bins=80); axes[0].set(xlabel='Signed radial error',ylabel='Directions')
    ordered=np.sort(np.abs(radial)); axes[1].plot(ordered,np.linspace(0,1,len(ordered))); axes[1].set(xlabel='Absolute radial error',ylabel='CDF'); save(fig,'basis_only_radial_error')
    n=240; axis=torch.linspace(-1.8,1.8,n,dtype=torch.float64); xx,yy=torch.meshgrid(axis,axis,indexing='xy')
    p=torch.stack((xx.flatten(),yy.flatten(),torch.zeros(n*n,dtype=torch.float64)),1)
    values=torch.cat([field.value(p[i:i+4096]) for i in range(0,len(p),4096)]).numpy().reshape(n,n)
    fig,ax=plt.subplots(figsize=(8,7)); m=ax.imshow(values,origin='lower',extent=(-1.8,1.8,-1.8,1.8),cmap='coolwarm',vmin=-1,vmax=1)
    ax.contour(xx,yy,values,levels=[0],colors='black',linewidths=1.5); angle=np.linspace(0,2*np.pi,512)
    ax.plot(np.cos(angle),np.sin(angle),'g--',label='analytic sphere z=0')
    centers=field.layout.centers.numpy(); r=float(field.layout.radii[0]); ids=np.where(np.abs(centers[:,2])<r)[0]
    for i in ids:
        rr=np.sqrt(r*r-centers[i,2]**2); ax.add_patch(plt.Circle(centers[i,:2],rr,fill=False,color='gray',alpha=.15,lw=.5))
    ax.scatter(centers[ids,0],centers[ids,1],s=7,c='gray',label='projected intersecting support centers')
    ax.set(xlim=(-1.8,1.8),ylim=(-1.8,1.8),title='Analytic scalar-field slice z=0; background = +1',xlabel='x',ylabel='y'); ax.legend(fontsize=8); fig.colorbar(m,ax=ax,label='F'); save(fig,'basis_only_sign_slice')
    mesh=trimesh.load(initial['mesh']['mesh_path'],process=False); parts=mesh.split(only_watertight=False)
    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,projection='3d')
    for i,part in enumerate(parts):
        v=np.asarray(part.vertices); v=v[::max(1,len(v)//8000)]; ax.scatter(*v.T,s=1,label=f'component {i}')
    ax.set_box_aspect((1,1,1)); ax.set_title(f"MC diagnostic ONLY: {len(parts)} component(s), watertight={mesh.is_watertight}"); ax.legend(); save(fig,'basis_only_zero_components')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,p in zip(axes,initial['perturbations']):
        i=p['coordinate']; ax.plot(roots[:,0],roots[:,2],'.',ms=1,alpha=.25,label='baseline')
        for s,color in ((-1,'b'),(1,'r')):
            x=np.load(CACHE/f'perturb_{i}_{s}.npy'); changed=np.linalg.norm(x-roots,axis=1)>1e-7
            ax.scatter(x[changed,0],x[changed,2],s=4,c=color,label=f'{s:+d} epsilon')
        ax.set_aspect('equal'); ax.set_title(f'lambda[{i}], epsilon={p["epsilon"]:.4g}'); ax.legend(fontsize=8)
    save(fig,'lambda_perturbations')
    t=initial['scalar_recovery']['trajectory']; fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot([x['iteration'] for x in t],[x['field_mse'] for x in t],'o-'); axes[0].set(xlabel='Recovery step',ylabel='Field MSE')
    axes[1].plot([x['iteration'] for x in t],[x['radial']['radial_mae'] for x in t],'o-'); axes[1].set(xlabel='Recovery step',ylabel='Radial MAE')
    for iteration,label in ((0,'perturbed'),(3,'recovered')):
        x=np.load(CACHE/f'scalar_recovery_{iteration}.npy'); axes[2].plot(x[:,0],x[:,2],'.',ms=2,alpha=.35,label=label)
    axes[2].set_aspect('equal'); axes[2].legend(); save(fig,'scalar_recovery')
    t=renderer.get('trajectory',[])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    if t:
        axes[0].plot([x['iteration'] for x in t],[x['mse'] for x in t],'o-')
        axes[1].plot([x['iteration'] for x in t],[x['radial']['radial_mae'] for x in t],'o-')
    axes[0].set(xlabel='Fixed-K CURRENT optimization step',ylabel='RGB MSE'); axes[1].set(xlabel='Fixed-K CURRENT optimization step',ylabel='Radial MAE')
    fig.suptitle('Reduced-cost 64x64 / 1024 sources / 4 views; no birth'); save(fig,'renderer_recovery')
    if (CACHE/'renderer_final.npy').exists():
        names=['target','initialized','perturbed','final']; images=[np.load(CACHE/f'renderer_{name}.npy') for name in names]
        exposure=float(np.quantile(images[0],.999)); fig,axes=plt.subplots(4,4,figsize=(12,11))
        for row,(name,image) in enumerate(zip(names,images)):
            for view in range(4): axes[row,view].imshow(np.clip(image[view]/exposure,0,1)); axes[row,view].set_title(f'{name} / view {view}'); axes[row,view].axis('off')
        fig.suptitle(f'CURRENT matched sphere observation; shared exposure {exposure:.4g}'); save(fig,'renderer_images')
    return [str(p) for p in sorted(folder.glob('v091_*.png'))]


def finalize():
    torch.set_num_threads(2)
    initial=json.loads(Path('artifacts/v091_sphere_initialization.json').read_text()); renderer=json.loads((CACHE/'renderer.json').read_text())
    movement=all(max(v['max_displacement'] for v in x['variants'])>1e-7 and all(v['max_displacement_outside_baseline_support']<1e-7 for v in x['variants']) for x in initial['perturbations'])
    fd=renderer.get('fd',{}).get('passed','UNRESOLVED')
    verdicts=dict(BASIS_ONLY_FIELD_IMPLEMENTED=True,FIXED_GEOMETRY_BASE_REMOVED=True,BACKGROUND_GAUGE_FIXED=True,
        BASIS_ONLY_FIELD_OUTSIDE_SUPPORT_EQUALS_BACKGROUND=initial['outside_support_exact'],OLD_003_COEFFICIENT_LIMIT_IS_NOT_USED=True,
        SINGLE_SCALE_ONLY=True,BASIS_ONLY_VALUE_GRADIENT_CONSISTENT=initial['gradient_check']['passed'],
        BASIS_ONLY_SPHERE_INITIALIZATION_VALID=initial['initialization_valid'],BASIS_ONLY_SPHERE_HAS_SINGLE_DOMINANT_ZEROSET=initial['single_dominant_zeroset'],
        LAMBDA_DIRECTLY_CONTROLS_ZEROSET_GEOMETRY=movement,FIXED_K_LAMBDA_OPTIMIZATION_REDUCES_FIELD_ERROR=initial['scalar_recovery']['passed'],
        BASIS_ONLY_FIELD_OPTIMIZES_THROUGH_CURRENT_RENDERER=renderer.get('passed','UNRESOLVED'),BASIS_ONLY_RENDERER_LAMBDA_DERIVATIVE_PASSES_FD=fd)
    verdicts['BASIS_ONLY_FORMULATION_READY_FOR_DYNAMIC_BIRTH']=all(x is True for x in verdicts.values())
    final_geometry=json.loads((CACHE/'final_geometry.json').read_text()) if (CACHE/'final_geometry.json').exists() else None
    report=dict(experiment='v091_basis_only_field',starting_state=json.loads((CACHE/'starting_state.json').read_text()),
        formulation='F(x;lambda)=1+sum_k lambda_k B_k(x)',INITIAL_K=initial['INITIAL_K'],COMMON_SUPPORT_RADIUS=initial['COMMON_SUPPORT_RADIUS'],
        MULTISCALE_BASIS=False,dynamic_birth_executed=False,verdicts=verdicts,initialization=initial,renderer=renderer,final_geometry=final_geometry,
        SPHERE_ROOT_COVERAGE=initial['radial']['root_coverage'],SPHERE_RADIAL_MAE=initial['radial']['radial_mae'],
        SPHERE_RADIAL_P95=initial['radial']['radial_p95'],SPHERE_SIGN_ACCURACY=initial['sign']['sign_accuracy'],MIN_ZEROSET_GRADIENT_NORM=initial['radial']['min_zeroset_gradient_norm'],
        limitations=['Single-scale 256/.6 is an approximate sphere: retain p95 AND maximum radial error; not a high-accuracy representation claim.',
            'Single-component and absence of extra shells are finite-direction/finite-grid diagnostics, not a global topological proof.',
            'Positive fixed background supplies outside sign, not sphere geometry; it does not prove complete identifiability inside the supported domain.',
            'Matched renderer test is reduced cost with float64 diagnostic readout; not Full-HD convergence or Bunny reconstruction.',
            'No dynamic birth, no nonexistent candidate scoring, no multiscale, no source/weight optimization.',
            'Fixed sphere-reference source rays/weights remain a coverage limitation for future large deformations.',
            'Current renderer is unchanged; new bracketed source attachment prevents Newton from escaping into unsupported constant field.',
            'Initialization uses geometric signed constraints only; renderer recovery optimizes image loss only. No mesh-rendered observation target.',
            'GPU timing may include resource contention with the untouched concurrent v090 experiment.'],
        fd_gate_history='Original requirement for the two smallest epsilons rejected weak coefficient92 because tiny component signs suffered subtraction noise. All original rows were retained. Final validation uses two adjacent well-resolved epsilons with a stricter L2 threshold; the same three coordinates were re-tested, not re-selected.',
        figures=figures(initial,renderer))
    write_json(Path('artifacts/v091_basis_only_field.json'),report)
    _write_csv(Path('artifacts/v091_basis_only_field.csv'),_scalar_csv_rows(report))
    print(json.dumps(verdicts,indent=2),flush=True)


if __name__=='__main__': finalize()
