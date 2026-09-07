"""v091 isolated representation validation; no birth, no historical writes."""
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .basis_only import BasisOnlyZeroSetField
from .basis_initialization import initialize,directions,shell_points,radial_roots
from .transverse_packet import write_json,digest

CACHE=Path('runs/v091_basis_only_field')


def preserve():
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'starting_state.json'
    if path.exists(): return
    # v090 reports legitimately change in their OTHER running process. Record
    # them, but never rewrite or require those live reports to stay frozen.
    paths=[p for folder in ('artifacts','figures') for p in Path(folder).iterdir() if p.is_file() and 'v091' not in p.name]
    write_json(path,dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        historical_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if 'v090' not in p.name},
        concurrent_v090_sha256_at_start={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if 'v090' in p.name},
        protected_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/zlt').glob('*.py') if not p.name.startswith('basis_')},
        policy='No modification/killing of concurrent v090; no old artifacts written; no commit/push.'))


def signs(field):
    inside=shell_points(8192,0,.90,291).to(field.coefficients.device)
    outside=shell_points(8192,1.10,2.,292).to(field.coefficients.device)
    iv=field.value(inside); ov=field.value(outside)
    return dict(center_value=float(field.value(inside.new_zeros((1,3)))[0]),inside_negative_fraction=float((iv<0).double().mean()),
        outside_positive_fraction=float((ov>0).double().mean()),sign_accuracy=float(torch.cat((iv<0,ov>0)).double().mean()),
        inside_radial_range=[0,.90],outside_radial_range=[1.10,2.],samples_per_class=8192,
        note='Sign accuracy excludes the near-surface uncertainty band; radial errors are reported separately.')


def mesh_diagnostic(field,label,n=112):
    import trimesh
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree
    device=field.coefficients.device; axis=torch.linspace(-1.8,1.8,n,device=device,dtype=torch.float64)
    values=np.empty(n**3)
    for i in range(0,n**3,8192):
        ids=torch.arange(i,min(i+8192,n**3),device=device)
        ijk=torch.stack((ids//(n*n),(ids//n)%n,ids%n),1)
        values[i:i+len(ids)]=field.value(axis[ijk]).cpu().numpy()
    v,f,_,_=marching_cubes(values.reshape(n,n,n),0.,spacing=(3.6/(n-1),)*3); v-=1.8
    mesh=trimesh.Trimesh(v,f,process=False); mesh.export(CACHE/(label+'.ply'))
    components=mesh.split(only_watertight=False)
    from .mesh_field import mesh_surface_samples
    points,_=mesh_surface_samples(v,f,8192); gt=directions(8192,293).numpy()
    chamfer=.5*(np.abs(np.linalg.norm(points,axis=1)-1).mean()+cKDTree(points).query(gt)[0].mean())
    near=np.abs(values)<.02; selected=np.flatnonzero(near)
    if len(selected)>8192: selected=selected[np.linspace(0,len(selected)-1,8192).astype(int)]
    ids=torch.as_tensor(selected,device=device); ijk=torch.stack((ids//(n*n),(ids//n)%n,ids%n),1)
    g=field.gradient(axis[ijk]).norm(dim=1)
    return dict(grid_resolution=n,components=len(components),component_areas=sorted([float(x.area) for x in components],reverse=True),
        watertight=bool(mesh.is_watertight),surface_area=float(mesh.area),analytic_sphere_area=4*np.pi,
        approximate_symmetric_chamfer=float(chamfer),near_zero_sample_count=len(selected),near_zero_gradient_min=float(g.min()),
        near_zero_degenerate_fraction=float((g<1e-5).double().mean()),mesh_path=str(CACHE/(label+'.ply')),
        note='Temporary MC diagnostic only, never representation/source/observation target; finite-grid topology and sampled Chamfer.')


def gradient_check(field,roots):
    device=field.coefficients.device; points=shell_points(2048,0,1.8,294).to(device)
    boundary=field.layout.centers+directions(len(field.coefficients),295).to(device)*field.layout.radii[:,None]*(1+1e-7)
    groups={'random':points,'support_boundary':boundary,'zero_set':roots[:1024]}; result={}
    for name,p in groups.items():
        analytic=field.gradient(p); rows=[]
        for eps in (1e-4,1e-5,1e-6):
            fd=[]
            for a in range(3):
                offset=torch.zeros_like(p); offset[:,a]=eps
                fd.append((field.value(p+offset)-field.value(p-offset))/(2*eps))
            fd=torch.stack(fd,1); error=(fd-analytic).norm()/analytic.norm().clamp_min(1e-30)
            rows.append(dict(epsilon=eps,relative_l2=float(error),cosine=float((fd*analytic).sum()/(fd.norm()*analytic.norm())),max_abs_error=float((fd-analytic).abs().max())))
        result[name]=rows
    return dict(groups=result,passed=all(rows[-1]['relative_l2']<1e-5 for rows in result.values()))


def run():
    preserve(); torch.set_num_threads(2); started=time.perf_counter()
    attempts=[]; selected=None
    for k,r in ((128,.6),(256,.6)):
        field,initial,system=initialize(k,r)
        _,_,radial=radial_roots(field,directions(512,296)); sign=signs(field)
        valid=radial['root_coverage']==1 and radial['radial_p95']<.05 and radial['multiple_crossing_fraction']==0 and sign['sign_accuracy']>.995
        attempts.append(dict(**initial,preliminary_radial=radial,preliminary_sign=sign,preliminary_valid=valid))
        if valid: selected=field; break
    write_json(CACHE/'initialization_preflight.json',dict(attempts=attempts,
        search='Only K=128 then 256, common radius .60. No multiscale; no residual-based tuning.',
        gates='100% roots; p95 radial error < .05; no multiple radial crossings; sign accuracy > .995.',
        development_note='An early constraint generator used separate Sobol streams for radial and angular coordinates, leaving correlated interior sampling. Replaced BEFORE selection by joint 3D Sobol volume samples. The first provisional 256/.6 fit had 34.375% multiple radial crossings on 512 directions; it is not the selected initialization.'))
    if selected is None: raise RuntimeError('SINGLE_SCALE_INITIALIZATION_FAILED: see isolated preflight')
    field=selected; torch.save(field.state_dict(),CACHE/'initialized_field.pt')
    # Representation diagnostics stay on CPU while the previous experiment uses CUDA.
    d=directions(4096,297); t,valid,radial=radial_roots(field,d,scan_steps=161); roots=d*t[:,None]
    sign=signs(field); mesh=mesh_diagnostic(field,'sphere_initial'); grad=gradient_check(field,roots)
    outside=shell_points(1024,3.,4.,298); ov=field.value(outside); og=field.gradient(outside)
    outside_exact=bool(torch.equal(ov,torch.ones_like(ov)) and torch.equal(og,torch.zeros_like(og)))
    # Choose high/mid/weak magnitude among coordinates with actual surface support.
    pi,bi=field.pairs(roots); supported=torch.unique(bi); order=supported[torch.argsort(field.coefficients[supported].abs(),descending=True)]
    chosen=[int(order[0]),int(order[len(order)//2]),int(order[-1])]; perturbations=[]
    for index in chosen:
        eps=.1*float(field.coefficients.square().mean().sqrt()); variants=[]
        support=(roots-field.layout.centers[index]).norm(dim=1)<field.layout.radii[index]
        normals=field.gradient(roots); normals/=normals.norm(dim=1,keepdim=True)
        for s in (-1,1):
            coeff=field.coefficients.clone(); coeff[index]+=s*eps; f=field.with_coefficients(coeff)
            tt,v,rr=radial_roots(f,d,scan_steps=161); displaced=d*tt[:,None]; n=f.gradient(displaced); n/=n.norm(dim=1,keepdim=True)
            delta=(displaced-roots).norm(dim=1)
            variants.append(dict(sign=s,radial=rr,max_displacement=float(delta.max()),rms_displacement=float(delta.square().mean().sqrt()),
                max_displacement_outside_baseline_support=float(delta[~support].max()) if (~support).any() else 0.,
                affected_direction_fraction=float((delta>1e-7).double().mean()),affected_sphere_area_estimate=float((delta>1e-7).double().mean())*4*np.pi,
                normal_change_rms=float((n-normals).square().sum(1).mean().sqrt())))
            np.save(CACHE/f'perturb_{index}_{s}.npy',displaced.numpy())
        perturbations.append(dict(coordinate=index,coefficient=float(field.coefficients[index]),epsilon=eps,variants=variants))
    # Full coefficient vector perturbation, no tight clamp; fixed K recovery.
    generator=torch.Generator().manual_seed(299); noise=torch.randn(len(field.coefficients),generator=generator,dtype=torch.float64)
    noise=noise/noise.square().mean().sqrt()*.2*field.coefficients.square().mean().sqrt()
    perturbed=field.coefficients+noise; torch.save(field.with_coefficients(perturbed).state_dict(),CACHE/'perturbed_field.pt')
    p,y,w,a,gram=system; eta=initial['ridge']; rhs=(a*w[:,None]).T@((y-1)*w)
    optimum=torch.linalg.solve(gram,rhs); coeff=perturbed; trajectory=[]
    for iteration,alpha in enumerate((0.,.5,.5,1.)):
        coeff=coeff+alpha*(optimum-coeff); f=field.with_coefficients(coeff)
        rr,_,rs=radial_roots(f,directions(1024,301),scan_steps=161)
        err=a@coeff+1-y; objective=float((err*w).square().sum()+eta*coeff.square().sum())
        trajectory.append(dict(iteration=iteration,alpha=alpha,objective=objective,field_mse=float(err.square().mean()),radial=rs))
        np.save(CACHE/f'scalar_recovery_{iteration}.npy',(directions(1024,301)*rr[:,None]).numpy())
    timings=[]; query=shell_points(16384,0,2,302)
    for name,fun in [('value',field.value),('gradient',field.gradient)]:
        begin=time.perf_counter(); fun(query); timings.append(dict(operation=name,seconds=time.perf_counter()-begin,points=len(query),device='cpu'))
    pi,bi=field.pairs(query)
    report=dict(formulation='F(x)=1+sum lambda_k B_k(x); no sphere/grid/base term',background=1.,background_fixed=True,
        INITIAL_K=field.layout.count,COMMON_SUPPORT_RADIUS=float(field.layout.radii[0]),MULTISCALE_BASIS=False,
        initialization=initial,preliminary_attempts=attempts,radial=radial,sign=sign,mesh=mesh,gradient_check=grad,
        outside_support_exact=outside_exact,perturbations=perturbations,
        scalar_recovery=dict(trajectory=trajectory,perturbation='Seed299 Gaussian coefficient noise, RMS .2 times initialized coefficient RMS; no clamp',
            optimizer='Exact regularized least-squares optimum with declared half/half/full steps; all K coefficients optimized',
            passed=trajectory[-1]['objective']<trajectory[0]['objective'] and trajectory[-1]['radial']['radial_mae']<trajectory[0]['radial']['radial_mae']),
        performance=dict(cpu_queries=timings,support_pairs=len(pi),query_points=len(query),average_active_bases_per_point=len(pi)/len(query),
            cpu_rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.perf_counter()-started),
        field_state_path=str(CACHE/'initialized_field.pt'),centers=field.layout.centers.tolist(),coefficients=field.coefficients.tolist(),
        field_digest=digest(field.coefficients),layout_digest=digest(field.layout.centers),
        initialization_valid=radial['root_coverage']==1 and radial['radial_p95']<.05 and sign['sign_accuracy']>.995 and sign['center_value']<0,
        single_dominant_zeroset=mesh['components']==1 and mesh['watertight'] and radial['multiple_crossing_fraction']==0 and mesh['near_zero_degenerate_fraction']==0)
    np.save(CACHE/'initial_roots.npy',roots.numpy())
    write_json(Path('artifacts/v091_sphere_initialization.json'),report)
    print(json.dumps({k:report[k] for k in ('INITIAL_K','COMMON_SUPPORT_RADIUS','radial','sign','mesh','initialization_valid','single_dominant_zeroset')}),flush=True)


if __name__=='__main__': run()
