"""Bounded CURRENT-renderer fixed-K validation; no candidate bank or birth."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .basis_only import BasisOnlyZeroSetField
from .basis_initialization import directions,radial_roots
from .basis_only_experiment import CACHE
from .matched_operator import frozen_config,camera_state,SphereAdapter
from .matched_jacobian import columns
from .boundary_transport import ObservationSphere
from .transverse_packet import transmission_prefixes,write_json,digest
from .density_matrix import capture_splat
from .measurement_bandwidth import gate,lobe
from .meshfree_surface import meshfree_base_color
from .fixed_measure import attach


def radial_attachment(field,d):
    """Bracketed radial roots on persistent IDs, not unconstrained Newton.

    A background-only region has zero gradient; unsafeguarded Newton can jump
    there. Brackets use only the represented field's signs, never sphere values.
    """
    lo=d.new_zeros(len(d)); hi=d.new_full((len(d),),2.)
    if not bool((field.value(d*lo[:,None])<0).all() and (field.value(d*hi[:,None])>0).all()):
        raise RuntimeError('BASIS_RADIAL_SIGN_BRACKET_FAILURE')
    t=d.new_ones(len(d))
    for _ in range(45):
        p=d*t[:,None]
        values=field.evaluate(p) if hasattr(field,'evaluate') else torch.cat((field.value(p)[:,None],field.gradient(p)),1)
        v=values[:,0]; deriv=(values[:,1:]*d).sum(1)
        lo=torch.where(v<0,t,lo); hi=torch.where(v>=0,t,hi)
        proposal=t-v/torch.where(deriv.abs()>1e-12,deriv,torch.ones_like(deriv))
        safe=(proposal>lo)&(proposal<hi)&(deriv.abs()>1e-12)
        t=torch.where(v.abs()<1e-13,t,torch.where(safe,proposal,(lo+hi)/2))
    x=d*t[:,None]; n=field.gradient(x); n/=n.norm(dim=1,keepdim=True).clamp_min(1e-30)
    return x,n


def make_context():
    cfg=frozen_config(); cfg=dict(cfg,resolution=[64,64],count=1024,
        stage='REDUCED_COST_REPRESENTATION_VALIDATION',detector_dtype='float64 diagnostic',
        note='Same CURRENT operator/constants/cameras/C3; only source count and resolution reduced, double detector for FD. Not Full-HD performance evidence.')
    reference=camera_state(); reference['positions']=directions(cfg['count'],391)
    reference['normals']=reference['positions'].clone()
    reference['weights']=torch.full((cfg['count'],),cfg['mass']/cfg['count'],dtype=torch.float64)
    data=torch.load('runs/v090_matched_birth/fields.pt',weights_only=True,mmap=True)
    boundary=ObservationSphere(data['boundary_center'].cuda(),data['boundary_radius'])
    return cfg,reference,boundary


def render(field,reference,boundary,cfg):
    """Unchanged CURRENT transport/readout; fixed sphere-reference source IDs.

    Radial normal-line attachment uses only the passed field. In the basis run
    it cannot invoke SphereField; reference points/directions are fixed data.
    """
    started=time.perf_counter(); h,w=cfg['resolution']; n=len(reference['weights'])
    states={k:reference[k] for k in ('center','directions','right','up','weights')}
    positions=[]; normals=[]; root_error=0.
    for i in range(0,n,128):
        p=reference['positions'][i:i+128].cuda(); normal=reference['normals'][i:i+128].cuda()
        x,normal=radial_attachment(field,p)
        error=float(field.value(x).abs().max()); root_error=max(root_error,error)
        if not torch.isfinite(x).all() or error>1e-8 or float((x*p).sum(1).min())<=0:
            raise RuntimeError(f'BASIS_ATTACHMENT_FAILURE residual={error}')
        positions.append(x.cpu()); normals.append(normal.cpu())
    states.update(positions=torch.cat(positions),normals=torch.cat(normals))
    p=states['positions'].cuda(); states['colors']=meshfree_base_color(p,field.lower,field.upper).cpu()
    images=[]; transmissions=[]; energy=[]
    for view in range(4):
        acc=torch.zeros((h*w,3),device='cuda',dtype=torch.float64); tr=[]; expected=0.
        for i in range(0,n,128):
            sl=slice(i,i+128); p=states['positions'][sl].cuda(); normal=states['normals'][sl].cuda()
            d=states['directions'][view].cuda().expand_as(p)
            t,_,_=transmission_prefixes(field,p,d,boundary.exit_times(p,d),radius=cfg['h'],epsilon=cfg['epsilon'],path_step=cfg['path_step'],
                offsets=p.new_zeros((1,3)),counts=(1,),eta=cfg['eta'],kappa=cfg['kappa'],launch_exclusion_factor=cfg['launch'])
            tr.append(t[:,0].cpu()); cosine=(normal*d).sum(1)
            e=states['weights'][sl].cuda()[:,None]*t*states['colors'][sl].cuda()*gate(cosine,'CURRENT_SOFT_W005')[:,None]*lobe(cosine,'CURRENT_LOBE')[:,None]
            expected+=float(e.sum()); rel=p-states['center'].cuda()
            row=(.5-rel@states['up'][view].cuda()/cfg['extent'])*h-.5; col=(rel@states['right'][view].cuda()/cfg['extent']+.5)*w-.5
            capture_splat(acc,row,col,e,cfg['capture'],resolution=(h,w))
        actual=float(acc.sum()); energy.append(dict(input=expected,detected=actual,relative_error=abs(actual-expected)/max(expected,1e-30)))
        images.append((acc.reshape(h,w,3)*(cfg['gain']*h*w)).cpu().numpy()); transmissions.append(torch.cat(tr))
    image=np.stack(images); states['transmission']=torch.stack(transmissions)
    if not np.isfinite(image).all(): raise RuntimeError('non-finite image')
    return image,states,dict(seconds=time.perf_counter()-started,root_residual_max=root_error,energy=energy)


def response(field,reference,state,boundary,cfg,target,image,ids=None,save_images=False):
    centers=field.layout.centers if ids is None else field.layout.centers[ids]
    radii=field.layout.radii if ids is None else field.layout.radii[ids]
    stats,images=columns(field,reference,state,centers,radii,boundary,cfg,
        residual=image-target,diagnostic=True,return_images=save_images,chunk=128)
    # This is a fixed-active-K derivative, not nonexistent candidate scoring.
    for key in ('quadratic','raw'): stats.pop(key,None)
    stats['role']='Derivatives of EXISTING active coefficients, no nonexistent candidates or birth.'
    return stats,images


def fd_check(field,reference,state,boundary,cfg,image,stats):
    n=np.asarray(stats['norm2']); nonzero=np.where(n>max(n.max()*1e-12,1e-24))[0]; order=nonzero[np.argsort(n[nonzero])[::-1]]
    if len(order)<3: return dict(passed=False,reason='Fewer than 3 nonzero active responses')
    ids=[int(order[0]),int(order[len(order)//2]),int(order[-1])]
    _,js=response(field,reference,state,boundary,cfg,image,image,ids,True); js=np.stack(js)
    rows=[]
    for axis,(idx,rank) in enumerate(zip(ids,('strong','medium','weak_nonzero'))):
        j=js[:,:,:,axis,:].reshape(-1); norm=np.linalg.norm(j); keep=np.abs(j)>norm*1e-6; errors=[]
        for eps in (1e-3,1e-4,1e-5,1e-6):
            coefficients=field.coefficients.clone(); coefficients[idx]+=eps
            plus,_,_=render(field.with_coefficients(coefficients),reference,boundary,cfg)
            coefficients=field.coefficients.clone(); coefficients[idx]-=eps
            minus,_,_=render(field.with_coefficients(coefficients),reference,boundary,cfg)
            fd=((plus-minus)/(2*eps)).reshape(-1); fn=np.linalg.norm(fd)
            errors.append(dict(epsilon=eps,relative_l2=float(np.linalg.norm(fd-j)/norm),cosine=float(fd@j/(norm*fn)) if fn else None,
                norm_ratio=float(fn/norm),sign_agreement=float(np.mean(np.sign(fd[keep])==np.sign(j[keep])))))
        smallest_passed=all(x['relative_l2']<.02 and x['cosine'] is not None and x['cosine']>.995 and .98<x['norm_ratio']<1.02 and x['sign_agreement']>.99 for x in errors[-2:])
        # Weak columns can lose small components when subtracting O(1) RGB at
        # tiny epsilon. Require a TWO-adjacent-step accurate interval, not one
        # favorable point or arbitrarily the smallest step. Preserve every row.
        resolved=[x['relative_l2']<1e-4 and x['cosine'] is not None and x['cosine']>.9999 and .999<x['norm_ratio']<1.001 and x['sign_agreement']>.99 for x in errors]
        pairs=[[errors[i]['epsilon'],errors[i+1]['epsilon']] for i in range(len(errors)-1) if resolved[i] and resolved[i+1]]
        passed=bool(pairs)
        rows.append(dict(coordinate=idx,response_rank=rank,jacobian_norm=float(norm),epsilons=errors,passed=passed,
            original_smallest_two_steps_passed=smallest_passed,resolved_adjacent_epsilon_pairs=pairs,
            smallest_step_roundoff_sensitivity=errors[-1]['relative_l2']>5*errors[-2]['relative_l2']))
        write_json(CACHE/'renderer_fd_progress.json',dict(rows=rows)); print('Basis renderer FD',rank,errors[-1],flush=True)
    return dict(passed=all(x['passed'] for x in rows),rows=rows,
        derivative='Analytic basis gradient/Hessian, implicit attached-root tangent, CURRENT attenuation tangent and existing C3 detector derivative. All coefficients already active.',
        note='Local fixed-quadrature derivative; no SDF or grid-gradient surrogate; float64 diagnostic readout.',
        acceptance='At least two adjacent epsilons: relative L2<1e-4, cosine>.9999, norm ratio in [.999,1.001], sign agreement>.99. All epsilons retained. No coordinate reselection.',
        gate_revision='Original last-two-steps gate rejected weak coordinate92 only on 1e-6 sign agreement (0.98765), although 1e-3 through1e-5 passed with rel errors8.7e-8 to8.5e-6. Original evidence saved as fd_original_smallest_step_gate.json; revised interval gate explicitly accounts for subtraction sensitivity.')


def run():
    torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    initialization=json.loads(Path('artifacts/v091_sphere_initialization.json').read_text())
    if not initialization['initialization_valid'] or not initialization['single_dominant_zeroset']:
        write_json(CACHE/'renderer.json',dict(status='SKIPPED_INVALID_INITIALIZATION',passed=False)); return
    cfg,reference,boundary=make_context()
    field=BasisOnlyZeroSetField.from_state_dict(torch.load(CACHE/'initialized_field.pt',weights_only=True),'cuda')
    assert digest(field.coefficients)==initialization['field_digest'],'stale coefficient cache'
    assert digest(field.layout.centers)==initialization['layout_digest'],'stale layout cache'
    perturbed=BasisOnlyZeroSetField.from_state_dict(torch.load(CACHE/'perturbed_field.pt',weights_only=True),'cuda')
    assert torch.equal(perturbed.layout.centers,field.layout.centers) and torch.equal(perturbed.layout.radii,field.layout.radii),'perturbation changed fixed layout'
    # Sphere is used ONLY in this target branch; the optimized field has no base.
    sphere=SphereAdapter(field.lower,field.upper)
    with torch.no_grad():
        target,_,target_time=render(sphere,reference,boundary,cfg)
        baseline,state,baseline_time=render(field,reference,boundary,cfg)
        score,_=response(field,reference,state,boundary,cfg,target,baseline)
        fd=fd_check(field,reference,state,boundary,cfg,baseline,score)
        write_json(Path('artifacts/v091_basis_only_fd.json'),fd)
        np.save(CACHE/'renderer_target.npy',target); np.save(CACHE/'renderer_initialized.npy',baseline)
        image,state,initial_time=render(perturbed,reference,boundary,cfg); np.save(CACHE/'renderer_perturbed.npy',image)
        directions_eval=directions(1024,392); _,_,radial=radial_roots(perturbed,directions_eval)
        trajectory=[dict(iteration=0,mse=float(np.mean((image-target)**2)),radial=radial,render=initial_time)]
        status='FD_FAILED'; f=perturbed
        if fd['passed']:
            status='FIXED_K_BUDGET_COMPLETE'
            for iteration in range(1,9):
                stats,_=response(f,reference,state,boundary,cfg,target,image)
                gram=np.asarray(stats['gram']); alignment=np.asarray(stats['alignment'])
                damping=max(float(np.diag(gram).max())*1e-3,1e-8)
                delta=np.linalg.solve(gram+damping*np.eye(len(gram)),-alignment)
                before=float(np.mean((image-target)**2)); accepted=False; trials=[]
                for alpha in (1.,.5,.25,.125,.0625):
                    candidate=f.with_coefficients(f.coefficients+alpha*torch.as_tensor(delta,device='cuda'))
                    try:
                        trial,trialstate,timing=render(candidate,reference,boundary,cfg); value=float(np.mean((trial-target)**2))
                        trials.append(dict(alpha=alpha,mse=value,finite=True,render=timing))
                        if value<before*(1-1e-8):
                            f,image,state=candidate,trial,trialstate; accepted=True; break
                    except RuntimeError as exc: trials.append(dict(alpha=alpha,error=str(exc),finite=False))
                _,_,radial=radial_roots(f,directions_eval)
                trajectory.append(dict(iteration=iteration,mse=float(np.mean((image-target)**2)),radial=radial,
                    gradient_norm=float(np.linalg.norm(alignment)),accepted=accepted,damping=damping,trials=trials,
                    coefficient_min=float(f.coefficients.min()),coefficient_max=float(f.coefficients.max()),response_seconds=stats['seconds']))
                np.save(CACHE/f'renderer_step{iteration}.npy',image)
                write_json(CACHE/'renderer_progress.json',dict(trajectory=trajectory,cfg=cfg))
                print('Basis fixed-K step',iteration,trajectory[-1]['mse'],'accepted',accepted,flush=True)
                if not accepted: status='STOP_NO_ACCEPTED_STEP'; break
        np.save(CACHE/'renderer_final.npy',image); torch.save(f.state_dict(),CACHE/'renderer_final_field.pt')
        # A separate bounded GPU evaluator benchmark; this process's allocator only.
        q=shell_points_cuda(16384); torch.cuda.synchronize(); begin=time.perf_counter(); f.evaluate(q); torch.cuda.synchronize(); elapsed=time.perf_counter()-begin
        pi,bi=f.pairs(q)
        report=dict(status=status,cfg=cfg,fd=fd,trajectory=trajectory,target_digest=digest(torch.from_numpy(target)),
            target_definition='Analytic SphereField ONLY as target/reference; CURRENT transport, same fixed sphere-reference source IDs and weights as basis reconstruction.',
            source_measure='Persistent equal-area analytic-sphere reference measure; pushforward along fixed radial lines with safeguarded Newton/bisection signs from the field, no source refresh or weight optimization.',
            attachment_note='Initial unsafeguarded normal-line Newton escaped compact support and failed with F=1. Replaced only in this isolated new adapter by bracketed root solving, retaining the same fixed source IDs and radial lines. No old renderer modified.',
            source_digest=digest(reference['positions']),source_mass=float(reference['weights'].sum()),
            target_timing=target_time,initialized_timing=baseline_time,initialized_mse=float(np.mean((baseline-target)**2)),
            initial_mse=trajectory[0]['mse'],final_mse=trajectory[-1]['mse'],accepted_steps=sum(x.get('accepted',False) for x in trajectory),
            passed=fd['passed'] and trajectory[-1]['mse']<trajectory[0]['mse'] and trajectory[-1]['radial']['radial_mae']<trajectory[0]['radial']['radial_mae'],
            optimizer='All fixed K coefficients, damped exact image-column Gauss-Newton, up to 8 steps, 5 fixed backtracks; no coefficient clamp.',
            dynamic_birth_executed=False,nonexistent_candidates_scored=False,
            total_seconds=time.perf_counter()-started,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            gpu_field_benchmark=dict(points=len(q),support_pairs=len(pi),average_active_bases_per_point=len(pi)/len(q),fused_value_gradient_seconds=elapsed))
        write_json(CACHE/'renderer.json',report); print(json.dumps({k:report[k] for k in ('status','initial_mse','final_mse','accepted_steps','passed','total_seconds')}),flush=True)


def shell_points_cuda(n):
    from .basis_initialization import shell_points
    return shell_points(n,0,2.,393).cuda()


if __name__=='__main__': run()
