"""Geometry closure, edge figures and evidence-based v0.8.17 verdicts."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .collision_transfer import CACHE, LAWS, from_micro, transfer
from .transverse_packet import digest, write_json


def diagnostic_field(state):
    from .mesh_field import prepare_stanford_bunny
    from .meshfree_surface import sample_meshfree_zero_set
    from .boundary_transport import enclosing_observation_sphere
    from .corrected_birth import CorrectedBirthConfig, _layout
    from .emitter_scaling import _center_bank
    from .geodesic_source import _eta
    # Only reconstruct the historical scalar grid from the original input.
    # No marching cubes, mesh visibility images, or production sample generation.
    prepared=prepare_stanford_bunny(Path('data/stanford_bunny/cache/bun_zipper.ply'),build_surface_scaffold=False)
    base=prepared.base_field.to(torch.device('cuda'))
    frozen=json.loads(Path('artifacts/v0816_global_sample_convergence.json').read_text())['frozen']
    assert digest(base.grid)==frozen['grid_digest']
    surface=sample_meshfree_zero_set(base,4096,sobol_scramble_seed=101)
    layout,_=_layout(surface.points,CorrectedBirthConfig(dictionary_count=128,initial_count=32))
    context=SimpleNamespace(base=base,master_layout=layout,reference_points=surface.points,reference_normals=surface.normals)
    bank=_center_bank(context,1024)
    boundary=enclosing_observation_sphere(surface.points)
    torch.testing.assert_close(boundary.center.cpu(),state['center'],rtol=0,atol=1e-12)
    return context,boundary,_eta(base,bank['centers']),frozen['packet_radius']


def closure(laws):
    from .fixed_measure import OneCoefficientField, attach, diagnostic_ids
    from .finite_packet import _transmission
    from .density_matrix import capture_splat
    from .measurement_bandwidth import gate,lobe
    from .meshfree_surface import meshfree_base_color
    state=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
    family=torch.load('runs/v0816_global_convergence/transport_16777216.pt',weights_only=True,mmap=True)
    context,boundary,eta,h=diagnostic_field(state)
    epsilons=(1e-3,1e-4,1e-5,1e-6); results=[]; regression=[]
    pool=diagnostic_ids(len(state['weights']),65536,107)
    for parameter in (0,17):
        center=context.master_layout.centers[parameter]; radius=context.master_layout.radii[parameter]
        distance=(state['positions'][pool]-center.cpu()).norm(dim=1)
        p=1-state['transmission'][0,pool]
        affected=distance<radius.cpu(); transition=(p>.02)&(p<.2)
        rng=np.random.default_rng(817+parameter)
        def pick(mask,n):
            ids=pool[mask.numpy()]; return rng.choice(ids,min(n,len(ids)),replace=False)
        owners=np.unique(np.concatenate((pick(affected & transition,256),pick(affected,256),pick(torch.ones(len(pool),dtype=torch.bool),256))))
        points=state['positions'][owners].cuda(); normals=state['normals'][owners].cuda()
        weights=state['weights'][owners].cuda() # original weights, no diagnostic renormalization
        direction=state['directions'][0].cuda(); up=state['up'][0].cuda(); right=state['right'][0].cuda()
        with torch.no_grad():
            x0,n0=attach(context.base,points,normals)
            attachment_error=float((x0-points).abs().max())
            normal_error=float((n0-normals).abs().max())
            # Always run a small CURRENT/cache comparison even when a previous
            # diagnostic law result is resumed from its checkpoint.
            d=direction.expand_as(points[:128])
            old,tau,_,micro=_transmission(context.base,points[:128],d,boundary.exit_times(points[:128],d),radius=h,epsilon=h,path_step=.5*h,
                offsets=d.new_zeros((1,3)),eta=eta,kappa=-math.log(.01),surface_barrier=True,launch_exclusion_factor=1.05,return_micro=True)
            regression.append(dict(parameter=parameter,micro_mean_max_abs=float((tau-micro.mean(-1)).abs().max()),
                cached_tau_max_abs=float((tau.cpu()-family['tau'][0,owners[:128],0]).abs().max())))
        for law in laws:
            path=CACHE/f"closure_p{parameter}_{law['name']}.json"
            if path.exists():
                cached=json.loads(path.read_text())
                if cached['law']==law and cached['sample_ids_digest']==digest(torch.from_numpy(owners)):
                    results.append(cached); continue
            def block(c,p,n,w):
                field=OneCoefficientField(context.base,center,radius,c)
                x,normal=attach(field,p,n); directions=direction.expand_as(x)
                old,tau,_,micro=_transmission(field,x,directions,boundary.exit_times(x,directions),radius=h,epsilon=h,path_step=.5*h,
                    offsets=x.new_zeros((1,3)),eta=eta,kappa=-math.log(.01),surface_barrier=True,launch_exclusion_factor=1.05,return_micro=True)
                t=from_micro(micro,law,tau); cosine=normal@direction
                e=w[:,None]*t[:,None]*gate(cosine,'CURRENT_SOFT_W005')[:,None]*lobe(cosine,'CURRENT_LOBE')[:,None]*meshfree_base_color(x,field.lower,field.upper)
                rel=x-boundary.center; row=(.5-rel@up/2.8)*1080-.5; col=(rel@right/2.8+.5)*1920-.5
                image=x.new_zeros((1080*1920,3)); capture_splat(image,row,col,e,4)
                return image.flatten()*(1.5*1080*1920)
            analytic=np.zeros(1080*1920*3); fds={e:np.zeros_like(analytic) for e in epsilons}
            started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for start in range(0,len(points),128):
                sl=slice(start,start+128)
                function=lambda c:block(c,points[sl],normals[sl],weights[sl])
                zero=torch.tensor(0.,device='cuda',dtype=torch.float64,requires_grad=True)
                _,j=torch.autograd.functional.jvp(function,zero,torch.ones_like(zero))
                analytic+=j.detach().cpu().numpy(); del j
                with torch.no_grad():
                    for eps in epsilons:
                        fds[eps]+=((function(zero+eps)-function(zero-eps))/(2*eps)).cpu().numpy()
                    if law['family']=='CURRENT':
                        d=direction.expand_as(points[sl]); old,tau,_,micro=_transmission(context.base,points[sl],d,boundary.exit_times(points[sl],d),radius=h,epsilon=h,path_step=.5*h,
                            offsets=d.new_zeros((1,3)),eta=eta,kappa=-math.log(.01),surface_barrier=True,launch_exclusion_factor=1.05,return_micro=True)
                        regression.append(dict(parameter=parameter,micro_mean_max_abs=float((tau-micro.mean(-1)).abs().max()),
                            cached_tau_max_abs=float((tau.cpu()-family['tau'][0,owners[sl],0]).abs().max())))
            norm=np.linalg.norm(analytic); active=np.abs(analytic)>norm*1e-6
            assert np.isfinite(analytic).all()
            null=bool(norm<=1e-20 or not active.any())
            rows=[]
            for eps,fd in fds.items():
                b=np.linalg.norm(fd)
                assert np.isfinite(fd).all()
                rows.append(dict(epsilon=eps,relative_l2=float(np.linalg.norm(fd-analytic)/norm) if not null else None,
                    cosine=float(np.dot(fd,analytic)/(b*norm)) if not null and b>1e-30 else None,
                    sign_agreement=float(np.mean(np.sign(fd[active])==np.sign(analytic[active]))) if not null else None,
                    norm_ratio=float(b/norm) if not null else None,fd_norm=float(b)))
            result=dict(law=law,parameter=parameter,samples=len(owners),sample_ids_digest=digest(torch.from_numpy(owners)),
                baseline_reattachment_position_max_abs=attachment_error,baseline_reattachment_normal_max_abs=normal_error,
                selection='Fixed seed; support-intersecting transition packets plus support and uniform controls; identical IDs for all laws',
                views=[0],resolution=[1080,1920],analytic_norm=float(norm),rows=rows,null_analytic_response=null,
                passed=not null and all(r['cosine'] is not None and r['relative_l2']<.05 and r['cosine']>.99 and .95<r['norm_ratio']<1.05 and r['sign_agreement']>.95 for r in rows[-2:]),
                runtime_seconds=time.perf_counter()-started,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            write_json(path,result); results.append(result)
            print(f"Closure {parameter} {law['name']} {rows[-1]}",flush=True)
    return dict(results=results,regression=regression,grid_digest=digest(context.base.grid),eta=eta,h=h,
                note='Only original input mesh-to-scalar-grid reconstruction; grid SHA256 matches frozen production. No marching cubes, mesh readout, or production source regeneration. Diagnostic basis points reproduce historical 4096-point dictionary construction.')


def sensitivity(laws):
    family=torch.load('runs/v0816_global_convergence/transport_16777216.pt',weights_only=True,mmap=True)
    state=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
    tau=family['tau']; results=[]
    for law in laws:
        near_zero=0; transition_zero=0; transition_count=0; active_count=0; active_zero=0; max_derivative=0.; samples=[]
        rng=np.random.default_rng(817)
        for v in range(4):
            for i in range(0,tau.shape[1],131072):
                q=tau[v,i:i+131072,0]; p=(-torch.expm1(-q)).detach().requires_grad_(True)
                t=transfer(p,law); dp=torch.autograd.grad(t.sum(),p)[0]; dtau=dp*torch.exp(-q)
                tiny=dtau.abs()<1e-8; trans=(p>.02)&(p<.2)
                c=state['normals'][i:i+len(q)]@state['directions'][v]; outward=c>0
                near_zero+=int(tiny.sum()); transition_zero+=int((tiny&trans).sum()); transition_count+=int(trans.sum())
                active_count+=int(outward.sum()); active_zero+=int((tiny&outward).sum())
                max_derivative=max(max_derivative,float(dtau.abs().max()))
                ids=rng.choice(len(q),min(128,len(q)),replace=False)
                samples.append(dtau.detach()[ids].abs().numpy())
        results.append(dict(law=law,near_zero_dT_dtau_fraction=near_zero/(4*tau.shape[1]),
            outward_near_zero_fraction=active_zero/active_count,transition_near_zero_fraction=transition_zero/max(transition_count,1),
            max_abs_dT_dtau=max_derivative,abs_dT_dtau_quantiles=np.quantile(np.concatenate(samples),[.5,.9,.99,.999]).tolist(),
            quantile_sampling='128 seeded random IDs per block, seed 817; never stride-decimate Sobol. Fractions and maximum are exact over all packets.',
            caveat='Transfer-chain sensitivity only; zero tau has no collision derivative in a compact shell. Full lambda closure reported separately.'))
    return results


def figures(report,selected):
    import matplotlib.pyplot as plt
    from .dense_reference import load_mse_reference
    reference=load_mse_reference(); paths=['figures/v0817_transfer_curves.png']
    data=[(r['law']['name'],np.load(CACHE/(r['law']['name']+'.npy'))) for r in selected]+[('DENSE HARD',reference)]
    def save(fig,name):
        path=f'figures/v0817_{name}.png'; fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    fig,axes=plt.subplots(len(data),4,figsize=(15,10))
    for i,(name,images) in enumerate(data):
        for v in range(4): axes[i,v].imshow(np.clip(images[v]*.52,0,1)); axes[i,v].set_title(f'{name}, view {v}',fontsize=8); axes[i,v].axis('off')
    save(fig,'fullhd_comparison')
    sites=report['reference_profiles'][::2]; fig,axes=plt.subplots(len(sites),len(data),figsize=(12,12))
    for i,s in enumerate(sites):
        y,x=s['row'],s['column']
        for j,(name,images) in enumerate(data):
            axes[i,j].imshow(np.clip(images[0,max(0,y-64):y+64,max(0,x-64):x+64]*.52,0,1)); axes[i,j].set_title(f"{s['region']}\n{name}",fontsize=8); axes[i,j].axis('off')
    save(fig,'matched_edge_crops')
    fig,axes=plt.subplots(5,2,figsize=(12,15))
    for k,ax in enumerate(axes.flat):
        for r in selected: ax.plot(r['profiles'][k]['distance'],r['profiles'][k]['profile'],label=r['law']['name'])
        p=report['reference_profiles'][k]; ax.plot(p['distance'],p['profile'],'k--',label='dense hard')
        ax.set_title(p['region']); ax.set_xlabel('Normal distance (pixels)'); ax.legend(fontsize=7)
    save(fig,'edge_profiles')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in selected:
        b=r['blocking_bins']; counts=[sum(x['count'] for x in b if x['lower']==lo) for lo in (0,.01,.02,.05,.1,.2,.5)]
        energies=[sum(x['detected_rgb_energy'] for x in b if x['lower']==lo) for lo in (0,.01,.02,.05,.1,.2,.5)]
        axes[0].plot(range(7),counts,'o-',label=r['law']['name']); axes[1].plot(range(7),energies,'o-',label=r['law']['name'])
    for ax in axes: ax.set_xticks(range(7),['0-.01','.01-.02','.02-.05','.05-.1','.1-.2','.2-.5','.5-1'],rotation=30); ax.legend(fontsize=7); ax.set_yscale('log')
    axes[0].set_ylabel('Packet count'); axes[1].set_ylabel('Detected RGB (sensor scaled)'); save(fig,'blocking_bins')
    return paths


def finalize():
    from .high_sample import _write_csv
    from .finite_packet import _scalar_csv_rows
    torch.set_num_threads(4); started=time.perf_counter()
    report=json.loads(Path('artifacts/v0817_collision_transfer.json').read_text()); rows=report['rows']
    from .collision_transfer import edge_locations,edge_profiles
    from .dense_reference import load_mse_reference
    reference=load_mse_reference(); sites=edge_locations(reference)
    report['reference_profiles']=edge_profiles(reference,sites)
    for row in rows:
        row['profiles']=edge_profiles(np.load(CACHE/(row['law']['name']+'.npy'),mmap_mode='r'),sites)
        for p,r in zip(row['profiles'],report['reference_profiles']):
            p['edge_offset_vs_reference']=p['location_50']-r['location_50'] if p['location_50'] is not None and r['location_50'] is not None else None
            p['width_error_vs_reference']=abs(p['width_10_90']-r['width_10_90']) if p['width_10_90'] is not None and r['width_10_90'] is not None else None
            p['contrast_ratio_vs_reference']=p['contrast']/max(r['contrast'],1e-30)
            p['valid_edge']=p['contrast_ratio_vs_reference']>=.1 and p['width_10_90'] is not None and p['width_10_90']>=0
    selected=[rows[0]]+[min((r for r in rows if r['law']['family']==family),key=lambda r:r['whole_image_mse']) for family in ('POWER_SHARP','SIGMOID_SHARP')]
    path=CACHE/'closure.json'
    if path.exists(): fd=json.loads(path.read_text())
    else: fd=closure([r['law'] for r in selected]); write_json(path,fd)
    report['geometry_closure']=fd
    report['sensitivity']=sensitivity([r['law'] for r in rows])
    current=rows[0]; best=min(rows[1:],key=lambda r:r['whole_image_mse'])
    def average(row,key):
        values=[p[key] for p in row['profiles'] if p[key] is not None and p['valid_edge']]
        return float(np.mean(np.abs(values))) if values else None
    for r in rows:
        r['mean_edge_width_error']=average(r,'width_error_vs_reference'); r['mean_abs_edge_offset']=average(r,'edge_offset_vs_reference')
        r['valid_edge_profile_count']=sum(p['valid_edge'] for p in r['profiles'])
        r['missing_or_collapsed_edge_profile_count']=len(r['profiles'])-r['valid_edge_profile_count']
        r['gradient_relative_l2']=float(np.mean([v['gradient_relative_l2'] for v in r['per_view']]))
        if 'total_image_energy' in r: r['squared_image_energy']=r.pop('total_image_energy')
    mse=best['whole_image_mse']<current['whole_image_mse']*.95
    grad=best['gradient_cosine']>current['gradient_cosine']+.02
    interior=best['interior_gt8px_mse']<=current['interior_gt8px_mse']*1.05
    edge=(best['valid_edge_profile_count']>=current['valid_edge_profile_count']
        and best['mean_edge_width_error'] is not None and best['mean_abs_edge_offset'] is not None
        and best['mean_edge_width_error']<current['mean_edge_width_error']*.9
        and best['mean_abs_edge_offset']<=current['mean_abs_edge_offset']+.25)
    results=[r for r in fd['results'] if r['law']['name']==best['law']['name']]
    closure_ok=len(results)==2 and all(r['passed'] for r in results)
    current_norms={r['parameter']:r['analytic_norm'] for r in fd['results'] if r['law']['family']=='CURRENT'}
    norm_ratios={str(r['parameter']):r['analytic_norm']/current_norms[r['parameter']] for r in results}
    usable=len(norm_ratios)==2 and all(.1<v<10 for v in norm_ratios.values())
    nearzero=float(transfer(torch.tensor(.2,dtype=torch.float64),best['law']))<=.01
    preferred=mse and grad and interior and edge and closure_ok and usable
    report['verdicts']=dict(CURRENT_PARTIAL_COLLISION_ATTENUATION_IS_TOO_SOFT=preferred,
        SHARP_COLLISION_TRANSFER_REDUCES_FULLHD_MSE=mse,
        SHARP_COLLISION_TRANSFER_IMPROVES_EDGE_ALIGNMENT=edge,
        SHARP_COLLISION_TRANSFER_IMPROVES_SPATIAL_GRADIENT_ALIGNMENT=grad,
        SHARP_COLLISION_TRANSFER_PRESERVES_INTERIOR_FIDELITY=interior,
        SHARP_COLLISION_TRANSFER_PRESERVES_LAMBDA_FD_CLOSURE=closure_ok,
        TWENTY_PERCENT_BLOCKING_EFFECTIVELY_ZEROES_PACKET=nearzero,
        SHARP_COLLISION_TRANSFER_IS_PREFERRED_FORWARD_OPERATOR=preferred,
        BEST_TRANSFER_LAW=min(rows,key=lambda r:r['whole_image_mse'])['law'])
    report['best_sharp_candidate']=best['law']
    report['best_per_family']={r['law']['family']:r['law'] for r in selected}
    report['selection_note']='Best means minimum whole-image MSE among tested sharp laws; promotion requires ALL fidelity and closure gates. Unpassed hypothesis is not proof attenuation never matters. Geometric 20% blockage was not measured.'
    report['input_caches']=['runs/v0816_global_convergence/source_16777216.pt','runs/v0816_global_convergence/transport_16777216.pt','runs/v0815_dense_reference/reference.npy']
    report['production_default_changed']=False
    report['geometry_sensitivity_usable']=usable
    report['best_lambda_norm_ratios_vs_current']=norm_ratios
    report['criteria']['lambda_norm_ratio_bounds']=[.1,10.]
    report['criteria']['edge_minimum_reference_contrast_ratio']=.1
    report['blocking_bin_note']='Detected RGB bins are pre-boundary conservative splat energies including unchanged gate/lobe/gain; their sum is checked against actual detector RGB to 1e-5. Silhouette contribution uses projected-center membership, not full footprint decomposition.'
    report['profile_note']='View 0; ten deterministic reference-only sites in five spatial/anatomical regions. Normals from sigma=1 reference gradient; profiles themselves are unsmoothed bilinear samples. Contrast uses far -12:-9 and +9:+12 px plateaus. Multiple crossings choose nearest site; overshoot is measured relative to these local plateaus, not proof of ringing.'
    report['profile_roi_audit']='Reference-only annotation showed initial broad upper/small-ear regions picked neck/ear-junction sites. Tightened spatial ROIs to the actual long appendage and small ear before interpreting profiles; all laws use identical corrected sites. Full-HD images and whole-image metrics were not changed.'
    report['profile_validity_audit']='A near-zero remaining edge can have a normalized 10-90 width despite being extinguished. Raw profiles/widths are retained; summary widths/offsets require >=10% of reference local contrast and valid crossings, with missing counts explicit. This QC cannot promote a vanished edge; alignment had already failed before this guard.'
    report['figures']=figures(report,selected)
    report['diagnostics_runtime_seconds']=time.perf_counter()-started
    write_json(Path('artifacts/v0817_collision_transfer.json'),report)
    _write_csv(Path('artifacts/v0817_collision_transfer.csv'),_scalar_csv_rows(report))
    print(json.dumps(report['verdicts'],indent=2),flush=True)


if __name__=='__main__': finalize()
