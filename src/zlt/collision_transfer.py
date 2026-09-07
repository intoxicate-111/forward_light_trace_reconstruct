"""v0.8.17: transfer-only ablation of frozen scalar collision evidence.

No mesh rendering, source generation, detector changes, or optimization.
With production M=1, p is opacity, NOT a measured geometric blocking fraction.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter
from .density_matrix import render
from .dense_reference import load_mse_reference, relative_change
from .finite_packet import _image_metrics, _scalar_csv_rows
from .measurement_bandwidth import masks, gate, lobe
from .polar_aliasing import artifact_metrics, project
from .transverse_packet import digest, write_json

CACHE = Path('runs/v0817_collision_transfer')
LAWS = [dict(name='CURRENT', family='CURRENT')]
LAWS += [dict(name=f'POWER_Q{q}', family='POWER_SHARP', q=q, p0=.1,
              alpha=math.log(2)) for q in (2, 4, 8)]
LAWS += [dict(name=f'SIGMOID_S{s:g}', family='SIGMOID_SHARP', pc=.1, s=s)
         for s in (.02, .01)]
BINS = (0., .01, .02, .05, .1, .2, .5, 1.)


def transfer(p, law):
    """CURRENT(p)=1-p is only the M=1 transfer-curve interpretation.

    Runtime CURRENT always uses exp(-historical_tau), not 1-p cancellation.
    """
    if law['family'] == 'CURRENT': return 1-p
    if law['family'] == 'POWER_SHARP':
        return torch.exp(-law['alpha'] * (p/law['p0'])**law['q'])
    if law['family'] == 'SIGMOID_SHARP':
        return torch.sigmoid((law['pc']-p)/law['s']) / torch.sigmoid(p.new_tensor(law['pc']/law['s']))
    raise ValueError(law)


def from_micro(tau_micro, law, current_tau=None):
    p = (-torch.expm1(-tau_micro)).mean(-1)
    if law['family'] == 'CURRENT':
        return torch.exp(-(tau_micro.mean(-1) if current_tau is None else current_tau))
    return transfer(p, law)


def curves():
    import matplotlib.pyplot as plt
    p = torch.linspace(0, 1, 10001, dtype=torch.float64, requires_grad=True)
    probes = torch.tensor([0,.01,.02,.05,.1,.2,.3,.5,1.], dtype=torch.float64, requires_grad=True)
    rows=[]; fig, axes=plt.subplots(1,2,figsize=(11,4))
    for law in LAWS:
        t=transfer(p,law); d=torch.autograd.grad(t.sum(),p)[0]
        tp=transfer(probes,law); dp=torch.autograd.grad(tp.sum(),probes)[0]
        accepted=bool(torch.isfinite(t).all() and torch.isfinite(d).all()
            and tp[2]>=.95 and tp[5]<=.01 and d.abs().max()<=100)
        rows.append(dict(**law,accepted_for_render=accepted or law['family']=='CURRENT',
            semantic_gate_passed=accepted, p=probes.detach().tolist(), transmission=tp.detach().tolist(),
            derivative=dp.detach().tolist(), max_abs_derivative=float(d.abs().max())))
        axes[0].plot(p.detach(),t.detach(),label=law['name']); axes[1].plot(p.detach(),d.detach())
    for ax in axes: ax.set_xlim(0,.3); ax.set_xlabel('p: mean micro opacity (not geometric volume fraction)'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Transmission'); axes[1].set_ylabel('dT/dp'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figures/v0817_transfer_curves.png',dpi=150); plt.close(fig)
    write_json(CACHE/'curves.json',rows)
    return rows


def blocking_stats(state,tau,law,regions):
    rows=[]; total_input=0.
    for v in range(4):
        counts=np.zeros(7,dtype=np.int64); mass=np.zeros(7); tsum=np.zeros(7); energy=np.zeros(7); edge=np.zeros(7)
        for i in range(0,len(state['weights']),131072):
            sl=slice(i,i+131072); p=-torch.expm1(-tau[v,sl,0]); t=state['transmission'][v,sl]
            ids=torch.bucketize(p,p.new_tensor(BINS[1:-1]),right=True)
            c=state['normals'][sl]@state['directions'][v]
            w=state['weights'][sl]; color=state['colors'][sl].sum(1)
            before=w*color*gate(c,'CURRENT_SOFT_W005')*lobe(c,'CURRENT_LOBE')
            total_input+=float((w*color).sum())
            coords=np.rint(project(state['positions'][sl].numpy(),state,v)).astype(int)
            inside=(coords[:,0]>=0)&(coords[:,0]<1080)&(coords[:,1]>=0)&(coords[:,1]<1920)
            near=np.zeros(len(coords),dtype=bool); near[inside]=regions[v]['silhouette'][coords[inside,0],coords[inside,1]]
            for k in range(7):
                selected=ids==k; counts[k]+=int(selected.sum()); mass[k]+=float(w[selected].sum())
                tsum[k]+=float(t[selected].sum()); energy[k]+=float((before*t)[selected].sum())
                edge[k]+=float((before*t)[selected & torch.from_numpy(near)].sum())
        for k in range(7):
            rows.append(dict(view=v,lower=BINS[k],upper=BINS[k+1],count=int(counts[k]),
                packet_fraction=float(counts[k]/len(state['weights'])),source_mass=float(mass[k]),
                mean_transmission=float(tsum[k]/counts[k]) if counts[k] else None,
                detected_rgb_energy=float(energy[k]*1.5*1080*1920),
                silhouette_center_energy=float(edge[k]*1.5*1080*1920)))
    return rows,total_input


def edge_locations(reference):
    """Fixed reference-only silhouette sites, never chosen by sharp-law results.

    Four anatomical regions in view 0, two sites each. Select strongest
    smoothed reference gradient in foreground's 1-4 px boundary band.
    """
    from scipy.ndimage import distance_transform_edt
    r=reference[0].mean(2).astype(float); fg=r>r.max()*.02
    yx=np.argwhere(fg); lo=yx.min(0); span=yx.max(0)-lo
    groups=[('upper_long_ear',(0,.18,.7,1)),('front_small_ear',(.18,.30,.6,.9)),
            ('left_torso',(.50,.8,0,.3)),('right_torso',(.50,.8,.6,1)),
            ('lower_pink_base',(.85,1.,0,1))]
    smooth=gaussian_filter(r,1); gy,gx=np.gradient(smooth); mag=np.hypot(gy,gx)
    dist=distance_transform_edt(fg); sites=[]
    yy,xx=np.indices(r.shape)
    for name,(a,b,c,d) in groups:
        sel=(yy>=lo[0]+a*span[0])&(yy<lo[0]+b*span[0])&(xx>=lo[1]+c*span[1])&(xx<lo[1]+d*span[1])&(dist>=1)&(dist<=4)
        score=np.where(sel,mag,0)
        for n in range(2):
            y,x=np.unravel_index(score.argmax(),score.shape)
            norm=math.hypot(gy[y,x],gx[y,x]); assert norm>0
            sites.append(dict(region=name,row=int(y),column=int(x),normal=[float(gy[y,x]/norm),float(gx[y,x]/norm)]))
            score[max(0,y-30):y+31,max(0,x-30):x+31]=0
    return sites


def edge_profiles(images,sites):
    distance=np.linspace(-12,12,193); rows=[]
    for site in sites:
        coords=np.array([[site['row']+distance*site['normal'][0]],[site['column']+distance*site['normal'][1]]]).reshape(2,-1)
        value=map_coordinates(images[0].mean(2).astype(float),coords,order=1,mode='nearest')
        low=float(np.mean(value[:24])); high=float(np.mean(value[-24:])); contrast=high-low
        normalized=(value-low)/max(contrast,1e-15)
        crossings=[]
        for level in (.1,.5,.9):
            idx=np.where((normalized[:-1]<level)&(normalized[1:]>=level))[0]
            if not len(idx): crossings.append(None); continue
            j=idx[np.argmin(np.abs(distance[idx]))]
            crossings.append(float(distance[j]+(level-normalized[j])/(normalized[j+1]-normalized[j])*.125))
        width=crossings[2]-crossings[0] if None not in crossings else None
        rows.append(dict(**site,low=low,high=high,contrast=contrast,width_10_90=width,
            location_50=crossings[1],peak_gradient=float(np.max(np.gradient(value,distance))),
            overshoot_fraction=float(max(0,normalized.max()-1)),distance=distance.tolist(),profile=value.tolist()))
    return rows


def run_experiment():
    from .high_sample import _write_csv
    torch.set_num_threads(4); CACHE.mkdir(parents=True,exist_ok=True)
    manifest=CACHE/'historical_hashes.json'
    if not manifest.exists(): write_json(manifest,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for f in ('artifacts','figures') for p in Path(f).iterdir() if p.is_file() and 'v0817' not in p.name})
    started=time.perf_counter(); curve_rows=curves()
    state=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
    family=torch.load('runs/v0816_global_convergence/transport_16777216.pt',weights_only=True,mmap=True)
    tau=family['tau']; assert tau.shape==(4,16777216,1)
    reference=load_mse_reference(); regions=masks(reference); sites=edge_locations(reference)
    ref_profiles=edge_profiles(reference,sites)
    frozen={k:digest(state[k]) for k in ('positions','normals','weights','colors','directions','right','up','center')}
    frozen.update(tau_digest=digest(tau),reference_digest=digest(torch.from_numpy(reference)),
                  transport_settings=family['settings'],source_mass=float(state['weights'].sum()))
    rows=[]
    for law,curve in zip(LAWS,curve_rows):
        if not curve['accepted_for_render']: continue
        print('Rendering '+law['name'],flush=True)
        # New storage: never mutate memory-mapped historical state/caches.
        t=torch.empty(tau.shape[:2],dtype=tau.dtype)
        for v in range(4):
            for i in range(0,t.shape[1],131072):
                sl=slice(i,i+131072); t[v,sl]=from_micro(tau[v,sl],law,tau[v,sl,0])
        work=dict(state,transmission=t)
        if law['family']=='CURRENT':
            assert torch.equal(t,state['transmission'])
        path=CACHE/(law['name']+'.npy'); meta=CACHE/(law['name']+'_render.json')
        signature=dict(law=law,frozen=frozen)
        info=json.loads(meta.read_text()) if meta.exists() else {}
        recorded=info.get('signature')
        if recorded is None and Path('artifacts/v0817_collision_transfer.json').exists():
            previous=json.loads(Path('artifacts/v0817_collision_transfer.json').read_text())
            old=next((r for r in previous['rows'] if r['law']['name']==law['name']),None)
            if old is not None: recorded=dict(law=old['law'],frozen=previous['frozen'])
        if path.exists() and meta.exists() and recorded==signature:
            images=np.load(path); accounting,timing=info['accounting'],info['timing']
        else:
            images,occupancy,accounting,timing=render(work,4); del occupancy
            images=np.stack(images); np.save(path,images)
            write_json(meta,dict(accounting=accounting,timing=timing,signature=signature))
        assert np.isfinite(images).all()
        row=dict(law=law,**_image_metrics(list(images),list(reference)),**artifact_metrics(images,reference))
        for v in range(4):
            row['per_view'][v].update(_image_metrics([images[v]],[reference[v]]))
            g=np.stack(np.gradient(images[v].mean(2).astype(float))); r=np.stack(np.gradient(reference[v].mean(2).astype(float)))
            row['per_view'][v]['gradient_relative_l2']=float(np.linalg.norm(g-r)/np.linalg.norm(r))
        row['blocking_bins'],input_rgb=blocking_stats(work,tau,law,regions)
        row.update(detected_rgb=float(images.sum(dtype=np.float64)),input_rgb_per_four_views=input_rgb,
                   detected_input_ratio=float(images.sum(dtype=np.float64)/(input_rgb*1.5*1080*1920)),
                   energy_accounting=accounting,timing=timing,profiles=edge_profiles(images,sites))
        for p,r in zip(row['profiles'],ref_profiles):
            p['edge_offset_vs_reference']=p['location_50']-r['location_50'] if p['location_50'] is not None and r['location_50'] is not None else None
            p['width_error_vs_reference']=abs(p['width_10_90']-r['width_10_90']) if p['width_10_90'] is not None and r['width_10_90'] is not None else None
        if law['family']=='CURRENT': row['historical_rgb_relative_l2']=relative_change(images,np.load('runs/v0816_global_convergence/rgb_16777216.npy'))
        rows.append(row); write_json(CACHE/'partial_rows.json',rows)
        print(json.dumps({k:row[k] for k in ('whole_image_mse','gradient_cosine','interior_gt8px_mse')}),flush=True)
    report=dict(version='0.8.17',rows=rows,curves=curve_rows,frozen=frozen,reference_profiles=ref_profiles,
        interpretation='M=1: p=1-exp(-tau) is opacity, not geometric fraction. Prior N16m has MSE plateau, not strict source convergence. No source or reference regenerated.',
        criteria=dict(mse_relative_improvement=.05,gradient_cosine_increase=.02,interior_mse_max_relative_growth=.05,
                      edge_width_error_reduction=.1,edge_offset_max_growth_px=.25,fd_relative_l2=.05,fd_cosine=.99),
        runtime_seconds=time.perf_counter()-started)
    write_json(Path('artifacts/v0817_collision_transfer.json'),report)
    _write_csv(Path('artifacts/v0817_collision_transfer.csv'),_scalar_csv_rows(report))
    return report


if __name__=='__main__': run_experiment()
