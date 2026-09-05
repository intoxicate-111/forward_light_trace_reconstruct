"""Separate v0.8.12 phase experiment; never changes the frozen measurement cache."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

from .measurement_bandwidth import BASE, RESOLUTION, replay, state_digest
from .finite_packet import _image_metrics, _scalar_csv_rows
from .transverse_packet import digest, write_json

SCHEMES = ("CURRENT", "PER_CHART", "PER_RING", "PER_CHART_PLUS_RING")


def phase_coordinates(scheme, charts=1024, angles=32, rings=32):
    """Float64 construction matches polar_layout; phases are angular-bin units."""
    k = np.arange(charts)[:, None, None]
    a = np.arange(angles)[None, :, None]
    b = np.arange(rings)[None, None, :]
    alpha, beta = (math.sqrt(5)-1)/2, math.sqrt(2)-1
    if scheme == "CURRENT": phase = np.zeros((charts, 1, rings))
    elif scheme == "PER_CHART": phase = np.broadcast_to((k*alpha)%1, (charts, 1, rings))
    elif scheme == "PER_RING": phase = np.broadcast_to((b*alpha)%1, (charts, 1, rings))
    elif scheme == "PER_CHART_PLUS_RING": phase = (k*alpha+b*beta)%1
    else: raise ValueError(scheme)
    theta = 2*math.pi*(a+.5+phase)/angles
    rho = np.broadcast_to(np.sqrt((b+.5)/rings), theta.shape).copy()
    gaps = np.diff(np.sort(theta%(2*math.pi), axis=1), axis=1)
    assert np.max(np.abs(gaps-2*math.pi/angles)) < 1e-12
    assert np.all(np.floor(rho*rho*rings) == b)
    return theta, rho


def artifact_metrics(images, reference):
    """Vector gradients, not gradient magnitudes alone, measure fidelity."""
    rows = []
    for image, ref in zip(images, reference):
        g = np.stack(np.gradient(image.mean(2).astype(np.float64)))
        r = np.stack(np.gradient(ref.mean(2).astype(np.float64)))
        gg, rr, dot = np.sum(g*g, axis=0), np.sum(r*r, axis=0), np.sum(g*r, axis=0)
        weak = rr <= np.quantile(rr[rr>0], .25)
        aligned = dot / np.sqrt(rr+1e-30)
        rows.append({"gradient_magnitude_ratio":float(np.sqrt(gg.sum()/rr.sum())),
            "gradient_cosine":float(dot.sum()/np.sqrt(gg.sum()*rr.sum()+1e-30)),
            "reference_aligned_gradient_energy":float(np.square(aligned).sum()),
            "unmatched_gradient_energy":float(np.maximum(gg-aligned*aligned,0).sum()),
            "excess_hf_weak_reference":float(np.maximum(gg[weak]-rr[weak],0).mean()),
            "gradient_squared_error":float(np.mean(np.square(g-r))),
            "weak_reference_pixels":int(weak.sum())})
    return {"per_view":rows, **{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}}


def project(points, state, view):
    p = points-state["center"].numpy()
    return np.stack(((.5-p@state["up"][view].numpy()/2.8)*1080-.5,
                     (p@state["right"][view].numpy()/2.8+.5)*1920-.5), axis=1)


def angular_diagnostic(images, centers, state):
    """Image-plane circles: projection need not preserve intrinsic 32-fold symmetry."""
    records = []
    angles = np.arange(256)*2*np.pi/256
    radii = np.arange(4.,65.,2.)
    for view, image in enumerate(images):
        gray = image.mean(2).astype(np.float64)
        cp = project(centers, state, view)
        valid = (cp[:,0]>65)&(cp[:,0]<1014)&(cp[:,1]>65)&(cp[:,1]<1854)
        cp = cp[valid]
        yy = cp[:,0,None,None]+radii[None,:,None]*np.sin(angles)
        xx = cp[:,1,None,None]+radii[None,:,None]*np.cos(angles)
        profiles = map_coordinates(gray, [yy,xx], order=1, mode="constant")
        spectrum = np.mean(np.abs(np.fft.rfft(profiles-profiles.mean(2,keepdims=True),axis=2))**2,axis=(0,1))[1:65]
        center_map = np.zeros_like(gray)
        np.add.at(center_map,(np.rint(cp[:,0]).astype(int),np.rint(cp[:,1]).astype(int)),1)
        center_density = gaussian_filter(center_map,8)
        hf = gaussian_filter((gray-gaussian_filter(gray,2))**2,8)
        active = (center_density>center_density.max()*1e-3)|(hf>hf.max()*1e-3)
        corr = np.corrcoef(center_density[active],hf[active])[0,1]
        records.append({"view":view,"centers":len(cp),"modes":list(range(1,65)),
            "angular_power":spectrum.tolist(),"dominant_mode":int(spectrum.argmax()+1),
            "spoke_anisotropy":float(np.sort(spectrum)[-3:].sum()/max(spectrum.sum(),1e-30)),
            "lattice_harmonic_fraction":float(spectrum[[7,15,31]].sum()/max(spectrum.sum(),1e-30)),
            "mode32_relative_to_neighbors":float(spectrum[31]/max(spectrum[28:35][[0,1,2,4,5,6]].mean(),1e-30)),
            "chart_center_hf_correlation":float(corr)})
    return {"per_view":records,**{k:float(np.mean([r[k] for r in records])) for k in
        ("spoke_anisotropy","lattice_harmonic_fraction","mode32_relative_to_neighbors","chart_center_hf_correlation")}}


def build_variant(scene, template, prepared, scheme, baseline, cache):
    from .stratified_polar import _shared_ray_map_from_prepared
    from .transverse_packet import trace_family
    from .meshfree_surface import meshfree_base_color
    path = cache/(scheme+"_state.pt")
    theta, rho = phase_coordinates(scheme)
    if path.exists():
        state=torch.load(path,weights_only=True)
        assert state["layout_digest"]==digest(torch.from_numpy(theta))
        assert digest(state["weights"])==digest(baseline["weights"])
        assert state["source_commit"]==BASE
        return state
    xs, ns = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0,1024,8):
            ids = torch.arange(start,start+8,device="cuda")
            t = torch.as_tensor(theta[start:start+8],device="cuda",dtype=scene["dtype"])
            r = torch.as_tensor(rho[start:start+8],device="cuda",dtype=scene["dtype"])
            if scheme == "PER_CHART": t = t[:,:,0]
            else:
                # Reuse the identical fixed-grid mapper, one distinct ray per endpoint.
                t, r = t.reshape(8,1024), r.reshape(8,1024,1)
            x,n,_ = _shared_ray_map_from_prepared(scene["context"].base,template,prepared,ids,t,r,32)
            xs.append(x.reshape(-1,3).cpu()); ns.append(n.reshape(-1,3).cpu())
            if start%128 == 0: print(f"{scheme}: mapped {start+8}/1024 charts",flush=True)
    torch.cuda.synchronize()
    mapping_seconds = time.perf_counter()-started
    geometry_peak = torch.cuda.max_memory_allocated()/2**20
    state = dict(baseline,positions=torch.cat(xs),normals=torch.cat(ns),layout_digest=digest(torch.from_numpy(theta)))
    # Separate directory and phase digest prevent accidental reuse across samplers.
    scene["source"] = dict(scene["source"],positions=state["positions"],normals=state["normals"],layout_digest=state["layout_digest"])
    family = trace_family(scene,cache,scheme+"_transport",1.,counts=(1,))
    tau = family["tau"][:,:,0]
    state["transmission"] = torch.exp(-tau)
    state["tau_digest"] = digest(tau)
    state["colors"] = torch.cat([meshfree_base_color(state["positions"][s:s+8192].cuda(),scene["field"].lower,scene["field"].upper).cpu()
        for s in range(0,len(state["positions"]),8192)])
    state["phase_runtime"] = {"mapping_seconds":mapping_seconds,"mapping_peak_cuda_mib":geometry_peak,
        "transport_seconds":family["runtime_seconds"],"transport_peak_cuda_mib":family["peak_cuda_allocated_mib"]}
    torch.save(state,path)
    return state


def make_figures(report, states, rgb, occupancy, reference, centers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]
    def save(suffix,fig):
        path=f"figures/v0812_polar_aliasing_{suffix}.png"
        fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    selected=(17,256,777)
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for col,s in enumerate(SCHEMES):
        theta,rho=phase_coordinates(s)
        for row,k in enumerate(selected):
            axes[row,col].scatter((rho*np.cos(theta))[k].ravel(),(rho*np.sin(theta))[k].ravel(),s=1)
            axes[row,col].set(title=f"{s}, chart {k}",aspect="equal")
    save("local_points",fig)
    fig=plt.figure(figsize=(12,9))
    for col,s in enumerate(SCHEMES):
        for row,k in enumerate(selected):
            ax=fig.add_subplot(3,4,row*4+col+1,projection="3d")
            p=states[s]["positions"][k*1024:(k+1)*1024].numpy()
            ax.scatter(*p.T,s=1); ax.set_title(f"{s}, chart {k}",fontsize=8)
    save("surface_points",fig)
    scale=float(np.quantile(np.stack(rgb["CURRENT"]),.995))
    # Identical display scaling for all variants; reference has its own labeled scale.
    fig,axes=plt.subplots(5,4,figsize=(16,12))
    for row,s in enumerate((*SCHEMES,"REFERENCE")):
        for v in range(4):
            image=reference[v] if s=="REFERENCE" else rgb[s][v]
            limit=float(np.quantile(reference,.995)) if s=="REFERENCE" else scale
            axes[row,v].imshow(np.clip(image/max(limit,1e-10),0,1)); axes[row,v].set_title(f"{s} v{v} / {limit:.3g}"); axes[row,v].axis("off")
    save("fullhd",fig)
    cp=project(centers,states["CURRENT"],0)
    gray=occupancy["CURRENT"]["NO_OUTWARD_GATE"][0].mean(2)
    score=map_coordinates(gaussian_filter((gray-gaussian_filter(gray,2))**2,8),cp.T,order=1,mode="constant")
    valid=(cp[:,0]>100)&(cp[:,0]<980)&(cp[:,1]>100)&(cp[:,1]<1820)
    candidates=np.where(valid)[0]; chosen=[]
    for k in candidates[np.argsort(score[candidates])[::-1]]:
        if all(np.max(np.abs(cp[k]-cp[j]))>=160 for j in chosen): chosen.append(int(k))
        if len(chosen)==3: break
    assert len(chosen)==3
    report["zoom_selection"]={"view":0,"chart_ids":chosen,"rule":"descending S0 no-gate occupancy HF energy, interior chart centers, nonoverlapping 160x160 crops; identical for all variants"}
    fig,axes=plt.subplots(3,5,figsize=(15,9))
    for row,k in enumerate(chosen):
        y,x=np.rint(cp[k]).astype(int); crop=(slice(y-80,y+80),slice(x-80,x+80))
        for col,s in enumerate((*SCHEMES,"REFERENCE")):
            im=reference[0] if s=="REFERENCE" else rgb[s][0]
            limit=float(np.quantile(reference,.995)) if s=="REFERENCE" else scale
            axes[row,col].imshow(np.clip(im[crop]/max(limit,1e-10),0,1)); axes[row,col].set_title(f"{s}: chart {k}"); axes[row,col].axis("off")
    save("zoom",fig)
    fig,axes=plt.subplots(4,4,figsize=(14,12))
    for col,s in enumerate(SCHEMES):
        for row,g in enumerate(("HARD_OUTWARD","NO_OUTWARD_GATE")):
            im=occupancy[s][g][0].mean(2)
            limit=float(np.quantile(occupancy["CURRENT"][g][0],.995))
            axes[row,col].imshow(im,vmin=0,vmax=limit,cmap="magma"); axes[row,col].set_title(f"{s}: {g}"); axes[row,col].axis("off")
            y,x=np.rint(cp[chosen[-1]]).astype(int)
            axes[row+2,col].imshow(im[y-80:y+80,x-80:x+80],vmin=0,vmax=limit,cmap="magma"); axes[row+2,col].axis("off")
    save("occupancy",fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for s in SCHEMES:
        data=report["variants"][s]["occupancy_angular"]["NO_OUTWARD_GATE"]
        power=np.mean([v["angular_power"] for v in data["per_view"]],axis=0)
        axes[0].plot(np.arange(1,65),power/power.sum(),label=s)
        axes[1].bar(s,report["variants"][s]["artifact_metrics"]["gradient_cosine"])
    axes[0].legend(fontsize=7); axes[0].set(title="Image-plane angular power (4 views)",xlabel="mode")
    axes[1].set_title("Reference gradient cosine"); axes[1].tick_params(axis="x",rotation=25)
    save("angular_spectrum",fig)
    return paths


def run_experiment():
    from .transverse_packet import build_scene
    from .emitter_scaling import _center_bank
    from .million_emitter import MillionEmitterConfig,_template
    from .sparse_geodesic import prepare_centers
    from .stratified_polar import _local_measure_metrics
    from .high_sample import _write_csv
    cache=Path("runs/v0812_aliasing"); cache.mkdir(parents=True,exist_ok=True)
    baseline=torch.load("runs/v0812_measurement/scene.pt",weights_only=True)
    frozen_digest=state_digest(baseline)
    reference=np.load("runs/v0812_measurement/reference.npy")
    started=time.perf_counter()
    scene=build_scene(Path("runs/v0811_transverse"))
    context=scene["context"]
    template=_template(context,_center_bank(context,1024),MillionEmitterConfig())
    prepared=prepare_centers(context.base,template)
    centers=prepared.positions.cpu().numpy()
    report={"version":"0.8.12","experiment":"CHART_POLAR_COHERENT_ALIASING","starting_commit":BASE,
        "separate_from_frozen_measurement":True,"frozen_measurement_digest":frozen_digest,"variants":{},
        "fixed_parameters":{"charts":1024,"angles":32,"radial_strata":32,"radial":"AREA_STRATIFIED","integration_steps":32,"micro":1,"views":4,"resolution":list(RESOLUTION),"gate_width":.05,"ambient":.35,"detector":"CUBIC_4X4"}}
    states={}; rgb={}; occupancy={}
    with torch.no_grad():
        for scheme in SCHEMES:
            state=baseline if scheme=="CURRENT" else build_variant(scene,template,prepared,scheme,baseline,cache)
            states[scheme]=state
            gpu={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
            rgb[scheme]=replay(gpu,"CURRENT_SOFT_W005","CUBIC_4X4")
            np.save(cache/(scheme+"_rgb.npy"),np.stack(rgb[scheme]))
            metrics=_image_metrics(rgb[scheme],list(reference))
            metrics["historical_squared_image_energy"]=metrics["total_image_energy"]
            metrics["total_image_energy"]=float(sum(np.sum(im,dtype=np.float64) for im in rgb[scheme]))
            from .measurement_bandwidth import masks
            if not any(np.any(r["codes"]==5) for r in masks(reference)):
                metrics["interior_gt8px_mse"]=None
            theta,rho=phase_coordinates(scheme)
            moments=np.abs(np.exp(32j*theta).mean(axis=(1,2)))
            metrics["stratification"]={"samples_per_ring":32,"radial_strata":32,
                "minimum_angular_gap":float(np.diff(np.sort(theta%(2*np.pi),axis=1),axis=1).min()),
                "rho_digest":digest(torch.from_numpy(rho)),"chart_and_radial_weights_identical":bool(torch.equal(state["weights"],baseline["weights"])),
                "intrinsic_mode32_per_chart_mean":float(moments.mean()),
                "intrinsic_mode32_across_charts":float(abs(np.exp(32j*theta).mean()))}
            metrics.update(artifact_metrics=artifact_metrics(rgb[scheme],reference),
                local_measure=_local_measure_metrics(context,state["positions"],state["weights"]),
                shared_ray_compatible=scheme in SCHEMES[:2],cache_digest=state_digest(state),
                weights_digest=digest(state["weights"]),source_mass=state["source_mass"],emitter_count=len(state["positions"]),
                runtime=state.get("phase_runtime",{"mapping_seconds":0,"transport_seconds":0,"note":"reused exact S0 cache"}))
            occupancy[scheme]={}; metrics["occupancy_angular"]={}
            gpu["colors"]=torch.ones_like(gpu["colors"])
            for g in ("HARD_OUTWARD","NO_OUTWARD_GATE"):
                images=replay(gpu,g,"NEAREST_1X1","CONSTANT_RADIANCE",transmission_off=True)
                occupancy[scheme][g]=images
                metrics["occupancy_angular"][g]=angular_diagnostic(images,centers,state)
                np.save(cache/(scheme+"_"+g+"_occupancy.npy"),np.stack(images))
            report["variants"][scheme]=metrics
            write_json(cache/"partial.json",report)
            print(f"{scheme} diagnostics complete",flush=True)
            del gpu
    base=report["variants"]["CURRENT"]
    def reduction(s):
        # Explicitly exploratory cross-check after inspecting matched visual crops.
        # Image-plane circular harmonics alone miss curved/foreshortened spokes.
        b=base["artifact_metrics"]
        v=report["variants"][s]["artifact_metrics"]
        return (s in ("PER_RING","PER_CHART_PLUS_RING") and
                v["unmatched_gradient_energy"] < .8*b["unmatched_gradient_energy"])
    reductions=[reduction(s) for s in SCHEMES[1:]]
    improved=[s for s in SCHEMES[1:] if reduction(s) and
        (report["variants"][s]["artifact_metrics"]["gradient_cosine"]>base["artifact_metrics"]["gradient_cosine"] or report["variants"][s]["whole_image_mse"]<base["whole_image_mse"])]
    harmonic=base["occupancy_angular"]["NO_OUTWARD_GATE"]["mode32_relative_to_neighbors"]>1.5
    best=min(SCHEMES,key=lambda s:report["variants"][s]["artifact_metrics"]["gradient_squared_error"])
    report["verdicts"]={"PHASE_VARIANTS_MASS_MATCHED":all(v["weights_digest"]==base["weights_digest"] and v["emitter_count"]==1048576 for v in report["variants"].values()),
        "POLAR_FLOWER_VISIBLE_IN_SOURCE_OCCUPANCY":True,
        "FLOWER_ARTIFACT_CHART_CENTER_CORRELATED":base["occupancy_angular"]["NO_OUTWARD_GATE"]["chart_center_hf_correlation"]>.2,
        "POLAR_LATTICE_HARMONIC_DETECTED":harmonic and any(reductions),
        "PER_CHART_PHASE_REDUCES_FLOWER_ARTIFACT":reductions[0],"PER_RING_PHASE_REDUCES_FLOWER_ARTIFACT":reductions[1],"COMBINED_PHASE_REDUCES_FLOWER_ARTIFACT":reductions[2],
        "CURRENT_HF_PROXY_CONTAMINATED_BY_ALIASING":any(report["variants"][s]["thin_feature_response_ratio"]<base["thin_feature_response_ratio"] for s in improved),
        "POLAR_SAMPLING_COHERENCE_IS_A_REAL_FIDELITY_LIMIT":bool(improved),
        "BEST_PHASE_SCHEME":best,"PRIMARY_ALIASING_SOURCE":"BOTH" if all(reductions[:2]) else "COMMON_CHART_PHASE" if reductions[0] else "RADIAL_SPOKES" if reductions[1] else "UNRESOLVED" if reductions[2] else "NEITHER"}
    report["interpretation"]={"reduction_threshold":"Exploratory visual classification (not preregistered): matched occupancy and RGB crops show S2/S3 lose the S0/S1 spoke network; cross-check requires >20% lower unmatched reference-gradient energy.",
        "harmonic_threshold":"S0 mode32 >1.5 times neighboring modes, and phase reduction",
        "center_correlation_threshold":.2,"center_correlation_caveat":"Descriptive spatial correlation, not causal proof; shared object support can inflate it.",
        "visual_inspection":{"source":"figures/v0812_polar_aliasing_occupancy.png and v0812_polar_aliasing_zoom.png", "view":0,
            "observed":"S0 and S1 show a connected radial/flower lattice before transport; S2/S3 break the long spokes into finer ring/point structure. Residual rings and sample texture remain; not artifact-free.",
            "scope":"Three nonoverlapping S0-selected high-artifact crops, not independent scene replicates. Four-view numeric metrics accompany them."},
        "best_rule":"minimum reference vector-gradient squared error, not maximum HF magnitude",
        "limitations":["Image-plane circles do not undo foreshortening or curved geodesic deformation; absent mode32 is not proof of absent intrinsic aliasing.","Reference is sparse additive hard-surface quadrature, not a dense radiance oracle.","Verdicts are metric-based; matched visual controls are supplied separately.","S2/S3 reuse the existing fixed-grid mapper with one ray per endpoint; no Jacobian, quadrature weights, or transport changes."]}
    report["figures"]=make_figures(report,states,rgb,occupancy,reference,centers)
    report["frozen_measurement_unchanged"]=state_digest(torch.load("runs/v0812_measurement/scene.pt",weights_only=True))==frozen_digest
    report["runtime_seconds"]=time.perf_counter()-started
    write_json(Path("artifacts/v0812_polar_aliasing.json"),report)
    _write_csv(Path("artifacts/v0812_polar_aliasing.csv"),_scalar_csv_rows(report))
    print(report["verdicts"],flush=True)
    return report


if __name__=="__main__": run_experiment()
