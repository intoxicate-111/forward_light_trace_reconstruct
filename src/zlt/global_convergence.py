"""v0.8.16: one-axis fixed global source quadrature convergence and birth gate."""
import hashlib
import json
import time
import resource
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .fixed_measure import attach,diagnostic_ids
from .density_matrix import sparse_scene,render,artifacts_score
from .dense_reference import load_mse_reference,relative_change
from .measurement_bandwidth import masks
from .meshfree_surface import meshfree_base_color
from .polar_aliasing import project,artifact_metrics
from .finite_packet import _image_metrics,_scalar_csv_rows
from .transverse_packet import trace_family,digest,write_json

CACHE=Path("runs/v0816_global_convergence")
LEVELS=tuple(2**p for p in range(19,25))


def prefix_weights(count,mass):
    return torch.full((count,),mass/count,dtype=torch.float64)


def classify(rows):
    first,last=rows[0],rows[-1]; previous=rows[-2]
    delta=(previous["whole_image_mse"]-last["whole_image_mse"])/previous["whole_image_mse"]
    plateau=abs(delta)<.05
    gaps=max(x["missing_energy_0_5px"] for x in last["coverage"])<.01 and max(x["missing_energy_1px"] for x in last["coverage"])<.001
    rgb=max(last["adjacent_rgb_per_view"])<.01
    structure=last["radial_banding_score"]<=1.1*previous["radial_banding_score"]
    converged=plateau and gaps and rgb and structure
    return {"GLOBAL_SAMPLE_COUNT_INCREASE_IMPROVES_FULLHD_FIDELITY":last["whole_image_mse"]<.95*first["whole_image_mse"],
        "GLOBAL_SAMPLE_COUNT_REDUCES_PIXEL_DISCRETENESS":max(last["adjacent_rgb_per_view"])<max(rows[1]["adjacent_rgb_per_view"]),
        "GLOBAL_SAMPLE_COUNT_REDUCES_PROJECTED_GAPS":np.mean([x["no_center_0_5px"] for x in last["coverage"]])<np.mean([x["no_center_0_5px"] for x in first["coverage"]]),
        "GLOBAL_SAMPLE_COUNT_HAS_REACHED_FULLHD_PLATEAU":plateau,
        "FULLHD_GLOBAL_SOURCE_QUADRATURE_CONVERGED":converged,
        "REMAINING_FULLHD_ERROR_DOMINATED_BY_SOURCE_DISCRETENESS":False if converged else "UNRESOLVED",
        "RECOMMENDED_GLOBAL_SAMPLE_COUNT":last["samples"] if converged else "UNRESOLVED"}, {"mse_relative_improvement":delta,"rgb_gate":rgb,"coverage_gate":gaps,"structured_artifact_proxy_gate":structure}


def generate(scene,baseline,count):
    path=CACHE/f"source_{count}.pt"
    if path.exists(): return torch.load(path,weights_only=True)
    mass=float(baseline["weights"].sum()); original=len(baseline["positions"])
    if count<=original:
        state={k:baseline[k] for k in ("center","directions","right","up")}
        for k in ("positions","normals","colors"): state[k]=baseline[k][:count].clone()
        state["transmission"]=baseline["transmission"][:,:count].clone()
        state["weights"]=prefix_weights(count,mass)
        state["timing"]={"source_seconds":0.,"transport_seconds":0.,"cache_reuse":"exact prefix of v0814 source and transport"}
    else:
        identity=torch.load("runs/v0814_fixed_measure/global_identity.pt",weights_only=True)
        triangles=identity["vertices"].numpy()[identity["faces"].numpy()]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        cdf=np.cumsum(area); sampler=torch.quasirandom.SobolEngine(3,scramble=True,seed=101)
        base=scene["context"].base
        state={k:baseline[k] for k in ("center","directions","right","up")}
        for k in ("positions","normals","colors"): state[k]=torch.empty((count,3),dtype=torch.float64)
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for i in range(0,count,8192):
            s=sampler.draw(min(8192,count-i),dtype=torch.float64).numpy()
            ids=np.searchsorted(cdf,s[:,0]*cdf[-1]); u=np.sqrt(s[:,1]); v=s[:,2]
            bary=np.stack((1-u,u*(1-v),u*v),1)
            p=torch.as_tensor(np.einsum("ni,nij->nj",bary,triangles[ids]),device="cuda",dtype=torch.float64)
            n=base.gradient(p); n/=n.norm(dim=1,keepdim=True).clamp_min(1e-30)
            x,n=attach(base,p,n)
            state["positions"][i:i+len(p)]=x.cpu(); state["normals"][i:i+len(p)]=n.cpu()
            state["colors"][i:i+len(p)]=meshfree_base_color(x,base.lower,base.upper).cpu()
        state["weights"]=prefix_weights(count,mass)
        state["timing"]={"source_seconds":time.perf_counter()-start,"source_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"source_peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
        assert torch.allclose(state["positions"][:original],baseline["positions"],atol=1e-12,rtol=0)
        state["layout_digest"]=digest(state["positions"])
        scene["source"]=state
        family=trace_family(scene,CACHE,f"transport_{count}",1.,counts=(1,))
        state["transmission"]=torch.exp(-family["tau"][:,:,0])
        state["timing"].update(transport_seconds=family["runtime_seconds"],transport_peak_allocated_mib=family["peak_cuda_allocated_mib"],transport_peak_reserved_mib=family["peak_cuda_reserved_mib"])
    state["position_digest"]=digest(state["positions"])
    torch.save(state,path); return state


def plots(rows,reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]; ns=[r["samples"] for r in rows]
    def save(fig,name):
        path=f"figures/v0816_global_sample_{name}.png"; fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    for name,key in (("mse","whole_image_mse"),("adjacent_rgb","adjacent_rgb_per_view")):
        fig,ax=plt.subplots(figsize=(6,4)); data=[r[key] for r in rows if key in r]
        ax.plot(ns[-len(data):],data,marker="o"); ax.set(xscale="log",yscale="log",xlabel="Global emitters",ylabel=key)
        if key=="adjacent_rgb_per_view": ax.legend([f"view {i}" for i in range(4)])
        save(fig,name)
    fig,ax=plt.subplots(figsize=(6,4))
    for key in ("p50","p90","p95","p99"):
        ax.plot(ns,[np.mean([v[key] for v in r["projected_spacing"]]) for r in rows],marker="o",label=key)
    ax.set(xscale="log",yscale="log",xlabel="Global emitters",ylabel="Emitter nearest-neighbor distance (px)"); ax.legend(); save(fig,"spacing")
    fig,ax=plt.subplots(figsize=(6,4))
    for key,label in (("no_center_0_5px","Pixels, 0.5 px"),("no_center_1px","Pixels, 1 px"),("missing_energy_0_5px","Energy, 0.5 px"),("missing_energy_1px","Energy, 1 px")):
        ax.plot(ns,[np.mean([v[key] for v in r["coverage"]]) for r in rows],marker="o",label=label)
    ax.set(xscale="log",yscale="log",xlabel="Global emitters",ylabel="Missing fraction"); ax.legend(); save(fig,"coverage")
    crop=next(c for c in json.loads(Path("artifacts/v0813_density_matrix.json").read_text())["zoom_crops"] if c["name"]=="ring")
    y,x=crop["row"],crop["column"]; exposure=.52
    fig,axes=plt.subplots(1,len(rows),figsize=(18,3))
    for ax,r in zip(axes,rows):
        image=np.load(CACHE/f"rgb_{r['samples']}.npy",mmap_mode="r")[0]
        ax.imshow(np.clip(image[y-80:y+80,x-80:x+80]*exposure,0,1)); ax.set_title(f"N={r['samples']:,}",fontsize=8); ax.axis("off")
    save(fig,"matched_crop")
    representatives=[rows[0]["samples"],2**22,rows[-1]["samples"]]
    fig,axes=plt.subplots(4,4,figsize=(14,10)); errfig,erraxes=plt.subplots(3,4,figsize=(14,8))
    for row,n in enumerate(representatives):
        images=np.load(CACHE/f"rgb_{n}.npy",mmap_mode="r")
        for v in range(4):
            axes[row,v].imshow(np.clip(images[v]*exposure,0,1)); axes[row,v].set_title(f"{n:,}, view {v}"); axes[row,v].axis("off")
            erraxes[row,v].imshow((images[v]-reference[v]).mean(2),cmap="coolwarm",vmin=-.5,vmax=.5); erraxes[row,v].set_title(f"{n:,}, view {v}"); erraxes[row,v].axis("off")
    for v in range(4): axes[3,v].imshow(np.clip(reference[v]*exposure,0,1)); axes[3,v].set_title("Dense hard reference"); axes[3,v].axis("off")
    save(fig,"full_views"); save(errfig,"residuals"); return paths


def run_experiment():
    from .high_sample import _write_csv
    torch.set_num_threads(8); CACHE.mkdir(parents=True,exist_ok=True)
    # Snapshot all existing outputs including uncommitted v0814/v0815 evidence.
    manifest_path=CACHE/"historical_hashes.json"
    if not manifest_path.exists():
        historical={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("artifacts","figures") for p in Path(folder).iterdir() if p.is_file() and "v0816" not in p.name}
        write_json(manifest_path,historical)
    started=time.perf_counter(); reference=load_mse_reference(); target_masks=masks(reference)
    baseline=torch.load("runs/v0814_fixed_measure/global_energy_matched_state.pt",weights_only=True)
    legacy=torch.load("runs/v0812_aliasing/PER_CHART_PLUS_RING_state.pt",weights_only=True)
    with torch.no_grad(): scene,_,prepared,_=sparse_scene(legacy)
    rows=[]; previous=None
    for count in LEVELS:
        print(f"Global source convergence N={count:,}",flush=True); begin=time.perf_counter()
        with torch.no_grad(): state=generate(scene,baseline,count)
        readout_cache=CACHE/f"readout_{count}.json"
        if readout_cache.exists():
            cached=json.loads(readout_cache.read_text()); accounting,timing=cached["accounting"],cached["timing"]
            images=np.load(CACHE/f"rgb_{count}.npy")
        else:
            with torch.no_grad(): images,occupancy,accounting,timing=render(state,4)
            del occupancy
            images=np.stack(images); np.save(CACHE/f"rgb_{count}.npy",images)
            write_json(readout_cache,{"accounting":accounting,"timing":timing})
        row={"samples":count,**_image_metrics(list(images),list(reference)),**artifact_metrics(images,reference)}
        row["historical_squared_image_energy"]=row.pop("total_image_energy")
        row["source_weight_sum"]=float(state["weights"].sum()); row["emitted_rgb"]=float((state["weights"][:,None]*state["colors"]).sum())
        row["detected_rgb"]=float(images.sum(dtype=np.float64)); row["energy_accounting"]=accounting
        row["relative_energy_accounting_error"]=max(a["relative_energy_difference"] for a in accounting)
        row["timing"]={**state["timing"],**timing}; row["position_digest"]=state["position_digest"]
        row["coverage"]=[]; row["projected_spacing"]=[]
        for view in range(4):
            coords=project(state["positions"].numpy(),state,view); tree=cKDTree(coords)
            ids=diagnostic_ids(count,65536)
            d=tree.query(coords[ids],k=2,workers=1)[0][:,1]
            row["projected_spacing"].append(dict(zip(("p50","p90","p95","p99"),np.quantile(d,[.5,.9,.95,.99]).tolist())))
            distances=tree.query(np.argwhere(target_masks[view]["foreground"]),workers=1)[0]
            energy=reference[view].sum(2)[target_masks[view]["foreground"]].astype(np.float64)
            row["coverage"].append({"view":view,"no_center_0_5px":float(np.mean(distances>.5)),"no_center_1px":float(np.mean(distances>1.)),
                "missing_energy_0_5px":float(energy[distances>.5].sum()/energy.sum()),"missing_energy_1px":float(energy[distances>1.].sum()/energy.sum())})
            row["per_view"][view]["whole_image_mse"]=_image_metrics([images[view]],[reference[view]])["whole_image_mse"]
            del tree,coords
        row["radial_banding_score"]=artifacts_score(list(images),prepared.positions.cpu().numpy(),state)["radial_banding_score"]
        if previous is not None:
            row["adjacent_rgb_per_view"]=[relative_change(a,b) for a,b in zip(previous,images)]
            row["adjacent_gradient_per_view"]=[relative_change(np.stack(np.gradient(a.mean(2))),np.stack(np.gradient(b.mean(2)))) for a,b in zip(previous,images)]
        if count==2**22:
            row["v0814_replay_relative_l2"]=relative_change(images,np.load("runs/v0814_fixed_measure/global_rgb.npy"))
        row["total_seconds"]=time.perf_counter()-begin; row["cpu_rss_peak_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        rows.append(row); previous=images; write_json(CACHE/"partial_rows.json",rows)
        print(json.dumps({k:row[k] for k in ("samples","whole_image_mse","coverage")}),flush=True)
        del state
    verdicts,evidence=classify(rows)
    verdicts={k:bool(v) if isinstance(v,np.bool_) else v for k,v in verdicts.items()}
    report={"version":"0.8.16","levels":rows,"verdicts":verdicts,"gate_evidence":evidence,
        "criteria":{"max_view_rgb_change":.01,"absolute_relative_mse_change":.05,"max_view_missing_reference_energy_0_5px":.01,"max_view_missing_reference_energy_1px":.001,"max_radial_banding_growth":1.1},
        "frozen":{"seed":101,"source_measure":baseline["global_measure"]["definition"],"source_weight_sum":float(baseline["weights"].sum()),"geometry_bases":64,"lambda":"all zero","views":4,"resolution":[1080,1920],"C":3,"capture_grid":4,"micro":1,"packet_radius":scene["h"],"shell_width":scene["h"],"path_step":.5*scene["h"],"gate_width":.05,"ambient":.35,"sensor_gain":1.5,"window":2.8,"grid_digest":digest(scene["context"].base.grid),"reference_digest":digest(torch.from_numpy(reference))},
        "energy_note":"One fixed scalar weight sum at all N; weights S/N, unchanged color. Colored integral varies only by deterministic quadrature error; require relative variation <1e-4, never normalize brightness per level.",
        "spacing_note":"2D projected emitter-to-emitter nearest neighbor at 65536 fixed random IDs (seed 103), includes hidden emitters. Coverage queries all foreground pixel centers of corrected dense reference. Gate uses reference-energy-weighted missing fractions, because cubic boundary halos create irreducible unweighted center gaps. Not a visibility-weighted observation rank.",
        "phase0":{"retained":"Density reduces aliasing; capture integration adds no new observations; geodesic CV .29503 versus global .11124; lambda tangential redistribution exists; tested own global lambda JVP closes at 4.78e-8 relative L2.","corrected":"Old sparse-reference MSE and near-zero spatial cosine are invalid Full-HD fidelity evidence. Corrected dense reference gives MSE .032109 geodesic and .018477 global. Spatial gradients are not dI/dlambda; .003 cosine differences are below reference-gradient convergence uncertainty."},
        "birth_gate":"PASS_REQUIRES_PHASE2" if verdicts["FULLHD_GLOBAL_SOURCE_QUADRATURE_CONVERGED"] else "STOP_PHASE1_SAMPLING_NOT_CONVERGED",
        "birth_experiments_run":False,"runtime_seconds":time.perf_counter()-started}
    report["figures"]=plots(rows,reference)
    write_json(Path("artifacts/v0816_global_sample_convergence.json"),report)
    _write_csv(Path("artifacts/v0816_global_sample_convergence.csv"),_scalar_csv_rows(report))
    print(json.dumps(verdicts,indent=2),flush=True); return report


if __name__=="__main__": run_experiment()
