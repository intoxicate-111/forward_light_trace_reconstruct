"""v0.8.13: source density x detector integration, fixed scene and radiometry."""
from __future__ import annotations
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.spatial import cKDTree
from .measurement_bandwidth import RESOLUTION, gate, lobe, kernel_pairs, state_digest, masks
from .polar_aliasing import phase_coordinates, artifact_metrics, angular_diagnostic, project
from .finite_packet import _image_metrics, _scalar_csv_rows
from .transverse_packet import digest, write_json

BASE = "fa141e346284d2a15117922046a66e588ca7d21a"
LEVELS = (32,48,64)
CAPTURE = (1,2,4)
CACHE = Path("runs/v0813_density_sparse")


def streamed_chart_spacing(centers,normals):
    """Same fourth normal-aware neighbor as _local_spacing, O(K) workspace."""
    result=[]
    for index,center in enumerate(centers):
        chord=torch.linalg.vector_norm(centers-center,dim=1)
        bend=1+.5*(1-(normals*normals[index]).sum(1).clamp(-1,1))
        distance=chord*bend; distance[index]=torch.inf
        result.append(torch.topk(distance,4,largest=False).values[-1])
    return torch.stack(result)


def sparse_scene(baseline):
    from .corrected_birth import CorrectedBirthConfig,_build_context
    from .mesh_field import prepare_stanford_bunny
    from .emitter_scaling import _center_bank
    from .geodesic_source import _eta,_reference_frames
    from .sparse_geodesic import GeodesicTemplate,prepare_centers
    from .boundary_transport import nested_fibonacci_atlas,enclosing_observation_sphere
    from .finite_packet import _surface_spacing
    from .transverse_packet import FusedGridLookup
    from .stratified_polar import _shared_ray_map_from_prepared
    mesh=prepare_stanford_bunny(Path("data/stanford_bunny/cache/bun_zipper.ply"),build_surface_scaffold=False)
    context=_build_context(mesh,CorrectedBirthConfig(dictionary_count=128,initial_count=32,surface_samples=4096,views=4,resolution=256,surface_scramble_seed=101))
    bank=_center_bank(context,1024); centers,normals=bank["centers"],bank["normals"]
    first,second=_reference_frames(normals)
    template=GeodesicTemplate(centers,normals,first,second,streamed_chart_spacing(centers,normals),_eta(context.base,centers),4)
    prepared=prepare_centers(context.base,template)
    scene={"context":context,"source":baseline,"device":centers.device,"dtype":centers.dtype,
        "field":FusedGridLookup(context.base),"h":_surface_spacing(context.reference_points),"eta":template.eta,
        "atlas":nested_fibonacci_atlas(centers.device,(4,)),"boundary":enclosing_observation_sphere(context.reference_points)}
    # Guard the equivalent row-streamed radius calculation against source changes.
    theta,rho=phase_coordinates("PER_CHART_PLUS_RING")
    ids=torch.tensor([0,17,166,953],device=centers.device)
    angles=torch.as_tensor(theta[ids.cpu().numpy()],device=centers.device,dtype=centers.dtype).reshape(4,1024)
    radial=torch.as_tensor(rho[ids.cpu().numpy()],device=centers.device,dtype=centers.dtype).reshape(4,1024,1)
    x,n,_=_shared_ray_map_from_prepared(context.base,template,prepared,ids,angles,radial,32)
    oldx=baseline["positions"].reshape(1024,1024,3)[ids.cpu()]
    oldn=baseline["normals"].reshape(1024,1024,3)[ids.cpu()]
    equivalence={"charts":[0,17,166,953],"position_max_abs":float((x.reshape_as(oldx).cpu()-oldx).abs().max()),
        "normal_max_abs":float((n.reshape_as(oldn).cpu()-oldn).abs().max())}
    assert max(equivalence["position_max_abs"],equivalence["normal_max_abs"])<1e-9
    return scene,template,prepared,equivalence


def capture_offsets(count):
    if count not in CAPTURE: raise ValueError(count)
    return (np.arange(count)+.5)/count-.5


def capture_splat(acc, row, col, energy, count, resolution=RESOLUTION):
    """Midpoint quadrature of the same translated cubic response over one pixel.

    Offsets are detector-space coordinates, never source/world-space displacements.
    Each translated partition of unity has unit mass. C1 is exactly the old splat.
    Workspace is chunk x 16 taps x channels, never emitter x geometry parameter.
    """
    for y in capture_offsets(count):
        for x in capture_offsets(count):
            ids,weights=kernel_pairs(row-float(y),col-float(x),"CUBIC_4X4",resolution)
            acc.index_add_(0,ids.flatten(),(weights[...,None]*energy[:,None]/count**2).reshape(-1,energy.shape[1]))


def source_state(scene,template,prepared,n,baseline):
    from .stratified_polar import _shared_ray_map_from_prepared
    from .transverse_packet import trace_family
    from .meshfree_surface import meshfree_base_color
    if n==32: return baseline
    path=CACHE/f"source_{n}.pt"
    if path.exists():
        state=torch.load(path,weights_only=True)
        assert state["density_start_commit"]==BASE and len(state["positions"])==1024*n*n
        return state
    theta,rho=phase_coordinates("PER_CHART_PLUS_RING",angles=n,rings=n)
    xs,ns=[],[]
    charts_per_chunk=max(1,8192//(n*n))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start_time=time.perf_counter()
    with torch.no_grad():
        for start in range(0,1024,charts_per_chunk):
            stop=min(1024,start+charts_per_chunk); size=stop-start
            ids=torch.arange(start,stop,device="cuda")
            t=torch.as_tensor(theta[start:stop],device="cuda",dtype=scene["dtype"]).reshape(size,n*n)
            r=torch.as_tensor(rho[start:stop],device="cuda",dtype=scene["dtype"]).reshape(size,n*n,1)
            x,norm,_=_shared_ray_map_from_prepared(scene["context"].base,template,prepared,ids,t,r,32)
            xs.append(x.reshape(-1,3).cpu()); ns.append(norm.reshape(-1,3).cpu())
        torch.cuda.synchronize()
    runtime={"mapping_seconds":time.perf_counter()-start_time,
        "mapping_peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "mapping_peak_cuda_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
    chart_mass=baseline["weights"].reshape(1024,1024).sum(1)
    weights=(chart_mass[:,None].expand(-1,n*n)/(n*n)).reshape(-1).clone()
    state=dict(baseline,positions=torch.cat(xs),normals=torch.cat(ns),weights=weights,
        layout_digest=digest(torch.from_numpy(theta)),density_start_commit=BASE)
    scene["source"]=dict(scene["source"],positions=state["positions"],normals=state["normals"],weights=weights,
        layout_digest=state["layout_digest"])
    family=trace_family(scene,CACHE,f"transport_{n}",1.,counts=(1,))
    tau=family["tau"][:,:,0]
    state["transmission"]=torch.exp(-tau); state["tau_digest"]=digest(tau)
    state["colors"]=torch.cat([meshfree_base_color(state["positions"][i:i+8192].cuda(),scene["field"].lower,scene["field"].upper).cpu()
        for i in range(0,len(weights),8192)])
    runtime.update(transport_seconds=family["runtime_seconds"],
        transport_peak_cuda_allocated_mib=family["peak_cuda_allocated_mib"],transport_peak_cuda_reserved_mib=family["peak_cuda_reserved_mib"])
    state["density_runtime"]=runtime
    torch.save(state,path)
    print(json.dumps({"phase":"source_complete","n":n,**runtime}),flush=True)
    return state


def render(state,capture):
    frames=[]; occupancy=[]; accounting=[]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    with torch.no_grad():
        for v in range(4):
            acc=torch.zeros((1080*1920,4),device="cuda",dtype=torch.float32)
            expected=0.
            for start in range(0,len(state["positions"]),8192):
                stop=start+8192
                p=state["positions"][start:stop].cuda(); normals=state["normals"][start:stop].cuda()
                cosine=normals@state["directions"][v].cuda()
                e=state["weights"][start:stop].cuda()*state["transmission"][v,start:stop].cuda()*gate(cosine,"CURRENT_SOFT_W005")*lobe(cosine,"CURRENT_LOBE")
                energy=(e[:,None]*state["colors"][start:stop].cuda()).float()
                expected+=float(energy.double().sum())
                relative=p.float()-state["center"].cuda().float()
                row=(.5-relative@state["up"][v].cuda().float()/2.8)*1080-.5
                col=(relative@state["right"][v].cuda().float()/2.8+.5)*1920-.5
                capture_splat(acc,row,col,torch.cat((energy,torch.ones_like(energy[:,:1])),1),capture)
            array=acc.reshape(1080,1920,4).cpu().numpy()
            frames.append(array[:,:,:3]*(1.5*1080*1920)); occupancy.append(array[:,:,3])
            actual=float(np.sum(array[:,:,:3],dtype=np.float64))
            accounting.append({"view":v,"input_energy":expected,"detected_energy":actual,
                "relative_energy_difference":abs(actual-expected)/max(expected,1e-30),
                "projected_count_sum":float(np.sum(array[:,:,3],dtype=np.float64))})
    torch.cuda.synchronize()
    timing={"readout_seconds":time.perf_counter()-started,
        "cuda_allocated_peak_mib":torch.cuda.max_memory_allocated()/2**20,
        "cuda_reserved_peak_mib":torch.cuda.max_memory_reserved()/2**20,
        "cpu_rss_peak_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    return frames,occupancy,accounting,timing


def density_stats(state,reference):
    results={c:[] for c in CAPTURE}
    for v,target in enumerate(reference):
        xy=project(state["positions"].numpy(),state,v)
        tree=cKDTree(xy)
        pixels=np.argwhere(np.linalg.norm(target,axis=2)>0).astype(np.float64)
        for c in CAPTURE:
            distances=[]; neighbors={r:[] for r in (.5,1.,2.)}
            for y in capture_offsets(c):
                for x in capture_offsets(c):
                    q=pixels+[y,x]
                    distances.append(tree.query(q,workers=1)[0])
                    for radius in neighbors:
                        neighbors[radius].append(tree.query_ball_point(q,radius,return_length=True,workers=1))
            d=np.concatenate(distances)
            counts={radius:np.concatenate(values) for radius,values in neighbors.items()}
            results[c].append({"view":v,"foreground_pixels":len(pixels),"capture_locations":len(d),
                "nearest_projected_distance_px":dict(zip(("p50","p90","p95","p99"),map(float,np.quantile(d,[.5,.9,.95,.99])))),
                "neighbors":{str(r):{"mean":float(a.mean()),"p50":float(np.median(a)),"p95":float(np.quantile(a,.95)),
                    "zero_fraction":float(np.mean(a==0)),"weak_lt2_fraction":float(np.mean(a<2))} for r,a in counts.items()}})
    return results


def artifacts_score(images,centers,state):
    angular=angular_diagnostic(images,centers,state)
    radial=[]
    theta=np.arange(128)*2*np.pi/128; radius=np.arange(4.,65.)
    for v,image in enumerate(images):
        cp=project(centers,state,v)
        yy=cp[:,0,None,None]+radius[None,:,None]*np.sin(theta)
        xx=cp[:,1,None,None]+radius[None,:,None]*np.cos(theta)
        a=map_coordinates(image.mean(2).astype(float),[yy,xx],order=1,mode="constant").mean(2)
        residual=a-gaussian_filter1d(a,2,axis=1)
        radial.append(float(np.mean(residual**2)/max(np.mean(a*a),1e-30)))
    return {"spoke_anisotropy":angular["spoke_anisotropy"],"angular_lattice_fraction":angular["lattice_harmonic_fraction"],
        "radial_banding_score":float(np.mean(radial)),"radial_banding_per_view":radial,
        "scope":"Fixed image-plane circles r=4..64px around all chart centers; geometry/overlap can also cause anisotropy or radial variation."}


def classify_fidelity(report):
    """Cosine is signed: improvement must use a difference, never a ratio."""
    def cell(e,c): return next(r for r in report["cells"] if r["E"]==e and r["C"]==c)
    evidence={}; improved={}
    for axis in ("E","C"):
        pairs=[(cell(1,j),cell(3,j)) if axis=="E" else (cell(j,1),cell(j,3)) for j in range(1,4)]
        evidence[axis]={key+"_ratio":float(np.mean([b[key]/a[key] for a,b in pairs])) for key in
            ("whole_image_mse","gradient_squared_error","unmatched_gradient_energy")}
        delta=float(np.mean([b["gradient_cosine"]-a["gradient_cosine"] for a,b in pairs]))
        evidence[axis]["gradient_cosine_delta"]=delta
        e=evidence[axis]
        improved[axis]=e["whole_image_mse_ratio"]<=1 and (e["gradient_squared_error_ratio"]<=.99 or
            (e["gradient_squared_error_ratio"]<1 and e["unmatched_gradient_energy_ratio"]<=.8 and delta>0))
    return improved,evidence


def refresh_classification(report):
    improved,evidence=classify_fidelity(report)
    report["verdicts"]["EMITTER_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY"]=improved["E"]
    report["verdicts"]["CAPTURE_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY"]=improved["C"]
    report["verdicts"]["PRIMARY_LIMITATION"]=("JOINT_SOURCE_AND_CAPTURE_DENSITY" if all(improved.values()) else
        "SOURCE_EMITTER_DENSITY" if improved["E"] else "CAPTURE_DENSITY" if improved["C"] else "OTHER")
    for axis in ("E","C"):
        report["axis_evidence"][axis].pop("gradient_cosine_ratio",None)
        report["axis_evidence"][axis]["gradient_cosine_delta"]=evidence[axis]["gradient_cosine_delta"]
    report["artifact_verdict_scope"]="Flower flags mean >=20% lower unmatched gradient energy; ring flags mean >=20% lower radial-banding score, averaged over the other axis. These pixel-scale proxy reductions do not assert that coherent chart-scale petals or rings disappear. Inspect the matched crops; smoothing can lower these scores without adding correct geometric detail."
    return report


def figures(report,reference,centers,state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]
    def save(name,fig):
        path=f"figures/v0813_{name}.png"; fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    def read(e,c,kind): return np.load(CACHE/f"E{e}C{c}_{kind}.npy",mmap_mode="r")
    rgb_limit=float(np.quantile(read(1,1,"rgb"),.995))
    for view in range(4):
        fig,axes=plt.subplots(3,3,figsize=(12,9))
        for e in range(1,4):
            for c in range(1,4):
                axes[e-1,c-1].imshow(np.clip(read(e,c,"rgb")[view]/rgb_limit,0,1)); axes[e-1,c-1].axis("off"); axes[e-1,c-1].set_title(f"E{e} C{c} / view {view}")
        save(f"density_fullhd_view{view}",fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9)); limit=float(np.quantile(read(3,1,"occupancy"),.995))
    for e in range(1,4):
        for c in range(1,4):
            axes[e-1,c-1].imshow(read(e,c,"occupancy")[0],vmin=0,vmax=limit,cmap="magma"); axes[e-1,c-1].axis("off"); axes[e-1,c-1].set_title(f"E{e} C{c}: integrated emitter count")
    save("density_occupancy",fig)
    cp=project(centers,state,0)
    points=state["positions"].numpy()
    thin_center=project(points[np.argsort(points[:,2])[-1000:]],state,0).mean(0)
    crops=[("spoke",cp[953]),("ring",cp[166]),("thin_world_top_z",thin_center)]
    report["zoom_crops"]=[]
    for name,(y,x) in crops:
        y=int(np.clip(round(y),80,1000)); x=int(np.clip(round(x),80,1840))
        report["zoom_crops"].append({"name":name,"view":0,"row":y,"column":x,"size":160})
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        for e in range(1,4):
            for c in range(1,4):
                axes[e-1,c-1].imshow(np.clip(read(e,c,"rgb")[0,y-80:y+80,x-80:x+80]/rgb_limit,0,1)); axes[e-1,c-1].set_title(f"E{e} C{c}"); axes[e-1,c-1].axis("off")
            axes[e-1,3].imshow(np.clip(reference[0,y-80:y+80,x-80:x+80],0,1)); axes[e-1,3].set_title("Unchanged sparse reference /1"); axes[e-1,3].axis("off")
        save(f"density_zoom_{name}",fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for j,key in enumerate(("whole_image_mse","gradient_squared_error")):
        for c in range(1,4):
            rr=[r for r in report["cells"] if r["C"]==c]
            axes[j,0].plot([r["total_emitters"] for r in rr],[r[key] for r in rr],"o-",label=f"C{c}")
        for e in range(1,4):
            rr=[r for r in report["cells"] if r["E"]==e]
            axes[j,1].plot([r["capture_samples_per_pixel"] for r in rr],[r[key] for r in rr],"o-",label=f"E{e}")
        for ax in axes[j]: ax.set_ylabel(key); ax.legend()
    axes[1,0].set_xlabel("emitters"); axes[1,1].set_xlabel("subpixel detector samples")
    save("density_fidelity_axes",fig)
    return paths


def run_experiment():
    from .high_sample import _write_csv
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False); CACHE.mkdir(parents=True,exist_ok=True)
    baseline=torch.load("runs/v0812_aliasing/PER_CHART_PLUS_RING_state.pt",weights_only=True)
    old=json.loads(Path("artifacts/v0812_polar_aliasing.json").read_text())
    assert state_digest(baseline)==old["variants"]["PER_CHART_PLUS_RING"]["cache_digest"]
    reference=np.load("runs/v0812_measurement/reference.npy")
    start_time=time.perf_counter()
    scene,template,prepared,initialization_equivalence=sparse_scene(baseline); context=scene["context"]
    centers=prepared.positions.cpu().numpy()
    report={"version":"0.8.13","starting_commit":BASE,"seed_list":[101],"phase":"PER_CHART_PLUS_RING",
        "resolution":list(RESOLUTION),"views":4,"geometry_digest":digest(context.base.grid),
        "reference_digest":digest(torch.from_numpy(reference)),"cells":[],"sources":[],
        "capture_definition":"Cubic B-spline detector response, tensor midpoint quadrature on pixel [-.5,.5]^2, C1=1x1, C2=2x2, C3=4x4. Each translated cubic partition is averaged with weight 1/q^2. No post-render resizing and no source movement. Increased numerical detector integration, not extra independent scene transport observations.",
        "frozen":{"K_chart":1024,"integration_steps":32,"radial":"AREA_STRATIFIED","micro":1,"packet_radius_over_h":1,"epsilon_over_h":1,"path_step_over_h":.5,"launch_start_over_h":2.1,"gate_width":.05,"ambient":.35,"sensor_gain":1.5,"window":2.8},
        "decision_rules":{"material_fidelity":"Nonincreased MSE and either >=1% lower vector-gradient squared error, or >=20% lower unmatched gradient energy together with lower vector-gradient error and increased gradient cosine, averaged across the other axis.","artifact_reduction":"At least 20% lower fixed radial-banding score (rings) or unmatched gradient energy (spoke/texture proxy); inspect matched crops separately.","source_undersampling":"More than 5% of reference foreground pixel centers have no emitter center within .5px at E3.","capture_undersampling":"At E3, C2->C3 image relative L2 exceeds 1%; numerical convergence flag, not scene information rank."},
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR":True,"geometry_optimization":False,"birth":False,
        "initialization_equivalence":initialization_equivalence,
        "sparse_implementation":"Existing grid geodesic mapper and finite-packet streamer; row-streamed exact fourth-neighbor chart spacing, cached chart masses, KD-tree detector locality. No all-point x all-chart/basis tensor, no autograd graph. Geodesic point chunks <=8192; detector chunks=8192 with 16 cubic taps per subpixel."}
    for ei,n in enumerate(LEVELS,1):
        state=source_state(scene,template,prepared,n,baseline); identity=state_digest(state)
        mass=state["weights"].reshape(1024,n*n).sum(1); base_mass=baseline["weights"].reshape(1024,1024).sum(1)
        source={"E":ei,"K_chart":1024,"N_theta":n,"N_r":n,"emitters_per_chart":n*n,"total_emitters":len(state["positions"]),
            "state_digest":identity,"source_mass":state["source_mass"],"weight_sum":float(state["weights"].sum()),
            "maximum_chart_mass_difference":float((mass-base_mass).abs().max()),"runtime":state.get("density_runtime",{"mapping_seconds":0,"transport_seconds":0,"cache_reused":True})}
        report["sources"].append(source)
        stats=density_stats(state,reference)
        for ci,capture in enumerate(CAPTURE,1):
            started=time.perf_counter(); frames,occupancy,accounting,timing=render(state,capture)
            np.save(CACHE/f"E{ei}C{ci}_rgb.npy",np.stack(frames)); np.save(CACHE/f"E{ei}C{ci}_occupancy.npy",np.stack(occupancy))
            metric=_image_metrics(frames,list(reference)); aware=artifact_metrics(frames,reference)
            metric["historical_squared_image_energy"]=metric["total_image_energy"]
            metric["total_image_energy"]=float(sum(np.sum(a,dtype=float) for a in frames))
            if not any(np.any(r["codes"]==5) for r in masks(reference)): metric["interior_gt8px_mse"]=None
            score=artifacts_score(frames,centers,state)
            row={**source,"C":ci,"capture_grid":capture,"capture_samples_per_pixel":capture**2,**metric,
                **{k:v for k,v in aware.items() if k!="per_view"},"gradient_per_view":aware["per_view"],
                "high_frequency_response":aware["gradient_magnitude_ratio"],"artifact_scores":score,
                "foreground_pixel_center_density":stats[1],"capture_location_density":stats[capture],
                "energy_accounting":accounting,**timing,"cell_analysis_seconds":time.perf_counter()-started}
            report["cells"].append(row)
            if ei==1 and ci==1:
                old_rgb=np.load("runs/v0812_aliasing/PER_CHART_PLUS_RING_rgb.npy")
                delta=np.stack(frames)-old_rgb
                report["baseline_replay"]={"relative_l2":float(np.linalg.norm(delta)/np.linalg.norm(old_rgb)),"max_absolute_error":float(abs(delta).max())}
            assert state_digest(state)==identity
            write_json(CACHE/"partial.json",report)
            print(json.dumps({"cell":f"E{ei}C{ci}","mse":metric["whole_image_mse"],"gradient_error":aware["gradient_squared_error"],**timing}),flush=True)
        del state
    cells=report["cells"]
    def cell(e,c): return next(r for r in cells if r["E"]==e and r["C"]==c)
    def change(axis,key):
        return float(np.mean([cell(3,j)[key]/cell(1,j)[key] if axis=="E" else cell(j,3)[key]/cell(j,1)[key] for j in range(1,4)]))
    improvements,_=classify_fidelity(report)
    def artifact_change(axis,key):
        return float(np.mean([cell(3,j)["artifact_scores"][key]/cell(1,j)["artifact_scores"][key] if axis=="E" else cell(j,3)["artifact_scores"][key]/cell(j,1)["artifact_scores"][key] for j in range(1,4)]))
    best=min(cells,key=lambda r:(r["gradient_squared_error"],r["whole_image_mse"]))
    a=np.load(CACHE/"E3C2_rgb.npy"); b=np.load(CACHE/"E3C3_rgb.npy")
    capture_change=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    primary="JOINT_SOURCE_AND_CAPTURE_DENSITY" if all(improvements.values()) else "SOURCE_EMITTER_DENSITY" if improvements["E"] else "CAPTURE_DENSITY" if improvements["C"] else "OTHER"
    report["verdicts"]={"EMITTER_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY":improvements["E"],"CAPTURE_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY":improvements["C"],
        "EMITTER_DENSITY_INCREASE_REDUCES_FLOWER_ARTIFACT":change("E","unmatched_gradient_energy")<.8,
        "EMITTER_DENSITY_INCREASE_REDUCES_RING_ARTIFACT":artifact_change("E","radial_banding_score")<.8,
        "CAPTURE_DENSITY_INCREASE_REDUCES_FLOWER_ARTIFACT":change("C","unmatched_gradient_energy")<.8,
        "CAPTURE_DENSITY_INCREASE_REDUCES_RING_ARTIFACT":artifact_change("C","radial_banding_score")<.8,
        "THIN_PROXY_STILL_ALIASING_CONTAMINATED":any(r["thin_feature_response_ratio"]<cell(1,1)["thin_feature_response_ratio"] and r["whole_image_mse"]<cell(1,1)["whole_image_mse"] and r["unmatched_gradient_energy"]<.8*cell(1,1)["unmatched_gradient_energy"] for r in cells),
        "FULLHD_SOURCE_OCCUPANCY_STILL_UNDERSAMPLED":float(np.mean([r["neighbors"]["0.5"]["zero_fraction"] for r in cell(3,1)["foreground_pixel_center_density"]]))>.05,
        "FULLHD_CAPTURE_SAMPLING_STILL_UNDERSAMPLED":capture_change>.01,"BEST_CONFIGURATION":f"E{best['E']}C{best['C']}","PRIMARY_LIMITATION":primary}
    report["axis_evidence"]={a:{"gradient_error_ratio":change(a,"gradient_squared_error"),"mse_ratio":change(a,"whole_image_mse"),"unmatched_hf_ratio":change(a,"unmatched_gradient_energy"),"radial_banding_ratio":artifact_change(a,"radial_banding_score")} for a in ("E","C")}
    refresh_classification(report)
    report["capture_C2_C3_relative_l2_at_E3"]=capture_change
    report["limitations"]=["Reference definition frozen: 4096 hard-visible weighted splats, not dense radiance; empty >8px interiors remain null.","Hierarchical chart measure bias is unchanged; denser quadrature cannot remove it.","Geometry/grid bandwidth is fixed and not independently resolved by this matrix.","Micro=1 finite-packet attenuation remains fixed; this matrix cannot rule out transport formulation bias.","Capture quadrature averages the same cubic response; it converges pixel integration, cannot recover independent geometry observations, and may modestly smooth rather than sharpen.","Artifact scalar proxies also respond to curvature and overlap; they are not isolated counts of visual flowers or rings.","No independent-seed ensemble: deterministic phase and seed 101 only. Classification is descriptive, using declared thresholds."]
    report["figures"]=figures(report,reference,centers,baseline)
    report["runtime_seconds"]=time.perf_counter()-start_time
    report["commands"]=["PYTHONPATH=src python demo.py --density-matrix","PYTHONPATH=src python scripts/validate_v0813.py"]
    write_json(Path("artifacts/v0813_density_matrix.json"),report)
    _write_csv(Path("artifacts/v0813_density_matrix.csv"),_scalar_csv_rows(report))
    print(json.dumps(report["verdicts"],indent=2),flush=True)
    return report


if __name__=="__main__": run_experiment()
