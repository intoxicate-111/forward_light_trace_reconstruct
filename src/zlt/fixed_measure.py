"""v0.8.14: global reference-surface measure versus geodesic chart measure."""
from __future__ import annotations
import json
import math
import time
import resource
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from .density_matrix import sparse_scene,render,capture_splat,artifacts_score
from .measurement_bandwidth import state_digest,masks,gate,lobe
from .polar_aliasing import phase_coordinates,artifact_metrics,project
from .finite_packet import _image_metrics,_scalar_csv_rows,_transmission
from .transverse_packet import digest,write_json,trace_family
from .locality import wendland_values,wendland_gradients
from .meshfree_surface import meshfree_base_color

BASE="975f820fa7cba98a49c3f27fb6e3f487187a6464"
CACHE=Path("runs/v0814_fixed_measure")


def diagnostic_ids(count,size,seed=103):
    """Never decimate a digital Sobol sequence with a power-of-two stride."""
    return np.sort(np.random.default_rng(seed).choice(count,min(count,size),replace=False))


class OneCoefficientField:
    """One selected coordinate of the unchanged 64-basis field; other λ=0.

    Compact support evaluated as point-basis pairs, never a point x 64 array.
    """
    def __init__(self,base,center,radius,coefficient):
        self.base=base; self.center=center; self.radius=radius; self.coefficient=coefficient
        self.lower,self.upper=base.lower,base.upper
    def value(self,p):
        d=p-self.center; ids=torch.nonzero((d*d).sum(1)<self.radius**2).flatten()
        value=self.base.value(p)
        return value.index_add(0,ids,self.coefficient*wendland_values(d[ids],self.radius.expand(len(ids))))
    def gradient(self,p):
        d=p-self.center; ids=torch.nonzero((d*d).sum(1)<self.radius**2).flatten()
        return self.base.gradient(p).index_add(0,ids,self.coefficient*wendland_gradients(d[ids],self.radius.expand(len(ids))))


def attach(field,points,normals,iterations=6):
    """Fixed baseline normal lines, fixed identities/weights; no chart regeneration."""
    t=torch.zeros_like(points[:,0])
    for _ in range(iterations):
        x=points+t[:,None]*normals
        denom=(field.gradient(x)*normals).sum(1)
        denom=torch.where(denom.abs()>1e-10,denom,torch.full_like(denom,1e-10))
        t=t-field.value(x)/denom
    x=points+t[:,None]*normals
    n=field.gradient(x); n=n/torch.linalg.vector_norm(n,dim=1,keepdim=True).clamp_min(1e-30)
    return x,n


def global_source(scene,baseline):
    path=CACHE/"global_state.pt"
    if path.exists():
        state=torch.load(path,weights_only=True)
        assert len(state["positions"])==len(baseline["positions"])
        return state
    from skimage.measure import marching_cubes
    base=scene["context"].base
    grid=base.grid.cpu().numpy(); lower=base.lower.cpu().numpy(); upper=base.upper.cpu().numpy()
    spacing=np.broadcast_to((upper-lower)/(np.asarray(grid.shape)-1),(3,))
    verts,faces,_,_=marching_cubes(grid,0,spacing=tuple(spacing)); verts=verts+lower
    triangles=verts[faces]; areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    cdf=np.cumsum(areas); count=len(baseline["positions"])
    sequence=torch.quasirandom.SobolEngine(3,scramble=True,seed=101).draw(count,dtype=torch.float64).numpy()
    face_ids=np.searchsorted(cdf,sequence[:,0]*cdf[-1]); u=np.sqrt(sequence[:,1]); v=sequence[:,2]
    bary=np.stack((1-u,u*(1-v),u*v),1)
    points=np.einsum("ni,nij->nj",bary,triangles[face_ids])
    xs=[]; ns=[]; residual=0.; displacement=0.; start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,count,8192):
            p=torch.as_tensor(points[i:i+8192],device="cuda",dtype=torch.float64)
            normal=base.gradient(p); normal/=normal.norm(dim=1,keepdim=True).clamp_min(1e-30)
            x,n=attach(base,p,normal)
            residual=max(residual,float(base.value(x).abs().max())); displacement=max(displacement,float((x-p).norm(dim=1).max()))
            xs.append(x.cpu()); ns.append(n.cpu())
    state=dict(baseline,positions=torch.cat(xs),normals=torch.cat(ns),weights=torch.full((count,),float(baseline["weights"].sum())/count,dtype=torch.float64))
    state["layout_digest"]=digest(torch.from_numpy(np.stack((face_ids,sequence[:,1],sequence[:,2]),1)))
    state["global_measure"]={"definition":"Uniform area on fixed marching-cubes polyhedral F0=0 surface, pushed to trilinear F0 by fixed-normal attachment. Under λ, push forward the same reference measure on persistent normal lines; NOT equal area on every deformed surface.",
        "triangles":len(faces),"polyhedral_area":float(cdf[-1]),"seed":101,"sample_ids":"Sobol index, fixed face ID and barycentric coordinate",
        "face_ids_digest":digest(torch.from_numpy(face_ids)),"barycentric_digest":digest(torch.from_numpy(bary)),
        "max_projection_displacement":displacement,"max_surface_residual":residual,"attachment_seconds":time.perf_counter()-start}
    torch.save({"face_ids":torch.from_numpy(face_ids),"barycentric":torch.from_numpy(bary),"vertices":torch.from_numpy(verts),"faces":torch.from_numpy(faces.copy())},CACHE/"global_identity.pt")
    scene["source"]=state
    family=trace_family(scene,CACHE,"global_transport",1.,counts=(1,))
    tau=family["tau"][:,:,0]; state["transmission"]=torch.exp(-tau); state["tau_digest"]=digest(tau)
    state["colors"]=torch.cat([meshfree_base_color(state["positions"][i:i+8192].cuda(),base.lower,base.upper).cpu() for i in range(0,count,8192)])
    state["transport_runtime"]={k:family[k] for k in ("runtime_seconds","peak_cuda_allocated_mib","peak_cuda_reserved_mib","cpu_rss_mib")}
    torch.save(state,path); return state


def surface_density(state,probes):
    tree=cKDTree(state["positions"].numpy())
    distance,ids=tree.query(probes,k=17,workers=1)
    area=np.pi*distance[:,-1]**2
    raw=16/np.maximum(area,1e-30)/len(state["weights"])
    weighted=state["weights"].numpy()[ids[:,1:]].sum(1)/np.maximum(area,1e-30)
    def stats(a): return {"mean":float(a.mean()),"cv":float(a.std()/a.mean()),"p05":float(np.quantile(a,.05)),"p50":float(np.median(a)),"p95":float(np.quantile(a,.95))}
    return {"raw_count_density":stats(raw),"weighted_source_density":stats(weighted),"neighbor_radius":stats(distance[:,-1]),
        "definition":"17-nearest Euclidean query at identical global area probes; discard nearest and use 16 neighbors / pi*r17^2. Thin nearby sheets/curvature bias this local surface-area approximation."},weighted


def density_image(points,weights,state,view=0):
    acc=torch.zeros((1080*1920,1),device="cuda",dtype=torch.float64)
    for i in range(0,len(points),8192):
        p=points[i:i+8192].cuda(); rel=p-state["center"].cuda()
        row=(.5-rel@state["up"][view].cuda()/2.8)*1080-.5
        col=(rel@state["right"][view].cuda()/2.8+.5)*1920-.5
        capture_splat(acc,row,col,weights[i:i+8192,None].cuda(),4)
    return acc.reshape(1080,1920).cpu().numpy()


def lambda_density(scene,template,current,global_state):
    from .sparse_geodesic import prepare_centers
    from .stratified_polar import _shared_ray_map_from_prepared
    context=scene["context"]; theta,rho=phase_coordinates("PER_CHART_PLUS_RING",angles=64,rings=64)
    rng=np.random.default_rng(105)
    local_ids=np.stack([np.sort(rng.choice(4096,64,replace=False)) for _ in range(1024)])
    ids=(np.arange(1024)[:,None]*4096+local_ids).ravel()
    gid=diagnostic_ids(len(global_state["positions"]),65536)
    p0=current["positions"][ids]; n0=current["normals"][ids]
    gp=global_state["positions"][gid]; gn=global_state["normals"][gid]
    weights=current["weights"][ids]; weights=weights/weights.sum()
    gw=global_state["weights"][gid]; gw=gw/gw.sum()
    rows=[]; maps={}; epsilon=1e-3
    with torch.no_grad():
        for parameter in (0,17):
            records={}; moves={}; residuals={}
            for sign in (-1,0,1):
                field=OneCoefficientField(context.base,context.master_layout.centers[parameter],context.master_layout.radii[parameter],torch.tensor(sign*epsilon,device="cuda",dtype=torch.float64))
                prepared=prepare_centers(field,template)
                xs=[]
                for k in range(0,1024,128):
                    tid=torch.arange(k,k+128,device="cuda")
                    t=torch.as_tensor(np.take_along_axis(theta[k:k+128].reshape(128,4096),local_ids[k:k+128],1),device="cuda")
                    r=torch.as_tensor(np.take_along_axis(rho[k:k+128].reshape(128,4096),local_ids[k:k+128],1),device="cuda")[:,:,None]
                    x,_,_=_shared_ray_map_from_prepared(field,template,prepared,tid,t,r,32); xs.append(x.reshape(-1,3).cpu())
                regenerated=torch.cat(xs)
                attached=[]; global_attached=[]
                for start in range(0,len(p0),8192):
                    x,_=attach(field,p0[start:start+8192].cuda(),n0[start:start+8192].cuda()); attached.append(x.cpu())
                    x,_=attach(field,gp[start:start+8192].cuda(),gn[start:start+8192].cuda()); global_attached.append(x.cpu())
                for name,points,w in (("current",regenerated,weights),("frozen",torch.cat(attached),weights),("global",torch.cat(global_attached),gw)):
                    records[name,sign]=density_image(points,w,current)
                    moves[name,sign]=points
                    residuals[name,sign]=max(float(field.value(points[i:i+8192].cuda()).abs().max()) for i in range(0,len(points),8192))
            for name in ("current","frozen","global"):
                diff=(records[name,1]-records[name,-1])/(2*epsilon)
                smooth0=gaussian_filter(records[name,0],2)
                change=gaussian_filter(records[name,1]-records[name,-1],2)
                shift=(moves[name,1]-moves[name,-1])/(2*epsilon)
                tangential=shift-(shift*(gn if name=="global" else n0)).sum(1,keepdim=True)*(gn if name=="global" else n0)
                rows.append({"parameter":parameter,"method":name,"epsilon":epsilon,"samples":len(p0),
                    "density_plus_minus_relative_l2":float(np.linalg.norm(change)/np.linalg.norm(smooth0)),
                    "density_derivative_l2":float(np.linalg.norm(gaussian_filter(diff,2))),
                    "position_derivative_rms":float(shift.square().sum(1).mean().sqrt()),
                    "tangential_derivative_rms":float(tangential.square().sum(1).mean().sqrt()),
                    "mass_difference":float(records[name,1].sum()-records[name,-1].sum()),
                    "max_surface_residual_over_probes":max(residuals[name,s] for s in (-1,0,1)),
                    "sample_ids_digest":digest(torch.from_numpy(gid if name=="global" else ids)),
                    "baseline_regeneration_error":float((moves[name,0]-(gp if name=="global" else p0)).abs().max())})
                maps[f"p{parameter}_{name}"]=diff
            maps[f"p{parameter}_redistribution_excess"]=maps[f"p{parameter}_current"]-maps[f"p{parameter}_frozen"]
            print(f"lambda density p{parameter} complete",flush=True)
    return rows,maps


def closure(scene,state,parameter=17):
    context=scene["context"]; owners=diagnostic_ids(len(state["positions"]),1024,107)
    points=state["positions"][owners].cuda(); normals=state["normals"][owners].cuda()
    weights=state["weights"][owners].cuda(); weights=weights/weights.sum()
    atlas,boundary=scene["atlas"],scene["boundary"]
    center=context.master_layout.centers[parameter]; radius=context.master_layout.radii[parameter]
    def block(coefficient,p,n,w):
        field=OneCoefficientField(context.base,center,radius,coefficient)
        x,normal=attach(field,p,n)
        directions=atlas.directions[0].expand_as(x)
        t,_,_=_transmission(field,x,directions,boundary.exit_times(x,directions),radius=scene["h"],epsilon=scene["h"],path_step=.5*scene["h"],offsets=torch.zeros((1,3),device="cuda",dtype=torch.float64),eta=scene["eta"],kappa=-math.log(.01),surface_barrier=True,launch_exclusion_factor=1.05)
        c=normal@atlas.directions[0]
        energy=w[:,None]*t[:,None]*gate(c,"CURRENT_SOFT_W005")[:,None]*lobe(c,"CURRENT_LOBE")[:,None]*meshfree_base_color(x,field.lower,field.upper)
        rel=x-boundary.center; row=(.5-rel@atlas.up[0]/2.8)*1080-.5; col=(rel@atlas.right[0]/2.8+.5)*1920-.5
        image=torch.zeros((1080*1920,3),device="cuda",dtype=torch.float64)
        capture_splat(image,row,col,energy,4)
        return image.flatten()*(1.5*1080*1920)
    epsilons=(1e-3,3e-4,1e-4,3e-5,1e-5)
    analytic=np.zeros(1080*1920*3); fds={e:np.zeros_like(analytic) for e in epsilons}
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad():
        for start in range(0,len(points),128):
            def function(c): return block(c,points[start:start+128],normals[start:start+128],weights[start:start+128])
            zero=torch.tensor(0.,device="cuda",dtype=torch.float64,requires_grad=True)
            _,jvp=torch.autograd.functional.jvp(function,zero,torch.ones_like(zero),create_graph=False)
            analytic+=jvp.detach().cpu().numpy()
            with torch.no_grad():
                for eps in epsilons:
                    plus=function(zero.detach()+eps); minus=function(zero.detach()-eps)
                    fds[eps]+=((plus-minus)/(2*eps)).cpu().numpy()
            print(f"closure {start+128}/1024",flush=True)
    rows=[]
    for eps,fd in fds.items():
        a=np.linalg.norm(analytic); b=np.linalg.norm(fd); active=np.abs(analytic)>a*1e-6
        if not np.isfinite(analytic).all() or a<=1e-20 or not active.any():
            raise RuntimeError("Nonfinite or null lambda JVP is not a valid closure test")
        rows.append({"epsilon":eps,"cosine":float(np.dot(fd,analytic)/max(a*b,1e-30)),"relative_l2":float(np.linalg.norm(fd-analytic)/max(a,1e-30)),
            "norm_ratio":float(b/max(a,1e-30)),"active_component_sign_agreement":float(np.mean(np.sign(fd[active])==np.sign(analytic[active]))),"analytic_norm":float(a),"fd_norm":float(b)})
    result={"parameter":parameter,"samples":1024,"views":[0],"resolution":[1080,1920],"capture":4,
        "sample_ids_digest":digest(torch.from_numpy(owners)),"id_selection":"1024 fixed random IDs, seed=107; not power-of-two Sobol decimation",
        "precision":"float64 diagnostic; identical transport and cubic subpixel family, 128-emitter AD blocks, one selected λ coordinate",
        "analytic_method":"automatic-differentiation JVP of own fixed-identity forward including normal attachment, normals, color, emission gate, transmission, boundary exit and detector weights",
        "rows":rows,"runtime_seconds":time.perf_counter()-started,"peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "passed":sum(r["cosine"]>.99 and r["relative_l2"]<.05 and .95<r["norm_ratio"]<1.05 for r in rows)>=2}
    np.save(CACHE/"global_geometry_jvp.npy",analytic.reshape(1080,1920,3))
    return result


def density_difference_figure(difference_maps):
    """Zoom to the affected support; shared scales, no invisible full-frame dots."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for row,p in enumerate((0,17)):
        names=("current","frozen","global","redistribution_excess")
        maps=[gaussian_filter(difference_maps[f"p{p}_{name}"],2) for name in names]
        magnitude=np.maximum.reduce([abs(a) for a in maps])
        yy,xx=np.nonzero(magnitude>magnitude.max()*.005)
        y0,y1=max(0,yy.min()-15),min(1080,yy.max()+16)
        x0,x1=max(0,xx.min()-15),min(1920,xx.max()+16)
        limit=float(np.quantile(magnitude[magnitude>magnitude.max()*.005],.99))
        for ax,name,a in zip(axes[row],names,maps):
            ax.imshow(a[y0:y1,x0:x1],cmap="coolwarm",vmin=-limit,vmax=limit,extent=(x0,x1,y1,y0))
            ax.set_title(f"λ{p} {name}",fontsize=9)
            ax.set_xlabel(f"pixel x; shared ±{limit:.3g}",fontsize=8)
        axes[row,0].set_ylabel("pixel y")
    fig.suptitle("dρ/dλ, ε=0.001; Gaussian σ=2 px (same as density metrics); matched support crops")
    return fig


def make_figures(report,states,images,reference,probes,densities,difference_maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]
    def save(name,fig):
        path=f"figures/v0814_{name}.png"; fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,("current","global")):
        p=states[name]["positions"].numpy()[diagnostic_ids(len(states[name]["positions"]),8192)]; ax.scatter(p[:,0],p[:,2],s=.3); ax.set(title=name,aspect="equal")
    save("surface_emitters",fig)
    limit=np.quantile(np.concatenate(list(densities.values())),.98)
    for name in ("current","global"):
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.scatter(probes[:,0],probes[:,2],s=4,c=densities[name],vmin=0,vmax=limit,cmap="magma"); fig.colorbar(im,ax=ax); ax.set(title=f"{name}: weighted surface density",aspect="equal"); save(f"density_{name}",fig)
    save("lambda_density_difference",density_difference_figure(difference_maps))
    fig,axes=plt.subplots(3,4,figsize=(14,9))
    for row,name in enumerate(("current","global","reference")):
        for v in range(4):
            a=reference[v] if name=="reference" else images[name][v]
            axes[row,v].imshow(np.clip(a,0,1)); axes[row,v].set_title(f"{name} view {v}"); axes[row,v].axis("off")
    save("fullhd_comparison",fig)
    old=json.loads(Path("artifacts/v0813_density_matrix.json").read_text())
    crop=next(x for x in old["zoom_crops"] if x["name"]=="ring"); y,x=crop["row"],crop["column"]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name in zip(axes,("current","global","reference")):
        a=reference[0] if name=="reference" else images[name][0]
        ax.imshow(np.clip(a[y-80:y+80,x-80:x+80],0,1)); ax.set_title(name); ax.axis("off")
    save("ring_crop",fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,name in enumerate(("current","global")):
        gray=images[name][0].mean(2); target=reference[0].mean(2)
        grad=np.hypot(*np.gradient(gray)); tgrad=np.hypot(*np.gradient(target))
        for ax,a,title in zip(axes[row],(grad,gray-target,grad-tgrad),("spatial gradient","RGB error","gradient magnitude error")):
            ax.imshow(a,cmap="coolwarm"); ax.set_title(name+" "+title,fontsize=9); ax.axis("off")
    save("gradient_error",fig)
    return paths


def run_experiment():
    from .high_sample import _write_csv
    torch.set_num_threads(8)
    CACHE.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    print("Loading fixed baseline and constructing unchanged scene",flush=True)
    current=torch.load("runs/v0813_density_sparse/source_64.pt",weights_only=True)
    baseline=torch.load("runs/v0812_aliasing/PER_CHART_PLUS_RING_state.pt",weights_only=True)
    with torch.no_grad(): scene,template,prepared,initialization=sparse_scene(baseline)
    print("Constructing/reusing global reference-area source",flush=True)
    global_state=global_source(scene,current)
    # One explicit, geometry-independent radiometric budget normalization, fixed
    # thereafter for every lambda probe. No material or transport retuning.
    target_budget=float((current["weights"][:,None]*current["colors"]).sum())
    original_budget=float((global_state["weights"][:,None]*global_state["colors"]).sum())
    budget_scale=target_budget/original_budget
    global_state=dict(global_state,weights=global_state["weights"]*budget_scale)
    torch.save(global_state,CACHE/"global_energy_matched_state.pt")
    states={"current":current,"global":global_state}
    reference=np.load("runs/v0812_measurement/reference.npy")
    images={"current":list(np.load("runs/v0813_density_sparse/E3C3_rgb.npy"))}
    with torch.no_grad(): global_images,_,energy,timing=render(global_state,4)
    images["global"]=global_images; np.save(CACHE/"global_rgb.npy",np.stack(global_images))
    probes=global_state["positions"].numpy()[diagnostic_ids(len(global_state["positions"]),8192)]
    report={"version":"0.8.14","starting_commit":BASE,"seed_list":[101],"emitters":len(current["positions"]),"views":4,"resolution":[1080,1920],"capture_grid":4,
        "global_measure":global_state["global_measure"],"comparisons":{},"cache_digests":{k:state_digest(v) for k,v in states.items()},
        "gradient_semantics":"v0.8.13 cosine is spatial image gradient ∇pixel I, NOT dI/dλ. External reference geometry-Jacobian columns are not supplied. Own λ-JVP closure is reported separately; no external geometry-gradient recovery is claimed.",
        "lambda_semantics":"1024 chart centers are not 1024 λ parameters. Use the established first 64 geometry bases; coefficient probes 0 and 17, remaining coefficients zero. Reference chart centers/radii and basis centers/support are fixed; projected centers and geodesic trajectories respond to Fλ.",
        "energy_definition":"Equal total RGB emitted source budget sum_i sum_RGB w_i*C_i before view-dependent selection/shading and transport. A single explicit baseline normalization of all global quadrature weights preserves uniform reference area and unchanged material; fixed thereafter under lambda. Per-view post-gate/lobe flux is an outcome, not renormalized.",
        "source_budget_normalization":{"scale":budget_scale,"unscaled_global_rgb":original_budget,"target_rgb":target_budget,"fixed_under_lambda":True},
        "NO_DENSE_POINT_BY_LAMBDA_TENSOR":True,"geometry_optimization":False,"birth":False,
        "configuration":{"K_chart":1024,"N_theta":64,"N_r":64,"phase":"PER_CHART_PLUS_RING","geometry_bases":64,"micro":1,"gate_width":.05,"ambient":.35,"capture":"4x4 midpoint quadrature of unchanged cubic detector"},
        "initialization_equivalence":initialization,
        "energy_accounting":energy,"global_readout_runtime":timing,"transport_runtime":global_state["transport_runtime"]}
    densities={}
    for name,state in states.items():
        print(f"Full-HD metrics and surface density: {name}",flush=True)
        metric=_image_metrics(images[name],list(reference)); aware=artifact_metrics(images[name],reference)
        density,densities[name]=surface_density(state,probes)
        per_view=[]
        for v in range(4):
            m=_image_metrics([images[name][v]],[reference[v]]); m.update(aware["per_view"][v]); m["gradient_relative_l2"]=float(np.sqrt(1+m["gradient_magnitude_ratio"]**2-2*m["gradient_cosine"]*m["gradient_magnitude_ratio"])); per_view.append(m)
        metric.update({k:v for k,v in aware.items() if k!="per_view"})
        metric["gradient_relative_l2"]=float(np.mean([m["gradient_relative_l2"] for m in per_view]))
        metric["historical_squared_image_energy"]=metric.pop("total_image_energy",None)
        for m in per_view:
            m["historical_squared_image_energy"]=m.pop("total_image_energy",None)
        metric.update(surface_density=density,per_view=per_view,source_weight_sum=float(state["weights"].sum()),raw_source_mass=state["source_mass"],
            colored_source_budget=float((state["weights"][:,None]*state["colors"]).sum()),total_rendered_rgb=float(sum(np.sum(i,dtype=float) for i in images[name])),
            artifact_scores=artifacts_score(images[name],prepared.positions.cpu().numpy(),state))
        fg=masks(reference); occupancy=[]
        metric["interior_gt8px_mse"]=None  # Historical Full-HD sparse reference has no >8px interior.
        for v,m in enumerate(per_view):
            m["interior_gt8px_mse"]=None
            c=state["normals"]@state["directions"][v]
            base_energy=state["weights"][:,None]*state["colors"]
            m["emitted_rgb_before_view_selection"]=float(base_energy.sum())
            m["pre_transmission_rgb_after_gate_lobe"]=float((base_energy*gate(c,"CURRENT_SOFT_W005")[:,None]*lobe(c,"CURRENT_LOBE")[:,None]).sum())
            m["total_rendered_rgb"]=float(np.sum(images[name][v],dtype=float))
        for v in range(4):
            d=cKDTree(project(state["positions"].numpy(),state,v)).query(np.argwhere(fg[v]["foreground"]),workers=1)[0]
            occupancy.append({"view":v,"nearest_px_p50":float(np.median(d)),"nearest_px_p95":float(np.quantile(d,.95)),"no_center_within_0_5px":float(np.mean(d>.5))})
        metric["occupancy"]=occupancy; report["comparisons"][name]=metric
    write_json(CACHE/"forward_partial.json",report)
    report["lambda_density"],difference_maps=lambda_density(scene,template,current,global_state)
    np.savez_compressed(CACHE/"lambda_density_maps.npz",**difference_maps)
    write_json(CACHE/"density_partial.json",report)
    report["closure"]=closure(scene,global_state)
    a=report["comparisons"]["current"]; b=report["comparisons"]["global"]
    alignment=b["gradient_cosine"]>a["gradient_cosine"]+.05
    correlations=[]
    for name,state in states.items():
        with torch.no_grad(): den=density_image(state["positions"],state["weights"],state)
        gray=images[name][0].mean(2)
        d=den-gaussian_filter(den,4); residual=gray-gaussian_filter(gray,4)
        mask=gaussian_filter(den,4)>den.max()*1e-4
        correlations.append({"method":name,"view":0,"density_rgb_highpass_correlation":float(np.corrcoef(d[mask],residual[mask])[0,1]),"scope":"Matched high-pass fields, sigma=4px, source support mask; not raw object-occupancy correlation or proof of causation."})
    report["density_residual_correlations"]=correlations
    current_rows=[r for r in report["lambda_density"] if r["method"]=="current"]
    global_rows=[r for r in report["lambda_density"] if r["method"]=="global"]
    strong_redistribution=all(c["density_derivative_l2"]>2*g["density_derivative_l2"] and c["density_plus_minus_relative_l2"]>.01 for c,g in zip(current_rows,global_rows))
    report["verdicts"]={"LAMBDA_CENTERED_EMISSION_HAS_STRONG_DENSITY_NONUNIFORMITY":a["surface_density"]["weighted_source_density"]["cv"]>.5,
        "LAMBDA_PERTURBATION_CHANGES_EMITTER_DENSITY":max(r["density_plus_minus_relative_l2"] for r in current_rows)>1e-5,
        "GLOBAL_FIXED_MEASURE_REDUCES_DENSITY_NONUNIFORMITY":b["surface_density"]["weighted_source_density"]["cv"]<.8*a["surface_density"]["weighted_source_density"]["cv"],
        "GLOBAL_FIXED_MEASURE_IMPROVES_REFERENCE_GRADIENT_ALIGNMENT":alignment,
        "FROZEN_EMITTER_LAYOUT_IMPROVES_REFERENCE_GRADIENT_ALIGNMENT":False,
        "GLOBAL_FORWARD_GRADIENT_PASSES_FINITE_DIFFERENCE_CLOSURE":report["closure"]["passed"],
        "RING_FLOWER_STRUCTURE_CORRELATES_WITH_EMITTER_DENSITY":correlations[0]["density_rgb_highpass_correlation"]>.5,
        "LAMBDA_DEPENDENT_EMITTER_REDISTRIBUTION_IS_PRIMARY_LIMITATION":alignment and strong_redistribution,
        "PRIMARY_LIMITATION":"LAMBDA_DEPENDENT_EMITTER_MEASURE" if alignment and strong_redistribution else "OTHER"}
    report["interpretation"]={"thresholds":{"strong_density_cv":.5,"density_cv_reduction_ratio":.8,"material_spatial_gradient_cosine_gain":.05,"strong_redistribution":"current density derivative >2x global AND ±epsilon relative density change >1%, both selected parameters","correlation":.5},
        "frozen_layout":"Frozen reference alignment is NOT TESTED; required boolean false means improvement not established, not evidence of no effect. Frozen normal attachment is only a +/-epsilon density control. Its lambda=0 reattachment residual relative to the approximate geodesic endpoints is explicitly reported; it is not assumed identically zero.",
        "causal_limit":"A moving surface changes Eulerian density even with a fixed material measure. Current-minus-frozen density derivatives and tangential motions help separate geodesic redistribution from unavoidable normal shape motion. Smoother images alone never establish primary λ-measure failure.",
        "remaining_alternatives":["Sparse hard-reference measure mismatch","Unchanged geometry/grid bandwidth","Unchanged attenuation/visibility formulation","Chart overlap and non-area geodesic quadrature"]}
    report["figures"]=make_figures(report,states,images,reference,probes,densities,difference_maps)
    report["runtime_seconds"]=time.perf_counter()-started; report["cpu_rss_peak_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report["commands"]=["PYTHONPATH=src python -m zlt.fixed_measure","PYTHONPATH=src python scripts/validate_v0814.py"]
    write_json(Path("artifacts/v0814_fixed_measure.json"),report); _write_csv(Path("artifacts/v0814_fixed_measure.csv"),_scalar_csv_rows(report))
    print(json.dumps(report["verdicts"],indent=2),flush=True); return report


if __name__=="__main__": run_experiment()
