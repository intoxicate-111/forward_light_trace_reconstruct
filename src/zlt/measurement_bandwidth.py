"""v0.8.12: measurement-only replay of immutable v0.8.11 scene transport."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .finite_packet import _cubic, _image_metrics, _scalar_csv_rows
from .pixel_diagnostic import _silhouette_regions
from .transverse_packet import digest, write_json

BASE = "3dd93c8672c432c007290c0f25facc518cfbb0a7"
RESOLUTION = (1080, 1920)
KERNELS = ("NEAREST_1X1", "BILINEAR_2X2", "CUBIC_4X4")
GATES = {"CURRENT_SOFT_W005": .05, "SOFT_W010": .10, "SOFT_W002": .02,
         "SOFT_W001": .01, "SOFT_W0005": .005, "HARD_OUTWARD": None,
         "NO_OUTWARD_GATE": None}
PRIMARY = ("CURRENT_SOFT_W005", "SOFT_W002", "SOFT_W001", "HARD_OUTWARD", "NO_OUTWARD_GATE")
LOBES = ("CURRENT_LOBE", "CONSTANT_RADIANCE", "AMBIENT_ONLY", "COSINE_ONLY")
BIN_EDGES = (0., .005, .01, .02, .05, .1, .25, 1.0000001)


def gate(c, name):
    if name == "HARD_OUTWARD": return (c > 0).to(c.dtype)
    if name == "NO_OUTWARD_GATE": return torch.ones_like(c)
    return 1 - torch.exp(-c.clamp_min(0).square()/GATES[name]**2)


def lobe(c, name):
    if name == "CONSTANT_RADIANCE": return torch.ones_like(c)
    if name == "AMBIENT_ONLY": return torch.full_like(c, .35)
    if name == "COSINE_ONLY": return c.clamp_min(0)
    return .35 + .65*c.clamp_min(0)


def kernel_pairs(row, column, kernel, resolution):
    """Pixel centers are integer coordinates, shared by all three kernels."""
    height, width = resolution
    if kernel == "NEAREST_1X1":
        ri, ci = torch.floor(row[:, None]+.5).long(), torch.floor(column[:, None]+.5).long()
        rw, cw = torch.ones_like(ri, dtype=row.dtype), torch.ones_like(ci, dtype=row.dtype)
    else:
        offsets = torch.arange(2, device=row.device) if kernel == "BILINEAR_2X2" else torch.arange(-1, 3, device=row.device)
        ri = torch.floor(row).long()[:, None]+offsets
        ci = torch.floor(column).long()[:, None]+offsets
        if kernel == "BILINEAR_2X2":
            rw = (1-(row[:, None]-ri).abs()).clamp_min(0)
            cw = (1-(column[:, None]-ci).abs()).clamp_min(0)
        elif kernel == "CUBIC_4X4":
            rw, cw = _cubic(row[:, None]-ri), _cubic(column[:, None]-ci)
        else:
            raise ValueError(kernel)
    n = len(row)
    rr = ri[:, :, None].expand(-1, ri.shape[1], ci.shape[1]).reshape(n, -1)
    cc = ci[:, None, :].expand(-1, ri.shape[1], ci.shape[1]).reshape(n, -1)
    valid = (rr>=0)&(rr<height)&(cc>=0)&(cc<width)
    weights = (rw[:, :, None]*cw[:, None, :]).reshape(n, -1)*valid
    pixels = rr.clamp(0,height-1)*width+cc.clamp(0,width-1)
    return pixels, weights


def splat(image, positions, energy, right, up, center, kernel, resolution=RESOLUTION):
    relative = positions-center
    row = (.5-relative@up/2.8)*resolution[0]-.5
    column = (relative@right/2.8+.5)*resolution[1]-.5
    pixels, weights = kernel_pairs(row, column, kernel, resolution)
    image.index_add_(0, pixels.flatten(), (weights[..., None]*energy[:, None]).reshape(-1, 3))


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(digest(value).encode() if isinstance(value, torch.Tensor) else json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def frozen_state(cache):
    """Only acquisition builds context; production transport is never invoked."""
    path = cache/"scene.pt"
    if path.exists():
        state = torch.load(path, weights_only=True)
    else:
        from .transverse_packet import build_scene, readout, reference_images
        from .meshfree_surface import meshfree_base_color
        old_cache = Path("runs/v0811_transverse")
        if not (old_cache/"source.pt").exists() or not (old_cache/"radius_1.pt").exists():
            raise RuntimeError("Exact v0.8.11 caches required; this diagnostic does not regenerate transport.")
        scene = build_scene(old_cache)
        old = json.loads(Path("artifacts/v0811_transverse_packet_integration.json").read_text())
        source = scene["source"]
        for key in ("positions_digest", "normals_digest", "weights_digest", "layout_digest"):
            if source[key] != old["source"][key]: raise RuntimeError(f"source mismatch: {key}")
        family = torch.load(old_cache/"radius_1.pt", weights_only=True)
        assert family["settings"]["radius_over_h"] == 1 and family["settings"]["epsilon_over_h"] == 1
        tau = family["tau"][:, :, family["settings"]["micro_counts"].index(1)]
        colors = []
        for start in range(0, len(source["positions"]), 8192):
            colors.append(meshfree_base_color(source["positions"][start:start+8192].cuda(),
                scene["field"].lower, scene["field"].upper).cpu())
        state = {key:source[key] for key in ("positions", "normals", "weights")}
        state.update(colors=torch.cat(colors), transmission=torch.exp(-tau),
            directions=scene["atlas"].directions.cpu(), right=scene["atlas"].right.cpu(),
            up=scene["atlas"].up.cpu(), center=scene["boundary"].center.cpu(),
            source_mass=source["source_mass"], layout_digest=source["layout_digest"], micro=1,
            resolution=list(RESOLUTION), source_commit=BASE, surface_h=scene["h"],
            tau_digest=digest(tau))
        np.save(cache/"reference.npy", np.stack(reference_images(scene, RESOLUTION)))
        np.save(cache/"original_replay.npy", np.stack(readout(scene, tau, RESOLUTION)))
        torch.save(state, path)
    return state, np.load(cache/"reference.npy"), np.load(cache/"original_replay.npy")


def replay(state, gate_name, kernel, lobe_name="CURRENT_LOBE", projected="NONE", transmission_off=False):
    images = []
    for view in range(4):
        image = torch.zeros((math.prod(RESOLUTION), 3), device="cuda", dtype=torch.float32)
        for start in range(0, len(state["positions"]), 8192):
            stop = start+8192
            x, n = state["positions"][start:stop], state["normals"][start:stop]
            c = n@state["directions"][view]
            t = torch.ones_like(c) if transmission_off else state["transmission"][view,start:stop]
            energy = state["weights"][start:stop]*t*gate(c, gate_name)*lobe(c, lobe_name)
            if projected == "PROJECTED_AREA": energy *= c.abs()
            elif projected == "POSITIVE_PROJECTED_AREA": energy *= c.clamp_min(0)
            energy = energy[:,None]*state["colors"][start:stop]
            splat(image, x.float(), energy.float(), state["right"][view].float(),
                  state["up"][view].float(), state["center"].float(), kernel)
        images.append((image*1.5*math.prod(RESOLUTION)).reshape(*RESOLUTION,3).cpu().numpy())
    return images


def masks(reference):
    result = []
    for target in reference:
        fg, _, codes = _silhouette_regions(target)
        gy, gx = np.gradient(target.mean(2).astype(np.float64))
        gradient = np.hypot(gx, gy)
        thin = gradient >= np.quantile(gradient[fg], .95)
        result.append(dict(foreground=fg, codes=codes, gradient=gradient, thin=thin,
                           silhouette=(codes>=1)&(codes<=4)))
    return result


def band_metrics(images, reference, regions, grazing=None):
    rows = []
    for view, (image, target, region) in enumerate(zip(images, reference, regions)):
        dy, dx = np.gradient(image.mean(2).astype(np.float64))
        gradient = np.hypot(dx, dy)
        for code, label in enumerate(("0-1", "1-2", "2-4", "4-8", ">8 interior"), 1):
            mask = region["codes"]==code
            count = int(mask.sum())
            current = float(gradient[mask].mean()) if count else 0.
            ref = float(region["gradient"][mask].mean()) if count else 0.
            rows.append({"view":view, "band":label, "pixels":count,
                "mse":float(np.square(image[mask]-target[mask]).mean()) if count else None,
                "current_gradient":current if count else None, "reference_gradient":ref if count else None,
                "response_ratio":current/ref if ref>0 else None,
                "grazing_energy_fraction":float(grazing[view][mask].sum()/max(image[mask].sum(),1e-30)) if grazing is not None and count else None})
    return rows


def impulse_diagnostic():
    frequency = torch.linspace(0, .5, 129, dtype=torch.float64)
    frac = (torch.arange(16, dtype=torch.float64)+.5)/16
    a, b = torch.meshgrid(frac, frac, indexing="ij")
    row, col = 8+a.flatten(), 8+b.flatten()
    results, captures = [], {}
    for kernel in KERNELS:
        pixels, weights = kernel_pairs(row, col, kernel, (20,20))
        dr, dc = pixels//20-row[:,None], pixels%20-col[:,None]
        r2 = dr.square()+dc.square()
        radii = r2.sqrt().flatten(); mass = weights.flatten()/len(row)
        order = radii.argsort(); cum = mass[order].cumsum(0)
        p95 = radii[order][torch.searchsorted(cum, cum.new_tensor(.95))]
        curves = []
        for distance in (dc, (dc+dr)/math.sqrt(2)):
            response = (weights[:,:,None]*torch.exp(-2j*math.pi*distance[:,:,None]*frequency)).sum(1)
            curves.append({"mean_magnitude":response.abs().mean(0).tolist(),
                "coherent_mean_magnitude":response.mean(0).abs().tolist()})
        psf = torch.zeros(400,dtype=torch.float64)
        psf.index_add_(0,pixels[85],weights[85]); captures[kernel]=psf.reshape(20,20).numpy()
        results.append({"kernel":kernel, "subpixel_samples":256,
            "partition_unity_max_error":float((weights.sum(1)-1).abs().max()),
            "rms_footprint_pixels":float((r2*weights).sum(1).mean().sqrt()),
            "p95_footprint_radius":float(p95), "second_moment":float((r2*weights).sum(1).mean()),
            "mean_peak":float(weights.max(1).values.mean()), "frequency":frequency.tolist(),
            "mtf_x":curves[0], "mtf_diagonal":curves[1]})
    return results, captures


def contribution_diagnostic(state, regions):
    """Bin contributions are summed with the actual unchanged cubic detector."""
    rows, captures, grazing, positive_rows = [], {}, [], []
    for view in range(4):
        c = state["normals"]@state["directions"][view]
        bins = torch.bucketize(c.abs(), c.new_tensor(BIN_EDGES[1:-1]), right=True)
        base = state["weights"]*state["transmission"][view]*lobe(c,"CURRENT_LOBE")
        base = base[:,None]*state["colors"]
        current_gate = gate(c,"CURRENT_SOFT_W005")
        grazing_image = np.zeros((*RESOLUTION,3),np.float32)
        for bin_id in range(7):
            ids = torch.nonzero(bins==bin_id).flatten()
            row = {"view":view,"bin":bin_id,"abs_cosine_range":list(BIN_EDGES[bin_id:bin_id+2]),
                "emitter_count":len(ids),"source_mass":float(state["weights"][ids].sum()/state["weights"].sum())*state["source_mass"],
                "pre_gate_energy":float(base[ids].sum()),"post_gate_energy":float((base[ids]*current_gate[ids,None]).sum())}
            row["suppression_ratio"] = 1-row["post_gate_energy"]/max(row["pre_gate_energy"],1e-30)
            for phase in ("pre", "post"):
                acc = torch.zeros((math.prod(RESOLUTION),3),device="cuda",dtype=torch.float32)
                for start in range(0,len(ids),8192):
                    selected=ids[start:start+8192]
                    energy=base[selected]*(current_gate[selected,None] if phase=="post" else 1)
                    splat(acc,state["positions"][selected].float(),energy.float(),state["right"][view].float(),
                          state["up"][view].float(),state["center"].float(),"CUBIC_4X4")
                image=(acc*1.5*math.prod(RESOLUTION)).reshape(*RESOLUTION,3).cpu().numpy()
                row[phase+"_thin_energy"]=float(image[regions[view]["thin"]].sum())
                row[phase+"_silhouette_energy"]=float(image[regions[view]["silhouette"]].sum())
                if phase=="post" and bin_id<4: grazing_image+=image
            rows.append(row)
        grazing.append(grazing_image)
        # Positive-cosine maps requested separately from absolute-cosine bins.
        for index,(lo,hi) in enumerate(((0,.01),(.01,.02),(.02,.05),(.05,.1),(.1,1.0000001))):
            ids=torch.nonzero((c>lo if lo==0 else c>=lo)&(c<hi)).flatten()
            positive_row={"view":view,"cosine_range":[lo,hi],"emitter_count":len(ids)}
            for phase in ("pre","post"):
                acc=torch.zeros((math.prod(RESOLUTION),3),device="cuda",dtype=torch.float32)
                for start in range(0,len(ids),8192):
                    selected=ids[start:start+8192]
                    energy=base[selected]*(current_gate[selected,None] if phase=="post" else 1)
                    splat(acc,state["positions"][selected].float(),energy.float(),state["right"][view].float(),
                        state["up"][view].float(),state["center"].float(),"CUBIC_4X4")
                image=(acc*1.5*math.prod(RESOLUTION)).reshape(*RESOLUTION,3).cpu().numpy()
                positive_row[phase+"_thin_energy"]=float(image[regions[view]["thin"]].sum())
                positive_row[phase+"_total_energy"]=float(image.sum())
                if view==0:
                    captures[f"{phase}_{index}"]=image*regions[view]["thin"][...,None]
            positive_rows.append(positive_row)
    return rows,captures,grazing,positive_rows


def reference_forensic(reference, regions):
    return {"measure_consistency":"unresolved", "reference_sample_count":4096,
        "production_sample_count":1048576,
        "reference_definition":"finite_packet._hard_images takes corrected_birth._render_cell.numerator, NOT its normalized/clipped image; hard visible owners, equal 1/4096 sample masses, cubic splats, sensor_gain*H*W scaling",
        "production_definition":"hierarchical surface-side mass / reference_area, soft transmission, outward gate, cubic splats; no projected-area Jacobian",
        "interpretation":"Both are additive sample measures, not raster occupancy or per-pixel normalized visible radiance. Different source sampling/weights and visibility mean equality of underlying continuum measures is not established. No reference was changed.",
        "sources":["src/zlt/finite_packet.py::_hard_images","src/zlt/corrected_birth.py::_render_cell","src/zlt/corrected_birth.py::_visibility_cells"],
        "foreground_pixels":[int(r["foreground"].sum()) for r in regions],
        "interior_gt8_pixels":[int((r["codes"]==5).sum()) for r in regions],
        "reference_maxima":[float(x.max()) for x in reference]}


def figures(report, images, psfs, contributions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths=[]
    def save(name,fig):
        p=Path("figures")/f"v0812_{name}.png"; fig.tight_layout(); fig.savefig(p,dpi=140); plt.close(fig); paths.append(str(p))
    fig,ax=plt.subplots(figsize=(7,4)); c=torch.linspace(-.05,.15,501)
    for name in GATES: ax.plot(c,gate(c,name),label=name)
    ax.legend(fontsize=7); ax.set(xlabel="n dot omega",ylabel="gate"); save("emission_gate_curves",fig)
    fig,ax=plt.subplots(figsize=(8,4)); rows=[x for x in report["cosine_bins"] if x["view"]==0]
    x=np.arange(7); ax.bar(x-.2,[r["pre_gate_energy"] for r in rows],.4,label="pre gate"); ax.bar(x+.2,[r["post_gate_energy"] for r in rows],.4,label="post gate")
    ax.set(xticks=x,xticklabels=[str(r["abs_cosine_range"]) for r in rows],ylabel="RGB energy sum (view 0)"); ax.tick_params(axis="x",rotation=30); ax.legend(); save("grazing_energy_suppression",fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,(name,p) in zip(axes,psfs.items()): ax.imshow(p[6:12,6:12],vmin=0,vmax=1); ax.set_title(name)
    save("detector_psf",fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for row in report["detector_impulses"]:
        for ax,key in zip(axes,("mtf_x","mtf_diagonal")):
            ax.plot(row["frequency"],row[key]["mean_magnitude"],label=row["kernel"]); ax.set(xlabel="cycles/pixel",ylabel="mean normalized magnitude",title=key)
    axes[0].legend(); save("detector_mtf",fig)
    primary=[r for r in report["measurements"] if r["primary"]]
    for name,metric in (("gate_detector_heatmap","whole_image_mse"),("thin_response_heatmap","thin_feature_response_ratio")):
        fig,ax=plt.subplots(figsize=(7,5)); data=[[next(r[metric] for r in primary if r["gate"]==g and r["kernel"]==k) for k in KERNELS] for g in PRIMARY]
        im=ax.imshow(data,aspect="auto"); fig.colorbar(im,ax=ax); ax.set(xticks=range(3),xticklabels=KERNELS,yticks=range(5),yticklabels=PRIMARY,title=metric); save(name,fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name,rows in report["silhouette_bands"].items():
        current=[r for r in rows if r["view"]==0]; ax.plot([r["band"] for r in current],[r["response_ratio"] if r["response_ratio"] is not None else np.nan for r in current],"o-",label=name)
    ax.legend(fontsize=7); ax.set(ylabel="gradient response, view 0"); save("silhouette_band_response",fig)
    fig,axes=plt.subplots(2,5,figsize=(15,6))
    for i,phase in enumerate(("pre","post")):
        for j in range(5): axes[i,j].imshow(np.clip(contributions[f"{phase}_{j}"],0,1)); axes[i,j].set_title(f"{phase} positive bin {j}"); axes[i,j].axis("off")
    save("cosine_bin_contributions",fig)
    for name,subset in (("projected_area_control",[r for r in report["measurements"] if r["group"]=="projected"]),("transmission_off_control",[r for r in report["measurements"] if r["group"]=="transport_off"] )):
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(range(len(subset)),[r["thin_feature_response_ratio"] for r in subset]); ax.set(xticks=range(len(subset)),xticklabels=[r["id"] for r in subset],ylabel="thin response"); ax.tick_params(axis="x",rotation=30); save(name,fig)
    fig,axes=plt.subplots(len(images),4,figsize=(15,3*len(images)))
    for i,(name,views) in enumerate(images.items()):
        for v in range(4): axes[i,v].imshow(np.clip(views[v],0,1)); axes[i,v].axis("off"); axes[i,v].set_title(f"{name} / {v}",fontsize=8)
    save("fullhd_measurement_comparison",fig)
    return paths


def run_experiment():
    from .polar_aliasing import artifact_metrics
    if not torch.cuda.is_available(): raise RuntimeError("Full-HD replay requires local CUDA")
    cache=Path("runs/v0812_measurement"); cache.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with torch.no_grad():
        cpu,reference,original=frozen_state(cache)
        frozen_digest=state_digest(cpu)
        state={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
        regions=masks(reference)
        report={"version":"0.8.12","starting_commit":BASE,"cache_digest":frozen_digest,
            "source_configuration":{"K_chart":1024,"N_theta":32,"N_r":32,"radial":"AREA_STRATIFIED","angular_jitter":0,"radial_jitter":0,"micro":1},
            "production_transmission_calls":0,"cache_origin":"exact v0.8.11 source.pt and radius_1.pt micro=1",
            "source_mass":cpu["source_mass"],"resolution":list(RESOLUTION),"views":4,
            "gate_label_disambiguation":"Task G3/G6 share a textual label: actual widths 0.10 and 0.01 are distinct here; G4 is an exact alias of G0 (0.05).",
            "gate_widths":GATES,"reference_analysis":reference_forensic(reference,regions),"measurements":[]}
        impulses,psfs=impulse_diagnostic(); report["detector_impulses"]=impulses
        configs=[]
        for g in GATES: configs.append((f"gate_{g}",g,"CUBIC_4X4","CURRENT_LOBE","NONE",False,"gate"))
        configs.append(("gate_G4_SOFT_W050_alias",PRIMARY[0],"CUBIC_4X4","CURRENT_LOBE","NONE",False,"gate"))
        for g in PRIMARY:
            for k in KERNELS: configs.append((f"primary_{g}_{k}",g,k,"CURRENT_LOBE","NONE",False,"primary"))
        for g in (PRIMARY[0],"HARD_OUTWARD","NO_OUTWARD_GATE"):
            for l in LOBES: configs.append((f"lobe_{g}_{l}",g,"CUBIC_4X4",l,"NONE",False,"lobe"))
        for p in ("NONE","PROJECTED_AREA","POSITIVE_PROJECTED_AREA"):
            configs.append((f"projected_{p}","HARD_OUTWARD","NEAREST_1X1","CONSTANT_RADIANCE",p,False,"projected"))
        for g,k in ((PRIMARY[0],"CUBIC_4X4"),("HARD_OUTWARD","NEAREST_1X1")):
            for off in (False,True): configs.append((f"T{'1' if off else 'real'}_{g}_{k}",g,k,"CURRENT_LOBE","NONE",off,"transport_off"))
        selected_images={}; cached_metrics={}; band_rows={}
        for label,g,k,l,p,off,group in configs:
            key=(g,k,l,p,off)
            if key in cached_metrics:
                metrics=cached_metrics[key]
            else:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
                current=replay(state,g,k,l,p,off); torch.cuda.synchronize()
                replay_seconds=time.perf_counter()-begin
                metrics={**_image_metrics(current,list(reference)),"replay_seconds":replay_seconds,
                    "artifact_aware":artifact_metrics(current,reference),
                    "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                    "cpu_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
                metrics["historical_squared_image_energy"]=metrics["total_image_energy"]
                metrics["total_image_energy"]=float(sum(np.sum(im,dtype=np.float64) for im in current))
                if not any(np.any(region["codes"]==5) for region in regions):
                    metrics["interior_gt8px_mse"]=None
                cached_metrics[key]=metrics
                if g==PRIMARY[0] and k=="CUBIC_4X4" and l=="CURRENT_LOBE" and p=="NONE" and not off:
                    delta=np.stack(current)-original
                    report["replay_equivalence"]={"max_absolute_error":float(np.abs(delta).max()),"relative_l2":float(np.linalg.norm(delta)/np.linalg.norm(original))}
                    selected_images["current"]=current
                if (g=="HARD_OUTWARD" and k=="NEAREST_1X1" and l=="CURRENT_LOBE" and p=="NONE"):
                    selected_images["hard_nearest_T1" if off else "hard_nearest"]=current
                if g=="NO_OUTWARD_GATE" and k=="NEAREST_1X1" and l=="CURRENT_LOBE": selected_images["no_gate_nearest"]=current
                if group=="projected" and p=="PROJECTED_AREA": selected_images["projected_constant"]=current
                if group in ("primary","gate"):
                    band_rows[label]=band_metrics(current,reference,regions)
            report["measurements"].append({"id":label,"gate":g,"width":GATES[g],"kernel":k,"lobe":l,
                "projected":p,"transmission_off":off,"group":group,"primary":group=="primary",
                "cache_digest":frozen_digest,**metrics})
            print(json.dumps({"phase":"measurement","id":label,"thin":metrics["thin_feature_response_ratio"]}),flush=True)
        rows,maps,grazing,positive_rows=contribution_diagnostic(state,regions)
        report["cosine_bins"]=rows
        report["positive_cosine_thin_decomposition"]=positive_rows
        band_rows["current_with_grazing_decomposition"]=band_metrics(selected_images["current"],reference,regions,grazing)
        report["silhouette_bands"]=band_rows
        report["cache_digest_after"]=state_digest(cpu)
    def find(g,k,l="CURRENT_LOBE",p="NONE",off=False):
        return next(r for r in report["measurements"] if (r["gate"],r["kernel"],r["lobe"],r["projected"],r["transmission_off"])==(g,k,l,p,off))
    control=find(PRIMARY[0],KERNELS[2]); hard=find("HARD_OUTWARD",KERNELS[2]); near=find(PRIMARY[0],KERNELS[0]); minimal=find("HARD_OUTWARD",KERNELS[0]); off=find("HARD_OUTWARD",KERNELS[0],off=True)
    gain=lambda r:r["thin_feature_response_ratio"]/control["thin_feature_response_ratio"]
    gate_gain,detector_gain=gain(hard),gain(near)
    constant=find(PRIMARY[0],KERNELS[2],"CONSTANT_RADIANCE")
    lobe_gain=gain(constant)
    grazing_rows=[r for r in rows if r["bin"]<4]
    pre=sum(r["pre_thin_energy"] for r in grazing_rows); post=sum(r["post_thin_energy"] for r in grazing_rows)
    projected=find("HARD_OUTWARD",KERNELS[0],"CONSTANT_RADIANCE","PROJECTED_AREA")
    unprojected=find("HARD_OUTWARD",KERNELS[0],"CONSTANT_RADIANCE")
    improves=projected["whole_image_mse"]<.9*unprojected["whole_image_mse"] and projected["thin_feature_response_ratio"]>1.1*unprojected["thin_feature_response_ratio"]
    recovered=minimal["thin_feature_response_ratio"]>=.75
    transport_primary=off["thin_feature_response_ratio"]>=.75 and minimal["thin_feature_response_ratio"]<.75
    primary_failure=("TRANSPORT_ATTENUATION_FORMULATION" if transport_primary else
        "PRE_MEASUREMENT_SOURCE_OR_GEOMETRY_REPRESENTATION" if off["thin_feature_response_ratio"]<.75 else
        "GRAZING_EMISSION_GATE" if gate_gain>1.5 and detector_gain<1.2 else
        "DETECTOR_KERNEL_BANDWIDTH" if detector_gain>1.5 and gate_gain<1.2 else "COMBINED_MEASUREMENT_LOWPASS")
    primary_rows=[r for r in report["measurements"] if r["primary"]]
    best=min(primary_rows,key=lambda r:(r["artifact_aware"]["gradient_squared_error"],r["whole_image_mse"]))
    report["best_configuration_rule"]="Minimum reference vector-gradient squared error among primary rows; whole-image MSE breaks ties. Larger thin/HF magnitude is not automatically preferred. Reference convention caveat applies."
    report.update(BEST_EMISSION_GATE=best["gate"],BEST_GATE_WIDTH=best["width"],BEST_DETECTOR_KERNEL=best["kernel"],
        BEST_MEASUREMENT_CONFIGURATION=best["id"],PRIMARY_FAILURE=primary_failure,
        PRIMARY_GRAZING_SUPPRESSION_SOURCE="BOTH" if gate_gain>1.5 and lobe_gain>1.5 else "EMISSION_GATE" if gate_gain>1.5 else "SHADING_LOBE" if lobe_gain>1.5 else "NEITHER")
    report["verdicts"]={"SCENE_TRANSPORT_FROZEN_ACROSS_MEASUREMENT_ABLATION":frozen_digest==report["cache_digest_after"],
        "CACHED_MEASUREMENT_REPLAY_EQUIVALENT":report["replay_equivalence"]["relative_l2"]<1e-6,
        "GRAZING_GATE_STRONGLY_SUPPRESSES_THIN_SUPPORT":pre>0 and post/pre<.5,
        "CURRENT_CUBIC_HAS_MEASURABLE_HIGH_FREQUENCY_ATTENUATION":impulses[-1]["mtf_x"]["mean_magnitude"][-1]<.5,
        "DETECTOR_ENERGY_CONSERVING":max(r["partition_unity_max_error"] for r in impulses)<1e-12,
        "MINIMAL_READOUT_RECOVERS_THIN_FEATURE":recovered,"PROJECTED_AREA_WEIGHT_IMPROVES_REFERENCE_MATCH":improves,
        "REFERENCE_AND_SOFT_READOUT_MEASURE_CONSISTENT":"unresolved","TRANSMISSION_IS_PRIMARY_BANDWIDTH_LOSS":transport_primary}
    report["decision_evidence"]={"hard_gate_thin_gain":gate_gain,"nearest_thin_gain":detector_gain,"constant_lobe_thin_gain":lobe_gain,
        "minimal_thin_response":minimal["thin_feature_response_ratio"],"T1_minimal_thin_response":off["thin_feature_response_ratio"],
        "grazing_pre_thin_energy":pre,"grazing_post_thin_energy":post,"recovery_target":.75,
        "grazing_fraction_of_all_pre_gate_thin_energy":pre/max(sum(r["pre_thin_energy"] for r in rows),1e-30),
        "grazing_fraction_of_current_thin_energy":post/max(sum(r["post_thin_energy"] for r in rows),1e-30),
        "threshold_note":"Strong gain >1.5x; recovery >=0.75 historical response ratio. Classification is diagnostic, not a geometry-vs-reference causal identification."}
    report["limitations"]=["Reference and source measures are not established equivalent; no reference was altered.",
        "All primary configurations are four-view FullHD; thin and edge responses are historical image-gradient proxies.",
        "No gate and T=1 include nonphysical diagnostic contributions; projected area is not promoted to production.",
        "Empty silhouette interiors have null statistics, not fabricated zero-error observations.",
        "MTF reports mean magnitude and coherent subpixel mean separately; nearest phase quantization is not reconstruction accuracy."]
    report["figures"]=figures(report,selected_images,psfs,maps)
    report["runtime_seconds"]=time.perf_counter()-started
    write_json(Path("artifacts/v0812_measurement_bandwidth.json"),report)
    from .high_sample import _write_csv
    _write_csv(Path("artifacts/v0812_measurement_bandwidth.csv"),_scalar_csv_rows(report))
    print(json.dumps(report["verdicts"],indent=2),flush=True)
    return report


if __name__=="__main__":
    run_experiment()
