"""Independent dense hard-visibility, matched surface-measure MSE reference.

Historical reference files are never overwritten. This is a visibility oracle
on the fixed grid-isosurface triangulation, not a physical radiance photograph.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .density_matrix import sparse_scene, capture_splat
from .measurement_bandwidth import gate, lobe
from .meshfree_surface import meshfree_base_color
from .finite_packet import _image_metrics, _scalar_csv_rows
from .polar_aliasing import artifact_metrics
from .transverse_packet import digest, write_json

CACHE=Path("runs/v0815_dense_reference")
LEVELS=(2**23,2**24,2**25,2**26,2**27,2**28)


def load_mse_reference():
    """Load only an explicitly validated reference; never fall back to sparse GT."""
    config=json.loads(Path("artifacts/mse_reference.json").read_text())
    if not config.get("validated"): raise RuntimeError("MSE reference is not validated")
    array=np.load(config["path"])
    if digest(torch.from_numpy(array))!=config["digest"]: raise RuntimeError("MSE reference digest mismatch")
    return array


def relative_change(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def compare_reference(previous,current):
    rgb=[]; gradients=[]
    for a,b in zip(previous,current):
        rgb.append(relative_change(a,b))
        gradients.append(relative_change(np.stack(np.gradient(a.mean(2))),np.stack(np.gradient(b.mean(2)))))
    return {"rgb_relative_l2_per_view":rgb,"spatial_gradient_relative_l2_per_view":gradients,
        "passed":max(rgb)<.02 and max(gradients)<.10}


def build_reference(scene,state):
    import open3d as o3d
    from .fixed_measure import CACHE as SOURCE_CACHE
    identity=torch.load(SOURCE_CACHE/"global_identity.pt",weights_only=True)
    vertices=identity["vertices"].numpy(); faces=identity["faces"].numpy()
    triangles=vertices[faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    cdf=np.cumsum(area); base=scene["context"].base
    ray_scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    ray_scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),o3d.core.Tensor(faces.astype(np.uint32)))
    generator=torch.quasirandom.SobolEngine(3,scramble=True,seed=211)
    accum=torch.zeros((4,1080*1920,3),dtype=torch.float64,device="cuda")
    emitted=0.; visible=np.zeros(4,dtype=np.int64); previous=None; rows=[]
    start=time.perf_counter(); done=0; weight_sum=float(state["weights"].sum())
    prior_seconds=0.
    prior_path=Path("artifacts/v0815_dense_reference.json")
    if prior_path.exists():
        prior=json.loads(prior_path.read_text())
        assert prior["seed"]==211 and prior["resolution"]==[1080,1920]
        previous=np.load(prior["reference_path"])
        assert digest(torch.from_numpy(previous))==prior["reference_digest"]
        rows=prior["levels"]; done=rows[-1]["samples"]
        if prior["REFERENCE_CONVERGENCE_PASSED"]: return previous,rows
        # Resume a validated prefix. The saved float32 image introduces only
        # image-storage roundoff, far below the declared quadrature tolerance.
        accum.copy_(torch.from_numpy(previous).cuda().reshape_as(accum)/(weight_sum/done*(1.5*1080*1920)))
        generator.fast_forward(done)
        emitted=rows[-1]["emitted_rgb"]*done/weight_sum
        visible=np.rint(np.asarray(rows[-1]["visible_fraction"])*done).astype(np.int64)
        prior_seconds=rows[-1]["elapsed_seconds"]
        print(f"Resuming independent reference at {done:,} samples",flush=True)
    for count in LEVELS:
        if count<=done: continue
        while done<count:
            size=min(16384,count-done)
            s=generator.draw(size,dtype=torch.float64).numpy()
            ids=np.searchsorted(cdf,s[:,0]*cdf[-1]); u=np.sqrt(s[:,1])
            bary=np.stack((1-u,u*(1-s[:,2]),u*s[:,2]),1)
            points=np.einsum("ni,nij->nj",bary,triangles[ids])
            p=torch.as_tensor(points,device="cuda",dtype=torch.float64)
            normal=base.gradient(p); normal/=normal.norm(dim=1,keepdim=True).clamp_min(1e-30)
            color=meshfree_base_color(p,base.lower,base.upper)
            emitted+=float(color.sum())
            for view in range(4):
                direction=state["directions"][view].numpy()
                # Camera-to-surface first intersection avoids launch-offset and
                # self-hit epsilon changing the transport definition.
                rays=np.concatenate((points+4*direction,np.broadcast_to(-direction,points.shape)),1).astype(np.float32)
                hits=ray_scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)
                hit_ids=hits["primitive_ids"].numpy()
                hit=hit_ids==ids
                visible[view]+=hit.sum()
                cosine=normal@state["directions"][view].cuda()
                energy=color*gate(cosine,"CURRENT_SOFT_W005")[:,None]*lobe(cosine,"CURRENT_LOBE")[:,None]*torch.as_tensor(hit,device="cuda")[:,None]
                rel=p-state["center"].cuda()
                row=(.5-rel@state["up"][view].cuda()/2.8)*1080-.5
                col=(rel@state["right"][view].cuda()/2.8+.5)*1920-.5
                capture_splat(accum[view],row,col,energy,4)
            done+=size
            if done%(2**20)==0: print(f"dense hard reference: {done:,} samples",flush=True)
        images=(accum*(weight_sum/count)*(1.5*1080*1920)).reshape(4,1080,1920,3).cpu().float().numpy()
        np.save(CACHE/f"reference_{count}.npy",images)
        row={"samples":count,"emitted_rgb":emitted*weight_sum/count,"visible_fraction":(visible/count).tolist(),"elapsed_seconds":prior_seconds+time.perf_counter()-start}
        if previous is not None: row["convergence"]=compare_reference(previous,images)
        rows.append(row); print(json.dumps(row),flush=True)
        previous=images
        if row.get("convergence",{}).get("passed",False): break
    return images,rows


def run_experiment():
    from .high_sample import _write_csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(8); CACHE.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    state=torch.load("runs/v0814_fixed_measure/global_energy_matched_state.pt",weights_only=True)
    baseline=torch.load("runs/v0812_aliasing/PER_CHART_PLUS_RING_state.pt",weights_only=True)
    with torch.no_grad():
        scene,_,_,_=sparse_scene(baseline)
        reference,levels=build_reference(scene,state)
    passed=levels[-1].get("convergence",{}).get("passed",False)
    path=CACHE/("reference.npy" if passed else "reference_candidate.npy")
    np.save(path,reference)
    report={"version":"0.8.15","reference_path":str(path),"reference_digest":digest(torch.from_numpy(reference)),
        "REFERENCE_CONVERGENCE_PASSED":passed,"seed":211,"resolution":[1080,1920],"views":4,
        "definition":"Uniform fixed grid-isosurface triangle-area measure; independent BVH first-hit hard visibility, unchanged position color, gate .05, ambient .35, source budget, sensor gain 1.5, window 2.8 and C3 pixel-integrated cubic readout. NOT point-sampled radiance. No soft transmission or chart pattern enters the target.",
        "geometry_caveat":"Piecewise-planar marching-cubes approximation of the same grid; not exact intersection of the trilinear zero-set. Normals use unchanged grid gradient. Same triangles used for source integration and independent BVH visibility.",
        "convergence_thresholds":{"max_view_rgb_relative_l2":.02,"max_view_spatial_gradient_relative_l2":.10,"scope":"Nested prefixes of seed 211, independent of candidate seed 101; empirical numerical tolerance, not an error bound or seed-independent proof."},
        "resume_precision":"Accumulation float64; resumed prefixes reconstructed from stored float32 images (image-storage roundoff, not a change in samples or weights).",
        "levels":levels,"comparisons":{},"historical_reference_unchanged":"runs/v0812_measurement/reference.npy",
        "interpretation":"Image spatial-gradient metrics only, not external dI/dlambda. A remaining discrepancy may combine source measure and hard-vs-soft visibility; this reference alone does not identify the primary cause."}
    images={"current":np.load("runs/v0813_density_sparse/E3C3_rgb.npy"),"global":np.load("runs/v0814_fixed_measure/global_rgb.npy")}
    for name,image in images.items():
        m=_image_metrics(list(image),list(reference)); m["historical_squared_image_energy"]=m.pop("total_image_energy")
        m.update(artifact_metrics(image,reference)); report["comparisons"][name]=m
    fig,axes=plt.subplots(3,4,figsize=(14,9))
    exposure=1/max(float(np.quantile(reference,.995)),1.)
    for row,(name,frames) in enumerate([*images.items(),("dense hard reference",reference)]):
        for v in range(4):
            axes[row,v].imshow(np.clip(frames[v]*exposure,0,1)); axes[row,v].set_title(f"{name}, view {v}"); axes[row,v].axis("off")
    fig.suptitle(f"Shared display exposure {exposure:.4g}; metrics use unclipped linear RGB")
    fig.tight_layout(); figure="figures/v0815_dense_reference.png"; fig.savefig(figure,dpi=150); plt.close(fig)
    report["figures"]=[figure]; report["runtime_seconds"]=time.perf_counter()-start
    report["peak_cuda_allocated_mib"]=torch.cuda.max_memory_allocated()/2**20
    write_json(Path("artifacts/v0815_dense_reference.json"),report)
    _write_csv(Path("artifacts/v0815_dense_reference.csv"),_scalar_csv_rows(report))
    print(json.dumps({"converged":passed,"comparisons":report["comparisons"]},indent=2),flush=True)
    return report


if __name__=="__main__": run_experiment()
