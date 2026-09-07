"""State preservation and BOTH-branch CURRENT response preflight."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .matched_operator import CACHE,build_geometry,source,WendlandField,forward,config_digest
from .matched_jacobian import columns,diagnostic_forward
from .transverse_packet import digest,write_json


def preserve():
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'starting_state.json'
    if not path.exists():
        old=[p for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
             if 'v090' not in p and 'matched_' not in p]
        report=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            tracked_diff=subprocess.check_output(['git','diff'],text=True),
            preexisting_untracked={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old},
            note='Preserves local v0817 uncommitted work; excludes newly added v090 files. No commit/push/reset/checkout.')
        write_json(path,report)
    manifest=CACHE/'historical_hashes.json'
    if not manifest.exists():
        write_json(manifest,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('artifacts','figures')
                            for p in Path(folder).iterdir() if p.is_file() and 'v090' not in p.name})


def propose(state,count=32,active_centers=None,seed=90):
    """Deterministic FPS of global source IDs; proposal changes no source state."""
    rng=np.random.default_rng(seed); size=min(4096,len(state['positions']))
    ids=np.sort(rng.choice(len(state['positions']),size,replace=False)); pool=state['positions'][ids].numpy()
    eligible=np.ones(size,dtype=bool)
    if active_centers is not None:
        for center in active_centers.detach().cpu().numpy(): eligible &= np.linalg.norm(pool-center,axis=1)>.054
    pool=pool[eligible]; ids=ids[eligible]
    if len(pool)<count: raise RuntimeError('NO_DISTINCT_CANDIDATE_CENTERS')
    selected=[]; distances=np.full(len(pool),np.inf); current=0
    for _ in range(count):
        selected.append(current); distances=np.minimum(distances,np.linalg.norm(pool-pool[current],axis=1)); distances[selected]=-1
        current=int(np.argmax(distances))
    centers=torch.as_tensor(pool[selected],device='cuda',dtype=torch.float64)
    # Two declared levels, never adapted to the image residual.
    radii=torch.full((count,),.45,device='cuda',dtype=torch.float64); radii[count//2:]/=math_sqrt_two()
    return centers,radii,ids[selected].tolist()


def math_sqrt_two(): return 2.**.5


def derivative_preflight(base,reference,boundary,cfg,branch):
    small=dict(cfg,resolution=[128,128]); centers,radii,source_ids=propose(reference,16)
    zero=torch.zeros(16,device='cuda',dtype=torch.float64)
    image,state=diagnostic_forward(base,reference,centers,radii,zero,boundary,small)
    score,_=columns(WendlandField(base),reference,state,centers,radii,boundary,small,diagnostic=True)
    norms=np.asarray(score['norm2']); positive=np.where(norms>max(norms.max()*1e-12,1e-30))[0]
    if len(positive)<3: return dict(branch=branch,passed=False,reason='Fewer than three measurable preflight candidates',score=score)
    ordered=positive[np.argsort(norms[positive])[::-1]]; chosen=[int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[-1])]
    _,images=columns(WendlandField(base),reference,state,centers[chosen],radii[chosen],boundary,small,diagnostic=True,return_images=True)
    analytic=np.stack(images); rows=[]; epsilons=(1e-3,1e-4,1e-5,1e-6)
    for col,(candidate,rank) in enumerate(zip(chosen,('high','medium','weak'))):
        j=analytic[:,:,:,col,:].reshape(-1); norm=np.linalg.norm(j); active=np.abs(j)>norm*1e-6
        records=[]
        for eps in epsilons:
            c=zero.clone(); c[candidate]=eps; plus,_=diagnostic_forward(base,reference,centers,radii,c,boundary,small)
            c[candidate]=-eps; minus,_=diagnostic_forward(base,reference,centers,radii,c,boundary,small)
            fd=((plus-minus)/(2*eps)).reshape(-1); fn=np.linalg.norm(fd)
            records.append(dict(epsilon=eps,relative_l2=float(np.linalg.norm(fd-j)/norm),cosine=float(fd@j/(fn*norm)) if fn else None,
                norm_ratio=float(fn/norm),sign_agreement=float(np.mean(np.sign(fd[active])==np.sign(j[active])))))
        passed=all(x['relative_l2']<.02 and x['cosine'] is not None and x['cosine']>.995 and .98<x['norm_ratio']<1.02 and x['sign_agreement']>.99 for x in records[-2:])
        rows.append(dict(candidate=candidate,rank=rank,center=centers[candidate].tolist(),radius=float(radii[candidate]),analytic_norm=float(norm),passed=passed,epsilons=records))
        print(f'{branch} FD {rank}: {records[-1]} passed={passed}',flush=True)
    result=dict(branch=branch,stage='PREFLIGHT_ONLY',resolution=[128,128],samples=len(reference['weights']),
        source_ids=source_ids,rows=rows,passed=all(r['passed'] for r in rows),analytic_method='Compact-support implicit attachment tangent + exact CURRENT barrier chain rule + algebraically identical C3 detector tangent',
        precision='float64 FD diagnostic, matching v0817 validation policy; primary renders remain float32 detector accumulation',
        performance=score)
    write_json(CACHE/f'{branch.lower()}_fd_preflight.json',result)
    return result


def preflight():
    preserve(); torch.set_num_threads(4)
    commands={'compileall':[sys.executable,'-m','compileall','-q','src','scripts','tests','demo.py'],
        'demo_verify':[sys.executable,'demo.py','--verify'],'tests':[sys.executable,'-m','unittest','discover','-s','tests']}
    for name,cmd in commands.items():
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (CACHE/f'preflight_{name}.log').write_text(p.stdout)
        if p.returncode: raise RuntimeError(f'Preflight {name} failed; see log')
    base,gt,sphere,boundary,cfg=build_geometry(); outputs={}
    with torch.no_grad():
        gt_source,gt_meta=source(gt,'GT',1024,cfg)
        empty=base.grid.new_empty((0,3)); radii=base.grid.new_empty(0)
        smoke_cfg=dict(cfg,resolution=[128,128])
        target,_=diagnostic_forward(gt,gt_source,empty,radii,radii,boundary,smoke_cfg)
        np.save(CACHE/'gt_smoke_PREFLIGHT_ONLY.npy',target)
        assert np.isfinite(target).all() and target.sum()>0
        for branch,f in (('SOFT',base),('GLOBE',sphere)):
            try:
                reference,_=source(f,branch,1024,cfg)
                outputs[branch]=derivative_preflight(f,reference,boundary,cfg,branch)
                # One bounded smoke birth uses exactly the same scalar objective.
                centers,rad,ids=propose(reference,16); coefficients=f.lower.new_zeros(16)
                image,state=diagnostic_forward(f,reference,centers,rad,coefficients,boundary,smoke_cfg)
                scores,_=columns(WendlandField(f),reference,state,centers,rad,boundary,smoke_cfg,residual=image-target,diagnostic=True)
                k=int(np.argmax(scores['quadratic'])); step=float(np.clip(-scores['alignment'][k]/(scores['norm2'][k]+1e-7),-.03,.03))
                before=float(np.mean((image-target)**2)); trials=[]
                for alpha in (1.,.5,.25,.125):
                    c=coefficients.clone(); c[k]=alpha*step
                    rendered,_=diagnostic_forward(f,reference,centers,rad,c,boundary,smoke_cfg)
                    trials.append(dict(alpha=alpha,mse=float(np.mean((rendered-target)**2))))
                outputs[branch]['smoke_birth']=dict(stage='PREFLIGHT_ONLY',candidate=k,coefficient_step=step,baseline_mse=before,trials=trials,
                    finite=all(np.isfinite(t['mse']) for t in trials))
            except Exception as exc:
                outputs[branch]=dict(passed=False,execution_error=str(exc),traceback=traceback.format_exc(),stage='PREFLIGHT_ONLY')
                print(outputs[branch]['traceback'],flush=True)
            write_json(Path('artifacts/v090_birth_fd.json'),outputs)
    write_json(CACHE/'preflight.json',dict(branches=outputs,config=cfg,gt_smoke=dict(signature=gt_meta['signature'],finite=True),completed=True))
    print(json.dumps({b:x['passed'] for b,x in outputs.items()}),flush=True)


if __name__=='__main__': preflight()
