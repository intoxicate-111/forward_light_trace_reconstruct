"""Matched CURRENT operator, two independent geometry-birth regimes.

Never commits/pushes. Branch failures are checkpointed and do not cancel the
other branch. All scientific losses use the one cached CURRENT-soft GT target.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr,spearmanr
from .matched_operator import CACHE,build_geometry,source,forward,WendlandField,config_digest,field_digest
from .matched_preflight import preserve,propose
from .matched_jacobian import columns
from .transverse_packet import digest,write_json
from .finite_packet import _image_metrics,_scalar_csv_rows
from .polar_aliasing import artifact_metrics
from .bunny import BunnyGeometryEvaluator,mesh_surface_samples

PROTOCOL=dict(damping=1e-7,coefficient_limit=.03,optimization_steps=1,line_search=[1.,.5,.25,.125],
    maximum_accepted_births=32,maximum_consecutive_no_gain=3,flattening_window=5,flattening_relative_gain=2e-4,
    minimum_births_before_flattening=8,candidate_bank=32,solo_per_stratum=2,
    candidate_rank_spearman_gate=.2,relative_gain_floor=1e-7,geometry_relative_gain_floor=1e-3,
    source_coverage_subset=65536,geometry_samples=16384,initial_preallocated_dof=0,
    support_rule='Two levels: first half radius .45, second half .45/sqrt(2). Independent of residual.',
    optimizer='Damped Gauss-Newton with exact accumulated CURRENT image Gram, coefficient box projection, deterministic four-trial backtracking',
    scoring='Full 16M source, four Full-HD views; exact image-space column norms, no stochastic norm sketch',
    evaluation_subset='Two top, two middle, two low non-null, two seeded random. Largest initial bounded comparison budget; overlap deduplicated and disclosed.',
    smoke_use='FD and runtime only; never headline predicted-vs-actual or birth verdicts.')


def loss(image,target):
    return .5*float(np.square(image.astype(np.float64)-target).mean())


def image_metrics(image,target):
    out={**_image_metrics(list(image),list(target)),**artifact_metrics(image,target)}
    out['loss']=loss(image,target); out['detected_energy']=float(image.sum(dtype=np.float64))
    out['relative_rgb_l2']=float(np.linalg.norm(image.astype(float)-target)/np.linalg.norm(target.astype(float)))
    out['per_view_mse']=[float(np.square(a.astype(float)-b).mean()) for a,b in zip(image,target)]
    overlap=[]
    for a,b in zip(image,target):
        fg_a=a.max(2)>max(float(a.max())*.01,1e-12); fg_b=b.max(2)>max(float(b.max())*.01,1e-12)
        overlap.append(float((fg_a&fg_b).sum()/max((fg_a|fg_b).sum(),1)))
    out['foreground_mask_iou']=overlap
    if 'total_image_energy' in out: out['squared_image_energy']=out.pop('total_image_energy')
    return out


class GeometryMetrics:
    """Reuse project P2S/Chamfer evaluator, but against GT FIELD, not old mesh.

    Marching cubes is isolated here for geometry metrics/exports only. No image
    target, candidate score, optimizer residual, or source cache is read from it.
    """
    def __init__(self,gt):
        self.gt=gt; self.gt_mesh=self.extract(WendlandField(gt),'gt_geometry_evaluation')
        points,_=mesh_surface_samples(self.gt_mesh.vertices,self.gt_mesh.faces,PROTOCOL['geometry_samples'])
        normals=gt.gradient(torch.as_tensor(points,device='cuda',dtype=torch.float64)); normals/=normals.norm(dim=1,keepdim=True)
        prepared=SimpleNamespace(original_vertices=np.asarray(self.gt_mesh.vertices),original_faces=np.asarray(self.gt_mesh.faces))
        self.evaluator=BunnyGeometryEvaluator(prepared,torch.from_numpy(points),normals,PROTOCOL['geometry_samples'])
        self.gt_max_radius=float(np.linalg.norm(self.gt_mesh.vertices,axis=1).max())

    def extract(self,field,label):
        import trimesh
        from skimage.measure import marching_cubes
        n=160; spacing=(field.upper-field.lower)/(n-1)
        values=np.empty(n**3,dtype=np.float64)
        for i in range(0,n**3,65536):
            ids=torch.arange(i,min(i+65536,n**3),device='cuda'); ijk=torch.stack((ids//(n*n),(ids//n)%n,ids%n),1)
            values[i:i+len(ids)]=field.value(field.lower+ijk*spacing).cpu().numpy()
        grid=values.reshape(n,n,n)
        vertices,faces,_,_=marching_cubes(grid,0.,spacing=tuple(spacing.cpu().tolist())); vertices+=field.lower.cpu().numpy()
        mesh=trimesh.Trimesh(vertices,faces,process=False)
        folder=CACHE/'geometry'; folder.mkdir(exist_ok=True)
        mesh.export(folder/(label+'.ply'))
        return mesh

    def evaluate(self,field,state,label):
        started=time.perf_counter(); mesh=self.extract(field,label)
        points,_=mesh_surface_samples(np.asarray(mesh.vertices),np.asarray(mesh.faces),PROTOCOL['geometry_samples'])
        p=torch.as_tensor(points,device='cuda',dtype=torch.float64); normal=field.jet(p)[0]
        normal/=normal.norm(dim=1,keepdim=True).clamp_min(1e-30)
        self.evaluator.reference_normals=normal.cpu().numpy()
        current=points
        self.evaluator.regions={'ear':current[:,1]>.45,'head':(current[:,1]<=.45)&(current[:,1]>-.05)&(current[:,0]<-.35),'leg':current[:,1]<-.70}
        self.evaluator.regions['torso']=~np.logical_or.reduce(list(self.evaluator.regions.values()))
        self.evaluator.regions={k:v for k,v in self.evaluator.regions.items() if v.any()}
        result=self.evaluator(torch.from_numpy(points))
        radii=np.linalg.norm(mesh.vertices,axis=1)
        rng=np.random.default_rng(909); ids=np.sort(rng.choice(len(state['positions']),min(PROTOCOL['source_coverage_subset'],len(state['positions'])),replace=False))
        nearest=cKDTree(state['positions'][ids].numpy()).query(points,workers=1)[0]
        result.update(bounds=np.asarray(mesh.bounds).tolist(),surface_area=float(mesh.area),components=int(mesh.body_count),
            maximum_outward_from_unit_sphere=float(max(0,radii.max()-1)),maximum_inward_from_unit_sphere=float(max(0,1-radii.min())),
            radial_deviation_p95=float(np.quantile(np.abs(radii-1),.95)),gt_extends_outside_unit_sphere=self.gt_max_radius>1,
            gt_max_radius=self.gt_max_radius,source_coverage_distance_p95=float(np.quantile(nearest,.95)),
            source_coverage_distance_max=float(nearest.max()),source_coverage_samples=len(ids),geometry_seconds=time.perf_counter()-started,
            geometry_path=str(CACHE/'geometry'/(label+'.ply')),
            geometry_target='MC approximation of prepared.gt_field, evaluation only; existing P2S/Chamfer definitions. Current normals updated at every evaluation.')
        return result


def score(base,reference,state,active_centers,active_radii,coefficients,candidates,candidate_radii,boundary,cfg,target,image,label):
    path=CACHE/(label+'_scores.json')
    all_centers=torch.cat((active_centers,candidates)); all_radii=torch.cat((active_radii,candidate_radii))
    signature=dict(config=config_digest(cfg),field=field_digest(base),source=digest(reference['positions']),
        current_positions=digest(state['positions']),coefficients=digest(coefficients),active_centers=digest(active_centers),active_radii=digest(active_radii),
        proposals=digest(candidates),proposal_radii=digest(candidate_radii),target=digest(torch.from_numpy(target)),image=digest(torch.from_numpy(image)),
        count=len(reference['weights']),shape=[4,1080,1920,3],dtype='float32 image columns / float64 reductions',seed=90,operator_version=cfg['operator'])
    if path.exists():
        result=json.loads(path.read_text()); assert result['signature']==signature,'stale scoring cache'; return result
    field=WendlandField(base,active_centers,active_radii,coefficients)
    result,_=columns(field,reference,state,all_centers,all_radii,boundary,cfg,residual=image.astype(float)-target,chunk=8192)
    result.update(signature=signature,active_count=len(coefficients),candidate_count=len(candidates),
        centers=all_centers.tolist(),radii=all_radii.tolist(),stage='FORMAL_FULLHD_16M')
    write_json(path,result); return result


def optimize(base,reference,centers,radii,coefficients,boundary,cfg,target,current_image,current_state,gram,alignment,label):
    started=time.perf_counter(); before=loss(current_image,target); k=len(coefficients)
    h=np.asarray(gram,dtype=float)+PROTOCOL['damping']*np.eye(k); a=np.asarray(alignment,dtype=float)
    try: delta=np.linalg.solve(h,-a)
    except np.linalg.LinAlgError:
        return current_image,current_state,coefficients,dict(accepted=False,linear_solver_failures=1,line_search_failures=0,trials=[],seconds=time.perf_counter()-started)
    initial=coefficients.cpu().numpy(); proposed=np.clip(initial+delta,-PROTOCOL['coefficient_limit'],PROTOCOL['coefficient_limit']); delta=proposed-initial
    trials=[]
    for attempt,alpha in enumerate(PROTOCOL['line_search']):
        c=torch.as_tensor(initial+alpha*delta,device='cuda',dtype=torch.float64)
        try:
            rendered,state,timing=forward(base,reference,centers,radii,c,boundary,cfg,f'{label}_trial{attempt}',persist_state=False)
            value=loss(rendered,target)
            trial=dict(alpha=alpha,loss=value,coefficient=c.tolist(),timing=timing,finite=True)
            trials.append(trial)
            if value<before*(1-PROTOCOL['relative_gain_floor']):
                return rendered,state,c,dict(accepted=True,linear_solver_failures=0,line_search_failures=0,trials=trials,
                    realized_gain=before-value,coefficient_limit_hits=int(np.sum(np.abs(c.cpu().numpy())>=.03-1e-12)),seconds=time.perf_counter()-started)
            del rendered,state
        except Exception as exc:
            trials.append(dict(alpha=alpha,error=str(exc),traceback=traceback.format_exc(),finite=False))
    return current_image,current_state,coefficients,dict(accepted=False,linear_solver_failures=0,line_search_failures=1,
        trials=trials,realized_gain=0.,coefficient_limit_hits=int(np.sum(np.abs(proposed)>=.03-1e-12)),seconds=time.perf_counter()-started)


def selection(scores,count_per=2):
    q=np.asarray(scores['quadratic']); n=np.asarray(scores['norm2']); eligible=np.where(n>max(float(n.max())*1e-12,1e-24))[0]
    ordered=eligible[np.argsort(q[eligible])[::-1]]; k=min(count_per,len(ordered)//3)
    if not k: return [],{}
    middle=max(k,len(ordered)//2-k//2)
    groups=dict(top=ordered[:k].tolist(),middle=ordered[middle:middle+k].tolist(),low=ordered[-k:].tolist())
    rng=np.random.default_rng(902); groups['random']=rng.choice(eligible,min(count_per,len(eligible)),replace=False).tolist()
    return sorted(set(i for values in groups.values() for i in values)),groups


def candidate_validation(branch,base,reference,state,image,target,boundary,cfg,geometry,baseline_geometry,bank):
    centers,radii,source_ids=bank; empty=centers.new_empty((0,3)); zero=radii.new_empty(0)
    scored=score(base,reference,state,empty,zero,zero,centers,radii,boundary,cfg,target,image,branch.lower()+'_initial')
    ids,groups=selection(scored,PROTOCOL['solo_per_stratum']); rows=[]
    baseline_loss=loss(image,target)
    for i in ids:
        path=CACHE/f'{branch.lower()}_solo_{i}.json'
        signature=dict(scoring_signature=scored['signature'],candidate=i,protocol=PROTOCOL)
        if path.exists():
            cached=json.loads(path.read_text()); assert cached['cache_signature']==signature,'stale solo cache'
            rows.append(cached); continue
        ci=centers[i:i+1]; ri=radii[i:i+1]; coefficient=ri.new_zeros(1)
        rendered,new_state,c,optimization=optimize(base,reference,ci,ri,coefficient,boundary,cfg,target,image,state,
            [[scored['norm2'][i]]],[scored['alignment'][i]],f'{branch.lower()}_solo_{i}')
        geom=geometry.evaluate(WendlandField(base,ci,ri,c),new_state,f'{branch.lower()}_solo_{i}')
        record=dict(cache_signature=signature,candidate=i,source_proposal_id=source_ids[i],center=ci[0].tolist(),radius=float(ri[0]),
            alignment=scored['alignment'][i],raw_alignment=scored['raw'][i],norm2=scored['norm2'][i],jacobian_norm=scored['jacobian_norm'][i],
            predicted_gain=scored['quadratic'][i],actual_gain=baseline_loss-loss(rendered,target),
            geometry_improvement=baseline_geometry['symmetric_chamfer']-geom['symmetric_chamfer'],geometry=geom,
            coefficient=float(c[0]),affected_views=scored['affected_views'][i],affected_pixels=scored['affected_pixels'][i],
            optimization=optimization,baseline_identical=True,coefficient_initialized_at_zero=True,birth_source_mass_change=0.,
            image_path=str(CACHE/f'{branch.lower()}_solo_{i}_trial{len(optimization["trials"])-1}.npy') if optimization['accepted'] else str(CACHE/(branch.lower()+'_initial.npy')))
        write_json(path,record); rows.append(record)
        print(f'{branch} candidate {i}: predicted={record["predicted_gain"]:.6g}, actual={record["actual_gain"]:.6g}',flush=True)
    lookup={r['candidate']:r for r in rows}; pred=np.asarray([r['predicted_gain'] for r in rows]); actual=np.asarray([r['actual_gain'] for r in rows])
    informative=len(rows)>=3 and np.std(pred)>0 and np.std(actual)>0
    pearson=float(pearsonr(pred,actual).statistic) if informative else None
    spearman=float(spearmanr(pred,actual).statistic) if informative else None
    top=float(np.mean([lookup[i]['actual_gain'] for i in groups.get('top',[])])) if rows else 0.
    random=float(np.mean([lookup[i]['actual_gain'] for i in groups.get('random',[])])) if rows else 0.
    allowed=informative and spearman>PROTOCOL['candidate_rank_spearman_gate'] and float(actual.max())>baseline_loss*PROTOCOL['relative_gain_floor']
    n=np.asarray(scored['norm2']); responsive=n>max(float(n.max())*1e-12,1e-24)
    return dict(scores=scored,rows=rows,groups=groups,pearson=pearson,spearman=spearman,
        top_mean_gain=top,random_mean_gain=random,top_k_enrichment=top/random if random>0 else None,
        top_candidates_outperform_random=top>random,rank_directionally_positive=bool(allowed),
        responsive_fraction=float(responsive.mean()),null_response_candidates=np.where(~responsive)[0].tolist(),
        false_positive_candidates=[r['candidate'] for r in rows if r['predicted_gain']>0 and r['actual_gain']<=0],
        statement='Primary Full-HD actual gains, same initial zero-coefficient state and fixed source for each candidate; reduced-source smoke gains excluded.')


def save_branch(branch,report):
    from .high_sample import _write_csv
    write_json(Path(f'artifacts/v090_{branch.lower()}_birth.json'),report)
    _write_csv(Path(f'artifacts/v090_{branch.lower()}_birth.csv'),_scalar_csv_rows(report))


def run():
    preserve(); torch.set_num_threads(4); started=time.perf_counter()
    if (CACHE/'run_complete.json').exists():
        from .matched_report import finalize
        finalize(); return
    setup=json.loads((CACHE/'setup.json').read_text()); pref=json.loads((CACHE/'preflight.json').read_text())
    base,gt,sphere,boundary,cfg=build_geometry(); assert setup['config']==cfg
    target=np.load(CACHE/'gt_current_soft_16m.npy'); assert digest(torch.from_numpy(target))==setup['target']['digest']
    geometry=GeometryMetrics(gt); reports={}; live={}
    write_json(CACHE/'protocol.json',PROTOCOL)
    code_paths=list(Path('src/zlt').glob('matched_*.py'))+[Path('src/zlt/finite_packet.py'),Path('src/zlt/density_matrix.py'),Path('src/zlt/transverse_packet.py')]
    code_state=dict(sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},note='Exact local files used for formal execution, including uncommitted CURRENT renderer.')
    history_path=CACHE/'formal_code_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    if (CACHE/'formal_code_state.json').exists() and not history: history.append(json.loads((CACHE/'formal_code_state.json').read_text()))
    if not history or history[-1]!=code_state: history.append(code_state)
    write_json(history_path,history); write_json(CACHE/'formal_code_state.json',code_state)
    with torch.no_grad():
        # Both validations precede either long sequential run.
        for branch,field in (('SOFT',base),('GLOBE',sphere)):
            previous_path=Path(f'artifacts/v090_{branch.lower()}_birth.json')
            previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
            if previous:
                assert previous['protocol']==PROTOCOL and previous['target_digest']==setup['target']['digest'],'stale branch checkpoint'
                if previous['status'] not in ('VALIDATING_CANDIDATES','SEQUENTIAL_READY','SEQUENTIAL_RUNNING'):
                    reports[branch]=previous; continue
            reference,source_meta=source(field,branch,cfg['count'],cfg)
            state=torch.load(CACHE/(branch.lower()+'_initial_state.pt'),weights_only=True,mmap=True)
            image=np.load(CACHE/(branch.lower()+'_initial.npy'))
            bank=propose(state,PROTOCOL['candidate_bank'])
            baseline_geometry=geometry.evaluate(WendlandField(field),state,branch.lower()+'_initial')
            report=dict(branch=branch,protocol=PROTOCOL,operator=cfg,target_digest=setup['target']['digest'],
                initial_preallocated_dof=0,source=source_meta,derivative_validation=pref['branches'][branch],
                baseline=dict(observation=image_metrics(image,target),geometry=baseline_geometry),births=[],status='VALIDATING_CANDIDATES')
            if previous: report=previous
            reports[branch]=report; save_branch(branch,report)
            try:
                if not pref['branches'][branch]['passed']:
                    report['status']='STOP_DERIVATIVE_GATE'; save_branch(branch,report); continue
                validation=report.get('candidate_validation') or candidate_validation(branch,field,reference,state,image,target,boundary,cfg,geometry,baseline_geometry,bank)
                report['candidate_validation']=validation
                if not validation['rank_directionally_positive']:
                    report['status']='STOP_UNINFORMATIVE_RANKING'; save_branch(branch,report); continue
                report['status']='SEQUENTIAL_READY'; save_branch(branch,report)
                live[branch]=dict(base=field,reference=reference,state=state,image=image,bank=bank,
                    centers=bank[0].new_empty((0,3)),radii=bank[1].new_empty(0),coefficients=bank[1].new_empty(0),
                    remaining=list(range(len(bank[1]))),no_gain=0,accepted=0)
                if report['births']:
                    last=report['births'][-1]; s=live[branch]
                    s['centers']=torch.tensor(last['centers'],device='cuda',dtype=torch.float64)
                    s['radii']=torch.tensor(last['radii'],device='cuda',dtype=torch.float64)
                    s['coefficients']=torch.tensor(last['coefficients'],device='cuda',dtype=torch.float64)
                    s['image'],s['state'],_=forward(field,reference,s['centers'],s['radii'],s['coefficients'],boundary,cfg,f'{branch.lower()}_resume{last["round"]}')
                    s['accepted']=sum(b['optimization']['accepted'] for b in report['births'])
                    for b in reversed(report['births']):
                        if b['optimization']['accepted']: break
                        s['no_gain']+=1
                    if branch=='SOFT': s['remaining']=[i for i in s['remaining'] if bank[2][i] not in {b['proposal_source_id'] for b in report['births']}]
            except Exception as exc:
                report['status']='BRANCH_EXECUTION_FAILURE'; report['error']=str(exc); report['traceback']=traceback.format_exc(); save_branch(branch,report)
        # Interleave branches so neither consumes the whole run first.
        for pass_id in range(PROTOCOL['maximum_accepted_births']+PROTOCOL['maximum_consecutive_no_gain']):
            if not live: break
            for branch in list(live):
                s=live[branch]; report=reports[branch]; round_start=time.perf_counter()
                outer=len(report['births'])
                try:
                    if branch=='SOFT':
                        ids=s['remaining']; proposal=s['bank'][0][ids]; radii=s['bank'][1][ids]; source_ids=[s['bank'][2][i] for i in ids]
                    else:
                        proposal,radii,source_ids=propose(s['state'],PROTOCOL['candidate_bank'],s['centers'],seed=90+outer)
                        ids=list(range(len(radii)))
                    if not len(radii): report['status']='STOP_NO_INACTIVE_CANDIDATES'; del live[branch]; save_branch(branch,report); continue
                    scored=score(s['base'],s['reference'],s['state'],s['centers'],s['radii'],s['coefficients'],proposal,radii,boundary,cfg,target,s['image'],f'{branch.lower()}_round{outer+1}')
                    active=len(s['coefficients']); utility=np.asarray(scored['quadratic'][active:]); norm=np.asarray(scored['norm2'][active:])
                    valid=norm>max(float(norm.max())*1e-12,1e-24)
                    if not valid.any(): report['status']='STOP_NO_VALID_RESPONSE'; del live[branch]; save_branch(branch,report); continue
                    local=int(np.argmax(np.where(valid,utility,-np.inf))); selected=active+local
                    c0=torch.cat((s['coefficients'],s['coefficients'].new_zeros(1)))
                    new_centers=torch.cat((s['centers'],proposal[local:local+1])); new_radii=torch.cat((s['radii'],radii[local:local+1]))
                    selected_ids=list(range(active))+[selected]; gram=np.asarray(scored['gram'])[np.ix_(selected_ids,selected_ids)]; alignment=np.asarray(scored['alignment'])[selected_ids]
                    before=loss(s['image'],target)
                    image,state,c,optimization=optimize(s['base'],s['reference'],new_centers,new_radii,c0,boundary,cfg,target,s['image'],s['state'],gram,alignment,f'{branch.lower()}_birth{outer+1}')
                    s.update(centers=new_centers,radii=new_radii,coefficients=c,image=image,state=state)
                    if branch=='SOFT': s['remaining'].remove(ids[local])
                    accepted=optimization['accepted']; s['accepted']+=int(accepted); s['no_gain']=0 if accepted else s['no_gain']+1
                    field=WendlandField(s['base'],new_centers,new_radii,c)
                    geom=geometry.evaluate(field,state,f'{branch.lower()}_birth{outer+1}')
                    point=proposal[local:local+1]
                    prefield=WendlandField(s['base'],new_centers[:-1],new_radii[:-1],c0[:-1])
                    record=dict(round=outer+1,deterministic_id=f'{branch}:{outer+1}:{ids[local]}',center=point[0].tolist(),radius=float(radii[local]),
                        proposal_source_id=source_ids[local],zero_set_residual_at_birth=float(prefield.value(point).abs()[0]),
                        selected_rank=1,candidate_bank_size=len(radii),responsive_fraction=float(valid.mean()),
                        predicted_gain=float(utility[local]),alignment=scored['alignment'][selected],jacobian_norm=scored['jacobian_norm'][selected],
                        affected_views=scored['affected_views'][selected],affected_pixels=scored['affected_pixels'][selected],
                        coefficient=float(c[-1]),active_dof_count=len(c),born_dof_count=len(c),accepted_births=s['accepted'],
                        initialized_at_zero=True,birth_image_jump=0.,birth_source_count_change=0,birth_source_mass_change=0.,
                        source_count=len(state['weights']),source_weight_digest=digest(state['weights']),
                        coefficients=c.tolist(),centers=new_centers.tolist(),radii=new_radii.tolist(),
                        realized_gain=before-loss(image,target),optimization=optimization,observation=image_metrics(image,target),geometry=geom,
                        scoring_seconds=scored['seconds'],total_seconds=time.perf_counter()-round_start,
                        peak_cuda_allocated_mib=max([scored['peak_cuda_allocated_mib']]+[x['timing']['peak_cuda_allocated_mib'] for x in optimization['trials'] if 'timing' in x]),
                        peak_cuda_reserved_mib=max([scored['peak_cuda_reserved_mib']]+[x['timing']['peak_cuda_reserved_mib'] for x in optimization['trials'] if 'timing' in x]),
                        cpu_rss_peak_mib=max([scored['cpu_rss_peak_mib']]+[x['timing']['cpu_rss_peak_mib'] for x in optimization['trials'] if 'timing' in x]),
                        image_path=str(CACHE/f'{branch.lower()}_birth{outer+1}_trial{len(optimization["trials"])-1}.npy') if accepted else None)
                    report['births'].append(record)
                    checkpoint=outer+1 in (1,2,4,8,16,32,64) or not accepted
                    if checkpoint:
                        label=f'{branch.lower()}_checkpoint{outer+1}'; torch.save(state,CACHE/(label+'_state.pt'))
                        np.save(CACHE/(label+'.npy'),image); write_json(CACHE/(label+'.json'),dict(record=record,config_digest=config_digest(cfg),source_field_digest=field_digest(s['base']),sample_count=cfg['count'],seed=cfg['seed'],operator_version=cfg['operator'],shape=[4,1080,1920,3],dtype='float32 image / float64 source'))
                        record['checkpoint_image']=str(CACHE/(label+'.npy'))
                    report['status']='SEQUENTIAL_RUNNING'
                    if s['no_gain']>=PROTOCOL['maximum_consecutive_no_gain']: report['status']='STOP_REPEATED_NO_GAIN'
                    if s['accepted']>=PROTOCOL['maximum_accepted_births']: report['status']='BIRTH_BUDGET_COMPLETE'
                    window=report['births'][-PROTOCOL['flattening_window']:]
                    if len(report['births'])>=PROTOCOL['minimum_births_before_flattening'] and len(window)==PROTOCOL['flattening_window']:
                        relative=sum(x['realized_gain'] for x in window)/max(window[0]['observation']['loss'],1e-30)
                        if relative<PROTOCOL['flattening_relative_gain']: report['status']='STOP_DECLARED_FLATTENING'
                    if report['status']!='SEQUENTIAL_RUNNING': del live[branch]
                    save_branch(branch,report); print(f'{branch} birth {outer+1}: {record["realized_gain"]:.6g}, {report["status"]}',flush=True)
                except Exception as exc:
                    report['status']='BRANCH_EXECUTION_FAILURE'; report['error']=str(exc); report['traceback']=traceback.format_exc(); save_branch(branch,report); del live[branch]
        for branch,report in reports.items():
            if report['status']=='SEQUENTIAL_RUNNING': report['status']='OUTER_ROUND_BUDGET_COMPLETE'
            save_branch(branch,report)
    write_json(CACHE/'run_complete.json',dict(branches={b:r['status'] for b,r in reports.items()},total_seconds=time.perf_counter()-started))
    from .matched_report import finalize
    finalize()


if __name__=='__main__': run()
