"""Evidence-only reporting for the matched CURRENT geometry-birth benchmark."""
import json
import time
from pathlib import Path
import numpy as np
from .matched_operator import CACHE
from .transverse_packet import write_json
from .high_sample import _write_csv
from .finite_packet import _scalar_csv_rows


LIMITATIONS=[
    'All primary images and candidate columns use 16,777,216 emitters, four 1920x1080 views. FD/smoke and timing subsets are PREFLIGHT_ONLY.',
    'GT uniform implicit-area, SOFT historical polygon-reference area pushed onto the base, and GLOBE analytic sphere area share normalized fixed-reference-measure semantics, but are not identical reference measures. No independent-layout control was run.',
    'v0816 established an MSE plateau, not strict quadrature convergence. Optional 32M GT convergence has not been run.',
    'Exact accumulated image columns include cross-emitter terms. The derivative is local to fixed quadrature/topology; finite steps can cross cells and invalidate quadratic gain predictions.',
    'Small deterministic two-level candidate bank, one damped Gauss-Newton step per birth, coefficient bound .03, and sparse single-candidate validation limit generality.',
    'Geometry metrics use a 160^3 marching-cubes evaluation of the GT scalar field, never the observation target. Source-coverage distances use a fixed 65,536-ID diagnostic subset.',
    'Coefficient saturation alone is not proof that relaxing the bound would improve geometry. Small nonzero sphere deformation is not recovery of Bunny-scale structure.',
]


def load(path): return json.loads(Path(path).read_text())


def final_image(branch,report):
    for row in reversed(report.get('births',[])):
        path=row.get('checkpoint_image') or row.get('image_path')
        if path and Path(path).exists(): return np.load(path,mmap_mode='r')
    return np.load(CACHE/(branch.lower()+'_initial.npy'),mmap_mode='r')


def verdicts(setup,reports):
    v=dict(MATCHED_CURRENT_SOFT_GT_TARGET_VALID=setup['target']['valid'],
        TARGET_AND_RECONSTRUCTION_USE_SAME_FORWARD_OPERATOR=True,
        BIRTH_SIGNAL_IS_NOT_DOMINATED_BY_SOURCE_LAYOUT='UNRESOLVED')
    for branch,r in reports.items():
        births=r.get('births',[]); cv=r.get('candidate_validation',{})
        if len(cv.get('rows',[]))>=3:
            from scipy.stats import pearsonr,spearmanr
            pred=np.asarray([x['predicted_gain'] for x in cv['rows']]); actual=np.asarray([x['actual_gain'] for x in cv['rows']])
            if pred.std()>0 and actual.std()>0:
                cv['pearson_pvalue']=float(pearsonr(pred,actual).pvalue); cv['spearman_pvalue']=float(spearmanr(pred,actual).pvalue)
            cv['best_predicted_actual_gain']=float(actual[pred.argmax()])
            cv['validation_sample_count']=len(actual)
            cv['inference_caveat']='Small deterministic stratified sample, overlapping random/top strata; directional correlation is not population-level proof.'
        initial=r['baseline']; end=births[-1] if births else initial
        image_gain=(initial['observation']['loss']-end['observation']['loss'])/initial['observation']['loss']
        geometry_gain=(initial['geometry']['symmetric_chamfer']-end['geometry']['symmetric_chamfer'])/initial['geometry']['symmetric_chamfer']
        evidence=bool(births)
        values=dict(CURRENT_FORWARD_CANDIDATE_DERIVATIVE_PASSES_FD=r['derivative_validation']['passed'],
            NONEXISTENT_CANDIDATE_RESPONSE_IS_MEASURABLE=cv.get('responsive_fraction',0)>0,
            PREDICTED_GAIN_CORRELATES_WITH_ACTUAL_GAIN=cv.get('rank_directionally_positive','UNRESOLVED'),
            TOP_CANDIDATES_OUTPERFORM_RANDOM=cv.get('top_candidates_outperform_random','UNRESOLVED'),
            SEQUENTIAL_BIRTH_REDUCES_OBSERVATION_ERROR=bool(image_gain>1e-7) if evidence else 'UNRESOLVED',
            SEQUENTIAL_BIRTH_REDUCES_GEOMETRY_ERROR=bool(geometry_gain>1e-3) if evidence else 'UNRESOLVED',
            DYNAMIC_BIRTH_IS_SUCCESSFUL=bool(image_gain>1e-7 and geometry_gain>1e-3) if evidence else 'UNRESOLVED')
        v.update({branch+'_'+k:value for k,value in values.items()})
        r['summary']=dict(accepted_births=sum(x['optimization']['accepted'] for x in births),born_dofs=len(births),
            relative_observation_improvement=image_gain,relative_chamfer_improvement=geometry_gain,
            initial_mse=2*initial['observation']['loss'],final_mse=2*end['observation']['loss'],
            initial_chamfer=initial['geometry']['symmetric_chamfer'],final_chamfer=end['geometry']['symmetric_chamfer'])
        if branch=='GLOBE':
            # Explicit conservative threshold for leaving a thin shell, not merely nonzero lambda.
            displacement=max(end['geometry']['maximum_outward_from_unit_sphere'],end['geometry']['maximum_inward_from_unit_sphere'])
            v['GLOBE_BIRTH_ESCAPES_INITIAL_SPHERE']=bool(displacement>.1) if evidence else 'UNRESOLVED'
            v['GLOBE_COEFFICIENT_LIMIT_IS_ACTIVE_BOTTLENECK']='UNRESOLVED'
            v['GLOBE_FIXED_REFERENCE_SOURCE_REMAINS_USABLE']=bool(end['geometry']['source_coverage_distance_p95']<max(.1,2*initial['geometry']['source_coverage_distance_p95'])) if evidence else 'UNRESOLVED'
            r['sphere_diagnostics']=dict(shell_escape_threshold=.1,maximum_absolute_radial_deviation=displacement,
                coefficient_limit_hit_rounds=sum(x['optimization'].get('coefficient_limit_hits',0)>0 for x in births),
                note='Saturation is reported, but causal bottleneck verdict requires a separate relaxed-bound control; not run.')
    success=any(v.get(b+'_DYNAMIC_BIRTH_IS_SUCCESSFUL') is True for b in reports)
    v['CURRENT_SOFT_OPERATOR_SUPPORTS_GEOMETRY_BIRTH']=True if success else 'UNRESOLVED'
    v['GLOBAL_FIXED_MEASURE_IS_COMPATIBLE_WITH_GEOMETRY_BIRTH']=True if success else 'UNRESOLVED'
    v['OBSERVATION_DRIVEN_FUNCTION_SPACE_GROWTH_IS_SUPPORTED']=True if success else 'UNRESOLVED'
    v['STRONGER_GLOBE_TO_BUNNY_GROWTH_IS_SUPPORTED']=bool(v.get('GLOBE_DYNAMIC_BIRTH_IS_SUCCESSFUL') is True and v.get('GLOBE_BIRTH_ESCAPES_INITIAL_SPHERE') is True)
    return v


def figures(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=np.load(CACHE/'gt_current_soft_16m.npy',mmap_mode='r')
    exposure=float(np.quantile(target,.999)); exposure=max(exposure,1e-12)
    folder=Path('figures'); folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/('v090_'+name+'.png'),dpi=140,bbox_inches='tight'); plt.close(fig)
    def montage(rows,name):
        fig,axes=plt.subplots(len(rows),4,figsize=(16,2.5*len(rows)),squeeze=False)
        for axrow,(label,images) in zip(axes,rows):
            for view,ax in enumerate(axrow):
                ax.imshow(np.clip(images[view]/exposure,0,1)); ax.axis('off'); ax.set_title(f'{label} / view {view}')
        fig.suptitle(f'CURRENT / shared exposure {exposure:.4g}; Full-HD originals in runs/v090_matched_birth')
        save(fig,name)
    montage([('GT',target)],'gt_target')
    finalrows=[]
    for branch,r in reports.items():
        baseline=np.load(CACHE/(branch.lower()+'_initial.npy'),mmap_mode='r')
        montage([('GT',target),(branch+' initial',baseline)],branch.lower()+'_baseline')
        fig,axes=plt.subplots(1,4,figsize=(16,3))
        for view,ax in enumerate(axes):
            im=ax.imshow(np.mean(baseline[view]-target[view],axis=2),cmap='coolwarm',vmin=-exposure,vmax=exposure); ax.axis('off')
        fig.colorbar(im,ax=axes.tolist(),shrink=.6); fig.suptitle(branch+' signed RGB-mean residual (shared scale)'); save(fig,branch.lower()+'_residuals')
        rows=[('initial',baseline)]
        for b in r.get('births',[]):
            if b['round'] in (1,4,8,16,32,64) and b.get('checkpoint_image'): rows.append((str(b['round']),np.load(b['checkpoint_image'],mmap_mode='r')))
        ongoing=r['status'] in ('VALIDATING_CANDIDATES','SEQUENTIAL_READY','SEQUENTIAL_RUNNING')
        end_label='latest (IN PROGRESS)' if ongoing else 'final'
        final=final_image(branch,r); rows.append((end_label+': '+r['status'],final))
        montage(rows,branch.lower()+'_birth_checkpoints'); finalrows.append((branch+' '+end_label,final))
        cv=r.get('candidate_validation',{}); evaluated=cv.get('rows',[])
        fig,ax=plt.subplots(figsize=(6,5))
        if evaluated:
            x=[e['predicted_gain'] for e in evaluated]; y=[e['actual_gain'] for e in evaluated]; ax.scatter(x,y)
            for e in evaluated: ax.annotate(str(e['candidate']),(e['predicted_gain'],e['actual_gain']))
            ax.set_xscale('symlog',linthresh=1e-9); ax.set_yscale('symlog',linthresh=1e-9)
        ax.set(xlabel='Predicted quadratic gain',ylabel='Actual single-birth gain',title=f'{branch}: Spearman={cv.get("spearman")}'); save(fig,branch.lower()+'_predicted_vs_actual')
    montage([('GT',target)]+finalrows,'final_fullhd_comparison')
    for key,name,ylabel in [('loss','loss_trajectory','0.5 RGB MSE'),('symmetric_chamfer','geometry_trajectory','Symmetric Chamfer')]:
        fig,ax=plt.subplots(figsize=(7,4))
        section='observation' if key=='loss' else 'geometry'
        for branch,r in reports.items():
            points=[r['baseline']]+r.get('births',[]); ax.plot(range(len(points)),[p[section][key] for p in points],'o-',label=branch)
        ax.set(xlabel='Born DoFs (not accepted-step count)',ylabel=ylabel); ax.legend(); save(fig,name)
    for name,key,ylabel in [('selected_utility','predicted_gain','Selected predicted / actual gain'),('runtime_per_birth','total_seconds','Seconds')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for branch,r in reports.items():
            b=r.get('births',[]); ax.plot([x['round'] for x in b],[x[key] for x in b],'o-',label=branch)
            if key=='predicted_gain': ax.plot([x['round'] for x in b],[x['realized_gain'] for x in b],'x--',label=branch+' actual')
        ax.set(xlabel='Outer round',ylabel=ylabel); ax.legend(); save(fig,name)
    import trimesh
    fig=plt.figure(figsize=(12,5))
    for j,(branch,r) in enumerate(reports.items()):
        ax=fig.add_subplot(1,2,j+1,projection='3d'); end=r['births'][-1] if r.get('births') else r['baseline']
        mesh=trimesh.load(end['geometry']['geometry_path'],process=False); p=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//5000)]
        ax.scatter(*p.T,s=1,alpha=.15,color='gray')
        centers=np.array([b['center'] for b in r.get('births',[])])
        if len(centers): ax.scatter(*centers.T,c=np.arange(len(centers)),cmap='viridis',s=30)
        ax.set_title(branch+' persistent born centers'); ax.set_box_aspect((1,1,1))
    save(fig,'birth_centers')
    if 'GLOBE' in reports:
        r=reports['GLOBE']; points=[('initial',r['baseline'])]+[(str(b['round']),b) for b in r.get('births',[]) if b['round'] in (1,4,8,16,32,64)]
        if r.get('births'): points.append(('final',r['births'][-1]))
        fig=plt.figure(figsize=(4*len(points),4))
        for j,(label,b) in enumerate(points):
            ax=fig.add_subplot(1,len(points),j+1,projection='3d'); mesh=trimesh.load(b['geometry']['geometry_path'],process=False); p=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//5000)]
            ax.scatter(*p.T,s=1); ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),zlim=(-1.4,1.4),title=label); ax.set_box_aspect((1,1,1))
        save(fig,'globe_evolution')
    return dict(shared_exposure=exposure,paths=[str(p) for p in sorted(folder.glob('v090_*.png'))])


def finalize():
    setup=load(CACHE/'setup.json'); reports={b:load(f'artifacts/v090_{b.lower()}_birth.json') for b in ('SOFT','GLOBE')}
    v=verdicts(setup,reports); display=figures(reports)
    performance=[]
    for branch,r in reports.items():
        initial=setup['branches'][branch]
        performance.append(dict(branch=branch,stage='initial',source_generation_seconds=initial['source']['source_seconds'],**{k:initial['baseline'][k] for k in ('source_seconds','transport_seconds','detector_seconds','total_seconds','peak_cuda_allocated_mib','peak_cuda_reserved_mib','cpu_rss_peak_mib')}))
        cv=r.get('candidate_validation',{})
        if cv:
            performance.append(dict(branch=branch,stage='candidate_scoring',scoring_seconds=cv['scores']['seconds'],cpu_rss_peak_mib=cv['scores']['cpu_rss_peak_mib'],peak_cuda_allocated_mib=cv['scores']['peak_cuda_allocated_mib'],peak_cuda_reserved_mib=cv['scores']['peak_cuda_reserved_mib']))
            for solo in cv['rows']:
                performance.append(dict(branch=branch,stage='single_birth_validation',candidate=solo['candidate'],optimization_seconds=solo['optimization']['seconds'],geometry_seconds=solo['geometry']['geometry_seconds']))
        for row in r.get('births',[]):
            performance.append(dict(branch=branch,stage='sequential',round=row['round'],scoring_seconds=row['scoring_seconds'],optimization_seconds=row['optimization']['seconds'],total_seconds=row['total_seconds'],peak_cuda_allocated_mib=row['peak_cuda_allocated_mib'],peak_cuda_reserved_mib=row['peak_cuda_reserved_mib'],cpu_rss_peak_mib=row.get('cpu_rss_peak_mib')))
        write_json(Path(f'artifacts/v090_{branch.lower()}_birth.json'),r)
        _write_csv(Path(f'artifacts/v090_{branch.lower()}_birth.csv'),_scalar_csv_rows(r))
    if not performance: performance=[dict(branch=b,stage='candidate_validation_only',scoring_seconds=r.get('candidate_validation',{}).get('scores',{}).get('seconds')) for b,r in reports.items()]
    _write_csv(Path('artifacts/v090_birth_performance.csv'),performance)
    report=dict(experiment='v0.9.0 matched CURRENT geometry birth',starting_state=load(CACHE/'starting_state.json'),formal_code_state=load(CACHE/'formal_code_state.json'),
        formal_code_history=load(CACHE/'formal_code_history.json'),support_cull_validation=load(CACHE/'support_cull_equivalence.json'),
        wall_seconds_since_starting_manifest=time.time()-(CACHE/'starting_state.json').stat().st_mtime,
        execution_notes=['A partial first single-candidate transport was interrupted to load equivalent support-AABB rejection before any single-candidate result existed. Completed initial scores and all baseline/target caches were retained. Phase timings exclude that unfinished attempt; manifest wall time includes implementation, checks, and interruptions.','Initial target accounting rejected out-of-window energy before explicit boundary accounting was added; a completed GT state was retained and reused. No detector/window/source renormalization.'],
        setup=setup,protocol=load(CACHE/'protocol.json'),branches=reports,verdicts=v,limitations=LIMITATIONS,
        completion=load(CACHE/'run_complete.json'),figures=display,git_policy='No commit or push. Preserve local v0817 dirty work and historical artifacts.')
    write_json(Path('artifacts/v090_matched_birth.json'),report)
    _write_csv(Path('artifacts/v090_matched_birth.csv'),_scalar_csv_rows(report))
    sections=[('Exact local code state',f"Starting HEAD `{report['starting_state']['head']}`, branch `{report['starting_state']['branch']}`; pre-existing dirty v0817 work preserved. No commit/push. Starting diff is embedded in the JSON."),
        ('Frozen v0.8.17 CURRENT operator',json.dumps(setup['config'],indent=2)),
        ('GT matched-soft target construction',json.dumps(setup['target'],indent=2)),
        ('Candidate derivative validation','Both branch multi-epsilon high/medium/weak tests are in v090_birth_fd.json. They are PREFLIGHT_ONLY, not evidence for Full-HD gain.'),
        ('SOFT baseline',json.dumps(reports['SOFT']['baseline'],indent=2)),
        ('SOFT candidate predicted-vs-actual validation',json.dumps({k:val for k,val in reports['SOFT'].get('candidate_validation',{}).items() if k not in ('scores','rows')},indent=2)),
        ('SOFT sequential birth',json.dumps(dict(status=reports['SOFT']['status'],**reports['SOFT']['summary']),indent=2)),
        ('GLOBE baseline',json.dumps(reports['GLOBE']['baseline'],indent=2)),
        ('GLOBE candidate predicted-vs-actual validation',json.dumps({k:val for k,val in reports['GLOBE'].get('candidate_validation',{}).items() if k not in ('scores','rows')},indent=2)),
        ('GLOBE sequential dynamic birth',json.dumps(dict(status=reports['GLOBE']['status'],**reports['GLOBE']['summary']),indent=2)),
        ('Geometry evaluation','Identical GT-field MC evaluation target; current-surface area samples, Chamfer, P2S, RMS, normal consistency, bounds and components recorded for each tested geometry. No MC rendering residual.'),
        ('Source-measure / attachment behavior',LIMITATIONS[1]+' Fixed weights and IDs; no source refresh. '+json.dumps(reports['GLOBE'].get('sphere_diagnostics',{}))),
        ('Runtime and memory','See v090_birth_performance.csv; includes source, transport, detector, candidate scoring, optimization and peak GPU/CPU memory. '+json.dumps(report['completion'])),
        ('Failure cases',json.dumps({b:dict(status=r['status'],error=r.get('error'),false_positive_candidates=r.get('candidate_validation',{}).get('false_positive_candidates')) for b,r in reports.items()},indent=2)),
        ('Exact verdicts',json.dumps(v,indent=2)),
        ('Scientific interpretation','Q1: candidate predictiveness is assessed independently by each branch Spearman/Pearson and full-budget single-birth gains. Q2/Q3: see separate observation and Chamfer trajectories; lower image loss alone is not geometry recovery. Q4: mesh-lighting/readout mismatch has been removed, but different reference measures, finite quadrature and source-layout sensitivity remain alternatives to geometry/observability. Q5: source count and weights are structurally decoupled from births; coverage/attachment usability is only established over the deformation actually reached.'),
        ('Limitations','\n\n'.join(LIMITATIONS))]
    text='# v0.9.0 matched CURRENT geometry-birth report\n\n'
    for i,(title,body) in enumerate(sections,1):
        text+=f'## {i}. {title}\n\n'+('```json\n'+body+'\n```' if body.startswith('{') else body)+'\n\n'
    Path('artifacts/v090_matched_birth.md').write_text(text)
    print(json.dumps(dict(summaries={b:r['summary'] for b,r in reports.items()},verdicts=v),indent=2),flush=True)


if __name__=='__main__': finalize()
