"""Full-HD matched GT target, canonical regression and both initial states."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .matched_operator import CACHE,build_geometry,source,forward,WendlandField,config_digest
from .matched_preflight import preserve,propose
from .matched_jacobian import columns
from .density_matrix import render
from .transverse_packet import digest,write_json


def setup():
    preserve(); torch.set_num_threads(4)
    pref=json.loads((CACHE/'preflight.json').read_text()); assert pref['completed']
    base,gt,sphere,boundary,cfg=build_geometry(); assert pref['config']==cfg
    empty=base.grid.new_empty((0,3)); coefficients=base.grid.new_empty(0); report=dict(config=cfg,branches={})
    with torch.no_grad():
        # Full production regression precedes the new GT formal render.
        soft_ref,soft_meta=source(base,'SOFT',cfg['count'],cfg)
        soft_image,soft_state,soft_time=forward(base,soft_ref,empty,coefficients,coefficients,boundary,cfg,'soft_initial')
        old=np.load('runs/v0817_collision_transfer/CURRENT.npy')
        error=float(np.linalg.norm(soft_image.astype(float)-old)/np.linalg.norm(old.astype(float)))
        assert error<1e-6; report['v0817_current_replay_relative_l2']=error
        del old
        gt_ref,gt_meta=source(gt,'GT',cfg['count'],cfg)
        target,target_state,target_meta=forward(gt,gt_ref,empty,coefficients,coefficients,boundary,cfg,'gt_current_soft_16m')
        replay,occupancy,accounting,timing=render(target_state,cfg['capture']); del occupancy
        replay=np.stack(replay); delta=float(np.linalg.norm(replay.astype(float)-target)/np.linalg.norm(target.astype(float)))
        assert delta<1e-6; del replay
        report['target']=dict(source=gt_meta,render=target_meta,replay_relative_l2=delta,replay_seconds=timing['readout_seconds'],
            digest=digest(torch.from_numpy(target)),field_digest=gt_meta['signature']['field_digest'],lambda_parameters=0,
            valid=True,path=str(CACHE/'gt_current_soft_16m.npy'),definition='CURRENT finite-packet integration of the GT implicit field; independent globally uniform implicit-area source; fixed normalized source measure; no hard reference or mesh readout.')
        report['branches']['SOFT']=dict(source=soft_meta,baseline=soft_time)
        globe_ref,globe_meta=source(sphere,'GLOBE',cfg['count'],cfg)
        globe_image,globe_state,globe_time=forward(sphere,globe_ref,empty,coefficients,coefficients,boundary,cfg,'globe_initial')
        report['branches']['GLOBE']=dict(source=globe_meta,baseline=globe_time)
        # Full-HD derivative runtime estimate only, NEVER a birth result.
        budgets=[]
        for branch,f,reference,state,image in (('SOFT',base,soft_ref,soft_state,soft_image),('GLOBE',sphere,globe_ref,globe_state,globe_image)):
            centers,radii,_=propose(state,32)
            for n in (16384,32768):
                sub={k:(v[:,:n] if k=='transmission' else v[:n] if k in ('positions','normals','colors','weights') else v) for k,v in state.items()}
                ref={k:(v[:n] if k in ('positions','normals','colors','weights') else v) for k,v in reference.items()}
                scored,_=columns(WendlandField(f),ref,sub,centers,radii,boundary,cfg,residual=image.astype(float)-target,chunk=8192)
                budgets.append(dict(branch=branch,stage='PREFLIGHT_ONLY',samples=n,**scored))
        report['runtime_estimates']=budgets
    write_json(CACHE/'setup.json',report); print('Matched setup complete',flush=True)


if __name__=='__main__': setup()
