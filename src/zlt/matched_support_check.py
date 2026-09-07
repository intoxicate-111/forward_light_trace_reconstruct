"""Exact sparse-query culling equivalence; bounded performance preflight only."""
import time
import torch
from .matched_operator import CACHE,build_geometry,source,WendlandField
from .matched_preflight import propose
from .transverse_packet import write_json


class UnculledField(WendlandField):
    def pairs(self,p):
        support=self.index.query(p,float(self.radii[self.active].max()))
        bi=self.active[support.basis_ids]; pi=support.point_ids
        keep=(p[pi]-self.centers[bi]).square().sum(1)<self.radii[bi].square()
        return pi[keep],bi[keep]


def check():
    torch.set_num_threads(4); base,gt,sphere,boundary,cfg=build_geometry()
    state,_=source(base,'SOFT',1024,cfg); centers,radii,_=propose(state,32)
    p=(torch.quasirandom.SobolEngine(3,scramble=True,seed=908).draw(65536,dtype=torch.float64)*3.6-1.8).cuda()
    rows=[]
    for k in (1,8,32):
        coefficient=torch.linspace(-.003,.002,k,device='cuda',dtype=torch.float64)
        before=UnculledField(base,centers[:k],radii[:k],coefficient); after=WendlandField(base,centers[:k],radii[:k],coefficient)
        a=before.evaluate(p); b=after.evaluate(p); aj=before.jet(p); bj=after.jet(p)
        errors=[float((a-b).abs().max())]+[float((x-y).abs().max()) for x,y in zip(aj,bj)]
        assert max(errors)<1e-12,errors  # CUDA sparse reductions may reorder roundoff.
        times=[]
        for field in (before,after):
            torch.cuda.synchronize(); started=time.perf_counter()
            for _ in range(8): field.evaluate(p)
            torch.cuda.synchronize(); times.append((time.perf_counter()-started)/8)
        rows.append(dict(active_k=k,point_count=len(p),value_gradient_max_abs_error=errors[0],scalar_spatial_derivative_max_abs_error=errors[1],gradient_jacobian_max_abs_error=errors[2],before_seconds=times[0],after_seconds=times[1]))
    write_json(CACHE/'support_cull_equivalence.json',dict(stage='PREFLIGHT_ONLY',passed=True,tolerance=1e-12,rows=rows,
        unchanged='Only zero-support path points are rejected before the existing radius query; no forward transport, source, detector, score, or optimizer change.'))
    print(rows,flush=True)


if __name__=='__main__': check()
