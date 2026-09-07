"""Bounded analytic CURRENT geometry columns, accumulated at detector pixels.

Compact support is queried before basis evaluation. Only chunk x candidate-block
derivative workspace exists, never a 16M-emitter graph or global point x K field.
Column norms include cross-emitter overlap: sum the image column BEFORE squaring.
"""
import math
import time
import resource
import numpy as np
import torch
from .locality import UniformGridIndex,wendland_values,wendland_gradients
from .finite_packet import _compact_shell,_cubic
from .density_matrix import capture_offsets,capture_splat
from .measurement_bandwidth import gate,lobe
from .meshfree_surface import meshfree_base_color
from .fixed_measure import attach
from .transverse_packet import transmission_prefixes
from .matched_operator import WendlandField


class CandidateSupports:
    def __init__(self,centers,radii):
        self.centers,self.radii=centers,radii
        self.index=UniformGridIndex(centers,float(radii.max()))
    def query(self,p):
        s=self.index.query(p,float(self.radii.max())); pi,bi=s.point_ids,s.basis_ids
        d=p[pi]-self.centers[bi]; r=self.radii[bi]; keep=d.square().sum(1)<r.square()
        pi,bi,d,r=pi[keep],bi[keep],d[keep],r[keep]
        return pi,bi,wendland_values(d,r),wendland_gradients(d,r)


def source_tangent(field,p,nref,supports):
    values=field.evaluate(p); g=values[:,1:]; norm=g.norm(dim=1); normal=g/norm[:,None].clamp_min(1e-30)
    true_df,hg=field.jet(p); denominator=(true_df*nref).sum(1)
    pi,bi,b,db=supports.query(p); count=len(supports.radii)
    if len(pi) and bool((denominator[pi].abs()<1e-10).any()): raise RuntimeError('DEGENERATE_ATTACHMENT_DERIVATIVE')
    dx=p.new_zeros((len(p),count,3)); dn=torch.zeros_like(dx)
    dx_e=-nref[pi]*(b/denominator[pi])[:,None]
    dg=db+torch.einsum('eab,eb->ea',hg[pi],dx_e)
    dn_e=(dg-normal[pi]*(dg*normal[pi]).sum(1,keepdim=True))/norm[pi,None]
    dx[pi,bi]=dx_e; dn[pi,bi]=dn_e
    return dx,dn


def optical_tangent(field,p,d,maximum,dx,supports,cfg,path_chunk=32):
    """Derivative of the existing shell and surface-barrier path sum.

    Spatial derivative uses the true scalar-interpolant derivative and the
    Jacobian of the separately interpolated gradient, never an SDF identity.
    The exit-mask derivative is zero inside a quadrature cell, as in AD.
    """
    start=cfg['launch']*(cfg['h']+cfg['epsilon']); step=cfg['path_step']; eps=cfg['epsilon']; eta=cfg['eta']
    steps=max(1,math.ceil(float((maximum-start).clamp_min(0).max())/step))
    k=len(supports.radii); tau=p.new_zeros(len(p)); dtau=p.new_zeros((len(p),k)); spatial=p.new_zeros((len(p),3))
    for a in range(0,steps,path_chunk):
        times=start+(torch.arange(a,min(a+path_chunk,steps),device=p.device,dtype=p.dtype)+.5)*step
        valid=times[None]<maximum[:,None]; points=p[:,None]+times[None,:,None]*d[:,None]
        flat=points.reshape(-1,3); vals=field.evaluate(flat); f,g=vals[:,0],vals[:,1:]
        norm=(g.square().sum(1)+eta*eta).sqrt().clamp_min(1e-30); q=f/(norm*eps)
        owner=torch.arange(len(p),device=p.device).repeat_interleave(len(times)); direction=d[owner]
        dot=(g*direction).sum(1)/norm
        influence=_compact_shell(q)/eps*dot.abs()*valid.flatten()
        tau+=influence.reshape(len(p),len(times)).sum(1)*cfg['kappa']*step
        hit=(q.abs()<1)&valid.flatten(); ids=hit.nonzero().flatten()
        if not len(ids): continue
        x=flat[ids]; own=owner[ids]; gv=g[ids]; nv=norm[ids]; fv=f[ids]; qv=q[ids]; av=dot[ids]; dv=direction[ids]
        psi=_compact_shell(qv)/eps; dpsi=(-15/4*qv*(1-qv*qv))/eps**2
        af=dpsi*av.abs()/nv
        ag=-dpsi[:,None]*av.abs()[:,None]*fv[:,None]*gv/nv[:,None]**3
        ag+=psi[:,None]*av.sign()[:,None]*(dv/nv[:,None]-(gv*dv).sum(1)[:,None]*gv/nv[:,None]**3)
        factor=cfg['kappa']*step; af*=factor; ag*=factor
        true_df,hg=field.jet(x)
        spatial.index_add_(0,own,af[:,None]*true_df+torch.einsum('eab,ea->eb',hg,ag))
        pi,bi,b,db=supports.query(x)
        dtau.flatten().index_add_(0,own[pi]*k+bi,af[pi]*b+(ag[pi]*db).sum(1))
    dtau+=torch.einsum('na,nka->nk',spatial,dx)
    return tau,dtau


def cubic_derivative(x):
    a=x.abs(); return x.sign()*torch.where(a<1,-2*a+1.5*a*a,-.5*(2-a).clamp_min(0)**2)


def integrated_kernel(row,col,capture,resolution):
    """Adjoint/tangent only: exact algebraic separability of unchanged C3.

    The forward renderer still executes historical 16 translated cubic splats.
    This six-slot union contains all nonzero taps; unused taps have zero weight.
    """
    offsets=torch.arange(-2,4,device=row.device); ri=torch.floor(row).long()[:,None]+offsets
    ci=torch.floor(col).long()[:,None]+offsets
    rw=torch.zeros_like(ri,dtype=row.dtype); cw=torch.zeros_like(ci,dtype=row.dtype)
    dr=torch.zeros_like(rw); dc=torch.zeros_like(cw)
    for shift in capture_offsets(capture):
        rr=row[:,None]-float(shift)-ri; cc=col[:,None]-float(shift)-ci
        rw+=_cubic(rr)/capture; cw+=_cubic(cc)/capture
        dr+=cubic_derivative(rr)/capture; dc+=cubic_derivative(cc)/capture
    h,w=resolution; rr=ri[:,:,None].expand(-1,6,6).reshape(-1,36); cc=ci[:,None,:].expand(-1,6,6).reshape(-1,36)
    valid=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
    ids=rr.clamp(0,h-1)*w+cc.clamp(0,w-1)
    return ids,(rw[:,:,None]*cw[:,None,:]).reshape(-1,36)*valid,(dr[:,:,None]*cw[:,None,:]).reshape(-1,36)*valid,(rw[:,:,None]*dc[:,None,:]).reshape(-1,36)*valid


def differential_block(field,reference,state,owners,view,supports,boundary,cfg,diagnostic=False):
    p=state['positions'][owners].cuda(); normal=state['normals'][owners].cuda(); nref=reference['normals'][owners].cuda()
    d=state['directions'][view].cuda().expand_as(p); cosine=(normal*d).sum(1)
    keep=cosine>0; p,normal,nref,d,cosine=p[keep],normal[keep],nref[keep],d[keep],cosine[keep]
    if not len(p): return None
    dx,dn=source_tangent(field,p,nref,supports)
    tau,dtau=optical_tangent(field,p,d,boundary.exit_times(p,d),dx,supports,cfg)
    transmission=state['transmission'][view,owners].cuda()[keep]
    error=float((torch.exp(-tau)-transmission).abs().max())
    if error>1e-8: raise RuntimeError(f'CURRENT_TANGENT_FORWARD_MISMATCH:{error}')
    color=state['colors'][owners].cuda()[keep]; weight=state['weights'][owners].cuda()[keep]
    g=gate(cosine,'CURRENT_SOFT_W005'); ell=lobe(cosine,'CURRENT_LOBE')
    energy=weight[:,None]*transmission[:,None]*color*(g*ell)[:,None]
    dc=torch.einsum('nka,na->nk',dn,d)
    dg=2*cosine/cfg['gate_width']**2*torch.exp(-cosine.square()/cfg['gate_width']**2)
    grad_color=p.new_tensor([.72,.70,-.62])/(field.upper-field.lower)
    inside=((p>=field.lower)&(p<=field.upper)).to(p.dtype)
    dcolor=dx*grad_color[None,None,:]*inside[:,None,:]
    de=(weight*transmission)[:,None,None]*(dcolor*(g*ell)[:,None,None]+color[:,None,:]*(dc*(dg*ell+.65*g)[:,None])[:,:,None])
    de-=energy[:,None,:]*dtau[:,:,None]
    right=state['right'][view].cuda(); up=state['up'][view].cuda(); center=state['center'].cuda()
    h,w=cfg['resolution']
    if diagnostic:
        rel=p-center; row=(.5-rel@up/cfg['extent'])*h-.5; col=(rel@right/cfg['extent']+.5)*w-.5
    else:
        rel=p.float()-center.float(); row=(.5-rel@up.float()/cfg['extent'])*h-.5; col=(rel@right.float()/cfg['extent']+.5)*w-.5
        up,right=up.float().double(),right.float().double()
    drow=-torch.einsum('nka,a->nk',dx,up)*h/cfg['extent']; dcol=torch.einsum('nka,a->nk',dx,right)*w/cfg['extent']
    return row,col,energy,de,drow,dcol,error


def columns(field,reference,state,centers,radii,boundary,cfg,residual=None,diagnostic=False,return_images=False,chunk=2048):
    k=len(radii); supports=CandidateSupports(centers,radii); h,w=cfg['resolution']; m=4*h*w*3
    gram=np.zeros((k,k)); alignment=np.zeros(k); per_view=[]; affected=np.zeros(k,dtype=np.int64); saved=[]
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); error=0.
    for v in range(4):
        acc=torch.zeros((h*w,k,3),device='cuda',dtype=torch.float64 if diagnostic else torch.float32)
        for i in range(0,len(state['weights']),chunk):
            block=differential_block(field,reference,state,slice(i,i+chunk),v,supports,boundary,cfg,diagnostic)
            if block is None: continue
            row,col,energy,de,drow,dcol,err=block; error=max(error,err)
            ids,weights,dy,dx=integrated_kernel(row,col,cfg['capture'],cfg['resolution'])
            # One tap at a time bounds temporary storage independently of K*N.
            for tap in range(36):
                contribution=weights[:,tap,None,None]*de+(dy[:,tap,None]*drow+dx[:,tap,None]*dcol)[:,:,None]*energy[:,None,:]
                acc.index_add_(0,ids[:,tap],contribution.to(acc.dtype))
            if i and i%(chunk*256)==0: print(f'Jacobian view={v} {i:,}/{len(state["weights"]):,}, K={k}',flush=True)
        acc*=cfg['gain']*h*w
        g=torch.zeros((k,k),device='cuda',dtype=torch.float64); a=torch.zeros(k,device='cuda',dtype=torch.float64)
        counts=torch.zeros(k,device='cuda',dtype=torch.long)
        for i in range(0,h*w,65536):
            part=acc[i:i+65536]; j=part.permute(0,2,1).reshape(-1,k).double()
            g+=j.T@j
            if residual is not None:
                r=torch.as_tensor(residual[v].reshape(-1,3)[i:i+65536],device='cuda',dtype=torch.float64).flatten(); a+=j.T@r
            counts+=(part.abs().amax(2)>1e-12).sum(0)
        gram+=g.cpu().numpy()/m; alignment+=a.cpu().numpy()/m; affected+=counts.cpu().numpy()
        per_view.append((g.diag()/m).cpu().tolist())
        if return_images: saved.append(acc.reshape(h,w,k,3).cpu().numpy())
        del acc
    norms=np.maximum(np.diag(gram),0)
    if not np.isfinite(gram).all() or not np.isfinite(alignment).all():
        raise RuntimeError('NONFINITE_CURRENT_CANDIDATE_RESPONSE')
    result=dict(alignment=alignment.tolist(),raw=np.abs(alignment).tolist(),norm2=norms.tolist(),
        quadratic=(.5*alignment**2/(norms+1e-7)).tolist(),jacobian_norm=np.sqrt(norms).tolist(),gram=gram.tolist(),
        affected_pixels=affected.tolist(),affected_views=(np.asarray(per_view)>1e-20).sum(0).tolist(),per_view_norm2=per_view,
        norm_method='Image columns accumulated over ALL emitters before inner products; no independent-packet norm approximation. Float32 production accumulation, float64 reductions.',
        objective='0.5*mean((I-current_target)^2), columns and residual normalized by sqrt(4*H*W*3)',
        current_transmission_max_abs_error=error,seconds=time.perf_counter()-started,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        cpu_rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,source_count=len(state['weights']),candidate_block=k)
    return result,saved


def diagnostic_forward(base,reference,centers,radii,coefficients,boundary,cfg):
    """PREFLIGHT_ONLY: full precision readout for meaningful finite differences."""
    field=WendlandField(base,centers,radii,coefficients); p=reference['positions'].cuda(); nr=reference['normals'].cuda()
    x,n=attach(field,p,nr); state=dict(reference,positions=x.cpu(),normals=n.cpu(),colors=meshfree_base_color(x,base.lower,base.upper).cpu())
    trans=[]; images=[]; h,w=cfg['resolution']
    for v in range(4):
        d=state['directions'][v].cuda().expand_as(x)
        t,_,_=transmission_prefixes(field,x,d,boundary.exit_times(x,d),radius=cfg['h'],epsilon=cfg['epsilon'],path_step=cfg['path_step'],offsets=x.new_zeros((1,3)),counts=(1,),eta=cfg['eta'],kappa=cfg['kappa'],launch_exclusion_factor=cfg['launch'])
        trans.append(t[:,0].cpu()); c=(n*d).sum(1)
        e=state['weights'].cuda()[:,None]*t*gate(c,'CURRENT_SOFT_W005')[:,None]*lobe(c,'CURRENT_LOBE')[:,None]*state['colors'].cuda()
        rel=x-state['center'].cuda(); row=(.5-rel@state['up'][v].cuda()/cfg['extent'])*h-.5; col=(rel@state['right'][v].cuda()/cfg['extent']+.5)*w-.5
        acc=x.new_zeros((h*w,3)); capture_splat(acc,row,col,e,cfg['capture'],resolution=(h,w))
        images.append((acc.reshape(h,w,3)*(cfg['gain']*h*w)).cpu().numpy())
    state['transmission']=torch.stack(trans)
    return np.stack(images),state
