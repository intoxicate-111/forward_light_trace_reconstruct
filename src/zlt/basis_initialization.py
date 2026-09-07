"""Deterministic signed sphere constraints; sphere is only initialization data."""
import numpy as np
import torch
from .basis_only import BasisOnlyZeroSetField
from .locality import BasisLayout,wendland_values


def directions(n,seed=191):
    u=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed).draw(n,dtype=torch.float64)
    z=1-2*u[:,0]; phi=2*torch.pi*u[:,1]; r=(1-z*z).sqrt()
    return torch.stack((r*phi.cos(),r*phi.sin(),z),1)


def layout(k,radius):
    # Deterministic volumetric farthest-point sampling; no residual or surface-only centers.
    u=torch.quasirandom.SobolEngine(3,scramble=True,seed=192).draw(16384,dtype=torch.float64).numpy()*2.3-1.15
    pool=np.concatenate((np.zeros((1,3)),u[np.linalg.norm(u,axis=1)<1.15]))
    distance=np.full(len(pool),np.inf); selected=[]; current=0
    for _ in range(k):
        selected.append(current); distance=np.minimum(distance,np.sum((pool-pool[current])**2,axis=1)); distance[selected]=-1
        current=int(distance.argmax())
    return BasisLayout(torch.from_numpy(pool[selected].copy()),torch.full((k,),radius,dtype=torch.float64))


def shell_points(n,low,high,seed):
    # Joint 3D Sobol coordinates: separate sequences for radius/direction can
    # introduce correlations and leave entire interior sectors unconstrained.
    u=torch.quasirandom.SobolEngine(3,scramble=True,seed=seed).draw(n,dtype=torch.float64)
    z=1-2*u[:,0]; phi=2*torch.pi*u[:,1]; radial=(1-z*z).sqrt()
    d=torch.stack((radial*phi.cos(),radial*phi.sin(),z),1)
    radius=(low**3+(high**3-low**3)*u[:,2]).pow(1/3)
    return d*radius[:,None]


def constraints():
    surf=directions(4096)
    inner=shell_points(4096,0,.95,193)
    outer=shell_points(4096,1.05,1.6,195)
    boundary=directions(1024,196)*2.
    p=torch.cat((surf,inner,outer,boundary,torch.zeros((1,3),dtype=torch.float64)))
    y=(p.square().sum(1)-1).clamp(-1,1)
    w=torch.cat((torch.full((4096,),3.),torch.ones(8192),torch.ones(1024),torch.tensor([4.]))).double()
    return p,y,w


def design(p,layout):
    # Modest initialization matrix only; rendering always uses sparse support queries.
    a=p.new_empty((len(p),layout.count))
    for i in range(0,len(p),512):
        d=p[i:i+512,None]-layout.centers[None]
        a[i:i+512]=wendland_values(d.reshape(-1,3),layout.radii.expand(len(d),-1).reshape(-1)).reshape(len(d),-1)
    return a


def initialize(k,radius,ridge=1e-6):
    basis=layout(k,radius); p,y,w=constraints(); a=design(p,basis)
    aw=a*w[:,None]; rhs=(y-1)*w
    gram=aw.T@aw+ridge*torch.eye(k,dtype=torch.float64)
    coeff=torch.linalg.solve(gram,aw.T@rhs)
    field=BasisOnlyZeroSetField(basis,coeff)
    return field,dict(k=k,radius=radius,ridge=ridge,constraints=len(p),surface_count=4096,interior_count=4097,
        exterior_count=4096,boundary_count=1024,target='clip(norm(x)^2-1,-1,1)',surface_weight=3.,center_weight=4.,other_weight=1.,
        condition=float(torch.linalg.cond(gram)),training_mse=float(((a@coeff+1-y)**2).mean()),
        coefficient_min=float(coeff.min()),coefficient_max=float(coeff.max()),coefficient_rms=float(coeff.square().mean().sqrt()),coefficient_norm=float(coeff.norm())),(p,y,w,a,gram)


def radial_roots(field,d,scan_steps=81):
    d=d.to(field.coefficients.device); ts=torch.linspace(0,2,scan_steps,device=d.device,dtype=d.dtype)
    # Chunk directions: bound support-pair workspace during radial scanning.
    values=[]
    for i in range(0,len(d),128): values.append(field.value((d[i:i+128,None]*ts[None,:,None]).reshape(-1,3)).reshape(-1,scan_steps))
    values=torch.cat(values)
    # Count a root exactly on a scan node once, not zero times or twice.
    crossing=((values[:,:-1]<0)&(values[:,1:]>=0))|((values[:,:-1]>0)&(values[:,1:]<=0))
    counts=crossing.sum(1); distances=(ts[:-1]+ts[1:])*.5
    chosen=torch.where(crossing,(distances-1).abs()[None],torch.inf).argmin(1)
    lo=ts[chosen]; hi=ts[chosen+1]; flo=values.gather(1,chosen[:,None])[:,0]
    for _ in range(35):
        mid=(lo+hi)/2; fm=field.value(d*mid[:,None]); left=flo*fm<=0
        hi=torch.where(left,mid,hi); lo=torch.where(left,lo,mid); flo=torch.where(left,flo,fm)
    t=(lo+hi)/2; valid=counts>0; error=(t[valid]-1).abs(); g=field.gradient(d[valid]*t[valid,None]).norm(dim=1)
    stats=dict(root_coverage=float(valid.double().mean()),radial_mae=float(error.mean()) if len(error) else None,
        radial_median=float(error.median()) if len(error) else None,radial_p95=float(torch.quantile(error,.95)) if len(error) else None,
        radial_max=float(error.max()) if len(error) else None,signed_radial_bias=float((t[valid]-1).mean()) if len(error) else None,
        min_zeroset_gradient_norm=float(g.min()) if len(g) else None,median_zeroset_gradient_norm=float(g.median()) if len(g) else None,
        multiple_crossing_fraction=float((counts>1).double().mean()),no_crossing_count=int((counts==0).sum()),directions=len(d),scan_steps=scan_steps)
    return t,valid,stats
