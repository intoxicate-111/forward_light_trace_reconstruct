"""v0.9.0 frozen CURRENT observation operator and independent source measures.

GT sampling uses dominant-axis implicit-area rejection, not marching cubes.
No candidate or active coefficient is ever consulted by initial source sampling.
"""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .fields import SphereField
from .locality import UniformGridIndex,wendland_values,wendland_gradients
from .transverse_packet import FusedGridLookup,transmission_prefixes,digest,write_json
from .fixed_measure import attach
from .meshfree_surface import meshfree_base_color,sign_changing_cells
from .density_matrix import capture_splat,render
from .boundary_transport import ObservationSphere

CACHE=Path('runs/v090_matched_birth')
OPERATOR_VERSION='v0817_CURRENT_fixed_measure_C3'


def frozen_config():
    previous=json.loads(Path('artifacts/v0817_collision_transfer.json').read_text())
    earlier=json.loads(Path('artifacts/v0816_global_sample_convergence.json').read_text())
    assert previous['verdicts']['BEST_TRANSFER_LAW']['family']=='CURRENT'
    f=earlier['frozen']; t=previous['geometry_closure']
    return dict(operator=OPERATOR_VERSION,h=t['h'],epsilon=f['shell_width'],path_step=f['path_step'],eta=t['eta'],
        kappa=-math.log(.01),launch=1.05,micro=1,ambient=f['ambient'],gate_width=f['gate_width'],
        gain=f['sensor_gain'],extent=f['window'],capture=f['capture_grid'],resolution=f['resolution'],views=f['views'],
        count=16777216,mass=previous['frozen']['source_mass'],seed=101,base_grid_digest=f['grid_digest'],
        source_dtype='float64',transport_dtype='float64',detector_dtype='float32',precision_note='As v0817; float64 detector in bounded FD diagnostics only.')


def config_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class SphereAdapter(SphereField):
    def __init__(self,lower,upper):
        super().__init__(radius=1.)
        object.__setattr__(self,'lower',lower); object.__setattr__(self,'upper',upper)


def field_digest(base):
    return digest(base.grid) if hasattr(base,'grid') else config_digest(dict(kind='SphereField',radius=base.radius,lower=base.lower.tolist(),upper=base.upper.tolist()))


def basis_hessian(offsets,radii):
    r=offsets.norm(dim=1); q=r/radii; a=(1-q).clamp_min(0)
    eye=torch.eye(3,device=r.device,dtype=r.dtype)
    return (-20*a**3/radii**2)[:,None,None]*eye+(60*a**2/(radii**3*r.clamp_min(1e-15)))[:,None,None]*offsets[:,:,None]*offsets[:,None,:]


def grid_jet(base,points):
    """Exact derivatives of scalar and separately interpolated gradient channels.

    This is NOT an SDF Hessian. Border-clamp coordinate derivatives are zero
    outside the grid, matching the actual trilinear evaluator's AD semantics.
    """
    dims=points.new_tensor(base.grid.shape); scale=(dims-1)/(base.upper-base.lower)
    coord=(points-base.lower)*scale; index=torch.floor(coord).long()
    index=torch.minimum(index.clamp_min(0),dims.long()-2)
    fraction=(coord-index).clamp(0,1)
    live=((coord>=0)&(coord<=dims-1)).to(points.dtype)
    if not hasattr(base,'_matched_jet_channels'):
        object.__setattr__(base,'_matched_jet_channels',torch.cat((base.grid[...,None],base.gradient_grid),-1).reshape(-1,4))
    channels=base._matched_jet_channels
    jac=points.new_zeros((len(points),4,3)); ny,nz=base.grid.shape[1:]
    for dx,dy,dz in itertools.product((0,1),repeat=3):
        bits=(dx,dy,dz); weights=[fraction[:,a] if bits[a] else 1-fraction[:,a] for a in range(3)]
        values=channels[((index[:,0]+dx)*ny+index[:,1]+dy)*nz+index[:,2]+dz]
        for a in range(3):
            other=[b for b in range(3) if b!=a]
            w=(1 if bits[a] else -1)*weights[other[0]]*weights[other[1]]*scale[a]*live[:,a]
            jac[:,:,a]+=values*w[:,None]
    return jac[:,0],jac[:,1:]


class WendlandField(FusedGridLookup):
    """CURRENT-compatible fused queries, local COO updates, no global N x K."""
    def __init__(self,base,centers=None,radii=None,coefficients=None):
        self.base=base; self.lower,self.upper=base.lower,base.upper
        self.lookup=FusedGridLookup(base) if hasattr(base,'grid') else None
        self.centers=centers; self.radii=radii; self.coefficients=coefficients
        self.active=(coefficients!=0).nonzero().flatten() if coefficients is not None else None
        self.index=None
        if self.active is not None and len(self.active):
            self.index=UniformGridIndex(centers[self.active],float(radii[self.active].max()))
            self.support_lower=(centers[self.active]-radii[self.active,None]).amin(0)
            self.support_upper=(centers[self.active]+radii[self.active,None]).amax(0)

    def pairs(self,p):
        # Exact support-union AABB rejection BEFORE the radius query. This is
        # only O(N,3), never point-by-basis broadcasting. Path points outside
        # this box cannot have nonzero Wendland value, gradient, or Hessian.
        owners=((p>=self.support_lower)&(p<=self.support_upper)).all(1).nonzero().flatten()
        if not len(owners): return owners,owners
        support=self.index.query(p[owners],float(self.radii[self.active].max()))
        bi=self.active[support.basis_ids]; pi=owners[support.point_ids]
        keep=(p[pi]-self.centers[bi]).square().sum(1)<self.radii[bi].square()
        return pi[keep],bi[keep]

    def evaluate(self,p):
        out=self.lookup.evaluate(p) if self.lookup is not None else torch.cat((self.base.value(p)[:,None],self.base.gradient(p)),1)
        if self.index is not None:
            pi,bi=self.pairs(p); d=p[pi]-self.centers[bi]; radii=self.radii[bi]
            terms=torch.cat((wendland_values(d,radii)[:,None],wendland_gradients(d,radii)),1)
            out=out.index_add(0,pi,terms*self.coefficients[bi,None])
        return out

    def jet(self,p):
        if self.lookup is None:
            df=self.base.gradient(p); hg=torch.eye(3,device=p.device,dtype=p.dtype)[None].expand(len(p),3,3)*2
        else: df,hg=grid_jet(self.base,p)
        if self.index is not None:
            pi,bi=self.pairs(p); d=p[pi]-self.centers[bi]; r=self.radii[bi]; c=self.coefficients[bi]
            df=df.index_add(0,pi,wendland_gradients(d,r)*c[:,None])
            hg=hg.index_add(0,pi,basis_hessian(d,r)*c[:,None,None])
        return df,hg


def build_geometry():
    from .mesh_field import prepare_stanford_bunny
    from .meshfree_surface import sample_meshfree_zero_set
    from .boundary_transport import enclosing_observation_sphere
    cfg=frozen_config()
    cached=CACHE/'fields.pt'; metadata=CACHE/'fields.json'
    if cached.exists():
        from .mesh_field import GridZeroSetField
        report=json.loads(metadata.read_text()); assert report['config_digest']==config_digest(cfg)
        data=torch.load(cached,weights_only=True)
        def restore(name):
            f=data[name]; p=f['grid'].new_empty((0,3)); face=torch.empty((0,3),dtype=torch.long)
            return GridZeroSetField(f['grid'],f['gradient_grid'],f['lower'],f['upper'],p,face,f['grid'].new_empty(0)).to(torch.device('cuda'))
        base,gt=restore('base'),restore('gt')
        assert digest(base.grid)==cfg['base_grid_digest'] and digest(gt.grid)==report['gt_digest']
        boundary=ObservationSphere(data['boundary_center'].cuda(),data['boundary_radius'])
        return base,gt,SphereAdapter(base.lower,base.upper),boundary,cfg
    prepared=prepare_stanford_bunny(Path('data/stanford_bunny/cache/bun_zipper.ply'),build_surface_scaffold=False)
    base=prepared.base_field.to(torch.device('cuda')); gt=prepared.gt_field.to(torch.device('cuda'))
    assert digest(base.grid)==cfg['base_grid_digest']
    historical=sample_meshfree_zero_set(base,4096,sobol_scramble_seed=101)
    boundary=enclosing_observation_sphere(historical.points)
    data={name:{k:getattr(f,k).cpu() for k in ('grid','gradient_grid','lower','upper')} for name,f in (('base',base),('gt',gt))}
    data.update(boundary_center=boundary.center.cpu(),boundary_radius=boundary.radius)
    CACHE.mkdir(parents=True,exist_ok=True); torch.save(data,cached)
    write_json(metadata,dict(config_digest=config_digest(cfg),base_digest=digest(base.grid),gt_digest=digest(gt.grid),
        shape=list(base.grid.shape),dtype='float64',seed=101,sample_count=4096,operator_version=OPERATOR_VERSION,
        source='Unchanged prepared.base_field and prepared.gt_field; no surface scaffold.'))
    return base,gt,SphereAdapter(base.lower,base.upper),boundary,cfg


def camera_state():
    old=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
    return {k:old[k] for k in ('center','directions','right','up')}


def implicit_area_points(field,count,seed=101,block=131072):
    """Uniform-area samples on the regular trilinear zero set, by rejection.

    Uniform cell / axis / projected uv; exact linear root along selected axis.
    Keep only the unique largest-|normal| axis, then accept with
    1/(sqrt(3)*|n_axis|). Coarea cancels projected-area density. No mesh, chart
    centers, learned coefficients, or SDF assumption enters this construction.
    """
    cells=sign_changing_cells(field.grid); dims=field.grid.new_tensor(field.grid.shape)
    voxel=(field.upper-field.lower)/(dims-1)
    if not torch.allclose(voxel,voxel[0].expand_as(voxel)): raise ValueError('equal voxel lengths required by uniform cell/axis measure')
    sampler=torch.quasirandom.SobolEngine(5,scramble=True,seed=seed)
    produced=attempts=0
    while produced<count:
        u=sampler.draw(block,dtype=torch.float64).to(field.grid.device); attempts+=len(u)
        ids=(u[:,0]*len(cells)).long(); axis=(u[:,1]*3).long(); ij=cells[ids]
        a=field.lower+ij*voxel
        # cyclic tangential axes, identical proposal density for each axis
        row=torch.arange(len(u),device=u.device); b=(axis+1)%3; c=(axis+2)%3
        a[row,b]+=u[:,2]*voxel[b]; a[row,c]+=u[:,3]*voxel[c]
        end=a.clone(); end[row,axis]+=voxel[axis]
        fa=field.value(a); fb=field.value(end); den=fa-fb
        ok=(fa*fb<=0)&(den.abs()>1e-20)
        p=a.clone(); p[row,axis]+=(fa/torch.where(ok,den,torch.ones_like(den)))*voxel[axis]
        p=p[ok]; axes=axis[ok]; coin=u[ok,4]
        grad=field.interpolant_gradient(p); norm=grad.norm(dim=1); unit=grad/norm[:,None].clamp_min(1e-30)
        dominant=unit.abs().argmax(1); component=unit.abs().gather(1,axes[:,None])[:,0]
        keep=(dominant==axes)&(norm>1e-12)&(coin<1/(math.sqrt(3)*component.clamp_min(1e-30)))
        p=p[keep][:count-produced]
        produced+=len(p)
        if len(p): yield p,attempts,produced


def source(base,branch,count,cfg):
    name=f'{branch.lower()}_source_{count}'; path=CACHE/(name+'.pt'); meta=CACHE/(name+'.json')
    signature=dict(config_digest=config_digest(cfg),field_digest=field_digest(base),count=count,seed=cfg['seed'],
        operator_version=OPERATOR_VERSION,shape=[count,3],dtype='float64',branch=branch)
    if path.exists():
        old=json.loads(meta.read_text()); assert old['signature']==signature,'stale source cache'
        result=torch.load(path,weights_only=True,mmap=True); return result,old
    started=time.perf_counter(); state=camera_state()
    if branch=='SOFT':
        old=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
        for k in ('positions','normals','colors'): state[k]=old[k][:count].clone()
        state['transmission']=old['transmission'][:,:count].clone()
        definition='Unchanged v0817 uniform polyhedral reference-area source, already attached to the base field; persistent normal-line pushforward under lambda.'
        attempted=count
    else:
        state.update({k:torch.empty((count,3),dtype=torch.float64) for k in ('positions','normals','colors')})
        if branch=='GT':
            iterator=implicit_area_points(base,count,cfg['seed'])
            definition='Uniform area on the GT trilinear implicit surface by dominant-axis coarea rejection; fixed thereafter. No marching cubes or hard visibility.'
        else:
            sampler=torch.quasirandom.SobolEngine(2,scramble=True,seed=cfg['seed'])
            def sphere_parts():
                for start in range(0,count,131072):
                    u=sampler.draw(min(131072,count-start),dtype=torch.float64).cuda()
                    z=1-2*u[:,0]; phi=2*torch.pi*u[:,1]; r=(1-z*z).clamp_min(0).sqrt()
                    yield torch.stack((r*phi.cos(),r*phi.sin(),z),1),start+len(u),start+len(u)
            iterator=sphere_parts(); definition='Analytic radius=1 sphere uniform area, Sobol seed 101; fixed normal-line pushforward under lambda.'
        done=0; attempted=0; residual=0.
        for points,attempted,produced in iterator:
            n=base.gradient(points); n=n/n.norm(dim=1,keepdim=True).clamp_min(1e-30)
            state['positions'][done:produced]=points.cpu(); state['normals'][done:produced]=n.cpu()
            state['colors'][done:produced]=meshfree_base_color(points,base.lower,base.upper).cpu()
            residual=max(residual,float(base.value(points).abs().max())); done=produced
            if done//1048576>(done-len(points))//1048576: print(f'{branch} source {done:,}/{count:,}',flush=True)
        assert done==count and residual<1e-8
    state['weights']=torch.full((count,),cfg['mass']/count,dtype=torch.float64)
    metadata=dict(signature=signature,definition=definition,attempted_proposals=attempted,source_seconds=time.perf_counter()-started,
        position_digest=digest(state['positions']),normal_digest=digest(state['normals']),weight_digest=digest(state['weights']),source_mass=float(state['weights'].sum()))
    torch.save(state,path); write_json(meta,metadata); return state,metadata


def detector_accounting(state,accounting,cfg):
    """Separate legitimate fixed-window clipping from numerical energy error."""
    from .matched_jacobian import integrated_kernel
    from .measurement_bandwidth import gate,lobe
    for v,record in enumerate(accounting):
        if record.get('boundary_checked'): continue
        lost=0.
        for i in range(0,len(state['weights']),65536):
            sl=slice(i,i+65536); p=state['positions'][sl].cuda()
            rel=p.float()-state['center'].cuda().float()
            row=(.5-rel@state['up'][v].cuda().float()/cfg['extent'])*1080-.5
            col=(rel@state['right'][v].cuda().float()/cfg['extent']+.5)*1920-.5
            near=(row<3)|(row>1076)|(col<3)|(col>1916)
            if not near.any(): continue
            _,w,_,_=integrated_kernel(row[near],col[near],cfg['capture'],cfg['resolution'])
            cosine=state['normals'][sl].cuda()[near]@state['directions'][v].cuda()
            e=state['weights'][sl].cuda()[near]*state['transmission'][v,sl].cuda()[near]*gate(cosine,'CURRENT_SOFT_W005')*lobe(cosine,'CURRENT_LOBE')
            rgb=(e[:,None]*state['colors'][sl].cuda()[near]).float()
            lost+=float((rgb.double().sum(1)*(1-w.double().sum(1))).sum())
        expected=record['input_energy']-lost
        record.update(boundary_checked=True,boundary_lost_energy=lost,expected_in_window_energy=expected,
            numerical_energy_relative_error=abs(record['detected_energy']-expected)/max(expected,1e-30))
    assert max(x['numerical_energy_relative_error'] for x in accounting)<1e-5,accounting
    return accounting


def forward(base,reference,centers,radii,coefficients,boundary,cfg,label,*,persist_state=True):
    """Persistent IDs and weights. Regenerate geometry-dependent state, not measure."""
    signature=dict(config_digest=config_digest(cfg),field_digest=field_digest(base),count=len(reference['weights']),seed=cfg['seed'],
        operator_version=OPERATOR_VERSION,shape=[cfg['views'],*cfg['resolution'],3],dtype='float32',
        source_digest=digest(reference['positions']),centers=digest(centers),radii=digest(radii),coefficients=digest(coefficients))
    path=CACHE/(label+'.npy'); spath=CACHE/(label+'_state.pt'); mpath=CACHE/(label+'.json')
    if path.exists() and spath.exists() and mpath.exists():
        meta=json.loads(mpath.read_text()); assert meta['signature']==signature,'stale render cache'
        state=torch.load(spath,weights_only=True,mmap=True)
        detector_accounting(state,meta['energy_accounting'],cfg); write_json(mpath,meta)
        return np.load(path),state,meta
    started=time.perf_counter(); field=WendlandField(base,centers,radii,coefficients)
    state={k:reference[k] for k in ('center','directions','right','up','weights')}; n=len(reference['weights'])
    changed=bool((coefficients!=0).any()); torch.cuda.reset_peak_memory_stats()
    residual=movement=0.; failures=0; normal_min=float('inf')
    if not changed:
        state.update({k:reference[k] for k in ('positions','normals','colors')})
    else:
        state.update({k:torch.empty((n,3),dtype=torch.float64) for k in ('positions','normals','colors')})
        for i in range(0,n,8192):
            sl=slice(i,i+8192); p=reference['positions'][sl].cuda(); nr=reference['normals'][sl].cuda()
            x,normal=attach(field,p,nr)
            f=field.value(x).abs(); failures+=int((~torch.isfinite(x).all(1)|(f>1e-6)).sum())
            residual=max(residual,float(f.max())); movement=max(movement,float((x-p).norm(dim=1).max()))
            normal_min=min(normal_min,float(field.gradient(x).norm(dim=1).min()))
            state['positions'][sl]=x.cpu(); state['normals'][sl]=normal.cpu()
            state['colors'][sl]=meshfree_base_color(x,field.lower,field.upper).cpu()
    source_seconds=time.perf_counter()-started
    if failures: raise RuntimeError(f'ATTACHMENT_FAILURE:{failures}, max residual {residual}')
    if not changed and 'transmission' in reference: state['transmission']=reference['transmission']
    else:
        t=torch.empty((4,n),dtype=torch.float64)
        for view in range(4):
            for i in range(0,n,4096):
                p=state['positions'][i:i+4096].cuda(); d=state['directions'][view].cuda().expand_as(p)
                tr,_,_=transmission_prefixes(field,p,d,boundary.exit_times(p,d),radius=cfg['h'],epsilon=cfg['epsilon'],
                    path_step=cfg['path_step'],offsets=p.new_zeros((1,3)),counts=(1,),eta=cfg['eta'],kappa=cfg['kappa'],launch_exclusion_factor=cfg['launch'])
                t[view,i:i+len(p)]=tr[:,0].cpu()
        state['transmission']=t
    transport_seconds=time.perf_counter()-started-source_seconds
    pre_peak=torch.cuda.max_memory_allocated()/2**20; pre_reserved=torch.cuda.max_memory_reserved()/2**20
    images,occupancy,accounting,timing=render(state,cfg['capture']); del occupancy
    images=np.stack(images); assert images.shape==(4,1080,1920,3) and np.isfinite(images).all()
    meta=dict(signature=signature,source_seconds=source_seconds,transport_seconds=transport_seconds,
        detector_seconds=timing['readout_seconds'],total_seconds=time.perf_counter()-started,
        peak_cuda_allocated_mib=max(pre_peak,timing['cuda_allocated_peak_mib']),peak_cuda_reserved_mib=max(pre_reserved,timing['cuda_reserved_peak_mib']),
        cpu_rss_peak_mib=timing['cpu_rss_peak_mib'],energy_accounting=accounting,root_failures=failures,
        attachment_residual_max=residual,maximum_displacement=movement,image_digest=digest(torch.from_numpy(images)))
    meta['state_persisted']=persist_state
    np.save(path,images)
    if persist_state: torch.save(state,spath)
    write_json(mpath,meta)
    detector_accounting(state,accounting,cfg); write_json(mpath,meta)
    print(f'Forward {label}: {meta["total_seconds"]:.1f}s',flush=True)
    return images,state,meta
