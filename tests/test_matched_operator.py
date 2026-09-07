import unittest
import numpy as np
import torch
from zlt.matched_operator import SphereAdapter,WendlandField,basis_hessian,grid_jet,implicit_area_points
from zlt.matched_jacobian import CandidateSupports,source_tangent,optical_tangent,integrated_kernel
from zlt.locality import wendland_gradients
from zlt.mesh_field import _grid_field
from zlt.density_matrix import capture_splat
from zlt.boundary_transport import ObservationSphere
from zlt.transverse_packet import transmission_prefixes
from zlt.fixed_measure import attach


class MatchedOperatorTests(unittest.TestCase):
    def test_support_aabb_cull_matches_unculled_sparse_query(self):
        base=SphereAdapter(torch.tensor(-1.2,dtype=torch.float64),torch.tensor(1.2,dtype=torch.float64))
        centers=torch.tensor([[0.,0.,1.],[.3,0.,.9],[0.,0.,-1.]],dtype=torch.float64)
        radii=centers.new_tensor([.45,.3,.2]); coefficients=centers.new_tensor([.01,-.004,.003])
        field=WendlandField(base,centers,radii,coefficients)
        p=torch.rand((4096,3),generator=torch.Generator().manual_seed(19),dtype=torch.float64)*6-3
        pi,bi=field.pairs(p); raw=field.index.query(p,.45)
        keep=(p[raw.point_ids]-centers[raw.basis_ids]).square().sum(1)<radii[raw.basis_ids].square()
        expected=torch.sort(raw.point_ids[keep]*3+raw.basis_ids[keep]).values
        torch.testing.assert_close(torch.sort(pi*3+bi).values,expected,atol=0,rtol=0)

    def test_zero_birth_does_not_change_field_or_source(self):
        base=SphereAdapter(torch.tensor(-1.2,dtype=torch.float64),torch.tensor(1.2,dtype=torch.float64))
        p=torch.tensor([[0.,0.,1.],[.8,0.,.6]],dtype=torch.float64)
        centers=p[:1].clone(); radius=p.new_tensor([.45]); coefficient=p.new_zeros(1)
        before=WendlandField(base); after=WendlandField(base,centers,radius,coefficient)
        torch.testing.assert_close(before.evaluate(p),after.evaluate(p),atol=0,rtol=0)
        x,n=attach(after,p,p)
        torch.testing.assert_close(x,p,atol=1e-14,rtol=0)
        torch.testing.assert_close(n,p,atol=1e-14,rtol=0)

    def test_detector_clipping_is_not_renormalized(self):
        row=torch.tensor([16.,-.5,-10.],dtype=torch.float64); col=torch.full_like(row,16.)
        ids,w,_,_=integrated_kernel(row,col,4,(32,32))
        torch.testing.assert_close(w.sum(1),row.new_tensor([1.,.5,0.]),atol=1e-14,rtol=0)
        energy=row.new_tensor([[.2,.3,.4]]).expand(3,-1)
        acc=row.new_zeros((32*32,3)); capture_splat(acc,row,col,energy,4,resolution=(32,32))
        torch.testing.assert_close(acc.sum(),(energy.sum(1)*w.sum(1)).sum(),atol=1e-14,rtol=0)

    def test_wendland_hessian(self):
        p=torch.tensor([[.1,.2,.05],[0.,0.,0.],[1.,2.,3.]],dtype=torch.float64)
        r=torch.full((3,),.6,dtype=p.dtype); h=basis_hessian(p,r)
        for axis in range(3):
            step=torch.zeros_like(p); step[:,axis]=1e-6
            fd=(wendland_gradients(p+step,r)-wendland_gradients(p-step,r))/(2e-6)
            torch.testing.assert_close(h[:,:,axis],fd,atol=1e-3,rtol=2e-5)

    def test_grid_jet_matches_actual_interpolant_ad(self):
        rng=np.random.default_rng(90); base,_=_grid_field(rng.normal(size=(6,6,6)).astype(np.float32),-1.,1.,build_surface_scaffold=False)
        p=torch.tensor([[.15,.27,-.12],[-1.1,.7,.7],[.5,1.2,.4]],dtype=torch.float64,requires_grad=True)
        df,hg=grid_jet(base,p.detach())
        expected=torch.autograd.grad(base.value(p).sum(),p)[0]
        torch.testing.assert_close(df,expected,atol=1e-12,rtol=1e-12)
        for a in range(3):
            row=torch.autograd.grad(base.gradient(p)[:,a].sum(),p)[0]
            torch.testing.assert_close(hg[:,a],row,atol=1e-12,rtol=1e-12)

    def test_integrated_detector_is_same_operator(self):
        row=torch.tensor([10.01,11.5,12.98,-.1],dtype=torch.float64); col=row+2
        energy=torch.tensor([[.2,.4,.6]]*4,dtype=torch.float64)
        acc=torch.zeros(32*32,3,dtype=torch.float64); capture_splat(acc,row,col,energy,4,resolution=(32,32))
        ids,w,dy,dx=integrated_kernel(row,col,4,(32,32)); other=torch.zeros_like(acc)
        other.index_add_(0,ids.flatten(),(w[:,:,None]*energy[:,None]).reshape(-1,3))
        torch.testing.assert_close(other,acc,atol=1e-14,rtol=1e-14)
        eps=1e-6
        # Compare image-valued derivatives; integer tap identities may differ.
        plus=torch.zeros_like(acc); minus=torch.zeros_like(acc)
        capture_splat(plus,row+eps,col,energy,4,resolution=(32,32)); capture_splat(minus,row-eps,col,energy,4,resolution=(32,32))
        analytic=torch.zeros_like(acc); analytic.index_add_(0,ids.flatten(),(dy[:,:,None]*energy[:,None]).reshape(-1,3))
        torch.testing.assert_close(analytic,(plus-minus)/(2*eps),atol=1e-9,rtol=1e-7)

    def test_sphere_attachment_and_optical_tangent(self):
        base=SphereAdapter(torch.tensor(-1.2,dtype=torch.float64),torch.tensor(1.2,dtype=torch.float64))
        p=torch.tensor([[0.,0.,1.],[.8,0.,.6]],dtype=torch.float64); normal=p.clone()
        centers=torch.tensor([[0.,0.,1.],[0.,0.,-1.]],dtype=torch.float64); radii=torch.full((2,),.6,dtype=p.dtype)
        support=CandidateSupports(centers,radii); field=WendlandField(base)
        dx,dn=source_tangent(field,p,normal,support)
        direction=torch.tensor([[0.,0.,-1.]],dtype=p.dtype).expand_as(p); boundary=ObservationSphere(p.new_zeros(3),1.5)
        cfg=dict(h=.08,epsilon=.08,path_step=.04,eta=1e-6,kappa=-np.log(.01),launch=1.05)
        tau,dtau=optical_tangent(field,p,direction,boundary.exit_times(p,direction),dx,support,cfg)
        for k in range(2):
            answers=[]; xs=[]; ns=[]
            for sign in (-1,1):
                c=p.new_zeros(2); c[k]=sign*1e-6; f=WendlandField(base,centers,radii,c)
                x,n=attach(f,p,normal); xs.append(x); ns.append(n)
                _,t,_=transmission_prefixes(f,x,direction,boundary.exit_times(x,direction),radius=.08,epsilon=.08,path_step=.04,offsets=p.new_zeros((1,3)),counts=(1,),eta=1e-6,kappa=cfg['kappa'])
                answers.append(t[:,0])
            torch.testing.assert_close(dx[:,k],(xs[1]-xs[0])/(2e-6),atol=1e-8,rtol=1e-6)
            torch.testing.assert_close(dn[:,k],(ns[1]-ns[0])/(2e-6),atol=1e-8,rtol=1e-6)
            torch.testing.assert_close(dtau[:,k],(answers[1]-answers[0])/(2e-6),atol=1e-6,rtol=1e-5)

    def test_active_and_unborn_response_after_deformation(self):
        base=SphereAdapter(torch.tensor(-1.2,dtype=torch.float64),torch.tensor(1.2,dtype=torch.float64))
        ref=torch.tensor([[0.,0.,1.],[.2,0.,np.sqrt(.96)]],dtype=torch.float64)
        centers=torch.tensor([[0.,0.,1.],[.15,0.,np.sqrt(1-.15**2)],[0.,0.,-1.]],dtype=ref.dtype)
        radii=ref.new_full((3,),.6); coeff=ref.new_tensor([.01,0.,-.005])
        field=WendlandField(base,centers,radii,coeff); x,n=attach(field,ref,ref)
        support=CandidateSupports(centers,radii); dx,dn=source_tangent(field,x,ref,support)
        direction=ref.new_tensor([[0.,0.,-1.]]).expand_as(ref); boundary=ObservationSphere(ref.new_zeros(3),1.5)
        cfg=dict(h=.08,epsilon=.08,path_step=.04,eta=1e-6,kappa=-np.log(.01),launch=1.05)
        _,dtau=optical_tangent(field,x,direction,boundary.exit_times(x,direction),dx,support,cfg)
        for k in range(3):
            xs=[]; ns=[]; taus=[]
            for sign in (-1,1):
                c=coeff.clone(); c[k]+=sign*1e-6; perturbed=WendlandField(base,centers,radii,c)
                xp,np_=attach(perturbed,ref,ref); xs.append(xp); ns.append(np_)
                _,t,_=transmission_prefixes(perturbed,xp,direction,boundary.exit_times(xp,direction),radius=.08,epsilon=.08,path_step=.04,offsets=xp.new_zeros((1,3)),counts=(1,),eta=1e-6,kappa=cfg['kappa'])
                taus.append(t[:,0])
            torch.testing.assert_close(dx[:,k],(xs[1]-xs[0])/(2e-6),atol=1e-8,rtol=1e-6)
            torch.testing.assert_close(dn[:,k],(ns[1]-ns[0])/(2e-6),atol=1e-8,rtol=1e-6)
            torch.testing.assert_close(dtau[:,k],(taus[1]-taus[0])/(2e-6),atol=1e-6,rtol=1e-5)

    def test_implicit_area_sampler_planar_uniformity(self):
        axis=np.linspace(-1,1,8); grid=np.broadcast_to(axis[:,None,None],(8,8,8)).copy().astype(np.float32)
        base,_=_grid_field(grid,-1.,1.,build_surface_scaffold=False)
        points=torch.cat([p for p,_,_ in implicit_area_points(base,4096,block=8192)])
        self.assertLess(float(base.value(points).abs().max()),1e-12)
        self.assertLess(float(points[:,1:].mean(0).abs().max()),.03)
        self.assertLess(float((points[:,1:].square().mean(0)-1/3).abs().max()),.03)


if __name__=='__main__': unittest.main()
