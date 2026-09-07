import unittest
import numpy as np
import torch
from zlt.collision_transfer import LAWS, transfer, from_micro, edge_profiles
from zlt.finite_packet import _transmission, _micro_offsets
from zlt.transverse_packet import ToyField


class CollisionTransferTests(unittest.TestCase):
    def test_edge_profile_on_known_smooth_step(self):
        axis=np.arange(64,dtype=float)-32
        image=np.broadcast_to((1/(1+np.exp(-axis/1.2)))[None,:,None],(64,64,3)).copy()
        result=edge_profiles([image],[dict(region='test',row=32,column=32,normal=[0.,1.])])[0]
        self.assertAlmostEqual(result['location_50'],0.,places=3)
        self.assertLess(abs(result['width_10_90']-2*np.log(9)*1.2),.25)

    def test_multiple_micro_current_is_not_one_minus_mean_opacity(self):
        tau=torch.tensor([[0.,4.]],dtype=torch.float64)
        self.assertAlmostEqual(float(from_micro(tau,LAWS[0])),float(torch.exp(-tau.mean())))
        self.assertGreater(abs(float(from_micro(tau,LAWS[0])-(torch.exp(-tau)).mean())),.1)

    def test_transfer_derivative_closes(self):
        for law in LAWS:
            p=torch.tensor([.01,.05,.1,.15,.2],dtype=torch.float64,requires_grad=True)
            d=torch.autograd.grad(transfer(p,law).sum(),p)[0]
            eps=1e-7; fd=(transfer(p.detach()+eps,law)-transfer(p.detach()-eps,law))/(2*eps)
            torch.testing.assert_close(d,fd,atol=1e-8,rtol=1e-6)

    def test_curves(self):
        for law in LAWS[1:]:
            p=torch.linspace(0,1,10001,dtype=torch.float64,requires_grad=True)
            t=transfer(p,law); d=torch.autograd.grad(t.sum(),p)[0]
            self.assertTrue(torch.isfinite(t).all() and torch.isfinite(d).all())
            self.assertAlmostEqual(float(t[0]),1.,places=14)
            self.assertTrue(bool((t[1:]<=t[:-1]).all()))
            self.assertLess(float(d.abs().max()),100.)
        for law in LAWS[2:]:
            self.assertGreaterEqual(float(transfer(torch.tensor(.02,dtype=torch.float64),law)),.95)
            self.assertLessEqual(float(transfer(torch.tensor(.2,dtype=torch.float64),law)),.01)

    def test_micro_equivalence_and_default_unchanged(self):
        x=torch.tensor([[.2,0.,-1.],[.4,0.,-1.]],dtype=torch.float64)
        d=torch.tensor([[0.,0.,1.]],dtype=torch.float64).expand_as(x)
        for m in (1,8):
            kw=dict(radius=.1,epsilon=.04,path_step=.02,offsets=_micro_offsets(m,x.device,x.dtype),eta=1e-6,kappa=4.605170185988092,surface_barrier=True,launch_exclusion_factor=1.05)
            old=_transmission(ToyField('filament',.3),x,d,x.new_full((2,),2.),**kw)
            new=_transmission(ToyField('filament',.3),x,d,x.new_full((2,),2.),return_micro=True,**kw)
            self.assertTrue(torch.equal(old[0],new[0]) and torch.equal(old[1],new[1]))
            torch.testing.assert_close(new[1],new[3].mean(-1),atol=1e-14,rtol=1e-14)
            self.assertTrue(torch.equal(from_micro(new[3],LAWS[0],new[1]),old[0]))


if __name__=='__main__': unittest.main()
