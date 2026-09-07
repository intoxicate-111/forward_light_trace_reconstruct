import io
import unittest
import torch
from zlt.basis_only import BasisOnlyZeroSetField
from zlt.basis_initialization import initialize,directions,radial_roots,design
from zlt.locality import BasisLayout,wendland_values,wendland_gradients


class BasisOnlyTests(unittest.TestCase):
    def simple(self):
        centers=torch.tensor([[0.,0.,0.],[.3,.2,.1]],dtype=torch.float64)
        return BasisOnlyZeroSetField(BasisLayout(centers,torch.full((2,),.6,dtype=torch.float64)),torch.tensor([-2.,.2],dtype=torch.float64))

    def test_formula_has_no_base(self):
        f=self.simple(); p=torch.rand((40,3),generator=torch.Generator().manual_seed(71),dtype=torch.float64)-.5
        expected=1+design(p,f.layout)@f.coefficients
        torch.testing.assert_close(f.value(p),expected,atol=1e-14,rtol=1e-14)
        self.assertFalse(any(hasattr(f,k) for k in ('base','grid','field','lookup')))

    def test_outside_exact_background(self):
        f=self.simple(); p=torch.rand((40,3),dtype=torch.float64)+3
        self.assertTrue(torch.equal(f.value(p),torch.ones(40,dtype=p.dtype)))
        self.assertTrue(torch.equal(f.gradient(p),torch.zeros_like(p)))

    def test_zero_coefficients_and_zero_derivative_cell(self):
        f=self.simple().with_coefficients(torch.zeros(2,dtype=torch.float64,requires_grad=True))
        p=torch.tensor([[.1,.1,.1]],dtype=torch.float64)
        self.assertEqual(float(f.value(p)),1.)
        self.assertTrue(torch.equal(f.gradient(p),torch.zeros_like(p)))
        derivative=torch.autograd.grad(f.value(p).sum(),f.coefficients)[0]
        self.assertGreater(float(derivative.norm()),0.)

    def test_gradient_fd_and_autograd(self):
        f=self.simple(); p=torch.tensor([[.1,.12,-.1],[.59999,0,0],[.60001,0,0],[2.,0,0]],dtype=torch.float64,requires_grad=True)
        analytic=f.gradient(p); ad=torch.autograd.grad(f.value(p).sum(),p)[0]
        torch.testing.assert_close(analytic,ad,atol=1e-12,rtol=1e-12)
        for axis in range(3):
            h=torch.zeros_like(p); h[:,axis]=1e-6
            fd=(f.value(p+h)-f.value(p-h))/(2e-6)
            torch.testing.assert_close(analytic[:,axis],fd,atol=1e-8,rtol=1e-5)

    def test_serialization(self):
        f=self.simple(); buffer=io.BytesIO(); torch.save(f.state_dict(),buffer); buffer.seek(0)
        g=BasisOnlyZeroSetField.from_state_dict(torch.load(buffer,weights_only=True))
        p=torch.rand((30,3),dtype=torch.float64)
        torch.testing.assert_close(f.evaluate(p),g.evaluate(p),rtol=0,atol=0)

    def test_single_scale_and_fixed_background(self):
        f=self.simple()
        with self.assertRaises(ValueError): BasisOnlyZeroSetField(f.layout,f.coefficients,0.)
        with self.assertRaises(ValueError): BasisOnlyZeroSetField(BasisLayout(f.layout.centers,torch.tensor([.5,.6])),f.coefficients)

    def test_deterministic_initialization(self):
        # Small matrix only checks determinism, not sphere approximation capacity.
        a,_,_=initialize(16,.6); b,_,_=initialize(16,.6)
        torch.testing.assert_close(a.layout.centers,b.layout.centers,atol=0,rtol=0)
        torch.testing.assert_close(a.coefficients,b.coefficients,atol=0,rtol=0)

    def test_coefficients_move_actual_roots(self):
        # One radial compact basis is a valid exact spherical zero set.
        layout=BasisLayout(torch.zeros((1,3),dtype=torch.float64),torch.tensor([2.],dtype=torch.float64))
        b=wendland_values(torch.tensor([[1.,0,0]],dtype=torch.float64),layout.radii)[0]
        f=BasisOnlyZeroSetField(layout,(-1/b).reshape(1)); d=directions(32)
        t,valid,_=radial_roots(f,d,scan_steps=81)
        self.assertTrue(bool(valid.all())); self.assertLess(float((t-1).abs().max()),1e-9)
        self.assertLess(float(f.value(torch.zeros(1,3,dtype=torch.float64))),0.)
        self.assertGreater(float(f.value(torch.tensor([[2.,0,0]],dtype=torch.float64))),0.)
        moved,_,_=radial_roots(f.with_coefficients(f.coefficients*1.1),d,scan_steps=82)
        self.assertGreater(float((moved-t).min()),0.)

    def test_safeguarded_radial_attachment(self):
        from zlt.basis_renderer import radial_attachment
        layout=BasisLayout(torch.zeros((1,3),dtype=torch.float64),torch.tensor([2.],dtype=torch.float64))
        f=BasisOnlyZeroSetField(layout,torch.tensor([-6.],dtype=torch.float64)); d=directions(32)
        x,n=radial_attachment(f,d)
        self.assertLess(float(f.value(x).abs().max()),1e-10)
        torch.testing.assert_close(n,d,atol=1e-10,rtol=0)
        with self.assertRaises(RuntimeError): radial_attachment(f.with_coefficients(torch.zeros(1,dtype=torch.float64)),d)


if __name__=='__main__': unittest.main()
