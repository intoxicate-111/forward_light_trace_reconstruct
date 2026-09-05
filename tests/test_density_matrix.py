import unittest
import numpy as np
import torch
from zlt.density_matrix import capture_offsets,capture_splat,CAPTURE,streamed_chart_spacing,classify_fidelity
from zlt.measurement_bandwidth import kernel_pairs
from zlt.polar_aliasing import phase_coordinates


class DensityTests(unittest.TestCase):
    def test_negative_cosine_is_not_improved_by_magnitude(self):
        report={"cells":[{"E":e,"C":c,"whole_image_mse":10-.01*c,
            "gradient_squared_error":10-.01*c,"unmatched_gradient_energy":100/c,
            "gradient_cosine":-.001*c} for e in range(1,4) for c in range(1,4)]}
        improved,evidence=classify_fidelity(report)
        self.assertFalse(improved["C"])
        self.assertLess(evidence["C"]["gradient_cosine_delta"],0)

    def test_streamed_chart_spacing(self):
        centers=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.],[4.,0.,0.]],dtype=torch.float64)
        normals=torch.tensor([[0.,0.,1.]],dtype=torch.float64).expand(5,3)
        torch.testing.assert_close(streamed_chart_spacing(centers,normals),torch.tensor([4.,3.,2.,3.,4.],dtype=torch.float64),atol=0,rtol=0)

    def test_midpoint_quadrature(self):
        self.assertEqual(capture_offsets(1).tolist(),[0.])
        for c in CAPTURE:
            self.assertAlmostEqual(capture_offsets(c).mean(),0.)
            self.assertTrue(np.all(np.abs(capture_offsets(c))<.5))

    def test_energy_and_baseline(self):
        rng=torch.Generator().manual_seed(101)
        row=4+20*torch.rand(100,generator=rng,dtype=torch.float64)
        col=4+20*torch.rand(100,generator=rng,dtype=torch.float64)
        energy=torch.rand((100,3),generator=rng,dtype=torch.float64)
        for c in CAPTURE:
            acc=torch.zeros((32*32,3),dtype=torch.float64)
            capture_splat(acc,row,col,energy,c,(32,32))
            torch.testing.assert_close(acc.sum(0),energy.sum(0),atol=1e-12,rtol=0)
            if c==1:
                ids,weights=kernel_pairs(row,col,"CUBIC_4X4",(32,32))
                expected=torch.zeros_like(acc)
                expected.index_add_(0,ids.flatten(),(weights[...,None]*energy[:,None]).reshape(-1,3))
                torch.testing.assert_close(acc,expected,atol=0,rtol=0)

    def test_density_axes_preserve_strata(self):
        for n in (32,48,64):
            theta,rho=phase_coordinates("PER_CHART_PLUS_RING",charts=3,angles=n,rings=n)
            self.assertEqual(theta.shape,(3,n,n))
            np.testing.assert_allclose(np.diff(theta,axis=1),2*np.pi/n,atol=2e-15)
            np.testing.assert_equal(np.floor(rho*rho*n),np.broadcast_to(np.arange(n),rho.shape))
            self.assertAlmostEqual((np.ones((3,n,n))/(3*n*n)).sum(),1.)


if __name__=="__main__": unittest.main()
