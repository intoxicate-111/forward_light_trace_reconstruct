import unittest
import numpy as np
import torch
from types import SimpleNamespace
from zlt.polar_aliasing import SCHEMES, phase_coordinates, artifact_metrics


class PolarAliasingTests(unittest.TestCase):
    def test_endpoint_factorization_preserves_fixed_grid(self):
        from zlt.stratified_polar import _shared_ray_map_from_prepared
        class Plane:
            def value(self,x): return x[:,2]
            def gradient(self,x):
                result=torch.zeros_like(x); result[:,2]=1
                return result
        prepared=SimpleNamespace(positions=torch.zeros((1,3),dtype=torch.float64),
            normals=torch.tensor([[0.,0.,1.]],dtype=torch.float64),
            tangent_1=torch.tensor([[1.,0.,0.]],dtype=torch.float64),
            tangent_2=torch.tensor([[0.,1.,0.]],dtype=torch.float64))
        template=SimpleNamespace(chart_radii=torch.tensor([.3],dtype=torch.float64),eta=1e-6)
        theta,rho=map(torch.from_numpy,phase_coordinates("CURRENT",charts=1,angles=4,rings=4))
        ids=torch.tensor([0])
        shared=_shared_ray_map_from_prepared(Plane(),template,prepared,ids,theta[:,:,0],rho,32)
        independent=_shared_ray_map_from_prepared(Plane(),template,prepared,ids,theta.reshape(1,16),rho.reshape(1,16,1),32)
        for a,b in zip(shared,independent): torch.testing.assert_close(a.flatten(),b.flatten(),atol=0,rtol=0)

    def test_matched_strata_and_phase_factorization(self):
        for scheme in SCHEMES:
            theta,rho=phase_coordinates(scheme,charts=9)
            self.assertEqual(theta.shape,(9,32,32))
            np.testing.assert_allclose(rho,phase_coordinates("CURRENT",charts=9)[1],atol=0,rtol=0)
            np.testing.assert_allclose(np.diff(theta,axis=1),2*np.pi/32,atol=2e-15)
            shared=np.all(theta==theta[:,:,:1])
            self.assertEqual(shared,scheme in SCHEMES[:2])

    def test_gradient_alignment_rejects_unmatched_frequency(self):
        x=np.arange(64)[None,:]; y=np.arange(64)[:,None]
        a=np.broadcast_to(np.sin(x), (64,64))
        b=np.broadcast_to(np.sin(y), (64,64))
        rgb=lambda z:np.repeat(z[:,:,None],3,axis=2)
        same=artifact_metrics([rgb(a)],[rgb(a)])
        wrong=artifact_metrics([rgb(b)],[rgb(a)])
        self.assertAlmostEqual(same["gradient_cosine"],1)
        self.assertLess(abs(wrong["gradient_cosine"]),1e-12)
        self.assertGreater(wrong["unmatched_gradient_energy"],same["unmatched_gradient_energy"])


if __name__=="__main__": unittest.main()
