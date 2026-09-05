import unittest
import torch
from zlt.measurement_bandwidth import KERNELS, gate, lobe, kernel_pairs, splat, impulse_diagnostic
from zlt.finite_packet import _accumulate_continuous


class MeasurementTests(unittest.TestCase):
    def test_gates_and_lobes_are_separate(self):
        c=torch.tensor([-.1,0.,.005,.05,1.],dtype=torch.float64)
        self.assertTrue(torch.equal(gate(c,"HARD_OUTWARD"),torch.tensor([0.,0.,1.,1.,1.],dtype=c.dtype)))
        self.assertTrue(torch.equal(gate(c,"NO_OUTWARD_GATE"),torch.ones_like(c)))
        self.assertAlmostEqual(float(gate(c,"CURRENT_SOFT_W005")[3]),1-torch.exp(torch.tensor(-1.)).item(),places=7)
        self.assertTrue(torch.equal(lobe(c,"CONSTANT_RADIANCE"),torch.ones_like(c)))
        self.assertAlmostEqual(float(lobe(c,"CURRENT_LOBE")[0]),.35)

    def test_partition_unity_and_boundary_loss(self):
        generator=torch.Generator().manual_seed(31)
        row=3+torch.rand(500,generator=generator,dtype=torch.float64)*10
        col=3+torch.rand(500,generator=generator,dtype=torch.float64)*10
        for kernel in KERNELS:
            _,weights=kernel_pairs(row,col,kernel,(20,20))
            torch.testing.assert_close(weights.sum(1),torch.ones_like(row),atol=1e-12,rtol=0)
            _,edge=kernel_pairs(row*0-.49,col*0-.49,kernel,(20,20))
            self.assertTrue(bool((edge.sum(1)<=1+1e-12).all()))

    def test_cubic_matches_original(self):
        g=torch.Generator().manual_seed(14)
        p=torch.randn(50,3,generator=g,dtype=torch.float64)*.5
        energy=torch.rand(50,3,generator=g,dtype=torch.float64)
        right=torch.tensor([1.,0.,0.],dtype=p.dtype); up=torch.tensor([0.,1.,0.],dtype=p.dtype)
        center=torch.zeros(3,dtype=p.dtype)
        a=torch.zeros(32*32,3,dtype=p.dtype); b=torch.zeros_like(a)
        _accumulate_continuous(a,p,energy,right,up,center,2.8,(32,32))
        splat(b,p,energy,right,up,center,"CUBIC_4X4",(32,32))
        torch.testing.assert_close(a,b,atol=1e-14,rtol=0)

    def test_detector_nyquist_attenuation(self):
        rows,_=impulse_diagnostic()
        self.assertGreater(rows[0]["mtf_x"]["mean_magnitude"][-1],.99)
        self.assertLess(rows[2]["mtf_x"]["mean_magnitude"][-1],.5)
        self.assertLess(max(r["partition_unity_max_error"] for r in rows),1e-12)


if __name__=="__main__": unittest.main()
