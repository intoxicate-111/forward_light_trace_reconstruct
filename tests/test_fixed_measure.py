import unittest
import torch
from zlt.fixed_measure import OneCoefficientField, attach, diagnostic_ids
from zlt.locality import wendland_values, wendland_gradients


class Plane:
    lower=torch.full((3,),-2.,dtype=torch.float64)
    upper=-lower
    def value(self,p): return p[:,2]
    def gradient(self,p):
        return torch.tensor([0.,0.,1.],dtype=p.dtype,device=p.device).expand_as(p)


class FixedMeasureTests(unittest.TestCase):
    def test_diagnostic_ids_do_not_alias_sobol_digits(self):
        sequence=torch.quasirandom.SobolEngine(3,scramble=True,seed=101).draw(65536,dtype=torch.float64)
        ids=diagnostic_ids(65536,1024)
        self.assertEqual(len(set(ids)),1024)
        self.assertTrue((ids==diagnostic_ids(65536,1024)).all())
        self.assertGreater(float(sequence[ids,0].max()-sequence[ids,0].min()),.99)
        self.assertLess(abs(float(sequence[ids,0].mean())-.5),.04)

    def test_sparse_single_coordinate_matches_dense_diagnostic(self):
        p=torch.tensor([[.1,.2,.05],[2.,1.,0.],[.3,-.1,.2]],dtype=torch.float64)
        radius=torch.tensor(.7,dtype=p.dtype); coefficient=torch.tensor(.01,dtype=p.dtype)
        field=OneCoefficientField(Plane(),torch.zeros(3,dtype=p.dtype),radius,coefficient)
        torch.testing.assert_close(field.value(p),p[:,2]+coefficient*wendland_values(p,radius.expand(3)))
        torch.testing.assert_close(field.gradient(p),Plane().gradient(p)+coefficient*wendland_gradients(p,radius.expand(3)))
        self.assertEqual(field.value(p)[1],0.)

    def test_fixed_normal_identity_attachment_and_derivative(self):
        p=torch.tensor([[.1,.2,0.],[.2,-.1,0.]],dtype=torch.float64)
        n=Plane().gradient(p); center=torch.zeros(3,dtype=p.dtype); r=torch.tensor(.7,dtype=p.dtype)
        def forward(c): return attach(OneCoefficientField(Plane(),center,r,c),p,n)[0]
        c=torch.tensor(0.,dtype=p.dtype,requires_grad=True)
        x,jvp=torch.autograd.functional.jvp(forward,c,torch.ones_like(c))
        torch.testing.assert_close(x,p)
        eps=1e-6; fd=(forward(c+eps)-forward(c-eps))/(2*eps)
        torch.testing.assert_close(jvp,fd,rtol=1e-7,atol=1e-10)
        torch.testing.assert_close(jvp[:,:2],torch.zeros_like(jvp[:,:2]))
        moved=forward(c+.001)
        self.assertLess(float(OneCoefficientField(Plane(),center,r,c+.001).value(moved).abs().max()),1e-12)


if __name__=="__main__": unittest.main()
