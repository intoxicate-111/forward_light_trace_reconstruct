import unittest
import torch
from zlt.jet_experiment import initial_field
from zlt.manifold_jets import JetLayout, ManifoldJetField, tangent_frame
from zlt.basis_renderer import radial_attachment


class ManifoldJetTests(unittest.TestCase):
    def field(self, order=2):
        base = initial_field()
        c = torch.tensor([[1., 0, 0]], dtype=torch.float64)
        layout = JetLayout(c, base.gradient(c), c.new_tensor([.8]), order)
        return ManifoldJetField(base, layout, c.new_full((layout.count,), .01))

    def test_frames(self):
        n = torch.randn(30, 3, dtype=torch.float64)
        f = tangent_frame(n)
        torch.testing.assert_close(f@f.transpose(1, 2), torch.eye(3, dtype=n.dtype).expand(30, 3, 3))
        torch.testing.assert_close(torch.linalg.det(f), torch.ones(30, dtype=n.dtype))
        with self.assertRaises(ValueError): tangent_frame(torch.zeros_like(n))

    def test_orders_and_normalized_modes(self):
        for order, count in ((0, 1), (1, 3), (2, 6)):
            f = self.field(order); frame = f.layout.frames[0]
            p = f.layout.centers+.8*(.2*frame[0]+.3*frame[1])
            _, _, h, _ = f.layout.query(p)
            expected = p.new_tensor([1, .2, .3, .04, .06, .09])[:count]
            torch.testing.assert_close(h/h[0], expected)
        with self.assertRaises(ValueError): self.field(3)

    def test_support_and_disabled(self):
        f = self.field(); p = torch.tensor([[1., 2, 0], [-1., 0, 0], [3., 0, 0]], dtype=torch.float64)
        self.assertEqual(len(f.layout.query(p)[0]), 0)
        torch.testing.assert_close(f.evaluate(p), f.reference.evaluate(p), rtol=0, atol=0)
        f.enabled = False
        p = torch.randn(50, 3, dtype=p.dtype)
        torch.testing.assert_close(f.evaluate(p), f.reference.evaluate(p), rtol=0, atol=0)

    def test_spatial_gradient_and_hessian(self):
        f = self.field()
        p = torch.tensor([[1.03, .12, .18], [1.5, .1, .2], [.8, -.15, .11]], dtype=torch.float64)
        g, H = f.jet(p)
        self.assertTrue(bool(torch.isfinite(H).all()))
        torch.testing.assert_close(H, H.transpose(1, 2), rtol=1e-9, atol=1e-9)
        for a in range(3):
            step = torch.zeros_like(p); step[:, a] = 1e-6
            torch.testing.assert_close(g[:, a], (f.value(p+step)-f.value(p-step))/2e-6, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(H[:, :, a], (f.gradient(p+step)-f.gradient(p-step))/2e-6, rtol=1e-6, atol=1e-8)

    def test_boundary_c2(self):
        f = self.field()
        for axis in (0, 1):
            # Normal collar boundary and tangent disk boundary.
            d = torch.zeros((1, 3), dtype=torch.float64); d[:, axis] = 1
            errors = []
            for eps in (1e-2, 1e-3, 1e-4):
                p = f.layout.centers+.8*(1-eps)*d
                g, H = f.jet(p); g0, H0 = f.reference.jet(p)
                errors.append(float((H-H0).norm()))
            self.assertLess(errors[-1], errors[0]*.02)

    def test_displacement_sign_and_scale(self):
        f = self.field(0); c = f.layout.centers
        self.assertLess(float(f.value(c)), 0.)
        for eps in (1e-3, 1e-4):
            moved = f.with_coefficients(c.new_tensor([eps]))
            x, _ = radial_attachment(moved, c)
            self.assertGreater(float(x[0, 0]), 1.)
            self.assertLess(abs(float((x[0, 0]-1)/eps)-1), .01)
            torch.testing.assert_close(moved.value(c)-f.reference.value(c), -eps*f.reference.gradient(c).norm(dim=1))

    def test_chart_origin_hessian(self):
        f = self.field(0)
        p = f.layout.centers
        g, H = f.jet(p)
        self.assertTrue(bool(torch.isfinite(H).all()))
        for a in range(3):
            step = torch.zeros_like(p); step[:, a] = 1e-6
            torch.testing.assert_close(H[:, :, a], (f.gradient(p+step)-f.gradient(p-step))/2e-6,
                                       rtol=2e-5, atol=1e-7)

    def test_deformed_reference_preserves_spatial_chain(self):
        reference = self.field()
        f = ManifoldJetField(reference, reference.layout, reference.coefficients*.3)
        p = torch.tensor([[1.01, .14, .23]], dtype=torch.float64)
        _, H = f.jet(p)
        for a in range(3):
            step = torch.zeros_like(p); step[:, a] = 1e-6
            torch.testing.assert_close(H[:, :, a], (f.gradient(p+step)-f.gradient(p-step))/2e-6,
                                       rtol=1e-5, atol=1e-7)


if __name__ == '__main__': unittest.main()
