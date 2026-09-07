import unittest
import numpy as np
import torch
from zlt.critical_continuation import scalar_control, sphere_torus_control, local_equation
from zlt.critical_experiment import CriticalBase, field, source, local_real_critical_curve, make_cells
from zlt.fields import implicit_position_jacobian, deform_reference_surface
from zlt.locality import BasisLayout, UniformGridIndex
from zlt.birth import _local_sparse_jacobian
from zlt.jacobian import render_fixed_transport_cell


class CriticalContinuationTests(unittest.TestCase):
    def test_scalar_monodromy(self):
        report, result = scalar_control()
        self.assertTrue(report['passed'])
        self.assertLess(report['max_residual'], 1e-12)
        self.assertAlmostEqual(report['max_implicit_derivative'], .5)
        self.assertLess(report['real_path'][-1]['spatial_gradient'], 1e-6)
        self.assertGreater(report['real_path'][-1]['implicit_derivative'], 1e6)

    def test_analytic_branches_and_real_reentry_failure(self):
        report, _ = sphere_torus_control()
        self.assertTrue(report['regular_complex_paths'])
        self.assertFalse(report['all_tracked_branches_return_real'])
        self.assertFalse(report['branches']['pole']['returns_real'])
        self.assertTrue(report['branches']['outer']['returns_real'])

    def test_sphere_and_known_critical_mode(self):
        p = torch.randn(30, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        p /= p.norm(dim=1, keepdim=True)
        self.assertLess(float(CriticalBase(0.).value(p).abs().max()), 1e-14)
        origin = torch.zeros((1, 3), dtype=torch.float64)
        for c in (-.005, 0., .005):
            self.assertAlmostEqual(float(field(c).value(origin)), c)
            self.assertEqual(float(field(c).gradient(origin).norm()), 0.)

    def test_full_complex_equation_gradient(self):
        p = np.array([[.1+.01j, .12-.02j, .2+.01j]])
        c = .003+.002j
        _, gradient, basis = local_equation(p, c)
        for axis in range(3):
            delta = np.zeros_like(p); delta[:, axis] = 1e-6
            fd = (local_equation(p+delta, c)[0]-local_equation(p-delta, c)[0])/2e-6
            np.testing.assert_allclose(fd, gradient[:, axis], rtol=1e-7, atol=1e-9)
        fd = (local_equation(p, c+1e-6)[0]-local_equation(p, c-1e-6)[0])/2e-6
        np.testing.assert_allclose(fd, basis, rtol=1e-8, atol=1e-10)

    def test_real_critical_collapse_not_chart_only(self):
        rows = local_real_critical_curve()
        self.assertLess(rows[-1]['spatial_gradient'], rows[0]['spatial_gradient']*1e-4)
        self.assertGreater(rows[-1]['implicit_derivative'], rows[0]['implicit_derivative']*1e4)
        self.assertLess(max(x['equation_residual'] for x in rows), 1e-14)
        p, n, _ = source(-.005, 8)
        tangent = torch.linalg.cross(n, n.new_tensor([0., 0., 1.]).expand_as(n))
        _, stable, denominator = implicit_position_jacobian(field(-.005), p, tangent)
        self.assertFalse(bool(stable.any()))
        self.assertGreater(float(field(-.005).gradient(p).norm(dim=1).min()), .1)

    def test_existing_sparse_image_chain(self):
        c = -.005; f = field(c); p, n, colors = source(c, 16)
        cell = make_cells(f, p, n, colors)[0]
        layout = BasisLayout(f.centers, f.radii)
        supports = UniformGridIndex(f.centers, .8).query(p, .8)
        denominator = (f.gradient(p)*n).sum(1)
        J = _local_sparse_jacobian(layout, p, n, denominator, supports, cell)
        column = torch.sparse.mm(J, torch.ones((1, 1), dtype=torch.float64)).flatten()
        self.assertGreater(float(column.norm()), 1e-6)
        images = []
        for dc in (1e-5, -1e-5):
            moved, success, _ = deform_reference_surface(field(c+dc), p, n)
            self.assertTrue(bool(success.all()))
            images.append(render_fixed_transport_cell(cell, moved).flatten())
        fd = (images[0]-images[1])/2e-5
        self.assertLess(float((fd-column).norm()/column.norm()), 1e-5)


if __name__ == '__main__': unittest.main()
