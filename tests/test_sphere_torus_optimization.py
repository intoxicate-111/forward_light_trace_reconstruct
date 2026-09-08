import unittest
import numpy as np
import torch
from zlt.sphere_torus_optimization import C0, TARGET_C, _complex_bridge, _critical_indicators
from zlt.critical_experiment import field, source


class SphereTorusOptimizationTests(unittest.TestCase):
    def test_known_critical_coordinate_and_collapse(self):
        origin = torch.zeros((1, 3), dtype=torch.float64)
        for c in (C0, 0., TARGET_C):
            self.assertAlmostEqual(float(field(c).value(origin)), c)
            self.assertEqual(float(field(c).gradient(origin).norm()), 0.)
        far = _critical_indicators(-1e-3)
        near = _critical_indicators(-1e-9)
        self.assertLess(near['critical_branch_gradient_norm'], far['critical_branch_gradient_norm']*.01)
        self.assertGreater(near['implicit_dc'], far['implicit_dc']*100)

    def test_complex_bridge_returns_to_independent_real_solve(self):
        p, n, _ = source(C0, 12)
        result, raw = _complex_bridge(C0, 3e-4, p, n)
        self.assertEqual(result['rendered_complex_states'], 0)
        self.assertLess(result['max_residual'], 1e-12)
        self.assertGreater(result['min_spatial_gradient_norm'], 0.)
        self.assertGreater(result['min_distance_to_discriminant'], 0.)
        self.assertLess(result['endpoint_point_imaginary_max'], 1e-12)
        self.assertLess(result['endpoint_real_resolve_max_error'], 1e-12)
        self.assertTrue(np.isfinite(raw['roots']).all())


if __name__ == '__main__': unittest.main()
