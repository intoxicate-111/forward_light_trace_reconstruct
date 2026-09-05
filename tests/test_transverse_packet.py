"""Small CPU checks; scientific CUDA sweeps live in zlt.transverse_packet."""
import math
import unittest

import torch

from zlt.finite_packet import _micro_offsets, _transmission
from zlt.mesh_field import GridZeroSetField
from zlt.transverse_packet import COUNTS, FusedGridLookup, ToyField, convergence, transmission_prefixes


class TransversePacketTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.offsets = _micro_offsets(128, self.device, self.dtype)
        self.x = torch.tensor([[0., 0., -2.], [.04, .1, -1.8]], dtype=self.dtype)
        self.w = torch.tensor([[0., 0., 1.]], dtype=self.dtype).expand_as(self.x)
        self.end = torch.tensor([4., 3.8], dtype=self.dtype)
        self.kwargs = dict(radius=1., epsilon=.07, path_step=.04, eta=1e-6,
                           kappa=-math.log(.01), launch_exclusion_factor=0.)

    def test_historical_center_and_nested_prefix(self):
        self.assertEqual(int(torch.count_nonzero(_micro_offsets(1, self.device, self.dtype))), 0)
        for m in (4, 8, 16, 32, 64):
            # Independent CPU transcendental kernels can differ by one ulp;
            # production always slices the one common offset tensor.
            torch.testing.assert_close(self.offsets[:m], _micro_offsets(m, self.device, self.dtype),
                                       atol=1e-15, rtol=1e-15)
        self.assertLessEqual(float(self.offsets.norm(dim=1).max()), 1.)

    def test_original_operator_and_streaming_agree(self):
        for kind in ("slab", "filament", "grazing", "two_surfaces"):
            field = ToyField(kind, .5)
            result, tau, _ = transmission_prefixes(field, self.x, self.w, self.end,
                offsets=self.offsets, counts=(1, 8, 32, 128), path_chunk=7,
                micro_chunk=5, **self.kwargs)
            for i, m in enumerate((1, 8, 32, 128)):
                t0, tau0, _ = _transmission(field, self.x, self.w, self.end,
                    offsets=_micro_offsets(m, self.device, self.dtype),
                    surface_barrier=True, **self.kwargs)
                torch.testing.assert_close(result[:, i], t0, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(tau[:, i], tau0, atol=1e-12, rtol=1e-12)

    def test_chunk_invariance_and_zero_radius(self):
        kw = {**self.kwargs, "radius": 0.}
        results = []
        for pc, mc in ((3, 1), (19, 7)):
            _, tau, _ = transmission_prefixes(ToyField("grazing", .1), self.x, self.w,
                self.end, offsets=self.offsets, counts=(1, 4, 8, 32),
                path_chunk=pc, micro_chunk=mc, **kw)
            torch.testing.assert_close(tau, tau[:, :1].expand_as(tau), atol=1e-12, rtol=1e-12)
            results.append(tau)
        torch.testing.assert_close(*results, atol=1e-12, rtol=1e-12)

    def test_invalid_chunk_rejected(self):
        with self.assertRaises(ValueError):
            transmission_prefixes(ToyField("slab", .5), self.x, self.w, self.end,
                offsets=self.offsets, path_chunk=0, **self.kwargs)

    def test_fixed_launch_keeps_centerline_independent_of_radius(self):
        values = []
        for radius in (0., .25, 1., 2.):
            _, tau, _ = transmission_prefixes(ToyField("grazing", .1), self.x, self.w,
                self.end, offsets=self.offsets, counts=(1,), launch_start=.2,
                **{**self.kwargs, "radius":radius})
            values.append(tau)
        for tau in values[1:]:
            torch.testing.assert_close(tau, values[0], atol=1e-12, rtol=1e-12)

    def test_mean_preserves_crossing_calibration(self):
        class Plane:
            def value(self, p): return p[:, 2]
            def gradient(self, p): return p.new_tensor([0., 0., 1.]).expand_as(p)
        t, _, _ = transmission_prefixes(Plane(), self.x, self.w, self.end,
            offsets=self.offsets, counts=(1, 4, 8, 32, 128),
            **{**self.kwargs, "path_step": .005})
        torch.testing.assert_close(t, torch.full_like(t, .01), atol=2e-6, rtol=0.)

    def test_fused_grid_preserves_independent_gradient_interpolation(self):
        generator = torch.Generator().manual_seed(42)
        grid = torch.randn((4, 5, 6), generator=generator, dtype=self.dtype)
        gradient = torch.randn((4, 5, 6, 3), generator=generator, dtype=self.dtype)
        lower = torch.tensor([-2., -1., 0.], dtype=self.dtype)
        upper = torch.tensor([3., 2., 4.], dtype=self.dtype)
        empty = torch.empty(0, dtype=self.dtype)
        field = GridZeroSetField(grid, gradient, lower, upper, empty, empty, empty)
        points = torch.randn((99, 3), generator=generator, dtype=self.dtype) * 5
        fused = FusedGridLookup(field).evaluate(points)
        torch.testing.assert_close(fused[:, 0], field.value(points), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(fused[:, 1:], field.gradient(points), atol=1e-12, rtol=1e-12)

    def test_reference_count_is_not_automatic_convergence(self):
        rows = [{"micro":m, "radius_over_h":1.,
                 "rgb_difference_to_max_micro":{"relative_l2":0. if m==128 else 1.},
                 "transmission_difference_to_max_micro":{"rms":0.},
                 "tau_difference_to_max_micro":{"relative_l2":0.}} for m in COUNTS]
        self.assertFalse(convergence(rows)["converged"])
        rows[-2]["rgb_difference_to_max_micro"]["relative_l2"] = 0.
        self.assertEqual(convergence(rows)["minimum_converged_count"], 64)
        rows[1]["rgb_difference_to_max_micro"]["relative_l2"] = 0.
        self.assertEqual(convergence(rows)["minimum_converged_count"], 64)

    def test_zero_radius_has_no_micro_quadrature_requirement(self):
        rows = [{"micro":m, "radius_over_h":0.,
                 "rgb_difference_to_max_micro":{"relative_l2":0.},
                 "transmission_difference_to_max_micro":{"rms":0.},
                 "tau_difference_to_max_micro":{"relative_l2":0.}} for m in COUNTS]
        self.assertEqual(convergence(rows)["minimum_converged_count"], 1)


if __name__ == "__main__":
    unittest.main()
