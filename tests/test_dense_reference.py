import unittest
import numpy as np
import torch
import importlib.util
from zlt.dense_reference import relative_change,compare_reference


class DenseReferenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("open3d"),"Open3D optional dependency")
    def test_camera_first_hit_rejects_occluded_surface_sample(self):
        import open3d as o3d
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,-.5],[1,0,-.5],[0,1,-.5]],dtype=np.float32)
        faces=np.array([[0,1,2],[3,4,5]],dtype=np.uint32)
        scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
        rays=np.array([[.25,.25,4,0,0,-1],[.25,.25,3.5,0,0,-1]],dtype=np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)["primitive_ids"].numpy()
        np.testing.assert_array_equal(hit==np.array([0,1]),[True,False])

    def test_convergence_requires_all_views_and_gradients(self):
        base=np.random.default_rng(1).random((4,16,16,3))
        self.assertTrue(compare_reference(base,base)["passed"])
        changed=base.copy(); changed[3]*=1.1
        self.assertFalse(compare_reference(base,changed)["passed"])

    def test_zero_reference_is_finite(self):
        self.assertEqual(relative_change(np.zeros(10),np.zeros(10)),0.)

    def test_sobol_resume_preserves_sample_identity(self):
        full=torch.quasirandom.SobolEngine(3,scramble=True,seed=211).draw(1024,dtype=torch.float64)
        resumed=torch.quasirandom.SobolEngine(3,scramble=True,seed=211)
        resumed.fast_forward(512)
        torch.testing.assert_close(resumed.draw(512,dtype=torch.float64),full[512:],atol=0,rtol=0)


if __name__=="__main__": unittest.main()
