import unittest
import torch
from zlt.global_convergence import prefix_weights,classify


def row(n,mse,rgb,gap=.001):
    return {"samples":n,"whole_image_mse":mse,"adjacent_rgb_per_view":[rgb]*4,
        "coverage":[{"no_center_0_5px":gap,"no_center_1px":gap/100,"missing_energy_0_5px":gap,"missing_energy_1px":gap/100} for _ in range(4)],"radial_banding_score":.1}


class GlobalConvergenceTests(unittest.TestCase):
    def test_weights_preserve_source_mass(self):
        for n in (512,1024,4096):
            self.assertAlmostEqual(float(prefix_weights(n,1.0034).sum()),1.0034,places=13)

    def test_mse_plateau_alone_does_not_allow_birth(self):
        verdict,_=classify([row(1,1,.3),row(2,.8,.2),row(4,.79,.08)])
        self.assertTrue(verdict["GLOBAL_SAMPLE_COUNT_HAS_REACHED_FULLHD_PLATEAU"])
        self.assertFalse(verdict["FULLHD_GLOBAL_SOURCE_QUADRATURE_CONVERGED"])
        self.assertEqual(verdict["RECOMMENDED_GLOBAL_SAMPLE_COUNT"],"UNRESOLVED")

    def test_all_gates_required(self):
        rows=[row(1,1,.3),row(2,.8,.1),row(4,.79,.005)]
        self.assertTrue(classify(rows)[0]["FULLHD_GLOBAL_SOURCE_QUADRATURE_CONVERGED"])
        rows[-1]["coverage"][0]["missing_energy_1px"]=.01
        self.assertFalse(classify(rows)[0]["FULLHD_GLOBAL_SOURCE_QUADRATURE_CONVERGED"])


if __name__=="__main__": unittest.main()
