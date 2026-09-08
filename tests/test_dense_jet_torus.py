import numpy as np
import torch

from zlt.dense_jet_torus import DEVICE, SphereReference, TorusTarget, make_field, topology_only


def test_zero_coefficients_are_exact_sphere():
    field = make_field(32)
    points = torch.randn((257, 3), dtype=torch.float64, device=DEVICE)
    reference = SphereReference(DEVICE)
    assert torch.equal(field.evaluate(points), reference.evaluate(points))
    assert bool((field.coefficients == 0).all())


def test_dense_layout_has_sparse_complete_surface_coverage():
    field = make_field(64)
    points = field.layout.centers
    pi, modes, values, gradients = field.layout.query(points)
    assert len(pi) == len(modes) == len(values) == len(gradients)
    assert torch.bincount(pi, minlength=len(points)).min() > 0
    assert modes.min() >= 0 and modes.max() < field.layout.count


def test_target_is_independent_genus_one_geometry():
    topology = topology_only(TorusTarget(DEVICE), n=32)
    assert topology["watertight"]
    assert topology["components"] == 1
    assert topology["euler_number"] == 0
    assert topology["inferred_genus"] == 1


def test_committed_trajectory_is_finite_and_freshly_retraced():
    import json
    from pathlib import Path

    report = json.loads(Path("artifacts/v095_dense_jet_torus.json").read_text())
    assert report["verdicts"]["TARGET_INFORMATION_IMAGE_ONLY"]
    assert report["verdicts"]["FRESH_REAL_RETRACE_FOR_ACCEPTANCE"]
    for run in report["runs"]:
        values = [row["actual_retraced_mse"] for row in run["trajectory"]]
        assert np.isfinite(values).all()
        assert all(b < a for a, b in zip(values, values[1:]))
        for row in run["trajectory"][:-1]:
            assert row["accepted"]
            assert "predicted_local_mse" in row["trials"][-1]
            assert "actual_fresh_retrace_mse" in row["trials"][-1]
