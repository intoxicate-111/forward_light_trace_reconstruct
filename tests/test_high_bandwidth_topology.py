"""Focused invariants for the v0.9.7 sparse high-bandwidth diagnostic."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

import zlt.high_bandwidth_topology as high_bandwidth

ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_interface_cannot_receive_topology_oracle():
    assert list(inspect.signature(high_bandwidth.optimize_image_only).parameters) == [
        "initial_field", "target_image", "cfg", "label", "maximum_steps"]


def test_experiment_has_hard_cuda_device_and_no_dense_claim():
    assert high_bandwidth.DEVICE.type == "cuda"
    source = inspect.getsource(high_bandwidth.observation_group)
    assert "torch.sparse.mm" in source
    assert "range_sketch" in source


def test_stateless_output_probe_repeats_per_detector_row():
    original = high_bandwidth.DEVICE
    high_bandwidth.DEVICE = torch.device("cpu")
    try:
        rows = torch.tensor([3, 11, 3, 19], dtype=torch.int64)
        values = high_bandwidth._output_rademacher(rows, view=1, probes=8)
    finally:
        high_bandwidth.DEVICE = original
    assert torch.equal(values[0], values[2])
    assert set(values.unique().tolist()) == {-1/(8**.5), 1/(8**.5)}


def test_committed_artifact_distinguishes_observability_from_alignment():
    path = ROOT/"artifacts/v097_high_bandwidth_topology.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    rows = result["phase_A_observation"]["configurations"]
    assert all(row["Jv_oracle_norm"] > 0 for row in rows)
    assert all(-.02 < row["negative_gradient_oracle_cosine"] < 0 for row in rows)
    assert result["verdicts"]["SUCCESS_HIERARCHY"] == "HIGH_BANDWIDTH_STILL_MISALIGNED"


def test_correct_range_projection_does_not_reuse_vt_containing_probe_projection():
    path = ROOT/"artifacts/v097_high_bandwidth_topology.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["phase_A_observation"]["configurations"]
    assert all(row["seeded_randomized_projection_estimate"] > .99 for row in rows)
    assert all(row["oracle_projection_on_randomized_range_JT"] < .1 for row in rows)
    assert all("not used" in row["legacy_seeded_projection_interpretation"] for row in rows)


def test_primary_and_genus_zero_basin_endpoints_have_single_root_fibers():
    path = ROOT/"artifacts/v097_high_bandwidth_topology.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    assert result["phase_B_optimization"]["final_fibers"]["root_count_histogram"] == {"1": 2048}
    for run in result["basin_of_attraction"]["runs"]:
        if run["initial_fraction"] < .75:
            assert run["topology_penetrated"] is False
            assert run["final_fibers"]["root_count_histogram"] == {"1": 2048}
