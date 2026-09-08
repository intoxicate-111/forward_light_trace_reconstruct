"""Focused invariant and artifact tests for v0.9.6."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

import zlt.topology_penetration as topology_penetration
from zlt.topology_penetration import DEVICE, _image_step

ROOT = Path(__file__).resolve().parents[1]


def test_image_optimizer_cannot_receive_oracle_geometry():
    assert list(inspect.signature(_image_step).parameters) == ["J", "residual", "damping_fraction"]


def test_experiment_has_no_cpu_fallback_device():
    assert DEVICE.type == "cuda"


def test_normal_fiber_root_counter_on_exact_sphere():
    original = topology_penetration.DEVICE
    topology_penetration.DEVICE = torch.device("cpu")
    try:
        field = topology_penetration.make_field(16, 0, .8)
        roots = topology_penetration.fiber_roots(
            field, direction_count=32, samples=65, s_min=-.5, s_max=.5)
    finally:
        topology_penetration.DEVICE = original
    assert roots["root_count_histogram"] == {"1": 32}
    assert roots["near_even_root_fibers"] == 0


def test_capacity_and_negative_reachability_are_not_conflated():
    path = ROOT / "artifacts/v096_topology_penetration.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    assert report["verdicts"]["FROZEN_CHART_FAMILY_CONTAINS_GENUS1"] is True
    assert report["verdicts"]["SAME_FAMILY_RGB_ONLY_REACHED_GENUS1"] is False
    assert report["verdicts"]["FINAL_CLASSIFICATION"] == "CAPACITY_YES_REACHABILITY_NO"


def test_certified_capacity_is_stable_and_away_from_boundary():
    path = ROOT / "artifacts/v096_topology_penetration.json"
    if not path.exists():
        return
    winner = json.loads(path.read_text())["representation_capacity"]["winner"]
    for topology in winner["topology_verification"].values():
        assert topology["components"] == 1
        assert topology["watertight"] is True
        assert topology["euler_number"] == 0
        assert topology["inferred_genus"] == 1
        assert topology["extraction_boundary_contact"] is False


def test_final_rgb_endpoint_retains_one_root_per_fiber():
    path = ROOT / "artifacts/v096_topology_penetration.json"
    if not path.exists():
        return
    fibers = json.loads(path.read_text())["image_penetration"]["final_fibers"]
    assert fibers["root_count_histogram"] == {"1": 2048}
    assert fibers["near_even_root_fibers"] == 0


def test_same_family_target_was_generated_and_is_genus_one():
    path = ROOT / "artifacts/v096_topology_penetration.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    assert (ROOT / report["saved_theta_T"]["path"]).exists()
    assert report["image_penetration"]["target_topology"]["inferred_genus"] == 1
    assert report["image_penetration"]["optimizer_target_inputs"] == ["immutable RGB target image"]
