"""Validate generated v0.8.11 evidence without altering historical artifacts."""
import csv
from array import array
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image


def validate():
    root = Path(__file__).resolve().parents[1]
    path = root / "artifacts/v0811_transverse_packet_integration.json"
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    report = json.loads(path.read_text(), parse_constant=invalid_constant)
    def finite(value):
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
        elif isinstance(value, float):
            assert math.isfinite(value)
    finite(report)
    counts = {1, 4, 8, 16, 32, 64, 128}
    radii = {0., .25, .5, 1., 1.5, 2.}
    assert {(r["radius_over_h"], r["micro"]) for r in report["screening"]} == {
        (radius, micro) for radius in radii for micro in counts}
    assert {r["epsilon_over_radius"] for r in report["controls"]["epsilon_rows"]} == {.25, .5, 1., 2.}
    assert report["configuration"]["primary_launch_start_over_h"] == 2.1
    assert all(abs(row["actual_launch_start"]-report["configuration"]["primary_launch_start"]) < 1e-14
               for row in report["screening"])
    assert report["centerline_radius_control"]["tau_mean_range"] < 1e-10
    assert {r["path_step_over_epsilon"] for r in report["controls"]["path_rows"]} == {1., .5, .25, .125}
    assert report["preflight"]["nested_prefix_exact"]
    assert len(report["preflight"]["sequence_digest"]) == 64
    assert len(report["forensic"]["production_sequence_digest"]) == 64
    offsets = report["forensic"]["production_micro_offset_statistics"]
    reference = offsets[-1]["offsets"]
    assert hashlib.sha256(array("d", (v for point in reference for v in point)).tobytes()).hexdigest() == report["forensic"]["production_sequence_digest"]
    for row in offsets:
        assert row["offsets"] == ([[0., 0., 0.]] if row["micro"]==1 else reference[:row["micro"]])
    assert {r["source_mass"] for r in report["screening"] + report["fullhd"]} == {report["source"]["source_mass"]}
    assert report["verdicts"]["NO_DENSE_POINT_BY_LAMBDA_TENSOR"]
    assert report["streaming_validation"]["equivalent"]
    assert report["streaming_validation"]["memory_bounded"]
    assert report["source"]["mass_error"] < 1e-10
    assert max(report["grid_lookup_validation"].values()) < 1e-9
    assert max(r["tau_error"] for r in report["preflight"]["streaming_equivalence"]) < 1e-10
    assert max(abs(r["transmission"]-.01) for r in report["controls"]["calibration"]) < 1e-5
    with path.with_suffix(".csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["metric", "value"]
        rows = list(reader)
        assert len(rows) > 1000 and all(set(row) == {"metric", "value"} for row in rows)
    assert len(report["figures"]) == 14
    for filename in report["figures"]:
        with Image.open(root / filename) as image:
            image.load()
            assert image.width > 100 and image.height > 100
    # Compare every historical tracked artifact byte-for-byte with starting
    # HEAD, not just the current git index. This catches uncommitted changes.
    baseline = report["starting_commit"]
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", baseline,
                                     "artifacts"], cwd=root, text=True).splitlines()
    hashes = {}
    for filename in files:
        expected = subprocess.check_output(["git", "show", f"{baseline}:{filename}"], cwd=root)
        actual = (root / filename).read_bytes()
        assert actual == expected, f"historical artifact changed: {filename}"
        hashes[filename] = hashlib.sha256(actual).hexdigest()
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
    result = {"strict_json": True, "finite_values": True, "csv_rows": len(rows),
              "decoded_figures": len(report["figures"]), "matrix_cells": len(report["screening"]),
              "historical_artifacts_unchanged": len(hashes), "historical_sha256": hashes}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    validate()
