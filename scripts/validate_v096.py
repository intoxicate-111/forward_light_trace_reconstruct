"""Strict validation for v0.9.6 without rerunning the expensive experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zlt.transverse_packet import write_json


def command(argv):
    run = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    return {"command": argv, "returncode": run.returncode,
            "stdout_tail": run.stdout[-4000:], "stderr_tail": run.stderr[-4000:]}


def main():
    artifact_path = ROOT / "artifacts/v096_topology_penetration.json"
    artifact = json.loads(artifact_path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    snapshot = json.loads((ROOT / "runs/v096_topology_penetration/starting_state.json").read_text())
    allowed_code_change = {"src/zlt/dense_jet_torus.py"}
    mismatches = {}
    for name, expected in snapshot["protected_sha256"].items():
        if name in allowed_code_change:
            continue
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    historical_artifact_mismatches = {k: v for k, v in mismatches.items()
                                      if k.startswith(("artifacts/v093", "artifacts/v094", "artifacts/v095",
                                                       "figures/v093", "figures/v094", "figures/v095"))}
    checks = {
        "cuda_hard_gate": artifact["verdicts"]["CUDA_HARD_GATE_PASSED"],
        "historical_v093_v094_v095_artifacts_unchanged": not historical_artifact_mismatches,
        "other_protected_files_unchanged_except_documented_renderer_streaming": not mismatches,
        "capacity_genus1": artifact["verdicts"]["FROZEN_CHART_FAMILY_CONTAINS_GENUS1"],
        "capacity_high_resolution_stable": all(
            x["watertight"] and x["components"] == 1 and x["inferred_genus"] == 1 and not x["extraction_boundary_contact"]
            for x in artifact["representation_capacity"]["winner"]["topology_verification"].values()),
        "target_coefficients_saved_and_digest_matches": hashlib.sha256((ROOT / artifact["saved_theta_T"]["path"]).read_bytes()).hexdigest() == artifact["saved_theta_T"]["sha256"],
        "image_only_optimizer_contract": not any(artifact["image_penetration"]["optimizer_forbidden_inputs"].values()),
        "fresh_retrace_trials_recorded": all(
            "actual_fresh_retrace_mse" in trial or "failure" in trial
            for row in artifact["image_penetration"]["trajectory"][:-1] for trial in row.get("trials", [])),
        "final_genus0_negative_result": not artifact["image_penetration"]["topology_penetrated"],
        "final_fibers_single_root": artifact["image_penetration"]["final_fibers"]["one_root_fraction"] == 1.0,
        "no_complex_without_critical_event": not artifact["verdicts"]["COMPLEX_CONTINUATION_ACTIVATED"],
        "classification": artifact["verdicts"]["FINAL_CLASSIFICATION"] == "CAPACITY_YES_REACHABILITY_NO",
        "all_cached_arrays_finite": all(np.isfinite(np.load(path)).all() for path in (ROOT / "runs/v096_topology_penetration").glob("*.npy")),
    }
    csv_path = ROOT / artifact["csv"]
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    checks["csv_parses"] = bool(rows)
    for name in artifact["figures"]:
        with Image.open(ROOT / name) as image:
            image.verify()
    checks["figures_decode"] = len(artifact["figures"]) >= 10
    report_text = (ROOT / "artifacts/v096_topology_penetration.md").read_text()
    headings = ("CUDA environment", "Frozen-chart topology analysis", "Representation capacity",
                "Normal-fiber root behavior", "Same-family genus-1 construction", "Image-driven topology penetration",
                "Emergent critical mode", "Image observability", "Need for complex continuation",
                "Analytic-torus generalization", "Limitations", "Final verdict")
    checks["required_report_headings"] = all(f"## {i}. {name}" in report_text for i, name in enumerate(headings, 1))
    generated_text = [artifact_path, csv_path, ROOT / "artifacts/v096_topology_penetration.md",
                      ROOT / "artifacts/v096_cuda_environment.json", ROOT / "artifacts/v096_cuda_environment.txt"]
    checks["generated_text_has_no_trailing_whitespace"] = all(
        all(line == line.rstrip() for line in path.read_text().splitlines()) for path in generated_text)

    runs = [
        command([sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts", "demo.py"]),
        command([sys.executable, "-m", "pytest", "-q"]),
        command([sys.executable, "demo.py", "--verify"]),
        command(["git", "diff", "--check"]),
    ]
    checks.update({"compileall": runs[0]["returncode"] == 0, "unit_tests": runs[1]["returncode"] == 0,
                   "demo_verify": runs[2]["returncode"] == 0, "git_diff_check": runs[3]["returncode"] == 0})
    passed = all(checks.values())
    result = {"version": "v0.9.6", "passed": passed, "checks": checks,
              "documented_allowed_code_change": {"src/zlt/dense_jet_torus.py": "boundary-clearance audit and exact emitter-axis transport streaming"},
              "historical_mismatches": historical_artifact_mismatches, "other_protected_mismatches": mismatches,
              "commands": runs}
    write_json(ROOT / "artifacts/v096_validation.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
