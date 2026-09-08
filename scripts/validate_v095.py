"""Strict v0.9.5 validation without rerunning the long experiment."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from zlt.transverse_packet import write_json


def command(argv):
    run = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    return {"command": argv, "returncode": run.returncode,
            "stdout_tail": run.stdout[-4000:], "stderr_tail": run.stderr[-4000:]}


def main():
    report = json.loads((ROOT/"artifacts/v095_dense_jet_torus.json").read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    snapshot = json.loads((ROOT/"runs/v095_dense_jet_torus/starting_state.json").read_text())
    checks = {}
    protected = {}
    for name, expected in snapshot["protected_sha256"].items():
        path = ROOT/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected: protected[name] = {"expected": expected, "actual": actual}
    checks["historical_files_unchanged"] = not protected

    csv_path = ROOT/"artifacts/v095_dense_jet_torus.csv"
    with csv_path.open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    checks["csv_parses"] = bool(csv_rows)
    generated = [ROOT/"artifacts/v095_dense_jet_torus.csv", ROOT/"artifacts/v095_dense_jet_torus.json",
                 ROOT/"artifacts/v095_dense_jet_torus.md", ROOT/"artifacts/v095_validation.json"]
    checks["generated_text_has_no_trailing_whitespace"] = all(
        all(line == line.rstrip() for line in path.read_text().splitlines()) for path in generated if path.exists())
    figures = []
    for name in report["figures"]:
        with Image.open(ROOT/name) as image:
            image.verify()
        figures.append(name)
    checks["figures_decode"] = len(figures) >= 5
    checks["arrays_finite"] = all(np.isfinite(np.load(path)).all() for path in (ROOT/"runs/v095_dense_jet_torus").glob("*.npy"))

    main_run = report["runs"][0]; high = report["high_resolution_topology"]
    checks.update({
        "exact_sphere_initialization": report["verdicts"]["INITIAL_DENSE_JET_IS_EXACT_SPHERE"],
        "image_only_target": report["verdicts"]["TARGET_INFORMATION_IMAGE_ONLY"],
        "fresh_retrace": report["verdicts"]["FRESH_REAL_RETRACE_FOR_ACCEPTANCE"],
        "loss_reduced": main_run["final_mse"] < main_run["initial_mse"],
        "target_genus_one": high["target"]["inferred_genus"] == 1,
        "endpoint_genus_zero_retained": high["final"]["inferred_genus"] == 0,
        "negative_claim_retained": report["verdicts"]["SPONTANEOUS_SPHERE_TO_TORUS_SUPPORTED"] is False,
        "no_forced_complex": all(not x["complex_continuation_activated"] for x in report["runs"]),
        "predicted_actual_separate": all(
            "predicted_local_mse" in row["trials"][-1] and "actual_fresh_retrace_mse" in row["trials"][-1]
            for run in report["runs"] for row in run["trajectory"][:-1]),
        "required_report_headings": all(f"## {i}. {name}" in (ROOT/"artifacts/v095_dense_jet_torus.md").read_text()
            for i, name in enumerate(("Dense representation validity", "Image-driven deformation", "Emergent critical mode",
                                      "Topology outcome", "Rendered-loss outcome", "Need for complex continuation", "Limitations"), 1)),
    })
    runs = [
        command([sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts", "demo.py"]),
        command([sys.executable, "-m", "pytest", "-q"]),
        command([sys.executable, "demo.py", "--verify"]),
        command(["git", "diff", "--check"]),
    ]
    checks.update({"compileall": runs[0]["returncode"] == 0, "unit_tests": runs[1]["returncode"] == 0,
                   "demo_verify": runs[2]["returncode"] == 0, "git_diff_check": runs[3]["returncode"] == 0})
    passed = all(checks.values())
    result = {"version": "v095", "passed": passed, "checks": checks,
              "historical_mismatches": protected, "commands": runs}
    write_json(ROOT/"artifacts/v095_validation.json", result)
    print(json.dumps(result, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
