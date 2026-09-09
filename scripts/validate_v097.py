"""Strict validation for v0.9.7 without rerunning multi-hour CUDA experiments."""
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
sys.path.insert(0, str(ROOT/"src"))
from zlt.transverse_packet import write_json

EXPECTED_TARGET_SHA256 = "e137b58e55d8184b67feecf537575435a6a5ce58b0541d667dc08311592e9d4c"


def strict(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def command(argv):
    run = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    return {"command": argv, "returncode": run.returncode,
            "stdout_tail": run.stdout[-4000:], "stderr_tail": run.stderr[-4000:]}


def main():
    artifact_path = ROOT/"artifacts/v097_high_bandwidth_topology.json"
    artifact = strict(artifact_path)
    obs = artifact["phase_A_observation"]; opt = artifact["phase_B_optimization"]
    path = artifact["phase_C_oracle_path"]; basins = artifact["basin_of_attraction"]
    slice_result = artifact["two_dimensional_loss_slice"]
    fiber_audit = artifact["fixed_2048_fiber_audit"]
    configurations = obs["configurations"]
    names = {row["name"] for row in configurations}
    expected = {"512x512_16K", "512x512_64K", "1024x1024_16K",
                "1024x1024_64K", "1920x1080_64K"}
    target_path = ROOT/obs["target"]["path"]
    report = (ROOT/"artifacts/v097_high_bandwidth_topology.md").read_text()
    required_headings = ("CUDA and bandwidth configuration", "Oracle topology-direction observability",
                         "Gradient alignment", "Source-density effect", "Detector-resolution effect",
                         "Jacobian spectrum", "High-bandwidth RGB optimization", "Topology trajectory",
                         "Oracle path tomography", "Basin of attraction", "Need for complex continuation",
                         "Final verdict")
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout.splitlines()
    allowed_prefixes = ("README.md", "src/zlt/high_bandwidth_topology.py", "scripts/report_v097.py",
                        "scripts/validate_v097.py", "tests/test_high_bandwidth_topology.py",
                        "artifacts/v097", "figures/v097")
    changed_paths = [line[3:] for line in git_status]
    checks = {
        "strict_json_parses": artifact["version"] == "v0.9.7",
        "cuda_hard_gate": artifact["verdicts"]["CUDA_HARD_GATE_PASSED"] and not artifact["cuda_environment"]["cpu_fallback"],
        "no_hpc": not artifact["cuda_environment"]["hpc_used"],
        "target_hash": hashlib.sha256(target_path.read_bytes()).hexdigest() == EXPECTED_TARGET_SHA256 == obs["target"]["sha256"],
        "target_topology_reproduced": all(t["watertight"] and t["components"] == 1 and
                                           t["inferred_genus"] == 1 and not t["extraction_boundary_contact"]
                                           for t in obs["target"]["topology"].values()),
        "high_bandwidth_matrix_complete": names == expected,
        "fullhd_attempted": "1920x1080_64K" in names,
        "finite_configuration_metrics": all(all(np.isfinite(float(row[key])) for key in
            ("mse", "Jv_oracle_norm", "Jv_oracle_norm_per_sqrt_observation", "gradient_norm",
             "negative_gradient_oracle_cosine", "oracle_g", "oracle_h", "oracle_alpha_star",
             "oracle_projection_on_randomized_range_JT")) for row in configurations),
        "correct_output_space_range_sketch": all(row["output_sketch_probe_count"] == 32 and
            0 <= row["oracle_projection_on_randomized_range_JT"] <= 1 and
            "output-space" in row["range_projection_scope"] for row in configurations),
        "all_alignments_near_zero_and_negative": all(-.02 < row["negative_gradient_oracle_cosine"] < 0
                                                       for row in configurations),
        "all_oracle_gn_steps_negative": all(row["oracle_alpha_star"] < 0 for row in configurations),
        "sparse_equivalence": artifact["sparse_equivalence"]["equivalent"] and
                              artifact["sparse_equivalence"]["maximum_absolute_error"] < 1e-9,
        "no_dense_jacobian": not any(row["dense_jacobian_allocated"] for row in configurations),
        "image_only_contract": not any(opt["optimizer_forbidden_inputs"].values()),
        "primary_budget_and_fresh_stop": opt["maximum_accepted_steps"] >= 100 and opt["accepted_steps"] == 20 and
                                         opt["stop"] == "STABLE_NO_PROGRESS_AFTER_32_FRESH_RETRACES",
        "every_trial_fresh_or_failed": all("actual_fresh_retrace_mse" in trial or "failure" in trial
            for row in opt["trajectory"][:-1] for trial in row.get("trials", [])),
        "primary_remains_genus0": not opt["topology_penetrated"] and
            all(t["watertight"] and t["components"] == 1 and t["inferred_genus"] == 0
                for t in opt["final_topology"].values()),
        "primary_final_fibers_single_root": opt["final_fibers"]["root_count_histogram"] == {"1": 2048},
        "primary_selected_2048_fibers_single_root":
            [r["iteration"] for r in fiber_audit["primary_selected_iterations"]] == [0, 5, 10, 15, 20] and
            all(r["fibers"]["root_count_histogram"] == {"1": 2048}
                for r in fiber_audit["primary_selected_iterations"]),
        "oracle_path_endpoints": path["samples"][0]["t"] == 0 and path["samples"][-1]["t"] == 1 and
                                 path["samples"][-1]["mse"] < 1e-20,
        "oracle_path_topology_bracket": next(r for r in path["samples"] if r["t"] == .725)["topology_grid72"]["inferred_genus"] == 0 and
                                        next(r for r in path["samples"] if r["t"] == .74)["topology_grid72"]["inferred_genus"] == 1,
        "oracle_path_has_loss_barrier": max(r["mse"] for r in path["samples"][:-1]) > path["samples"][0]["mse"],
        "oracle_path_2048_fibers_reproduce_target": len(fiber_audit["oracle_path"]) == len(path["samples"]) and
                                                    fiber_audit["oracle_path"][-1]["fibers"]["root_count_histogram"] == {"1": 1959, "3": 89},
        "basin_starts_complete": {r["initial_fraction"] for r in basins["runs"]} == {.25, .5, .75},
        "only_already_topological_basin_succeeds": [r["initial_fraction"] for r in basins["runs"] if r["topology_penetrated"]] == [.75],
        "genus0_basins_end_single_root": all(r["final_fibers"]["root_count_histogram"] == {"1": 2048}
                                               for r in basins["runs"] if r["initial_fraction"] < .75),
        "loss_slice_complete": len(slice_result["samples"]) == 25,
        "loss_slice_finite_or_explicit_failure": all((row.get("finite") is True and np.isfinite(row["mse"])) or
                                                       (row.get("finite") is False and "failure" in row)
                                                       for row in slice_result["samples"]),
        "classification": artifact["verdicts"]["SUCCESS_HIERARCHY"] == "HIGH_BANDWIDTH_STILL_MISALIGNED" and
                          not artifact["verdicts"]["MEASUREMENT_BANDWIDTH_HYPOTHESIS_SUPPORTED"],
        "complex_not_forced": not artifact["verdicts"]["COMPLEX_CONTINUATION_ACTIVATED"],
        "required_report_headings_exact_order": [line[3:] for line in report.splitlines() if line.startswith("## ")] == list(required_headings),
        "historical_tracked_outputs_not_modified": all(any(path.startswith(prefix) for prefix in allowed_prefixes)
                                                         for path in changed_paths),
    }
    csv_path = ROOT/"artifacts/v097_high_bandwidth_topology.csv"
    with csv_path.open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    checks["csv_parses"] = bool(csv_rows) and {row["phase"] for row in csv_rows} == {"observation", "optimization", "oracle_path"}
    for name in artifact["figures"]:
        with Image.open(ROOT/name) as image:
            image.verify()
    checks["figures_decode"] = len(artifact["figures"]) >= 8
    generated = [artifact_path, csv_path, ROOT/"artifacts/v097_high_bandwidth_topology.md",
                 ROOT/"artifacts/v097_cuda_environment.json", ROOT/"artifacts/v097_cuda_environment.txt",
                 ROOT/"artifacts/v097_sparse_equivalence.json"]
    checks["generated_text_no_trailing_whitespace"] = all(
        all(line == line.rstrip() for line in item.read_text().splitlines()) for item in generated)

    runs = [command([sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts", "demo.py"]),
            command([sys.executable, "-m", "pytest", "-q"]),
            command([sys.executable, "demo.py", "--verify"]),
            command(["git", "diff", "--check"])]
    checks.update(compileall=runs[0]["returncode"] == 0, unit_tests=runs[1]["returncode"] == 0,
                  demo_verify=runs[2]["returncode"] == 0, git_diff_check=runs[3]["returncode"] == 0)
    passed = all(checks.values())
    result = {"version": "v0.9.7", "passed": passed, "checks": checks,
              "target_sha256": EXPECTED_TARGET_SHA256, "changed_paths_at_validation": changed_paths,
              "commands": runs}
    write_json(ROOT/"artifacts/v097_validation.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
