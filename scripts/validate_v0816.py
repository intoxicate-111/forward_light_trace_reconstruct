"""Validate phase-one evidence and preserve every pre-existing output."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import re
import numpy as np
import torch
from PIL import Image
from zlt.global_convergence import CACHE,LEVELS,classify
from zlt.transverse_packet import write_json,digest


def validate():
    def invalid(x): raise ValueError(x)
    report=json.loads(Path("artifacts/v0816_global_sample_convergence.json").read_text(),parse_constant=invalid)
    rows=report["levels"]; assert [r["samples"] for r in rows]==list(LEVELS)
    expected,_=classify(rows)
    assert report["verdicts"]==expected
    from zlt.dense_reference import load_mse_reference
    assert digest(torch.from_numpy(load_mse_reference()))==report["frozen"]["reference_digest"]
    mass=report["frozen"]["source_weight_sum"]; budgets=[]
    prior_positions=None; prior_transmission=None
    for row in rows:
        assert abs(row["source_weight_sum"]-mass)<1e-12
        budgets.append(row["emitted_rgb"])
        assert row["relative_energy_accounting_error"]<1e-5
        image=np.load(CACHE/f"rgb_{row['samples']}.npy",mmap_mode="r")
        assert image.shape==(4,1080,1920,3) and np.isfinite(image).all()
        state=torch.load(CACHE/f"source_{row['samples']}.pt",weights_only=True)
        for key in ("positions","normals","weights","colors","transmission"): assert torch.isfinite(state[key]).all()
        assert digest(state["positions"])==row["position_digest"]
        if prior_positions is not None:
            torch.testing.assert_close(state["positions"][:len(prior_positions)],prior_positions,atol=1e-12,rtol=0)
            torch.testing.assert_close(state["transmission"][:,:len(prior_positions)],prior_transmission,atol=1e-10,rtol=0)
        prior_positions=state["positions"]
        prior_transmission=state["transmission"]
        if row["samples"]==2**22: assert row["v0814_replay_relative_l2"]<1e-6
    assert max(budgets)/min(budgets)-1<1e-4
    for path in report["figures"]:
        with Image.open(path) as image: image.load()
    with Path("artifacts/v0816_global_sample_convergence.csv").open() as stream:
        csv_rows=list(csv.DictReader(stream)); assert len(csv_rows)>100
    manifest=json.loads((CACHE/"historical_hashes.json").read_text())
    for path,value in manifest.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
    commands={"compileall":[sys.executable,"-m","compileall","-q","src","scripts","tests","demo.py"],
        "demo_verify":[sys.executable,"demo.py","--verify"],"unit_tests":[sys.executable,"-m","unittest","discover","-s","tests"],
        "diff_check":["git","diff","--check"]}
    logs={}
    for name,cmd in commands.items():
        p=subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        logs[name]=p.stdout; (CACHE/f"validation_{name}.log").write_text(p.stdout)
    result={"finite_arrays":True,"strict_json_csv":True,"source_mass_conserved":True,"relative_emitted_rgb_range":max(budgets)/min(budgets)-1,
        "detector_energy_conserved":True,"nested_source_identity":True,"historical_files_unchanged":len(manifest),"figures_decoded":len(report["figures"]),
        "compileall":True,"demo_verify":True,"unit_tests":int(re.search(r"Ran (\d+) tests",logs["unit_tests"]).group(1)),"git_diff_check":True,"csv_rows":len(csv_rows)}
    write_json(Path("artifacts/v0816_validation.json"),result)
    summary="Cumulative working-tree diff, including pre-existing uncommitted v0814/v0815 work.\nHEAD: "+subprocess.check_output(["git","rev-parse","HEAD"],text=True)+subprocess.check_output(["git","diff","--stat"],text=True)+"\nUntracked outputs/code:\n"+subprocess.check_output(["git","ls-files","--others","--exclude-standard"],text=True)
    Path("artifacts/v0816_git_diff.txt").write_text(summary)
    print(json.dumps(result,indent=2))


if __name__=="__main__": validate()
