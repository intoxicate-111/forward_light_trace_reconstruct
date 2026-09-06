"""Strict validation of fixed-measure evidence without rewriting history."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import re
import numpy as np
import torch
from PIL import Image
from zlt.fixed_measure import BASE,CACHE
from zlt.measurement_bandwidth import state_digest
from zlt.transverse_packet import write_json


def validate():
    def invalid(x): raise ValueError(x)
    def finite(x):
        if isinstance(x,dict):
            for v in x.values(): finite(v)
        elif isinstance(x,list):
            for v in x: finite(v)
        elif isinstance(x,float): assert math.isfinite(x)
    r=json.loads(Path("artifacts/v0814_fixed_measure.json").read_text(),parse_constant=invalid)
    finite(r)
    assert r["starting_commit"]==BASE and r["emitters"]==4194304
    assert r["views"]==4 and r["resolution"]==[1080,1920] and r["capture_grid"]==4
    assert r["NO_DENSE_POINT_BY_LAMBDA_TENSOR"] and not r["birth"] and not r["geometry_optimization"]
    # Fixed six-step attachment approximates the trilinear zero surface.
    assert r["global_measure"]["max_surface_residual"]<1e-6
    a,b=(r["comparisons"][k] for k in ("current","global"))
    assert abs(a["colored_source_budget"]-b["colored_source_budget"])<1e-10
    assert max(e["relative_energy_difference"] for e in r["energy_accounting"])<1e-5
    for name,path in (("current",Path("runs/v0813_density_sparse/source_64.pt")),("global",CACHE/"global_energy_matched_state.pt")):
        state=torch.load(path,weights_only=True)
        assert state_digest(state)==r["cache_digests"][name]
        assert len(state["positions"])==4194304
        for key in ("positions","normals","colors","weights","transmission"):
            assert torch.isfinite(state[key]).all()
        assert len(r["comparisons"][name]["per_view"])==4
    rgb=np.load(CACHE/"global_rgb.npy",mmap_mode="r")
    assert rgb.shape==(4,1080,1920,3) and np.isfinite(rgb).all()
    assert all(abs(row["mass_difference"])<1e-10 for row in r["lambda_density"])
    assert r["closure"]["passed"] and r["verdicts"]["GLOBAL_FORWARD_GRADIENT_PASSES_FINITE_DIFFERENCE_CLOSURE"]
    for name in r["figures"]:
        with Image.open(name) as image: image.load(); assert image.width>100
    with Path("artifacts/v0814_fixed_measure.csv").open(newline="") as stream:
        reader=csv.DictReader(stream); assert reader.fieldnames==["metric","value"]
        rows=list(reader); assert len(rows)>100
    historical=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"artifacts","figures"],text=True).splitlines()
    for name in historical:
        assert Path(name).read_bytes()==subprocess.check_output(["git","show",f"{BASE}:{name}"]),name
    commands={"compileall":[sys.executable,"-m","compileall","-q","src","tests","scripts","demo.py"],
        "demo_verify":[sys.executable,"demo.py","--verify"],
        "unit_tests":[sys.executable,"-m","unittest","discover","-s","tests"],
        "git_diff_check":["git","diff","--check"]}
    logs={}
    for name,command in commands.items():
        completed=subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        logs[name]=completed.stdout
        (CACHE/f"validation_{name}.log").write_text(completed.stdout)
    result={"strict_json":True,"finite_values":True,"energy_accounting":True,"cache_digests":True,"lambda_closure":True,
        "csv_rows":len(rows),"decoded_figures":len(r["figures"]),"historical_files_unchanged":len(historical),
        "compileall":True,"demo_verify":True,"git_diff_check":True,"unit_tests":int(re.search(r"Ran (\d+) tests",logs["unit_tests"]).group(1))}
    write_json(Path("artifacts/v0814_validation.json"),result)
    diff=subprocess.check_output(["git","diff","--stat"],text=True)
    new=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],text=True)
    Path("artifacts/v0814_git_diff.txt").write_text("Base: "+BASE+"\nTracked modifications:\n"+diff+"\nNew files (not staged):\n"+new+"\nHistorical artifacts/figures unchanged: "+str(len(historical))+"\nNo commit or push performed by this task.\n")
    print(json.dumps(result,indent=2)); return result


if __name__=="__main__": validate()
