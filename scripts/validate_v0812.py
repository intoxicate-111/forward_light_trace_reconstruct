"""Read-only validation of both scientifically separate v0.8.12 experiments."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from zlt.measurement_bandwidth import BASE, PRIMARY, KERNELS, state_digest


def validate():
    def invalid(value): raise ValueError(value)
    def finite(value):
        if isinstance(value,dict):
            for item in value.values(): finite(item)
        elif isinstance(value,list):
            for item in value: finite(item)
        elif isinstance(value,float): assert math.isfinite(value)
    root=Path(__file__).resolve().parents[1]
    reports=[]; figures=0; csv_count=0
    for name in ("measurement_bandwidth","polar_aliasing"):
        path=root/f"artifacts/v0812_{name}.json"
        report=json.loads(path.read_text(),parse_constant=invalid); finite(report); reports.append(report)
        with path.with_suffix(".csv").open(newline="") as stream:
            reader=csv.DictReader(stream)
            assert reader.fieldnames==["metric","value"]
            rows=list(reader); assert rows and all(set(r)=={"metric","value"} for r in rows)
            csv_count+=len(rows)
        for filename in report["figures"]:
            with Image.open(root/filename) as image:
                image.load(); assert image.width>100 and image.height>100
            figures+=1
    main,alias=reports
    state=torch.load(root/"runs/v0812_measurement/scene.pt",weights_only=True)
    identity=state_digest(state)
    assert identity==main["cache_digest"]==main["cache_digest_after"]==alias["frozen_measurement_digest"]
    assert alias["frozen_measurement_unchanged"]
    for key in ("SCENE_TRANSPORT_FROZEN_ACROSS_MEASUREMENT_ABLATION","CACHED_MEASUREMENT_REPLAY_EQUIVALENT","DETECTOR_ENERGY_CONSERVING"):
        assert main["verdicts"][key]
    assert main["production_transmission_calls"]==0
    assert main["resolution"]==[1080,1920] and main["views"]==4
    assert {(r["gate"],r["kernel"]) for r in main["measurements"] if r["primary"]}=={(g,k) for g in PRIMARY for k in KERNELS}
    assert all(r["cache_digest"]==identity for r in main["measurements"])
    assert len(main["cosine_bins"])==28 and len(main["positive_cosine_thin_decomposition"])==20
    assert len(main["figures"])>=11 and len(alias["figures"])>=4
    assert alias["verdicts"]["PHASE_VARIANTS_MASS_MATCHED"]
    assert len(alias["variants"])==4
    for scheme,row in alias["variants"].items():
        assert row["emitter_count"]==1048576
        assert row["shared_ray_compatible"]==(scheme in ("CURRENT","PER_CHART"))
        for suffix in ("rgb","HARD_OUTWARD_occupancy","NO_OUTWARD_GATE_occupancy"):
            data=np.load(root/f"runs/v0812_aliasing/{scheme}_{suffix}.npy",mmap_mode="r")
            assert data.shape==(4,1080,1920,3) and np.isfinite(data).all()
    files=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"artifacts","figures"],cwd=root,text=True).splitlines()
    hashes={}
    for name in files:
        expected=subprocess.check_output(["git","show",f"{BASE}:{name}"],cwd=root)
        actual=(root/name).read_bytes()
        assert expected==actual,f"historical file modified: {name}"
        hashes[name]=hashlib.sha256(actual).hexdigest()
    subprocess.run(["git","diff","--check"],cwd=root,check=True)
    result={"strict_json":True,"finite_values":True,"csv_rows":csv_count,"decoded_figures":figures,
        "historical_files_unchanged":len(hashes),"cache_digest":identity,"primary_fullhd_rows":15,"phase_fullhd_rows":4}
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__": validate()
