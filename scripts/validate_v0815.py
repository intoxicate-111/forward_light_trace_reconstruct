"""Validate and activate the converged dense reference for new MSE comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from PIL import Image
from zlt.dense_reference import CACHE,compare_reference
from zlt.transverse_packet import digest,write_json


def validate():
    def invalid(x): raise ValueError(x)
    r=json.loads(Path("artifacts/v0815_dense_reference.json").read_text(),parse_constant=invalid)
    assert r["REFERENCE_CONVERGENCE_PASSED"]
    reference=np.load(r["reference_path"])
    assert reference.shape==(4,1080,1920,3) and np.isfinite(reference).all()
    assert digest(torch.from_numpy(reference))==r["reference_digest"]
    previous=np.load(CACHE/f"reference_{r['levels'][-2]['samples']}.npy")
    assert compare_reference(previous,reference)["passed"]
    assert abs(r["levels"][-1]["emitted_rgb"]/1.5489680757546012-1)<1e-4
    assert all((x>0).sum()>4096 for x in reference.mean(3))
    for method,m in r["comparisons"].items():
        assert np.isfinite(m["whole_image_mse"]) and np.isfinite(m["gradient_cosine"])
    for path in r["figures"]:
        with Image.open(path) as image: image.load()
    with Path("artifacts/v0815_dense_reference.csv").open() as stream:
        rows=list(csv.DictReader(stream)); assert len(rows)>100
    old=Path("runs/v0812_measurement/reference.npy")
    assert hashlib.sha256(old.read_bytes()).hexdigest()=="926988cf6c017fd24c203a54a44554d462b351afc2d71e440717679b1e8d7c34"
    for path,expected in {
        "artifacts/v0814_fixed_measure.json":"7903017ae2abb9bbdd67066bf27c83c74a607651316c38fbb0f7f821fa565d88",
        "artifacts/v0814_fixed_measure.csv":"07e3696a097e3d2a785cfa4d4327ec122661fce8b7d1add448842a791a018cd5",
    }.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    from zlt.fixed_measure import BASE
    historical=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"artifacts","figures"],text=True).splitlines()
    for path in historical:
        assert Path(path).read_bytes()==subprocess.check_output(["git","show",f"{BASE}:{path}"])
    subprocess.run(["git","diff","--check"],check=True)
    result={"converged":True,"finite_fullhd":True,"strict_json_csv":True,"figures_decode":True,
        "historical_files_unchanged":len(historical),"old_reference_unchanged":True,"csv_rows":len(rows)}
    write_json(Path("artifacts/v0815_validation.json"),result)
    write_json(Path("artifacts/mse_reference.json"),{"version":"0.8.15","path":r["reference_path"],"digest":r["reference_digest"],
        "report":"artifacts/v0815_dense_reference.json","definition":r["definition"],"validated":True})
    print(json.dumps(result,indent=2))


if __name__=="__main__": validate()
