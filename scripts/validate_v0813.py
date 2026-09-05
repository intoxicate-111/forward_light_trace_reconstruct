"""Validate the nine measured Full-HD density cells and historical immutability."""
import csv
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import torch
from zlt.density_matrix import BASE,CACHE,LEVELS,CAPTURE,classify_fidelity
from zlt.measurement_bandwidth import state_digest


def validate():
    def invalid(x): raise ValueError(x)
    def finite(x):
        if isinstance(x,dict):
            for v in x.values(): finite(v)
        elif isinstance(x,list):
            for v in x: finite(v)
        elif isinstance(x,float): assert math.isfinite(x)
    path=Path("artifacts/v0813_density_matrix.json")
    r=json.loads(path.read_text(),parse_constant=invalid); finite(r)
    assert r["starting_commit"]==BASE and r["resolution"]==[1080,1920] and r["views"]==4
    assert r["phase"]=="PER_CHART_PLUS_RING" and r["seed_list"]==[101]
    assert r["NO_DENSE_POINT_BY_LAMBDA_TENSOR"] and not r["geometry_optimization"] and not r["birth"]
    assert {(x["E"],x["C"]) for x in r["cells"]}=={(e,c) for e in range(1,4) for c in range(1,4)}
    assert r["baseline_replay"]["relative_l2"]<1e-6
    improved,_=classify_fidelity(r)
    assert r["verdicts"]["EMITTER_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY"]==improved["E"]
    assert r["verdicts"]["CAPTURE_DENSITY_INCREASE_IMPROVES_FULLHD_FIDELITY"]==improved["C"]
    assert r["initialization_equivalence"]["position_max_abs"]<1e-9
    assert r["initialization_equivalence"]["normal_max_abs"]<1e-9
    for e,n in enumerate(LEVELS,1):
        path_state=Path("runs/v0812_aliasing/PER_CHART_PLUS_RING_state.pt") if e==1 else CACHE/f"source_{n}.pt"
        state=torch.load(path_state,weights_only=True); identity=state_digest(state)
        rows=[x for x in r["cells"] if x["E"]==e]
        for row in rows:
            assert row["state_digest"]==identity
            assert row["total_emitters"]==1024*n*n and row["K_chart"]==1024
            assert row["N_theta"]==n and row["N_r"]==n and row["capture_samples_per_pixel"]==CAPTURE[row["C"]-1]**2
            assert row["maximum_chart_mass_difference"]<1e-12
            assert abs(row["weight_sum"]-r["sources"][0]["weight_sum"])<1e-10
            assert max(a["relative_energy_difference"] for a in row["energy_accounting"])<1e-5
            assert all(abs(a["projected_count_sum"]/row["total_emitters"]-1)<1e-5 for a in row["energy_accounting"])
            for kind,shape in (("rgb",(4,1080,1920,3)),("occupancy",(4,1080,1920))):
                image=np.load(CACHE/f"E{e}C{row['C']}_{kind}.npy",mmap_mode="r")
                assert image.shape==shape and np.isfinite(image).all()
        assert rows[0]["foreground_pixel_center_density"]==rows[1]["foreground_pixel_center_density"]==rows[2]["foreground_pixel_center_density"]
    with path.with_suffix(".csv").open(newline="") as stream:
        reader=csv.DictReader(stream); assert reader.fieldnames==["metric","value"]
        rows=list(reader); assert len(rows)>1000 and all(set(x)=={"metric","value"} for x in rows)
    for name in r["figures"]:
        with Image.open(name) as image: image.load(); assert image.width>100
    historical=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"artifacts","figures"],text=True).splitlines()
    for name in historical:
        expected=subprocess.check_output(["git","show",f"{BASE}:{name}"])
        assert Path(name).read_bytes()==expected,f"historical file changed: {name}"
    subprocess.run(["git","diff","--check"],check=True)
    result={"matrix_cells":9,"strict_json":True,"finite_values":True,"energy_accounting":True,
        "csv_rows":len(rows),"decoded_figures":len(r["figures"]),"historical_files_unchanged":len(historical),
        "baseline_replay_relative_l2":r["baseline_replay"]["relative_l2"]}
    print(json.dumps(result,indent=2)); return result


if __name__=="__main__": validate()
