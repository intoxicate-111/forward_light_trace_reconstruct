"""Validate isolated v091 outputs; old v090 may still write its own reports."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from zlt.transverse_packet import write_json


def reject(value): raise ValueError('Nonfinite JSON '+value)


def validate():
    cache=ROOT/'runs/v091_basis_only_field'; checks={}
    for name,command in [('compileall',[sys.executable,'-m','compileall','-q','src','scripts','tests','demo.py']),
        ('full_tests',[sys.executable,'-m','unittest','discover','-s','tests']),('demo_verify',[sys.executable,'demo.py','--verify'])]:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (cache/(name+'.log')).write_text(result.stdout+result.stderr); checks[name]=result.returncode==0
    before=json.loads((cache/'starting_state.json').read_text())
    checks['historical_artifacts_unchanged']=all((ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before['historical_sha256'].items())
    checks['all_preexisting_renderer_and_v090_code_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before['protected_code_sha256'].items())
    checks['v090_policy']='No old experiment writes/stops by v091. Its report hashes may legitimately change from its own ongoing process.'
    report=json.loads((ROOT/'artifacts/v091_basis_only_field.json').read_text(),parse_constant=reject)
    checks['fixed_background']=report['formulation']=='F(x;lambda)=1+sum_k lambda_k B_k(x)'
    files=list((ROOT/'artifacts').glob('v091_*.json'))+list(cache.glob('*.json'))
    for p in files: json.loads(p.read_text(),parse_constant=reject)
    checks['strict_json_files']=len(files)
    for p in (ROOT/'artifacts').glob('v091_*.csv'):
        with p.open(newline='') as stream:
            rows=list(csv.reader(stream)); assert rows and all(len(row)==len(rows[0]) for row in rows)
    checks['csv_parse']=True
    pictures=list((ROOT/'figures').glob('v091_*.png'))
    for p in pictures:
        with Image.open(p) as im: im.load()
    checks['figures_decode']=len(pictures)
    arrays=list(cache.glob('*.npy'))
    for p in arrays: assert np.isfinite(np.load(p,mmap_mode='r')).all(),p
    checks['finite_arrays']=len(arrays)
    import torch
    from zlt.basis_only import BasisOnlyZeroSetField
    field=BasisOnlyZeroSetField.from_state_dict(torch.load(cache/'initialized_field.pt',weights_only=True))
    checks['single_scale']=bool(torch.all(field.layout.radii==field.layout.radii[0]))
    checks['no_hidden_base']=not any(hasattr(field,k) for k in ('base','grid','field','lookup'))
    checks['all_geometry_in_coefficients']=bool(torch.all(field.with_coefficients(torch.zeros_like(field.coefficients)).value(torch.randn((512,3),dtype=torch.float64))==1))
    from zlt.transverse_packet import digest
    checks['coefficient_digest']=digest(field.coefficients)==report['initialization']['field_digest']
    checks['layout_digest']=digest(field.layout.centers)==report['initialization']['layout_digest']
    checks['no_birth']=report['dynamic_birth_executed'] is False
    renderer=report['renderer']
    if 'target_digest' in renderer:
        checks['target_digest']=digest(torch.from_numpy(np.load(cache/'renderer_target.npy')))==renderer['target_digest']
        checks['initial_detector_energy']=all(x['relative_error']<1e-10 for x in renderer['target_timing']['energy'])
    checks['head_unchanged']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    result=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    checks['git_diff_check']=result.returncode==0
    checks['code_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/zlt').glob('basis_*.py')}
    checks['passed']=all(v is not False for v in checks.values()) and len(pictures)>=7
    write_json(ROOT/'artifacts/v091_validation.json',checks)
    print(json.dumps(checks,indent=2)); assert checks['passed']


if __name__=='__main__': validate()
