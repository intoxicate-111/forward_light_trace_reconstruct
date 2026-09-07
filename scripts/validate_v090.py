"""Validate only the new v090 namespace; never rewrite historical reports."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from zlt.transverse_packet import write_json


def reject(value): raise ValueError('Non-finite JSON constant: '+value)


def validate():
    cache=ROOT/'runs/v090_matched_birth'; checks={}
    for name,cmd in [('compileall',[sys.executable,'-m','compileall','-q','src','scripts','tests','demo.py']),
        ('demo_verify',[sys.executable,'demo.py','--verify']),('unit_tests',[sys.executable,'-m','unittest','discover','-s','tests'])]:
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        (cache/('final_'+name+'.log')).write_text(result.stdout+result.stderr)
        checks[name]=result.returncode==0
    manifest=json.loads((cache/'historical_hashes.json').read_text())
    checks['historical_artifacts_unchanged']=all((ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    starting=json.loads((cache/'starting_state.json').read_text())
    checks['preexisting_untracked_preserved']=all((ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in starting['preexisting_untracked'].items())
    files=list((ROOT/'artifacts').glob('v090_*.json'))+list(cache.glob('*.json'))
    for p in files: json.loads(p.read_text(),parse_constant=reject)
    checks['strict_json']=len(files)
    tables=list((ROOT/'artifacts').glob('v090_*.csv'))
    for p in tables:
        with p.open(newline='') as f:
            rows=list(csv.reader(f)); assert rows and all(len(r)==len(rows[0]) for r in rows)
    checks['csv_parse']=len(tables)
    images=list((ROOT/'figures').glob('v090_*.png'))
    for p in images:
        with Image.open(p) as im: im.load()
    checks['figures_decode']=len(images)
    arrays=list(cache.glob('*.npy'))
    preflight_shapes={'gt_smoke_PREFLIGHT_ONLY.npy':(4,128,128,3)}
    primary_count=0; preflight_count=0
    for p in arrays:
        a=np.load(p,mmap_mode='r'); assert np.isfinite(a).all(),p
        if p.name in preflight_shapes:
            assert a.shape==preflight_shapes[p.name],p
            preflight_count+=1
        else:
            assert a.shape==(4,1080,1920,3),p
            primary_count+=1
    checks['all_primary_arrays_finite_fullhd']=primary_count
    checks['explicit_preflight_arrays_finite_expected_shape']=preflight_count
    report=json.loads((ROOT/'artifacts/v090_matched_birth.json').read_text(),parse_constant=reject)
    import torch
    from zlt.transverse_packet import digest
    checks['target_digest']=digest(torch.from_numpy(np.load(cache/'gt_current_soft_16m.npy')))==report['setup']['target']['digest']
    checks['current_regression']=report['setup']['v0817_current_replay_relative_l2']<1e-6
    checks['target_replay']=report['setup']['target']['replay_relative_l2']<1e-6
    checks['both_branches_reported']=set(report['branches'])=={'SOFT','GLOBE'}
    for b,r in report['branches'].items():
        checks[b+'_full_source']=r['source']['signature']['count']==16777216
        for row in r.get('births',[]):
            assert row['source_count']==16777216 and row['birth_source_mass_change']==0
            assert max(abs(c) for c in row['coefficients'])<=.03000000001
        for label in [b.lower()+'_initial']:
            accounting=json.loads((cache/(label+'.json')).read_text())['energy_accounting']
            assert max(x['numerical_energy_relative_error'] for x in accounting)<1e-5
    checks['diff_check']=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    checks['head_unchanged']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==starting['head']
    checks['git_diff_summary']=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True)
    checks['git_status']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    checks['passed']=all(v is not False for v in checks.values()) and len(images)>=11 and len(tables)>=4
    write_json(ROOT/'artifacts/v090_validation.json',checks)
    print(json.dumps(checks,indent=2)); assert checks['passed']


if __name__=='__main__': validate()
