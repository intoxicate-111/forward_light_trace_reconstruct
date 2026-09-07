"""Read-only checks of historical data; write only new v0817 validation outputs."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import torch
from PIL import Image
from zlt.collision_transfer import CACHE
from zlt.transverse_packet import digest,write_json


def validate():
    def invalid(x): raise ValueError(x)
    report=json.loads(Path('artifacts/v0817_collision_transfer.json').read_text(),parse_constant=invalid)
    state=torch.load('runs/v0816_global_convergence/source_16777216.pt',weights_only=True,mmap=True)
    for k in ('positions','normals','weights','colors','directions','right','up','center'):
        assert digest(state[k])==report['frozen'][k]
        assert torch.isfinite(state[k]).all()
    assert abs(float(state['weights'].sum())-report['frozen']['source_mass'])<1e-12
    tau=torch.load('runs/v0816_global_convergence/transport_16777216.pt',weights_only=True,mmap=True)['tau']
    assert digest(tau)==report['frozen']['tau_digest'] and torch.isfinite(tau).all()
    reference=np.load('runs/v0815_dense_reference/reference.npy')
    assert digest(torch.from_numpy(reference))==report['frozen']['reference_digest']
    assert report['rows'][0]['historical_rgb_relative_l2']<1e-6
    for row in report['rows']:
        image=np.load(CACHE/(row['law']['name']+'.npy'),mmap_mode='r')
        assert image.shape==(4,1080,1920,3) and np.isfinite(image).all()
        assert sum(b['count'] for b in row['blocking_bins'])==4*16777216
        assert abs(sum(b['source_mass'] for b in row['blocking_bins'])-4*report['frozen']['source_mass'])<1e-10
        expected=sum(b['detected_rgb_energy'] for b in row['blocking_bins'])
        assert abs(expected-row['detected_rgb'])/expected<1e-5
        assert max(a['relative_energy_difference'] for a in row['energy_accounting'])<1e-5
    regression=report['geometry_closure']['regression']; assert regression
    assert max(r['cached_tau_max_abs'] for r in regression)<1e-8
    assert max(r['micro_mean_max_abs'] for r in regression)<1e-12
    assert len(report['geometry_closure']['results'])==6
    # A failed derivative gate is a valid experiment outcome, never hide it.
    for r in report['geometry_closure']['results']:
        assert len(r['rows'])==4 and all(v is None or np.isfinite(v) for row in r['rows'] for v in row.values())
        if r.get('null_analytic_response',False): assert not r['passed']
    for path in report['figures']:
        with Image.open(path) as image: image.load()
    with Path('artifacts/v0817_collision_transfer.csv').open() as stream: records=list(csv.DictReader(stream))
    assert len(records)>100
    manifest=json.loads((CACHE/'historical_hashes.json').read_text())
    for path,expected in manifest.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    commands={'compileall':[sys.executable,'-m','compileall','-q','src','scripts','tests','demo.py'],
        'demo_verify':[sys.executable,'demo.py','--verify'],
        'unit_tests':[sys.executable,'-m','unittest','discover','-s','tests'],
        'diff_check':['git','diff','--check']}
    logs={}
    for name,command in commands.items():
        result=subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        logs[name]=result.stdout; (CACHE/f'validation_{name}.log').write_text(result.stdout)
    result=dict(strict_json_csv=True,finite_values=True,frozen_digests=True,source_mass=True,
        current_replay=True,micro_tau_regression=True,detector_energy_accounting=True,
        derivative_tests_recorded=6,historical_files_unchanged=len(manifest),figures_decoded=len(report['figures']),
        compileall=True,demo_verify=True,unit_tests=int(re.search(r'Ran (\d+) tests',logs['unit_tests']).group(1)),git_diff_check=True)
    write_json(Path('artifacts/v0817_validation.json'),result); print(json.dumps(result,indent=2))


if __name__=='__main__': validate()
