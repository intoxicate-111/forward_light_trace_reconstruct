"""Validate v094 and prove no prior experiment file changed."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from zlt.sphere_torus_optimization import CACHE, REPORT, write


def reject(value):
    raise ValueError('nonfinite JSON '+value)


def main():
    checks = {}
    for name, command in [
        ('compileall', [sys.executable, '-m', 'compileall', '-q', 'src', 'scripts', 'tests', 'demo.py']),
        ('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests']),
        ('demo_verify', [sys.executable, 'demo.py', '--verify']),
        ('diff_check', ['git', 'diff', '--check'])]:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (CACHE/(name+'.log')).write_text(result.stdout+result.stderr); checks[name] = result.returncode == 0
    start = json.loads((CACHE/'starting_state.json').read_text())
    checks['historical_files_unchanged'] = all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                                               for p, digest in start['protected_sha256'].items())
    report = json.loads(REPORT.read_text(), parse_constant=reject)
    for path in list(Path('artifacts').glob('v094*.json'))+list(CACHE.glob('*.json')):
        json.loads(path.read_text(), parse_constant=reject)
    checks['strict_json'] = True
    with Path('artifacts/v094_sphere_torus_optimization.csv').open(newline='') as stream:
        rows = list(csv.reader(stream)); checks['csv_parse'] = len(rows) > 10 and all(len(x) == len(rows[0]) for x in rows)
    arrays = list(CACHE.glob('*.npy'))+list(CACHE.glob('*.npz'))
    for path in arrays:
        data = np.load(path)
        if hasattr(data, 'files'):
            for key in data.files: assert np.isfinite(data[key]).all(), (path, key)
        else: assert np.isfinite(data).all(), path
    checks['finite_arrays'] = len(arrays)
    figures = list(Path('figures').glob('v094*.png'))
    for path in figures:
        with Image.open(path) as im: im.load()
    checks['figures_decode'] = len(figures) >= 5
    v = report['verdicts']
    checks['positive_mechanism_results'] = all(v[k] for k in ('CRITICAL_MODE_IMAGE_OBSERVABLE', 'IMAGE_TRIGGER_PREDICTS_CROSSING',
        'COMPLEX_BYPASS_REGULAR', 'COMPLEX_REAL_REENTRY', 'COMPLEX_RUN_REACHES_GENUS1', 'SAME_SIDE_DOES_NOT_TRIGGER',
        'WEAK_VIEW_UNDERDETERMINED'))
    checks['central_falsification_retained'] = (v['REAL_ONLY_DISCRETE_REACHES_GENUS1'] and not v['COMPLEX_BEATS_DISCRETE_REAL']
                                                and v['COMPLEX_MATCHES_DISCRETE_REAL'] and not v['FULL_CLAIM_SUPPORTED'])
    direct, continuous, complex_run = report['runs'][:3]
    checks['identical_real_optimizer_paths'] = ([x.get('accepted_c') for x in direct['trajectory']] ==
                                                [x.get('accepted_c') for x in complex_run['trajectory']])
    checks['complex_beats_only_continuity_baseline'] = (complex_run['final_loss'] < continuous['final_loss'] and
                                                        abs(complex_run['final_loss']-direct['final_loss']) < 1e-12)
    checks['passed'] = all(value is not False for value in checks.values())
    write(Path('artifacts/v094_validation.json'), checks)
    print(json.dumps(checks, indent=2))
    if not checks['passed']: raise RuntimeError('v094 validation failed')


if __name__ == '__main__': main()
