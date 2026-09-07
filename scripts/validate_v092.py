"""Validate manifold-jet artifacts without modifying previous results."""
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
from zlt.jet_experiment import CACHE, write


def reject(value):
    raise ValueError('Nonfinite JSON: '+value)


def main():
    checks = {}
    for name, command in [
        ('compileall', [sys.executable, '-m', 'compileall', '-q', 'src', 'scripts', 'tests', 'demo.py']),
        ('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests']),
        ('demo_verify', [sys.executable, 'demo.py', '--verify']),
        ('diff_check', ['git', 'diff', '--check'])]:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (CACHE/(name+'.log')).write_text(result.stdout+result.stderr)
        checks[name] = result.returncode == 0
    before = json.loads((CACHE/'starting_state.json').read_text())
    checks['historical_artifacts_and_renderer_unchanged'] = all(
        Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in before['protected_sha256'].items())
    checks['head_unchanged'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == before['head']
    files = list(Path('artifacts').glob('v092*.json'))+list(CACHE.glob('*.json'))
    for path in files:
        json.loads(path.read_text(), parse_constant=reject)
    checks['strict_json_files'] = len(files)
    report = json.loads(Path('artifacts/v092_manifold_jets.json').read_text(), parse_constant=reject)
    checks['fd_six_modes'] = report['fd']['passed'] and len(report['fd']['coefficients']) == 6
    checks['fixed_eight_cases'] = len(report['fixed']) == 8
    checks['birth_three_policies'] = {x['policy'] for x in report['birth']} == {'loss_driven', 'random', 'uniform'}
    checks['equal_birth_budgets'] = all(x['sequence'][-1]['dofs'] == 18 for x in report['birth'])
    checks['disabled_and_saved_image_replay'] = json.loads(Path('artifacts/v092_jet_audit.json').read_text())['passed']
    checks['finite_arrays'] = True
    arrays = list(CACHE.glob('*.npy'))
    for path in arrays:
        array = np.load(path)
        checks['finite_arrays'] &= bool(np.isfinite(array).all() and array.shape == (4, 64, 64, 3))
    checks['array_count'] = len(arrays)
    pictures = list(Path('figures').glob('v092*.png'))
    for path in pictures:
        with Image.open(path) as im:
            im.load()
    checks['figures_decode'] = len(pictures) >= 6
    with Path('artifacts/v092_manifold_jets.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
        checks['csv_parse'] = len(rows) == 18 and all(len(row) == len(rows[0]) for row in rows)
    checks['passed'] = all(value is not False for value in checks.values())
    write(Path('artifacts/v092_validation.json'), checks)
    print(json.dumps(checks, indent=2))
    if not checks['passed']:
        raise RuntimeError('v092 validation failed')


if __name__ == '__main__': main()
