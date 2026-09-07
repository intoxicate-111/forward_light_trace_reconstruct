"""Pre-commit verification; preserve old scientific files and dirty changes."""
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
from zlt.critical_experiment import CACHE, REPORT, write


def reject(x):
    raise ValueError('nonfinite JSON: '+x)


def main():
    checks = {}
    for name, command in [
        ('compileall', [sys.executable, '-m', 'compileall', '-q', 'src', 'tests', 'scripts', 'demo.py']),
        ('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests']),
        ('demo_verify', [sys.executable, 'demo.py', '--verify']),
        ('git_diff_check', ['git', 'diff', '--check']),
        ('git_staged_diff_check', ['git', 'diff', '--cached', '--check'])]:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (CACHE/(name+'.log')).write_text(completed.stdout+completed.stderr)
        checks[name] = completed.returncode == 0
    before = json.loads((CACHE/'starting_state.json').read_text())
    checks['historical_files_unchanged'] = all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                               for p, h in before['protected_sha256'].items())
    report = json.loads(REPORT.read_text(), parse_constant=reject)
    checks['strict_json'] = True
    for path in list(Path('artifacts').glob('v093*.json'))+list(CACHE.glob('*.json')):
        json.loads(path.read_text(), parse_constant=reject)
    with Path('artifacts/v093_critical_continuation.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
        checks['csv_parse'] = len(rows) == 4 and all(len(x) == len(rows[0]) for x in rows)
    for path in CACHE.glob('*.npy'):
        assert np.isfinite(np.load(path)).all(), path
    for path in CACHE.glob('*.npz'):
        arrays = np.load(path)
        for key in arrays.files: assert np.isfinite(arrays[key]).all(), (path, key)
    checks['finite_arrays'] = True
    figures = list(Path('figures').glob('v093*.png'))
    for path in figures:
        with Image.open(path) as im: im.load()
    checks['figures_decode'] = len(figures) >= 6
    checks['sparse_vs_original_AD'] = report['image']['sparse_vs_original_AD_max_error'] < 1e-12
    checks['image_FD'] = max(x['relative_l2'] for x in report['image']['fd']) < 1e-5
    checks['path_tangent_FD'] = max([report['scalar']['path_tangent_fd_relative_l2']]+
                                    [x['path_tangent_fd_relative_l2'] for x in report['sphere_torus']['branches'].values()]+
                                    [report['image']['candidates'][0]['continuation']['path_tangent_fd_relative_l2']]) < 5e-4
    positive = ['scalar_monodromy', 'analytic_detour_regular', 'meaningful_image_mode', 'real_target_predicts_crossing',
                'same_side_not_forced', 'invisible_mode_underdetermined', 'complex_sample_path_regular',
                'real_reentry', 'actual_retraced_loss_improves', 'endpoint_genus_changed']
    checks['bounded_experiment_acceptance'] = all(report['verdicts'][k] for k in positive)
    checks['negative_results_retained'] = (report['verdicts']['raw_full_GN_step_improves'] is False and
                                            report['verdicts']['every_analytic_branch_returns_real'] is False)
    checks['new_code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path('src/zlt/critical_continuation.py'), Path('src/zlt/critical_experiment.py'),
                                 Path('scripts/report_v093.py'), Path('scripts/validate_v093.py'), Path('tests/test_critical_continuation.py')]}
    checks['passed'] = all(v is not False for v in checks.values())
    write(Path('artifacts/v093_validation.json'), checks)
    print(json.dumps(checks, indent=2))
    if not checks['passed']: raise RuntimeError('v093 validation failed; do not commit')


if __name__ == '__main__': main()
