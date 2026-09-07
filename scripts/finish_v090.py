"""Wait for a specific local experiment, then report and validate its outputs.

Never restarts failed scientific branches, changes parameters, or commits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    cache=root/'runs/v090_matched_birth'; cmdline=Path(f'/proc/{args.pid}/cmdline')
    expected=b'zlt.matched_birth'
    if cmdline.exists() and expected not in cmdline.read_bytes(): raise RuntimeError('PID is not the specified birth experiment')
    status=dict(state='WAITING_FOR_FORMAL_RUN',pid=args.pid,started=time.time(),no_commit_or_push=True)
    def save(): (cache/'completion_monitor.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    save()
    while cmdline.exists() and expected in cmdline.read_bytes():
        status['checked_at']=time.time(); save(); time.sleep(30)
    if not (cache/'run_complete.json').exists():
        status.update(state='FORMAL_RUN_EXITED_WITHOUT_COMPLETION',finished=time.time()); save()
        raise RuntimeError('Formal run exited without completion marker; preserve partial evidence, do not fabricate verdicts')
    for name,command in [('report',[sys.executable,'-m','zlt.matched_report']),('validation',[sys.executable,'scripts/validate_v090.py'])]:
        status['state']='RUNNING_'+name.upper(); save()
        with (cache/('automatic_'+name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state=name.upper()+'_FAILED',returncode=result.returncode,finished=time.time()); save()
            raise RuntimeError(name+' failed; inspect log')
    status.update(state='REPORTED_AND_VALIDATED',finished=time.time()); save()
    print(json.dumps(status),flush=True)


if __name__=='__main__': main()
