"""Run a stage and record actual GPU identity plus sampled device memory without changing it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time


def sample():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        return [dict(uuid=x[0].strip(),name=x[1].strip(),total_mib=float(x[2]),used_mib=float(x[3]))
                for line in result.stdout.splitlines() if len(x:=line.split(','))==4]
    except (OSError,subprocess.SubprocessError,ValueError):
        return []


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('Provide the stage command after --.')
    samples=[]
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            samples.append(dict(elapsed_seconds=time.perf_counter()-started,gpus=sample()))
            stop.wait(.5)
    started=time.perf_counter()
    worker=threading.Thread(target=monitor,daemon=True)
    worker.start()
    code=1
    try:
        code=subprocess.call(command)
    finally:
        stop.set()
        worker.join(timeout=6)
        samples.append(dict(elapsed_seconds=time.perf_counter()-started,gpus=sample()))
        result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),command=command,
            returncode=code,seconds=time.perf_counter()-started,samples=samples,
            sampled_peak_device_used_mib=max((g['used_mib'] for s in samples for g in s['gpus']),default=None),
            limitation='Sampled total device usage includes other processes and can miss brief peaks. '
            'It is not an exact per-process peak or proof of 16-GB compatibility on a larger card.')
        path=Path(args.record)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2))
        print('Measured stage record:',path,flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
