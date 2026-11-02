"""One-shot process-tree monitor; stop this run if headroom is exhausted, never retry."""
import argparse
import ctypes
from ctypes import wintypes
import os
import subprocess
import sys
import time
from common import ROOT,EVIDENCE,memory,save,now

sys.path.insert(0,str(ROOT/"reproduction_setup"))
from run_observed import MemoryCounters,process_tree_memory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--label",required=True)
    parser.add_argument("--cwd",required=True)
    parser.add_argument("--max-working-gib",type=float,required=True)
    parser.add_argument("--min-available-gib",type=float,default=1)
    parser.add_argument("--min-commit-gib",type=float,default=1)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    logfile=EVIDENCE/(args.label+".log")
    if logfile.exists():
        raise RuntimeError("One-shot stage already attempted; refusing automatic rerun")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(EVIDENCE/"matplotlib_cache"),
               OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",OMP_NUM_THREADS="2")
    record={"recorded_at_utc":now(),"command":command,"cwd":args.cwd,"memory_before":memory(),
            "threads":2,"max_working_bytes":int(args.max_working_gib*1024**3),
            "stop_if_free_physical_below_bytes":int(args.min_available_gib*1024**3),
            "stop_if_free_commit_below_bytes":int(args.min_commit_gib*1024**3),"automatic_retries":0}
    initial=memory()
    if initial['available_physical_bytes']<record['stop_if_free_physical_below_bytes'] or initial['available_commit_bytes']<record['stop_if_free_commit_below_bytes']:
        raise RuntimeError('Insufficient headroom before launch; no retry')
    api=ctypes.WinDLL("psapi").GetProcessMemoryInfo
    api.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    api.restype=wintypes.BOOL
    start=time.monotonic()
    peak=commit=0
    minimum_free=memory()["available_physical_bytes"]
    pids_seen=set()
    guard_reason=None
    with logfile.open("xb") as output:
        process=subprocess.Popen(command,cwd=args.cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
        print(f"START {args.label} pid={process.pid}",flush=True)
        with (EVIDENCE/(args.label+"_resources.jsonl")).open("x",encoding="utf-8") as samples:
            last=-10
            while True:
                working,committed,pids=process_tree_memory(process.pid,api)
                snapshot=memory()
                peak=max(peak,working)
                commit=max(commit,committed)
                minimum_free=min(minimum_free,snapshot["available_physical_bytes"])
                pids_seen.update(pids)
                elapsed=time.monotonic()-start
                if elapsed-last>=5:
                    samples.write(__import__("json").dumps({"elapsed_seconds":round(elapsed,1),"working_set_bytes":working,
                        "commit_bytes":committed,**snapshot})+"\n")
                    samples.flush()
                    last=elapsed
                code=process.poll()
                if code is not None:
                    break
                if working>record["max_working_bytes"]:
                    guard_reason="Process tree exceeded working set budget"
                elif snapshot['available_physical_bytes']<record['stop_if_free_physical_below_bytes']:
                    guard_reason=f"Available physical RAM below {args.min_available_gib:g} GiB"
                elif snapshot['available_commit_bytes']<record['stop_if_free_commit_below_bytes']:
                    guard_reason=f"Available commit headroom below {args.min_commit_gib:g} GiB"
                if guard_reason:
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
                    code=process.wait()
                    break
                time.sleep(0.5)
    record.update(exit_code=code,elapsed_seconds=round(time.monotonic()-start,2),peak_working_set_bytes=peak,
        peak_commit_bytes=commit,minimum_available_physical_bytes=minimum_free,guard_reason=guard_reason,
        observed_pids=sorted(pids_seen),memory_after=memory(),
        measurement="0.5s sampled sum of process-tree working sets/commit (launcher + worker); not exact continuous peak")
    save(args.label+"_execution.json",record)
    print(__import__("json").dumps(record,indent=2),flush=True)
    return 2 if guard_reason else code


if __name__=="__main__":
    raise SystemExit(main())
