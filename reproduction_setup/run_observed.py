"""Run an explicitly supplied command, capture logs/exit and Windows peak working set."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reproduction_setup/evidence"


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class ProcessEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


def process_tree_memory(root_pid, memory_api):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return 0, 0, []
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    parents = {}
    try:
        has_entry = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            has_entry = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    pids = {root_pid}
    while True:
        expanded = pids | {pid for pid,parent in parents.items() if parent in pids}
        if expanded == pids:
            break
        pids = expanded
    working = committed = 0
    observed = []
    for pid in pids:
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            continue
        try:
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if memory_api(handle, ctypes.byref(counters), counters.cb):
                working += counters.WorkingSetSize
                committed += counters.PagefileUsage
                observed.append(pid)
        finally:
            kernel.CloseHandle(handle)
    return working, committed, observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a command after --")
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                       MPLCONFIGDIR=str(EVIDENCE / "matplotlib_cache"),
                       OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", OMP_NUM_THREADS="2")
    result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "cwd": str(args.cwd.resolve()), "command": command,
              "threads": {key:environment[key] for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS")}}
    started = time.monotonic()
    peak = committed_peak = 0
    observed_pids = set()
    memory_api = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    memory_api.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    memory_api.restype = wintypes.BOOL
    with (EVIDENCE / f"{args.label}.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=args.cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print(f"START {args.label} pid={process.pid}", flush=True)
        while True:
            working, committed, pids = process_tree_memory(process.pid, memory_api)
            peak = max(peak, working)
            committed_peak = max(committed_peak, committed)
            observed_pids.update(pids)
            status = process.poll()
            if status is not None:
                break
            time.sleep(0.5)
    result.update(exit_code=status, elapsed_seconds=round(time.monotonic()-started,2),
                  peak_working_set_bytes=peak, peak_commit_bytes=committed_peak,
                  memory_scope="sum of sampled current working sets/commit across launcher and descendants; 0.5 second sampling",
                  observed_process_ids=sorted(observed_pids),
                  log=str(EVIDENCE / f"{args.label}.log"))
    (EVIDENCE / f"{args.label}_execution.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
