"""Paths and evidence utilities for the BTC/ETH baseline; no research implementation."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/baseline_btc_eth_full_20261102"
EVIDENCE = ROOT / "baseline_reproduction/evidence"
PYTHON = ROOT / "reproduction_setup/venv/Scripts/python.exe"
CONFIG = {
    "horizon_min":600,"deadband_bps":10.0,"confidence_tau":0.8,
    "mlp_hidden_sizes":[256,128,64],"learning_rate":0.001,"epochs":80,
    "batch_size_cli":4096,"max_train_size":999999999,
    "validation_split":0.15,"test_split":0.15,"top_k_features":128,
    "feature_selection_method":"mutual_info","use_robust_scaler":True,
    "calibrate_probabilities":True,"optimize_threshold_for_profit":False,
    "early_stopping_patience":10,"profit_cost_bps":1.0,"seed":42,"device":"cpu",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, obj):
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def sha256(path):
    hasher=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class MemoryStatus(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
              ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
              ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
              ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
              ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]


def memory():
    status=MemoryStatus()
    status.dwLength=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {"total_physical_bytes":status.ullTotalPhys,"available_physical_bytes":status.ullAvailPhys,
            "commit_limit_bytes":status.ullTotalPageFile,"available_commit_bytes":status.ullAvailPageFile}
