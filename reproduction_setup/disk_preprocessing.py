"""Disk-backed aggregation variant; author source is never patched."""
import argparse
import ctypes
from ctypes import wintypes
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc
import pandas as pd
from memory_preprocessing import ROOT, StorageMicro, config_for, native
from run_observed import MemoryCounters
sys.path.insert(0, str(ROOT/'baseline_reproduction'))
from common import memory

API = ctypes.WinDLL('psapi').GetProcessMemoryInfo
API.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
API.restype = wintypes.BOOL
KERNEL = ctypes.WinDLL('kernel32')
KERNEL.GetCurrentProcess.restype = wintypes.HANDLE


class DiskMicro(StorageMicro):
    profile_previous = False

    def event(self, stage, **extra):
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not API(KERNEL.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        current, peak = tracemalloc.get_traced_memory()
        record = dict(stage=stage, elapsed=time.monotonic()-self.started,
            working_bytes=counters.WorkingSetSize, peak_working_bytes=counters.PeakWorkingSetSize,
            commit_bytes=counters.PagefileUsage, peak_commit_bytes=counters.PeakPagefileUsage,
            traced_current_bytes=current, traced_peak_since_event_bytes=peak, **memory(), **extra)
        with (self.config.output_dir/'aggregation_profile.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(record)+'\n')
        tracemalloc.reset_peak()

    def _perform_minute_aggregation(self, df):
        self.started = time.monotonic()
        tracemalloc.start()
        self.event('entry', rows=len(df), columns=len(df.columns))
        reference = None
        if self.profile_previous:
            # Smoke only: line-level measurements of the previously failing method.
            code = StorageMicro._perform_minute_aggregation.__code__
            def trace(frame, event, arg):
                if frame.f_code == code and event == 'line':
                    self.event('previous_before_line', line=frame.f_lineno)
                return trace
            sys.settrace(trace)
            try:
                reference = super()._perform_minute_aggregation(df.copy())
            finally:
                sys.settrace(None)
            self.event('previous_complete')
        if self.config.minute_aggregation_method != 'last':
            raise ValueError('Only last aggregation is verified')
        # clean_data already owns its independent copy; the load-stage frame is dead.
        self.data = None
        gc.collect()
        self.event('released_load_stage')
        df['timestamp_minute'] = df.timestamp.dt.floor('T')
        keys = df[['Symbol', 'timestamp', 'timestamp_minute']].sort_values(['Symbol', 'timestamp'])
        columns = [c for c in df if c not in ['Symbol', 'timestamp_minute']]
        folder = self.config.output_dir/'aggregation_blocks'
        folder.mkdir()
        self.event('sorted_keys')
        # Globally sorted key order is retained. Per-symbol groupby avoids a wide
        # sorted copy, with identical last-non-null behavior for every column.
        manifests = []
        for number, (symbol, group) in enumerate(keys.groupby('Symbol', sort=True)):
            order = group.index
            small_keys = group[['Symbol', 'timestamp_minute']]
            paths = []
            for start in range(0, len(columns), 8):
                self.event('before_select', symbol=str(symbol), block=start)
                # Select columns before rows: never reindex the full wide frame.
                block = df[columns[start:start+8]].loc[order]
                self.event('after_select', symbol=str(symbol), block=start)
                block = pd.concat([small_keys, block], axis=1)
                aggregated = block.groupby(['Symbol', 'timestamp_minute']).last().reset_index()
                self.event('after_groupby', symbol=str(symbol), block=start)
                path = folder/f'{number}_{start}.parquet'
                aggregated.to_parquet(path, index=False)
                paths.append(path)
                del block, aggregated
                gc.collect()
                self.event('after_write_release', symbol=str(symbol), block=start)
            manifests.append(paths)
        # Reconstruct each final column once; no list of wide aggregate frames.
        result = {}
        all_columns = ['Symbol', 'timestamp_minute'] + columns
        for col in all_columns:
            owner = 0 if col in ['Symbol', 'timestamp_minute'] else columns.index(col)//8
            result[col] = pd.concat([pd.read_parquet(paths[owner], columns=[col])[col]
                                     for paths in manifests], ignore_index=True)
            self.event('reconstruct_column', column=col)
        result['timestamp'] = result.pop('timestamp_minute')
        final = pd.DataFrame(result, copy=False)
        self.event('aggregation_complete', rows=len(final), columns=len(final.columns))
        if reference is not None:
            pd.testing.assert_frame_equal(reference, final, check_exact=True, rtol=0, atol=0)
            self.event('previous_aggregation_exact_pass')
        tracemalloc.stop()
        return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-previous-smoke', action='store_true')
    args = parser.parse_args()
    if args.profile_previous_smoke and 'smoke' not in args.work.name:
        raise RuntimeError('Previous-method profiling is allowed only on smoke')
    if args.output.exists():
        raise RuntimeError('Refuse overwrite or automatic retry')
    args.output.mkdir(parents=True)
    pipeline = native.FeatureEngineeringPipeline(config_for(args.work, args.output))
    pipeline.micro_processor = DiskMicro(pipeline.config)
    pipeline.micro_processor.profile_previous = args.profile_previous_smoke
    results = pipeline.run_full_pipeline()
    if not {'macro', 'micro', 'unified', 'train', 'validation'} <= results.keys():
        raise RuntimeError('Incomplete native pipeline; inspect report/log')
    print(json.dumps({k:list(v.shape) for k,v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
