"""Windows memory counters plus line-level native join profiling."""
import ctypes
from ctypes import wintypes
import json
import sys
import time
import tracemalloc
from disk_preprocessing import API, KERNEL, MemoryCounters, memory


class Performance(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in
        ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
         'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']] + [
        (n,wintypes.DWORD) for n in ['HandleCount','ProcessCount','ThreadCount']]


class Region(ctypes.Structure):
    _fields_=[('BaseAddress',ctypes.c_void_p),('AllocationBase',ctypes.c_void_p),
        ('AllocationProtect',wintypes.DWORD),('PartitionId',wintypes.WORD),
        ('RegionSize',ctypes.c_size_t),('State',wintypes.DWORD),
        ('Protect',wintypes.DWORD),('Type',wintypes.DWORD)]


def system_counters():
    info=Performance()
    info.cb=ctypes.sizeof(info)
    api=ctypes.WinDLL('psapi').GetPerformanceInfo
    api.argtypes=[ctypes.POINTER(Performance),wintypes.DWORD]
    if not api(ctypes.byref(info),info.cb):
        raise ctypes.WinError()
    return {f'system_{n}_bytes':int(getattr(info,n)*info.PageSize) for n in
        ['CommitTotal','CommitLimit','PhysicalTotal','PhysicalAvailable','SystemCache','KernelPaged','KernelNonpaged']}


def mapped_regions():
    query=KERNEL.VirtualQuery
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Region),ctypes.c_size_t]
    query.restype=ctypes.c_size_t
    address=0
    mapped=private=image=0
    while True:
        region=Region()
        if not query(ctypes.c_void_p(address),ctypes.byref(region),ctypes.sizeof(region)):
            break
        if region.State==0x1000:
            if region.Type==0x40000:
                mapped+=region.RegionSize
            elif region.Type==0x20000:
                private+=region.RegionSize
            elif region.Type==0x1000000:
                image+=region.RegionSize
        following=(region.BaseAddress or 0)+region.RegionSize
        if following<=address:
            raise RuntimeError('Invalid VirtualQuery progression')
        address=following
    return dict(mapped_committed_virtual_bytes=mapped,private_committed_virtual_region_bytes=private,
                image_committed_virtual_bytes=image)


class Profiler:
    def __init__(self,folder):
        self.path=folder/'join_profile.jsonl'
        self.started=time.monotonic()
        tracemalloc.start()

    def event(self,stage,regions=False,**extra):
        counters=MemoryCounters()
        counters.cb=ctypes.sizeof(counters)
        if not API(KERNEL.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError()
        current,peak=tracemalloc.get_traced_memory()
        entry=dict(stage=stage,elapsed=time.monotonic()-self.started,
            working_bytes=counters.WorkingSetSize,peak_working_bytes=counters.PeakWorkingSetSize,
            private_commit_bytes=counters.PagefileUsage,peak_private_commit_bytes=counters.PeakPagefileUsage,
            traced_current_bytes=current,traced_peak_bytes=peak,**memory(),**system_counters(),**extra)
        if regions:
            entry.update(mapped_regions())
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(entry)+'\n')
        tracemalloc.reset_peak()

    def native_call(self,method,macro,micro,mode):
        code=method.__func__.__code__
        previous=None
        def trace(frame,event,arg):
            nonlocal previous
            if frame.f_code!=code:
                return None
            if event in ['line','return']:
                self.event('join_line',mode=mode,completed_line=previous,next_line=frame.f_lineno,
                    shapes={k:list(v.shape) for k,v in frame.f_locals.items() if hasattr(v,'columns')})
                previous=frame.f_lineno
            return trace
        sys.settrace(trace)
        try:
            return method(macro,micro)
        finally:
            sys.settrace(None)
