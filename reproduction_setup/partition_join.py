"""Disk-backed native left join, with dtype decisions deferred to whole columns."""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import sys
import tracemalloc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from memory_preprocessing import ROOT,config_for,native
from join_profile import Profiler
sys.path.insert(0,str(ROOT/'baseline_reproduction'))
from common import sha256


def unoptimized_join():
    # Execute the exact native join AST, deferring ONLY the final dtype pass.
    tree=ast.parse((ROOT/'data_preprocessing.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureEngineeringPipeline')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='create_unified_dataset')
    removed=[]
    for i,node in enumerate(method.body):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='optimize_dtypes':
            removed.append(node.lineno)
            method.body[i]=ast.copy_location(ast.Pass(),node)
    if removed!=[1922]:
        raise RuntimeError('Unexpected native join source; review required')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace=dict(vars(native))
    exec(compile(module,'<partition native join; deferred dtype>', 'exec'),namespace)
    return namespace[method.name]


def verify_micro_before_save(source,profile):
    clean=source/'clean_checkpoint.parquet'
    narrow=source/'narrow_features.parquet'
    micro=source/'processed_data_micro.parquet'
    clean_cols=pq.read_schema(clean).names
    computed=pq.read_schema(narrow).names
    original=['symbol' if c=='Symbol' else c for c in clean_cols]
    expected=original+[c for c in computed if c not in original]
    if pq.read_schema(micro).names!=expected:
        raise RuntimeError('Micro column order differs from pre-save checkpoint')
    for col in expected:
        if col.startswith(('bids[','asks[')):
            frame=pd.read_parquet(clean,columns=[col])
            frame=native.optimize_dtypes(frame,True)
            before=frame[col]
        else:
            before=pd.read_parquet(narrow,columns=[col])[col]
        saved=pd.read_parquet(micro,columns=[col])[col]
        pd.testing.assert_series_equal(before,saved,check_exact=True,rtol=0,atol=0)
    profile.event('micro_saved_equals_before_save_exact',regions=True,columns=len(expected))


def assemble(partitions,folder,profile,rows):
    target=folder/'columns'
    target.mkdir()
    names=pq.read_schema(partitions[0]).names
    if any(pq.read_schema(p).names!=names for p in partitions):
        raise RuntimeError('Partition column order mismatch')
    paths=[]
    fields=[]
    metadata=None
    pandas_columns=[]
    dtype_manifest={}
    for number,col in enumerate(names):
        profile.event('before_global_column',column=col)
        series=pd.concat([pd.read_parquet(p,columns=[col])[col] for p in partitions],ignore_index=True)
        frame=series.to_frame()
        if len(frame)!=rows:
            raise RuntimeError('Partition row loss/expansion')
        frame=native.optimize_dtypes(frame,True,preserve_categories=False)
        path=target/f'{number:03d}.parquet'
        frame.to_parquet(path,index=False,compression='snappy',row_group_size=50000)
        dtype_manifest[col]=dict(dtype=str(frame[col].dtype),missing=int(frame[col].isna().sum()))
        schema=pq.read_schema(path)
        fields.append(schema.field(0))
        meta=json.loads(schema.metadata[b'pandas'])
        if metadata is None:
            metadata=meta
        pandas_columns.extend(meta['columns'])
        paths.append(path)
        del series,frame
        gc.collect()
    metadata['columns']=pandas_columns
    schema=pa.schema(fields,metadata={b'pandas':json.dumps(metadata).encode()})
    partial=folder/'unified_dataset.partial.parquet'
    count=0
    iterators=[pq.ParquetFile(p).iter_batches(batch_size=50000) for p in paths]
    with pq.ParquetWriter(partial,schema,compression='snappy') as writer:
        for row_group in range((rows+49999)//50000):
            profile.event('before_output_row_group',row_group=row_group,regions=True)
            batches=[next(iterator) for iterator in iterators]
            if len({b.num_rows for b in batches})!=1:
                raise RuntimeError('Column batch alignment failed')
            table=pa.Table.from_arrays([b.column(0) for b in batches],schema=schema)
            writer.write_table(table,row_group_size=50000)
            count+=table.num_rows
            del table,batches
        if any(next(it,None) is not None for it in iterators):
            raise RuntimeError('Unexpected extra column rows')
    if count!=rows:
        raise RuntimeError('Output row mismatch')
    del iterators
    partial.rename(folder/'unified_dataset.parquet')
    (folder/'global_dtype_manifest.json').write_text(json.dumps(dtype_manifest,indent=2),encoding='utf-8')
    profile.event('unified_complete',rows=count,columns=len(names),regions=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--macro-before-save',type=Path,required=True)
    parser.add_argument('--profile-wide-smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite or automatic retry')
    if args.profile_wide_smoke and 'smoke' not in args.source.name:
        raise RuntimeError('Wide native profiling only allowed on smoke')
    args.output.mkdir(parents=True)
    profile=Profiler(args.output)
    profile.event('entry',regions=True)
    config=config_for(args.work,args.output)
    manifest=json.loads((args.macro_before_save.parent/'macro_before_save_manifest.json').read_text())
    if sha256(config.macro_csv_path)!=manifest['raw_sha256'] or sha256(args.macro_before_save)!=manifest['checkpoint_sha256']:
        raise RuntimeError('Macro checkpoint identity failed')
    if not args.profile_wide_smoke:
        original=json.loads((ROOT/'baseline_reproduction/evidence/feature_partial_verification.json').read_text())
        if sha256(args.source/'processed_data_micro.parquet')!=original['sha256']:
            raise RuntimeError('Micro checkpoint checksum mismatch')
    verify_micro_before_save(args.source,profile)
    macro=pd.read_parquet(args.macro_before_save)
    if macro.duplicated(['symbol','date']).any():
        raise RuntimeError('Macro keys are not unique; partition strategy needs review')
    for name in ['processed_data_macro.parquet','processed_data_micro.parquet']:
        os.link(args.source/name,args.output/name)
    if args.profile_wide_smoke:
        reference=args.output/'profile_native'
        reference.mkdir()
        pipeline=native.FeatureEngineeringPipeline(config_for(args.work,reference))
        micro=pd.read_parquet(args.source/'processed_data_micro.parquet')
        profile.event('wide_native_start',regions=True)
        result=profile.native_call(pipeline.create_unified_dataset,macro,micro,'wide_native')
        profile.event('wide_native_complete',regions=True)
        del result,micro,pipeline
        gc.collect()
    method=unoptimized_join()
    partitions=[]
    total=0
    last=None
    reader=pq.ParquetFile(args.source/'processed_data_micro.parquet')
    expected=reader.metadata.num_rows
    batch_size=10000 if args.profile_wide_smoke else 50000
    for number,batch in enumerate(reader.iter_batches(batch_size=batch_size)):
        micro=batch.to_pandas()
        keys=list(zip(micro.symbol,micro.timestamp))
        if keys!=sorted(keys) or len(keys)!=len(set(keys)) or (last is not None and keys[0]<=last):
            raise RuntimeError('Micro checkpoint is not globally sorted/unique')
        last=keys[-1]
        folder=args.output/'partitions'/f'{number:03d}'
        folder.mkdir(parents=True)
        pipeline=native.FeatureEngineeringPipeline(config_for(args.work,folder))
        bound=method.__get__(pipeline,type(pipeline))
        profile.event('partition_start',partition=number,rows=len(micro),regions=True)
        result=profile.native_call(bound,macro,micro,'partition_native')
        if len(result)!=len(micro):
            raise RuntimeError('Unexpected join row expansion')
        total+=len(result)
        partitions.append(folder/'unified_dataset.parquet')
        del result,micro,batch,keys,pipeline,bound
        gc.collect()
        profile.event('partition_released',partition=number,regions=True)
    if total!=expected:
        raise RuntimeError('Input rows lost')
    del reader,macro
    gc.collect()
    assemble(partitions,args.output,profile,total)
    output=args.output/'unified_dataset.parquet'
    (args.output/'output_checksum.json').write_text(json.dumps(dict(sha256=sha256(output),bytes=output.stat().st_size,
        rows=total,columns=len(pq.read_schema(output).names)),indent=2),encoding='utf-8')
    tracemalloc.stop()
    print((args.output/'output_checksum.json').read_text(),flush=True)


if __name__=='__main__':
    main()
