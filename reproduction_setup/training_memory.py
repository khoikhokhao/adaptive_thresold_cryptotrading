"""Disk-backed preparation preserving native selection, imputation and scaling.

The MI sampling function is compiled verbatim from native source. Casting before
sampling is performed on one train column at a time, in the original order.
"""
import argparse
import ast
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'baseline_reproduction'))
import centralized_training_two_classes as native
from join_profile import Profiler
from common import sha256

META=['symbol','timestamp','mid_price','future_mid','ret','ret_bps','y','y_binary','y_direction']


def config(path,folder,cap):
    return native.TrainingConfig(unified_data_path=path,output_dir=folder,task='classification',
        horizon_min=600,deadband_bps=100.,model_type='mlp',epochs=20,batch_size=1024,learning_rate=.001,
        validation_split=.15,test_split=.15,mlp_hidden_sizes=[256,128,64],two_class_mode=True,
        use_two_stage=False,confidence_tau=.8,profit_cost_bps=1.,max_train_size=cap,
        top_k_features=128,use_robust_scaler=True,feature_selection_method='mutual_info',
        calibrate_probabilities=True,optimize_threshold_for_profit=True,confidence_grid_start=.5,
        confidence_grid_end=.94,confidence_grid_step=.02,optimize_tau_by='profit',seed=42,device='cpu')


def frame_info(frame):
    # Logical sizes are NOT additive: multiple names can share block buffers.
    return dict(shape=list(frame.shape),object_id=id(frame),
        blocks=[dict(dtype=str(b.dtype),shape=list(b.values.shape),buffer=int(b.values.__array_interface__['data'][0]))
                for b in frame._mgr.blocks if hasattr(b.values,'__array_interface__')])


class PreparationTrace:
    def __init__(self,folder,profile):
        self.folder=folder
        self.profile=profile
        self.previous={}
        methods=[native.TargetEncoder.create_targets,native.TemporalSplitter.split_data_symbol_wise,
            native.TemporalSplitter.filter_trade_only_samples,native.TemporalSplitter.cap_train_size,
            native.FeatureProcessor.fit_feature_list,native.FeatureProcessor.fit_preprocessors,
            native.FeatureProcessor.transform,native.CentralizedTrainer._prepare_features]
        self.codes={f.__code__:f.__name__ for f in methods}
        self.captured=False

    def __call__(self,frame,event,arg):
        if frame.f_code not in self.codes:
            return None
        if event in ['line','return']:
            key=id(frame)
            before=self.previous.get(key)
            loc=frame.f_locals
            frames={k:frame_info(v) for k,v in loc.items() if isinstance(v,pd.DataFrame)}
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes,object_id=id(v),
                          buffer=int(v.__array_interface__['data'][0]))
                    for k,v in loc.items() if isinstance(v,np.ndarray)}
            self.profile.event('preparation_line',function=self.codes[frame.f_code],
                completed_line=before,next_line=frame.f_lineno,frames=frames,arrays=arrays,
                n_max_mi=loc.get('n_max_mi'),
                retained_train=frame_info(loc['self'].train_df_capped) if isinstance(getattr(loc.get('self'),'train_df_capped',None),pd.DataFrame) else None)
            if frame.f_code==native.FeatureProcessor.fit_feature_list.__code__ and 'selector' in loc and 'X_sample' in loc and not self.captured:
                np.save(self.folder/'mi_positions.npy',loc['pos'])
                np.save(self.folder/'mi_input.npy',loc['X_sample'].to_numpy())
                np.save(self.folder/'mi_y.npy',loc['y_sample'])
                (self.folder/'mi_columns.json').write_text(json.dumps(list(loc['X_sample'].columns)),encoding='utf-8')
                loc['X_sample'].to_parquet(self.folder/'mi_sample.parquet',index=False)
                self.captured=True
            self.previous[key]=frame.f_lineno
            if event=='return':
                self.previous.pop(key,None)
        return self


def save_frame(frame,path):
    # Preserve actual split indices (including gaps after the oracle filter).
    frame[META].to_parquet(path,index=True)


def reference(path,folder,cap,profile):
    trainer=native.CentralizedTrainer(config(path,folder,cap))
    tracer=PreparationTrace(folder,profile)
    sys.settrace(tracer)
    try:
        data=trainer._load_data()
        labeled=trainer._create_targets(data)
        parts=trainer._split_data(labeled)
        for name,frame in zip(['train_uncapped','validation','test'],parts):
            save_frame(frame,folder/f'{name}_metadata.parquet')
        arrays=trainer._prepare_features(*parts)
        save_frame(trainer.train_df_capped,folder/'train_metadata.parquet')
    finally:
        sys.settrace(None)
    for name,X,y in zip(['train','validation','test'],arrays[::2],arrays[1::2]):
        np.save(folder/f'X_{name}.npy',X)
        np.save(folder/f'y_{name}.npy',y)
    joblib.dump(trainer.feature_processor,folder/'feature_processor.joblib')
    np.save(folder/'mi_scores.npy',trainer.feature_processor.feature_selector.scores_)
    profile.event('reference_preparation_complete',regions=True)


def sample_positions(metadata):
    tree=ast.parse((ROOT/'centralized_training_two_classes.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureProcessor')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit_feature_list')
    func=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='sample_positions_stratified')
    namespace=dict(vars(native),rng=np.random.default_rng(42),df_train=metadata,
                   X_all=pd.DataFrame(index=metadata.index))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),
                 '<native sampling unchanged>','exec'),namespace)
    return namespace[func.name](min(100000,len(metadata)))


def load_train_column(path,col,metadata):
    raw=pd.read_parquet(path,columns=[col])[col]
    selected=raw.iloc[metadata['_source_row'].to_numpy()].copy()
    selected.index=metadata.index
    return selected


def optimized(path,folder,cap,profile):
    key=pd.read_parquet(path,columns=['symbol','timestamp','mid_price'])
    key['_source_row']=np.arange(len(key),dtype=np.int64)
    order=key.sort_values(['symbol','timestamp']).index.to_numpy()
    if not np.array_equal(order,np.arange(len(key))):
        raise RuntimeError('Expected validated source row order')
    profile.event('label_start',regions=True)
    encoder=native.TargetEncoder(600,100.,'classification')
    traced=PreparationTrace(folder,profile)
    sys.settrace(traced)
    try:
        labeled=encoder.create_targets(key)
        split=native.TemporalSplitter(.15,.15)
        parts=split.filter_trade_only_samples(*split.split_data_symbol_wise(labeled))
        train=split.cap_train_size(parts[0],cap=cap)
    finally:
        sys.settrace(None)
    for name,frame in zip(['train_uncapped','validation','test'],parts):
        save_frame(frame,folder/f'{name}_metadata.parquet')
    save_frame(train,folder/'train_metadata.parquet')
    np.save(folder/'train_source_rows.npy',train['_source_row'].to_numpy())
    metas=[train,parts[1],parts[2]]
    for name,frame in zip(['train','validation','test'],metas):
        frame.to_parquet(folder/f'{name}_mapping.parquet',index=True)
    del key,labeled,parts,train,frame,metas,order
    gc.collect()
    train=pd.read_parquet(folder/'train_mapping.parquet')
    profile.event('labels_splits_released',train_rows=len(train),regions=True)
    processor=native.FeatureProcessor(top_k_features=128,use_robust_scaler=True,feature_selection_method='mutual_info')
    pos=sample_positions(train)
    np.save(folder/'mi_positions.npy',pos)
    sample={}
    valid=[]
    # The source selects np.number columns, excluding bool/category/object.
    schema=pq.read_schema(path)
    for col in schema.names:
        if col in processor.exclude_cols:
            continue
        series=load_train_column(path,col,train)
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            continue
        if series.notna().sum()==0 or series.nunique(dropna=True)<=1:
            continue
        valid.append(col)
        # Native: replace on all training data, float32-cast floats, then sample,
        # replace again, and fill using the SAMPLE mean. Never use full means.
        series=series.replace([np.inf,-np.inf],np.nan)
        if pd.api.types.is_float_dtype(series.dtype):
            series=series.astype(np.float32)
        sample[col]=series.iloc[pos].replace([np.inf,-np.inf],np.nan)
    X_sample=pd.DataFrame(sample,copy=False)
    X_sample=X_sample.fillna(X_sample.mean()).astype(np.float32)
    y_sample=np.asarray(train.y_direction.iloc[pos]).ravel()
    np.save(folder/'mi_input.npy',X_sample.to_numpy())
    np.save(folder/'mi_y.npy',y_sample)
    (folder/'mi_columns.json').write_text(json.dumps(valid),encoding='utf-8')
    X_sample.to_parquet(folder/'mi_sample.parquet',index=False)
    distribution=train.symbol.iloc[pos].value_counts().to_dict()
    profile.event('MI_actual_input',rows=len(X_sample),columns=len(valid),
                  dtype=str(X_sample.to_numpy().dtype),symbol_counts=distribution,
                  unique_positions=int(len(np.unique(pos))),regions=True)
    if len(valid)<=128 or len(np.unique(y_sample))!=2:
        raise RuntimeError('Unsupported feature-selection branch; no silent fallback')
    selector=native.SelectKBest(score_func=native.mi_score_func,k=128)
    selector.fit(X_sample,y_sample)
    processor.feature_selector=selector
    processor.feature_names=sorted(X_sample.columns[selector.get_support()].tolist())
    np.save(folder/'mi_scores.npy',selector.scores_)
    del X_sample,sample,series,train,y_sample,pos
    gc.collect()
    profile.event('MI_complete',features=processor.feature_names,regions=True)
    def matrix(metadata):
        return pd.DataFrame({c:load_train_column(path,c,metadata) for c in processor.feature_names},copy=False)
    train=pd.read_parquet(folder/'train_mapping.parquet')
    X=matrix(train)
    sys.settrace(PreparationTrace(folder,profile))
    try:
        processor.fit_preprocessors(X)
    finally:
        sys.settrace(None)
    del X,train
    gc.collect()
    joblib.dump(processor,folder/'feature_processor.joblib')
    processor.save_artifacts(folder)
    profile.event('imputer_scaler_fit_complete',regions=True)
    for name in ['train','validation','test']:
        metadata=pd.read_parquet(folder/f'{name}_mapping.parquet')
        X=matrix(metadata)
        profile.event('transform_start',split=name,rows=len(X),regions=True)
        sys.settrace(PreparationTrace(folder,profile))
        try:
            transformed=processor.transform(X)
        finally:
            sys.settrace(None)
        if not np.isfinite(transformed).all():
            raise RuntimeError('Nonfinite transformed matrix; no automatic repair')
        np.save(folder/f'X_{name}.npy',transformed)
        np.save(folder/f'y_{name}.npy',metadata.y_direction.to_numpy())
        profile.event('transform_complete',split=name,shape=list(transformed.shape),dtype=str(transformed.dtype),
                      missing=int(np.isnan(transformed).sum()),C_order=bool(transformed.flags.c_contiguous),
                      F_order=bool(transformed.flags.f_contiguous),regions=True)
        del X,transformed,metadata
        gc.collect()
    profile.event('optimized_preparation_complete',regions=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['reference','optimized'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-train-size',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite or automatic retry')
    if args.mode=='reference' and 'smoke' not in str(args.input):
        raise RuntimeError('Wide native reference restricted to smoke')
    args.output.mkdir(parents=True)
    np.random.seed(42)
    profile=Profiler(args.output)
    profile.event('entry',regions=True)
    cfg=config(args.input,args.output,args.max_train_size)
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},indent=2),encoding='utf-8')
    (reference if args.mode=='reference' else optimized)(args.input,args.output,args.max_train_size,profile)
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in args.output.glob('*.npy')}
    (args.output/'matrix_manifest.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
    tracemalloc.stop()
    print(json.dumps(dict(mode=args.mode,passed=True,artifacts=list(artifacts))),flush=True)


if __name__=='__main__':
    main()
