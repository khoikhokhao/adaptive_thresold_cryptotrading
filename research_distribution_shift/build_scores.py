"""Train-only feature-space references, validation scores, causal rolling covariates."""
from pathlib import Path
import json,time,sys,gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_volatility_threshold'))
from method_a import sha,write_json,memory
OUT=ROOT/'research_distribution_shift/validation_20261103'
SCORES=['md128_l01','md128_l10','pca_md16','pca_md32','pca_re32','knn5_pc16','knn20_pc16',
        'psi360_pc3','w1_360_pc1','tail_fraction','pooled_md128','rawspace_md128']

def fit_reference(x,standardize=True):
    x=np.asarray(x,dtype=np.float64)
    mean=x.mean(axis=0);sd=x.std(axis=0,ddof=0)
    scale=np.where(sd>1e-12,sd,1.) if standardize else np.ones(x.shape[1])
    x-=mean;x/=scale
    covariance=x.T@x/(len(x)-1)
    eig,U=np.linalg.eigh(covariance);order=np.argsort(eig)[::-1];eig=eig[order];U=U[:,order]
    mu=np.trace(covariance)/len(eig)
    assert mu>0
    positive=np.maximum(eig,0.)
    lo,hi=np.quantile(x,[.01,.99],axis=0)
    ref=dict(mean=mean,scale=scale,eigenvalues=positive,components=U,mu=mu,lower=lo,upper=hi,
        n=len(x),standardize=standardize,constant_columns=int((sd<=1e-12).sum()),
        numerical_rank=int((eig>eig[0]*1e-10).sum()),min_raw_eigen=float(eig[-1]),max_eigen=float(eig[0]),
        retained16=float(positive[:16].sum()/positive.sum()),retained32=float(positive[:32].sum()/positive.sum()),
        shrinkage_condition01=float(((.99*positive[0]+.01*mu)/(.99*positive[-1]+.01*mu))))
    pc=x@U
    scores=score_z(x,pc,ref)
    ref['train_pc16']=np.ascontiguousarray(pc[:,:16])
    ref['train_pc3']=np.ascontiguousarray(pc[:,:3])
    ref['train_pc1_sorted']=np.sort(pc[:,0])
    ref['pc1_iqr']=float(np.quantile(pc[:,0],.75)-np.quantile(pc[:,0],.25))
    ref['psi_edges']=[np.r_[-np.inf,np.unique(np.quantile(pc[:,j],np.linspace(0,1,11)))[1:-1],np.inf] for j in range(3)]
    ref['psi_train_counts']=[np.histogram(pc[:,j],bins=ref['psi_edges'][j])[0] for j in range(3)]
    ref['train_quantiles']={name:dict(q50=float(np.median(v)),q95=float(np.quantile(v,.95)),q99=float(np.quantile(v,.99))) for name,v in scores.items()}
    return ref

def score_z(z,pc,ref):
    e=ref['eigenvalues'];mu=ref['mu'];den=.99*e+.01*mu
    return dict(md128_l01=np.sqrt((pc**2/den).sum(axis=1)),
        md128_l10=np.sqrt((pc**2/(.9*e+.1*mu)).sum(axis=1)),
        pca_md16=np.sqrt((pc[:,:16]**2/den[:16]).sum(axis=1)),
        pca_md32=np.sqrt((pc[:,:32]**2/den[:32]).sum(axis=1)),
        pca_re32=np.sqrt((pc[:,32:]**2).sum(axis=1)/(z.shape[1]-32)),
        tail_fraction=((z<ref['lower'])|(z>ref['upper'])).mean(axis=1))

def point_scores(x,ref):
    z=(np.asarray(x,dtype=float)-ref['mean'])/ref['scale'];pc=z@ref['components']
    return score_z(z,pc,ref),pc

def exact_w1_windows(history,positions,reference,window=360):
    """Exact empirical one-dimensional W1 via integrated training quantile function."""
    x=reference;N=len(x);prefix=np.r_[0.,np.cumsum(x)]/N
    a=np.arange(window)/window;b=np.arange(1,window+1)/window
    def H(u):
        loc=u*N;idx=np.floor(loc).astype(np.int64);idx=np.minimum(idx,N)
        return prefix[idx]+(loc-idx)/N*x[np.minimum(idx,N-1)]
    ha,hb=H(a),H(b)
    view=np.lib.stride_tricks.sliding_window_view(history,window)
    result=np.full(len(positions),np.nan)
    for start in range(0,len(positions),512):
        ix=positions[start:start+512];valid=ix>=window-1
        if not valid.any():continue
        values=np.sort(view[ix[valid]-window+1],axis=1)
        cdf=np.searchsorted(x,values,side='right')/N
        t=np.minimum(np.maximum(cdf,a),b)
        d=(ha+hb-2*H(t)+values*(2*t-a-b)).sum(axis=1)
        assert np.nanmin(d)>-1e-7
        dest=np.arange(start,start+len(ix))[valid];result[dest]=np.maximum(d,0.)
    return result

def main():
    begun=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (OUT/'resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-begun,**m))+'\n')
        print(stage,flush=True)
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    event('start')
    prep=ROOT/'data/btc_eth_1pct/prepared';source=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    previous=ROOT/'research_autocorrelation_analysis/validation_20261103/validation_features.parquet'
    protected=[prep/'X_train.npy',prep/'X_validation.npy',prep/'imputer.joblib',prep/'scaler.joblib',prep/'feature_schema.json',
        prep/'train_metadata.parquet',prep/'validation_metadata.parquet',source,vp,previous,
        ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py',ROOT/'data/btc_eth_1pct/training/classifier_complete.joblib']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    manifest=json.loads((prep/'matrix_manifest.json').read_text())
    for name in ['X_train.npy','X_validation.npy']:assert sha(prep/name)==manifest[name]['sha256']
    trainmeta=pd.read_parquet(prep/'train_metadata.parquet',columns=['symbol','timestamp']).reset_index(drop=True)
    valmeta=pd.read_parquet(prep/'validation_metadata.parquet',columns=['symbol','timestamp']).reset_index(drop=True)
    validation=pd.read_parquet(vp,columns=['symbol','timestamp','confidence'])
    pd.testing.assert_frame_equal(valmeta.reset_index(drop=True),validation[['symbol','timestamp']].reset_index(drop=True))
    assert not pd.MultiIndex.from_frame(trainmeta).isin(pd.MultiIndex.from_frame(valmeta)).any()
    Xtr=np.load(prep/'X_train.npy',mmap_mode='r');Xv=np.load(prep/'X_validation.npy',mmap_mode='r')
    assert Xtr.shape==(292338,128) and Xv.shape==(80260,128) and Xtr.dtype==Xv.dtype==np.float32
    names=json.loads((prep/'feature_schema.json').read_text())['feature_names'];assert len(names)==128
    write_json(OUT/'protocol.json',dict(scores=SCORES,n_train=len(Xtr),n_validation=len(Xv),features=names,
        reference_fit='training features and symbol only; no training labels',
        primary_reference='per symbol; full training samples',analysis_scaling='train mean/std per symbol in MLP-scaled space; constants scale=1',
        covariance_shrinkage=[.01,.10],PCA_dimensions=[16,32],knn_k=[5,20],knn_exact=True,
        knn_reference_subsampled=False,training_anchor_diagnostic_n=2048,rolling_window_rows=360,
        PSI_bins='training quantiles 10 bins per first 3 PCs; 0.5 pseudocount',
        Wasserstein='exact W1 of rolling PC1 against full training PC1, normalized by training IQR',inputs_sha256=hashes))
    scores=pd.DataFrame(index=valmeta.index,columns=SCORES,dtype=float);refs={};audit=[];loadings=[]
    for symbol in ['BTC_USDT','ETH_USDT']:
        ti=np.flatnonzero((trainmeta.symbol==symbol).to_numpy());vi=np.flatnonzero((valmeta.symbol==symbol).to_numpy())
        assert trainmeta.iloc[ti].timestamp.max()<valmeta.iloc[vi].timestamp.min()
        ref=fit_reference(np.asarray(Xtr[ti],dtype=float));refs[symbol]=ref
        point,pc=point_scores(Xv[vi],ref)
        for name,value in point.items():scores.loc[vi,name]=value
        tree=cKDTree(ref['train_pc16'])
        for start in range(0,len(vi),2048):
            ix=vi[start:start+2048];dist,_=tree.query(pc[start:start+len(ix),:16],k=20,eps=0.,workers=2)
            scores.loc[ix,'knn5_pc16']=dist[:,:5].mean(axis=1);scores.loc[ix,'knn20_pc16']=dist.mean(axis=1)
        anchors=np.linspace(0,len(ti)-1,min(2048,len(ti))).astype(int)
        d,_=tree.query(ref['train_pc16'][anchors],k=21,eps=0.,workers=2)
        for name,value in [('knn5_pc16',d[:,1:6].mean(axis=1)),('knn20_pc16',d[:,1:21].mean(axis=1))]:
            ref['train_quantiles'][name]=dict(q50=float(np.median(value)),q95=float(np.quantile(value,.95)),q99=float(np.quantile(value,.99)),anchor_n=len(anchors))
        for j in range(3):
            for k in np.argsort(np.abs(ref['components'][:,j]))[-8:][::-1]:loadings.append(dict(symbol=symbol,PC=j+1,feature=names[k],loading=ref['components'][k,j]))
        audit.append(dict(symbol=symbol,train_n=len(ti),val_n=len(vi),train_start=str(trainmeta.iloc[ti].timestamp.min()),
            train_end=str(trainmeta.iloc[ti].timestamp.max()),val_start=str(valmeta.iloc[vi].timestamp.min()),
            constant_columns=ref['constant_columns'],numerical_rank=ref['numerical_rank'],min_eigen=ref['min_raw_eigen'],
            max_eigen=ref['max_eigen'],retained16=ref['retained16'],retained32=ref['retained32'],regularized_condition01=ref['shrinkage_condition01']))
        raw=fit_reference(np.asarray(Xtr[ti],dtype=float),standardize=False)
        rawpoint,_=point_scores(Xv[vi],raw);scores.loc[vi,'rawspace_md128']=rawpoint['md128_l01']
        ref['rawspace_reference']=dict(mean=raw['mean'],scale=raw['scale'],components=raw['components'],eigenvalues=raw['eigenvalues'],mu=raw['mu'])
        ref['train_quantiles']['rawspace_md128']=raw['train_quantiles']['md128_l01']
        del tree,pc,raw,rawpoint;gc.collect();event(symbol+'_train_reference_and_exact_knn_complete')
    pooled=fit_reference(np.array(Xtr,dtype=float))
    for start in range(0,len(Xv),8192):
        a,_=point_scores(Xv[start:start+8192],pooled);scores.loc[start:start+len(a['md128_l01'])-1,'pooled_md128']=a['md128_l01']
    pooled_q=pooled['train_quantiles']['md128_l01']
    pooled_meta=dict(train_end=str(trainmeta.timestamp.max()),early_val_rows=int((valmeta.timestamp<=trainmeta.timestamp.max()).sum()),
        warning='Pooled training reference contains ETH observations later than some early BTC validation timestamps; sensitivity only')
    del pooled;gc.collect();event('pooled_sensitivity_complete')
    # Build full-stream first-3-PC coordinates with the ORIGINAL imputer/scaler, in bounded batches.
    imputer=joblib.load(prep/'imputer.joblib');scaler=joblib.load(prep/'scaler.joblib')
    trainpos=pd.read_parquet(prep/'train_mapping.parquet',columns=['_source_row'])._source_row.to_numpy()
    valpos=pd.read_parquet(prep/'validation_mapping.parquet',columns=['_source_row'])._source_row.to_numpy()
    chunks=[];offset=0;verified_train=0;verified_val=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192,columns=['symbol','timestamp']+names):
        g=batch.to_pandas();count=len(g)
        raw=g[names].replace([np.inf,-np.inf],np.nan)
        x=scaler.transform(imputer.transform(raw)).astype(np.float32)
        assert np.isfinite(x).all()
        for positions,cache,label in [(trainpos,Xtr,'train'),(valpos,Xv,'validation')]:
            ix=np.flatnonzero((positions>=offset)&(positions<offset+count))
            if len(ix):
                np.testing.assert_array_equal(x[positions[ix]-offset],cache[ix])
                if label=='train':verified_train+=len(ix)
                else:verified_val+=len(ix)
        pc=np.zeros((count,3))
        for symbol,r in refs.items():
            sm=(g.symbol==symbol).to_numpy();z=(x[sm].astype(float)-r['mean'])/r['scale'];pc[sm]=z@r['components'][:,:3]
        a=g[['symbol','timestamp']].copy();a['source_row']=np.arange(offset,offset+count)
        for j in range(3):a[f'PC{j+1}']=pc[:,j]
        chunks.append(a);offset+=count
    assert verified_train==len(Xtr) and verified_val==len(Xv)
    stream=pd.concat(chunks,ignore_index=True).sort_values(['symbol','timestamp']).reset_index(drop=True)
    del chunks,x,g,raw;gc.collect();event('full_stream_transform_exact_training_validation_parity_PASS')
    windows=[]
    for symbol,g in stream.groupby('symbol',sort=True):
        g=g.reset_index(drop=True);ref=refs[symbol];N=ref['n'];psi=np.zeros(len(g))
        for j in range(3):
            edges=ref['psi_edges'][j];counts=ref['psi_train_counts'][j];B=len(counts)
            p=(counts+.5)/(N+.5*B);idx=np.searchsorted(edges[1:-1],g[f'PC{j+1}'],side='right')
            value=np.zeros(len(g));value[:359]=np.nan
            for b in range(B):
                c=pd.Series((idx==b).astype(float)).rolling(360,min_periods=360).sum().to_numpy()
                q=(c+.5)/(360+.5*B);value+=(q-p[b])*np.log(q/p[b])
            psi+=value/3
            # Direct histogram check on an actual trailing window.
            t=len(g)//2;hist=np.bincount(idx[t-359:t+1],minlength=B);q=(hist+.5)/(360+.5*B)
            assert np.isclose(value[t],np.sum((q-p)*np.log(q/p)),atol=1e-12)
        g['psi360_pc3']=psi
        g['stream_position']=np.arange(len(g))
        lookup=g.set_index(['symbol','timestamp']).stream_position
        sm=(valmeta.symbol==symbol).to_numpy();vi=np.flatnonzero(sm)
        positions=lookup.reindex(pd.MultiIndex.from_frame(valmeta.iloc[vi])).to_numpy(dtype=int)
        w1=exact_w1_windows(g.PC1.to_numpy(),positions,ref['train_pc1_sorted'])
        assert ref['pc1_iqr']>0
        scores.loc[vi,'w1_360_pc1']=w1/ref['pc1_iqr'];scores.loc[vi,'psi360_pc3']=g.psi360_pc3.to_numpy()[positions]
        for t in positions[[0,len(positions)//2,-1]]:
            direct=wasserstein_distance(ref['train_pc1_sorted'],g.PC1.to_numpy()[t-359:t+1])
            actual=exact_w1_windows(g.PC1.to_numpy(),np.array([t]),ref['train_pc1_sorted'])[0]
            assert np.isclose(direct,actual,atol=1e-9,rtol=1e-9)
        training_rows=np.flatnonzero(g.source_row.isin(trainpos[trainmeta.symbol==symbol]).to_numpy())
        eligible=training_rows[training_rows>=359];anchors=eligible[np.linspace(0,len(eligible)-1,min(2048,len(eligible))).astype(int)]
        tw=exact_w1_windows(g.PC1.to_numpy(),anchors,ref['train_pc1_sorted'])/ref['pc1_iqr']
        tp=g.psi360_pc3.to_numpy()[training_rows];tp=tp[np.isfinite(tp)]
        for name,value in [('w1_360_pc1',tw),('psi360_pc3',tp)]:
            ref['train_quantiles'][name]=dict(q50=float(np.median(value)),q95=float(np.quantile(value,.95)),q99=float(np.quantile(value,.99)),diagnostic_n=len(value))
        windows.append(g[['symbol','timestamp','source_row','PC1','PC2','PC3','psi360_pc3']])
        event(symbol+'_rolling_PSI_and_exact_Wasserstein_PASS')
    assert np.isfinite(scores[SCORES]).all().all()
    values=valmeta.join(scores)
    context=pd.read_parquet(previous)
    df=context.merge(values,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True)
    assert len(df)==80260 and (df._merge=='both').all()
    assert df.selected.sum()==20463
    df.drop(columns='_merge').to_parquet(OUT/'validation_scores.parquet',index=False)
    refrows=[]
    for symbol,r in refs.items():
        r['train_quantiles']['pooled_md128']=pooled_q
        for score,q in r['train_quantiles'].items():refrows.append(dict(symbol=symbol,score=score,**q))
    pd.DataFrame(refrows).to_csv(OUT/'training_score_quantiles.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'reference_audit.csv',index=False)
    pd.DataFrame(loadings).to_csv(OUT/'PCA_loadings.csv',index=False)
    joblib.dump(refs,OUT/'training_references.joblib')
    pd.concat(windows,ignore_index=True).to_parquet(OUT/'past_PC_stream.parquet',index=False)
    for path in protected:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    event('complete_all_inputs_unchanged')
    write_json(OUT/'score_status.json',dict(n_train=len(Xtr),n_validation=len(df),scores=SCORES,
        training_transform_rows_verified=verified_train,validation_transform_rows_verified=verified_val,
        exact_transform_equivalence='PASS',direct_PSI_and_exact_W1_checks='PASS',pooled_reference_audit=pooled_meta,
        elapsed_sec=time.monotonic()-begun,memory=memory(),test_predictions_read=False,
        training_labels_used=False,MLP_retrained=False,inputs_unchanged=True))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
