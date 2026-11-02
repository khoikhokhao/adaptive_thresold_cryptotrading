"""Causal trailing features and descriptive statistics on original validation only."""
from pathlib import Path
import json,time,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_volatility_threshold'))
from method_a import sha,write_json,memory
sys.path.insert(0,str(ROOT/'research_regime_analysis'))
from analyze_validation import stats

OUT=ROOT/'research_autocorrelation_analysis/validation_20261103'
ACF=[(w,k) for w in [60,360,1440] for k in [1,5]]+[(360,15)]
VR=[(360,5),(360,15),(1440,5)]
WINDOWS=[60,360,1440]
BASE=[f'acf_{k}_{w}' for w,k in ACF]+[f'vr_{q}_{w}' for w,q in VR]+[f'er_{w}' for w in WINDOWS]
FEATURES=BASE+[f'abs_acf_{k}_{w}' for w,k in ACF]

def rolling_features(price):
    lp=np.log(price.astype(float));r=lp.diff();res=pd.DataFrame(index=price.index)
    for w,k in ACF:
        mean=r.rolling(w,min_periods=w).mean()
        ss=r.pow(2).rolling(w,min_periods=w).sum()-w*mean.pow(2)
        paired=w-k
        prod=(r*r.shift(k)).rolling(paired,min_periods=paired).sum()
        sums=r.rolling(paired,min_periods=paired).sum()+r.shift(k).rolling(paired,min_periods=paired).sum()
        num=prod-mean*sums+paired*mean.pow(2)
        res[f'acf_{k}_{w}']=num/ss.where(ss>0)
    for w,q in VR:
        # All overlapping q-return sums are inside the SAME last w one-step returns.
        block=lp-lp.shift(q)
        res[f'vr_{q}_{w}']=block.rolling(w-q+1,min_periods=w-q+1).var(ddof=1)/(q*r.rolling(w,min_periods=w).var(ddof=1).replace(0,np.nan))
    for w in WINDOWS:
        res[f'er_{w}']=(lp-lp.shift(w)).abs()/r.abs().rolling(w,min_periods=w).sum().replace(0,np.nan)
    return res

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    cached=ROOT/'research_regime_analysis/validation_20261103/validation_regime_rows.parquet'
    vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    protected=[source,cached,vp,sel,ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    write_json(OUT/'feature_protocol.json',dict(acf=ACF,variance_ratio=VR,er_windows=WINDOWS,
        features=FEATURES,near_zero_acf_band=[-.05,.05],inputs_sha256=hashes,
        threshold_fixed=json.loads(sel.read_text())['tau'],no_test_predictions=True))
    u=pd.read_parquet(source,columns=['symbol','timestamp','mid_price']).sort_values(['symbol','timestamp']).reset_index(drop=True)
    assert not u.duplicated(['symbol','timestamp']).any()
    frames=[];checks=[]
    for s,g in u.groupby('symbol',sort=True):
        g=g.copy();f=rolling_features(g.mid_price)
        middle=len(g)//2
        prefix=rolling_features(g.mid_price.iloc[:middle])
        np.testing.assert_allclose(f.iloc[:middle].to_numpy(),prefix.to_numpy(),atol=0,rtol=0,equal_nan=True)
        # Direct finite-window calculations on actual price history validate estimators/alignment.
        lr=np.log(g.mid_price.astype(float)).diff().to_numpy()
        for w,k in ACF:
            for t in [w+10,middle,len(g)-1]:
                a=lr[t-w+1:t+1];a=a-a.mean()
                expected=np.dot(a[k:],a[:-k])/np.dot(a,a)
                assert np.isclose(f[f'acf_{k}_{w}'].iloc[t],expected,atol=1e-10,rtol=1e-8)
        for w,q in VR:
            for t in [w+10,middle,len(g)-1]:
                a=lr[t-w+1:t+1]
                block=np.convolve(a,np.ones(q),mode='valid')
                expected=np.var(block,ddof=1)/(q*np.var(a,ddof=1))
                assert np.isclose(f[f'vr_{q}_{w}'].iloc[t],expected,atol=1e-8,rtol=1e-7)
        part=g[['symbol','timestamp']].copy()
        for col in BASE:part[col]=f[col]
        part['label_endpoint']=g.timestamp.shift(-600)
        for w in WINDOWS:part[f'elapsed_{w}_minutes']=(g.timestamp-g.timestamp.shift(w)).dt.total_seconds()/60
        frames.append(part)
        checks.append(dict(symbol=s,prefix_invariance='PASS',direct_ACF_VR='PASS'))
    del u
    features=pd.concat(frames,ignore_index=True);del frames
    prior=pd.read_parquet(cached)
    df=prior.merge(features,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True,suffixes=('','_recomputed'))
    assert len(df)==80260 and (df._merge=='both').all() and df.label_endpoint.notna().all()
    for w in WINDOWS:
        np.testing.assert_array_equal(df[f'elapsed_{w}_minutes'],df[f'elapsed_{w}_minutes_recomputed'])
    np.testing.assert_allclose(df.er_360,df.efficiency_360.abs(),atol=1e-15,rtol=1e-12)
    for w,k in ACF:df[f'abs_acf_{k}_{w}']=df[f'acf_{k}_{w}'].abs()
    for w,k in ACF:
        x=df[f'acf_{k}_{w}'];assert x.dropna().abs().max()<=1+1e-10
    for w,q in VR:assert (df[f'vr_{q}_{w}'].dropna()>=0).all()
    for w in WINDOWS:assert df[f'er_{w}'].dropna().between(0,1+1e-10).all()
    tau=json.loads(sel.read_text())['tau']
    np.testing.assert_array_equal(df.selected,df.confidence>=tau)
    assert df.selected.sum()==20463
    distributions=[];quantiles=[];directions=[];weekly=[];states=[];correlations=[]
    for name in FEATURES:
        df[name+'_q']=np.nan
        for symbol,z in df.groupby('symbol'):
            x=z[name];finite=np.isfinite(x);valid=z.loc[finite].copy()
            q25,q75=x[finite].quantile([.25,.75]);iqr=q75-q25
            distributions.append(dict(feature=name,symbol=symbol,n=len(x),missing=int((~finite).sum()),
                minimum=x[finite].min(),p01=x[finite].quantile(.01),q25=q25,median=x[finite].median(),q75=q75,
                p99=x[finite].quantile(.99),maximum=x[finite].max(),tukey_outlier_share=((x[finite]<q25-1.5*iqr)|(x[finite]>q75+1.5*iqr)).mean(),
                sign_positive_share=(x[finite]>.05).mean() if name.startswith('acf_') else np.nan,
                sign_negative_share=(x[finite]<-.05).mean() if name.startswith('acf_') else np.nan))
            valid['quantile'],edges=pd.qcut(valid[name],5,labels=False,duplicates='drop',retbins=True)
            df.loc[valid.index,name+'_q']=valid['quantile']+1
            for q,g in valid.groupby('quantile'):
                quantiles.append(dict(feature=name,symbol=symbol,quantile=int(q)+1,lower=edges[int(q)],upper=edges[int(q)+1],**stats(g)))
                for d,h in g.groupby('pred_direction'):
                    directions.append(dict(feature=name,symbol=symbol,quantile=int(q)+1,direction=int(d),**stats(h)))
                for week,h in g.groupby('week'):
                    weekly.append(dict(feature=name,symbol=symbol,quantile=int(q)+1,week=week,feature_median=h[name].median(),**stats(h)))
            for control in ['volatility_30m','volatility_15d','trend_360']:
                correlations.append(dict(feature=name,symbol=symbol,control=control,spearman=z[name].corr(z[control],method='spearman')))
            if name.startswith('acf_'):
                valid['sign_band']=np.where(valid[name]>.05,'positive',np.where(valid[name]<-.05,'negative','near_zero'))
                for band,g in valid.groupby('sign_band'):states.append(dict(feature=name,symbol=symbol,sign_band=band,share=len(g)/len(valid),**stats(g)))
    for name,data in [('distributions',distributions),('quantile_metrics',quantiles),('direction_quantiles',directions),
                      ('weekly_quantiles',weekly),('acf_sign_bands',states),('control_correlations',correlations)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    df.drop(columns=['_merge']).to_parquet(OUT/'validation_features.parquet',index=False)
    mainfeatures=['acf_1_360','abs_acf_1_360','vr_5_360','er_360']
    tab=pd.DataFrame(quantiles)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,name in enumerate(mainfeatures):
        for s,z in tab[tab.feature==name].groupby('symbol'):
            axes[0,j].plot(z['quantile'],100*z.selective_accuracy,marker='o',label=s)
            axes[1,j].plot(z['quantile'],z.net_selected_bps,marker='o',label=s)
        axes[0,j].set(title=name,ylabel='Selective accuracy (%)',xlabel='Within-symbol quantile')
        axes[1,j].set(ylabel='Net bps / selected sample',xlabel='Within-symbol quantile')
        axes[1,j].axhline(0,color='gray',linestyle='--')
    axes[0,0].legend(fontsize=8);fig.suptitle('Validation only; fixed tau=0.64; oracle-filtered')
    fig.tight_layout();fig.savefig(OUT/'quantile_performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,name in zip(axes,mainfeatures):
        for s,z in df.groupby('symbol'):ax.hist(z[name],bins=50,alpha=.5,density=True,label=s)
        ax.set(title=name,ylabel='Density')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'feature_distributions.png',dpi=150);plt.close(fig)
    for path in protected:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    write_json(OUT/'exploration_status.json',dict(rows=len(df),checks=checks,
        missing={f:int((~np.isfinite(df[f])).sum()) for f in FEATURES},elapsed_sec=time.monotonic()-start,
        memory=memory(),test_predictions_read=False,threshold_modified=False,
        input_hashes_unchanged=True,base_feature_count=len(BASE),representations_count=len(FEATURES)))
    print('FEATURE EXPLORATION COMPLETE',len(df),len(FEATURES),flush=True)

if __name__=='__main__':main()
