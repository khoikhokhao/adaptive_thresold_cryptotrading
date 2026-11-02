"""Descriptive regime analysis only. No prediction/model fitting or test reads."""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_regime_analysis/validation_20261103'
DEFS={'trend_60':1.,'trend_360':1.,'trend_1440':1.,'efficiency_360':.2,'daily_ma_5_20':.005}
REGIMES=['Bear','Sideways','Bull']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def regime(x,cutoff):
    a=pd.Series('Sideways',index=x.index,dtype='object')
    a[x>cutoff]='Bull';a[x<-cutoff]='Bear';a[~np.isfinite(x)]='Missing'
    return a

def stats(g):
    z=g[g.selected]
    bins=np.minimum((g.confidence.to_numpy()*10).astype(int),9)
    gaps=[]
    for b in np.unique(bins):
        a=g.iloc[np.flatnonzero(bins==b)]
        gaps.append(len(a)/len(g)*abs(a.correct.mean()-a.confidence.mean()))
    return dict(n=len(g),up_n=int((g.y_direction==1).sum()),down_n=int((g.y_direction==0).sum()),
        up_share=(g.y_direction==1).mean(),pred_up_share=(g.pred_direction==1).mean(),
        accuracy=g.correct.mean(),selected_n=len(z),coverage=len(z)/len(g),
        selective_accuracy=z.correct.mean(),net_selected_bps=z.net_bps.mean(),
        expected_net_bps=z.net_bps.sum()/len(g),confidence_mean=g.confidence.mean(),
        confidence_min=g.confidence.min(),confidence_median=g.confidence.median(),
        confidence_p90=g.confidence.quantile(.9),confidence_p95=g.confidence.quantile(.95),
        confidence_max=g.confidence.max(),calibration_gap=g.correct.mean()-g.confidence.mean(),
        ece_10=sum(gaps),selected_days=z.timestamp.dt.floor('D').nunique(),few_selected=len(z)<200)

def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    pred=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [source,pred,sel]}
    tau=json.loads(sel.read_text())['tau']
    p=pd.read_parquet(pred)
    assert len(p)==80260 and not p.duplicated(['symbol','timestamp']).any()
    np.testing.assert_allclose(p.confidence,np.maximum(p.proba_up,p.proba_down),rtol=0,atol=1e-14)
    u=pd.read_parquet(source,columns=['symbol','timestamp','mid_price','volatility_30m','volatility_15d','ma_5d','ma_20d'])
    u=u.sort_values(['symbol','timestamp']).reset_index(drop=True)
    assert not u.duplicated(['symbol','timestamp']).any() and (u.mid_price>0).all()
    frames=[];switches=[];causal=[]
    periods=p.groupby('symbol').timestamp.agg(['min','max'])
    for symbol,x in u.groupby('symbol',sort=True):
        x=x.copy()
        logprice=np.log(x.mid_price)
        r=logprice.diff()
        for w in [60,360,1440]:
            numerator=logprice-logprice.shift(w)
            denom=np.sqrt((r*r).rolling(w,min_periods=w).sum())
            x[f'trend_{w}']=numerator/denom.replace(0,np.nan)
            if w==360:x['efficiency_360']=numerator/r.abs().rolling(w,min_periods=w).sum().replace(0,np.nan)
            x[f'elapsed_{w}_minutes']=(x.timestamp-x.timestamp.shift(w)).dt.total_seconds()/60
        x['daily_ma_5_20']=x.ma_5d/x.ma_20d-1
        # Prefix invariance checks: perturbation/removal of future rows cannot change trailing scores.
        middle=len(x)//2
        prefix=logprice.iloc[:middle]
        for w in [60,360,1440]:
            prefix_score=(prefix-prefix.shift(w))/np.sqrt(prefix.diff().pow(2).rolling(w,min_periods=w).sum()).replace(0,np.nan)
            np.testing.assert_allclose(prefix_score.to_numpy(),x[f'trend_{w}'].iloc[:middle].to_numpy(),equal_nan=True,rtol=0,atol=0)
        causal.append(dict(symbol=symbol,prefix_checks='PASS',full_rows=len(x)))
        for name,cut in DEFS.items():
            x[name+'_regime']=regime(x[name],cut)
            calendar=(x.timestamp>=periods.loc[symbol,'min'])&(x.timestamp<=periods.loc[symbol,'max'])
            a=x.loc[calendar,['timestamp',name+'_regime']].copy()
            state=a[name+'_regime']
            change=state.ne(state.shift())
            runs=change.cumsum()
            sizes=a.groupby(runs).size()
            run_times=a.groupby(runs).timestamp.agg(['min','max'])
            duration=(run_times['max']-run_times['min']).dt.total_seconds()/60+1
            switches.append(dict(definition=name,symbol=symbol,n_calendar_rows=len(a),
                switches=int(change.sum()-1),switch_rate=float((change.sum()-1)/(len(a)-1)),
                median_run_rows=float(sizes.median()),median_run_minutes=float(duration.median()),
                p90_run_rows=float(sizes.quantile(.9)),edge_runs_censored=True))
        frames.append(x[['symbol','timestamp','volatility_30m','volatility_15d']+
            list(DEFS)+[k+'_regime' for k in DEFS]+[f'elapsed_{w}_minutes' for w in [60,360,1440]]])
    del u
    features=pd.concat(frames,ignore_index=True)
    df=p.merge(features,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True)
    assert (df._merge=='both').all()
    df['correct']=(df.pred_direction==df.y_direction).astype(float)
    df['net_bps']=np.where(df.pred_direction==1,df.ret_bps,-df.ret_bps)-1
    df['selected']=df.confidence>=tau
    assert df.selected.sum()==20463
    df['day']=df.timestamp.dt.floor('D')
    df['week']=df.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
    df['vol_bin']=df.groupby('symbol').volatility_30m.transform(lambda x:pd.qcut(x,5,labels=False,duplicates='drop'))
    rows=[];direction=[];weekly=[];reliability=[];effects=[];corrs=[];elapsed=[]
    for symbol,g in df.groupby('symbol'):
        for w in [60,360,1440]:
            z=g[f'elapsed_{w}_minutes']
            elapsed.append(dict(symbol=symbol,window_rows=w,median_minutes=z.median(),p95_minutes=z.quantile(.95),
                                max_minutes=z.max(),share_over_1_5_window=(z>1.5*w).mean()))
    for name in DEFS:
        state=name+'_regime'
        for (symbol,label),g in df.groupby(['symbol',state]):
            denominator=(df.symbol==symbol).sum()
            rows.append(dict(definition=name,symbol=symbol,regime=label,regime_share=len(g)/denominator,**stats(g)))
            for d in [0,1]:
                z=g[g.pred_direction==d]
                if len(z):direction.append(dict(definition=name,symbol=symbol,regime=label,
                    condition='predicted_Down' if d==0 else 'predicted_Up',**stats(z)))
                # Class recall measures performance conditional on actual label, not prediction direction.
                a=g[g.y_direction==d]
                if len(a):direction.append(dict(definition=name,symbol=symbol,regime=label,
                    condition='actual_Down' if d==0 else 'actual_Up',**stats(a)))
            for week,z in g.groupby('week'):
                weekly.append(dict(definition=name,symbol=symbol,regime=label,week=week,**stats(z)))
            bins=np.minimum((g.confidence*10).astype(int),9)
            for b,z in g.groupby(bins):
                reliability.append(dict(definition=name,symbol=symbol,regime=label,bin=int(b),n=len(z),
                    confidence=z.confidence.mean(),accuracy=z.correct.mean()))
        for symbol,g in df.groupby('symbol'):
            z=g[np.isfinite(g[name])&g.volatility_30m.notna()].copy()
            corrs.append(dict(definition=name,symbol=symbol,
                spearman_score_vol30=z[name].corr(z.volatility_30m,method='spearman'),
                spearman_abs_score_vol30=z[name].abs().corr(z.volatility_30m,method='spearman')))
            # Within-confidence fixed-effects linear probability/return model.
            # Dummy coefficients are adjusted Bull/Bear minus Sideways differences, not causal effects.
            z['bull']=(z[state]=='Bull').astype(float);z['bear']=(z[state]=='Bear').astype(float)
            for control,keys in [('confidence',['confidence']),('confidence_vol',['confidence','vol_bin']),
                                 ('confidence_vol_day',['confidence','vol_bin','day'])]:
                cols=['bull','bear','correct','net_bps']
                means=z.groupby(keys,dropna=False)[cols].transform('mean')
                res=z[cols]-means
                X=res[['bull','bear']].to_numpy()
                singular=np.linalg.svd(X,compute_uv=False)
                rank=int((singular>1e-8).sum())
                for outcome in ['correct','net_bps']:
                    coef=np.linalg.lstsq(X,res[outcome].to_numpy(),rcond=1e-10)[0] if rank==2 else [np.nan,np.nan]
                    for label,beta in zip(['Bull','Bear'],coef):
                        effects.append(dict(definition=name,symbol=symbol,controls=control,outcome=outcome,
                            contrast=label+' minus Sideways',effect=beta,n=len(z),rank=rank,
                            groups=z.groupby(keys,dropna=False).ngroups,
                            regime_n=int((z[state]==label).sum()),sideways_n=int((z[state]=='Sideways').sum())))
    tables={'regime_metrics':rows,'direction_metrics':direction,'weekly_metrics':weekly,
            'reliability_bins':reliability,'controlled_effects':effects,'volatility_correlations':corrs,
            'calendar_switching':switches,'lookback_elapsed':elapsed}
    for name,table in tables.items():pd.DataFrame(table).to_csv(OUT/(name+'.csv'),index=False)
    df.drop(columns=['_merge']).to_parquet(OUT/'validation_regime_rows.parquet',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    tab=pd.DataFrame(rows)
    for symbol,g in tab[tab.definition=='trend_360'].groupby('symbol'):
        g=g.set_index('regime').reindex(REGIMES)
        axes[0].plot(REGIMES,100*g.selective_accuracy,marker='o',label=symbol)
        axes[1].plot(REGIMES,100*g.coverage,marker='o',label=symbol)
        axes[2].plot(REGIMES,g.net_selected_bps,marker='o',label=symbol)
    for ax,title,y in zip(axes,['Selective accuracy','Coverage at 0.64','Net / selected sample'],['%','%','bps']):
        ax.set(title=title,ylabel=y);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Validation only; trend 360 rows; oracle-filtered cohort')
    fig.tight_layout();fig.savefig(OUT/'trend_360_performance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rel=pd.DataFrame(reliability)
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        for label in REGIMES:
            z=rel[(rel.definition=='trend_360')&(rel.symbol==symbol)&(rel.regime==label)&(rel.n>=100)]
            ax.plot(z.confidence,z.accuracy,marker='o',label=label)
        ax.plot([.5,1],[.5,1],linestyle='--',color='gray');ax.set(title=symbol,xlabel='Mean confidence',ylabel='Accuracy')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Regime reliability; bins with n >= 100; in-sample calibration')
    fig.tight_layout();fig.savefig(OUT/'reliability_by_regime.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        z=df[df.symbol==symbol]
        q=pd.crosstab(z.day,z.trend_360_regime,normalize='index').reindex(columns=REGIMES,fill_value=0)
        for label in REGIMES:ax.plot(q.index,q[label],label=label)
        ax.set(title=symbol,ylabel='Share in oracle-filtered validation');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'daily_regime_shares.png',dpi=160);plt.close(fig)
    for path in [source,pred,sel]: assert sha(path)==hashes[str(path.relative_to(ROOT))]
    metadata=dict(definitions=DEFS,inputs_sha256=hashes,tau=tau,rows=len(df),elapsed_sec=time.monotonic()-started,
        causal_prefix_checks=causal,test_predictions_read=False,threshold_fit=False,
        missing={name:int((df[name+'_regime']=='Missing').sum()) for name in DEFS},
        baseline_selected=int(df.selected.sum()),baseline_selective_accuracy=float(df.loc[df.selected,'correct'].mean()),
        baseline_net_bps=float(df.loc[df.selected,'net_bps'].mean()))
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
    print(tab[tab.definition=='trend_360'].round(4).to_string(index=False))

if __name__=='__main__':main()
