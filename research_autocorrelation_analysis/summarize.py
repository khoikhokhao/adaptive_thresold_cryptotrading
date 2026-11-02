"""Summaries and nonparametric-strata robustness; no new threshold/model selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore import OUT,FEATURES

def main():
    d=pd.read_csv(OUT/'temporal_incremental_metrics.csv')
    rows=[]
    for (feature,scope),g in d.groupby(['feature','scope']):
        r=dict(feature=feature,scope=scope,n=int(g.n.sum()),positive_logloss_folds=int((g.gain_logloss>0).sum()),positive_rmse_folds=int((g.gain_rmse_net>0).sum()))
        for c in ['gain_logloss','gain_brier','gain_rmse_net','gain_mae_net','effect_correct_pp','effect_net_bps']:
            r[c]=np.average(g[c],weights=g.n)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'incremental_summary.csv',index=False)
    df=pd.read_parquet(OUT/'validation_features.parquet')
    preds=pd.read_parquet(OUT/'statistical_validation_predictions.parquet')
    preds=preds[preds.feature=='acf_1_60'].merge(df[['symbol','timestamp','confidence']],on=['symbol','timestamp'],validate='one_to_one')
    health=[]
    for fold,z in preds.groupby('fold'):
        health.append(dict(fold=int(fold),n=len(z),correctness=z.correct.mean(),baseline_p_mean=z.base_p_correct.mean(),
            baseline_p_min=z.base_p_correct.min(),baseline_p_max=z.base_p_correct.max(),
            baseline_fraction_over95=(z.base_p_correct>.95).mean(),baseline_logloss=log_loss(z.correct,z.base_p_correct),
            confidence_reference_logloss=log_loss(z.correct,z.confidence),baseline_brier=brier_score_loss(z.correct,z.base_p_correct),
            confidence_reference_brier=brier_score_loss(z.correct,z.confidence)))
    pd.DataFrame(health).to_csv(OUT/'control_model_health.csv',index=False)
    fixed=[]
    for symbol,g in df.groupby('symbol'):
        z=g.copy()
        z['trend_bin']=pd.qcut(z.trend_360,5,labels=False,duplicates='drop')
        z['lv']=np.log(z.volatility_30m.astype(float))
        z['ld']=np.log(z.volatility_15d.astype(float))
        z['trend_control']=np.tanh(z.trend_360/2)
        for spec,keys in [('strata',['confidence','pred_direction','vol_bin','trend_bin']),
                          ('strata_day',['confidence','pred_direction','vol_bin','trend_bin','day'])]:
            cols=FEATURES+['correct','net_bps','lv','ld','trend_control']
            means=z.groupby(keys)[cols].transform('mean')
            residual=z[cols]-means
            controls=residual[['lv','ld','trend_control']].to_numpy()
            keep=np.linalg.norm(controls,axis=0)>1e-8
            controls=controls[:,keep]
            def partial(v):
                return v-controls@np.linalg.lstsq(controls,v,rcond=1e-10)[0] if controls.shape[1] else v
            yc=partial(residual.correct.to_numpy());yg=partial(residual.net_bps.to_numpy())
            groupn=z.groupby(keys).confidence.transform('size')
            for name in FEATURES:
                f=partial(residual[name].to_numpy());den=np.dot(f,f)
                iqr=z[name].quantile(.75)-z[name].quantile(.25)
                def corr(y):
                    v=np.sqrt(den*np.dot(y,y));return np.dot(f,y)/v if v>0 else np.nan
                fixed.append(dict(symbol=symbol,feature=name,spec=spec,n=len(z),n_in_repeated_strata=int((groupn>=2).sum()),
                    partial_corr_correct=corr(yc),partial_corr_net=corr(yg),
                    effect_correct_pp_per_IQR=100*np.dot(f,yc)/den*iqr if den>1e-16 else np.nan,
                    effect_net_bps_per_IQR=np.dot(f,yg)/den*iqr if den>1e-16 else np.nan))
    pd.DataFrame(fixed).to_csv(OUT/'conditional_strata_effects.csv',index=False)
    reliability=[]
    df['acf_sign']=np.where(df.acf_1_360>.05,'positive',np.where(df.acf_1_360<-.05,'negative','near_zero'))
    bins=np.minimum((df.confidence*10).astype(int),9)
    for (symbol,band,b),g in df.groupby(['symbol','acf_sign',bins]):
        reliability.append(dict(symbol=symbol,band=band,bin=int(b),n=len(g),confidence=g.confidence.mean(),accuracy=g.correct.mean()))
    rel=pd.DataFrame(reliability);rel.to_csv(OUT/'reliability_bins.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        for band in ['negative','near_zero','positive']:
            g=rel[(rel.symbol==symbol)&(rel.band==band)&(rel.n>=100)]
            ax.plot(g.confidence,g.accuracy,marker='o',label=band)
        ax.plot([.5,1],[.5,1],color='gray',linestyle='--');ax.set(title=symbol,xlabel='Mean confidence',ylabel='Actual correctness')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('ACF lag-1 / W=360 bands; validation; calibration already fit here')
    fig.tight_layout();fig.savefig(OUT/'reliability_by_acf.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    s=pd.DataFrame(rows);s=s[s.scope=='pooled'].sort_values('feature')
    axes[0].barh(s.feature,s.gain_logloss,color=np.where(s.gain_logloss>=0,'steelblue','salmon'))
    axes[0].axvline(0,color='black',linewidth=.5);axes[0].set(xlabel='Weighted mean log-loss reduction (positive = better)',title='Temporal validation: correctness')
    axes[1].barh(s.feature,s.gain_rmse_net,color=np.where(s.gain_rmse_net>=0,'steelblue','salmon'))
    axes[1].axvline(0,color='black',linewidth=.5);axes[1].set(xlabel='Weighted mean fold RMSE reduction (bps)',title='Return statistical model, not trading utility')
    fig.tight_layout();fig.savefig(OUT/'incremental_effects.png',dpi=160);plt.close(fig)
    weekly=pd.read_csv(OUT/'weekly_quantiles.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        x=weekly[(weekly.feature=='abs_acf_1_360')&(weekly.symbol==symbol)&(weekly.selected_n>=200)]
        for q in [1,5]:
            y=x[x['quantile']==q]
            ax.plot(y.week,y.net_selected_bps,marker='o',label=f'Q{q}')
        ax.axhline(0,color='gray',linestyle='--');ax.set(title=symbol,ylabel='Net bps / selected sample')
        ax.tick_params(axis='x',rotation=45,labelsize=7);ax.legend()
    fig.suptitle('|ACF lag-1, W=360|: weeks with >=200 selections only')
    fig.tight_layout();fig.savefig(OUT/'weekly_stability.png',dpi=160);plt.close(fig)
    print('SUMMARIES AND ROBUSTNESS COMPLETE')

if __name__=='__main__':main()
