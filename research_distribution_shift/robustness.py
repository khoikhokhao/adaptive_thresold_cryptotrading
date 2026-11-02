"""Full-validation conditional associations and score-health diagnostics, not a method."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_scores import ROOT,OUT,SCORES

def main():
    df=pd.read_parquet(OUT/'validation_analysis_rows.parquet')
    rows=[]
    for symbol,z in df.groupby('symbol'):
        z=z.copy();z['trend_bin']=pd.qcut(z.trend_360,5,labels=False,duplicates='drop')
        z['acf_bin']=pd.qcut(z.abs_acf_1_360,5,labels=False,duplicates='drop')
        z['lv']=np.log(z.volatility_30m.astype(float));z['ld']=np.log(z.volatility_15d.astype(float))
        z['trend_control']=np.tanh(z.trend_360/2)
        for score in SCORES:z['L_'+score]=np.log1p(z[score])
        for spec,keys in [('strata',['confidence','pred_direction','vol_bin','trend_bin','acf_bin']),
            ('strata_day',['confidence','pred_direction','vol_bin','trend_bin','acf_bin','day'])]:
            cols=['L_'+s for s in SCORES]+['correct','net_bps','lv','ld','trend_control','acf_1_360','abs_acf_1_360']
            r=z[cols]-z.groupby(keys)[cols].transform('mean')
            controls=r[['lv','ld','trend_control','acf_1_360','abs_acf_1_360']].to_numpy()
            keep=np.linalg.norm(controls,axis=0)>1e-8;controls=controls[:,keep]
            def partial(v):return v-controls@np.linalg.lstsq(controls,v,rcond=1e-10)[0]
            yc=partial(r.correct.to_numpy());yg=partial(r.net_bps.to_numpy())
            counts=z.groupby(keys).confidence.transform('size')
            for score in SCORES:
                f=partial(r['L_'+score].to_numpy());den=np.dot(f,f)
                iqr=z['L_'+score].quantile(.75)-z['L_'+score].quantile(.25)
                def corr(y):
                    d=np.sqrt(den*np.dot(y,y));return np.dot(f,y)/d if d>1e-16 else np.nan
                rows.append(dict(symbol=symbol,score=score,spec=spec,n=len(z),n_repeated_strata=int((counts>=2).sum()),
                    partial_corr_correct=corr(yc),partial_corr_net=corr(yg),
                    effect_correct_pp_per_IQR=100*np.dot(f,yc)/den*iqr if den>1e-16 else np.nan,
                    effect_net_bps_per_IQR=np.dot(f,yg)/den*iqr if den>1e-16 else np.nan))
    pd.DataFrame(rows).to_csv(OUT/'conditional_strata_effects.csv',index=False)
    saturation=[]
    for symbol,g in df.groupby('symbol'):
        saturation.append(dict(symbol=symbol,psi_share_above_5_8=(g.psi360_pc3>5.8).mean(),
            psi_unique=g.psi360_pc3.nunique(),psi_top_value_count=int(g.psi360_pc3.value_counts().iloc[0]),
            reconstruction_median=g.pca_re32.median(),reconstruction_p99=g.pca_re32.quantile(.99),
            reconstruction_rank_spearman_vol=g.pca_re32.corr(g.volatility_30m,method='spearman')))
    pd.DataFrame(saturation).to_csv(OUT/'score_health.csv',index=False)
    scores=pd.read_csv(OUT/'incremental_summary.csv');scores=scores[scores.scope=='pooled'].sort_values('score')
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].barh(scores.score,scores.gain_logloss,color=np.where(scores.gain_logloss>=0,'steelblue','salmon'))
    axes[1].barh(scores.score,scores.gain_rmse_net,color=np.where(scores.gain_rmse_net>=0,'steelblue','salmon'))
    axes[0].set(xlabel='Weighted mean log-loss reduction',title='Validation temporal increments; positive=better')
    axes[1].set(xlabel='Weighted mean fold RMSE reduction (bps)',title='Statistical return prediction, not trading gain')
    for ax in axes:ax.axvline(0,color='black',linewidth=.5)
    fig.tight_layout();fig.savefig(OUT/'incremental_effects.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        z=df[df.symbol==symbol]
        daily=z.groupby('day')[['md128_l01','knn20_pc16','psi360_pc3','w1_360_pc1']].median()
        # Rank within validation is for plotting only, never a train reference or threshold.
        for score in daily.columns:ax.plot(daily.index,daily[score].rank(pct=True),label=score)
        ax.set(title=symbol,ylabel='Rank of daily median (plot only)');ax.tick_params(axis='x',rotation=30,labelsize=7);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'temporal_shift.png',dpi=160);plt.close(fig)
    print('ROBUSTNESS COMPLETE')

if __name__=='__main__':main()
