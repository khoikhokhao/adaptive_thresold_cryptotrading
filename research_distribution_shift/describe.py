"""Descriptive score cohorts only; no threshold/model change."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_scores import OUT,SCORES,ROOT
import sys
sys.path.insert(0,str(ROOT/'research_regime_analysis'))
from analyze_validation import stats

def main():
    df=pd.read_parquet(OUT/'validation_scores.parquet')
    refs=pd.read_csv(OUT/'training_score_quantiles.csv').set_index(['symbol','score'])
    distributions=[];metrics=[];direction=[];weekly=[];correlations=[];exceed=[];cal=[];examples=[]
    for score in SCORES:
        df[score+'_q']=np.nan;df[score+'_above_train99']=False
        for symbol,g in df.groupby('symbol'):
            x=g[score];q25,q75=x.quantile([.25,.75]);iqr=q75-q25;ref=refs.loc[(symbol,score)]
            groups,edges=pd.qcut(x,5,labels=False,retbins=True,duplicates='drop')
            df.loc[g.index,score+'_q']=groups+1
            high=x>ref.q99;df.loc[g.index,score+'_above_train99']=high
            distributions.append(dict(score=score,symbol=symbol,n=len(x),missing=int((~np.isfinite(x)).sum()),
                min=x.min(),p01=x.quantile(.01),median=x.median(),p90=x.quantile(.9),p95=x.quantile(.95),
                p99=x.quantile(.99),max=x.max(),train_q99=ref.q99,share_above_train99=high.mean(),
                tukey_outlier_share=((x<q25-1.5*iqr)|(x>q75+1.5*iqr)).mean()))
            for q in sorted(groups.unique()):
                z=g[groups==q]
                metrics.append(dict(score=score,scope=symbol,quantile=int(q)+1,lower=edges[int(q)],upper=edges[int(q)+1],**stats(z)))
                for d,h in z.groupby('pred_direction'):direction.append(dict(score=score,scope=symbol,quantile=int(q)+1,direction=int(d),**stats(h)))
                if q in [0,4]:examples.append(dict(score=score,symbol=symbol,quantile=int(q)+1,
                    correct_n=int(z.correct.sum()),wrong_n=int((1-z.correct).sum()),**stats(z)))
                bins=np.minimum((z.confidence*10).astype(int),9)
                for b,h in z.groupby(bins):cal.append(dict(score=score,symbol=symbol,quantile=int(q)+1,bin=int(b),n=len(h),confidence=h.confidence.mean(),accuracy=h.correct.mean()))
            for week,z in g.groupby('week'):
                for q in sorted(groups.unique()):
                    h=z[groups.loc[z.index]==q]
                    if len(h):weekly.append(dict(score=score,symbol=symbol,week=week,quantile=int(q)+1,score_median=h[score].median(),**stats(h)))
            for flag,z in g.groupby(high):exceed.append(dict(score=score,scope=symbol,above_train99=bool(flag),**stats(z)))
            for control in ['confidence','volatility_30m','volatility_15d','trend_360','abs_acf_1_360','mid_price']:
                correlations.append(dict(score=score,symbol=symbol,control=control,spearman=x.corr(g[control],method='spearman')))
        # Pooled quantile means each symbol's own quantile, not different asset mixtures.
        for q,g in df.groupby(score+'_q'):
            metrics.append(dict(score=score,scope='pooled_within_symbol_quantiles',quantile=int(q),lower=np.nan,upper=np.nan,**stats(g)))
        for flag,g in df.groupby(score+'_above_train99'):exceed.append(dict(score=score,scope='pooled',above_train99=bool(flag),**stats(g)))
    sensitivity=[]
    for symbol,g in df.groupby('symbol'):
        for a,b in [('md128_l01','md128_l10'),('pca_md16','pca_md32'),('knn5_pc16','knn20_pc16'),
                    ('md128_l01','rawspace_md128'),('md128_l01','pooled_md128')]:
            sensitivity.append(dict(symbol=symbol,first=a,second=b,spearman=g[a].corr(g[b],method='spearman')))
    data={'score_distributions':distributions,'quantile_metrics':metrics,'direction_quantiles':direction,
          'weekly_quantiles':weekly,'control_correlations':correlations,'training99_cohorts':exceed,
          'reliability_bins':cal,'novelty_counterexamples':examples,'sensitivity_correlations':sensitivity}
    for name,rows in data.items():pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    df.to_parquet(OUT/'validation_analysis_rows.parquet',index=False)
    primary=['md128_l01','knn20_pc16','psi360_pc3','w1_360_pc1']
    q=pd.DataFrame(metrics)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for j,score in enumerate(primary):
        for symbol,g in q[(q.score==score)&q.scope.isin(['BTC_USDT','ETH_USDT'])].groupby('scope'):
            axes[0,j].plot(g['quantile'],100*g.selective_accuracy,marker='o',label=symbol)
            axes[1,j].plot(g['quantile'],g.net_selected_bps,marker='o',label=symbol)
        axes[0,j].set(title=score,ylabel='Selective accuracy (%)',xlabel='Within-symbol validation quintile')
        axes[1,j].set(ylabel='Net bps / selected sample',xlabel='Within-symbol validation quintile')
        axes[1,j].axhline(0,color='gray',linestyle='--')
    axes[0,0].legend(fontsize=8);fig.suptitle('Validation only, fixed tau=0.64, oracle-filtered cohort')
    fig.tight_layout();fig.savefig(OUT/'quantile_performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));r=pd.DataFrame(cal)
    for ax,symbol in zip(axes,['BTC_USDT','ETH_USDT']):
        for k in [1,5]:
            a=r[(r.score=='md128_l01')&(r.symbol==symbol)&(r['quantile']==k)&(r.n>=100)]
            ax.plot(a.confidence,a.accuracy,marker='o',label=f'MD Q{k}')
        ax.plot([.5,1],[.5,1],color='gray',linestyle='--');ax.set(title=symbol,xlabel='Mean confidence',ylabel='Correctness');ax.legend()
    fig.suptitle('Reliability by train-reference Mahalanobis score; n>=100 bins')
    fig.tight_layout();fig.savefig(OUT/'reliability_by_shift.png',dpi=150);plt.close(fig)
    for symbol,g in df.groupby('symbol'):
        c=g[SCORES].corr(method='spearman');c.to_csv(OUT/(symbol+'_score_correlation.csv'))
    print('DESCRIPTIVE ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()
