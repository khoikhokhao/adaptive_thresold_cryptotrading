"""Temporal statistical comparators of correctness and sample returns, validation only."""
from pathlib import Path
import json,time,warnings,sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,mean_squared_error,mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from explore import ROOT,OUT,FEATURES
sys.path.insert(0,str(ROOT/'research_volatility_threshold'))
from method_a import write_json,memory

VARIANTS=FEATURES+['acf_1_360_interactions','acf_1_360_asset_interaction']

def base_numeric(z,origin):
    ell=np.log(np.clip(z.confidence.to_numpy(),1e-6,1-1e-6)/(1-np.clip(z.confidence.to_numpy(),1e-6,1-1e-6)))
    eth=(z.symbol=='ETH_USDT').to_numpy(dtype=float);d=2*z.pred_direction.to_numpy()-1
    trend=np.tanh(z.trend_360.to_numpy()/2)
    return np.column_stack([ell,eth,d,ell*d,ell*eth,
        np.log(z.volatility_30m.to_numpy(dtype=float)),np.log(z.volatility_15d.to_numpy(dtype=float)),
        trend,np.abs(trend),(z.trend_360>1).to_numpy(dtype=float),(z.trend_360< -1).to_numpy(dtype=float),
        (z.timestamp-origin).dt.total_seconds().to_numpy()/86400])

def keys(z):
    return z.symbol+'|'+z.confidence.map(lambda x:format(x,'.17g'))+'|'+z.pred_direction.astype(str)

def candidate(z,name,value=None):
    base='acf_1_360' if name in ['acf_1_360_interactions','acf_1_360_asset_interaction'] else name
    f=z[base].to_numpy() if value is None else np.full(len(z),value)
    if name=='acf_1_360_interactions':
        ell=np.log(np.clip(z.confidence.to_numpy(),1e-6,1-1e-6)/(1-np.clip(z.confidence.to_numpy(),1e-6,1-1e-6)))
        return np.column_stack([f,f*ell,f*(2*z.pred_direction.to_numpy()-1)])
    if name=='acf_1_360_asset_interaction':return np.column_stack([f,f*(z.symbol=='ETH_USDT').to_numpy()])
    return f.reshape(-1,1)

def evaluate(y,p,g,ghat):
    return dict(logloss=log_loss(y,p,labels=[0,1]),brier=brier_score_loss(y,p),
        auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        rmse_net=np.sqrt(mean_squared_error(g,ghat)),mae_net=mean_absolute_error(g,ghat))

def scopes(z):
    yield 'pooled',np.ones(len(z),bool)
    for s in ['BTC_USDT','ETH_USDT']:yield s,(z.symbol==s).to_numpy()
    for d in [0,1]:yield 'predicted_'+('Up' if d else 'Down'),(z.pred_direction==d).to_numpy()

def main():
    start=time.monotonic();modeldir=OUT/'statistical_controls';modeldir.mkdir(exist_ok=False)
    df=pd.read_parquet(OUT/'validation_features.parquet')
    assert len(df)==80260
    origin=df.timestamp.min();end=df.timestamp.max()+pd.Timedelta(1,'ns')
    bounds=[origin+(end-origin)*i/4 for i in range(5)]
    complete=np.isfinite(df[FEATURES+['volatility_30m','volatility_15d','trend_360']]).all(axis=1)
    common=df[complete].copy()
    write_json(OUT/'conditional_protocol.json',dict(variants=VARIANTS,bounds=list(map(str,bounds)),
        excluded_incomplete=int((~complete).sum()),common_n=len(common),folds='3 expanding chronological folds; label-endpoint purge train and evaluation',
        logistic_C=1.,ridge_alpha=10.,rare_stratum_min_train_n=20,hyperparameter_tuning=False,
        statistical_comparators_only=True,no_test_predictions=True,
        controls=['confidence logit and exact symbol/confidence/pred_direction strata',
                  'symbol','predicted direction','confidence*symbol and confidence*direction',
                  'log volatility_30m','log volatility_15d','tanh(trend_360/2) and absolute value',
                  'Bull/Bear trend_360 indicators','linear calendar time']))
    rows=[];folds=[];predictions=[];coefficients=[]
    for fold in range(1,4):
        lo,hi=bounds[fold],bounds[fold+1]
        tr0=common.timestamp<lo;va0=(common.timestamp>=lo)&(common.timestamp<hi)
        tr=common[tr0&(common.label_endpoint<lo)].copy()
        va=common[va0&(common.label_endpoint<hi)].copy()
        assert len(tr)>0 and len(va)>0 and tr.label_endpoint.max()<lo
        assert tr.timestamp.max()<va.timestamp.min()
        counts=keys(tr).value_counts();allowed=set(counts[counts>=20].index)
        kt=keys(tr).where(keys(tr).isin(allowed),'__OTHER__').to_numpy().reshape(-1,1)
        kv=keys(va).where(keys(va).isin(allowed),'__OTHER__').to_numpy().reshape(-1,1)
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        sc=StandardScaler();xt=sc.fit_transform(base_numeric(tr,origin));xv=sc.transform(base_numeric(va,origin))
        xt=np.column_stack([xt,enc.fit_transform(kt)]);xv=np.column_stack([xv,enc.transform(kv)])
        y=tr.correct.to_numpy();g=tr.net_bps.to_numpy();yv=va.correct.to_numpy();gv=va.net_bps.to_numpy()
        folds.append(dict(fold=fold,eval_start=str(lo),eval_end=str(hi),train_n=len(tr),eval_n=len(va),
            train_purged=int(tr0.sum()-len(tr)),eval_purged=int(va0.sum()-len(va)),
            rare_unseen_eval_share=float((~keys(va).isin(allowed)).mean()),control_columns=xt.shape[1]))
        print('FOLD',fold,'train',len(tr),'eval',len(va),flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            b=LogisticRegression(C=1.,solver='lbfgs',max_iter=500,tol=1e-6).fit(xt,y)
        br=Ridge(alpha=10.).fit(xt,g)
        bp=b.predict_proba(xv)[:,1];bg=br.predict(xv)
        base_scores={s:evaluate(yv[mask],bp[mask],gv[mask],bg[mask]) for s,mask in scopes(va)}
        bundle=dict(base_logistic=b,base_return_ridge=br,control_scaler=sc,encoder=enc,allowed_strata=allowed,
                    origin=origin,candidates={})
        for name in VARIANTS:
            cs=StandardScaler();ft=cs.fit_transform(candidate(tr,name));fv=cs.transform(candidate(va,name))
            at=np.column_stack([xt,ft]);av=np.column_stack([xv,fv])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model=LogisticRegression(C=1.,solver='lbfgs',max_iter=500,tol=1e-6).fit(at,y)
            ret=Ridge(alpha=10.).fit(at,g)
            ap=model.predict_proba(av)[:,1];ag=ret.predict(av)
            basefeature='acf_1_360' if name.endswith('_interactions') or name.endswith('_asset_interaction') else name
            q25,q75=tr[basefeature].quantile([.25,.75]).to_numpy()
            fl=cs.transform(candidate(va,name,value=q25));fh=cs.transform(candidate(va,name,value=q75))
            al=np.column_stack([xv,fl]);ah=np.column_stack([xv,fh])
            effectp=model.predict_proba(ah)[:,1]-model.predict_proba(al)[:,1]
            effectg=ret.predict(ah)-ret.predict(al)
            odds=float(np.exp(model.coef_[0,-1]*(q75-q25)/cs.scale_[0])) if ft.shape[1]==1 else np.nan
            for scope,mask in scopes(va):
                baseline=base_scores[scope];aug=evaluate(yv[mask],ap[mask],gv[mask],ag[mask])
                row=dict(fold=fold,feature=name,scope=scope,n=int(mask.sum()),effect_correct_pp=float(100*effectp[mask].mean()),
                    effect_net_bps=float(effectg[mask].mean()),odds_ratio_train_IQR=odds,train_q25=q25,train_q75=q75)
                for metric in baseline:
                    row['base_'+metric]=baseline[metric];row['aug_'+metric]=aug[metric]
                    row['gain_'+metric]=(aug[metric]-baseline[metric]) if metric=='auc' else (baseline[metric]-aug[metric])
                rows.append(row)
            p=va[['symbol','timestamp','correct','net_bps']].copy()
            p['fold']=fold;p['feature']=name;p['base_p_correct']=bp;p['aug_p_correct']=ap
            p['base_net_estimate']=bg;p['aug_net_estimate']=ag
            predictions.append(p)
            coefficients.append(dict(fold=fold,feature=name,added_columns=ft.shape[1],
                added_logistic_coefficients=model.coef_[0,-ft.shape[1]:].tolist(),
                added_ridge_coefficients=ret.coef_[-ft.shape[1]:].tolist(),
                feature_train_mean=cs.mean_.tolist(),feature_train_scale=cs.scale_.tolist(),
                logistic_iterations=int(model.n_iter_[0])))
            bundle['candidates'][name]=dict(logistic=model,return_ridge=ret,feature_scaler=cs)
        joblib.dump(bundle,modeldir/f'fold_{fold}.joblib')
        m=memory()
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    pd.DataFrame(rows).to_csv(OUT/'temporal_incremental_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'temporal_folds.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'statistical_validation_predictions.parquet',index=False)
    write_json(OUT/'statistical_coefficients.json',coefficients)
    write_json(OUT/'conditional_status.json',dict(elapsed_sec=time.monotonic()-start,memory=memory(),
        variants=len(VARIANTS),temporal_folds=len(folds),MLP_unchanged=True,adaptive_threshold_implemented=False,
        test_predictions_read=False,convergence='All logistic fits completed; warnings treated as errors'))
    print('CONDITIONAL ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
