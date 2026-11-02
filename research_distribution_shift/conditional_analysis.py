"""Temporal validation statistical controls; no test or adaptive execution rules."""
import sys,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,brier_score_loss,mean_squared_error,mean_absolute_error
from threadpoolctl import threadpool_limits
from build_scores import ROOT,OUT,SCORES
sys.path.insert(0,str(ROOT/'research_volatility_threshold'))
from method_a import memory,write_json
sys.path.insert(0,str(ROOT/'research_autocorrelation_analysis'))
from conditional_analysis import base_numeric,keys

MAIN=[s for s in SCORES if s!='pooled_md128']
VARIANTS=MAIN+[s+'_asset_interaction' for s in ['md128_l01','knn20_pc16','psi360_pc3','w1_360_pc1']]

def controls(z,origin):
    # Remove linear calendar extrapolation BEFORE looking at shift-score model outcomes.
    previous=base_numeric(z,origin)[:,:-1]
    return np.column_stack([previous,z.acf_1_360,z.abs_acf_1_360,z.acf_5_1440,z.er_360])

def fmatrix(z,name,value=None):
    score=name.removesuffix('_asset_interaction')
    f=np.log1p(z[score].to_numpy()) if value is None else np.full(len(z),value)
    return np.column_stack([f,f*(z.symbol=='ETH_USDT').to_numpy()]) if name.endswith('_asset_interaction') else f.reshape(-1,1)

def scopes(z):
    yield 'pooled',np.ones(len(z),bool)
    for s in ['BTC_USDT','ETH_USDT']:yield s,(z.symbol==s).to_numpy()
    for d in [0,1]:yield 'predicted_'+('Up' if d else 'Down'),(z.pred_direction==d).to_numpy()

def loss(y,p,g,ghat):
    return dict(logloss=log_loss(y,p,labels=[0,1]),brier=brier_score_loss(y,p),
        rmse_net=np.sqrt(mean_squared_error(g,ghat)),mae_net=mean_absolute_error(g,ghat))

def main():
    begun=time.monotonic();folder=OUT/'statistical_controls';folder.mkdir(exist_ok=False)
    df=pd.read_parquet(OUT/'validation_analysis_rows.parquet');assert len(df)==80260
    assert np.isfinite(df[MAIN]).all().all()
    originalfolds=pd.read_csv(ROOT/'research_autocorrelation_analysis/validation_20261103/temporal_folds.csv')
    write_json(OUT/'conditional_protocol.json',dict(variants=VARIANTS,logistic_C=.01,ridge_alpha=10.,
        new_score_transform='log1p(score), then train-prefix StandardScaler',
        controls=['confidence logit/exact confidence-symbol-direction strata','symbol','predicted direction',
            'log volatility_30m/15d','signed and absolute trend','Bull/Bear indicators',
            'signed/absolute ACF1_360','signed ACF5_1440','ER360'],
        calendar_predictor=False,reason='Avoid unstable calendar extrapolation observed in prior analysis; predeclared, not tuned on shift outcomes',
        pooled_reference_excluded_from_predictive_models=True,hyperparameters_tuned=False,no_test_predictions=True))
    output=[];predictions=[];health=[];foldrows=[];coef=[]
    for row in originalfolds.itertuples(index=False):
        lo,hi=pd.Timestamp(row.eval_start),pd.Timestamp(row.eval_end)
        tr=df[(df.timestamp<lo)&(df.label_endpoint<lo)].copy()
        ev=df[(df.timestamp>=lo)&(df.timestamp<hi)&(df.label_endpoint<hi)].copy()
        assert len(tr)==row.train_n and len(ev)==row.eval_n
        origin=tr.timestamp.min();allowed=set(keys(tr).value_counts().loc[lambda s:s>=20].index)
        kt=keys(tr).where(keys(tr).isin(allowed),'__OTHER__').to_numpy().reshape(-1,1)
        kv=keys(ev).where(keys(ev).isin(allowed),'__OTHER__').to_numpy().reshape(-1,1)
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        scaler=StandardScaler();xt=scaler.fit_transform(controls(tr,origin));xv=scaler.transform(controls(ev,origin))
        xt=np.column_stack([xt,enc.fit_transform(kt)]);xv=np.column_stack([xv,enc.transform(kv)])
        y=tr.correct.to_numpy();g=tr.net_bps.to_numpy();yv=ev.correct.to_numpy();gv=ev.net_bps.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model=LogisticRegression(C=.01,solver='lbfgs',max_iter=500,tol=1e-6).fit(xt,y)
        ridge=Ridge(alpha=10.).fit(xt,g);bp=model.predict_proba(xv)[:,1];bg=ridge.predict(xv)
        baseline={scope:loss(yv[mask],bp[mask],gv[mask],bg[mask]) for scope,mask in scopes(ev)}
        health.append(dict(fold=row.fold,n=len(ev),correctness=yv.mean(),predicted_correctness=bp.mean(),
            base_logloss=baseline['pooled']['logloss'],confidence_reference_logloss=log_loss(yv,ev.confidence),
            base_brier=baseline['pooled']['brier'],confidence_reference_brier=brier_score_loss(yv,ev.confidence)))
        bundle=dict(control_scaler=scaler,encoder=enc,allowed_strata=allowed,origin=origin,base_logistic=model,base_ridge=ridge,candidates={})
        print('FOLD',row.fold,'train',len(tr),'eval',len(ev),flush=True)
        for name in VARIANTS:
            fs=StandardScaler();ft=fs.fit_transform(fmatrix(tr,name));fv=fs.transform(fmatrix(ev,name))
            at=np.column_stack([xt,ft]);av=np.column_stack([xv,fv])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                aug=LogisticRegression(C=.01,solver='lbfgs',max_iter=500,tol=1e-6).fit(at,y)
            ar=Ridge(alpha=10.).fit(at,g);ap=aug.predict_proba(av)[:,1];ag=ar.predict(av)
            score=name.removesuffix('_asset_interaction');q25,q75=np.quantile(np.log1p(tr[score]),[.25,.75])
            low=np.column_stack([xv,fs.transform(fmatrix(ev,name,value=q25))]);high=np.column_stack([xv,fs.transform(fmatrix(ev,name,value=q75))])
            dp=100*(aug.predict_proba(high)[:,1]-aug.predict_proba(low)[:,1]);dg=ar.predict(high)-ar.predict(low)
            for scope,mask in scopes(ev):
                result=loss(yv[mask],ap[mask],gv[mask],ag[mask]);base=baseline[scope]
                r=dict(fold=row.fold,score=name,scope=scope,n=int(mask.sum()),effect_correct_pp=float(dp[mask].mean()),
                    effect_net_bps=float(dg[mask].mean()),train_log_q25=q25,train_log_q75=q75)
                for metric in base:r['base_'+metric]=base[metric];r['aug_'+metric]=result[metric];r['gain_'+metric]=base[metric]-result[metric]
                output.append(r)
            prediction=ev[['symbol','timestamp','correct','net_bps','confidence']].copy()
            prediction['fold']=row.fold;prediction['score']=name;prediction['base_p_correct']=bp;prediction['aug_p_correct']=ap
            prediction['base_net_estimate']=bg;prediction['aug_net_estimate']=ag;predictions.append(prediction)
            coef.append(dict(fold=int(row.fold),score=name,added_logistic_coefficients=aug.coef_[0,-ft.shape[1]:].tolist(),
                added_ridge_coefficients=ar.coef_[-ft.shape[1]:].tolist(),feature_train_mean=fs.mean_.tolist(),feature_train_scale=fs.scale_.tolist()))
            bundle['candidates'][name]=dict(logistic=aug,ridge=ar,feature_scaler=fs)
        joblib.dump(bundle,folder/f'fold_{row.fold}.joblib')
        foldrows.append(dict(fold=row.fold,train_n=len(tr),eval_n=len(ev),eval_start=str(lo),eval_end=str(hi),control_columns=xt.shape[1]))
        m=memory()
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    results=pd.DataFrame(output);results.to_csv(OUT/'temporal_incremental_metrics.csv',index=False)
    pd.DataFrame(health).to_csv(OUT/'control_model_health.csv',index=False)
    pd.DataFrame(foldrows).to_csv(OUT/'temporal_folds.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'statistical_validation_predictions.parquet',index=False)
    write_json(OUT/'statistical_coefficients.json',coef)
    rows=[]
    for (score,scope),z in results.groupby(['score','scope']):
        r=dict(score=score,scope=scope,n=int(z.n.sum()),positive_logloss_folds=int((z.gain_logloss>0).sum()),positive_rmse_folds=int((z.gain_rmse_net>0).sum()))
        for c in ['gain_logloss','gain_brier','gain_rmse_net','gain_mae_net','effect_correct_pp','effect_net_bps']:r[c]=np.average(z[c],weights=z.n)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'incremental_summary.csv',index=False)
    write_json(OUT/'conditional_status.json',dict(elapsed_sec=time.monotonic()-begun,memory=memory(),variants=len(VARIANTS),folds=3,
        convergence='PASS; warnings treated as errors',MLP_retrained=False,test_predictions_read=False,adaptive_method_implemented=False))
    print('CONDITIONAL ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
