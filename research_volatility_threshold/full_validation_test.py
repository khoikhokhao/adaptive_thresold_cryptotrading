"""Select Method A on the original full validation; evaluate locked rule on original test."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from method_a import ROOT, GRID, SYMBOLS, metrics, threshold, sha, write_json, memory

def main():
    started=time.monotonic()
    out=ROOT/'research_volatility_threshold/full_validation_test_20261103'
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'research_volatility_threshold/method_a_pooled_20261103'
    norm=json.loads((src/'setup_manifest.json').read_text())['normalization']
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    tau=json.loads(sel.read_text())['tau']
    vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    dp=src/'validation_decisions.parquet'
    df=pd.read_parquet(dp)
    v=pd.read_parquet(vp)
    assert len(df)==len(v)==80260
    pd.testing.assert_frame_equal(df[['symbol','timestamp','confidence','ret_bps']].reset_index(drop=True),
                                  v[['symbol','timestamp','confidence','ret_bps']].reset_index(drop=True))
    assert np.isfinite(df.h).all()
    rows=[]
    for beta in GRID:
        m=metrics(df,df.confidence>=threshold(df.h,beta,tau))
        rows.append(dict(beta=beta,feasible=m['coverage']>=.1,**m))
    grid=pd.DataFrame(rows)
    grid.to_csv(out/'full_validation_beta_grid.csv',index=False)
    feasible=grid[grid.feasible]
    if feasible.empty: raise RuntimeError('No pooled coverage-feasible beta')
    best=feasible.net_selected_bps.max()
    tied=feasible[abs(feasible.net_selected_bps-best)<=1e-8].copy()
    tied['abs_beta']=tied.beta.abs()
    beta=float(tied.sort_values(['abs_beta','beta']).iloc[0].beta)
    lock=dict(beta=beta,tau_fixed=tau,normalization=norm,beta_grid=GRID,min_coverage_validation_pooled=.1,
        objective='maximum mean net bps per selected sample',selection_n=80260,cost_bps=1,
        normalization_note='Reuse already frozen symbol median/IQR from early validation; no test fitting.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [vp,dp,sel,src/'setup_manifest.json']},
        test_not_loaded_at_lock=True,method='clip(tau_fixed + beta*tanh((v-median)/IQR), 0.5, 0.95)')
    write_json(out/'locked_config_before_test.json',lock)
    lockhash=sha(out/'locked_config_before_test.json')
    # First test read occurs AFTER selection and saving the immutable decision record.
    tp=ROOT/'data/btc_eth_1pct/training/test_predictions_complete.parquet'
    test=pd.read_parquet(tp)
    assert len(test)==65182 and not test.duplicated(['symbol','timestamp']).any()
    assert set(test.symbol)==set(SYMBOLS)
    assert np.isfinite(test[['confidence','ret_bps']]).all().all()
    np.testing.assert_allclose(test.confidence,np.maximum(test.proba_up,test.proba_down),atol=1e-14,rtol=0)
    assert (test.ret_bps.abs()>100).all()
    unified=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    features=pd.read_parquet(unified,columns=['symbol','timestamp','volatility_30m'])
    test=test.merge(features,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True)
    assert (test._merge=='both').all()
    test['h']=0.
    test['fallback']=False
    for s in SYMBOLS:
        sm=test.symbol==s
        n=norm[s]
        missing=sm&(~np.isfinite(test.volatility_30m)|(n['iqr']<=0))
        test.loc[missing,'fallback']=True
        if n['iqr']>0:
            ok=sm&~missing
            test.loc[ok,'h']=np.tanh((test.loc[ok,'volatility_30m']-n['median'])/n['iqr'])
    test['correct']=(test.pred_direction==test.y_direction).astype(float)
    test['net_bps']=np.where(test.pred_direction==1,test.ret_bps,-test.ret_bps)-1.
    results=[]
    for split,z in [('validation',df),('test',test)]:
        for method,b in [('fixed_064',0.),('method_a',beta)]:
            chosen=z.confidence>=threshold(z.h,b,tau)
            for s in ['pooled']+SYMBOLS:
                sm=np.ones(len(z),dtype=bool) if s=='pooled' else (z.symbol==s).to_numpy()
                results.append(dict(split=split,method=method,symbol=s,**metrics(z[sm],chosen[sm])))
    metrics_df=pd.DataFrame(results)
    metrics_df.to_csv(out/'metrics_validation_test.csv',index=False)
    for z,name in [(df,'validation'),(test,'test')]:
        z=z.copy()
        z['tau_a']=threshold(z.h,beta,tau)
        z['selected_a']=z.confidence>=z.tau_a
        z['selected_fixed']=z.confidence>=tau
        z.to_parquet(out/(name+'_decisions.parquet'),index=False)
    assert sha(out/'locked_config_before_test.json')==lockhash
    mem=memory()
    assert mem['peak_ws_gib']<9 and mem['available_gib']>=2
    write_json(out/'completion.json',dict(elapsed_sec=time.monotonic()-started,memory=mem,
        locked_config_sha256=lockhash,test_predictions_sha256=sha(tp),test_fallback_count=int(test.fallback.sum()),
        label='retrospective/exploratory; oracle-filtered; overlapping returns, not portfolio PnL'))
    print('BETA',beta)
    print(grid.to_string(index=False))
    print(metrics_df.to_string(index=False))

if __name__=='__main__':main()
