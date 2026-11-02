"""ACF-A: fixed protocol, full-validation selection, lock, then original test evaluation."""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_regime_threshold'))
from run_regime_a import GRID,FIXED_GRID,TARGETS,LAM,choose,scopes,metrics,fit_cutoff,sha,write_json,memory,md_table,display

def acf(price):
    r=np.log(price.astype(float)).diff()
    mu=r.rolling(360,min_periods=360).mean()
    denominator=r.pow(2).rolling(360,min_periods=360).sum()-360*mu.pow(2)
    prod=(r*r.shift(1)).rolling(359,min_periods=359).sum()
    sums=r.rolling(359,min_periods=359).sum()+r.shift(1).rolling(359,min_periods=359).sum()
    return (prod-mu*sums+359*mu.pow(2))/denominator.where(denominator>0)

def prepare(pred,features):
    assert not pred.duplicated(['symbol','timestamp']).any()
    assert set(pred.symbol)=={'BTC_USDT','ETH_USDT'}
    np.testing.assert_allclose(pred.confidence,np.maximum(pred.proba_up,pred.proba_down),rtol=0,atol=1e-14)
    assert np.isfinite(pred[['confidence','ret_bps']]).all().all() and (pred.ret_bps.abs()>100).all()
    z=pred.merge(features,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True)
    assert (z._merge=='both').all()
    z['abs_acf']=z.acf_1_360.abs()
    z['correct']=(z.pred_direction==z.y_direction).astype(float)
    z['net_bps']=np.where(z.pred_direction==1,z.ret_bps,-z.ret_bps)-1.
    return z.drop(columns='_merge')

def normalize(z,parameters=None):
    z=z.copy();fitted={} if parameters is None else parameters
    z['h']=0.;z['fallback']=False
    for symbol,g in z.groupby('symbol'):
        x=g.abs_acf[np.isfinite(g.abs_acf)]
        if parameters is None:
            fitted[symbol]=dict(median=float(x.median()),iqr=float(x.quantile(.75)-x.quantile(.25)),n=len(x))
        a=fitted[symbol]
        valid=(z.symbol==symbol)&np.isfinite(z.abs_acf)&(a['iqr']>0)
        z.loc[(z.symbol==symbol)&~valid,'fallback']=True
        if a['iqr']>0:z.loc[valid,'h']=np.tanh((z.loc[valid,'abs_acf']-a['median'])/a['iqr'])
    return z,fitted

def selections(z,beta,tau):
    return z.confidence.to_numpy()>=np.clip(tau+beta*z.h.to_numpy(),.5,.95)

def grid_search(z,tau):
    rows=[]
    for beta in GRID:
        m=metrics(z,selections(z,beta,tau));penalty=LAM*beta**2
        rows.append(dict(beta=beta,penalty_bps=penalty,objective_bps=m['net_selected_bps']-penalty,
                         feasible=m['coverage']>=.1,**m))
    return pd.DataFrame(rows)

def main(out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (out/'resources.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-started,**m))+'\n')
        print(stage,flush=True)
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    event('start')
    source=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    oldfeatures=ROOT/'research_autocorrelation_analysis/validation_20261103/validation_features.parquet'
    protected=[source,vp,sel,oldfeatures,ROOT/'proposoed_method_v2/autocorrelation_A.md',
        ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py',
        ROOT/'data/btc_eth_1pct/training/classifier_complete.joblib',
        ROOT/'research_regime_threshold/method_a_20261103/metrics_validation_test.csv',
        ROOT/'research_regime_threshold/method_b_20261103/metrics_validation_test.csv',
        ROOT/'research_volatility_threshold/method_b_benchmark_20261103/metrics_validation_test.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    original=json.loads(sel.read_text());tau=original['tau']
    u=pd.read_parquet(source,columns=['symbol','timestamp','mid_price']).sort_values(['symbol','timestamp']).reset_index(drop=True)
    assert not u.duplicated(['symbol','timestamp']).any()
    frames=[]
    for symbol,g in u.groupby('symbol',sort=True):
        f=acf(g.mid_price);n=len(g)//2
        np.testing.assert_array_equal(f.iloc[:n],acf(g.mid_price.iloc[:n]))
        returns=np.log(g.mid_price.astype(float)).diff().to_numpy()
        for t in [370,n,len(g)-1]:
            x=returns[t-359:t+1];x=x-x.mean()
            assert np.isclose(f.iloc[t],np.dot(x[1:],x[:-1])/np.dot(x,x),atol=1e-10,rtol=1e-8)
        a=g[['symbol','timestamp']].copy();a['acf_1_360']=f;a['label_endpoint']=g.timestamp.shift(-600)
        frames.append(a)
    features=pd.concat(frames,ignore_index=True);del u,frames
    val=prepare(pd.read_parquet(vp),features);assert len(val)==80260
    old=pd.read_parquet(oldfeatures,columns=['symbol','timestamp','acf_1_360','abs_acf_1_360'])
    parity=val.merge(old,on=['symbol','timestamp'],validate='one_to_one',suffixes=('','_old'))
    np.testing.assert_array_equal(parity.acf_1_360,parity.acf_1_360_old)
    np.testing.assert_array_equal(parity.abs_acf,parity.abs_acf_1_360)
    val,norm=normalize(val)
    assert (val.confidence>=tau).sum()==original['validation_count']
    assert abs(val.loc[val.confidence>=tau,'net_bps'].mean()-original['validation_mean_net_bps'])<.001
    np.testing.assert_array_equal(selections(val,0.,tau),val.confidence>=tau)
    event('causal_direct_and_exact_exploratory_equivalence_PASS')
    grid=grid_search(val,tau);grid.to_csv(out/'beta_grid_full_validation.csv',index=False)
    beta=choose(grid,'objective_bps','beta')
    fixed=[]
    for t in FIXED_GRID:
        m=metrics(val,val.confidence>=t);fixed.append(dict(tau=t,feasible=m['coverage']>=.1,**m))
    fixed=pd.DataFrame(fixed);fixed.to_csv(out/'fixed_tau_grid_validation.csv',index=False)
    taustar=choose(fixed,'net_selected_bps','tau',reference=tau)
    # Secondary validation diagnostics only. Primary selection above always uses all 80,260 rows.
    origin=val.timestamp.min();end=val.timestamp.max()+pd.Timedelta(1,'ns')
    bounds=[origin+(end-origin)*i/4 for i in range(5)]
    folds=[]
    for fold in range(1,4):
        lo,hi=bounds[fold],bounds[fold+1]
        tr=val[(val.timestamp<lo)&(val.label_endpoint<lo)].copy()
        ev=val[(val.timestamp>=lo)&(val.timestamp<hi)&(val.label_endpoint<hi)].copy()
        tr,fnorm=normalize(tr);ev,_=normalize(ev,fnorm)
        fg=grid_search(tr,tau);fg.to_csv(out/f'validation_prefix_{fold}_beta_grid.csv',index=False)
        fb=choose(fg,'objective_bps','beta') if fg.feasible.any() else None
        for method,b in [('fixed_064',0.),('acf_a_prefix',fb)]:
            if b is None:continue
            for scope,sm in scopes(ev):
                folds.append(dict(fold=fold,method=method,scope=scope,beta=fb,
                    train_n=len(tr),eval_n=len(ev),eval_start=str(lo),eval_end=str(hi),
                    **metrics(ev[sm],selections(ev,b,tau)[sm])))
        write_json(out/f'validation_prefix_{fold}_config.json',dict(beta=fb,normalization=fnorm,
            status='selected' if fb is not None else 'infeasible_on_past_prefix',train_n=len(tr),eval_n=len(ev)))
    pd.DataFrame(folds).to_csv(out/'validation_temporal_stability.csv',index=False)
    rules=[]
    for scope,sm in scopes(val):
        for method in ['fixed_score','acf_a_score']:
            score=val.confidence.to_numpy()-(beta*val.h.to_numpy() if method=='acf_a_score' else 0.)
            for target in TARGETS:rules.append(dict(scope=scope,method=method,target=target,**fit_cutoff(score[sm],target)))
    lock=dict(beta=beta,normalization=norm,acf_window_rows=360,acf_lag=1,acf_absolute=True,
        beta_grid=GRID,lambda_bps=LAM,tau_fixed=tau,tau_star=taustar,cost_bps=1.,validation_n=len(val),
        minimum_validation_pooled_coverage=.1,objective='mean net bps/selected - 100*beta^2',
        threshold='clip(tau_fixed+beta*tanh((abs_acf-median_symbol)/IQR_symbol),0.5,0.95)',
        fallback='fixed 0.64; keep samples in denominator',coverage_rules=rules,inputs_sha256=hashes,
        equivalence='Exact feature parity with exploratory cache, prefix-invariance and direct formula PASS',
        test_predictions_not_read_at_lock=True)
    write_json(out/'locked_config_before_test.json',lock);lockhash=sha(out/'locked_config_before_test.json')
    event('beta_normalizers_tau_star_and_all_cutoffs_locked')
    tp=ROOT/'data/btc_eth_1pct/training/test_predictions_complete.parquet'
    test=prepare(pd.read_parquet(tp),features);assert len(test)==65182
    test,_=normalize(test,norm)
    native=[];matched=[];weekly=[];thresholds=[]
    for split,df in [('validation',val),('test',test)]:
        masks=dict(scopes(df));c=df.confidence.to_numpy()
        chosen={'fixed_064':c>=tau,'fixed_tau_star':c>=taustar,'acf_a':selections(df,beta,tau)}
        week=df.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
        for method,selection in chosen.items():
            for scope,sm in masks.items():
                native.append(dict(split=split,method=method,scope=scope,**metrics(df[sm],selection[sm])))
                for w in week.unique():
                    mask=sm&(week==w).to_numpy()
                    if mask.any():weekly.append(dict(split=split,method=method,scope=scope,week=w,**metrics(df[mask],selection[mask])))
        for rule in rules:
            sm=masks[rule['scope']];score=c-(beta*df.h.to_numpy() if rule['method']=='acf_a_score' else 0.)
            m=metrics(df[sm],score[sm]>=rule['cutoff'])
            matched.append(dict(split=split,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=.01))
        decisions=df[['symbol','timestamp','confidence','proba_up','proba_down','pred_direction','y_direction',
            'ret_bps','net_bps','correct','acf_1_360','abs_acf','h','fallback']].copy()
        decisions['tau_acf_a']=np.clip(tau+beta*df.h.to_numpy(),.5,.95)
        for method,selection in chosen.items():
            decisions[method+'_selected']=selection
            decisions[method+'_action']=np.where(selection,np.where(df.pred_direction==1,'Long','Short'),'Abstain')
        for scope,sm in masks.items():
            t=decisions.loc[sm,'tau_acf_a']
            thresholds.append(dict(split=split,scope=scope,minimum=t.min(),median=t.median(),p90=t.quantile(.9),maximum=t.max()))
        decisions.to_parquet(out/(split+'_decisions.parquet'),index=False)
    native=pd.DataFrame(native);matched=pd.DataFrame(matched)
    native.to_csv(out/'metrics_validation_test.csv',index=False)
    matched.to_csv(out/'coverage_matched_metrics.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_metrics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'threshold_distributions.csv',index=False)
    b=native[(native.split=='test')&(native.method=='fixed_064')&native.scope.isin(['BTC_USDT','ETH_USDT'])].set_index('scope')
    a=native[(native.split=='test')&(native.method=='acf_a')&native.scope.isin(['BTC_USDT','ETH_USDT'])].set_index('scope')
    w0=b.selected_n/b.selected_n.sum();w1=a.selected_n/a.selected_n.sum()
    decomposition=dict(mean_net_composition_bps=float(((w1-w0)*b.net_selected_bps).sum()),
        mean_net_within_asset_bps=float((w1*(a.net_selected_bps-b.net_selected_bps)).sum()),
        accuracy_composition_pp=float(100*((w1-w0)*b.selective_accuracy).sum()),
        accuracy_within_asset_pp=float(100*(w1*(a.selective_accuracy-b.selective_accuracy)).sum()))
    write_json(out/'asset_mix_decomposition.json',decomposition)
    native[native.scope.isin(['BTC_USDT','ETH_USDT'])].groupby(['split','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean().to_csv(out/'macro_average_metrics.csv')
    reference=pd.read_csv(ROOT/'research_regime_threshold/method_a_20261103/metrics_validation_test.csv')
    previous=reference[reference.method.isin(['fixed_064','fixed_tau_star'])].reset_index(drop=True)
    actual=native[native.method.isin(['fixed_064','fixed_tau_star'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,previous[actual.columns],check_dtype=False,rtol=1e-12,atol=1e-10)
    for path in protected:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    assert sha(out/'locked_config_before_test.json')==lockhash
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(grid.beta,grid.net_selected_bps,marker='o',label='Raw net / selected')
    axes[0].plot(grid.beta,grid.objective_bps,marker='x',label='Penalized objective')
    axes[0].axvline(beta,color='gray',linestyle='--');axes[0].set(title='Full validation selection',xlabel='beta',ylabel='bps');axes[0].legend()
    for method in ['fixed_score','acf_a_score']:
        z=matched[(matched.split=='test')&(matched.scope=='pooled')&(matched.method==method)].sort_values('coverage')
        axes[1].plot(100*z.coverage,z.net_selected_bps,marker='o',label=method)
    axes[1].set(title='Test: validation-locked cutoffs',xlabel='Actual coverage (%)',ylabel='Net / selected sample (bps)');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'selection_and_coverage.png',dpi=160);plt.close(fig)
    event('complete_baseline_and_protected_inputs_integrity_PASS')
    status=dict(beta=beta,tau_star=taustar,elapsed_sec=time.monotonic()-started,memory=memory(),
        locked_config_sha256=lockhash,test_predictions_sha256=sha(tp),
        baseline_equivalence='PASS',exact_exploratory_feature_equivalence='PASS',
        protected_inputs_unchanged=True,MLP_retrained=False,test_used_for_selection=False,
        validation_fallback=int(val.fallback.sum()),test_fallback=int(test.fallback.sum()),
        label='retrospective/exploratory; oracle-filtered; overlapping sample returns, not portfolio PnL')
    write_json(out/'completion.json',status)
    report='''# ACF-A — Shrunk Absolute-ACF Tilt

Giữ nguyên MLP, predictions/calibration, labels/directions, phí 1 bps và toàn bộ cohort baseline: validation 80.260, test 65.182. Không train lại model. Không thêm VR/ER, không thay lag/window hoặc penalty sau kết quả.

## Method và lựa chọn

a=|ACF lag-1 của 360 returns|, finite-sample mean chung như exploratory analysis. h=tanh((a−median_symbol)/IQR_symbol). Median/IQR fit trên full validation, giữ nguyên khi sang test. Threshold=clip(0,64+beta*h,0,50,0,95). Thiếu feature/IQR zero dùng Fixed, không bỏ mẫu.

Grid beta −0,10..+0,10 bước 0,02; maximize mean net/mẫu chọn−100*beta², minimum pooled validation coverage 10%. Không constraint test hoặc từng coin/direction. Fixed tau* dùng grid 0,50..0,94 bước 0,02, maximize raw mean net với cùng validation constraint. Mọi coefficients/normalizers/native/matching cutoffs được khóa trước khi đọc test outcomes. Báo raw net không trừ regularization penalty.

'''+f'Beta chọn: **{beta:.2f}**; fixed tau*: **{taustar:.2f}**.\n\n'+'''## Tổng hợp

Coverage/accuracy trong bảng là %. Net/expected net là bps. Primary validation table dùng đủ 80.260 mẫu, không nhầm với temporal subperiods.

'''+md_table(display(native[native.scope=='pooled']))+'''\n\n## Test theo asset, predicted direction và asset×direction\n\n'''+md_table(display(native[(native.split=='test')&(native.scope!='pooled')]))+'''

## Temporal stability trên validation

Đây là secondary diagnostic, không thay main protocol. Ba expanding folds dùng đúng chronological boundaries và purge theo label endpoint 600 dòng như exploratory analysis. Mỗi prefix fit median/IQR và chọn beta chỉ từ prefix. Infeasible thì báo, không nới coverage để cứu; chỉ đánh giá adaptive của fold khi có beta hợp lệ. Có grid và config riêng từng prefix. Không dùng các diagnostics này để đổi beta chính sau khi xem test.

## Coverage-matched

coverage_matched_metrics.csv gồm targets 10%,20%,25,50%,40% và pooled/asset/direction/asset×direction scopes. Score fixed=c; score A=c−beta*h. Diagnostic A không clip, khác native rule. Chọn cutoff trên validation score counts sau khi khóa beta; không dùng test quantile/top-k. attainable báo plateau; actual_on_target báo actual drift, không coi target là actual coverage. Không chọn cutoff đẹp trên test làm threshold mới.

![Selection và coverage](selection_and_coverage.png)

## Integrity và giới hạn

Exact ACF/absolute ACF parity so với exploratory cache: PASS; prefix-invariance và direct formula checks trên source prices: PASS. Fixed/tau* pooled và mọi breakdown khớp baseline cũ: PASS. Hash source, model, dữ liệu, proposals và artifacts tham chiếu cũ không đổi. Guard theo stage: peak working set ≤9 GiB, available RAM ≥2 GiB; không phải continuous sampler.

Oracle-filtered cohort |future return|>1%, horizon 600 dòng, overlap/thiếu purge main baseline và reuse validation giữ nguyên. Test đã xem ở các lượt trước: kết quả retrospective/exploratory, không confirmatory OOS. Sample returns/selection counts không phải portfolio PnL hoặc vị thế độc lập. Không triển khai ACF-B trong lượt này.

## Tái lập

Lệnh: `reproduction_setup/venv/Scripts/python.exe research_autocorrelation_threshold/run_acf_a.py`; thêm `--out` đến thư mục mới nếu tái lập, không overwrite. Artifacts gồm đầy đủ beta/fixed grids, locked_config_before_test.json, validation/test_decisions.parquet (probabilities, confidence, labels, signed return, ACF, h, threshold, Long/Short/Abstain), metrics_validation_test.csv, weekly/temporal_stability/macro/coverage_matched/threshold CSVs, prefix configs/grids, completion.json và resources.jsonl.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(grid.to_string(index=False),flush=True)
    print(native[native.scope.isin(['pooled','BTC_USDT','ETH_USDT'])].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'research_autocorrelation_threshold/method_a_20261103')
    main(parser.parse_args().out)
