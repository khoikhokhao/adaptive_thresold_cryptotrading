"""Shift-A: cached TRAIN-only PCA reference and normalization, validation selection then test."""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_regime_threshold'))
from run_regime_a import GRID,FIXED_GRID,TARGETS,LAM,choose,scopes,metrics,fit_cutoff,sha,write_json,memory,md_table,display
sys.path.insert(0,str(ROOT/'research_distribution_shift'))
from build_scores import point_scores

REFDIR=ROOT/'research_distribution_shift/validation_20261103'

def normalize(df,normalization):
    df=df.copy();df['h']=0.;df['fallback']=False
    for symbol,n in normalization.items():
        sm=df.symbol==symbol;valid=sm&np.isfinite(df.pca_re32)&(n['q95']>n['median'])
        df.loc[sm&~valid,'fallback']=True
        if n['q95']>n['median']:df.loc[valid,'h']=np.tanh((df.loc[valid,'pca_re32']-n['median'])/(n['q95']-n['median']))
    return df

def selected(df,beta,tau):
    return df.confidence.to_numpy()>=np.clip(tau+beta*df.h.to_numpy(),.5,.95)

def search(df,tau):
    rows=[]
    for beta in GRID:
        m=metrics(df,selected(df,beta,tau));penalty=LAM*beta**2
        rows.append(dict(beta=beta,penalty_bps=penalty,objective_bps=m['net_selected_bps']-penalty,
            feasible=m['coverage']>=.1,**m))
    return pd.DataFrame(rows)

def prepare(pred):
    assert not pred.duplicated(['symbol','timestamp']).any()
    assert set(pred.symbol)=={'BTC_USDT','ETH_USDT'}
    np.testing.assert_allclose(pred.confidence,np.maximum(pred.proba_up,pred.proba_down),atol=1e-14,rtol=0)
    assert np.isfinite(pred[['confidence','ret_bps']]).all().all() and (pred.ret_bps.abs()>100).all()
    pred=pred.copy();pred['correct']=(pred.pred_direction==pred.y_direction).astype(float)
    pred['net_bps']=np.where(pred.pred_direction==1,pred.ret_bps,-pred.ret_bps)-1.
    return pred

def from_matrix(X,meta,refs):
    result=meta.copy();result['pca_re32']=np.nan
    for symbol,r in refs.items():
        ix=np.flatnonzero((meta.symbol==symbol).to_numpy())
        score,_=point_scores(X[ix],r)
        result.loc[ix,'pca_re32']=score['pca_re32']
    return result

def main(out):
    begun=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (out/'resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-begun,**m))+'\n')
        print(stage,flush=True)
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    event('start')
    prep=ROOT/'data/btc_eth_1pct/prepared';vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json';refpath=REFDIR/'training_references.joblib'
    protected=[refpath,REFDIR/'training_score_quantiles.csv',REFDIR/'validation_analysis_rows.parquet',
        prep/'X_train.npy',prep/'X_validation.npy',prep/'scaler.joblib',prep/'imputer.joblib',prep/'feature_schema.json',vp,sel,
        ROOT/'proposoed_method_v2/distribution_shift_A.md',ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py',
        ROOT/'data/btc_eth_1pct/training/classifier_complete.joblib',
        ROOT/'research_autocorrelation_threshold/method_a_20261103/locked_config_before_test.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    manifest=json.loads((prep/'matrix_manifest.json').read_text())
    for name in ['X_train.npy','X_validation.npy']:assert sha(prep/name)==manifest[name]['sha256']
    refs=joblib.load(refpath);norm={s:dict(median=r['train_quantiles']['pca_re32']['q50'],q95=r['train_quantiles']['pca_re32']['q95'],n_train=r['n']) for s,r in refs.items()}
    original=json.loads(sel.read_text());tau=original['tau']
    Xv=np.load(prep/'X_validation.npy',mmap_mode='r');assert Xv.shape==(80260,128) and Xv.dtype==np.float32
    vm=pd.read_parquet(prep/'validation_metadata.parquet',columns=['symbol','timestamp']).reset_index(drop=True)
    pred=prepare(pd.read_parquet(vp));assert len(pred)==80260
    pd.testing.assert_frame_equal(pred[['symbol','timestamp']].reset_index(drop=True),vm)
    recomputed=from_matrix(Xv,vm,refs)
    cached=pd.read_parquet(REFDIR/'validation_analysis_rows.parquet',columns=['symbol','timestamp','pca_re32','label_endpoint'])
    val=pred.merge(cached,on=['symbol','timestamp'],validate='one_to_one')
    delta=float(np.max(np.abs(val.pca_re32.to_numpy()-recomputed.pca_re32.to_numpy())))
    np.testing.assert_allclose(val.pca_re32,recomputed.pca_re32,atol=1e-14,rtol=1e-12)
    audits=[]
    for symbol,r in refs.items():
        ix=np.flatnonzero((vm.symbol==symbol).to_numpy());ix=ix[np.linspace(0,len(ix)-1,min(2048,len(ix))).astype(int)]
        z=(Xv[ix].astype(float)-r['mean'])/r['scale'];U=r['components']
        spectral=np.sqrt(((z@U)[:,32:]**2).sum(axis=1)/96)
        reconstructed=z-(z@U[:,:32])@U[:,:32].T
        direct=np.sqrt((reconstructed**2).sum(axis=1)/96)
        ortho=float(np.max(np.abs(U.T@U-np.eye(128))))
        np.testing.assert_allclose(spectral,direct,atol=1e-12,rtol=1e-5)
        rho=float(spearmanr(spectral,direct).statistic)
        if ortho>1e-10 or rho<.99999:raise RuntimeError('Measurement audit failed; stop before test')
        audits.append(dict(symbol=symbol,n_audit=len(ix),orthogonality_max_error=ortho,
            spectral_direct_max_abs_difference=float(np.max(np.abs(spectral-direct))),rank_spearman=rho))
    write_json(out/'measurement_audit.json',dict(cache_recomputed_max_abs_difference=delta,
        audits=audits,status='PASS',caveat='No alternative eigensolver/float32-reference refits or economic validity proven'))
    val=normalize(val,norm)
    np.testing.assert_array_equal(selected(val,0.,tau),val.confidence>=tau)
    assert (val.confidence>=tau).sum()==original['validation_count']
    assert abs(val.loc[val.confidence>=tau,'net_bps'].mean()-original['validation_mean_net_bps'])<.001
    event('train_only_reference_normalization_and_measurement_audit_PASS')
    grid=search(val,tau);grid.to_csv(out/'beta_grid_full_validation.csv',index=False)
    beta=choose(grid,'objective_bps','beta')
    fixed=[]
    for t in FIXED_GRID:
        m=metrics(val,val.confidence>=t);fixed.append(dict(tau=t,feasible=m['coverage']>=.1,**m))
    fixed=pd.DataFrame(fixed);fixed.to_csv(out/'fixed_tau_grid_validation.csv',index=False)
    taustar=choose(fixed,'net_selected_bps','tau',reference=tau)
    folds=pd.read_csv(ROOT/'research_distribution_shift/validation_20261103/temporal_folds.csv');temporal=[];foldconfigs=[]
    for row in folds.itertuples(index=False):
        lo,hi=pd.Timestamp(row.eval_start),pd.Timestamp(row.eval_end)
        tr=val[(val.timestamp<lo)&(val.label_endpoint<lo)]
        ev=val[(val.timestamp>=lo)&(val.timestamp<hi)&(val.label_endpoint<hi)]
        assert len(tr)==row.train_n and len(ev)==row.eval_n
        fg=search(tr,tau);fg.to_csv(out/f'prefix_{row.fold}_beta_grid.csv',index=False)
        b=choose(fg,'objective_bps','beta') if fg.feasible.any() else None
        foldconfigs.append(dict(fold=row.fold,beta=b,status='selected' if b is not None else 'infeasible_on_prefix',train_n=len(tr),eval_n=len(ev)))
        for method,bb in [('fixed_064',0.),('shift_a_prefix',b)]:
            if bb is not None:
                for scope,sm in scopes(ev):temporal.append(dict(fold=row.fold,method=method,scope=scope,beta=b,**metrics(ev[sm],selected(ev,bb,tau)[sm])))
    write_json(out/'prefix_configs.json',foldconfigs);pd.DataFrame(temporal).to_csv(out/'validation_temporal_stability.csv',index=False)
    rules=[]
    for scope,sm in scopes(val):
        for name in ['fixed_score','shift_a_score']:
            score=val.confidence.to_numpy()-(beta*val.h.to_numpy() if name=='shift_a_score' else 0.)
            for target in TARGETS:rules.append(dict(scope=scope,method=name,target=target,**fit_cutoff(score[sm],target)))
    lock=dict(beta=beta,beta_grid=GRID,lambda_bps=LAM,tau_fixed=tau,tau_star=taustar,cost_bps=1.,validation_n=len(val),
        minimum_validation_pooled_coverage=.1,objective='mean net bps per selected - 100*beta^2',
        reference='cached PCA32 TRAIN-only per-asset geometry; no refit',normalization=norm,
        normalization_formula='h=tanh((pca_re32-train_median)/(train_q95-train_median))',
        threshold='clip(tau_fixed+beta*h,0.5,0.95)',fallback='Fixed 0.64; retain all rows',
        matched_rules=rules,input_sha256=hashes,test_predictions_not_read_at_lock=True,measurement_audit='PASS')
    write_json(out/'locked_config_before_test.json',lock);lockhash=sha(out/'locked_config_before_test.json')
    event('beta_all_operating_points_and_train_normalization_locked_before_test')
    # First access to test data, strictly after locking every decision.
    tp=ROOT/'data/btc_eth_1pct/training/test_predictions_complete.parquet'
    Xtest_path=prep/'X_test.npy';assert sha(Xtest_path)==manifest['X_test.npy']['sha256']
    Xt=np.load(Xtest_path,mmap_mode='r');assert Xt.shape==(65182,128) and Xt.dtype==np.float32
    tm=pd.read_parquet(prep/'test_metadata.parquet',columns=['symbol','timestamp']).reset_index(drop=True)
    test=prepare(pd.read_parquet(tp));assert len(test)==65182
    pd.testing.assert_frame_equal(test[['symbol','timestamp']].reset_index(drop=True),tm)
    tscore=from_matrix(Xt,tm,refs);test=test.merge(tscore,on=['symbol','timestamp'],validate='one_to_one');test=normalize(test,norm)
    native=[];weekly=[];matched=[];thresholds=[]
    for split,df in [('validation',val),('test',test)]:
        c=df.confidence.to_numpy();choices={'fixed_064':c>=tau,'fixed_tau_star':c>=taustar,'shift_a':selected(df,beta,tau)}
        week=df.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str);masks=dict(scopes(df))
        exported=df.copy();exported['tau_shift_a']=np.clip(tau+beta*df.h.to_numpy(),.5,.95)
        for method,chosen in choices.items():
            exported[method+'_selected']=chosen;exported[method+'_action']=np.where(chosen,np.where(df.pred_direction==1,'Long','Short'),'Abstain')
            for scope,sm in masks.items():
                native.append(dict(split=split,method=method,scope=scope,**metrics(df[sm],chosen[sm])))
                for w in week.unique():
                    mask=sm&(week==w).to_numpy()
                    if mask.any():weekly.append(dict(split=split,method=method,scope=scope,week=w,**metrics(df[mask],chosen[mask])))
        for rule in rules:
            sm=masks[rule['scope']];score=c-(beta*df.h.to_numpy() if rule['method']=='shift_a_score' else 0.)
            m=metrics(df[sm],score[sm]>=rule['cutoff']);matched.append(dict(split=split,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=.01))
        for scope,sm in masks.items():
            t=exported.loc[sm,'tau_shift_a'];thresholds.append(dict(split=split,scope=scope,min_tau=t.min(),median_tau=t.median(),p90_tau=t.quantile(.9),max_tau=t.max(),fallback_n=int(df.loc[sm,'fallback'].sum())))
        exported.to_parquet(out/(split+'_decisions.parquet'),index=False)
    native=pd.DataFrame(native);matched=pd.DataFrame(matched)
    native.to_csv(out/'metrics_validation_test.csv',index=False);matched.to_csv(out/'coverage_matched_metrics.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_metrics.csv',index=False);pd.DataFrame(thresholds).to_csv(out/'threshold_distributions.csv',index=False)
    native[native.scope.isin(['BTC_USDT','ETH_USDT'])].groupby(['split','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean().to_csv(out/'macro_average_metrics.csv')
    previous=pd.read_csv(ROOT/'research_autocorrelation_threshold/method_a_20261103/metrics_validation_test.csv')
    actual=native[native.method.isin(['fixed_064','fixed_tau_star'])].reset_index(drop=True)
    expected=previous[previous.method.isin(['fixed_064','fixed_tau_star'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False,rtol=1e-12,atol=1e-10)
    b=native[(native.split=='test')&(native.method=='fixed_064')&native.scope.isin(['BTC_USDT','ETH_USDT'])].set_index('scope')
    a=native[(native.split=='test')&(native.method=='shift_a')&native.scope.isin(['BTC_USDT','ETH_USDT'])].set_index('scope')
    w0=b.selected_n/b.selected_n.sum();w1=a.selected_n/a.selected_n.sum()
    decomposition=dict(mean_net_composition_bps=float(((w1-w0)*b.net_selected_bps).sum()),mean_net_within_asset_bps=float((w1*(a.net_selected_bps-b.net_selected_bps)).sum()),
        accuracy_composition_pp=float(100*((w1-w0)*b.selective_accuracy).sum()),accuracy_within_asset_pp=float(100*(w1*(a.selective_accuracy-b.selective_accuracy)).sum()))
    write_json(out/'asset_mix_decomposition.json',decomposition)
    for path in protected:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    assert sha(out/'locked_config_before_test.json')==lockhash
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(grid.beta,grid.net_selected_bps,marker='o',label='Raw net/selected')
    axes[0].plot(grid.beta,grid.objective_bps,marker='x',label='Penalized objective');axes[0].axvline(beta,color='gray',linestyle='--')
    axes[0].set(title='Full validation selection',xlabel='beta',ylabel='bps');axes[0].legend()
    for method in ['fixed_score','shift_a_score']:
        z=matched[(matched.split=='test')&(matched.scope=='pooled')&(matched.method==method)].sort_values('coverage')
        axes[1].plot(100*z.coverage,z.net_selected_bps,marker='o',label=method)
    axes[1].set(title='Validation-locked cutoffs only',xlabel='Actual test coverage (%)',ylabel='Net bps / selected');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'selection_and_coverage.png',dpi=160);plt.close(fig)
    event('complete_baseline_sources_and_training_reference_integrity_PASS')
    status=dict(beta=beta,tau_star=taustar,elapsed_sec=time.monotonic()-begun,memory=memory(),
        locked_config_sha256=lockhash,test_predictions_sha256=sha(tp),test_matrix_sha256=sha(Xtest_path),
        baseline_equivalence='PASS',measurement_audit='PASS',protected_inputs_unchanged=True,
        reference_fitted_on_train_only=True,MLP_retrained=False,test_used_for_selection=False,
        validation_fallback=int(val.fallback.sum()),test_fallback=int(test.fallback.sum()),
        label='retrospective/exploratory; oracle-filtered; not portfolio PnL')
    write_json(out/'completion.json',status)
    report='''# Shift-A — Shrunk PCA-Residual Threshold

Giữ MLP, probabilities/labels/directions, phí 1 bps và cohort gốc: validation 80.260, test 65.182. Không train lại network; không fit lại PCA/reference/scaler hoặc chọn dimension theo test.

## Method và selection

Score RMS của spectral components 33..128 trong train-only, per-symbol standardized MLP-feature geometry. PCA32/reference tái sử dụng từ exploratory artifact. h=tanh((score−train_Q50)/(train_Q95−train_Q50)); normalization chỉ từ train, không fit bằng validation/test. Threshold=clip(0,64+beta*h,0,50,0,95). Missing/degenerate train spread override Fixed và giữ mẫu số.

Grid beta −0,10..+0,10 bước 0,02, maximize mean net/mẫu chọn−100*beta², coverage validation pooled>=10%; không constraint test/từng coin/direction. Fixed tau* grid 0,50..0,94 bước 0,02 theo raw mean net với cùng validation constraint. Khóa beta, normalizers, native và matched cutoffs trước khi đọc test. Báo raw return không trừ penalty.

'''+f'Beta chọn: **{beta:.2f}**; fixed tau*: **{taustar:.2f}**.\n\n'+'''## Metrics tổng hợp

Coverage/accuracy là %, return là bps. Main validation table dùng đủ 80.260 rows, không nhầm với temporal diagnostics.

'''+md_table(display(native[native.scope=='pooled']))+'''\n\n## Test theo asset, direction và asset×direction\n\n'''+md_table(display(native[(native.split=='test')&(native.scope!='pooled')]))+'''

## Measurement audit và temporal stability

Recomputed validation spectral residual khớp exploratory cache trong tolerance 1e−14 absolute/1e−12 relative (max delta lưu measurement_audit.json). Kiểm tra orthogonality, spectral tail so với direct projection residual trên 2.048 rows/coin: PASS. Audit này không chứng minh ý nghĩa tài chính hoặc ổn định qua mọi eigensolver/reference precision.

Ba expanding prefix diagnostics trên validation với label-endpoint purge. Geometry/normalization train-only giữ nguyên trong mọi fold, không refit bằng prefix; beta chỉ chọn từ past prefix. Infeasible thì báo và không lấp bằng full-validation beta. Grid từng prefix, prefix_configs.json và validation_temporal_stability.csv giữ đủ; không đổi beta chính theo test.

## Coverage matching và attribution

Targets 10%,20%,25,50%,40%; cutoffs chọn từ validation score counts, không test quantile/top-k. Fixed score=c, A score=c−beta*h (diagnostic không clip, khác native). Lưu attainable/actual_on_target, actual drift và cả hai coverage kề target. Báo riêng asset/direction và asset×direction; không gọi matched chỉ vì cùng target. asset_mix_decomposition.json tách composition/within-asset effects, không causal.

![Selection và coverage](selection_and_coverage.png)

## Integrity và giới hạn

Baseline Fixed/tau* native khớp kết quả cũ mọi scope: PASS. Cached X_validation và X_test khớp saved matrix manifests, original feature order đối chiếu qua metadata keys. Source/model/reference/train matrices/proposals và inputs cũ được hash trước/sau không đổi. Guard theo stage: peak WS<=9 GiB, available>=2 GiB.

PCA residual nhỏ/low-rank vẫn có thể nhạy tick/precision/subspace; measurement equivalence không chứng minh đây là causal risk score. Oracle filtering |future return|>1%, 600-row labels overlap/thiếu purge main legacy cohort, validation/model calibration reuse và test đã xem giữ nguyên. Kết quả retrospective/exploratory, không confirmatory OOS; sample signals/returns không phải capital-constrained positions/portfolio PnL. Không triển khai Shift-B.

## Tái lập và artifacts

Lệnh: `reproduction_setup/venv/Scripts/python.exe research_distribution_shift_threshold/run_shift_a.py`; dùng `--out` đến thư mục mới nếu tái lập, không overwrite. Có beta/fixed grids, measurement_audit.json, locked_config_before_test.json, metrics_validation_test.csv, weekly/macro/threshold/coverage CSVs, prefix grids/configs/stability CSV, validation/test_decisions.parquet giữ probabilities/confidence/labels/returns/residual/h/threshold/actions, asset_mix_decomposition.json, completion.json, resources.jsonl.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(grid.to_string(index=False),flush=True)
    print(native[native.scope.isin(['pooled','BTC_USDT','ETH_USDT'])].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'research_distribution_shift_threshold/method_a_20261103')
    with threadpool_limits(limits=2):main(parser.parse_args().out)
