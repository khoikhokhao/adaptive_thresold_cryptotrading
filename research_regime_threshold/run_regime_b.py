"""Direction-aware Regime-B and nested ablations; select on validation, freeze, evaluate test."""
from pathlib import Path
import argparse
import itertools
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_regime_a import ROOT, TARGETS, LAM, metrics, fit_cutoff, sha, write_json, memory, scopes, md_table, display

PRIOR=ROOT/'research_regime_threshold/method_a_20261103'
NEW=['regime_b','b_no_interaction','b_direction_only','b_alignment_only']
ALL=['fixed_064','fixed_tau_star','regime_a']+NEW

def adjustment(df,theta):
    beta,gamma,eta=theta
    d=2*df.pred_direction.to_numpy()-1
    z=df.z.to_numpy()
    value=beta*z+gamma*d*z+eta*d
    return np.where(df.fallback.to_numpy(),0.,value)

def choose(grid,selector):
    f=grid.loc[selector&grid.feasible].copy()
    if f.empty:raise RuntimeError('No feasible candidate; stop without test evaluation')
    maximum=f.objective_bps.max()
    f=f[(f.objective_bps-maximum).abs()<=1e-8]
    r=f.sort_values(['norm2','beta','gamma','eta']).iloc[0]
    return [float(r[c]) for c in ['beta','gamma','eta']]

def checks(df,tau,oldbeta):
    assert df.pred_direction.isin([0,1]).all()
    assert np.isfinite(df[['confidence','correct','net_bps','z']]).all().all()
    np.testing.assert_array_equal(df.correct,(df.pred_direction==df.y_direction).astype(float))
    np.testing.assert_array_equal(df.net_bps,np.where(df.pred_direction==1,df.ret_bps,-df.ret_bps)-1.)
    np.testing.assert_array_equal(df.fixed_064_selected,df.confidence>=np.clip(tau+adjustment(df,[0.,0.,0.]),.5,.95))
    np.testing.assert_array_equal(df.regime_a_selected,df.confidence>=np.clip(tau+adjustment(df,[oldbeta,0.,0.]),.5,.95))
    assert (df.z.abs()<=1).all()
    # Direction-only and alignment are functions of predicted direction, never labels.
    d=2*df.pred_direction.to_numpy()-1
    np.testing.assert_array_equal(adjustment(df,[0.,0.,.02]),np.where(df.fallback,0.,.02*d))

def main(out):
    started=time.monotonic()
    out.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (out/'resources.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-started,**m))+'\n')
        print(stage,flush=True)
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('Memory guard')
    event('start')
    protected=[p for p in PRIOR.iterdir() if p.is_file()]+[
        ROOT/'proposoed_method_v2/market_regime_B.md',ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py',
        ROOT/'data/btc_eth_1pct/training/classifier_complete.joblib',
        ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet',
        ROOT/'research_volatility_threshold/method_b_benchmark_20261103/metrics_validation_test.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    write_json(out/'previous_artifact_sha256.json',hashes)
    previous=json.loads((PRIOR/'locked_config_before_test.json').read_text())
    tau,taustar,abeta=previous['tau_fixed'],previous['tau_star'],previous['beta']
    val=pd.read_parquet(PRIOR/'validation_decisions.parquet')
    assert len(val)==80260 and not val.duplicated(['symbol','timestamp']).any()
    checks(val,tau,abeta)
    # Score one candidate at a time; no N x K allocation. Integer grid enforces budget exactly.
    z=val.z.to_numpy();d=2*val.pred_direction.to_numpy()-1
    c=val.confidence.to_numpy();net=val.net_bps.to_numpy();correct=val.correct.to_numpy();fallback=val.fallback.to_numpy()
    rows=[]
    for ib,ig,ie in itertools.product(range(-5,6),repeat=3):
        if abs(ib)+abs(ig)+abs(ie)>8:continue
        beta,gamma,eta=ib*.02,ig*.02,ie*.02
        f=beta*z+gamma*d*z+eta*d
        f=np.where(fallback,0.,f)
        selected=c>=np.clip(tau+f,.5,.95)
        n=int(selected.sum());cov=n/len(val)
        norm2=(ib**2+ig**2+ie**2)*.02**2
        avg=float(net[selected].mean()) if n else np.nan
        rows.append(dict(beta=beta,gamma=gamma,eta=eta,norm2=norm2,penalty_bps=LAM*norm2,
            selected_n=n,coverage=cov,selective_accuracy=float(correct[selected].mean()) if n else np.nan,
            net_selected_bps=avg,expected_net_bps=float(net[selected].sum()/len(val)),
            objective_bps=avg-LAM*norm2,feasible=cov>=.1))
    grid=pd.DataFrame(rows)
    assert len(grid)==sum(abs(b)+abs(g)+abs(e)<=8 for b,g,e in itertools.product(range(-5,6),repeat=3))
    grid.to_csv(out/'full_validation_parameter_grid.csv',index=False)
    selectors={'regime_b':np.ones(len(grid),bool),'b_no_interaction':grid.gamma==0,
               'b_direction_only':(grid.beta==0)&(grid.gamma==0),
               'b_alignment_only':(grid.beta==0)&(grid.eta==0)}
    models={name:choose(grid,mask) for name,mask in selectors.items()}
    # Verify nested A candidates exactly reproduce the saved A validation grid/objective.
    oldgrid=pd.read_csv(PRIOR/'beta_grid_full_validation.csv')
    nested=grid[(grid.gamma==0)&(grid.eta==0)].sort_values('beta').reset_index(drop=True)
    for col in ['beta','coverage','net_selected_bps','objective_bps']:
        np.testing.assert_allclose(nested[col],oldgrid.sort_values('beta')[col],atol=1e-10,rtol=1e-12)
    event('complete_validation_grid_and_nested_A_equivalence_PASS')
    rules=[]
    for scope,sm in scopes(val):
        for method,theta in [('fixed_score',[0.,0.,0.]),('regime_a',[abeta,0.,0.])]+list(models.items()):
            scores=c-adjustment(val,theta)
            for target in TARGETS:rules.append(dict(scope=scope,method=method,target=target,**fit_cutoff(scores[sm],target)))
    counts={name:dict(candidates=int(np.asarray(mask).sum()),feasible=int((np.asarray(mask)&grid.feasible.to_numpy()).sum())) for name,mask in selectors.items()}
    lock=dict(models=models,coefficients_order=['beta','gamma','eta'],grid_step=.02,each_coefficient_bound=.1,
        total_abs_coefficient_budget=.16,lambda_bps=LAM,grid_counts=counts,
        objective='mean net bps/selected sample - 100*(beta^2+gamma^2+eta^2)',
        minimum_validation_pooled_coverage=.1,validation_n=len(val),cost_bps=1.,
        tau_fixed=tau,tau_star=taustar,regime_a_beta=abeta,
        feature='z=tanh(trend_360/2), d=2*pred_direction-1',
        threshold='clip(tau_fixed+beta*z+gamma*d*z+eta*d,0.5,0.95)',
        fallback='fixed 0.64 if trend unavailable',coverage_rules=rules,
        input_sha256=hashes,test_predictions_not_read_at_lock=True,
        tie='tolerance 1e-8 bps; smallest squared coefficient norm; lexicographic beta,gamma,eta')
    write_json(out/'locked_config_before_test.json',lock)
    lockhash=sha(out/'locked_config_before_test.json')
    event('all_models_and_matching_cutoffs_locked_before_test')
    # Test outcomes are first accessed only after all native/matched parameters are frozen.
    test=pd.read_parquet(PRIOR/'test_decisions.parquet')
    assert len(test)==65182 and not test.duplicated(['symbol','timestamp']).any()
    checks(test,tau,abeta)
    native=[];matched=[];weekly=[];thresholds=[]
    for split,df in [('validation',val),('test',test)]:
        choices={'fixed_064':df.fixed_064_selected.to_numpy(),'fixed_tau_star':df.fixed_tau_star_selected.to_numpy(),
                 'regime_a':df.regime_a_selected.to_numpy()}
        decisions=df[['symbol','timestamp','confidence','pred_direction','y_direction','ret_bps','correct','net_bps','trend_360','z','fallback']].copy()
        for name,theta in models.items():
            t=np.clip(tau+adjustment(df,theta),.5,.95)
            choices[name]=df.confidence.to_numpy()>=t
            decisions[name+'_tau']=t
            for scope,sm in scopes(df):
                thresholds.append(dict(split=split,method=name,scope=scope,minimum=float(t[sm].min()),
                    median=float(np.median(t[sm])),p90=float(np.quantile(t[sm],.9)),maximum=float(t[sm].max())))
        week=df.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
        for method,chosen in choices.items():
            decisions[method+'_selected']=chosen
            for scope,sm in scopes(df):
                native.append(dict(split=split,method=method,scope=scope,**metrics(df[sm],chosen[sm])))
                for w in week.unique():
                    mask=sm&(week==w).to_numpy()
                    if mask.any():weekly.append(dict(split=split,method=method,scope=scope,week=w,**metrics(df[mask],chosen[mask])))
        masks=dict(scopes(df))
        scoremap={'fixed_score':df.confidence.to_numpy(),'regime_a':df.confidence.to_numpy()-adjustment(df,[abeta,0.,0.])}
        scoremap.update({name:df.confidence.to_numpy()-adjustment(df,theta) for name,theta in models.items()})
        for rule in rules:
            sm=masks[rule['scope']];chosen=scoremap[rule['method']][sm]>=rule['cutoff']
            m=metrics(df[sm],chosen)
            matched.append(dict(split=split,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=.01))
        decisions.to_parquet(out/(split+'_decisions.parquet'),index=False)
    native=pd.DataFrame(native);matched=pd.DataFrame(matched)
    native.to_csv(out/'metrics_validation_test.csv',index=False)
    matched.to_csv(out/'coverage_matched_metrics.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_metrics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'threshold_distributions.csv',index=False)
    native[native.scope.isin(['BTC_USDT','ETH_USDT'])].groupby(['split','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean().to_csv(out/'macro_average_metrics.csv')
    old=pd.read_csv(PRIOR/'metrics_validation_test.csv')
    actual=native[native.method.isin(['fixed_064','fixed_tau_star','regime_a'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[actual.columns],actual,check_dtype=False,rtol=1e-12,atol=1e-10)
    for p in protected:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    assert sha(out/'locked_config_before_test.json')==lockhash
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ['fixed_score','regime_a','regime_b','b_no_interaction','b_direction_only','b_alignment_only']:
        z=matched[(matched.split=='test')&(matched.scope=='pooled')&(matched.method==name)].sort_values('coverage')
        axes[0].plot(100*z.coverage,100*(1-z.selective_accuracy),marker='o',label=name)
        axes[1].plot(100*z.coverage,z.net_selected_bps,marker='o',label=name)
    axes[0].set(title='Retrospective test; validation-locked cutoffs',xlabel='Actual coverage (%)',ylabel='Selective error (%)')
    axes[1].set(title='Coverage can drift; no test cutoff selection',xlabel='Actual coverage (%)',ylabel='Net bps / selected sample')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'risk_profit_coverage.png',dpi=160);plt.close(fig)
    event('complete_prior_results_and_original_sources_unchanged_PASS')
    write_json(out/'completion.json',dict(elapsed_sec=time.monotonic()-started,memory=memory(),
        models=models,locked_config_sha256=lockhash,previous_artifacts_unchanged=True,
        previous_A_and_fixed_metrics_equivalence='PASS',nested_grid_A_equivalence='PASS',
        original_MLP_retrained=False,test_used_for_selection=False,
        label='retrospective/exploratory; oracle-filtered; overlapping sample returns, not portfolio PnL'))
    pooled=display(native[native.scope=='pooled'])
    breakdown=display(native[(native.split=='test')&(native.scope!='pooled')])
    report='''# Regime-B — Direction–Trend Interaction Threshold

Giữ MLP, predictions, labels, horizon, phí 1 bps và cohort gốc: validation 80.260, test 65.182 mẫu. Không train lại MLP; không fit logistic correctness mới. Regime-A, Fixed và toàn bộ kết quả cũ giữ nguyên, hash trước/sau kiểm tra PASS.

## Công thức và quy trình

Threshold = clip(0,64 + beta*z + gamma*d*z + eta*d, 0,50, 0,95), z=tanh(trend_360/2), d=+1 khi **MLP dự đoán Up**, −1 khi dự đoán Down; không thay d bằng future label. Tái sử dụng trend đã kiểm chứng tương đương ở Regime-A. Nếu thiếu trend, dùng Fixed.

Grid mỗi hệ số −0,10..+0,10 bước 0,02, tổng |beta|+|gamma|+|eta|≤0,16. Objective validation: mean net/mẫu chọn −100*(beta²+gamma²+eta²), minimum coverage pooled 10%; không yêu cầu từng coin/direction hoặc test đạt 10%. Chọn theo penalized objective, hòa thì ưu tiên norm nhỏ rồi lexicographic; tất cả hệ số, cutoffs matching và ablations được khóa trước lần đọc test outcomes trong script.

Ablations: bỏ interaction gamma=0; direction-only beta=gamma=0; alignment-only beta=eta=0. Chọn từng ablation bằng đúng validation objective, không chọn lại trên test. Fixed tau* và Regime-A dùng tham số cũ. Các return báo trong bảng là raw net sau phí, không trừ regularization penalty.

## Hệ số đã khóa

```json
'''+json.dumps(dict(models=models,candidate_counts=counts),indent=2)+'''
```

## Tổng hợp

Coverage và selective accuracy là %. Net là bps/mẫu. Không gọi accuracy toàn cohort là selective accuracy.

'''+md_table(pooled)+'''\n\n## Test theo asset, predicted direction và asset×direction\n\n'''+md_table(breakdown)+'''

## Coverage-matched và stability

coverage_matched_metrics.csv có targets 10%, 20%, 25,50%, 40%; scope pooled, mỗi asset, mỗi direction và asset×direction. Score là confidence trừ adjustment, diagnostic không clip (khác native). Khóa hệ số rồi chọn cutoff bằng validation score counts, không dùng return/label để phá tie. Không dùng test quantile/top-k để ép coverage. attainable và hai coverage kề target báo plateau; actual_on_target báo actual drift. Không coi hai phương pháp matched chỉ vì target cùng nhau. weekly_metrics.csv giữ mọi tuần có mẫu, few_samples đánh dấu selections <200.

![Risk/profit coverage](risk_profit_coverage.png)

## Integrity, giới hạn và tái lập

Nested grid gamma=eta=0 được đối chiếu với Regime-A cũ, raw và penalized metrics tương đương: PASS. Native Fixed/tau*/A đối chiếu nguyên kết quả cũ: PASS. Không chỉnh params sau bảng test. Toàn bộ kết quả retrospective/exploratory: test đã xem trước; validation dùng lại cho calibration/model selection/thresholds; oracle filter loại |future return|≤1%; horizon 600 dòng overlap, không purge thêm để giữ cohort baseline. Sample returns không phải portfolio PnL, không báo annualized Sharpe hoặc IID significance.

Lệnh: `reproduction_setup/venv/Scripts/python.exe research_regime_threshold/run_regime_b.py`; dùng `--out` cho thư mục mới nếu tái lập, không overwrite. Có parameter grid đầy đủ, locked_config_before_test.json, metrics_validation_test.csv, coverage_matched_metrics.csv, weekly_metrics.csv, macro_average_metrics.csv, threshold_distributions.csv, validation/test_decisions.parquet, previous_artifact_sha256.json, completion.json, resources.jsonl. Guard kiểm tra theo stage: peak working set ≤9 GiB, available RAM ≥2 GiB.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(models=models,counts=counts),indent=2),flush=True)
    print(native[(native.scope.isin(['pooled','BTC_USDT','ETH_USDT']))].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'research_regime_threshold/method_b_20261103')
    main(parser.parse_args().out)
