"""Regime-A: validation-only selection, freeze configuration, then original test evaluation."""
from pathlib import Path
import argparse
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_volatility_threshold'))
from method_a import metrics, threshold, fit_cutoff, sha, write_json, memory

GRID=[i/100 for i in range(-10,11,2)]
FIXED_GRID=[i/100 for i in range(50,95,2)]
TARGETS=[.1,.2,.255,.4]
LAM=100.

def choose(table, score, parameter, reference=0.):
    feasible=table[table.feasible].copy()
    if feasible.empty:raise RuntimeError('No coverage-feasible candidate; stop without test evaluation')
    maximum=feasible[score].max()
    tied=feasible[(feasible[score]-maximum).abs()<=1e-8].copy()
    tied['distance']=(tied[parameter]-reference).abs()
    return float(tied.sort_values(['distance',parameter]).iloc[0][parameter])

def prepare(pred, features, tau):
    assert not pred.duplicated(['symbol','timestamp']).any()
    assert set(pred.symbol)=={'BTC_USDT','ETH_USDT'}
    np.testing.assert_allclose(pred.confidence,np.maximum(pred.proba_up,pred.proba_down),rtol=0,atol=1e-14)
    assert np.isfinite(pred[['confidence','ret_bps']]).all().all()
    assert (pred.ret_bps.abs()>100).all()
    df=pred.merge(features,on=['symbol','timestamp'],how='left',validate='one_to_one',indicator=True)
    assert (df._merge=='both').all()
    df['fallback']=~np.isfinite(df.trend_360)
    df['z']=np.tanh(df.trend_360/2).where(~df.fallback,0.)
    df['correct']=(df.pred_direction==df.y_direction).astype(float)
    df['net_bps']=np.where(df.pred_direction==1,df.ret_bps,-df.ret_bps)-1.
    df['fixed_selected']=df.confidence>=tau
    return df.drop(columns='_merge')

def scopes(df):
    allrows=np.ones(len(df),bool)
    yield 'pooled',allrows
    for s in ['BTC_USDT','ETH_USDT']:yield s,(df.symbol==s).to_numpy()
    for d in [0,1]:yield 'predicted_'+('Up' if d else 'Down'),(df.pred_direction==d).to_numpy()
    for s in ['BTC_USDT','ETH_USDT']:
        for d in [0,1]:yield s+'_predicted_'+('Up' if d else 'Down'),((df.symbol==s)&(df.pred_direction==d)).to_numpy()

def md_table(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('undefined' if pd.isna(v) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def display(df):
    z=df[['split','method','scope','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps']].copy()
    for c in ['coverage','selective_accuracy']:z[c]=(100*z[c]).round(2)
    for c in ['net_selected_bps','expected_net_bps']:z[c]=z[c].round(3)
    return z

def main(out):
    started=time.monotonic()
    out.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (out/'resources.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-started,**m))+'\n')
        print(stage,flush=True)
        if m['peak_ws_gib']>9 or m['available_gib']<2:raise RuntimeError('9 GiB working set / 2 GiB available memory guard')
    event('start')
    source=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    vp=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    sel=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    cached=ROOT/'research_regime_analysis/validation_20261103/validation_regime_rows.parquet'
    proposal=ROOT/'proposoed_method_v2/market_regime_B.md'
    protected=[source,vp,sel,cached,proposal,
        ROOT/'centralized_training_two_classes.py',ROOT/'data_preprocessing.py',
        ROOT/'data/btc_eth_1pct/training/classifier_complete.joblib',
        ROOT/'research_volatility_threshold/full_validation_test_20261103/metrics_validation_test.csv',
        ROOT/'research_volatility_threshold/method_b_benchmark_20261103/metrics_validation_test.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    original=json.loads(sel.read_text());tau=original['tau']
    u=pd.read_parquet(source,columns=['symbol','timestamp','mid_price']).sort_values(['symbol','timestamp']).reset_index(drop=True)
    assert not u.duplicated(['symbol','timestamp']).any() and (u.mid_price>0).all()
    frames=[]
    for symbol,g in u.groupby('symbol',sort=True):
        logp=np.log(g.mid_price)
        r=logp.diff()
        t=(logp-logp.shift(360))/np.sqrt(r.pow(2).rolling(360,min_periods=360).sum()).replace(0,np.nan)
        g=g[['symbol','timestamp']].copy();g['trend_360']=t
        # Prefix invariance on actual source prices: no future rows needed.
        n=len(logp)//2;l=logp.iloc[:n]
        prefix=(l-l.shift(360))/np.sqrt(l.diff().pow(2).rolling(360,min_periods=360).sum()).replace(0,np.nan)
        np.testing.assert_allclose(prefix,t.iloc[:n],rtol=0,atol=0,equal_nan=True)
        frames.append(g)
    features=pd.concat(frames,ignore_index=True)
    del u,frames
    val=prepare(pd.read_parquet(vp),features,tau)
    assert len(val)==80260
    old=pd.read_parquet(cached,columns=['symbol','timestamp','trend_360','correct','net_bps','selected'])
    parity=val.merge(old,on=['symbol','timestamp'],suffixes=('','_cached'),validate='one_to_one')
    for col in ['trend_360','correct','net_bps']:
        np.testing.assert_array_equal(parity[col],parity[col+'_cached'])
    np.testing.assert_array_equal(parity.fixed_selected,parity.selected)
    assert val.fixed_selected.sum()==original['validation_count']
    assert abs(val.loc[val.fixed_selected,'net_bps'].mean()-original['validation_mean_net_bps'])<.001
    np.testing.assert_array_equal(val.confidence>=threshold(val.z,0.,tau),val.fixed_selected)
    assert np.isclose(LAM*.1**2,1.)
    event('feature_prefix_and_exploratory_equivalence_PASS')
    gridrows=[]
    for beta in GRID:
        m=metrics(val,val.confidence>=threshold(val.z,beta,tau))
        penalty=LAM*beta**2
        gridrows.append(dict(beta=beta,penalty_bps=penalty,objective_bps=m['net_selected_bps']-penalty,
                             feasible=m['coverage']>=.1,**m))
    grid=pd.DataFrame(gridrows);grid.to_csv(out/'beta_grid_full_validation.csv',index=False)
    beta=choose(grid,'objective_bps','beta')
    fixedrows=[]
    for t in FIXED_GRID:
        m=metrics(val,val.confidence>=t)
        fixedrows.append(dict(tau=t,feasible=m['coverage']>=.1,**m))
    fixedgrid=pd.DataFrame(fixedrows);fixedgrid.to_csv(out/'fixed_tau_grid_validation.csv',index=False)
    # Fixed tau* uses raw mean net, no adaptive-parameter penalty.
    taustar=choose(fixedgrid,'net_selected_bps','tau',reference=tau)
    rules=[]
    for scope,sm in scopes(val):
        for method in ['fixed_score','regime_a_score']:
            scores=val.confidence.to_numpy()-(beta*val.z.to_numpy() if method=='regime_a_score' else 0.)
            for target in TARGETS:rules.append(dict(scope=scope,method=method,target=target,**fit_cutoff(scores[sm],target)))
    lock=dict(beta=beta,lambda_bps=LAM,trend_window_rows=360,trend_scale=2.,tau_fixed=tau,
        tau_star=taustar,beta_grid=GRID,fixed_tau_grid=FIXED_GRID,cost_bps=1.,minimum_validation_pooled_coverage=.1,
        selection_n=len(val),objective='mean net bps per selected sample minus 100*beta^2',
        threshold='clip(tau_fixed + beta*tanh(trend_360/2),0.5,0.95)',
        missing_trend_policy='use fixed 0.64 and retain in denominator',
        tie='objective tolerance 1e-8 bps, smallest abs(beta), then smallest beta',
        coverage_rules=rules,inputs_sha256=hashes,test_predictions_not_read_at_lock=True,
        feature_equivalence='exact values vs cached exploratory trend and prefix invariance PASS')
    write_json(out/'locked_config_before_test.json',lock)
    lockhash=sha(out/'locked_config_before_test.json')
    event('beta_tau_star_and_matching_cutoffs_locked')
    # First test predictions read: all decisions above are already frozen.
    tp=ROOT/'data/btc_eth_1pct/training/test_predictions_complete.parquet'
    test=prepare(pd.read_parquet(tp),features,tau)
    assert len(test)==65182
    metricsrows=[];weekly=[];matched=[];thresholds=[]
    for split,df in [('validation',val),('test',test)]:
        choices={'fixed_064':df.fixed_selected.to_numpy(),'fixed_tau_star':(df.confidence>=taustar).to_numpy(),
                 'regime_a':(df.confidence>=threshold(df.z,beta,tau)).to_numpy()}
        week=df.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
        for method,chosen in choices.items():
            for scope,sm in scopes(df):
                metricsrows.append(dict(split=split,method=method,scope=scope,**metrics(df[sm],chosen[sm])))
                for w in week.unique():
                    mask=sm&(week==w).to_numpy()
                    if mask.any():weekly.append(dict(split=split,method=method,scope=scope,week=w,**metrics(df[mask],chosen[mask])))
        scope_masks=dict(scopes(df))
        for rule in rules:
            sm=scope_masks[rule['scope']]
            scores=df.confidence.to_numpy()-(beta*df.z.to_numpy() if rule['method']=='regime_a_score' else 0.)
            m=metrics(df[sm],scores[sm]>=rule['cutoff'])
            matched.append(dict(split=split,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=.01))
        decisions=df[['symbol','timestamp','confidence','pred_direction','y_direction','ret_bps','correct','net_bps','trend_360','z','fallback']].copy()
        decisions['tau_regime_a']=threshold(df.z,beta,tau)
        for method,chosen in choices.items():decisions[method+'_selected']=chosen
        decisions.to_parquet(out/(split+'_decisions.parquet'),index=False)
        for scope,sm in scopes(df):
            t=decisions.loc[sm,'tau_regime_a']
            thresholds.append(dict(split=split,scope=scope,minimum=t.min(),median=t.median(),p90=t.quantile(.9),maximum=t.max()))
    native=pd.DataFrame(metricsrows);match=pd.DataFrame(matched)
    native.to_csv(out/'metrics_validation_test.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_metrics.csv',index=False)
    match.to_csv(out/'coverage_matched_metrics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'threshold_distributions.csv',index=False)
    native[native.scope.isin(['BTC_USDT','ETH_USDT'])].groupby(['split','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean().to_csv(out/'macro_average_metrics.csv')
    reference=pd.read_csv(ROOT/'research_volatility_threshold/full_validation_test_20261103/metrics_validation_test.csv')
    for split in ['validation','test']:
        for s in ['pooled','BTC_USDT','ETH_USDT']:
            actual=native[(native.split==split)&(native.scope==s)&(native.method=='fixed_064')].iloc[0]
            expected=reference[(reference.split==split)&(reference.symbol==s)&(reference.method=='fixed_064')].iloc[0]
            for c in ['n','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps']:
                assert np.isclose(actual[c],expected[c],rtol=1e-12,atol=1e-10),(split,s,c)
    for p in protected:assert sha(p)==hashes[str(p.relative_to(ROOT))]
    assert sha(out/'locked_config_before_test.json')==lockhash
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(grid.beta,grid.net_selected_bps,marker='o',label='Raw net / selected')
    axes[0].plot(grid.beta,grid.objective_bps,marker='x',label='Penalized objective')
    axes[0].axvline(beta,color='gray',linestyle='--');axes[0].set(xlabel='beta',ylabel='bps',title='Full validation selection');axes[0].legend()
    for method in ['fixed_score','regime_a_score']:
        a=match[(match.split=='test')&(match.scope=='pooled')&(match.method==method)].sort_values('coverage')
        axes[1].plot(100*a.coverage,a.net_selected_bps,marker='o',label=method)
    axes[1].set(xlabel='Actual test coverage (%)',ylabel='Net bps / selected',title='Cutoffs locked on validation');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'selection_and_coverage.png',dpi=160);plt.close(fig)
    event('complete_baseline_and_source_integrity_PASS')
    status=dict(beta=beta,tau_star=taustar,elapsed_sec=time.monotonic()-started,memory=memory(),
        locked_config_sha256=lockhash,test_predictions_sha256=sha(tp),
        baseline_equivalence='PASS',source_and_prior_artifacts_unchanged=True,
        validation_fallback=int(val.fallback.sum()),test_fallback=int(test.fallback.sum()),
        label='Retrospective/exploratory; oracle-filtered; not portfolio PnL')
    write_json(out/'completion.json',status)
    pooled=display(native[native.scope=='pooled'])
    parts=display(native[(native.split=='test')&(native.scope!='pooled')])
    report='''# Regime-A — benchmark trên validation/test BTC/ETH gốc

Giữ MLP, probabilities, labels, direction và phí 1 bps; không train lại. Chọn beta trên toàn bộ 80.260 validation samples, minimum coverage pooled 10%; không constraint từng asset/direction hoặc test. Đánh giá trên cùng 65.182 test samples cũ. Không chia nhỏ validation để thay bảng baseline.

## Công thức và lựa chọn

Trend = log(P(t)/P(t−360)) / sqrt(sum của 360 log-returns bình phương); z=tanh(trend/2). Threshold = clip(0,64 + beta*z, 0,50, 0,95). Dùng 360 dòng theo từng symbol, không phải 360 phút cố định; tính từ toàn lịch sử unified, không rolling trên cohort lọc Neutral. Thiếu trend thì dùng fixed. Đã đối chiếu giá trị trend với exploratory artifact tuyệt đối và prefix-invariance trên dữ liệu thật: PASS.

Grid beta −0,10..+0,10 bước 0,02; chọn tối đa mean net bps/mẫu chọn −100*beta², coverage validation pooled ≥10%. Không sweep lại window, scale hoặc penalty. Fixed tau* tối đa raw mean net trên grid 0,50..0,94 bước 0,02 với cùng coverage constraint. Khóa beta, tau*, mọi cutoff matching trước khi đọc test. Không chọn lại theo test.

'''+f'Beta được chọn: **{beta:.2f}**; fixed tau*: **{taustar:.2f}**.\n\n'+'''## Kết quả tổng hợp

Coverage/accuracy trong bảng là %. Net và expected net là bps. Objective có penalty chỉ dùng chọn beta; metrics báo return thực không trừ penalty.

'''+md_table(pooled)+'''\n\n## Test theo asset và predicted direction\n\n'''+md_table(parts)+'''

## Coverage và stability

coverage_matched_metrics.csv gồm targets 10%, 20%, 25,50%, 40%, pooled, từng asset, từng direction và asset×direction. Score fixed=c, score A=c−beta*z. Diagnostic matching không clip threshold, khác native rule; không refit beta. Cutoff được chọn từ validation score counts, không dùng labels để phá tie và không dùng test quantiles/top-k. attainable=False nếu plateau không cho coverage validation trong ±1 điểm phần trăm; actual_on_target báo coverage thực ở evaluation. Không gọi matched nếu actual coverages chưa tương đương, không coi target là actual coverage.

weekly_metrics.csv giữ mọi tuần có eligible samples, gắn few_samples nếu selections <200. Không chọn tuần tốt để thay beta. fixed_tau_star có thể giống fixed_064; không tạo đối chứng yếu hơn để làm đẹp adaptive.

## Integrity và giới hạn

Baseline native khớp metrics cũ ở validation/test pooled và từng coin: PASS. Hash unified, predictions validation, original model, source tác giả, proposals và các artifacts tham chiếu A/B cũ không đổi. completion.json lưu memory, runtime, cấu hình khóa và hash test predictions. Guard theo stage: 9 GiB peak working set / available RAM tối thiểu 2 GiB. Đây không phải continuous memory sampler.

Cohort đã oracle-filter theo |future return|>1%; horizon 600 dòng chồng lấn, thiếu purge; labels/calibration/threshold reuse được giữ giống baseline. Test đã từng được xem. Tất cả kết quả retrospective/exploratory, không phải confirmatory OOS; sample net returns không phải portfolio PnL hay vị thế độc lập. Không triển khai Regime-B trong lượt này.

## Tái lập và artifacts

Lệnh: `reproduction_setup/venv/Scripts/python.exe research_regime_threshold/run_regime_a.py` (từ chối overwrite; dùng `--out` tới thư mục mới để tái lập). Gồm beta_grid_full_validation.csv, fixed_tau_grid_validation.csv, locked_config_before_test.json, metrics_validation_test.csv, weekly_metrics.csv, coverage_matched_metrics.csv, macro_average_metrics.csv, threshold_distributions.csv, validation/test_decisions.parquet, selection_and_coverage.png, completion.json và resources.jsonl.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(grid.to_string(index=False),flush=True)
    print(native[native.scope.isin(['pooled','BTC_USDT','ETH_USDT'])].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'research_regime_threshold/method_a_20261103')
    main(parser.parse_args().out)
