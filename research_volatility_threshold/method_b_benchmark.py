"""Fit conditional correctness odds on original validation, lock, then compare on test."""
from pathlib import Path
import json
import time
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from method_a import ROOT, SYMBOLS, metrics, fit_cutoff, sha, write_json, memory, threshold

PREVIOUS=ROOT/'research_volatility_threshold/full_validation_test_20261103'
OUT=ROOT/'research_volatility_threshold/method_b_benchmark_20261103'
LAM=.01
KAPPAS=[i/100 for i in range(50,95,2)]
METHODS=['fixed_064','method_a','method_b','b_symbol_only']

def design(df, full=True):
    ell=logit(np.clip(df.confidence.to_numpy(),1e-6,1-1e-6))
    btc=(df.symbol=='BTC_USDT').to_numpy(dtype=float)
    eth=(df.symbol=='ETH_USDT').to_numpy(dtype=float)
    h=df.h.to_numpy()
    X=np.column_stack([btc,eth,h,h*ell] if full else [btc,eth])
    return X,ell

def fit(df,full=True):
    X,ell=design(df,full)
    y=df.correct.to_numpy()
    sizes=df.groupby('symbol').symbol.transform('size').to_numpy()
    weights=.5/sizes
    assert abs(weights.sum()-1)<1e-12
    def objective(theta):
        eta=ell+X@theta
        loss=np.dot(weights,np.logaddexp(0,eta)-y*eta)+LAM*np.dot(theta,theta)
        grad=X.T@(weights*(expit(eta)-y))+2*LAM*theta
        return loss,grad
    # Check analytic gradient with actual validation observations; no fitting on test.
    start=np.zeros(X.shape[1])
    loss,gradient=objective(start)
    numerical=[]
    for j in range(len(start)):
        step=np.zeros_like(start);step[j]=1e-6
        numerical.append((objective(start+step)[0]-objective(start-step)[0])/2e-6)
    np.testing.assert_allclose(gradient,numerical,atol=1e-8,rtol=1e-5)
    bounds=[(None,None)]*X.shape[1]
    if full: bounds[-1]=(-.25,.25)
    result=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
                    options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Optimizer failure; will not retry other configs: '+str(result.message))
    return dict(coefficients=result.x.tolist(),loss=float(result.fun),iterations=int(result.nit),
                success=bool(result.success),message=str(result.message),full=full,
                gradient_check='PASS',delta_at_bound=bool(full and abs(result.x[-1])>=.25-1e-7))

def probabilities(df,model):
    X,ell=design(df,model['full'])
    return expit(ell+X@np.array(model['coefficients']))

def select_kappa(df,q):
    rows=[]
    for kappa in KAPPAS:
        m=metrics(df,q>=kappa)
        rows.append(dict(kappa=kappa,feasible=m['coverage']>=.1,**m))
    table=pd.DataFrame(rows)
    feasible=table[table.feasible]
    if feasible.empty: raise RuntimeError('No feasible kappa at coverage >=10%; no test evaluation')
    best=feasible.net_selected_bps.max()
    tied=feasible[abs(feasible.net_selected_bps-best)<=1e-8].copy()
    tied['distance']=(tied.kappa-.64).abs()
    kappa=float(tied.sort_values(['distance','kappa']).iloc[0].kappa)
    return kappa,table

def md_table(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))

def pretty_metrics(df):
    cols=['split','method','symbol','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps']
    z=df[cols].copy()
    for c in ['coverage','selective_accuracy']:z[c]=(100*z[c]).round(2)
    for c in ['net_selected_bps','expected_net_bps']:z[c]=z[c].round(3)
    return z

def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    def event(stage):
        m=memory()
        with (OUT/'resources.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,elapsed_sec=time.monotonic()-started,**m))+'\n')
        if m['peak_ws_gib']>9 or m['available_gib']<2: raise RuntimeError('Memory guard')
        print(stage,flush=True)
    event('start')
    retained={p.name:sha(p) for p in PREVIOUS.iterdir() if p.is_file()}
    write_json(OUT/'previous_results_sha256.json',retained)
    previous=json.loads((PREVIOUS/'locked_config_before_test.json').read_text())
    tau,beta=previous['tau_fixed'],previous['beta']
    df=pd.read_parquet(PREVIOUS/'validation_decisions.parquet')
    assert len(df)==80260 and not df.duplicated(['symbol','timestamp']).any()
    assert np.isfinite(df[['confidence','h','correct','net_bps']]).all().all()
    models={}
    kappas={}
    val_scores={'fixed_064':df.confidence.to_numpy(),'method_a':(df.confidence-beta*df.h).to_numpy()}
    for method,full in [('method_b',True),('b_symbol_only',False)]:
        model=fit(df,full)
        q=probabilities(df,model)
        kappa,grid=select_kappa(df,q)
        grid.to_csv(OUT/(method+'_kappa_grid_validation.csv'),index=False)
        models[method]=model;kappas[method]=kappa;val_scores[method]=q
    event('B_and_symbol_ablation_fitted_on_validation')
    # Matched-coverage rules are selected using validation score counts only, no test ranks.
    rules=[]
    for scope in ['pooled']+SYMBOLS:
        sm=np.ones(len(df),bool) if scope=='pooled' else (df.symbol==scope).to_numpy()
        for method in METHODS:
            for target in [.1,.2,.255,.4]:
                rules.append(dict(scope=scope,method=method,target=target,**fit_cutoff(val_scores[method][sm],target)))
    lock=dict(model_b=models,kappas=kappas,lambda_l2=LAM,kappa_grid=KAPPAS,
        fixed_tau=tau,method_a_beta=beta,normalization=previous['normalization'],matched_rules=rules,
        selection_n=80260,coverage_constraint='pooled validation >= 10%; no test coverage constraint',
        objective='maximum mean net bps per selected sample on full validation',
        coefficients_order=['a_BTC','a_ETH','b','delta'],symbol_only_order=['a_BTC','a_ETH'],
        test_not_read_at_lock=True,versions={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        source_previous_results_sha256=retained)
    write_json(OUT/'locked_config_before_test.json',lock)
    lockhash=sha(OUT/'locked_config_before_test.json')
    event('all_parameters_and_cutoffs_locked_before_test_read')
    # No further fit, optimizer, cutoff selection, or parameter updates beyond this point.
    test=pd.read_parquet(PREVIOUS/'test_decisions.parquet')
    assert len(test)==65182 and not test.duplicated(['symbol','timestamp']).any()
    assert set(test.symbol)==set(SYMBOLS)
    assert np.isfinite(test[['confidence','h','correct','net_bps']]).all().all()
    results=[];matched=[];weekly=[];threshold_stats=[]
    for split,z in [('validation',df),('test',test)]:
        scores={'fixed_064':z.confidence.to_numpy(),'method_a':(z.confidence-beta*z.h).to_numpy()}
        choices={'fixed_064':(z.confidence>=tau).to_numpy(),
                 'method_a':(z.confidence>=threshold(z.h,beta,tau)).to_numpy()}
        decisions=z[['symbol','timestamp','confidence','h','correct','net_bps']].copy()
        for method in ['method_b','b_symbol_only']:
            q=probabilities(z,models[method]);scores[method]=q;choices[method]=q>=kappas[method]
            decisions[method+'_q_correct']=q
            coef=models[method]['coefficients']
            a=np.where(z.symbol=='BTC_USDT',coef[0],coef[1])
            b,delta=coef[2:] if models[method]['full'] else (0.,0.)
            raw=expit((logit(kappas[method])-a-b*z.h)/(1+delta*z.h))
            effective=np.maximum(.5,raw)
            decisions[method+'_effective_tau']=effective
            for s in SYMBOLS:
                t=effective[z.symbol==s]
                threshold_stats.append(dict(split=split,method=method,symbol=s,
                    minimum=float(t.min()),median=float(t.median()),p90=float(t.quantile(.9)),maximum=float(t.max())))
        for method in METHODS:
            decisions[method+'_selected']=choices[method]
            for s in ['pooled']+SYMBOLS:
                sm=np.ones(len(z),bool) if s=='pooled' else (z.symbol==s).to_numpy()
                results.append(dict(split=split,method=method,symbol=s,**metrics(z[sm],choices[method][sm])))
            week=z.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
            for w in week.unique():
                for s in SYMBOLS:
                    sm=((week==w)&(z.symbol==s)).to_numpy()
                    weekly.append(dict(split=split,method=method,symbol=s,week=w,**metrics(z[sm],choices[method][sm])))
        for rule in rules:
            sm=np.ones(len(z),bool) if rule['scope']=='pooled' else (z.symbol==rule['scope']).to_numpy()
            m=metrics(z[sm],scores[rule['method']][sm]>=rule['cutoff'])
            matched.append(dict(split=split,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=.01))
        decisions.to_parquet(OUT/(split+'_decisions.parquet'),index=False)
    results=pd.DataFrame(results);matched=pd.DataFrame(matched)
    results.to_csv(OUT/'metrics_validation_test.csv',index=False)
    matched.to_csv(OUT/'coverage_matched_metrics.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_metrics.csv',index=False)
    pd.DataFrame(threshold_stats).to_csv(OUT/'threshold_distributions.csv',index=False)
    results[results.symbol!='pooled'].groupby(['split','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean().to_csv(OUT/'macro_average_metrics.csv')
    old=pd.read_csv(PREVIOUS/'metrics_validation_test.csv')
    compare=results[results.method.isin(['fixed_064','method_a'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[compare.columns],compare,check_dtype=False,atol=1e-10,rtol=1e-12)
    for p in PREVIOUS.iterdir():
        if p.is_file(): assert sha(p)==retained[p.name],str(p)+' changed'
    assert sha(OUT/'locked_config_before_test.json')==lockhash
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names={'fixed_064':'Fixed 0.64','method_a':'Method A','method_b':'Method B','b_symbol_only':'B symbol-only control'}
    for method in METHODS:
        z=matched[(matched.split=='test')&(matched.scope=='pooled')&(matched.method==method)].sort_values('coverage')
        axes[0].plot(z.coverage*100,(1-z.selective_accuracy)*100,marker='o',label=names[method])
        axes[1].plot(z.coverage*100,z.net_selected_bps,marker='o',label=names[method])
    axes[0].set(xlabel='Actual test coverage (%)',ylabel='Selective error (%)',title='Locked validation cutoffs; retrospective')
    axes[1].set(xlabel='Actual test coverage (%)',ylabel='Net bps / selected sample',title='No test coverage re-fitting')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'risk_profit_coverage.png',dpi=160);plt.close(fig)
    event('complete_original_A_results_unchanged')
    status=dict(elapsed_sec=time.monotonic()-started,memory=memory(),locked_config_sha256=lockhash,
        original_A_metrics_equivalence='PASS',original_A_artifact_hashes_unchanged=True,
        label='Retrospective/exploratory; oracle-filtered; not portfolio PnL')
    write_json(OUT/'completion.json',status)
    pooled=pretty_metrics(results[results.symbol=='pooled'])
    perasset=pretty_metrics(results[(results.split=='test')&(results.symbol!='pooled')])
    text='''# Fixed 0,64, Method A và Method B — cùng validation/test BTC/ETH

## Protocol

Giữ nguyên MLP, probabilities gốc, labels, phí 1 bps và cohort baseline: validation 80.260, test 65.182 mẫu. Lưu nguyên toàn bộ kết quả A ở thư mục full_validation_test_20261103; hash mọi artifact trước và sau chạy, kiểm tra metrics A/Fixed khớp kết quả cũ (PASS).

Fit B trên toàn validation, chọn κ trên toàn validation theo mean net bps/mẫu chọn cao nhất với coverage pooled ≥10%; không ràng buộc từng coin hoặc test. Median/IQR volatility tái sử dụng từ A, không fit trên test. Model meta-selection B khác với MLP: chỉ fit 4 hệ số correctness, không train lại MLP hoặc thay hướng dự đoán.

## Công thức và cấu hình

`h = tanh((volatility_30m − median_symbol)/IQR_symbol)`; `l = logit(clip(confidence, 1e−6, 1−1e−6))`.

`q_correct = sigmoid((1 + delta*h)*l + a_symbol + b*h)`; chọn khi `q_correct >= kappa`. Đây là xác suất correctness có điều kiện, không phải probability Up mới. |delta|≤0,25 bảo toàn tính đơn điệu theo confidence. Loss: binary cross-entropy correctness, tổng trọng số mỗi coin 0,5, cộng 0,01 × tổng bình phương hệ số; λ không tune. L-BFGS-B với gradient giải tích đã đối chiếu finite differences trên dữ liệu thật. Khởi tạo zero, không retry cấu hình khác. Grid κ=0,50–0,94 bước 0,02; chọn trên validation. Hệ số, κ và toàn bộ cutoff coverage-matched được lưu trước khi đọc test; không refit sau test.

Đối chứng `b_symbol_only`: cùng fit/selection nhưng b=delta=0, còn hai intercept asset. Đây là ablation nhằm tách hiệu ứng volatility khỏi hiệu chỉnh theo symbol. Method A beta=−0,06 và fixed τ=0,64 giữ nguyên.

## Hệ số và trạng thái fit

'''+json.dumps(dict(models=models,kappas=kappas),ensure_ascii=False,indent=2)+'''

## Metrics tổng hợp

Coverage và selective accuracy là %. Net là bps. `expected_net_bps = coverage × net_selected_bps`. Các method không có coverage bằng nhau ở operating point native.

'''+md_table(pooled)+'''

## Test theo asset

'''+md_table(perasset)+'''

## Coverage-matched và thời gian

coverage_matched_metrics.csv gồm cutoff chọn bằng validation, targets 10%, 20%, 25,5%, 40%, pooled và từng symbol. Fixed score=confidence; A score=confidence−beta*h (diagnostic không clip); B score=q_correct. Không dùng test quantile/top-k. Khi plateau khiến target không khả thi trong ±1 điểm phần trăm, attainable=False và có hai coverage khả thi kề nhau; actual_on_target cho biết drift trên test. Chỉ coi matched nếu cả hai method đều đạt tolerance thực tế trên cùng scope; biểu đồ không chứng minh ranking tốt hơn khi actual coverage khác nhau.

weekly_metrics.csv báo stability theo tuần và asset; mọi số này chỉ để đánh giá sau khi khóa, không chọn lại method hoặc tham số. Mẫu chọn <200/tuần/asset được gắn few_samples. Không kiểm định IID hoặc công bố statistical significance vì labels và returns chồng lấn.

## Giới hạn

Oracle filtering |future return|>1%, horizon 600 dòng, overlap, thiếu purge và calibration/threshold cùng dùng validation được giữ nguyên để so sánh đồng nhất baseline. B fit reliability và chọn κ trên cùng validation cũng có nguy cơ overfit. Test đã được xem ở các lần trước, nên mọi kết quả là retrospective/exploratory. Net/mẫu không phải portfolio PnL hay lợi nhuận annualized. Không tuyên bố novelty hoặc confirmatory OOS. Không chọn lại cấu hình theo bảng test.

## Artifacts

Script: research_volatility_threshold/method_b_benchmark.py. Lệnh: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/method_b_benchmark.py` (từ chối overwrite thư mục kết quả). Có locked_config_before_test.json, hai kappa grids validation, metrics_validation_test.csv, coverage_matched_metrics.csv, weekly_metrics.csv, macro_average_metrics.csv, threshold_distributions.csv, validation/test_decisions.parquet, risk_profit_coverage.png, previous_results_sha256.json, resources.jsonl và completion.json. Memory guard kiểm tra theo stage: peak working set ≤9 GiB và available ≥2 GiB.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(models=models,kappas=kappas),indent=2))
    print(results.to_string(index=False))

if __name__=='__main__':main()
