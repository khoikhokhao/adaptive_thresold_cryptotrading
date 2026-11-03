"""Validation-only Smooth Volatility Tilt. No model fit and no test reads."""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ['BTC_USDT', 'ETH_USDT']
GRID = [i / 100 for i in range(-10, 11, 2)]
TARGETS = [.10, .20, .255, .40]
TOL = .01

def memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [(s, ctypes.c_ulonglong) for s in
            ['total_phys', 'avail_phys', 'total_commit_limit', 'avail_commit', 'total_virtual', 'avail_virtual', 'extended']]
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(s, ctypes.c_size_t) for s in
            ['peak_ws', 'ws', 'peak_pool_paged', 'pool_paged', 'peak_pool_nonpaged', 'pool_nonpaged', 'pagefile', 'peak_pagefile', 'private']]
    s, c = Status(), Counters()
    s.length, c.cb = ctypes.sizeof(s), ctypes.sizeof(c)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    assert ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(c), c.cb)
    return {k: v / 2**30 for k, v in {'ws_gib': c.ws, 'peak_ws_gib': c.peak_ws,
        'private_commit_gib': c.private, 'available_gib': s.avail_phys, 'commit_headroom_gib': s.avail_commit}.items()}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def threshold(h, beta, tau):
    return np.clip(tau + beta * np.asarray(h), .5, .95)

def metrics(df, chosen):
    chosen = np.asarray(chosen, dtype=bool)
    n, k = len(df), int(chosen.sum())
    return dict(n=n, selected_n=k, coverage=k/n if n else np.nan,
        accuracy=df.correct.mean() if n else np.nan,
        selective_accuracy=df.correct.to_numpy()[chosen].mean() if k else np.nan,
        net_selected_bps=df.net_bps.to_numpy()[chosen].mean() if k else np.nan,
        expected_net_bps=df.net_bps.to_numpy()[chosen].sum()/n if n else np.nan,
        selected_days=df.loc[chosen, 'timestamp'].dt.floor('D').nunique(), few_samples=k<200)

def fit_cutoff(scores, target):
    """Deterministic threshold; never break confidence ties. Uses no labels/returns."""
    x = np.asarray(scores)
    vals, counts = np.unique(x, return_counts=True)
    vals, counts = vals[::-1], counts[::-1]
    cuts = np.r_[np.nextafter(vals[0], np.inf), vals]
    cov = np.r_[0., np.cumsum(counts)/len(x)]
    best = int(np.argmin(np.abs(cov-target)))
    lo = float(cov[cov <= target].max())
    hi = float(cov[cov >= target].min())
    return dict(cutoff=float(cuts[best]), fit_coverage=float(cov[best]),
        lower_coverage=lo, upper_coverage=hi, attainable=bool(abs(cov[best]-target)<=TOL))

def self_checks():
    h = np.array([-1., 0., 1.])
    assert np.array_equal(threshold(h, 0., .64), np.full(3, .64))
    assert np.all(np.diff(threshold(h, .1, .64))>0)
    assert np.all(np.diff(threshold(h, -.1, .64))<0)
    assert not fit_cutoff([.5]*8+[.8]*2, .5)['attainable']
    z = fit_cutoff([.5]*8+[.8]*2, .2)
    assert z['attainable'] and z['cutoff']==.8

def finish_infeasible(out):
    """Complete documentation from saved selection artifacts, without reading check outcomes."""
    grid=pd.read_csv(out/'beta_grid_select.csv')
    assert not grid.feasible.any()
    split=pd.read_csv(out/'split_counts.csv')
    resources=[json.loads(line) for line in (out/'resources.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for s in SYMBOLS:
        axes[0].plot(grid.beta,100*grid[s+'_coverage'],marker='o',label=s)
    axes[0].axhline(10,linestyle='--',color='gray',label='Required per symbol')
    axes[0].set(xlabel='beta',ylabel='Coverage (%)',title='V_select: every candidate infeasible')
    axes[0].legend()
    axes[1].plot(grid.beta,grid.net_selected_bps,marker='o')
    axes[1].axvline(0,linestyle='--',color='gray')
    axes[1].set(xlabel='beta',ylabel='Net bps / selected sample',title='Objective alone also favors beta = 0')
    fig.tight_layout();fig.savefig(out/'selection_grid.png',dpi=160);plt.close(fig)
    def table(frame):
        return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    tab=grid[['beta','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps','BTC_USDT_coverage','ETH_USDT_coverage']].copy()
    for c in ['coverage','selective_accuracy','BTC_USDT_coverage','ETH_USDT_coverage']: tab[c]=(100*tab[c]).round(2)
    for c in ['net_selected_bps','expected_net_bps']: tab[c]=tab[c].round(3)
    report='''# Method A — kết quả validation BTC/ETH

**Trạng thái: không có β khả thi; chưa chọn adaptive threshold.** Không train lại MLP, không đọc predictions test. Không đánh giá performance V_check và không chạy coverage-matched vì chưa có β thỏa protocol đã khóa. Không thay baseline τ=0,64 hoặc tự nới constraint.

## Cấu hình

Volatility 30 dòng đã lưu, chuẩn hóa bằng median/IQR theo coin chỉ trên V_fit. h = tanh((v−median)/IQR); threshold = clip(τ_fixed + βh, 0,50, 0,95). Dùng chính xác τ_fixed trong selection.json (0.6400000000000001). Grid β từ −0,10 đến +0,10, bước 0,02. Objective: mean net bps/mẫu chọn; phí 1 bps. Coverage tổng và **từng coin** phải ≥10%; ít nhất 200 mẫu chọn/coin. Mọi hệ số dùng chung BTC/ETH.

Ranh giới theo timestamp chung, 50%/25%/25% khoảng lịch validation. Phục hồi future timestamp bằng shift −600 dòng riêng từng symbol trên unified; kiểm tra future price khớp tuyệt đối float32 với predictions và ret_bps khớp tới 1e−10 bps. Purge mẫu có endpoint label vượt ranh giới đoạn, không thay horizon theo dòng bằng 600 phút lịch.

## Chia dữ liệu và purge

'''+table(split)+'''

Tổng sau purge: V_fit 36.411, V_select 19.061, V_check 23.708; tổng loại 1.080 mẫu. V_check chỉ được xác định membership theo timestamp, chưa dùng correctness/return để đánh giá hoặc chọn β. Các outcome của toàn validation đã được xem trong Bước 1, nên việc chia lại không tạo holdout độc lập.

## Grid trên V_select

Các cột coverage và accuracy trong bảng là %. Returns là bps. BTC coverage / ETH coverage tính trên mẫu số riêng từng coin.

'''+table(tab)+'''

**Nguyên nhân dừng:** ETH coverage chỉ 5,48–5,66%, thấp hơn 10% cho mọi β; số mẫu ETH 533–550 vẫn vượt minimum 200. Coverage tổng 18,16–19,24% đều vượt 10%. Nếu chỉ ràng buộc coverage tổng thì các ứng viên sẽ hợp lệ, nhưng đó là một protocol khác. Không thay protocol sau khi thấy kết quả.

Ngay cả khi chỉ nhìn objective và bỏ qua feasibility, β=0 đạt mean net cao nhất 183,929 bps/mẫu chọn; không có β khác cải thiện metric này trên V_select. Tuy nhiên β=−0,10 đạt expected net/mẫu đủ điều kiện 34,998 bps, cao hơn 33,406 ở β=0 do chọn thêm mẫu; đây là objective khác, không dùng để đổi nghiệm sau thử nghiệm. Không thể kết luận adaptive thất bại ở mọi hoàn cảnh hoặc trên test.

## Giới hạn và bước tiếp theo

Cohort oracle-filtered |future return|>1%; kết quả không phải backtest có thể giao dịch. Labels 600 dòng chồng lấn; không coi selections là số vị thế độc lập. Calibration/model selection và fixed threshold cũ dùng validation; toàn bộ nghiên cứu vẫn exploratory. Không có significance IID hoặc annualized Sharpe.

Giữ fixed 0,64. Nếu tiếp tục nghiên cứu, cần chốt trước liệu minimum coverage áp dụng pooled hay từng coin, rồi mở một thí nghiệm mới có nhãn rõ; không thay kết quả lần này. Hiện tại **NO-GO cho việc chuyển sang adaptive đã chọn**, không phải kết luận NO-GO cho mọi volatility-adaptive method.

## Artifacts và tái lập

- Script: `research_volatility_threshold/method_a.py`.
- Chạy lại cùng protocol vào thư mục mới: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/method_a.py --out research_volatility_threshold/method_a_reproduction_NEW`.
- `beta_grid_select.csv`: đủ 11 ứng viên, mọi feasibility=False.
- `split_counts.csv`: timestamps, counts và purge.
- `selection_grid.png`: coverage và objective theo β.
- `resources.jsonl`: các lần kiểm tra working set, peak, private commit và available RAM; guard 9 GiB working set, dừng nếu available <2 GiB. Đây là kiểm tra theo stage, không phải continuous sampler.
- `status.json`: infeasible. Không có model mới hoặc selected adaptive configuration.

'''
    status=json.loads((out/'status.json').read_text())
    runtime=status.get('elapsed_sec',resources[-1]['elapsed_sec'])
    report+='Peak working set ghi nhận: '+str(round(max(r['peak_ws_gib'] for r in resources),3))+' GiB; available RAM thấp nhất được ghi nhận: '+str(round(min(r['available_gib'] for r in resources),3))+' GiB. Runtime ghi nhận: '+str(round(runtime,2))+' giây (không gồm bước hoàn thiện báo cáo/biểu đồ).\n'
    (out/'METHOD_A_VALIDATION_REPORT.md').write_text(report,encoding='utf-8')

def main(out, coverage_scope='each_symbol'):
    started = time.monotonic()
    self_checks()
    out.mkdir(parents=True, exist_ok=False)
    def event(stage):
        m = memory()
        with (out/'resources.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage, elapsed_sec=time.monotonic()-started, **m))+'\n')
        print(stage, m, flush=True)
        if m['ws_gib']>9 or m['peak_ws_gib']>9 or m['available_gib']<2:
            raise RuntimeError('Memory guard: 9 GiB process / 2 GiB available')
    event('start')
    predpath=ROOT/'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
    unified=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    selpath=ROOT/'data/btc_eth_1pct/coverage_10pct/selection.json'
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [predpath, unified, selpath]}
    original=json.loads(selpath.read_text())
    tau=original['tau']
    p=pd.read_parquet(predpath)
    assert len(p)==80260 and not p.duplicated(['symbol','timestamp']).any()
    assert set(p.symbol)==set(SYMBOLS)
    assert np.isfinite(p[['confidence','ret_bps']]).all().all()
    np.testing.assert_allclose(p.confidence, np.maximum(p.proba_up,p.proba_down), rtol=0, atol=1e-14)
    u=pd.read_parquet(unified, columns=['symbol','timestamp','mid_price','volatility_30m'])
    u=u.sort_values(['symbol','timestamp']).reset_index(drop=True)
    assert not u.duplicated(['symbol','timestamp']).any()
    u['future_timestamp']=u.groupby('symbol').timestamp.shift(-600)
    u['verify_future_mid']=u.groupby('symbol').mid_price.shift(-600).astype(np.float32)
    df=p.merge(u.drop(columns='mid_price'), on=['symbol','timestamp'], how='left', validate='one_to_one', indicator=True)
    assert (df._merge=='both').all() and df.future_timestamp.notna().all()
    np.testing.assert_array_equal(df.future_mid.to_numpy(),df.verify_future_mid.to_numpy())
    np.testing.assert_allclose(df.ret_bps,(df.future_mid/df.mid_price-1)*10000, rtol=0, atol=1e-10)
    assert (df.ret_bps.abs()>100).all()
    df['correct']=(df.pred_direction==df.y_direction).astype(float)
    df['net_bps']=np.where(df.pred_direction==1,df.ret_bps,-df.ret_bps)-1.
    base=metrics(df,df.confidence>=tau)
    assert base['selected_n']==original['validation_count']
    assert abs(base['net_selected_bps']-original['validation_mean_net_bps'])<.001
    del u,p
    event('join_and_label_endpoint_verified')
    start,end=df.timestamp.min(),df.timestamp.max()+pd.Timedelta(1,'ns')
    b1,b2=start+(end-start)*.5,start+(end-start)*.75
    bounds=[start,b1,b2,end]
    df['slice']='purged'
    splitrows=[]
    for i,name in enumerate(['fit','select','check']):
        raw=(df.timestamp>=bounds[i])&(df.timestamp<bounds[i+1])
        valid=raw&(df.future_timestamp<bounds[i+1])
        df.loc[valid,'slice']=name
        for s in SYMBOLS:
            sm=df.symbol==s
            splitrows.append(dict(slice=name,symbol=s,start=str(bounds[i]),end_exclusive=str(bounds[i+1]),
                before=int((raw&sm).sum()),retained=int((valid&sm).sum()),purged=int((raw&~valid&sm).sum())))
    pd.DataFrame(splitrows).to_csv(out/'split_counts.csv',index=False)
    norm={}
    df['h']=0.
    df['fallback']=False
    for s in SYMBOLS:
        v=df.loc[(df.slice=='fit')&(df.symbol==s),'volatility_30m']
        v=v[np.isfinite(v)]
        if len(v)==0: raise RuntimeError('No finite fit volatility for '+s)
        median,iqr=float(v.median()),float(v.quantile(.75)-v.quantile(.25))
        norm[s]=dict(median=median,iqr=iqr,n_fit=len(v))
        sm=df.symbol==s
        fallback=sm&(~np.isfinite(df.volatility_30m)|(iqr<=0))
        df.loc[fallback,'fallback']=True
        if iqr>0:
            ok=sm&~fallback
            df.loc[ok,'h']=np.tanh((df.loc[ok,'volatility_30m']-median)/iqr)
    write_json(out/'setup_manifest.json',dict(normalization=norm,inputs_sha256=inputs,
        beta_grid=GRID,tau_fixed=tau,split_bounds=list(map(str,bounds)),
        coverage_scope=coverage_scope,minimum_coverage=.1,
        minimum_selected_each_symbol=200 if coverage_scope=='each_symbol' else None))
    vs=df[df.slice=='select'].copy()
    assert len(vs)>0
    gridrows=[]
    for beta in GRID:
        chosen=vs.confidence>=threshold(vs.h,beta,tau)
        row=dict(beta=beta,**metrics(vs,chosen))
        feasible=row['coverage']>=.1
        for s in SYMBOLS:
            sm=vs.symbol==s
            m=metrics(vs[sm],chosen[sm])
            row[s+'_coverage'],row[s+'_n_selected']=m['coverage'],m['selected_n']
            if coverage_scope=='each_symbol':
                feasible=feasible and m['coverage']>=.1 and m['selected_n']>=200
        row['feasible']=bool(feasible)
        gridrows.append(row)
    grid=pd.DataFrame(gridrows)
    grid.to_csv(out/'beta_grid_select.csv',index=False)
    feasible=grid[grid.feasible]
    if feasible.empty:
        write_json(out/'status.json',{'status':'infeasible','reason':'No beta meets predeclared coverage/count constraints', 'elapsed_sec':time.monotonic()-started})
        event('infeasible_no_check_evaluation')
        finish_infeasible(out)
        return
    best=feasible.net_selected_bps.max()
    tied=feasible[abs(feasible.net_selected_bps-best)<=1e-8].copy()
    tied['abs_beta']=tied.beta.abs()
    beta=float(tied.sort_values(['abs_beta','beta']).iloc[0].beta)
    # Lock every decision BEFORE computing check-slice outcomes.
    rules=[]
    for scope in ['pooled']+SYMBOLS:
        z=vs if scope=='pooled' else vs[vs.symbol==scope]
        for method in ['fixed_score','adaptive_score']:
            scores=z.confidence.to_numpy()-(beta*z.h.to_numpy() if method=='adaptive_score' else 0.)
            for target in TARGETS:
                rules.append(dict(scope=scope,method=method,target=target,**fit_cutoff(scores,target)))
    lock=dict(beta=beta,tau_fixed=tau,normalization=norm,beta_grid=GRID,cost_bps=1,
        constraints={'coverage_scope':coverage_scope,'min_coverage':.1,
                     'min_selected_each_symbol':200 if coverage_scope=='each_symbol' else None},
        split_bounds=list(map(str,bounds)),matched_rules=rules,inputs_sha256=inputs,
        clipping=[.5,.95],coverage_tolerance=TOL,selection_objective='mean net bps per selected sample',
        limitations='Oracle-filtered; original calibrator/model/fixed threshold already used validation; retrospective only.')
    write_json(out/'locked_config.json',lock)
    event('beta_and_all_cutoffs_locked')
    native=[]
    weekly=[]
    for name in ['select','check']:
        z=df[df.slice==name]
        assert len(z)>0
        for method,b in [('fixed_064',0.),('method_a',beta)]:
            selected=z.confidence>=threshold(z.h,b,tau)
            for s in ['pooled']+SYMBOLS:
                sm=np.ones(len(z),dtype=bool) if s=='pooled' else (z.symbol==s).to_numpy()
                native.append(dict(slice=name,method=method,symbol=s,**metrics(z[sm],selected[sm])))
            week=z.timestamp.dt.tz_localize(None).dt.to_period('W-SUN').astype(str)
            for w in week.unique():
                for s in SYMBOLS:
                    sm=((week==w)&(z.symbol==s)).to_numpy()
                    weekly.append(dict(slice=name,method=method,symbol=s,week=w,**metrics(z[sm],selected[sm])))
    native=pd.DataFrame(native)
    native.to_csv(out/'native_metrics.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'weekly_metrics.csv',index=False)
    macro=native[native.symbol!='pooled'].groupby(['slice','method'])[['coverage','selective_accuracy','net_selected_bps','expected_net_bps']].mean()
    macro.to_csv(out/'macro_average_metrics.csv')
    matched=[]
    for name in ['select','check']:
        for rule in rules:
            z=df[df.slice==name]
            if rule['scope']!='pooled': z=z[z.symbol==rule['scope']]
            score=z.confidence-(beta*z.h if rule['method']=='adaptive_score' else 0.)
            m=metrics(z,score>=rule['cutoff'])
            matched.append(dict(slice=name,**rule,**m,actual_on_target=abs(m['coverage']-rule['target'])<=TOL))
    matched=pd.DataFrame(matched)
    matched.to_csv(out/'coverage_matched_metrics.csv',index=False)
    outputs=df[['symbol','timestamp','future_timestamp','slice','volatility_30m','h','fallback','confidence','correct','ret_bps','net_bps']].copy()
    outputs['tau_a']=threshold(df.h,beta,tau)
    outputs['selected_a']=df.confidence>=outputs.tau_a
    outputs['selected_fixed']=df.confidence>=tau
    outputs.to_parquet(out/'validation_decisions.parquet',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for s in SYMBOLS:
        q=grid
        axes[0].plot(q.beta,q[s+'_coverage']*100,marker='o',label=s)
    axes[0].set(xlabel='beta',ylabel='Coverage (%)',title='V_select: coverage constraints')
    axes[0].axhline(10,color='gray',linestyle='--')
    axes[1].plot(grid.beta,grid.net_selected_bps,marker='o')
    axes[1].axvline(beta,color='gray',linestyle='--')
    axes[1].set(xlabel='beta',ylabel='Net bps / selected sample',title='V_select: predeclared grid')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'selection_grid.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ['fixed_score','adaptive_score']:
        z=matched[(matched.slice=='check')&(matched.scope=='pooled')&(matched.method==method)].sort_values('coverage')
        axes[0].plot(100*z.coverage,100*(1-z.selective_accuracy),marker='o',label=method)
        axes[1].plot(100*z.coverage,z.net_selected_bps,marker='o',label=method)
    axes[0].set(xlabel='Actual coverage (%)',ylabel='Selective error (%)',title='V_check: locked validation cutoffs')
    axes[1].set(xlabel='Actual coverage (%)',ylabel='Net bps / selected sample',title='Coverage may differ; no re-tuning')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'locked_coverage_curves.png',dpi=160);plt.close(fig)
    event('completed')
    assert sha(predpath)==inputs[str(predpath.relative_to(ROOT))]
    assert sha(selpath)==inputs[str(selpath.relative_to(ROOT))]
    write_json(out/'status.json',dict(status='complete',beta=beta,elapsed_sec=time.monotonic()-started,
        fallback_count=int(df.fallback.sum()),self_checks='passed',memory=memory(),test_read=False,
        original_validation_baseline=base))
    print(native.to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'research_volatility_threshold/method_a_run_20261103')
    parser.add_argument('--coverage-scope',choices=['pooled','each_symbol'],default='each_symbol')
    args=parser.parse_args()
    main(args.out,args.coverage_scope)
