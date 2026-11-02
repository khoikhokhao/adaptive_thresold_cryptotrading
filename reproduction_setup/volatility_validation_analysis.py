"""Descriptive BTC/ETH validation analysis; never reads test predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/volatility_validation_btc_eth'
OUT.mkdir(parents=True, exist_ok=True)
PRED = ROOT / 'data/btc_eth_1pct/training/validation_predictions_complete.parquet'
UNIFIED = ROOT / 'data/join_full_btc_eth_20261102/unified_dataset.parquet'
features = [f'volatility_{w}m' for w in (5, 15, 30)] + [f'volatility_{w}d' for w in (5, 15, 30)]
extra = ['atr_14', 'volatility_regime']
p = pd.read_parquet(PRED)
assert len(p) == 80260 and not p.duplicated(['symbol', 'timestamp']).any()
assert set(p.symbol.unique()) == {'BTC_USDT', 'ETH_USDT'}
assert np.allclose(p.confidence, np.maximum(p.proba_up, p.proba_down))
assert (p.ret_bps.abs() > 100).all()
u = pd.read_parquet(UNIFIED, columns=['symbol', 'timestamp'] + features + extra)
assert not u.duplicated(['symbol', 'timestamp']).any()
df = p.merge(u, on=['symbol', 'timestamp'], how='left', validate='one_to_one', indicator=True)
assert (df._merge == 'both').all()
df['correct'] = (df.pred_direction == df.y_direction).astype(float)
df['net_bps'] = np.where(df.pred_direction == 1, df.ret_bps, -df.ret_bps) - 1.0
# Match selection artifact's 0.6400000000000001 boundary, avoiding a floating-point mismatch.
selection = json.loads((ROOT / 'data/btc_eth_1pct/coverage_10pct/selection.json').read_text())
tau = selection['tau']
df['selected'] = df.confidence >= tau
assert int(df.selected.sum()) == selection['validation_count']
assert np.isclose(df.loc[df.selected, 'correct'].mean(), selection['validation_accuracy'])
assert np.isclose(df.loc[df.selected, 'net_bps'].mean(), selection['validation_mean_net_bps'], atol=0.001)

schema = json.loads((ROOT / 'data/btc_eth_1pct/training/feature_schema.json').read_text())
names = schema['feature_names']
assert len(names) == 128
summary = {'n': len(df), 'tau': tau, 'matched': len(df), 'feature_in_mlp': {f: f in names for f in features + extra},
           'periods': {str(k): [str(g.timestamp.min()), str(g.timestamp.max())] for k, g in df.groupby('symbol')}}

rows, controls, plateaus = [], [], []
for feature in features:
    for symbol, s in df.groupby('symbol', sort=True):
        s = s.copy()
        s = s[s[feature].notna()].copy()
        # Never arbitrarily split ties; duplicate quantile edges reduce the number of groups.
        s['q'], edges = pd.qcut(s[feature], 5, labels=False, duplicates='drop', retbins=True)
        s['vrank'] = s[feature].rank(method='average', pct=True)
        for q, g in s.groupby('q'):
            selected = g[g.selected]
            rows.append(dict(feature=feature, symbol=symbol, quantile=int(q)+1, groups=len(edges)-1,
                lower=edges[int(q)], upper=edges[int(q)+1], n=len(g), accuracy=g.correct.mean(),
                selected_n=len(selected), coverage=g.selected.mean(), selected_accuracy=selected.correct.mean(),
                selected_net_bps=selected.net_bps.mean(), confidence_min=g.confidence.min(),
                confidence_median=g.confidence.median(), confidence_p90=g.confidence.quantile(.9),
                confidence_p95=g.confidence.quantile(.95), confidence_max=g.confidence.max(),
                confidence_mean=g.confidence.mean(), calibration_gap=g.correct.mean()-g.confidence.mean()))
        # Residualization on exact calibrated confidence plateau AND symbol.
        # Ranks are within symbol, calculated from validation for descriptive analysis only.
        grouped = s.groupby('confidence', sort=False)
        counts = grouped.confidence.transform('size')
        v = s.vrank - grouped.vrank.transform('mean')
        yc = s.correct - grouped.correct.transform('mean')
        yr = s.net_bps - grouped.net_bps.transform('mean')
        denom = np.dot(v, v)
        def corr(x):
            d = np.sqrt(denom*np.dot(x,x))
            return float(np.dot(v,x)/d) if d else np.nan
        controls.append(dict(feature=feature, symbol=symbol, n=len(s), n_missing=len(df[df.symbol==symbol])-len(s),
            strata=int(s.confidence.nunique()), n_in_repeated_strata=int((counts>=2).sum()),
            residual_rank_variance=float(denom), partial_corr_correct=corr(yc),
            slope_correct=float(np.dot(v,yc)/denom) if denom else np.nan,
            partial_corr_net=corr(yr), slope_net_bps=float(np.dot(v,yr)/denom) if denom else np.nan))
        s['day'] = s.timestamp.dt.floor('D')
        gd = s.groupby(['day', 'confidence'], sort=False)
        vd = s.vrank - gd.vrank.transform('mean')
        for outcome in ('correct', 'net_bps'):
            yd = s[outcome] - gd[outcome].transform('mean')
            dd = np.sqrt(np.dot(vd, vd) * np.dot(yd, yd))
            controls[-1]['partial_corr_' + outcome + '_also_day'] = float(np.dot(vd, yd)/dd) if np.dot(vd, vd) > 1e-16 and dd else np.nan
        mode = s.confidence.value_counts().index[0]
        for q,g in s[s.confidence == mode].groupby('q'):
            plateaus.append(dict(feature=feature, symbol=symbol, confidence=mode, quantile=int(q)+1,
                                 n=len(g), accuracy=g.correct.mean(), net_bps=g.net_bps.mean()))

pd.DataFrame(rows).to_csv(OUT/'quantiles.csv', index=False)
pd.DataFrame(controls).to_csv(OUT/'controlled_associations.csv', index=False)
pd.DataFrame(plateaus).to_csv(OUT/'most_common_confidence_plateau.csv', index=False)
(OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
tab = pd.DataFrame(rows)
for symbol, g in tab[tab.feature == 'volatility_30m'].groupby('symbol'):
    axs[0].plot(g['quantile'], 100*g.accuracy, marker='o', label=symbol)
    axs[0].plot(g['quantile'], 100*g.selected_accuracy, marker='x', linestyle='--', label=symbol+' selected')
    axs[1].plot(g['quantile'], 100*g.coverage, marker='o', label=symbol)
    axs[2].plot(g['quantile'], g.selected_net_bps, marker='o', label=symbol)
for ax,title,ylabel in zip(axs, ['Accuracy','Coverage at fixed threshold','Selected net return'], ['%','%','bps/sample']):
    ax.set(title=title, xlabel='Volatility 30-row quantile', ylabel=ylabel, xticks=[1,2,3,4,5])
    ax.grid(alpha=.2)
axs[0].legend(fontsize=7)
fig.suptitle('Oracle-filtered validation; descriptive only')
fig.tight_layout()
fig.savefig(OUT/'volatility_30m_groups.png', dpi=160)
print(json.dumps(summary, indent=2))
print('Wrote', OUT)
