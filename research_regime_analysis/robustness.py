"""Additional descriptive controls on cached validation rows, never model/test fitting."""
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path(__file__).resolve().parent/'validation_20261103'

def main():
    df=pd.read_parquet(OUT/'validation_regime_rows.parquet')
    rows=[]
    definitions=['trend_60','trend_360','trend_1440','efficiency_360','daily_ma_5_20']
    for name in definitions:
        for symbol,z in df.groupby('symbol'):
            z=z.copy()
            z['bull']=(z[name+'_regime']=='Bull').astype(float)
            z['bear']=(z[name+'_regime']=='Bear').astype(float)
            z['lv30']=np.log(z.volatility_30m.astype(np.float64).replace(0,np.nan))
            z['lv15']=np.log(z.volatility_15d.astype(np.float64).replace(0,np.nan))
            z=z.dropna(subset=['lv30','lv15'])
            for controls,keys in [('confidence_continuous_vol',['confidence']),
                                  ('confidence_continuous_vol_day',['confidence','day'])]:
                columns=['bull','bear','lv30','lv15','correct','net_bps']
                r=z[columns]-z.groupby(keys)[columns].transform('mean')
                names=[s for s in ['bull','bear','lv30','lv15'] if np.linalg.norm(r[s])>1e-8]
                X=r[names].to_numpy()
                rank=np.linalg.matrix_rank(X,tol=1e-8)
                for outcome in ['correct','net_bps']:
                    coef=np.linalg.lstsq(X,r[outcome],rcond=1e-10)[0]
                    for reg,namecol in [('Bull','bull'),('Bear','bear')]:
                        effect=float(coef[names.index(namecol)]) if namecol in names and rank==len(names) else np.nan
                        rows.append(dict(definition=name,symbol=symbol,controls=controls,outcome=outcome,
                            contrast=reg+' minus Sideways',effect=effect,rank=int(rank),columns=','.join(names),n=len(z)))
    pd.DataFrame(rows).to_csv(OUT/'continuous_volatility_controls.csv',index=False)
    print(pd.DataFrame(rows).query("definition=='trend_360'").round(4).to_string(index=False))

if __name__=='__main__':main()
