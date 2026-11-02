"""Compare per-symbol preserved-precision features to the existing BTC/ETH run."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from memory_preprocessing import ROOT,native

def main():
 paths=[ROOT/'data/five_asset_work/precise_v2_BTC_USDT/processed_data_micro.parquet',
        ROOT/'data/five_asset_work/precise_v2_ETH_USDT/processed_data_micro.parquet']
 reference=ROOT/'data/join_full_btc_eth_20261102/processed_data_micro.parquet'
 schemas=[pq.read_schema(p) for p in paths];ref=pq.read_schema(reference)
 if any(s.names!=ref.names for s in schemas):raise RuntimeError('Column names/order mismatch')
 n=pq.ParquetFile(paths[0]).metadata.num_rows
 mid=pd.read_parquet(paths[0],columns=['mid_price']).mid_price.to_numpy()
 boundary=abs(mid[-1]-mid[-2])/mid[-2]
 failed=[];native.logger.disabled=True
 for start in range(0,len(ref.names),8):
  cols=ref.names[start:start+8]
  groups=[pd.read_parquet(p,columns=cols) for p in paths]
  original=pd.read_parquet(reference,columns=cols).reset_index(drop=True)
  for col in cols:
   x=pd.concat([g[col] for g in groups],ignore_index=True)
   if col=='price_deviation':x.iloc[n]=boundary
   x=native.optimize_dtypes(pd.DataFrame({col:x}))[col]
   y=original[col]
   try:
    if x.dtype!=y.dtype:raise AssertionError(f'dtype {x.dtype} != {y.dtype}')
    if pd.api.types.is_numeric_dtype(x.dtype):
     if not np.array_equal(x.to_numpy(),y.to_numpy(),equal_nan=True):raise AssertionError('numeric values differ')
    else:pd.testing.assert_series_equal(x,y,check_exact=True,check_names=False)
   except AssertionError as e:
    failed.append(dict(column=col,reason=str(e)[:400]));raise RuntimeError('Equivalence failure: '+str(failed[-1]))
  print('checked columns',start,'to',start+len(cols)-1,flush=True)
 out=ROOT/'data/five_asset_work/precision_equivalence.json'
 out.write_text(json.dumps(dict(passed=not failed,columns=len(ref),rows=len(y),failed=failed,boundary_value=boundary),indent=2),encoding='utf-8')
 if failed:raise RuntimeError('Precision equivalence failed: '+str(failed[:5]))
 print('TWO-SYMBOL GLOBAL PRECISION EQUIVALENCE PASS',flush=True)

if __name__=='__main__':main()
