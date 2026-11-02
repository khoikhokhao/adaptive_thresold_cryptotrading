"""Run the author's micro pipeline for one complete symbol, as a guarded stage."""
import argparse,ast,gc,json,time,sys,inspect,textwrap
from pathlib import Path
import pandas as pd
from memory_preprocessing import ROOT,config_for,native

def main():
 p=argparse.ArgumentParser();p.add_argument('--symbol',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preserve-precision',action='store_true');a=p.parse_args()
 if a.symbol not in {'BTC_USDT','ETH_USDT','SOL_USDT','DOGE_USDT','BNB_USDT'}:raise ValueError('Unexpected symbol')
 if a.output.exists():raise RuntimeError('Refuse overwrite')
 a.output.mkdir(parents=True);cfg=config_for(ROOT,a.output);cfg.save_debug_info=False
 f=ROOT/'data/micro'/f'{a.symbol}.csv';start=time.monotonic()
 chunks=[]
 for c in pd.read_csv(f,chunksize=cfg.chunk_size):c['Symbol']=a.symbol;chunks.append(c)
 df=pd.concat(chunks,ignore_index=True);del chunks
 df['timestamp']=pd.to_datetime(df['time_exchange_minute'],errors='coerce')
 proc=native.MicroDataProcessor(cfg);proc.data=df;del df;gc.collect()
 print('clean start',a.symbol,flush=True);proc.clean_data();print('feature start',len(proc.data),flush=True)
 # pandas groupby.apply pivots the return into a DataFrame for exactly one
 # group. Replace only that expression with its equivalent grouped diff/shift.
 # Verified bit-exact on 20,000 actual BTC/ETH rows before execution.
 tree=ast.parse(textwrap.dedent(inspect.getsource(native.MicroDataProcessor.engineer_features)))
 fn=tree.body[0]
 target=next(node for node in fn.body if isinstance(node,ast.If) and 'total_bids_volume' in ast.unparse(node.test))
 target.body=ast.parse("df['ofi_proxy'] = df.groupby('Symbol')['total_bids_volume'].diff().sub(df.groupby('Symbol')['total_asks_volume'].diff()).groupby(df['Symbol']).shift(1)").body
 ns=dict(vars(native))
 if a.preserve_precision:
  # The compiled function resolves globals in this namespace, not in the
  # original module. Deferring downcast must therefore happen here.
  ns['optimize_dtypes']=lambda df,*args,**kwargs:df
 exec(compile(ast.fix_missing_locations(tree),'<native-single-symbol-ofi>', 'exec'),ns)
 proc.engineer_features=ns['engineer_features'].__get__(proc,type(proc))
 if a.preserve_precision:
  # The author downcasts across the entire universe. Defer that one operation
  # until all symbols are available, otherwise each symbol loses precision.
  proc.engineer_features()
  proc.data.to_parquet(a.output/'processed_data_micro.parquet',index=False)
 else:
  proc.engineer_features();proc.save_processed_data('_micro')
 print(json.dumps(dict(symbol=a.symbol,rows=len(proc.data),seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
