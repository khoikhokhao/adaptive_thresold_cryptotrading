"""Train the author's native two-class MLP using freshly prepared cached arrays."""
import json,time,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from training_memory import ROOT,config,load_train_column,native
from run_mlp_baseline import ObservedTrainer
from common import sha256,memory

def main():
    prep=ROOT/'data/btc_eth_1pct/prepared'; out=ROOT/'data/btc_eth_1pct/training'
    if out.exists(): raise RuntimeError('Refuse overwrite or retry')
    dataset=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet'
    expected=json.loads((dataset.parent/'output_checksum.json').read_text())['sha256']
    if sha256(dataset)!=expected: raise RuntimeError('Dataset hash mismatch')
    manifest=json.loads((prep/'matrix_manifest.json').read_text())
    for name,info in manifest.items():
        if sha256(prep/name)!=info['sha256']: raise RuntimeError('Matrix hash mismatch '+name)
    cfg=config(dataset,out,1500000)
    saved=json.loads((prep/'config.json').read_text())
    assert saved['deadband_bps']==100 and saved['horizon_min']==600 and saved['confidence_grid_step']==.02
    frames=[pd.read_parquet(prep/f'{n}_metadata.parquet') for n in ['train','validation','test']]
    arrays=[np.load(prep/f'{p}_{n}.npy',mmap_mode='r') for n in ['train','validation','test'] for p in ['X','y']]
    assert [len(x) for x in frames]==[292338,80260,65182]
    for f,X,y in zip(frames,arrays[::2],arrays[1::2]):
        assert X.shape==(len(f),128) and X.dtype==np.float32 and np.isfinite(X).all()
        np.testing.assert_array_equal(y,f.y_direction.to_numpy())
    processor=joblib.load(prep/'feature_processor.joblib')
    mapping=pd.read_parquet(prep/'validation_mapping.parquet')
    # Native decision-threshold export re-transforms validation raw features.
    for col in processor.feature_names: frames[1][col]=load_train_column(dataset,col,mapping)
    np.testing.assert_array_equal(processor.transform(processor.select_features(frames[1])),arrays[2])
    out.mkdir(parents=True)
    trainer=ObservedTrainer(cfg);trainer.started=time.monotonic();trainer.feature_processor=processor
    processor.save_artifacts(out);joblib.dump(processor,out/'feature_processor.joblib')
    trainer.train_df_capped=frames[0]
    trainer._val_y_direction=frames[1].y_direction.to_numpy().astype(np.int8)
    trainer._val_ret_bps=frames[1].ret_bps.to_numpy().astype(np.float32)
    (out/'run_config.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
    trainer.event('training_start',samples=[len(x) for x in frames],features=128)
    results=trainer._train_mlp_model(*arrays,*frames)
    results['test_df']=frames[2]
    trainer._evaluate_and_save_results(results);trainer._plot_training_progress(out)
    trainer.event('training_complete')
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
