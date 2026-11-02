"""Native trainer with observation/checkpoint/export hooks; no fitting logic changes."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'baseline_reproduction'))
import centralized_training_two_classes as native
from common import memory


class ObservedTrainer(native.CentralizedTrainer):
    def event(self,stage,**extra):
        with (self.config.output_dir/'stage_resources.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(stage=stage,elapsed=time.monotonic()-self.started,**memory(),**extra))+'\n')

    def _load_data(self):
        self.event('load_start')
        result=super()._load_data()
        self.event('load_complete',shape=list(result.shape))
        return result

    def _create_targets(self,df):
        self.event('target_start')
        result=super()._create_targets(df)
        self.event('target_complete',shape=list(result.shape))
        return result

    def _split_data(self,df):
        self.event('split_start')
        result=super()._split_data(df)
        self.event('split_complete',shapes=[list(x.shape) for x in result])
        return result

    def _prepare_features(self,*args):
        self.event('features_start')
        result=super()._prepare_features(*args)
        joblib.dump(self.feature_processor,self.config.output_dir/'feature_processor.joblib')
        self.event('features_complete',shapes=[list(a.shape) for a in result])
        return result

    def _train_two_class_sklearn(self,X_train,y_train,X_val,y_val,X_test,y_test,train_df,val_df,test_df):
        from sklearn.neural_network import MLPClassifier
        original_fit=MLPClassifier.fit
        folder=self.config.output_dir
        (folder/'checkpoints').mkdir()
        calls=[]
        self.event('model_start',shapes=[list(x.shape) for x in [X_train,X_val,X_test]])
        def observed_fit(model,*args,**kwargs):
            begin=time.monotonic()
            self.event('fit_start',call=len(calls),requested_max_iter=model.max_iter,
                       shape=list(args[0].shape) if args else None)
            outcome=original_fit(model,*args,**kwargs)
            record=dict(call=len(calls),requested_max_iter=model.max_iter,actual_n_iter=model.n_iter_,
                        seconds=time.monotonic()-begin,loss=float(model.loss_))
            calls.append(record)
            joblib.dump(model,folder/'checkpoints'/f'fit_{len(calls):03d}.joblib')
            (folder/'fit_timings.json').write_text(json.dumps(calls,indent=2),encoding='utf-8')
            (folder/'actual_mlp_params.json').write_text(json.dumps(model.get_params(),indent=2),encoding='utf-8')
            self.event('fit_complete',**record)
            return outcome
        MLPClassifier.fit=observed_fit
        try:
            results=super()._train_two_class_sklearn(X_train,y_train,X_val,y_val,X_test,y_test,train_df,val_df,test_df)
        finally:
            MLPClassifier.fit=original_fit
        classifier=results['model']
        joblib.dump(classifier,folder/'classifier_complete.joblib')
        for name,frame,X in [('validation',val_df,X_val),('test',test_df,X_test)]:
            probability=classifier.predict_proba(X)
            prediction,confidence,execute=classifier.predict_with_confidence(X)
            columns=['symbol','timestamp','mid_price','future_mid','ret','ret_bps','y','y_binary','y_direction']
            exported=frame[columns].copy()
            exported['proba_down']=probability[:,0]
            exported['proba_up']=probability[:,1]
            exported['pred_direction']=prediction
            exported['confidence']=confidence
            exported['execute']=execute
            exported.to_parquet(folder/f'{name}_predictions_complete.parquet',index=False)
            metrics=native.MetricsCalculator('classification').calculate_two_class_metrics(
                y_val if name=='validation' else y_test,prediction,confidence,execute,
                frame.ret_bps.to_numpy(),self.config.profit_cost_bps)
            metrics['accuracy_all']=float((prediction==frame.y_direction.to_numpy()).mean())
            (folder/f'{name}_metrics_complete.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        self.train_df_capped[['symbol','timestamp','y_direction','ret_bps']].to_parquet(folder/'train_index_labels.parquet',index=False)
        self.event('model_and_exports_complete',fit_calls=len(calls))
        return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite or automatic retry')
    args.output.mkdir(parents=True)
    config=native.TrainingConfig(unified_data_path=ROOT/'data/join_full_btc_eth_20261102/unified_dataset.parquet',
        output_dir=args.output,task='classification',horizon_min=600,deadband_bps=10.,model_type='mlp',
        epochs=args.epochs,batch_size=4096,learning_rate=.001,validation_split=.15,test_split=.15,
        mlp_hidden_sizes=[256,128,64],dropout_rate=.2,early_stopping_patience=10,
        use_class_weights=True,use_two_stage=False,two_class_mode=True,confidence_tau=.8,
        profit_cost_bps=1.,max_train_size=999999999,top_k_features=128,use_robust_scaler=True,
        feature_selection_method='mutual_info',calibrate_probabilities=True,
        optimize_threshold_for_profit=False,seed=42,device='cpu',save_predictions=True)
    serial={k:str(v) if isinstance(v,Path) else v for k,v in asdict(config).items()}
    (args.output/'config_actual_requested.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
    native.validate_config(config)
    trainer=ObservedTrainer(config)
    trainer.started=time.monotonic()
    trainer.train()
    trainer.event('pipeline_complete')


if __name__=='__main__':
    main()
