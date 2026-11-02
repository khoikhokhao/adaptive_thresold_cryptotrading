"""Separate storage-only variant; original research methods are imported unchanged.

Chunk CSV parsing uses the author's chunk size and pandas inference. Intermediate
Parquet preserves each chunk's types; concat preserves native dtype promotion.
Minute aggregation uses the same sorted row order and pandas groupby.last,
including per-column NaN semantics, but processes eight value columns at a time.
"""
import argparse
import ast
import gc
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_preprocessing as native


def config_for(work, output):
    tree = ast.parse((ROOT / 'data_preprocessing.py').read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
    assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
    config = eval(compile(ast.Expression(assignment.value), '<native config>', 'eval'), vars(native))
    config.macro_csv_path = work / config.macro_csv_path
    config.micro_data_dir = work / config.micro_data_dir
    config.output_dir = output
    return config


class StorageMicro(native.MicroDataProcessor):
    def load_data(self):
        staging = self.config.output_dir / 'staging'
        staging.mkdir()
        paths = []
        files = [f for f in self.config.micro_data_dir.glob('*.csv') if 'top_100' not in f.name.lower()]
        for number, file in enumerate(files):
            symbol = native.SymbolNormalizer.extract_symbol_from_filename(file)
            chunks = (pd.read_csv(file, chunksize=self.config.chunk_size)
                      if file.stat().st_size > 50 * 1024**2 else [pd.read_csv(file)])
            parts = []
            for i, chunk in enumerate(chunks):
                chunk['Symbol'] = symbol
                target = staging / f'{number}_{i}.parquet'
                chunk.to_parquet(target, index=False)
                parts.append(target)
            del chunks, chunk
            # Read one column at a time: no list of wide chunk frames is retained.
            import pyarrow.parquet as pq
            columns = pq.read_schema(parts[0]).names
            series = {}
            for column in columns:
                series[column] = pd.concat([pd.read_parquet(p, columns=[column])[column]
                                            for p in parts], ignore_index=True)
            frame = pd.DataFrame(series, copy=False)
            del series
            timestamp = next(c for c in ['time_exchange_minute', 'timestamp', 'time'] if c in frame)
            frame['timestamp'] = pd.to_datetime(frame[timestamp], errors='coerce')
            target = staging / f'symbol_{number}.parquet'
            frame.to_parquet(target, index=False)
            paths.append(target)
            del frame
            gc.collect()
        if not paths:
            raise native.DataQualityError('No micro files')
        columns = pq.read_schema(paths[0]).names
        series = {}
        for column in columns:
            series[column] = pd.concat([pd.read_parquet(p, columns=[column])[column]
                                        for p in paths], ignore_index=True)
        self.data = pd.DataFrame(series, copy=False)
        if self.data.Symbol.nunique() < 2:
            raise native.DataQualityError('Expected >=2 symbols')
        self.metadata['processing_steps'].append('loaded_micro_data')
        self.metadata['symbol_stats']['loaded'] = dict(unique_symbols=self.data.Symbol.nunique(),
            total_observations=len(self.data), files_processed=len(files))
        return self.data

    def _perform_minute_aggregation(self, df):
        if self.config.minute_aggregation_method != 'last':
            raise ValueError('Only verified last aggregation is supported')
        df['timestamp_minute'] = df.timestamp.dt.floor('T')
        order = df[['Symbol', 'timestamp']].sort_values(['Symbol', 'timestamp']).index
        keys = df.loc[order, ['Symbol', 'timestamp_minute']]
        columns = [c for c in df if c not in keys.columns]
        pieces = []
        for start in range(0, len(columns), 8):
            block = df.loc[order, columns[start:start+8]]
            block = pd.concat([keys, block], axis=1)
            pieces.append(block.groupby(['Symbol', 'timestamp_minute']).last())
        result = pd.concat(pieces, axis=1).reset_index()
        result['timestamp'] = result.timestamp_minute
        return result.drop(columns='timestamp_minute')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite or automatic retry')
    args.output.mkdir(parents=True)
    pipeline = native.FeatureEngineeringPipeline(config_for(args.work, args.output))
    pipeline.micro_processor = StorageMicro(pipeline.config)
    results = pipeline.run_full_pipeline()
    if not {'macro', 'micro', 'unified', 'train', 'validation'} <= results.keys():
        raise RuntimeError('Native pipeline reported incomplete outputs; inspect report')
    print(json.dumps({k: list(v.shape) for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
