#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge shard prediction tables into one TSV.')
    ap.add_argument('--output', required=True, help='Output TSV path')
    ap.add_argument('inputs', nargs='+', help='Input shard prediction TSV files')
    args = ap.parse_args()

    frames = []
    for fp in args.inputs:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f'Missing input prediction table: {fp}')
        df = pd.read_csv(p, sep='\t')
        df['_source_file'] = p.name
        frames.append(df)

    if not frames:
        raise SystemExit('No input prediction tables provided.')

    merged = pd.concat(frames, ignore_index=True)

    key_cols = [c for c in ['id', 'Ids', 'seq_id', 'header'] if c in merged.columns]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep='first')
    else:
        merged = merged.drop_duplicates(keep='first')

    merged.to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
