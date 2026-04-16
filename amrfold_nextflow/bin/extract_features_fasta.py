#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from features import ProstT5Config, ProstT5FeatureExtractor


def str2bool(x):
    return str(x).strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def main():
    ap = argparse.ArgumentParser(description='Extract ProstT5/3Di features for one FASTA shard directly into LMDB.')
    ap.add_argument('--input_fasta', required=True)
    ap.add_argument('--lmdb_dir', required=True)
    ap.add_argument('--map_size_gb', type=float, default=8.0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--half_precision', default='true')
    ap.add_argument('--prostt5_dir', default=None)
    ap.add_argument('--local_files_only', default='false')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--token_budget', type=int, default=4096)
    ap.add_argument('--commit_every', type=int, default=100)
    args = ap.parse_args()

    model_name = args.prostt5_dir if args.prostt5_dir else 'Rostlab/ProstT5'
    cfg = ProstT5Config(
        model_name=model_name,
        device=args.device,
        half_precision=str2bool(args.half_precision),
        local_files_only=str2bool(args.local_files_only),
    )
    extractor = ProstT5FeatureExtractor(cfg)

    Path(args.lmdb_dir).mkdir(parents=True, exist_ok=True)
    extractor.encode_fasta_to_lmdb(
        fasta_path=args.input_fasta,
        lmdb_dir=args.lmdb_dir,
        map_size_gb=float(args.map_size_gb),
        batch_size=int(args.batch_size),
        token_budget=int(args.token_budget),
        commit_every=int(args.commit_every),
        overwrite=True,
        progress=True,
    )


if __name__ == '__main__':
    sys.exit(main())
