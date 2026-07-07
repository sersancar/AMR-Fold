#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

from features import ProstT5Config, ProstT5FeatureExtractor


def str2bool(x):
    return str(x).strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def read_manifest(path: str):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if 'shard' not in reader.fieldnames:
            raise ValueError(f"Manifest must contain a 'shard' column: {path}")
        for row in reader:
            shard = row['shard'].strip()
            if shard:
                rows.append(shard)
    if not rows:
        raise ValueError(f'No shards found in manifest: {path}')
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Extract ProstT5/3Di features for several FASTA shards using one persistent ProstT5 instance.'
    )
    ap.add_argument('--manifest', required=True, help='TSV with at least a shard column')
    ap.add_argument('--shards_dir', required=True, help='Directory containing shard FASTA files')
    ap.add_argument('--lmdb_root', required=True, help='Output directory where one LMDB dir per shard will be written')
    ap.add_argument('--map_size_gb', type=float, default=8.0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--half_precision', default='true')
    ap.add_argument('--prostt5_dir', default=None)
    ap.add_argument('--local_files_only', default='false')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--token_budget', type=int, default=4096)
    ap.add_argument('--commit_every', type=int, default=100)
    ap.add_argument('--skip_completed', default='true')
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    lmdb_root = Path(args.lmdb_root)
    lmdb_root.mkdir(parents=True, exist_ok=True)

    shard_names = read_manifest(args.manifest)

    model_name = args.prostt5_dir if args.prostt5_dir else 'Rostlab/ProstT5'
    cfg = ProstT5Config(
        model_name=model_name,
        device=args.device,
        half_precision=str2bool(args.half_precision),
        local_files_only=str2bool(args.local_files_only),
    )

    print(f'[GROUP_EXTRACT] Loading ProstT5 once for {len(shard_names)} shards', flush=True)
    extractor = ProstT5FeatureExtractor(cfg)

    skip_completed = str2bool(args.skip_completed)
    out_manifest = lmdb_root / 'extract_manifest.tsv'

    with open(out_manifest, 'w', newline='') as mf:
        writer = csv.writer(mf, delimiter='\t')
        writer.writerow(['shard', 'fasta', 'lmdb_dir', 'status'])

        for i, shard_name in enumerate(shard_names, start=1):
            fasta_path = shards_dir / shard_name
            if not fasta_path.exists():
                raise FileNotFoundError(f'Missing shard FASTA: {fasta_path}')

            lmdb_dir = lmdb_root / f'{Path(shard_name).stem}.lmdb'
            success_file = lmdb_dir / '_SUCCESS'

            if skip_completed and success_file.exists():
                print(f'[GROUP_EXTRACT] [{i}/{len(shard_names)}] SKIP completed {shard_name}', flush=True)
                writer.writerow([shard_name, str(fasta_path), str(lmdb_dir), 'skipped'])
                continue

            print(f'[GROUP_EXTRACT] [{i}/{len(shard_names)}] Extracting {shard_name} -> {lmdb_dir}', flush=True)
            lmdb_dir.mkdir(parents=True, exist_ok=True)
            extractor.encode_fasta_to_lmdb(
                fasta_path=str(fasta_path),
                lmdb_dir=str(lmdb_dir),
                map_size_gb=float(args.map_size_gb),
                batch_size=int(args.batch_size),
                token_budget=int(args.token_budget),
                commit_every=int(args.commit_every),
                overwrite=True,
                progress=True,
            )
            success_file.touch()
            writer.writerow([shard_name, str(fasta_path), str(lmdb_dir), 'done'])
            mf.flush()

    print(f'[GROUP_EXTRACT] Done. Manifest: {out_manifest}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
