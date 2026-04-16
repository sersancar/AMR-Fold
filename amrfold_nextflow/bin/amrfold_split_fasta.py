#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
from typing import Iterator, Tuple, TextIO


def open_maybe_gzip(path: str) -> TextIO:
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path, 'rt')


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    with open_maybe_gzip(path) as fh:
        header = None
        seq_chunks = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description='Split FASTA into shard FASTA files.')
    ap.add_argument('--input', required=True, help='Input FASTA or FASTA.gz')
    ap.add_argument('--outdir', required=True, help='Output directory for shard FASTA files')
    ap.add_argument('--manifest', required=True, help='Output manifest TSV')
    ap.add_argument('--seqs-per-shard', type=int, default=256, help='Sequences per shard')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seqs_per_shard = max(1, int(args.seqs_per_shard))
    shard_idx = 0
    n_total = 0
    n_in_shard = 0
    shard_path = None
    shard_fh = None
    manifest_rows = []

    def start_new_shard() -> Tuple[Path, TextIO]:
        nonlocal shard_idx
        shard_idx += 1
        p = outdir / f'shard_{shard_idx:05d}.fa'
        return p, open(p, 'wt')

    try:
        for header, seq in read_fasta(args.input):
            if shard_fh is None or n_in_shard >= seqs_per_shard:
                if shard_fh is not None:
                    shard_fh.close()
                    manifest_rows.append((shard_path.name, n_in_shard))
                shard_path, shard_fh = start_new_shard()
                n_in_shard = 0

            shard_fh.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                shard_fh.write(seq[i:i+80] + '\n')
            n_in_shard += 1
            n_total += 1

        if shard_fh is not None:
            shard_fh.close()
            manifest_rows.append((shard_path.name, n_in_shard))
    finally:
        if shard_fh is not None and not shard_fh.closed:
            shard_fh.close()

    with open(args.manifest, 'wt') as mf:
        mf.write('shard\tn_sequences\n')
        for name, count in manifest_rows:
            mf.write(f'{name}\t{count}\n')

    if n_total == 0:
        raise SystemExit('No sequences found in input FASTA.')


if __name__ == '__main__':
    main()
