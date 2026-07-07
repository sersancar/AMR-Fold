#!/usr/bin/env python3
"""
Build an LMDB of precomputed ProstT5 features.

This is an optimized/safe-ish version of your original script:
- periodic commits (avoid one gigantic transaction)
- optional "fast IO" mode for much faster writes on shared filesystems
- optional map_size auto-estimation from .npy sizes
- CLI so the Slurm script can control paths cleanly

Notes:
- LMDB is single-writer. Two GPUs won't help here; this is disk IO + CPU (pickle/NumPy).
- If you enable --fast-io, durability is reduced until the final env.sync() completes.
"""

import os
import csv
import argparse
import pickle
from typing import List, Set, Tuple, Optional

import lmdb
import numpy as np
from tqdm import tqdm


def collect_ids(tsv_paths: List[str], id_col: str = "Ids") -> List[str]:
    ids: Set[str] = set()
    for tsv in tsv_paths:
        with open(tsv, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if id_col not in reader.fieldnames:
                raise ValueError(f"Column '{id_col}' not found in {tsv}. Columns: {reader.fieldnames}")
            for row in reader:
                sid = row[id_col]
                if sid:
                    ids.add(sid)
    return sorted(ids)


def estimate_map_size_bytes(ids: List[str], features_dir: str, factor: float = 2.2) -> int:
    """
    Estimate LMDB map_size from raw .npy file sizes.
    Factor ~2.0-2.5 is usually safe because LMDB has overhead and pickle expands slightly.

    If files are on a shared filesystem, stat() is cheap and safe.
    """
    total = 0
    missing = 0
    for sid in ids:
        paths = (
            os.path.join(features_dir, f"{sid}.plm.npy"),
            os.path.join(features_dir, f"{sid}.3di_tokens.npy"),
            os.path.join(features_dir, f"{sid}.3di_conf.npy"),
        )
        ok = True
        for p in paths:
            try:
                total += os.path.getsize(p)
            except FileNotFoundError:
                ok = False
                break
        if not ok:
            missing += 1
    # add 1 GiB cushion
    return int(total * factor) + (1 * 1024**3)


def open_lmdb(
    lmdb_dir: str,
    map_size: int,
    fast_io: bool,
) -> lmdb.Environment:
    """
    fast_io=False: more conservative settings (closer to your original script)
    fast_io=True:  significantly faster, especially on networked filesystems
    """
    os.makedirs(lmdb_dir, exist_ok=True)

    if fast_io:
        return lmdb.open(
            lmdb_dir,
            map_size=map_size,
            subdir=True,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            max_dbs=1,
            # speed-focused
            sync=False,
            metasync=False,
            map_async=True,
        )

    # conservative defaults (similar to original)
    return lmdb.open(
        lmdb_dir,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=True,
        meminit=False,
        max_dbs=1,
        metasync=True,
        sync=True,
        map_async=False,
    )


def main():
    ap = argparse.ArgumentParser(description="Pack ProstT5 feature .npy files into an LMDB.")
    ap.add_argument("--tsvs", nargs="+", required=True, help="TSV files containing IDs (column 'Ids' by default).")
    ap.add_argument("--id-col", default="Ids", help="Column name for sequence IDs in the TSVs. Default: Ids")
    ap.add_argument("--features-dir", required=True, help="Directory containing *.plm.npy, *.3di_tokens.npy, *.3di_conf.npy")
    ap.add_argument("--lmdb-dir", required=True, help="Output LMDB directory (will contain data.mdb/lock.mdb)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing LMDB directory contents.")
    ap.add_argument("--commit-every", type=int, default=1000, help="Commit every N records. Default: 1000")
    ap.add_argument("--map-size-gb", type=int, default=0, help="LMDB map size in GiB. If 0, auto-estimate.")
    ap.add_argument("--map-factor", type=float, default=2.2, help="Auto map_size factor over raw .npy sizes. Default: 2.2")
    ap.add_argument("--fast-io", action="store_true", help="Faster LMDB write settings (sync/metasync off, map_async on).")
    ap.add_argument("--fp16", action="store_true", help="Store plm/conf as float16 to reduce LMDB size (training must handle).")
    args = ap.parse_args()

    # Overwrite handling
    if args.overwrite and os.path.isdir(args.lmdb_dir):
        # Remove existing files; keep directory
        for fn in ("data.mdb", "lock.mdb"):
            p = os.path.join(args.lmdb_dir, fn)
            if os.path.exists(p):
                os.remove(p)

    ids = collect_ids(args.tsvs, id_col=args.id_col)
    if not ids:
        raise SystemExit("No IDs found in TSVs.")

    if args.map_size_gb > 0:
        map_size = args.map_size_gb * 1024**3
    else:
        map_size = estimate_map_size_bytes(ids, args.features_dir, factor=args.map_factor)

    print(f"IDs: {len(ids)}")
    print(f"Features dir: {args.features_dir}")
    print(f"LMDB dir: {args.lmdb_dir}")
    print(f"map_size: {map_size / 1024**3:.1f} GiB (fast_io={args.fast_io})")
    print(f"commit_every: {args.commit_every}")

    env = open_lmdb(args.lmdb_dir, map_size=map_size, fast_io=args.fast_io)

    packed = 0
    missing = 0
    txn = env.begin(write=True)
    since_commit = 0

    try:
        for sid in tqdm(ids, desc="Packing to LMDB"):
            plm_path = os.path.join(args.features_dir, f"{sid}.plm.npy")
            di_path = os.path.join(args.features_dir, f"{sid}.3di_tokens.npy")
            conf_path = os.path.join(args.features_dir, f"{sid}.3di_conf.npy")

            if not (os.path.exists(plm_path) and os.path.exists(di_path) and os.path.exists(conf_path)):
                missing += 1
                continue

            plm = np.load(plm_path)
            di = np.load(di_path)
            conf = np.load(conf_path)

            if args.fp16:
                if plm.dtype != np.float16:
                    plm = plm.astype(np.float16, copy=False)
                if conf.dtype != np.float16:
                    conf = conf.astype(np.float16, copy=False)

            blob = pickle.dumps((plm, di, conf), protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(sid.encode("utf-8"), blob)

            packed += 1
            since_commit += 1

            if since_commit >= args.commit_every:
                txn.commit()
                txn = env.begin(write=True)
                since_commit = 0

        # final commit
        txn.commit()
    finally:
        # Ensure data is flushed (especially important in fast_io mode)
        env.sync()
        env.close()

    print(f"Done. Packed: {packed} | Missing feature triplets: {missing}")
    print("LMDB:", args.lmdb_dir)


if __name__ == "__main__":
    main()
