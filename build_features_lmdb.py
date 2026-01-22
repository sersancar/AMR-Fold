#!/usr/bin/env python3
import os
import csv
import lmdb
import pickle
import numpy as np
from tqdm import tqdm

TSVS = [
    "/data3/ssanchez/amr_fold/data/DB_train.tsv",
    "/data3/ssanchez/amr_fold/data/DB_val.tsv",
    "/data3/ssanchez/amr_fold/data/DB_test.tsv",
]
FEATURES_DIR = "/data3/ssanchez/amr_fold/features"
LMDB_DIR = "/data3/ssanchez/amr_fold/features.lmdb"  # directory that will contain data.mdb, lock.mdb

def collect_ids(tsv_paths):
    ids = set()
    for tsv in tsv_paths:
        with open(tsv, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ids.add(row["Ids"])
    return sorted(ids)

def main():
    ids = collect_ids(TSVS)
    os.makedirs(LMDB_DIR, exist_ok=True)

    # Map size: choose safely above your dataset size. 73G raw features -> set ~160G.
    map_size = 160 * 1024**3

    env = lmdb.open(
        LMDB_DIR,
        map_size=map_size,
        subdir=True,
        readonly=False,
        metasync=True,
        sync=True,
        map_async=False,
        lock=True,
        readahead=True,
        meminit=False,
        max_dbs=1,
    )

    with env.begin(write=True) as txn:
        for sid in tqdm(ids, desc="Packing to LMDB"):
            plm_path  = os.path.join(FEATURES_DIR, f"{sid}.plm.npy")
            di_path   = os.path.join(FEATURES_DIR, f"{sid}.3di_tokens.npy")
            conf_path = os.path.join(FEATURES_DIR, f"{sid}.3di_conf.npy")

            plm  = np.load(plm_path)   # (L, 1024)
            di   = np.load(di_path)    # (L,)
            conf = np.load(conf_path)  # (L,)

            # Optional space/perf lever:
            # plm = plm.astype(np.float16, copy=False)
            # conf = conf.astype(np.float16, copy=False)

            blob = pickle.dumps((plm, di, conf), protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(sid.encode("utf-8"), blob)

    env.sync()
    env.close()
    print("Done:", LMDB_DIR)

if __name__ == "__main__":
    main()

