#!/usr/bin/env python3
import argparse
import csv
import gzip
import io
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DBDatasetLMDB, AMRFoldModel, arg_collate_fn

CLASS_NAMES = [
    'non_ARG', 'LSAP-like', 'aminoglycoside', 'bacitracin', 'beta_lactam',
    'fosfomycin', 'glycopeptide', 'multidrug', 'others', 'peptide', 'phenicol',
    'quinolone', 'rifamycin', 'sulfonamide', 'tetracycline',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NON_ARG_LABEL = 'non_ARG'


def open_text(path: str):
    if path.endswith('.gz'):
        return io.TextIOWrapper(gzip.open(path, 'rb'))
    return open(path, 'r')


def read_ids_from_fasta(path: str) -> List[str]:
    ids = []
    with open_text(path) as f:
        for line in f:
            if line.startswith('>'):
                ids.append(line[1:].split()[0])
    return ids


def make_dummy_tsv(ids: List[str], path: str):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['ids', 'Ids', 'class', 'bin'])
        for sid in ids:
            w.writerow([sid, sid, NON_ARG_LABEL, 0])


def probs_from_class_logits(logits: torch.Tensor, class_loss_type: str) -> torch.Tensor:
    if str(class_loss_type).lower() == 'ce':
        return torch.softmax(logits, dim=1)
    return torch.sigmoid(logits)


def apply_rare_class_gate(probs: np.ndarray, gate_class_indices: Tuple[int, ...], tau: float, delta: float) -> np.ndarray:
    if probs.size == 0:
        return np.array([], dtype=np.int64)
    if not gate_class_indices:
        return probs.argmax(axis=1).astype(np.int64)
    gate_set = set(int(x) for x in gate_class_indices)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    pred2 = top2[:, 0]
    pred1 = top2[:, 1]
    p1 = probs[np.arange(probs.shape[0]), pred1]
    p2 = probs[np.arange(probs.shape[0]), pred2]
    mask = np.array([p in gate_set for p in pred1], dtype=bool)
    if tau > 0:
        mask &= (p1 < float(tau)) | ((p1 - p2) < float(delta))
    else:
        mask &= ((p1 - p2) < float(delta))
    adj = pred1.copy()
    adj[mask] = pred2[mask]
    return adj.astype(np.int64)


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def build_model(n_classes: int, dropout: float, max_len: int, device: torch.device) -> AMRFoldModel:
    m = AMRFoldModel(n_classes=n_classes, dropout=float(dropout), max_len=int(max_len))
    return m.to(device)


def load_models(checkpoints: List[str], device: torch.device, dropout: float, max_len: int):
    models = []
    hps = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        hp = ckpt.get('hp', {}) if isinstance(ckpt, dict) else {}
        model = build_model(n_classes=len(CLASS_NAMES), dropout=dropout, max_len=max_len, device=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        hps.append(hp)
    return models, hps


def read_score_manifest(path: str, shards_dir: str, lmdb_root: str):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if 'shard' not in reader.fieldnames:
            raise ValueError(f"Manifest must contain a 'shard' column: {path}")
        for row in reader:
            shard = row['shard'].strip()
            if not shard:
                continue
            fasta = Path(shards_dir) / shard
            lmdb_dir = Path(lmdb_root) / f'{Path(shard).stem}.lmdb'
            rows.append((shard, fasta, lmdb_dir))
    if not rows:
        raise ValueError(f'No shards found in manifest: {path}')
    return rows


def score_one_shard(shard_name: str, fasta_path: Path, lmdb_dir: Path, output_path: Path, models, device, args, max_len: int, class_loss_type: str):
    ids = read_ids_from_fasta(str(fasta_path))
    if not ids:
        raise ValueError(f'No sequences found in FASTA shard: {fasta_path}')
    if not lmdb_dir.exists():
        raise FileNotFoundError(f'Missing LMDB dir for {shard_name}: {lmdb_dir}')

    with tempfile.TemporaryDirectory(prefix='amrfold_infer_tsv_') as tmpdir:
        dummy_tsv = os.path.join(tmpdir, 'infer.tsv')
        make_dummy_tsv(ids, dummy_tsv)
        ds = DBDatasetLMDB(dummy_tsv, str(lmdb_dir), class_to_idx=CLASS_TO_IDX)
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == 'cuda'),
            collate_fn=lambda b: arg_collate_fn(b, l_max=max_len, random_crop=False),
        )

        seq_ids_ref = None
        bin_probs_seeds = []
        class_probs_seeds = []

        with torch.inference_mode():
            for model in models:
                seq_ids_all = []
                bin_probs_all = []
                class_probs_all = []
                for batch in loader:
                    plm = batch['plm'].to(device, non_blocking=True)
                    di = batch['di'].to(device, non_blocking=True)
                    conf = batch['conf'].to(device, non_blocking=True)
                    attn = batch['attention_mask'].to(device, non_blocking=True)
                    out = model(plm=plm, di=di, conf=conf, attention_mask=attn, return_attn=False)
                    bin_probs_all.append(torch.sigmoid(out['logits_bin']).detach().cpu().numpy())
                    class_probs_all.append(probs_from_class_logits(out['logits_class'], class_loss_type).detach().cpu().numpy())
                    seq_ids_all.extend(batch['seq_ids'])

                if seq_ids_ref is None:
                    seq_ids_ref = seq_ids_all
                elif seq_ids_ref != seq_ids_all:
                    raise RuntimeError(f'Seed outputs are not aligned for {shard_name}.')
                bin_probs_seeds.append(np.concatenate(bin_probs_all, axis=0) if bin_probs_all else np.array([], dtype=np.float32))
                class_probs_seeds.append(np.concatenate(class_probs_all, axis=0) if class_probs_all else np.zeros((0, len(CLASS_NAMES)), dtype=np.float32))

    bin_probs = np.mean(np.stack(bin_probs_seeds, axis=0), axis=0)
    class_probs = np.mean(np.stack(class_probs_seeds, axis=0), axis=0)

    raw_preds = class_probs.argmax(axis=1).astype(np.int64)
    top2 = np.argsort(class_probs, axis=1)[:, -2:]
    pred2 = top2[:, 0].astype(np.int64)
    pred1 = top2[:, 1].astype(np.int64)
    top1_prob = class_probs[np.arange(class_probs.shape[0]), pred1]
    top2_prob = class_probs[np.arange(class_probs.shape[0]), pred2]
    margins = top1_prob - top2_prob

    gate_names = [x.strip() for x in str(args.gate_classes).split(',') if x.strip()]
    gate_idx = tuple(CLASS_TO_IDX[x] for x in gate_names if x in CLASS_TO_IDX and x != NON_ARG_LABEL)
    gated_preds = apply_rare_class_gate(class_probs, gate_idx, float(args.gate_tau), float(args.gate_delta))
    gate_applied = (gated_preds != raw_preds).astype(np.int64)
    pred_bin = (bin_probs >= float(args.bin_threshold)).astype(np.int64)
    final_label = [NON_ARG_LABEL if b == 0 else IDX_TO_CLASS[int(c)] for b, c in zip(pred_bin, gated_preds)]

    prob_cols = [f'prob_{c.replace("-", "_").replace(" ", "_")}' for c in CLASS_NAMES]
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([
            'seq_id', 'bin_prob', 'pred_bin', 'pred_class_raw', 'pred_class', 'final_label',
            'second_class', 'top1_prob', 'top2_prob', 'margin', 'gate_applied', 'threshold_used'
        ] + prob_cols)
        for i, sid in enumerate(seq_ids_ref):
            row = [
                sid, float(bin_probs[i]), int(pred_bin[i]), IDX_TO_CLASS[int(raw_preds[i])],
                IDX_TO_CLASS[int(gated_preds[i])], final_label[i], IDX_TO_CLASS[int(pred2[i])],
                float(top1_prob[i]), float(top2_prob[i]), float(margins[i]), int(gate_applied[i]),
                float(args.bin_threshold),
            ] + [float(x) for x in class_probs[i, :]]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description='Score many LMDB feature stores with one persistent AMR-Fold seed ensemble.')
    p.add_argument('--manifest', required=True)
    p.add_argument('--shards_dir', required=True)
    p.add_argument('--lmdb_root', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--bin_threshold', type=float, default=0.39917078614234924)
    p.add_argument('--gate_classes', default='multidrug,others,sulfonamide,rifamycin,quinolone')
    p.add_argument('--gate_tau', type=float, default=0.7108869316825098)
    p.add_argument('--gate_delta', type=float, default=0.10345399260721301)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--max_len', type=int, default=None)
    p.add_argument('--class_loss_type', default=None, choices=[None, 'bce', 'ce'])
    p.add_argument('--skip_completed', default='true')
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f'[GROUP_SCORE] device={device}', flush=True)
    if device.type == 'cuda':
        print(f'[GROUP_SCORE] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}', flush=True)
        print(f'[GROUP_SCORE] torch.cuda.device_count={torch.cuda.device_count()}', flush=True)
        print(f'[GROUP_SCORE] torch.cuda.current_device={torch.cuda.current_device()}', flush=True)

    first_ckpt = torch.load(args.checkpoints[0], map_location='cpu')
    hp = first_ckpt.get('hp', {}) if isinstance(first_ckpt, dict) else {}
    dropout = float(args.dropout if args.dropout is not None else hp.get('dropout', 0.1038155492514013))
    max_len = int(args.max_len if args.max_len is not None else hp.get('max_len', 1024))
    class_loss_type = str(args.class_loss_type if args.class_loss_type is not None else hp.get('class_loss_type', 'bce')).lower()

    print(f'[GROUP_SCORE] Loading {len(args.checkpoints)} AMR-Fold checkpoints once', flush=True)
    models, _ = load_models(args.checkpoints, device, dropout, max_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_score_manifest(args.manifest, args.shards_dir, args.lmdb_root)
    skip_completed = str(args.skip_completed).strip().lower() in {'1', 'true', 't', 'yes', 'y'}

    for i, (shard, fasta, lmdb_dir) in enumerate(rows, start=1):
        out = output_dir / f'{Path(shard).stem}.predictions.tsv'
        if skip_completed and out.exists() and out.stat().st_size > 0:
            print(f'[GROUP_SCORE] [{i}/{len(rows)}] SKIP completed {shard}', flush=True)
            continue
        print(f'[GROUP_SCORE] [{i}/{len(rows)}] Scoring {shard}', flush=True)
        score_one_shard(shard, fasta, lmdb_dir, out, models, device, args, max_len, class_loss_type)

    print('[GROUP_SCORE] Done', flush=True)


if __name__ == '__main__':
    main()
