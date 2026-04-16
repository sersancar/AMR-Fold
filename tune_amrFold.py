#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning + final retrain for AMR-Fold.

End-goal aligned objective:
  - Primary: performance of ARG class prediction CONDITIONAL on bin==1
    (i.e., class quality on true positives): f1_class_macro_pos

Recommended multi-GPU pattern:
  - Run multiple *processes* sharing the same Optuna storage, each pinned to a GPU
    via CUDA_VISIBLE_DEVICES (see tune_AMRFold.sh).

This script is self-contained and mirrors the loss/metrics logic of train_amrFold.py:
  - Binary BCE on all samples (weighted)
  - Class loss on positives only (weighted)
  - Attention regularisation (entropy + continuity)
  - ReduceLROnPlateau + early stopping

IMPORTANT (your new considerations):
  1) Class head uses sigmoid (multi-label-style probabilities) rather than softmax.
     We still evaluate with argmax for "single-label" class assignment.
  2) You can add more gate classes via --gate_classes (names must match TSV "class" values).
"""

import argparse
import gc
import json
import os
import time
import random
import sqlite3
import shutil
import sys
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import DBDataset, DBDatasetLMDB, arg_collate_fn, AMRFoldModel

try:
    import optuna
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Optuna is required. Install with e.g. `mamba install -c conda-forge optuna` "
        "or `pip install optuna`.\nOriginal error:\n"
        + str(e)
    )

# Ensure stdout/stderr are line-buffered (fixes “prints all at once” on some schedulers)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Fixed label-space contract (14 ARG classes + non_ARG)
NON_ARG_LABEL = "non_ARG"
EXPECTED_N_CLASSES = 15  # 14 ARG classes + non_ARG
EXPECTED_N_ARG_CLASSES = 14


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def build_model(n_classes: int, *, dropout: float, max_len: int, device: torch.device) -> AMRFoldModel:
    model = AMRFoldModel(
        n_classes=n_classes,
        dropout=float(dropout),
        max_len=int(max_len),
    )
    return model.to(device)

def build_fixed_class_mapping(train_tsv: str, val_tsv: str, test_tsv: str) -> Dict[str, int]:
    """Build a fixed, validated class_to_idx mapping (14 ARG classes + non_ARG)."""
    import csv

    all_classes = set()
    for tsv_path in [train_tsv, val_tsv, test_tsv]:
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                all_classes.add(row["class"])

    classes = sorted(all_classes)
    if NON_ARG_LABEL not in classes:
        raise ValueError(
            f"Expected label '{NON_ARG_LABEL}' to exist in TSVs, but it was not found. "
            f"Found classes={classes}"
        )

    # Force non_ARG at index 0 (stable mapping)
    classes = [NON_ARG_LABEL] + [c for c in classes if c != NON_ARG_LABEL]

    if len(classes) != EXPECTED_N_CLASSES:
        raise ValueError(
            f"Expected exactly {EXPECTED_N_CLASSES} classes (={EXPECTED_N_ARG_CLASSES} ARG + non_ARG), "
            f"but found {len(classes)}: {classes}"
        )

    return {c: i for i, c in enumerate(classes)}


def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Return (best_thr, best_f1) using PR curve; robust to edge cases."""
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = (2.0 * prec * rec) / (prec + rec + 1e-12)
    best_i = int(np.nanargmax(f1))
    if best_i >= len(thr):
        return 0.5, float(f1[best_i])
    return float(thr[best_i]), float(f1[best_i])


def apply_rare_class_gate(
    probs: np.ndarray,
    gate_class_indices: Optional[Tuple[int, ...]] = None,
    tau: float = 0.0,
    delta: float = 0.0,
) -> np.ndarray:
    """Apply a simple confidence/margin gate for specified classes.

    If the top-1 predicted class is in gate_class_indices, only accept it when:
      - p(top1) >= tau  AND  (p(top1) - p(top2)) >= delta
    Otherwise, fall back to the 2nd-best class.

    NOTE: Works with any "prob-like" scores per class (softmax or sigmoid outputs).
    We apply it to sigmoid outputs in this script.

    Args:
        probs: (N, C) probabilities/scores
        gate_class_indices: tuple of class indices to gate (e.g. peptide, rifamycin)
        tau: minimum probability for gated classes
        delta: minimum margin p1 - p2 for gated classes

    Returns:
        preds: (N,) int64 adjusted predictions
    """
    if probs.size == 0:
        return np.array([], dtype=np.int64)
    if gate_class_indices is None or len(gate_class_indices) == 0:
        return probs.argmax(axis=1).astype(np.int64)

    gate_set = set(int(x) for x in gate_class_indices)

    # Top-2 classes for each row
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


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_classes: int
    class_to_idx: Dict[str, int]
    bce_pos_weight: float
    class_weights: np.ndarray


def build_data(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    *,
    feature_dir: Optional[str] = None,
    features_lmdb: Optional[str] = None,
    max_len: int = 1024,
    batch_size: int = 16,
    num_workers: int = 4,
    class_weight_power: float = 1.0,
    class_weight_cap: float = 1e9,
    random_crop_train: bool = True,
) -> DataBundle:
    """Build datasets + loaders, and compute loss weights from TRAIN only.

    - class weights can be tempered/capped: w=(ref/c)^power, clipped at cap
    - training loader can use random cropping for long proteins

    Uses LMDB features if features_lmdb is provided; otherwise uses feature_dir.

    IMPORTANT: class_to_idx is FIXED from TSVs (stable split/seed reproducibility).
    """
    # Build a validated, fixed class mapping (15 classes; non_ARG forced index 0)
    class_to_idx = build_fixed_class_mapping(train_tsv, val_tsv, test_tsv)

    if features_lmdb:
        train_ds = DBDatasetLMDB(train_tsv, features_lmdb, class_to_idx=class_to_idx)
        val_ds = DBDatasetLMDB(val_tsv, features_lmdb, class_to_idx=class_to_idx)
        test_ds = DBDatasetLMDB(test_tsv, features_lmdb, class_to_idx=class_to_idx)
    else:
        if not feature_dir:
            raise ValueError("Provide either --features_lmdb or --feature_dir.")
        # KEEP FIXED mapping (do NOT overwrite from dataset)
        train_ds = DBDataset(train_tsv, feature_dir, class_to_idx=class_to_idx)
        val_ds = DBDataset(val_tsv, feature_dir, class_to_idx=class_to_idx)
        test_ds = DBDataset(test_tsv, feature_dir, class_to_idx=class_to_idx)

    n_classes = len(class_to_idx)

    # ------------------ Loss weighting (computed from TRAIN only) ------------------
    train_bins = [int(r["bin"]) for r in train_ds.records]
    n_train = len(train_bins)
    n_pos = int(sum(train_bins))
    n_neg = int(n_train - n_pos)
    bce_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # Class weights: only among POSITIVES (since class loss only on positives)
    from collections import Counter
    pos_class_counts = Counter()
    for r in train_ds.records:
        if int(r["bin"]) == 1:
            pos_class_counts[r["class"]] += 1

    power = float(class_weight_power)
    cap = float(class_weight_cap)

    # Reference count anchors typical classes ~1.0 before normalization
    counts = [
        pos_class_counts.get(cls, 0)
        for cls in class_to_idx.keys()
        if cls != NON_ARG_LABEL and pos_class_counts.get(cls, 0) > 0
    ]
    ref = float(np.median(counts)) if len(counts) > 0 else 1.0
    if ref <= 0:
        ref = 1.0

    class_weights = np.zeros(n_classes, dtype=np.float32)
    for cls, idx in class_to_idx.items():
        if cls == NON_ARG_LABEL:
            class_weights[idx] = 0.0
            continue
        c = int(pos_class_counts.get(cls, 0))
        if c <= 0:
            class_weights[idx] = 0.0
            continue
        w = (ref / float(c)) ** power
        if cap > 0:
            w = min(float(w), cap)
        class_weights[idx] = float(w)

    # Normalize non-zero weights to mean=1 (keeps loss scale stable)
    nz = class_weights > 0
    if nz.any():
        class_weights[nz] = class_weights[nz] / class_weights[nz].mean()

    train_collate = lambda b: arg_collate_fn(b, l_max=max_len, random_crop=bool(random_crop_train))
    eval_collate = lambda b: arg_collate_fn(b, l_max=max_len, random_crop=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=eval_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=eval_collate,
        drop_last=False,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_classes=n_classes,
        class_to_idx=class_to_idx,
        bce_pos_weight=bce_pos_weight,
        class_weights=class_weights,
    )


def one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    # labels: (N,) int64
    y = torch.zeros((labels.size(0), n_classes), device=labels.device, dtype=torch.float32)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y


def probs_from_class_logits(logits: torch.Tensor, class_loss_type: str) -> torch.Tensor:
    if str(class_loss_type).lower() == "ce":
        return torch.softmax(logits, dim=1)
    return torch.sigmoid(logits)


def build_class_criteria(class_loss_type: str, class_weights_t: torch.Tensor, label_smoothing: float = 0.05) -> Tuple[nn.Module, nn.Module]:
    class_loss_type = str(class_loss_type).lower()
    if class_loss_type == "ce":
        ce_weight = class_weights_t.clone().float()
        if ce_weight.numel() > 0:
            ce_weight[0] = 0.0
        ls = float(label_smoothing)
        return (
            nn.CrossEntropyLoss(weight=ce_weight, reduction="none", label_smoothing=ls),
            nn.CrossEntropyLoss(weight=ce_weight, reduction="none", label_smoothing=ls),
        )

    pos_weight = class_weights_t.clone().float()
    pos_weight[pos_weight <= 0] = 1.0
    return (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none"),
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none"),
    )


def compute_class_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    class_loss_type: str,
    n_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.size(0) == 0:
        z = torch.tensor(0.0, device=logits.device)
        return z, torch.zeros((0,), device=logits.device)

    class_loss_type = str(class_loss_type).lower()
    if class_loss_type == "ce":
        loss_per = criterion(logits, labels)
        return loss_per.mean(), loss_per

    y = one_hot(labels, n_classes)
    loss_mat = criterion(logits, y)
    loss_per = loss_mat.mean(dim=1)
    return loss_per.mean(), loss_per


def save_tsv_rows(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for c in columns:
                v = row.get(c, "")
                if isinstance(v, float):
                    if np.isnan(v):
                        vals.append("nan")
                    else:
                        vals.append(repr(float(v)))
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def sanitize_colname(x: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(x))


def evaluate_arrays(
    *,
    seq_ids: List[str],
    bin_labels: np.ndarray,
    bin_probs: np.ndarray,
    class_labels_all: np.ndarray,
    class_probs_all: np.ndarray,
    loss_total: float,
    loss_bin: float,
    loss_class_pos: float,
    threshold: float = 0.5,
    tune_threshold: bool = False,
    gate_class_indices: Optional[Tuple[int, ...]] = None,
    gate_tau: float = 0.0,
    gate_delta: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if class_probs_all.ndim != 2:
        raise ValueError(f"class_probs_all must be 2D, got shape={class_probs_all.shape}")

    n = len(seq_ids)
    if n != len(bin_labels) or n != len(bin_probs) or n != len(class_labels_all) or n != class_probs_all.shape[0]:
        raise ValueError("Mismatched sequence/prediction array lengths in evaluate_arrays().")

    n_classes = int(class_probs_all.shape[1])
    raw_preds = class_probs_all.argmax(axis=1).astype(np.int64) if n > 0 else np.array([], dtype=np.int64)

    if n > 0 and n_classes >= 2:
        top2 = np.argsort(class_probs_all, axis=1)[:, -2:]
        pred2 = top2[:, 0].astype(np.int64)
        pred1 = top2[:, 1].astype(np.int64)
        top1_prob = class_probs_all[np.arange(n), pred1].astype(np.float32)
        top2_prob = class_probs_all[np.arange(n), pred2].astype(np.float32)
    elif n > 0:
        pred1 = raw_preds.copy()
        pred2 = raw_preds.copy()
        top1_prob = class_probs_all[:, 0].astype(np.float32)
        top2_prob = np.zeros_like(top1_prob)
    else:
        pred1 = np.array([], dtype=np.int64)
        pred2 = np.array([], dtype=np.int64)
        top1_prob = np.array([], dtype=np.float32)
        top2_prob = np.array([], dtype=np.float32)

    margins = (top1_prob - top2_prob).astype(np.float32)
    gated_preds = apply_rare_class_gate(
        class_probs_all,
        gate_class_indices=gate_class_indices,
        tau=float(gate_tau),
        delta=float(gate_delta),
    ).astype(np.int64)
    gate_applied = (gated_preds != raw_preds)

    pos_mask = (bin_labels.astype(np.int64) == 1)
    pos_class_labels = class_labels_all[pos_mask].astype(np.int64)
    pos_class_preds = gated_preds[pos_mask].astype(np.int64)
    pos_class_probs = class_probs_all[pos_mask].astype(np.float32)

    try:
        auc_roc = roc_auc_score(bin_labels, bin_probs)
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = average_precision_score(bin_labels, bin_probs)
    except ValueError:
        auc_pr = float("nan")

    if tune_threshold:
        best_thr, f1_bin_best = _best_f1_threshold(bin_labels, bin_probs)
        thr_to_use = float(best_thr)
    else:
        f1_bin_best = float("nan")
        thr_to_use = float(threshold)

    preds_0p5 = (bin_probs >= 0.5).astype(np.int64)
    preds_used = (bin_probs >= thr_to_use).astype(np.int64)

    acc_0p5 = accuracy_score(bin_labels, preds_0p5)
    f1_0p5 = f1_score(bin_labels, preds_0p5, zero_division=0)
    prec_0p5 = precision_score(bin_labels, preds_0p5, zero_division=0)
    rec_0p5 = recall_score(bin_labels, preds_0p5, zero_division=0)

    acc_used = accuracy_score(bin_labels, preds_used)
    f1_used = f1_score(bin_labels, preds_used, zero_division=0)
    prec_used = precision_score(bin_labels, preds_used, zero_division=0)
    rec_used = recall_score(bin_labels, preds_used, zero_division=0)

    f1_class_macro_all = f1_score(class_labels_all, gated_preds, average="macro", zero_division=0)
    if pos_class_labels.size > 0:
        f1_class_macro_pos = f1_score(pos_class_labels, pos_class_preds, average="macro", zero_division=0)
    else:
        f1_class_macro_pos = float("nan")

    metrics = {
        "loss_total": float(loss_total),
        "loss_bin": float(loss_bin),
        "loss_class_pos": float(loss_class_pos),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "acc_bin": float(acc_used),
        "f1_bin": float(f1_used),
        "precision_bin": float(prec_used),
        "recall_bin": float(rec_used),
        "pred_pos_rate": float(preds_used.mean()) if n > 0 else float("nan"),
        "acc_bin_0p5": float(acc_0p5),
        "f1_bin_0p5": float(f1_0p5),
        "precision_bin_0p5": float(prec_0p5),
        "recall_bin_0p5": float(rec_0p5),
        "pred_pos_rate_0p5": float(preds_0p5.mean()) if n > 0 else float("nan"),
        "threshold_used": float(thr_to_use),
        "threshold_best_f1": float(best_thr) if tune_threshold else float("nan"),
        "f1_bin_best": float(f1_bin_best),
        "f1_class_macro_all": float(f1_class_macro_all),
        "f1_class_macro_pos": float(f1_class_macro_pos),
        "n_samples": float(n),
        "n_pos": float(pos_mask.sum()),
        "gate_override_count": float(gate_applied.sum()),
        "gate_override_rate": float(gate_applied.mean()) if n > 0 else float("nan"),
    }

    outputs = {
        "seq_ids": list(seq_ids),
        "bin_labels": bin_labels.astype(np.int64),
        "bin_probs": bin_probs.astype(np.float32),
        "bin_preds_0p5": preds_0p5.astype(np.int64),
        "bin_preds_used": preds_used.astype(np.int64),
        "class_labels_all": class_labels_all.astype(np.int64),
        "class_probs_all": class_probs_all.astype(np.float32),
        "class_preds_raw_all": raw_preds.astype(np.int64),
        "class_preds_all": gated_preds.astype(np.int64),
        "class_top2_idx_all": pred2.astype(np.int64),
        "class_top1_prob_all": top1_prob.astype(np.float32),
        "class_top2_prob_all": top2_prob.astype(np.float32),
        "class_margin_all": margins.astype(np.float32),
        "gate_applied_all": gate_applied.astype(np.int64),
        "class_labels_pos": pos_class_labels.astype(np.int64),
        "class_preds_pos": pos_class_preds.astype(np.int64),
        "class_probs_pos": pos_class_probs.astype(np.float32),
    }
    return metrics, outputs


def metrics_rows_for_split(split_name: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    base = {
        "split": split_name,
        "loss_total": float(metrics.get("loss_total", float("nan"))),
        "loss_bin": float(metrics.get("loss_bin", float("nan"))),
        "loss_class_pos": float(metrics.get("loss_class_pos", float("nan"))),
        "auc_roc": float(metrics.get("auc_roc", float("nan"))),
        "auc_pr": float(metrics.get("auc_pr", float("nan"))),
        "f1_class_macro_all": float(metrics.get("f1_class_macro_all", float("nan"))),
        "f1_class_macro_pos": float(metrics.get("f1_class_macro_pos", float("nan"))),
        "n_samples": float(metrics.get("n_samples", float("nan"))),
        "n_pos": float(metrics.get("n_pos", float("nan"))),
        "gate_override_count": float(metrics.get("gate_override_count", float("nan"))),
        "gate_override_rate": float(metrics.get("gate_override_rate", float("nan"))),
    }
    row_0p5 = dict(base)
    row_0p5.update({
        "threshold_type": "0p5",
        "threshold_used": 0.5,
        "acc_bin": float(metrics.get("acc_bin_0p5", float("nan"))),
        "f1_bin": float(metrics.get("f1_bin_0p5", float("nan"))),
        "precision_bin": float(metrics.get("precision_bin_0p5", float("nan"))),
        "recall_bin": float(metrics.get("recall_bin_0p5", float("nan"))),
        "pred_pos_rate": float(metrics.get("pred_pos_rate_0p5", float("nan"))),
    })
    row_used = dict(base)
    row_used.update({
        "threshold_type": "tuned_or_used",
        "threshold_used": float(metrics.get("threshold_used", float("nan"))),
        "acc_bin": float(metrics.get("acc_bin", float("nan"))),
        "f1_bin": float(metrics.get("f1_bin", float("nan"))),
        "precision_bin": float(metrics.get("precision_bin", float("nan"))),
        "recall_bin": float(metrics.get("recall_bin", float("nan"))),
        "pred_pos_rate": float(metrics.get("pred_pos_rate", float("nan"))),
    })
    return [row_0p5, row_used]


def save_metrics_tsv(path: str, split_metrics: Dict[str, Dict[str, float]]) -> None:
    rows: List[Dict[str, Any]] = []
    for split_name, metrics in split_metrics.items():
        rows.extend(metrics_rows_for_split(split_name, metrics))
    cols = [
        "split", "threshold_type", "threshold_used",
        "loss_total", "loss_bin", "loss_class_pos",
        "auc_roc", "auc_pr",
        "acc_bin", "f1_bin", "precision_bin", "recall_bin", "pred_pos_rate",
        "f1_class_macro_all", "f1_class_macro_pos",
        "gate_override_count", "gate_override_rate",
        "n_samples", "n_pos",
    ]
    save_tsv_rows(path, rows, cols)


def save_prediction_tsv(path: str, outputs: Dict[str, Any], idx_to_class: Dict[int, str]) -> None:
    class_cols = []
    class_idx_sorted = sorted(idx_to_class)
    for i in class_idx_sorted:
        class_cols.append((i, f"prob_{sanitize_colname(idx_to_class[i])}"))

    rows = []
    n = len(outputs["seq_ids"])
    for i in range(n):
        row = {
            "seq_id": outputs["seq_ids"][i],
            "true_bin": int(outputs["bin_labels"][i]),
            "bin_prob": float(outputs["bin_probs"][i]),
            "pred_bin_0p5": int(outputs["bin_preds_0p5"][i]),
            "pred_bin_used": int(outputs["bin_preds_used"][i]),
            "true_class_index": int(outputs["class_labels_all"][i]),
            "true_class": idx_to_class.get(int(outputs["class_labels_all"][i]), str(int(outputs["class_labels_all"][i]))),
            "pred_class_raw_index": int(outputs["class_preds_raw_all"][i]),
            "pred_class_raw": idx_to_class.get(int(outputs["class_preds_raw_all"][i]), str(int(outputs["class_preds_raw_all"][i]))),
            "pred_class_index": int(outputs["class_preds_all"][i]),
            "pred_class": idx_to_class.get(int(outputs["class_preds_all"][i]), str(int(outputs["class_preds_all"][i]))),
            "second_class_index": int(outputs["class_top2_idx_all"][i]),
            "second_class": idx_to_class.get(int(outputs["class_top2_idx_all"][i]), str(int(outputs["class_top2_idx_all"][i]))),
            "top1_prob": float(outputs["class_top1_prob_all"][i]),
            "top2_prob": float(outputs["class_top2_prob_all"][i]),
            "margin": float(outputs["class_margin_all"][i]),
            "gate_applied": int(outputs["gate_applied_all"][i]),
        }
        for cls_idx, col in class_cols:
            row[col] = float(outputs["class_probs_all"][i, cls_idx])
        rows.append(row)

    cols = [
        "seq_id", "true_bin", "bin_prob", "pred_bin_0p5", "pred_bin_used",
        "true_class_index", "true_class",
        "pred_class_raw_index", "pred_class_raw",
        "pred_class_index", "pred_class",
        "second_class_index", "second_class",
        "top1_prob", "top2_prob", "margin", "gate_applied",
    ] + [c for _, c in class_cols]
    save_tsv_rows(path, rows, cols)


def compute_gate_audit_rows(
    outputs: Dict[str, Any],
    idx_to_class: Dict[int, str],
    gate_class_names: List[str],
    gate_tau: float,
    gate_delta: float,
) -> List[Dict[str, Any]]:
    mask = outputs["bin_labels"] == 1
    if mask.sum() == 0:
        return []

    true_cls = outputs["class_labels_all"][mask]
    raw_cls = outputs["class_preds_raw_all"][mask]
    gated_cls = outputs["class_preds_all"][mask]
    top2_cls = outputs["class_top2_idx_all"][mask]
    top1_prob = outputs["class_top1_prob_all"][mask]
    margin = outputs["class_margin_all"][mask]

    gate_set = set(gate_class_names)
    rows: List[Dict[str, Any]] = []

    overall_hyp = (top1_prob < float(gate_tau)) | (margin < float(gate_delta))
    rows.append({
        "class_index": "ALL",
        "class_name": "ALL",
        "currently_gated": "ALL",
        "n_raw_top1": int(raw_cls.size),
        "raw_correct": int((raw_cls == true_cls).sum()),
        "actual_overrides": int((raw_cls != gated_cls).sum()),
        "hypothetical_overrides": int(overall_hyp.sum()),
        "actual_correct": int((gated_cls == true_cls).sum()),
        "hypothetical_correct_if_gated": int(((np.where(overall_hyp, top2_cls, raw_cls)) == true_cls).sum()),
        "delta_correct_if_gated": int(((np.where(overall_hyp, top2_cls, raw_cls)) == true_cls).sum() - (raw_cls == true_cls).sum()),
        "suggested_for_gate": "",
        "override_rate_if_gated": float(overall_hyp.mean()) if raw_cls.size > 0 else float("nan"),
    })

    for cls_idx in sorted(idx_to_class):
        cls_name = idx_to_class[cls_idx]
        cls_mask = raw_cls == cls_idx
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            continue
        hypo = cls_mask & ((top1_prob < float(gate_tau)) | (margin < float(gate_delta)))
        hypo_pred = raw_cls.copy()
        hypo_pred[hypo] = top2_cls[hypo]
        raw_correct = int((cls_mask & (raw_cls == true_cls)).sum())
        actual_overrides = int((cls_mask & (raw_cls != gated_cls)).sum())
        actual_correct = int((cls_mask & (gated_cls == true_cls)).sum())
        hypo_overrides = int(hypo.sum())
        hypo_correct = int((cls_mask & (hypo_pred == true_cls)).sum())
        delta_correct = int(hypo_correct - raw_correct)
        suggested = (
            cls_name not in gate_set
            and n_cls >= 10
            and (hypo_overrides / max(n_cls, 1)) >= 0.10
            and delta_correct > 0
        )
        rows.append({
            "class_index": int(cls_idx),
            "class_name": cls_name,
            "currently_gated": int(cls_name in gate_set),
            "n_raw_top1": n_cls,
            "raw_correct": raw_correct,
            "actual_overrides": actual_overrides,
            "hypothetical_overrides": hypo_overrides,
            "actual_correct": actual_correct,
            "hypothetical_correct_if_gated": hypo_correct,
            "delta_correct_if_gated": delta_correct,
            "suggested_for_gate": int(bool(suggested)),
            "override_rate_if_gated": float(hypo_overrides / max(n_cls, 1)),
        })
    return rows


def save_gate_audit_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "class_index", "class_name", "currently_gated",
        "n_raw_top1", "raw_correct",
        "actual_overrides", "hypothetical_overrides",
        "actual_correct", "hypothetical_correct_if_gated",
        "delta_correct_if_gated", "override_rate_if_gated",
        "suggested_for_gate",
    ]
    save_tsv_rows(path, rows, cols)


def write_study_trials_tsv(study: "optuna.Study", out_path: str) -> None:
    param_keys = sorted({k for t in study.trials for k in t.params.keys()})
    rows = []
    for t in study.trials:
        row = {
            "trial_number": int(t.number),
            "state": str(t.state).split(".")[-1],
            "value": float(t.value) if t.value is not None else float("nan"),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else "",
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else "",
            "duration_seconds": (t.duration.total_seconds() if t.duration is not None else float("nan")),
            "user_attrs": json.dumps(t.user_attrs, sort_keys=True),
        }
        for k in param_keys:
            row[k] = t.params.get(k, "")
        rows.append(row)
    cols = ["trial_number", "state", "value", "datetime_start", "datetime_complete", "duration_seconds"] + param_keys + ["user_attrs"]
    save_tsv_rows(out_path, rows, cols)


# -------------------------
# Training / evaluation
# -------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_class: float,
    lambda_ent: float,
    lambda_cont: float,
    criterion_bin: nn.Module,
    criterion_class: nn.Module,
    class_loss_type: str,
    trial_number: Optional[int] = None,
    epoch_number: Optional[int] = None,
    log_batch_progress: bool = False,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bin_loss = 0.0
    total_class_loss_sum = 0.0  # SUM over positive samples (for consistent averaging)
    total_attn_loss = 0.0
    total_samples = 0
    total_pos = 0

    eps = 1e-8

    n_batches = len(data_loader)
    milestone_batches = set()
    if log_batch_progress and n_batches > 0:
        for frac in (0.25, 0.5, 0.75):
            b = int(round(frac * n_batches))
            b = max(1, min(n_batches, b))
            milestone_batches.add(b)

    batches_seen = 0
    samples_seen = 0

    for batch_i, batch in enumerate(data_loader, start=1):
        plm = batch["plm"].to(device, non_blocking=True)
        di = batch["di"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        bin_labels = batch["bin_labels"].to(device, non_blocking=True)
        class_labels = batch["class_labels"].to(device, non_blocking=True)

        B = int(bin_labels.size(0))

        optimizer.zero_grad(set_to_none=True)

        out = model(plm, di, conf, attn_mask, return_attn=True)
        logits_bin = out["logits_bin"]        # (B,)
        logits_class = out["logits_class"]    # (B, n_classes)
        attn_cls = out["attn_cls"]            # (B, L)

        # Binary loss (all samples)
        loss_bin = criterion_bin(logits_bin, bin_labels)

        # Class loss (positives only)
        pos_mask = (bin_labels > 0.5)
        n_pos_batch = int(pos_mask.sum().item())
        if n_pos_batch > 0:
            log_pos = logits_class[pos_mask]
            labels_pos = class_labels[pos_mask]
            loss_class, loss_per = compute_class_loss(
                log_pos,
                labels_pos,
                criterion_class,
                class_loss_type,
                logits_class.size(1),
            )
            total_pos += n_pos_batch
            total_class_loss_sum += float(loss_per.sum().item())
        else:
            loss_class = torch.tensor(0.0, device=device)

        batches_seen += 1
        samples_seen += B
        if log_batch_progress and (batches_seen in milestone_batches):
            lr_now = float(optimizer.param_groups[0]["lr"])
            avg_loss_so_far = total_loss / max(samples_seen, 1)
            avg_bin_so_far = total_bin_loss / max(samples_seen, 1)
            avg_class_pos_so_far = (total_class_loss_sum / total_pos) if total_pos > 0 else float("nan")

            mem_gb = None
            if torch.cuda.is_available() and device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)

            prefix = ""
            if trial_number is not None:
                prefix += f"[trial {trial_number}] "
            if epoch_number is not None:
                prefix += f"[ep {epoch_number}] "

            msg = (
                f"{prefix}batch {batches_seen}/{n_batches} "
                f"loss~{avg_loss_so_far:.4f} (bin~{avg_bin_so_far:.4f}, class_pos~{avg_class_pos_so_far:.4f}) "
                f"lr={lr_now:.2e}"
            )
            if mem_gb is not None:
                msg += f" mem={mem_gb:.1f}GB"
            print(msg, flush=True)

        # Attention regularisation
        mask = attn_mask.float()
        alpha = attn_cls * mask
        alpha_sum = alpha.sum(dim=1, keepdim=True) + eps
        alpha = alpha / alpha_sum

        ent = -(alpha * (alpha + eps).log()).sum(dim=1).mean()
        loss_ent = lambda_ent * ent

        diff = alpha[:, 1:] - alpha[:, :-1]
        diff_mask = mask[:, 1:] * mask[:, :-1]
        cont_per_seq = ((diff ** 2) * diff_mask).sum(dim=1) / (diff_mask.sum(dim=1) + eps)
        cont = cont_per_seq.mean()
        loss_cont = lambda_cont * cont

        loss_attn = loss_ent + loss_cont

        loss = loss_bin + lambda_class * loss_class + loss_attn
        total_samples += B

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * B
        total_bin_loss += float(loss_bin.item()) * B
        total_attn_loss += float(loss_attn.item()) * B

    avg_loss = total_loss / max(total_samples, 1)
    avg_bin_loss = total_bin_loss / max(total_samples, 1)
    avg_class_loss_pos = (total_class_loss_sum / total_pos) if total_pos > 0 else float("nan")
    avg_attn_loss = total_attn_loss / max(total_samples, 1)

    return {
        "loss_total": float(avg_loss),
        "loss_bin": float(avg_bin_loss),
        "loss_class_pos": float(avg_class_loss_pos),
        "loss_attn": float(avg_attn_loss),
        "n_samples": float(total_samples),
        "n_pos": float(total_pos),
    }


def evaluate_with_outputs(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    lambda_class: float,
    threshold: float = 0.5,
    tune_threshold: bool = False,
    criterion_bin: nn.Module,
    criterion_class: nn.Module,
    class_loss_type: str,
    gate_class_indices: Optional[Tuple[int, ...]] = None,
    gate_tau: float = 0.0,
    gate_delta: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.eval()

    total_bin_loss_sum = 0.0     # sum over all samples
    total_class_loss_sum = 0.0   # sum over positive samples
    total_samples = 0
    total_pos = 0

    seq_ids: List[str] = []
    all_bin_labels = []
    all_bin_probs = []
    all_class_labels = []
    all_class_probs = []

    with torch.no_grad():
        for batch in data_loader:
            plm = batch["plm"].to(device, non_blocking=True)
            di = batch["di"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            bin_labels = batch["bin_labels"].to(device, non_blocking=True)
            class_labels = batch["class_labels"].to(device, non_blocking=True)

            out = model(plm, di, conf, attn_mask, return_attn=False)
            logits_bin = out["logits_bin"]
            logits_class = out["logits_class"]

            B = int(bin_labels.size(0))
            total_samples += B
            seq_ids.extend(list(batch.get("seq_ids", [])))

            bin_loss = criterion_bin(logits_bin, bin_labels)
            total_bin_loss_sum += float(bin_loss.item())

            pos_mask = (bin_labels > 0.5)
            n_pos_batch = int(pos_mask.sum().item())
            if n_pos_batch > 0:
                total_pos += n_pos_batch
                log_pos = logits_class[pos_mask]
                labels_pos = class_labels[pos_mask]
                _, loss_per = compute_class_loss(
                    log_pos,
                    labels_pos,
                    criterion_class,
                    class_loss_type,
                    logits_class.size(1),
                )
                total_class_loss_sum += float(loss_per.sum().item())

            probs_bin = torch.sigmoid(logits_bin)
            probs_class = probs_from_class_logits(logits_class, class_loss_type)

            all_bin_labels.append(bin_labels.detach().cpu().numpy())
            all_bin_probs.append(probs_bin.detach().cpu().numpy())
            all_class_labels.append(class_labels.detach().cpu().numpy())
            all_class_probs.append(probs_class.detach().cpu().numpy())

    all_bin_labels_np = np.concatenate(all_bin_labels, axis=0) if all_bin_labels else np.array([], dtype=np.int64)
    all_bin_probs_np = np.concatenate(all_bin_probs, axis=0) if all_bin_probs else np.array([], dtype=np.float32)
    all_class_labels_np = np.concatenate(all_class_labels, axis=0) if all_class_labels else np.array([], dtype=np.int64)
    all_class_probs_np = np.concatenate(all_class_probs, axis=0) if all_class_probs else np.zeros((0, 1), dtype=np.float32)

    avg_bin_loss = total_bin_loss_sum / max(total_samples, 1)
    avg_class_loss = (total_class_loss_sum / total_pos) if total_pos > 0 else float("nan")
    avg_total_loss = avg_bin_loss + lambda_class * (avg_class_loss if total_pos > 0 else 0.0)

    return evaluate_arrays(
        seq_ids=seq_ids,
        bin_labels=all_bin_labels_np,
        bin_probs=all_bin_probs_np,
        class_labels_all=all_class_labels_np,
        class_probs_all=all_class_probs_np,
        loss_total=float(avg_total_loss),
        loss_bin=float(avg_bin_loss),
        loss_class_pos=float(avg_class_loss),
        threshold=threshold,
        tune_threshold=tune_threshold,
        gate_class_indices=gate_class_indices,
        gate_tau=gate_tau,
        gate_delta=gate_delta,
    )


def per_class_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    n_classes: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - classification_report dict (per-class precision/recall/F1/support + macro/weighted)
      - confusion matrix (n_classes x n_classes)
      - per-class ovr ROC-AUC (n_classes,)
      - per-class ovr PR-AUC (n_classes,)
    """
    labels = list(range(n_classes))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_ovr = np.full((n_classes,), np.nan, dtype=np.float64)
    pr_ovr = np.full((n_classes,), np.nan, dtype=np.float64)

    if y_prob is not None and y_prob.shape[0] == y_true.shape[0] and y_prob.shape[1] == n_classes:
        for c in range(n_classes):
            y_c = (y_true == c).astype(np.int64)
            # Need both classes present to compute AUC
            if y_c.sum() == 0 or y_c.sum() == len(y_c):
                continue
            try:
                roc_ovr[c] = roc_auc_score(y_c, y_prob[:, c])
            except Exception:
                pass
            try:
                pr_ovr[c] = average_precision_score(y_c, y_prob[:, c])
            except Exception:
                pass

    return report, cm, roc_ovr, pr_ovr


def save_reports(
    out_dir: str,
    *,
    split_name: str,
    report: Dict[str, Any],
    cm: np.ndarray,
    roc_ovr: np.ndarray,
    pr_ovr: np.ndarray,
    idx_to_class: Dict[int, str],
    prefix: str,
) -> None:
    """Write per-class report + confusion matrix as TSV only."""
    ensure_dir(out_dir)

    rows = []
    for k, v in report.items():
        if str(k).isdigit():
            c = int(k)
            row = dict(v)
            row["class_index"] = c
            row["class_name"] = idx_to_class.get(c, str(c))
            row["roc_auc_ovr"] = float(roc_ovr[c]) if (c < len(roc_ovr) and not np.isnan(roc_ovr[c])) else np.nan
            row["pr_auc_ovr"] = float(pr_ovr[c]) if (c < len(pr_ovr) and not np.isnan(pr_ovr[c])) else np.nan
            if "f1-score" in row and "f1" not in row:
                row["f1"] = row["f1-score"]
            rows.append(row)

    cols = ["class_index", "class_name", "precision", "recall", "f1", "support", "roc_auc_ovr", "pr_auc_ovr"]
    save_tsv_rows(
        os.path.join(out_dir, f"{prefix}_{split_name}_class_report.tsv"),
        sorted(rows, key=lambda x: x.get("class_index", 0)),
        cols,
    )

    labels = [idx_to_class.get(i, str(i)) for i in range(cm.shape[0])]
    cm_path = os.path.join(out_dir, f"{prefix}_{split_name}_confusion.tsv")
    with open(cm_path, "w") as f:
        f.write("true\\pred" + "\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(str(int(x)) for x in cm[i]) + "\n")


def _sqlite_path_from_url(url: str) -> str:
    u = url.split("?", 1)[0]
    if u.startswith("sqlite:////"):
        return "/" + u[len("sqlite:////"):]
    if u.startswith("sqlite:///"):
        return u[len("sqlite:///"):]
    return ""


def _init_sqlite_pragmas(sqlite_url: str) -> None:
    path = _sqlite_path_from_url(sqlite_url)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=60000;")
        con.commit()
    finally:
        con.close()


def _acquire_dir_lock(lock_dir: str, stale_seconds: int = 7200) -> None:
    start = time.time()
    while True:
        try:
            os.mkdir(lock_dir)
            try:
                with open(os.path.join(lock_dir, "pid"), "w") as f:
                    f.write(str(os.getpid()))
            except Exception:
                pass
            return
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_dir)
                if age > stale_seconds:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except Exception:
                pass

            if time.time() - start > 600:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(0.2 + random.random() * 0.3)


def _release_dir_lock(lock_dir: str) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)


def create_study_safe(args: argparse.Namespace, pruner, sampler):
    storage_obj: Any = args.storage

    lock_base = args.out_dir
    if isinstance(args.storage, str) and args.storage.startswith("sqlite"):
        p = _sqlite_path_from_url(args.storage)
        if p:
            lock_base = os.path.dirname(p)
    lock_dir = os.path.join(lock_base, f".optuna_init_lock_{args.study_name}")
    _acquire_dir_lock(lock_dir)
    try:
        if isinstance(args.storage, str) and args.storage.startswith("sqlite"):
            _init_sqlite_pragmas(args.storage)

            try:
                from sqlalchemy.pool import NullPool  # type: ignore
                engine_kwargs = {"connect_args": {"timeout": 120}, "poolclass": NullPool}
            except Exception:
                engine_kwargs = {"connect_args": {"timeout": 120}}

            storage_obj = optuna.storages.RDBStorage(args.storage, engine_kwargs=engine_kwargs)

        for attempt in range(10):
            try:
                return optuna.create_study(
                    study_name=args.study_name,
                    storage=storage_obj,
                    direction="maximize",
                    load_if_exists=True,
                    pruner=pruner,
                    sampler=sampler,
                )
            except Exception as e:
                if "database is locked" in str(e).lower():
                    time.sleep(0.5 * (attempt + 1) + random.random() * 0.2)
                    continue
                raise

        raise RuntimeError("Could not create/load Optuna study due to persistent SQLite locking.")
    finally:
        _release_dir_lock(lock_dir)


def objective_factory(args: argparse.Namespace):
    # Parse once (names -> indices happens per-trial after class mapping exists)
    gate_names: List[str] = []
    if getattr(args, "gate_classes", None):
        gate_names = [c.strip() for c in str(args.gate_classes).split(",") if c.strip()]

    def objective(trial: "optuna.Trial") -> float:
        # ---- Search space (high-ROI, end-goal aligned) ----
        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.30)
        class_loss_type = trial.suggest_categorical("class_loss_type", ["bce", "ce"])

        lambda_class = trial.suggest_float("lambda_class", 0.1, 1.0)
        lambda_ent = trial.suggest_float("lambda_ent", 1e-6, 1e-2, log=True)
        lambda_cont = trial.suggest_float("lambda_cont", 1e-6, 1e-2, log=True)

        class_weight_power = trial.suggest_float("class_weight_power", 0.3, 1.3)
        class_weight_cap = trial.suggest_float("class_weight_cap", 2.0, 50.0, log=True)

        t_k = trial.suggest_float("t_k", 0.55, 0.99)
        t_delta = trial.suggest_float("t_delta", 0.0, 0.35)

        max_len = int(trial.suggest_categorical("max_len", [512, 1024, 2048]))
        batch_size = int(trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]))
        tok_budget = int(trial.suggest_categorical("tok_budget", [8192, 16384, 32768]))
        max_allowed = tok_budget // max_len
        if batch_size > max_allowed:
            trial.set_user_attr(
                "pruned_reason",
                f"batch_size={batch_size} > max_allowed={max_allowed} (tok_budget={tok_budget}, max_len={max_len})",
            )
            raise optuna.exceptions.TrialPruned()

        lr_factor = trial.suggest_categorical("lr_factor", [0.3, 0.5, 0.7])
        lr_patience = trial.suggest_int("lr_patience", 2, 5)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        if args.print_trial_header:
            print(
                f"\n=== Trial {trial.number} start (CUDA_VISIBLE_DEVICES={visible}) ===\n"
                f"  lr={lr:.2e} wd={weight_decay:.2e} dropout={dropout:.3f} class_loss_type={class_loss_type}\n"
                f"  lambda_class={lambda_class:.3f} lambda_ent={lambda_ent:.2e} lambda_cont={lambda_cont:.2e}\n"
                f"  class_weight_power={class_weight_power:.2f} class_weight_cap={class_weight_cap:.2f}\n"
                f"  gate(t_k={t_k:.2f}, t_delta={t_delta:.2f}) gate_classes={','.join(gate_names) if gate_names else '(none)'}\n"
                f"  batch_size={batch_size} max_len={max_len} lr_factor={lr_factor} lr_patience={lr_patience}",
                flush=True,
            )

        data = build_data(
            args.train_tsv,
            args.val_tsv,
            args.test_tsv,
            feature_dir=args.feature_dir,
            features_lmdb=args.features_lmdb,
            max_len=int(max_len),
            batch_size=int(batch_size),
            num_workers=args.num_workers,
            class_weight_power=float(class_weight_power),
            class_weight_cap=float(class_weight_cap),
            random_crop_train=bool(getattr(args, "random_crop_train", True)),
        )

        gate_idx_tuple: Tuple[int, ...] = tuple(
            data.class_to_idx[c] for c in gate_names
            if (c in data.class_to_idx and c != NON_ARG_LABEL)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(data.n_classes, dropout=dropout, max_len=int(max_len), device=device)

        bce_pos_weight_t = torch.tensor([data.bce_pos_weight], device=device)
        class_weights_t = torch.tensor(data.class_weights, device=device)

        criterion_bin_train = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t)
        criterion_bin_eval = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t, reduction="sum")
        criterion_class_train, criterion_class_eval = build_class_criteria(class_loss_type, class_weights_t, label_smoothing=args.class_label_smoothing)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(lr_factor),
            patience=int(lr_patience),
        )

        best_score = -1.0
        epochs_no_improve = 0

        try:
            for epoch in range(1, args.max_epochs + 1):
                train_stats = train_one_epoch(
                    model,
                    data.train_loader,
                    optimizer,
                    device,
                    lambda_class=float(lambda_class),
                    lambda_ent=float(lambda_ent),
                    lambda_cont=float(lambda_cont),
                    criterion_bin=criterion_bin_train,
                    criterion_class=criterion_class_train,
                    class_loss_type=class_loss_type,
                    trial_number=trial.number,
                    epoch_number=epoch,
                    log_batch_progress=args.log_batch_progress,
                )

                val_metrics, _ = evaluate_with_outputs(
                    model,
                    data.val_loader,
                    device,
                    lambda_class=float(lambda_class),
                    tune_threshold=False,
                    criterion_bin=criterion_bin_eval,
                    criterion_class=criterion_class_eval,
                    class_loss_type=class_loss_type,
                    gate_class_indices=gate_idx_tuple if len(gate_idx_tuple) else None,
                    gate_tau=float(t_k),
                    gate_delta=float(t_delta),
                )

                score = float(val_metrics[args.metric])
                scheduler.step(score)

                if args.print_epoch_summary:
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"[trial {trial.number}] epoch {epoch}/{args.max_epochs} "
                        f"train_loss={train_stats['loss_total']:.4f} "
                        f"val_{args.metric}={score:.4f} "
                        f"val_auc={val_metrics['auc_roc']:.4f} "
                        f"val_f1_bin={val_metrics['f1_bin']:.4f} "
                        f"lr={lr_now:.2e}",
                        flush=True,
                    )

                trial.report(score, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if score > best_score + args.min_delta:
                    best_score = score
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= args.patience:
                    break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise optuna.exceptions.TrialPruned()
            raise
        finally:
            del model
            del optimizer
            del scheduler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if getattr(args, "print_completed_trials", False):
            print(
                f"[trial {trial.number}] COMPLETE best_{args.metric}={best_score:.6f} "
                f"(loss={class_loss_type}, max_len={max_len}, bs={batch_size}, lr={lr:.2e}, wd={weight_decay:.2e})",
                flush=True,
            )

        return ensure_float(best_score)

    return objective


def ensure_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def resolved_hp_from_params(params: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    hp = dict(params)
    return {
        "lr": float(hp.get("lr", args.default_lr)),
        "weight_decay": float(hp.get("weight_decay", args.default_weight_decay)),
        "dropout": float(hp.get("dropout", args.default_dropout)),
        "class_loss_type": str(hp.get("class_loss_type", args.default_class_loss_type)).lower(),
        "lambda_class": float(hp.get("lambda_class", args.default_lambda_class)),
        "lambda_ent": float(hp.get("lambda_ent", args.default_lambda_ent)),
        "lambda_cont": float(hp.get("lambda_cont", args.default_lambda_cont)),
        "batch_size": int(hp.get("batch_size", args.default_batch_size)),
        "max_len": int(hp.get("max_len", args.default_max_len)),
        "lr_factor": float(hp.get("lr_factor", args.default_lr_factor)),
        "lr_patience": int(hp.get("lr_patience", args.default_lr_patience)),
        "class_weight_power": float(hp.get("class_weight_power", args.default_class_weight_power)),
        "class_weight_cap": float(hp.get("class_weight_cap", args.default_class_weight_cap)),
        "t_k": float(hp.get("t_k", args.default_t_k)),
        "t_delta": float(hp.get("t_delta", args.default_t_delta)),
    }


def top_complete_trials(study: "optuna.Study", top_k: int) -> List["optuna.Trial"]:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    complete.sort(key=lambda t: (-float(t.value), int(t.number)))
    return complete[: int(top_k)]


def select_median_seed_summary(summaries: List[Dict[str, Any]], key: str = "val_f1_class_macro_pos") -> Dict[str, Any]:
    ordered = sorted(summaries, key=lambda d: (float(d.get(key, float("nan"))), int(d.get("seed", 0))))
    return ordered[len(ordered) // 2]


def aggregate_seed_stats(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "best_score",
        "val_f1_class_macro_pos", "test_f1_class_macro_pos",
        "val_f1_bin", "test_f1_bin",
        "val_auc_roc", "test_auc_roc",
        "val_loss_total", "test_loss_total",
        "threshold_used",
    ]
    rows = []
    for m in metrics:
        vals = [float(s.get(m, float("nan"))) for s in summaries]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        rows.append({
            "metric": m,
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        })
    return rows


def build_ensemble_outputs(seed_summaries: List[Dict[str, Any]], split_name: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not seed_summaries:
        raise ValueError("No seed summaries provided for ensemble building.")
    ref = seed_summaries[0]["_outputs"][split_name]
    seq_ids = list(ref["seq_ids"])
    bin_labels = ref["bin_labels"].copy()
    class_labels_all = ref["class_labels_all"].copy()

    bin_stack = []
    class_stack = []
    loss_total_vals = []
    loss_bin_vals = []
    loss_class_vals = []
    for s in seed_summaries:
        out = s["_outputs"][split_name]
        if list(out["seq_ids"]) != seq_ids:
            raise ValueError(f"Seed predictions are not aligned for split={split_name}; seq_id ordering differs.")
        bin_stack.append(out["bin_probs"])
        class_stack.append(out["class_probs_all"])
        loss_total_vals.append(float(s.get(f"{split_name}_loss_total", float("nan"))))
        loss_bin_vals.append(float(s.get(f"{split_name}_loss_bin", float("nan"))))
        loss_class_vals.append(float(s.get(f"{split_name}_loss_class_pos", float("nan"))))

    bin_probs = np.mean(np.stack(bin_stack, axis=0), axis=0)
    class_probs_all = np.mean(np.stack(class_stack, axis=0), axis=0)

    loss_total = float(np.nanmean(loss_total_vals)) if loss_total_vals else float("nan")
    loss_bin = float(np.nanmean(loss_bin_vals)) if loss_bin_vals else float("nan")
    loss_class_pos = float(np.nanmean(loss_class_vals)) if loss_class_vals else float("nan")

    return evaluate_arrays(
        seq_ids=seq_ids,
        bin_labels=bin_labels,
        bin_probs=bin_probs,
        class_labels_all=class_labels_all,
        class_probs_all=class_probs_all,
        loss_total=loss_total,
        loss_bin=loss_bin,
        loss_class_pos=loss_class_pos,
        threshold=0.5,
        tune_threshold=(split_name == "val"),
        gate_class_indices=seed_summaries[0]["_gate_idx"],
        gate_tau=float(seed_summaries[0]["_hp"]["t_k"]),
        gate_delta=float(seed_summaries[0]["_hp"]["t_delta"]),
    )


def run_retrain_seed(
    args: argparse.Namespace,
    hp: Dict[str, Any],
    *,
    seed: int,
    out_dir: str,
    early_stop_metric: str,
    epochs: int,
    patience: int,
    trial_number: Optional[int],
    evaluate_test: bool,
    save_full_outputs: bool,
    return_outputs: bool,
) -> Dict[str, Any]:
    set_seed(seed)
    ensure_dir(out_dir)

    gate_names = [s.strip() for s in (args.gate_classes or "").split(",") if s.strip()]
    data = build_data(
        args.train_tsv,
        args.val_tsv,
        args.test_tsv,
        feature_dir=args.feature_dir,
        features_lmdb=args.features_lmdb,
        max_len=int(hp["max_len"]),
        batch_size=int(hp["batch_size"]),
        num_workers=args.num_workers,
        class_weight_power=float(hp["class_weight_power"]),
        class_weight_cap=float(hp["class_weight_cap"]),
        random_crop_train=bool(args.random_crop_train),
    )

    gate_idx = tuple(
        data.class_to_idx[n] for n in gate_names
        if (n in data.class_to_idx and n != NON_ARG_LABEL)
    )
    idx_to_class = {i: c for c, i in data.class_to_idx.items()}

    hp_rows = [{"key": k, "value": hp[k]} for k in sorted(hp)]
    save_tsv_rows(os.path.join(out_dir, "hp.tsv"), hp_rows, ["key", "value"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(data.n_classes, dropout=float(hp["dropout"]), max_len=int(hp["max_len"]), device=device)

    bce_pos_weight_t = torch.tensor([data.bce_pos_weight], device=device)
    class_weights_t = torch.tensor(data.class_weights, device=device)

    criterion_bin_train = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t)
    criterion_bin_eval = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t, reduction="sum")
    criterion_class_train, criterion_class_eval = build_class_criteria(hp["class_loss_type"], class_weights_t, label_smoothing=args.class_label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(hp["lr_factor"]),
        patience=int(hp["lr_patience"]),
    )

    best_score = -1.0
    best_epoch = 0
    best_path = os.path.join(out_dir, "best_checkpoint.pt")
    epochs_no_improve = 0
    history_rows = []

    for epoch in range(1, int(epochs) + 1):
        train_stats = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            device,
            lambda_class=float(hp["lambda_class"]),
            lambda_ent=float(hp["lambda_ent"]),
            lambda_cont=float(hp["lambda_cont"]),
            criterion_bin=criterion_bin_train,
            criterion_class=criterion_class_train,
            class_loss_type=str(hp["class_loss_type"]),
            trial_number=trial_number,
            epoch_number=epoch,
            log_batch_progress=args.log_batch_progress,
        )

        val_metrics_epoch, _ = evaluate_with_outputs(
            model,
            data.val_loader,
            device,
            lambda_class=float(hp["lambda_class"]),
            threshold=0.5,
            tune_threshold=False,
            criterion_bin=criterion_bin_eval,
            criterion_class=criterion_class_eval,
            class_loss_type=str(hp["class_loss_type"]),
            gate_class_indices=gate_idx,
            gate_tau=float(hp["t_k"]),
            gate_delta=float(hp["t_delta"]),
        )

        score = float(val_metrics_epoch.get(early_stop_metric, val_metrics_epoch.get(args.metric, float("nan"))))
        scheduler.step(score)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history_rows.append({
            "epoch": epoch,
            "lr": lr_now,
            "class_loss_type": hp["class_loss_type"],
            "train_loss_total": train_stats["loss_total"],
            "train_loss_bin": train_stats["loss_bin"],
            "train_loss_class_pos": train_stats["loss_class_pos"],
            "val_loss_total": val_metrics_epoch["loss_total"],
            "val_f1_class_macro_pos": val_metrics_epoch.get("f1_class_macro_pos", float("nan")),
            "val_f1_bin": val_metrics_epoch.get("f1_bin", float("nan")),
            "val_auc_roc": val_metrics_epoch.get("auc_roc", float("nan")),
            "val_n_pos": val_metrics_epoch.get("n_pos", float("nan")),
        })

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hp": hp,
                "seed": seed,
                "trial_number": trial_number,
            }, best_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= int(patience):
            break

    save_tsv_rows(os.path.join(out_dir, "history.tsv"), history_rows, list(history_rows[0].keys()) if history_rows else ["epoch"])

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_out = evaluate_with_outputs(
        model,
        data.val_loader,
        device,
        lambda_class=float(hp["lambda_class"]),
        threshold=0.5,
        tune_threshold=True,
        criterion_bin=criterion_bin_eval,
        criterion_class=criterion_class_eval,
        class_loss_type=str(hp["class_loss_type"]),
        gate_class_indices=gate_idx,
        gate_tau=float(hp["t_k"]),
        gate_delta=float(hp["t_delta"]),
    )

    split_metrics = {"val": val_metrics}
    split_outputs = {"val": val_out}
    test_metrics = None
    test_out = None

    if evaluate_test:
        test_metrics, test_out = evaluate_with_outputs(
            model,
            data.test_loader,
            device,
            lambda_class=float(hp["lambda_class"]),
            threshold=float(val_metrics.get("threshold_used", 0.5)),
            tune_threshold=False,
            criterion_bin=criterion_bin_eval,
            criterion_class=criterion_class_eval,
            class_loss_type=str(hp["class_loss_type"]),
            gate_class_indices=gate_idx,
            gate_tau=float(hp["t_k"]),
            gate_delta=float(hp["t_delta"]),
        )
        split_metrics["test"] = test_metrics
        split_outputs["test"] = test_out

    save_metrics_tsv(os.path.join(out_dir, "metrics.tsv"), split_metrics)

    if save_full_outputs:
        for split_name, split_out in split_outputs.items():
            save_prediction_tsv(os.path.join(out_dir, f"{split_name}_predictions.tsv"), split_out, idx_to_class)
            gate_rows = compute_gate_audit_rows(split_out, idx_to_class, gate_names, float(hp["t_k"]), float(hp["t_delta"]))
            save_gate_audit_tsv(os.path.join(out_dir, f"gate_audit_{split_name}.tsv"), gate_rows)

            y_true = split_out["class_labels_pos"]
            y_pred = split_out["class_preds_pos"]
            y_prob = split_out["class_probs_pos"]
            rep, cm, roc_ovr, pr_ovr = per_class_reports(y_true, y_pred, y_prob, data.n_classes)
            save_reports(
                out_dir,
                split_name=split_name,
                report=rep,
                cm=cm,
                roc_ovr=roc_ovr,
                pr_ovr=pr_ovr,
                idx_to_class=idx_to_class,
                prefix="pos",
            )

    summary = {
        "trial_number": int(trial_number) if trial_number is not None else -1,
        "seed": int(seed),
        "class_loss_type": str(hp["class_loss_type"]),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "val_f1_class_macro_pos": float(val_metrics.get("f1_class_macro_pos", float("nan"))),
        "val_f1_bin": float(val_metrics.get("f1_bin", float("nan"))),
        "val_auc_roc": float(val_metrics.get("auc_roc", float("nan"))),
        "val_loss_total": float(val_metrics.get("loss_total", float("nan"))),
        "val_loss_bin": float(val_metrics.get("loss_bin", float("nan"))),
        "val_loss_class_pos": float(val_metrics.get("loss_class_pos", float("nan"))),
        "threshold_used": float(val_metrics.get("threshold_used", float("nan"))),
        "seed_dir": out_dir,
    }
    if test_metrics is not None:
        summary.update({
            "test_f1_class_macro_pos": float(test_metrics.get("f1_class_macro_pos", float("nan"))),
            "test_f1_bin": float(test_metrics.get("f1_bin", float("nan"))),
            "test_auc_roc": float(test_metrics.get("auc_roc", float("nan"))),
            "test_loss_total": float(test_metrics.get("loss_total", float("nan"))),
            "test_loss_bin": float(test_metrics.get("loss_bin", float("nan"))),
            "test_loss_class_pos": float(test_metrics.get("loss_class_pos", float("nan"))),
        })
    else:
        summary.update({
            "test_f1_class_macro_pos": float("nan"),
            "test_f1_bin": float("nan"),
            "test_auc_roc": float("nan"),
            "test_loss_total": float("nan"),
            "test_loss_bin": float("nan"),
            "test_loss_class_pos": float("nan"),
        })

    if return_outputs:
        summary["_outputs"] = split_outputs
        summary["_gate_idx"] = gate_idx
        summary["_idx_to_class"] = idx_to_class
        summary["_class_to_idx"] = data.class_to_idx
        summary["_hp"] = dict(hp)

    del model
    del optimizer
    del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _smart_parse_value(x: str) -> Any:
    s = str(x)
    if s == "":
        return ""
    sl = s.lower()
    if sl == "nan":
        return float("nan")
    if sl in {"true", "false"}:
        return sl == "true"
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s


def load_kv_tsv(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header[:2] != ["key", "value"]:
            raise ValueError(f"Expected key/value TSV at {path}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            out[parts[0]] = _smart_parse_value(parts[1])
    return out


def save_summary_tsv(path: str, summary: Dict[str, Any]) -> None:
    cols = list(summary.keys())
    save_tsv_rows(path, [summary], cols)


def load_single_row_tsv(path: str) -> Dict[str, Any]:
    import csv
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            return {k: _smart_parse_value(v) for k, v in row.items()}
    raise ValueError(f"No data rows found in {path}")


def parse_parallel_gpus(args: argparse.Namespace) -> List[str]:
    if getattr(args, "parallel_gpus", None):
        gpus = [g.strip() for g in str(args.parallel_gpus).split(",") if g.strip()]
        if gpus:
            return gpus
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        toks = [t.strip() for t in env.split(",") if t.strip()]
        if len(toks) > 1:
            return toks
        if len(toks) == 1:
            return [toks[0]]
    return ["0"]


def load_prediction_tsv(path: str, class_to_idx: Dict[str, int]) -> Dict[str, Any]:
    import csv
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    class_cols = [(i, f"prob_{sanitize_colname(idx_to_class[i])}") for i in sorted(idx_to_class)]
    seq_ids = []
    bin_labels = []
    bin_probs = []
    bin_preds_0p5 = []
    bin_preds_used = []
    class_labels_all = []
    class_preds_raw_all = []
    class_preds_all = []
    class_top2_idx_all = []
    class_top1_prob_all = []
    class_top2_prob_all = []
    class_margin_all = []
    gate_applied_all = []
    class_probs_rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            seq_ids.append(row["seq_id"])
            bin_labels.append(int(row["true_bin"]))
            bin_probs.append(float(row["bin_prob"]))
            bin_preds_0p5.append(int(row["pred_bin_0p5"]))
            bin_preds_used.append(int(row["pred_bin_used"]))
            class_labels_all.append(int(row["true_class_index"]))
            class_preds_raw_all.append(int(row["pred_class_raw_index"]))
            class_preds_all.append(int(row["pred_class_index"]))
            class_top2_idx_all.append(int(row["second_class_index"]))
            class_top1_prob_all.append(float(row["top1_prob"]))
            class_top2_prob_all.append(float(row["top2_prob"]))
            class_margin_all.append(float(row["margin"]))
            gate_applied_all.append(int(row["gate_applied"]))
            class_probs_rows.append([float(row[col]) for _, col in class_cols])
    class_probs_all = np.asarray(class_probs_rows, dtype=np.float32)
    bin_labels_arr = np.asarray(bin_labels, dtype=np.int64)
    outputs = {
        "seq_ids": seq_ids,
        "bin_labels": bin_labels_arr,
        "bin_probs": np.asarray(bin_probs, dtype=np.float32),
        "bin_preds_0p5": np.asarray(bin_preds_0p5, dtype=np.int64),
        "bin_preds_used": np.asarray(bin_preds_used, dtype=np.int64),
        "class_labels_all": np.asarray(class_labels_all, dtype=np.int64),
        "class_probs_all": class_probs_all,
        "class_preds_raw_all": np.asarray(class_preds_raw_all, dtype=np.int64),
        "class_preds_all": np.asarray(class_preds_all, dtype=np.int64),
        "class_top2_idx_all": np.asarray(class_top2_idx_all, dtype=np.int64),
        "class_top1_prob_all": np.asarray(class_top1_prob_all, dtype=np.float32),
        "class_top2_prob_all": np.asarray(class_top2_prob_all, dtype=np.float32),
        "class_margin_all": np.asarray(class_margin_all, dtype=np.float32),
        "gate_applied_all": np.asarray(gate_applied_all, dtype=np.int64),
    }
    pos_mask = bin_labels_arr == 1
    outputs["class_labels_pos"] = outputs["class_labels_all"][pos_mask]
    outputs["class_preds_pos"] = outputs["class_preds_all"][pos_mask]
    outputs["class_probs_pos"] = outputs["class_probs_all"][pos_mask]
    return outputs


def launch_worker_jobs(args: argparse.Namespace, jobs: List[Dict[str, Any]], phase_name: str) -> List[Dict[str, Any]]:
    gpus = parse_parallel_gpus(args)
    max_jobs = max(1, min(int(getattr(args, "max_parallel_jobs", 1)), len(gpus)))
    num_workers_per_job = max(1, int(args.num_workers) // max_jobs)
    queue = list(jobs)
    running: List[Tuple[subprocess.Popen, str, Dict[str, Any]]] = []
    done: List[Dict[str, Any]] = []
    free_gpus = list(gpus)
    script_path = os.path.abspath(__file__)

    def launch_one(job: Dict[str, Any], gpu: str):
        out_dir = ensure_dir(job["out_dir"])
        hp_tsv = os.path.join(out_dir, "worker_hp.tsv")
        save_tsv_rows(hp_tsv, [{"key": k, "value": job["hp"][k]} for k in sorted(job["hp"])], ["key", "value"])
        stdout_path = os.path.join(out_dir, "worker.out")
        stderr_path = os.path.join(out_dir, "worker.err")
        cmd = [
            sys.executable, script_path,
            "--mode", "retrain_seed_worker",
            "--train_tsv", args.train_tsv,
            "--val_tsv", args.val_tsv,
            "--test_tsv", args.test_tsv,
            "--out_dir", args.out_dir,
            "--study_name", args.study_name,
            "--storage", args.storage,
            "--num_workers", str(num_workers_per_job),
            "--metric", args.metric,
            "--retrain_metric", args.retrain_metric,
            "--gate_classes", args.gate_classes,
            "--class_label_smoothing", str(args.class_label_smoothing),
            "--worker_hp_tsv", hp_tsv,
            "--worker_seed", str(job["seed"]),
            "--worker_out_dir", out_dir,
            "--worker_trial_number", str(job.get("trial_number", -1)),
            "--worker_early_stop_metric", job["early_stop_metric"],
            "--worker_epochs", str(job["epochs"]),
            "--worker_patience", str(job["patience"]),
            "--optuna_verbosity", args.optuna_verbosity,
        ]
        if args.features_lmdb:
            cmd.extend(["--features_lmdb", args.features_lmdb])
        else:
            cmd.extend(["--feature_dir", args.feature_dir])
        if args.random_crop_train:
            cmd.append("--random_crop_train")
        if job.get("evaluate_test", False):
            cmd.append("--worker_evaluate_test")
        if job.get("save_full_outputs", False):
            cmd.append("--worker_save_full_outputs")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        so = open(stdout_path, "w")
        se = open(stderr_path, "w")
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, env=env)
        job["_stdout_fh"] = so
        job["_stderr_fh"] = se
        print(f"[{phase_name}] launched seed={job['seed']} trial={job.get('trial_number', -1)} on GPU {gpu}", flush=True)
        return proc

    while queue or running:
        while queue and free_gpus and len(running) < max_jobs:
            job = queue.pop(0)
            gpu = free_gpus.pop(0)
            proc = launch_one(job, gpu)
            running.append((proc, gpu, job))

        time.sleep(1.0)
        still_running: List[Tuple[subprocess.Popen, str, Dict[str, Any]]] = []
        for proc, gpu, job in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, gpu, job))
                continue
            try:
                job["_stdout_fh"].close(); job["_stderr_fh"].close()
            except Exception:
                pass
            free_gpus.append(gpu)
            if rc != 0:
                raise RuntimeError(f"{phase_name} worker failed for seed={job['seed']} trial={job.get('trial_number', -1)} on GPU {gpu}; see {job['out_dir']}/worker.err")
            done.append(load_single_row_tsv(os.path.join(job["out_dir"], "summary.tsv")))
        running = still_running

    return done


def retrain_seed_worker(args: argparse.Namespace) -> None:
    if not args.worker_hp_tsv or args.worker_seed is None or not args.worker_out_dir:
        raise ValueError("Worker mode requires --worker_hp_tsv, --worker_seed, and --worker_out_dir")
    hp = load_kv_tsv(args.worker_hp_tsv)
    summary = run_retrain_seed(
        args,
        hp,
        seed=int(args.worker_seed),
        out_dir=args.worker_out_dir,
        early_stop_metric=str(args.worker_early_stop_metric or args.retrain_metric or args.metric),
        epochs=int(args.worker_epochs or args.retrain_epochs),
        patience=int(args.worker_patience or args.retrain_patience),
        trial_number=int(args.worker_trial_number) if args.worker_trial_number is not None else -1,
        evaluate_test=bool(args.worker_evaluate_test),
        save_full_outputs=bool(args.worker_save_full_outputs),
        return_outputs=False,
    )
    save_summary_tsv(os.path.join(args.worker_out_dir, "summary.tsv"), summary)


def retrain_best(args: argparse.Namespace) -> None:
    import optuna

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    ensure_dir(args.out_dir)
    write_study_trials_tsv(study, os.path.join(args.out_dir, f"{args.study_name}_trials_summary.tsv"))

    top_trials = top_complete_trials(study, args.confirm_top_k)
    if not top_trials:
        raise RuntimeError("No COMPLETE Optuna trials found for confirmation/retraining.")

    top_rows = []
    for t in top_trials:
        row = {"trial_number": int(t.number), "value": float(t.value), "params_json": json.dumps(t.params, sort_keys=True)}
        row.update({k: t.params.get(k, "") for k in sorted(t.params)})
        top_rows.append(row)
    top_cols = ["trial_number", "value"] + sorted({k for r in top_rows for k in r if k not in {"trial_number", "value", "params_json"}}) + ["params_json"]
    save_tsv_rows(os.path.join(args.out_dir, f"{args.study_name}_top_trials.tsv"), top_rows, top_cols)

    confirm_seeds = [int(s.strip()) for s in str(args.confirm_seeds).split(",") if s.strip()] or [1,2,3]
    confirm_root = ensure_dir(os.path.join(args.out_dir, f"{args.retrain_tag}_confirm_top{int(args.confirm_top_k)}"))
    confirm_jobs = []
    for t in top_trials:
        hp = resolved_hp_from_params(t.params, args)
        for seed in confirm_seeds:
            seed_dir = ensure_dir(os.path.join(confirm_root, f"trial_{int(t.number)}", f"seed_{seed}"))
            confirm_jobs.append({
                "trial_number": int(t.number),
                "hp": hp,
                "seed": int(seed),
                "out_dir": seed_dir,
                "early_stop_metric": args.retrain_metric,
                "epochs": args.confirmation_epochs,
                "patience": args.confirmation_patience,
                "evaluate_test": False,
                "save_full_outputs": False,
            })
    confirmation_seed_rows = launch_worker_jobs(args, confirm_jobs, phase_name="confirm")
    save_tsv_rows(
        os.path.join(confirm_root, "confirmation_seed_metrics.tsv"),
        confirmation_seed_rows,
        ["trial_number", "seed", "class_loss_type", "val_f1_class_macro_pos", "val_f1_bin", "val_auc_roc", "best_epoch", "best_score", "seed_dir"],
    )

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in confirmation_seed_rows:
        grouped.setdefault(int(row["trial_number"]), []).append(row)
    confirmation_summary_rows = []
    for t in top_trials:
        rows = grouped.get(int(t.number), [])
        vals = np.asarray([float(r.get("val_f1_class_macro_pos", float("nan"))) for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        hp = resolved_hp_from_params(t.params, args)
        confirmation_summary_rows.append({
            "trial_number": int(t.number),
            "class_loss_type": hp["class_loss_type"],
            "mean_val_f1_class_macro_pos": float(vals.mean()) if vals.size else float("nan"),
            "sd_val_f1_class_macro_pos": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median_val_f1_class_macro_pos": float(np.median(vals)) if vals.size else float("nan"),
            "n_seeds": int(vals.size),
        })
    save_tsv_rows(
        os.path.join(confirm_root, "confirmation_summary.tsv"),
        confirmation_summary_rows,
        ["trial_number", "class_loss_type", "mean_val_f1_class_macro_pos", "sd_val_f1_class_macro_pos", "median_val_f1_class_macro_pos", "n_seeds"],
    )

    selected = sorted(
        confirmation_summary_rows,
        key=lambda r: (-float(r["mean_val_f1_class_macro_pos"]), float(r["sd_val_f1_class_macro_pos"]), -float(r["median_val_f1_class_macro_pos"]), int(r["trial_number"])),
    )[0]
    selected_trial = next(t for t in top_trials if int(t.number) == int(selected["trial_number"]))
    selected_hp = resolved_hp_from_params(selected_trial.params, args)
    save_tsv_rows(
        os.path.join(args.out_dir, f"{args.retrain_tag}_selected_trial.tsv"),
        [{
            "trial_number": int(selected_trial.number),
            "value": float(selected_trial.value),
            "class_loss_type": selected_hp["class_loss_type"],
            "mean_val_f1_class_macro_pos": float(selected["mean_val_f1_class_macro_pos"]),
            "sd_val_f1_class_macro_pos": float(selected["sd_val_f1_class_macro_pos"]),
            "median_val_f1_class_macro_pos": float(selected["median_val_f1_class_macro_pos"]),
            "params_json": json.dumps(selected_trial.params, sort_keys=True),
        }],
        ["trial_number", "value", "class_loss_type", "mean_val_f1_class_macro_pos", "sd_val_f1_class_macro_pos", "median_val_f1_class_macro_pos", "params_json"],
    )

    final_seeds = [int(s.strip()) for s in str(args.retrain_seeds).split(",") if s.strip()] if args.retrain_seeds else [int(args.seed)]
    out_root = ensure_dir(os.path.join(args.out_dir, args.retrain_tag))
    final_jobs = []
    for seed in final_seeds:
        seed_dir = ensure_dir(os.path.join(out_root, f"seed_{seed}"))
        final_jobs.append({
            "trial_number": int(selected_trial.number),
            "hp": selected_hp,
            "seed": int(seed),
            "out_dir": seed_dir,
            "early_stop_metric": args.retrain_metric,
            "epochs": args.retrain_epochs,
            "patience": args.retrain_patience,
            "evaluate_test": True,
            "save_full_outputs": True,
        })
    final_summaries = launch_worker_jobs(args, final_jobs, phase_name="final")

    per_seed_cols = [
        "trial_number", "seed", "class_loss_type", "best_epoch", "best_score",
        "val_f1_class_macro_pos", "test_f1_class_macro_pos",
        "val_f1_bin", "test_f1_bin",
        "val_auc_roc", "test_auc_roc",
        "val_loss_total", "test_loss_total",
        "threshold_used", "seed_dir",
    ]
    save_tsv_rows(os.path.join(out_root, "multiseed_summary.tsv"), final_summaries, per_seed_cols)
    save_tsv_rows(os.path.join(out_root, "multiseed_stats.tsv"), aggregate_seed_stats(final_summaries), ["metric", "n", "mean", "sd", "min", "median", "max"])

    representative = select_median_seed_summary(final_summaries, key="val_f1_class_macro_pos")
    save_tsv_rows(
        os.path.join(out_root, "representative_seed.tsv"),
        [{
            "seed": int(representative["seed"]),
            "val_f1_class_macro_pos": float(representative["val_f1_class_macro_pos"]),
            "test_f1_class_macro_pos": float(representative["test_f1_class_macro_pos"]),
            "seed_dir": representative["seed_dir"],
        }],
        ["seed", "val_f1_class_macro_pos", "test_f1_class_macro_pos", "seed_dir"],
    )

    rep_dir = str(representative["seed_dir"])
    for fn in [
        "best_checkpoint.pt", "history.tsv", "metrics.tsv",
        "val_predictions.tsv", "test_predictions.tsv",
        "gate_audit_val.tsv", "gate_audit_test.tsv",
        "pos_val_class_report.tsv", "pos_val_confusion.tsv",
        "pos_test_class_report.tsv", "pos_test_confusion.tsv",
    ]:
        src = os.path.join(rep_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_root, f"representative_{fn}"))

    class_to_idx = build_fixed_class_mapping(args.train_tsv, args.val_tsv, args.test_tsv)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    seed_outputs = []
    for s in final_summaries:
        sd = str(s["seed_dir"])
        seed_outputs.append({
            "summary": s,
            "val": load_prediction_tsv(os.path.join(sd, "val_predictions.tsv"), class_to_idx),
            "test": load_prediction_tsv(os.path.join(sd, "test_predictions.tsv"), class_to_idx),
        })

    ref_val = seed_outputs[0]["val"]
    ensemble_val_metrics, ensemble_val_out = evaluate_arrays(
        seq_ids=ref_val["seq_ids"],
        bin_labels=ref_val["bin_labels"],
        bin_probs=np.mean(np.stack([s["val"]["bin_probs"] for s in seed_outputs], axis=0), axis=0),
        class_labels_all=ref_val["class_labels_all"],
        class_probs_all=np.mean(np.stack([s["val"]["class_probs_all"] for s in seed_outputs], axis=0), axis=0),
        loss_total=float(np.nanmean([float(s["summary"].get("val_loss_total", float("nan"))) for s in seed_outputs])),
        loss_bin=float(np.nanmean([float(s["summary"].get("val_loss_bin", float("nan"))) for s in seed_outputs])),
        loss_class_pos=float(np.nanmean([float(s["summary"].get("val_loss_class_pos", float("nan"))) for s in seed_outputs])),
        threshold=0.5,
        tune_threshold=True,
        gate_class_indices=tuple(class_to_idx[n] for n in [x.strip() for x in (args.gate_classes or "").split(",") if x.strip()] if n in class_to_idx and n != NON_ARG_LABEL),
        gate_tau=float(selected_hp["t_k"]),
        gate_delta=float(selected_hp["t_delta"]),
    )
    ref_test = seed_outputs[0]["test"]
    ensemble_test_metrics, ensemble_test_out = evaluate_arrays(
        seq_ids=ref_test["seq_ids"],
        bin_labels=ref_test["bin_labels"],
        bin_probs=np.mean(np.stack([s["test"]["bin_probs"] for s in seed_outputs], axis=0), axis=0),
        class_labels_all=ref_test["class_labels_all"],
        class_probs_all=np.mean(np.stack([s["test"]["class_probs_all"] for s in seed_outputs], axis=0), axis=0),
        loss_total=float(np.nanmean([float(s["summary"].get("test_loss_total", float("nan"))) for s in seed_outputs])),
        loss_bin=float(np.nanmean([float(s["summary"].get("test_loss_bin", float("nan"))) for s in seed_outputs])),
        loss_class_pos=float(np.nanmean([float(s["summary"].get("test_loss_class_pos", float("nan"))) for s in seed_outputs])),
        threshold=float(ensemble_val_metrics.get("threshold_used", 0.5)),
        tune_threshold=False,
        gate_class_indices=tuple(class_to_idx[n] for n in [x.strip() for x in (args.gate_classes or "").split(",") if x.strip()] if n in class_to_idx and n != NON_ARG_LABEL),
        gate_tau=float(selected_hp["t_k"]),
        gate_delta=float(selected_hp["t_delta"]),
    )

    save_metrics_tsv(os.path.join(out_root, "ensemble_metrics.tsv"), {"val": ensemble_val_metrics, "test": ensemble_test_metrics})
    save_prediction_tsv(os.path.join(out_root, "ensemble_val_predictions.tsv"), ensemble_val_out, idx_to_class)
    save_prediction_tsv(os.path.join(out_root, "ensemble_test_predictions.tsv"), ensemble_test_out, idx_to_class)
    gate_names = [s.strip() for s in (args.gate_classes or "").split(",") if s.strip()]
    save_gate_audit_tsv(os.path.join(out_root, "ensemble_gate_audit_val.tsv"), compute_gate_audit_rows(ensemble_val_out, idx_to_class, gate_names, float(selected_hp["t_k"]), float(selected_hp["t_delta"])))
    save_gate_audit_tsv(os.path.join(out_root, "ensemble_gate_audit_test.tsv"), compute_gate_audit_rows(ensemble_test_out, idx_to_class, gate_names, float(selected_hp["t_k"]), float(selected_hp["t_delta"])))

    rep_val, cm_val, roc_val, pr_val = per_class_reports(ensemble_val_out["class_labels_pos"], ensemble_val_out["class_preds_pos"], ensemble_val_out["class_probs_pos"], len(idx_to_class))
    save_reports(out_root, split_name="val", report=rep_val, cm=cm_val, roc_ovr=roc_val, pr_ovr=pr_val, idx_to_class=idx_to_class, prefix="ensemble_pos")
    rep_test, cm_test, roc_test, pr_test = per_class_reports(ensemble_test_out["class_labels_pos"], ensemble_test_out["class_preds_pos"], ensemble_test_out["class_probs_pos"], len(idx_to_class))
    save_reports(out_root, split_name="test", report=rep_test, cm=cm_test, roc_ovr=roc_test, pr_ovr=pr_test, idx_to_class=idx_to_class, prefix="ensemble_pos")

    print("\nFinal retrain summary (multi-seed):", flush=True)
    stats_rows = aggregate_seed_stats(final_summaries)
    for metric in ["val_f1_class_macro_pos", "test_f1_class_macro_pos", "val_f1_bin", "test_f1_bin", "val_auc_roc", "test_auc_roc"]:
        row = next((r for r in stats_rows if r["metric"] == metric), None)
        if row is not None:
            print(f"  {metric}: {row['mean']:.4f} ± {row['sd']:.4f}", flush=True)
    print(f"  Representative (median seed by val_f1_class_macro_pos): seed {representative['seed']}", flush=True)
    print(f"  Ensemble val_f1_class_macro_pos={ensemble_val_metrics['f1_class_macro_pos']:.4f}", flush=True)
    print(f"  Ensemble test_f1_class_macro_pos={ensemble_test_metrics['f1_class_macro_pos']:.4f}", flush=True)
    print(f"  Reports written to: {out_root}", flush=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["tune", "retrain_best", "retrain_seed_worker"], default="tune")

    # Data
    p.add_argument("--train_tsv", required=True)
    p.add_argument("--val_tsv", required=True)
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--feature_dir", default=None, help="Directory with *.npy features (if not using LMDB)")
    p.add_argument("--features_lmdb", default=None, help="Path to features LMDB directory")
    p.add_argument("--out_dir", required=True)

    # Optuna
    p.add_argument("--study_name", default="amrfold_hpo")
    p.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////abs/path/amrfold_optuna.db")
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None, help="Seconds; optional")

    p.add_argument(
        "--metric",
        choices=["auc_roc", "auc_pr", "f1_bin", "f1_class_macro_pos"],
        default="f1_class_macro_pos",
        help="Optimisation metric computed on VAL (end-goal default: f1_class_macro_pos)",
    )

    p.add_argument("--max_epochs", type=int, default=20, help="Max epochs per trial")
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience per trial")
    p.add_argument("--min_delta", type=float, default=1e-4)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--pruner", choices=["none", "median"], default="median")

    # Confirmation + retrain
    p.add_argument("--confirm_top_k", type=int, default=3, help="Number of top Optuna trials to confirm.")
    p.add_argument("--confirm_seeds", default="1,2,3", help="Comma-separated seeds for top-k confirmation stage.")
    p.add_argument("--confirmation_epochs", type=int, default=40, help="Epochs per seed during top-k confirmation.")
    p.add_argument("--confirmation_patience", type=int, default=8, help="Early stopping patience during top-k confirmation.")

    p.add_argument("--retrain_epochs", type=int, default=40)
    p.add_argument("--retrain_patience", type=int, default=8)
    p.add_argument("--retrain_metric", default=None, help="Metric for retrain early stopping; default=--metric")
    p.add_argument("--retrain_tag", default=None, help="Subdir name under out_dir for retrain outputs")
    p.add_argument(
        "--retrain_seeds",
        default=None,
        help="Comma-separated seeds for final multi-seed retrain (e.g. '1,2,3,4,5'). If unset: uses --seed only.",
    )

    p.add_argument("--parallel_gpus", default="", help="Comma-separated visible GPU IDs to use for parallel confirmation/retrain workers, e.g. 0,1.")
    p.add_argument("--max_parallel_jobs", type=int, default=1, help="Maximum number of concurrent worker subprocesses in confirmation/retrain.")

    # Internal worker mode args
    p.add_argument("--worker_hp_tsv", default=None)
    p.add_argument("--worker_seed", type=int, default=None)
    p.add_argument("--worker_out_dir", default=None)
    p.add_argument("--worker_trial_number", type=int, default=-1)
    p.add_argument("--worker_early_stop_metric", default=None)
    p.add_argument("--worker_epochs", type=int, default=None)
    p.add_argument("--worker_patience", type=int, default=None)
    p.add_argument("--worker_evaluate_test", action="store_true")
    p.add_argument("--worker_save_full_outputs", action="store_true")

    # Rare-class gate (you can add new classes here)
    p.add_argument(
        "--gate_classes",
        default="peptide,rifamycin,sulfonamide",
        help="Comma-separated class names to apply the rare-class gate to.",
    )

    # Data augmentation
    # Keep your original behavior (default True), but allow disabling explicitly.
    p.add_argument(
        "--random_crop_train",
        action="store_true",
        default=True,
        help="Enable random cropping for proteins longer than max_len during TRAIN (recommended).",
    )
    p.add_argument(
        "--no_random_crop_train",
        action="store_false",
        dest="random_crop_train",
        help="Disable random cropping during TRAIN.",
    )

    # Defaults (used if a study has missing params)
    p.add_argument("--lr_default", dest="default_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay_default", dest="default_weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout_default", dest="default_dropout", type=float, default=0.1)
    p.add_argument("--class_loss_type_default", dest="default_class_loss_type", default="ce", choices=["bce", "ce"])
    p.add_argument("--class_label_smoothing", type=float, default=0.05, help="Label smoothing for CE class head.")
    p.add_argument("--lambda_class_default", dest="default_lambda_class", type=float, default=0.5)
    p.add_argument("--lambda_ent_default", dest="default_lambda_ent", type=float, default=1e-3)
    p.add_argument("--lambda_cont_default", dest="default_lambda_cont", type=float, default=1e-3)
    p.add_argument("--batch_size_default", dest="default_batch_size", type=int, default=16)
    p.add_argument("--max_len_default", dest="default_max_len", type=int, default=1024)
    p.add_argument("--lr_factor_default", dest="default_lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience_default", dest="default_lr_patience", type=int, default=3)

    p.add_argument("--class_weight_power_default", dest="default_class_weight_power", type=float, default=1.0)
    p.add_argument("--class_weight_cap_default", dest="default_class_weight_cap", type=float, default=10.0)

    p.add_argument("--t_k_default", dest="default_t_k", type=float, default=0.8)
    p.add_argument("--t_delta_default", dest="default_t_delta", type=float, default=0.15)

    # Logging / verbosity
    p.add_argument(
        "--optuna_verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Optuna internal logging level",
    )

    p.add_argument("--print_trial_header", action="store_true", help="Print trial hyperparameters at the start of each trial")
    p.add_argument("--print_epoch_summary", action="store_true", help="Print per-epoch train/val summaries inside each trial")
    p.add_argument("--log_batch_progress", action="store_true", help="Print intra-epoch milestone batch logs (25/50/75%)")
    p.add_argument("--print_completed_trials", action="store_true", help="Print one line for each COMPLETE trial")

    args = p.parse_args()

    # Retrain metric defaults to tuning metric
    if args.retrain_metric is None:
        args.retrain_metric = args.metric

    # Retrain tag defaults to 'retrain_best_<timestamp>' if not provided
    if args.retrain_tag is None:
        args.retrain_tag = f"retrain_best_{int(time.time())}"

    return args


def _set_optuna_verbosity(level: str) -> None:
    mapping = {
        "debug": optuna.logging.DEBUG,
        "info": optuna.logging.INFO,
        "warning": optuna.logging.WARNING,
        "error": optuna.logging.ERROR,
        "critical": optuna.logging.CRITICAL,
    }
    optuna.logging.set_verbosity(mapping.get(level, optuna.logging.INFO))


def main() -> None:
    args = parse_args()
    _set_optuna_verbosity(getattr(args, "optuna_verbosity", "info"))
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if args.mode == "retrain_best":
        retrain_best(args)
        return
    if args.mode == "retrain_seed_worker":
        retrain_seed_worker(args)
        return

    # Tuning mode
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=5,
            interval_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=args.seed, constant_liar=True)
    study = create_study_safe(args, pruner=pruner, sampler=sampler)

    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    write_study_trials_tsv(study, os.path.join(args.out_dir, f"{args.study_name}_trials_summary.tsv"))

    best = study.best_trial
    print("\nBest trial so far:", flush=True)
    print(f"  number={best.number} value={best.value}", flush=True)
    print(f"  metric={args.metric}", flush=True)
    print("  params:", json.dumps(best.params, indent=2), flush=True)


if __name__ == "__main__":
    main()
