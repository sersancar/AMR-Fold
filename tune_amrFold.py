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
  - Class CE on positives only (weighted)
  - Attention regularisation (entropy + continuity)
  - ReduceLROnPlateau + early stopping
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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

# Fixed label-space contract (15 ARG classes + non_ARG)
NON_ARG_LABEL = "non_ARG"
EXPECTED_N_CLASSES = 16  # 15 ARG classes + non_ARG
EXPECTED_N_ARG_CLASSES = 15



# -------------------------
# Utils
# -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path



def build_fixed_class_mapping(train_tsv: str, val_tsv: str, test_tsv: str) -> Dict[str, int]:
    """Build a fixed, validated class_to_idx mapping (15 ARG classes + non_ARG)."""
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
) -> DataBundle:
    """
    Build datasets + loaders, and compute loss weights from TRAIN only.

    Uses LMDB features if features_lmdb is provided; otherwise uses feature_dir.
    """
    # Build a validated, fixed class mapping (15 ARG + non_ARG)
    class_to_idx = build_fixed_class_mapping(train_tsv, val_tsv, test_tsv)

    if features_lmdb:
        train_ds = DBDatasetLMDB(train_tsv, features_lmdb, class_to_idx=class_to_idx)
        val_ds = DBDatasetLMDB(val_tsv, features_lmdb, class_to_idx=class_to_idx)
        test_ds = DBDatasetLMDB(test_tsv, features_lmdb, class_to_idx=class_to_idx)
    else:
        if not feature_dir:
            raise ValueError("Provide either --features_lmdb or --feature_dir.")
        train_ds = DBDataset(train_tsv, feature_dir, class_to_idx=None)
        class_to_idx = train_ds.class_to_idx
        val_ds = DBDataset(val_tsv, feature_dir, class_to_idx=class_to_idx)
        test_ds = DBDataset(test_tsv, feature_dir, class_to_idx=class_to_idx)

    n_classes = len(class_to_idx)

    # ------------------ Loss weighting (computed once; negligible overhead) ------------------
    train_bins = [int(r["bin"]) for r in train_ds.records]
    n_train = len(train_bins)
    n_pos = int(sum(train_bins))
    n_neg = int(n_train - n_pos)
    bce_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # Class weights: inverse-frequency among POSITIVES only (since CE only on positives)
    from collections import Counter
    pos_class_counts = Counter()
    for r in train_ds.records:
        if int(r["bin"]) == 1:
            pos_class_counts[r["class"]] += 1

    class_weights = np.zeros(n_classes, dtype=np.float32)
    for cls, idx in class_to_idx.items():
        if cls == "non_ARG":
            class_weights[idx] = 0.0
        else:
            c = pos_class_counts.get(cls, 0)
            class_weights[idx] = (1.0 / c) if c > 0 else 0.0

    nz = class_weights > 0
    if nz.any():
        class_weights[nz] = class_weights[nz] / class_weights[nz].mean()

    collate = lambda b: arg_collate_fn(b, l_max=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
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
    trial_number: Optional[int] = None,
    epoch_number: Optional[int] = None,
    log_batch_progress: bool = False,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bin_loss = 0.0
    total_class_loss = 0.0
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
            loss_class = criterion_class(logits_class[pos_mask], class_labels[pos_mask])
            total_pos += n_pos_batch
        else:
            loss_class = torch.tensor(0.0, device=device)

        batches_seen += 1
        samples_seen += B
        if log_batch_progress and (batches_seen in milestone_batches):
            lr_now = float(optimizer.param_groups[0]["lr"])
            avg_loss_so_far = total_loss / max(samples_seen, 1)
            avg_bin_so_far = total_bin_loss / max(samples_seen, 1)
            avg_class_pos_so_far = (total_class_loss / total_pos) if total_pos > 0 else float("nan")

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
        if n_pos_batch > 0:
            total_class_loss += float(loss_class.item()) * float(n_pos_batch)
        total_attn_loss += float(loss_attn.item()) * B

    avg_loss = total_loss / max(total_samples, 1)
    avg_bin_loss = total_bin_loss / max(total_samples, 1)
    avg_class_loss_pos = (total_class_loss / total_pos) if total_pos > 0 else float("nan")
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
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate and return:
      - metrics dict (bin + class macro_pos)
      - outputs dict (arrays) for per-class reporting

    Class metrics of interest for your stated end-goal:
      - f1_class_macro_pos computed only on true positives (bin==1).
    """
    model.eval()

    total_bin_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0
    total_pos = 0

    all_bin_labels = []
    all_bin_probs = []

    all_class_labels = []
    all_class_preds = []

    pos_class_labels = []
    pos_class_preds = []
    pos_class_probs = []  # softmax on positives only

    with torch.no_grad():
        for batch in data_loader:
            plm = batch["plm"].to(device, non_blocking=True)
            di = batch["di"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            bin_labels = batch["bin_labels"].to(device, non_blocking=True)
            class_labels = batch["class_labels"].to(device, non_blocking=True)

            out = model(plm, di, conf, attn_mask, return_attn=False)
            logits_bin = out["logits_bin"]         # (B,)
            logits_class = out["logits_class"]     # (B, n_classes)

            B = bin_labels.size(0)
            total_samples += B

            # Losses
            bin_loss = criterion_bin(logits_bin, bin_labels)
            total_bin_loss += float(bin_loss.item())

            pos_mask = (bin_labels > 0.5)
            if pos_mask.any():
                class_loss = criterion_class(logits_class[pos_mask], class_labels[pos_mask])
                total_class_loss += float(class_loss.item())
                total_pos += int(pos_mask.sum().item())

                preds_class_pos = logits_class[pos_mask].argmax(dim=1)
                probs_class_pos = torch.softmax(logits_class[pos_mask], dim=1)

                pos_class_labels.append(class_labels[pos_mask].cpu().numpy())
                pos_class_preds.append(preds_class_pos.cpu().numpy())
                pos_class_probs.append(probs_class_pos.cpu().numpy())

            probs_bin = torch.sigmoid(logits_bin)
            all_bin_labels.append(bin_labels.cpu().numpy())
            all_bin_probs.append(probs_bin.cpu().numpy())

            preds_class = logits_class.argmax(dim=1)
            all_class_labels.append(class_labels.cpu().numpy())
            all_class_preds.append(preds_class.cpu().numpy())

    all_bin_labels = np.concatenate(all_bin_labels, axis=0)
    all_bin_probs = np.concatenate(all_bin_probs, axis=0)
    all_class_labels = np.concatenate(all_class_labels, axis=0)
    all_class_preds = np.concatenate(all_class_preds, axis=0)

    if len(pos_class_labels) > 0:
        pos_class_labels = np.concatenate(pos_class_labels, axis=0)
        pos_class_preds = np.concatenate(pos_class_preds, axis=0)
        pos_class_probs = np.concatenate(pos_class_probs, axis=0)
    else:
        pos_class_labels = np.array([], dtype=np.int64)
        pos_class_preds = np.array([], dtype=np.int64)
        pos_class_probs = np.zeros((0, 1), dtype=np.float32)

    # Loss averages
    avg_bin_loss = total_bin_loss / max(total_samples, 1)
    avg_class_loss = (total_class_loss / total_pos) if total_pos > 0 else float("nan")
    avg_total_loss = avg_bin_loss + lambda_class * (avg_class_loss if total_pos > 0 else 0.0)

    # Ranking metrics (binary)
    try:
        auc_roc = roc_auc_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = average_precision_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_pr = float("nan")

    # Threshold selection / application (binary)
    best_thr = float("nan")
    f1_bin_best = float("nan")
    if tune_threshold:
        best_thr, f1_bin_best = _best_f1_threshold(all_bin_labels, all_bin_probs)
        thr_to_use = best_thr
    else:
        thr_to_use = float(threshold)

    all_bin_preds_0p5 = (all_bin_probs >= 0.5).astype(np.int64)
    all_bin_preds = (all_bin_probs >= thr_to_use).astype(np.int64)

    acc_bin_0p5 = accuracy_score(all_bin_labels, all_bin_preds_0p5)
    f1_bin_0p5 = f1_score(all_bin_labels, all_bin_preds_0p5, zero_division=0)

    acc_bin = accuracy_score(all_bin_labels, all_bin_preds)
    f1_bin = f1_score(all_bin_labels, all_bin_preds, zero_division=0)
    prec_bin = precision_score(all_bin_labels, all_bin_preds, zero_division=0)
    rec_bin = recall_score(all_bin_labels, all_bin_preds, zero_division=0)

    # Class metrics
    f1_class_macro_all = f1_score(all_class_labels, all_class_preds, average="macro", zero_division=0)
    if pos_class_labels.size > 0:
        f1_class_macro_pos = f1_score(pos_class_labels, pos_class_preds, average="macro", zero_division=0)
    else:
        f1_class_macro_pos = float("nan")

    metrics = {
        "loss_total": float(avg_total_loss),
        "loss_bin": float(avg_bin_loss),
        "loss_class_pos": float(avg_class_loss),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "acc_bin": float(acc_bin),
        "f1_bin": float(f1_bin),
        "precision_bin": float(prec_bin),
        "recall_bin": float(rec_bin),
        "acc_bin_at_0p5": float(acc_bin_0p5),
        "f1_bin_at_0p5": float(f1_bin_0p5),
        "threshold_used": float(thr_to_use),
        "pred_pos_rate": float(all_bin_preds.mean()),
        "f1_class_macro_all": float(f1_class_macro_all),
        "f1_class_macro_pos": float(f1_class_macro_pos),
        "n_samples": float(total_samples),
        "n_pos": float(total_pos),
    }
    if tune_threshold:
        metrics["threshold_best_f1"] = float(best_thr)
        metrics["f1_bin_best"] = float(f1_bin_best)

    outputs = {
        "bin_labels": all_bin_labels.astype(np.int64),
        "bin_probs": all_bin_probs.astype(np.float32),
        "class_labels_all": all_class_labels.astype(np.int64),
        "class_preds_all": all_class_preds.astype(np.int64),
        "class_labels_pos": pos_class_labels.astype(np.int64),
        "class_preds_pos": pos_class_preds.astype(np.int64),
        "class_probs_pos": pos_class_probs.astype(np.float32),
    }
    return metrics, outputs


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
    ensure_dir(out_dir)

    # Add AUC columns per class into a tabular CSV
    # Extract per-class entries (only integer keys in the report dict are class labels as strings)
    rows = []
    for k, v in report.items():
        if k.isdigit():
            c = int(k)
            row = dict(v)
            row["class_index"] = c
            row["class_name"] = idx_to_class.get(c, str(c))
            row["roc_auc_ovr"] = float(roc_ovr[c]) if not np.isnan(roc_ovr[c]) else np.nan
            row["pr_auc_ovr"] = float(pr_ovr[c]) if not np.isnan(pr_ovr[c]) else np.nan
            rows.append(row)

    # Write JSON report (full)
    with open(os.path.join(out_dir, f"{prefix}_{split_name}_class_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Write CSV (per-class only)
    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["support", "f1-score"], ascending=[False, False])
        df.to_csv(os.path.join(out_dir, f"{prefix}_{split_name}_class_report.csv"), index=False)
    except Exception:
        # Fallback: minimal TSV
        tsv_path = os.path.join(out_dir, f"{prefix}_{split_name}_class_report.tsv")
        with open(tsv_path, "w") as f:
            f.write("class_index\tclass_name\tprecision\trecall\tf1\tsupport\troc_auc_ovr\tpr_auc_ovr\n")
            for r in rows:
                f.write(
                    f"{r['class_index']}\t{r['class_name']}\t{r.get('precision',0):.6f}\t"
                    f"{r.get('recall',0):.6f}\t{r.get('f1-score',0):.6f}\t{int(r.get('support',0))}\t"
                    f"{r.get('roc_auc_ovr',np.nan)}\t{r.get('pr_auc_ovr',np.nan)}\n"
                )

    # Save confusion matrix
    np.save(os.path.join(out_dir, f"{prefix}_{split_name}_confusion.npy"), cm)
    # Also write a labeled CSV for convenience
    try:
        import pandas as pd
        labels = [idx_to_class.get(i, str(i)) for i in range(cm.shape[0])]
        dfcm = pd.DataFrame(cm, index=labels, columns=labels)
        dfcm.to_csv(os.path.join(out_dir, f"{prefix}_{split_name}_confusion.csv"))
    except Exception:
        pass


# -------------------------
# Optuna objective + retrain
# -------------------------

def build_model(n_classes: int, *, dropout: float, max_len: int, device: torch.device) -> AMRFoldModel:
    model = AMRFoldModel(n_classes=n_classes, dropout=float(dropout), max_len=int(max_len))
    return model.to(device)



# -------------------------
# Optuna storage helpers (robust multi-process SQLite)
# -------------------------

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

    lock_dir = os.path.join(args.out_dir, f".optuna_init_lock_{args.study_name}")
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
    def objective(trial: "optuna.Trial") -> float:
        # ---- Search space (keep high-ROI, end-goal aligned) ----
        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.30)
        lambda_class = trial.suggest_float("lambda_class", 0.1, 1.0)

        lambda_ent = trial.suggest_float("lambda_ent", 1e-6, 1e-2, log=True)
        lambda_cont = trial.suggest_float("lambda_cont", 1e-6, 1e-2, log=True)

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        max_len = trial.suggest_categorical("max_len", [512, 1024])

        lr_factor = trial.suggest_categorical("lr_factor", [0.3, 0.5, 0.7])
        lr_patience = trial.suggest_int("lr_patience", 2, 5)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        if args.print_trial_header:
            print(
                f"\n=== Trial {trial.number} start (CUDA_VISIBLE_DEVICES={visible}) ===\n"
                f"  lr={lr:.2e} wd={weight_decay:.2e} dropout={dropout:.3f}\n"
                f"  lambda_class={lambda_class:.3f} lambda_ent={lambda_ent:.2e} lambda_cont={lambda_cont:.2e}\n"
                f"  batch_size={batch_size} max_len={max_len} lr_factor={lr_factor} lr_patience={lr_patience}",
                flush=True,
            )

        # ---- Build data (depends on batch_size / max_len) ----
        data = build_data(
            args.train_tsv, args.val_tsv, args.test_tsv,
            feature_dir=args.feature_dir,
            features_lmdb=args.features_lmdb,
            max_len=int(max_len),
            batch_size=int(batch_size),
            num_workers=args.num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(data.n_classes, dropout=dropout, max_len=int(max_len), device=device)

        # Losses (match train_amrFold.py)
        bce_pos_weight_t = torch.tensor([data.bce_pos_weight], device=device)
        class_weights_t = torch.tensor(data.class_weights, device=device)

        criterion_bin_train = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t)
        criterion_bin_eval = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t, reduction="sum")

        criterion_class_train = nn.CrossEntropyLoss(weight=class_weights_t)
        criterion_class_eval = nn.CrossEntropyLoss(weight=class_weights_t, reduction="sum")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(lr_factor),
            patience=int(lr_patience),
        )

        best_score = -1.0
        epochs_no_improve = 0

        # ---- Train / Val loop (with pruning) ----
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
                )

                score = float(val_metrics[args.metric])

                # Step scheduler on the *optimisation metric* (end-goal aligned)
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
            # Handle CUDA OOM by pruning the trial instead of killing the worker
            if "out of memory" in str(e).lower():
                raise optuna.exceptions.TrialPruned()
            raise
        finally:
            # Free memory aggressively between trials
            del model
            del optimizer
            del scheduler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return float(best_score)

    return objective


def retrain_best(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = study.best_trial

    # Output directory
    tag = args.retrain_tag or f"retrain_best_trial{best.number}"
    out_dir = ensure_dir(os.path.join(args.out_dir, tag))

    # Persist best params
    with open(os.path.join(out_dir, "best_trial.json"), "w") as f:
        json.dump(
            {
                "study_name": args.study_name,
                "best_trial_number": best.number,
                "best_value": best.value,
                "metric": args.metric,
                "params": best.params,
            },
            f,
            indent=2,
        )

    # Apply best params
    hp = best.params
    lr = float(hp.get("lr", args.lr_default))
    weight_decay = float(hp.get("weight_decay", args.weight_decay_default))
    dropout = float(hp.get("dropout", args.dropout_default))
    lambda_class = float(hp.get("lambda_class", args.lambda_class_default))
    lambda_ent = float(hp.get("lambda_ent", args.lambda_ent_default))
    lambda_cont = float(hp.get("lambda_cont", args.lambda_cont_default))
    batch_size = int(hp.get("batch_size", args.batch_size_default))
    max_len = int(hp.get("max_len", args.max_len_default))
    lr_factor = float(hp.get("lr_factor", args.lr_factor_default))
    lr_patience = int(hp.get("lr_patience", args.lr_patience_default))

    # Data + model
    data = build_data(
        args.train_tsv, args.val_tsv, args.test_tsv,
        feature_dir=args.feature_dir,
        features_lmdb=args.features_lmdb,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(data.n_classes, dropout=dropout, max_len=max_len, device=device)

    idx_to_class = {v: k for k, v in data.class_to_idx.items()}

    bce_pos_weight_t = torch.tensor([data.bce_pos_weight], device=device)
    class_weights_t = torch.tensor(data.class_weights, device=device)

    criterion_bin_train = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t)
    criterion_bin_eval = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t, reduction="sum")

    criterion_class_train = nn.CrossEntropyLoss(weight=class_weights_t)
    criterion_class_eval = nn.CrossEntropyLoss(weight=class_weights_t, reduction="sum")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_factor, patience=lr_patience)

    # Retrain loop with early stopping on args.retrain_metric (defaults to args.metric)
    metric_name = args.retrain_metric or args.metric
    best_score = -1.0
    epochs_no_improve = 0
    best_state = None

    history = []

    for epoch in range(1, args.retrain_epochs + 1):
        train_m = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            device,
            lambda_class=lambda_class,
            lambda_ent=lambda_ent,
            lambda_cont=lambda_cont,
            criterion_bin=criterion_bin_train,
            criterion_class=criterion_class_train,
            epoch_number=epoch,
            log_batch_progress=args.log_batch_progress,
        )

        val_m, _ = evaluate_with_outputs(
            model,
            data.val_loader,
            device,
            lambda_class=lambda_class,
            tune_threshold=False,
            criterion_bin=criterion_bin_eval,
            criterion_class=criterion_class_eval,
        )

        score = float(val_m.get(metric_name, float("nan")))
        scheduler.step(score)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        history.append(row)

        if score > best_score + args.min_delta:
            best_score = score
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "score": best_score,
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.retrain_patience:
            break

    # Save history
    try:
        import pandas as pd
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    except Exception:
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    if best_state is None:
        raise SystemExit("Retrain did not produce a valid checkpoint (unexpected).")

    # Load best checkpoint
    model.load_state_dict(best_state["model_state"], strict=True)

    # --- Evaluate VAL (tune threshold for bin for diagnostics), and TEST using that threshold ---
    val_metrics, val_out = evaluate_with_outputs(
        model,
        data.val_loader,
        device,
        lambda_class=lambda_class,
        tune_threshold=True,
        criterion_bin=criterion_bin_eval,
        criterion_class=criterion_class_eval,
    )
    best_thr = float(val_metrics.get("threshold_best_f1", 0.5))

    test_metrics, test_out = evaluate_with_outputs(
        model,
        data.test_loader,
        device,
        lambda_class=lambda_class,
        threshold=best_thr,
        tune_threshold=False,
        criterion_bin=criterion_bin_eval,
        criterion_class=criterion_class_eval,
    )

    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # --- Per-class reports (POS subset: aligns with your end-goal) ---
    for split_name, out in [("val", val_out), ("test", test_out)]:
        y_true = out["class_labels_pos"]
        y_pred = out["class_preds_pos"]
        y_prob = out["class_probs_pos"]
        report, cm, roc_ovr, pr_ovr = per_class_reports(y_true, y_pred, y_prob, n_classes=data.n_classes)
        save_reports(
            out_dir,
            split_name=split_name,
            report=report,
            cm=cm,
            roc_ovr=roc_ovr,
            pr_ovr=pr_ovr,
            idx_to_class=idx_to_class,
            prefix="pos",
        )

    # Save checkpoint + metadata
    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": data.class_to_idx,
        "idx_to_class": idx_to_class,
        "best_epoch": best_state["epoch"],
        "best_score": best_state["score"],
        "retrain_metric": metric_name,
        "val_best_thr_f1_bin": best_thr,
        "hparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "lambda_class": lambda_class,
            "lambda_ent": lambda_ent,
            "lambda_cont": lambda_cont,
            "batch_size": batch_size,
            "max_len": max_len,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
        },
    }
    torch.save(ckpt, os.path.join(out_dir, "best_checkpoint.pt"))

    # Console summary (short, but immediately useful)
    print("\nFinal retrain summary (end-goal aligned):", flush=True)
    print(f"  Optimised metric: {metric_name}", flush=True)
    print(f"  Best epoch: {best_state['epoch']}  best_score: {best_state['score']:.6f}", flush=True)
    print(f"  VAL:  f1_class_macro_pos={val_metrics.get('f1_class_macro_pos', float('nan')):.4f}  "
          f"f1_bin={val_metrics.get('f1_bin', float('nan')):.4f}  auc_roc={val_metrics.get('auc_roc', float('nan')):.4f}", flush=True)
    print(f"  TEST: f1_class_macro_pos={test_metrics.get('f1_class_macro_pos', float('nan')):.4f}  "
          f"f1_bin={test_metrics.get('f1_bin', float('nan')):.4f}  auc_roc={test_metrics.get('auc_roc', float('nan')):.4f}", flush=True)
    print(f"  Reports written to: {out_dir}", flush=True)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["tune", "retrain_best"], default="tune")

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

    # Retrain
    p.add_argument("--retrain_epochs", type=int, default=40)
    p.add_argument("--retrain_patience", type=int, default=8)
    p.add_argument("--retrain_metric", default=None, help="Metric for retrain early stopping; default=--metric")
    p.add_argument("--retrain_tag", default=None, help="Subdir name under out_dir for retrain outputs")

    # Defaults (used if a study has missing params)
    p.add_argument("--lr_default", type=float, default=1e-4)
    p.add_argument("--weight_decay_default", type=float, default=1e-4)
    p.add_argument("--dropout_default", type=float, default=0.1)
    p.add_argument("--lambda_class_default", type=float, default=0.5)
    p.add_argument("--lambda_ent_default", type=float, default=1e-3)
    p.add_argument("--lambda_cont_default", type=float, default=1e-3)
    p.add_argument("--batch_size_default", type=int, default=16)
    p.add_argument("--max_len_default", type=int, default=1024)
    p.add_argument("--lr_factor_default", type=float, default=0.5)
    p.add_argument("--lr_patience_default", type=int, default=3)


    # Logging / verbosity
    # By default, keep console output minimal (Optuna logs only). Enable these flags if needed.
    p.add_argument('--print_trial_header', action='store_true', help='Print trial hyperparameters at the start of each trial')
    p.add_argument('--print_epoch_summary', action='store_true', help='Print per-epoch train/val summaries inside each trial')
    p.add_argument('--log_batch_progress', action='store_true', help='Print intra-epoch milestone batch logs (25/50/75%)')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if args.mode == "retrain_best":
        retrain_best(args)
        return

    # Tuning mode
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = create_study_safe(args, pruner=pruner, sampler=sampler)

    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # Print best so far
    best = study.best_trial
    print("\nBest trial so far:", flush=True)
    print(f"  number={best.number} value={best.value}", flush=True)
    print(f"  metric={args.metric}", flush=True)
    print("  params:", json.dumps(best.params, indent=2), flush=True)


if __name__ == "__main__":
    main()
