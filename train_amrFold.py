#!/usr/bin/env python3

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)

from model import DBDataset, DBDatasetLMDB, arg_collate_fn, AMRFoldModel


# -------------------------
# Evaluation
# -------------------------

def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray):
    """Return (best_thr, best_f1). Uses PR curve; robust to edge cases."""
    # precision_recall_curve returns precision/recall for thresholds in ascending order
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = (2.0 * prec * rec) / (prec + rec + 1e-12)

    best_i = int(np.nanargmax(f1))
    # thr has length len(prec)-1; last prec/rec point corresponds to thr=1.0 (no threshold)
    if best_i >= len(thr):
        best_thr = 0.5
    else:
        best_thr = float(thr[best_i])

    return best_thr, float(f1[best_i])


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    lambda_class: float = 0.5,
    threshold: float = 0.5,
    tune_threshold: bool = False,
    criterion_bin: Optional[nn.Module] = None,
    criterion_class: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Run evaluation on a loader: compute losses + metrics (no attention regularisation).

    Notes:
      - Class loss is computed ONLY on positives (bin==1), to match training (Option A1).
      - If tune_threshold=True, selects a threshold on this split that maximizes F1_bin.
        (Use this on VAL only, then reuse the resulting threshold on TEST.)
    """
    model.eval()

    # Default criteria (unweighted) if not provided
    if criterion_bin is None:
        criterion_bin = nn.BCEWithLogitsLoss(reduction="sum")
    if criterion_class is None:
        criterion_class = nn.CrossEntropyLoss(reduction="sum")

    total_bin_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0
    total_pos = 0

    all_bin_labels = []
    all_bin_probs = []

    all_class_labels = []
    all_class_preds = []
    all_class_labels_pos = []
    all_class_preds_pos = []

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

            # Binary loss across all samples
            bin_loss = criterion_bin(logits_bin, bin_labels)
            total_bin_loss += float(bin_loss.item())

            # Class loss on positives only (Option A1)
            pos_mask = (bin_labels > 0.5)
            if pos_mask.any():
                class_loss = criterion_class(logits_class[pos_mask], class_labels[pos_mask])
                total_class_loss += float(class_loss.item())
                total_pos += int(pos_mask.sum().item())

                preds_class_pos = logits_class[pos_mask].argmax(dim=1)
                all_class_labels_pos.append(class_labels[pos_mask].cpu().numpy())
                all_class_preds_pos.append(preds_class_pos.cpu().numpy())

            # Probabilities for metrics
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

    if len(all_class_labels_pos) > 0:
        all_class_labels_pos = np.concatenate(all_class_labels_pos, axis=0)
        all_class_preds_pos = np.concatenate(all_class_preds_pos, axis=0)
    else:
        all_class_labels_pos = np.array([], dtype=np.int64)
        all_class_preds_pos = np.array([], dtype=np.int64)

    # Losses
    avg_bin_loss = total_bin_loss / max(total_samples, 1)
    # class loss averaged over positives (not total samples)
    avg_class_loss = (total_class_loss / total_pos) if total_pos > 0 else float("nan")
    avg_total_loss = avg_bin_loss + lambda_class * (avg_class_loss if total_pos > 0 else 0.0)

    # Ranking metrics
    try:
        auc_roc = roc_auc_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = average_precision_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_pr = float("nan")

    # Threshold selection / application
    best_thr = float("nan")
    f1_bin_best = float("nan")
    if tune_threshold:
        best_thr, f1_bin_best = _best_f1_threshold(all_bin_labels, all_bin_probs)
        thr_to_use = best_thr
    else:
        thr_to_use = float(threshold)

    # Always compute metrics at the default 0.5 threshold for diagnostics
    all_bin_preds_0p5 = (all_bin_probs >= 0.5).astype(np.int64)
    acc_bin_0p5 = accuracy_score(all_bin_labels, all_bin_preds_0p5)
    f1_bin_0p5 = f1_score(all_bin_labels, all_bin_preds_0p5)

    all_bin_preds = (all_bin_probs >= thr_to_use).astype(np.int64)
    acc_bin = accuracy_score(all_bin_labels, all_bin_preds)
    f1_bin = f1_score(all_bin_labels, all_bin_preds)

    # Class F1
    f1_class_macro_all = f1_score(all_class_labels, all_class_preds, average="macro")

    if all_class_labels_pos.size > 0:
        f1_class_macro_pos = f1_score(all_class_labels_pos, all_class_preds_pos, average="macro")
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

    return metrics



# -------------------------
# Training one epoch
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
    criterion_bin: Optional[nn.Module] = None,
    criterion_class: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    One training epoch with attention regularisation.

    Changes vs the original:
      - Class loss is computed ONLY on positives (bin==1).
      - BCE and CE can be weighted; pass criterion_bin / criterion_class from main to avoid per-epoch rebuild.
    """
    model.train()

    if criterion_bin is None:
        criterion_bin = nn.BCEWithLogitsLoss()
    if criterion_class is None:
        criterion_class = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_bin_loss = 0.0
    total_class_loss = 0.0
    total_attn_loss = 0.0
    total_samples = 0
    total_pos = 0

    eps = 1e-8

    for batch in data_loader:
        plm = batch["plm"].to(device, non_blocking=True)
        di = batch["di"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        bin_labels = batch["bin_labels"].to(device, non_blocking=True)
        class_labels = batch["class_labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(plm, di, conf, attn_mask, return_attn=True)
        logits_bin = out["logits_bin"]        # (B,)
        logits_class = out["logits_class"]    # (B, n_classes)
        attn_cls = out["attn_cls"]            # (B, L)

        # Binary loss (all samples)
        loss_bin = criterion_bin(logits_bin, bin_labels)

        # Class loss (positives only)
        pos_mask = (bin_labels > 0.5)
        if pos_mask.any():
            loss_class = criterion_class(logits_class[pos_mask], class_labels[pos_mask])
            total_pos += int(pos_mask.sum().item())
        else:
            loss_class = torch.tensor(0.0, device=device)

        # Attention regularisation
        mask = attn_mask.float()              # (B, L)
        alpha = attn_cls * mask               # zero out pads
        alpha_sum = alpha.sum(dim=1, keepdim=True) + eps
        alpha = alpha / alpha_sum             # (B, L), per-sequence distribution

        # Entropy term (encourage focused attention)
        ent = -(alpha * (alpha + eps).log()).sum(dim=1).mean()
        loss_ent = lambda_ent * ent

        # Continuity term (encourage local smoothness)
        diff = alpha[:, 1:] - alpha[:, :-1]
        # Only penalize diffs where both positions are real tokens (avoids real→pad boundary)
        diff_mask = mask[:, 1:] * mask[:, :-1]             # (B, L-1), 1 for valid adjacent pairs
        # Length-normalize per sequence (avoids longer sequences getting larger penalties)
        cont_per_seq = ((diff ** 2) * diff_mask).sum(dim=1) / (diff_mask.sum(dim=1) + eps)  # (B,)
        cont = cont_per_seq.mean()
        loss_cont = lambda_cont * cont
        
        # Total
        loss = loss_bin + lambda_class * loss_class + loss_attn

        B = bin_labels.size(0)
        total_samples += B

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping for stability
        optimizer.step()

        total_loss += float(loss.item()) * B
        total_bin_loss += float(loss_bin.item()) * B
        if pos_mask.any():
            total_class_loss += float(loss_class.item()) * float(pos_mask.sum().item())
        
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



# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train AMR-Fold model with attention regularisation"
    )

    p.add_argument("--train_tsv", required=True, help="Path to DB_train.tsv")
    p.add_argument("--val_tsv", required=True, help="Path to DB_val.tsv")
    p.add_argument("--test_tsv", required=True, help="Path to DB_test.tsv")

    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for checkpoints and logs",
    )

    p.add_argument(
        "--features_dir", 
        default=None,
        help="Directory with *.plm.npy, *.3di.npy, *.conf.npy",
    )

    p.add_argument(
        "--features_lmdb",
        default=None,
        help="Path to LMDB directory (contains data.mdb/lock.mdb)")

  

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lambda_class", type=float, default=0.5)
    p.add_argument("--lambda_ent", type=float, default=0.001)
    p.add_argument("--lambda_cont", type=float, default=0.001)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience in epochs (0 = no early stopping)",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum AUROC improvement to reset patience",
    )

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Validate features input: exactly one of dir or lmdb
    if (args.features_dir is None) == (args.features_lmdb is None):
        raise ValueError("Provide exactly one of --features_dir or --features_lmdb")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------ Datasets & loaders ------------------
    print("Loading datasets...")

    # Collect all classes from train, val, test TSVs
    import csv
    all_classes = set()
    for tsv_path in [args.train_tsv, args.val_tsv, args.test_tsv]:
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                all_classes.add(row["class"])
    
    classes = sorted(all_classes)
    # Force non_ARG to index 0 if present
    if "non_ARG" in classes:
        classes = ["non_ARG"] + [c for c in classes if c != "non_ARG"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if args.features_lmdb:
        train_ds = DBDatasetLMDB(args.train_tsv, args.features_lmdb, class_to_idx=class_to_idx)
        val_ds   = DBDatasetLMDB(args.val_tsv,   args.features_lmdb, class_to_idx=class_to_idx)
        test_ds  = DBDatasetLMDB(args.test_tsv,  args.features_lmdb, class_to_idx=class_to_idx)
    else:
        train_ds = DBDataset(args.train_tsv, args.features_dir, class_to_idx=class_to_idx)
        val_ds   = DBDataset(args.val_tsv,   args.features_dir, class_to_idx=class_to_idx)
        test_ds  = DBDataset(args.test_tsv,  args.features_dir, class_to_idx=class_to_idx)


    n_classes = len(class_to_idx)
    print(f"n_classes = {n_classes}")
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ------------------ Loss weighting (computed once; negligible overhead) ------------------
    # Binary: BCE pos_weight = N_neg / N_pos (critical if imbalanced; ~1.0 if balanced).
    train_bins = [int(r["bin"]) for r in train_ds.records]
    n_train = len(train_bins)
    n_pos = int(sum(train_bins))
    n_neg = int(n_train - n_pos)
    if n_pos > 0:
        bce_pos_weight = float(n_neg / n_pos)
    else:
        bce_pos_weight = 1.0

    # Class weights: inverse-frequency among POSITIVES only (since we train CE only on positives).
    from collections import Counter
    pos_class_counts = Counter()
    for r in train_ds.records:
        if int(r["bin"]) == 1:
            pos_class_counts[r["class"]] += 1

    class_weights = np.zeros(n_classes, dtype=np.float32)
    for cls, idx_ in class_to_idx.items():
        if cls == "non_ARG":
            class_weights[idx_] = 0.0  # never used if Option A1 is enabled
        else:
            c = pos_class_counts.get(cls, 0)
            class_weights[idx_] = (1.0 / c) if c > 0 else 0.0

    # Normalize non-zero weights to mean 1 for stability
    nz = class_weights > 0
    if nz.any():
        class_weights[nz] = class_weights[nz] / class_weights[nz].mean()

    print(f"Train label stats: n={n_train} pos={n_pos} neg={n_neg} BCE pos_weight={bce_pos_weight:.4f}")
    # Show a few positive-class counts to confirm distribution
    if len(pos_class_counts) > 0:
        print("Top positive classes:", pos_class_counts.most_common(5))

    bce_pos_weight_t = torch.tensor([bce_pos_weight], device=device)
    class_weights_t = torch.tensor(class_weights, device=device)

    # Reuse loss objects across epochs (no per-epoch/per-batch rebuild)
    criterion_bin_train = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t)
    criterion_bin_eval = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_t, reduction="sum")

    criterion_class_train = nn.CrossEntropyLoss(weight=class_weights_t)
    criterion_class_eval = nn.CrossEntropyLoss(weight=class_weights_t, reduction="sum")


    collate = lambda b: arg_collate_fn(b, l_max=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
	    persistent_workers=True,
    	prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ------------------ Model, optimizer ------------------
    print("Building model...")

    model = AMRFoldModel(
        n_classes=n_classes,
        d_plm=1024,
        d_model=512,
        d_struct_tok=64,
        n_3di_tokens=20,
        n_layers=3,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=args.max_len,
        pad_token_id=20,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",      # we want to maximize AUROC
        factor=0.5,      # halve LR
        patience=3,      # after 3 epochs with no AUROC improvement
    )

    best_val_auc = -1.0
    best_ckpt_path = os.path.join(args.out_dir, "amrfold_best.pt")
    epochs_no_improve = 0

    # ------------------ Training loop ------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_class=args.lambda_class,
            lambda_ent=args.lambda_ent,
            lambda_cont=args.lambda_cont,
            criterion_bin=criterion_bin_train,
            criterion_class=criterion_class_train,
        )

        print(
            f"  Train: "
            f"loss={train_stats['loss_total']:.4f} "
            f"(bin={train_stats['loss_bin']:.4f}, "
            f"class_pos={train_stats['loss_class_pos']:.4f}, "
            f"attn={train_stats['loss_attn']:.4f})"
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            lambda_class=args.lambda_class,
            tune_threshold=True,
            criterion_bin=criterion_bin_eval,
            criterion_class=criterion_class_eval,
        )

        # Step scheduler with validation AUROC
        scheduler.step(val_metrics["auc_roc"])

        # Print current LR
        current_lr = scheduler.get_last_lr()[0]
        print(f"  LR={current_lr}")

        print(
            f"  Val:   "
            f"loss={val_metrics['loss_total']:.4f} "
            f"(bin={val_metrics['loss_bin']:.4f}, "
            f"class_pos={val_metrics['loss_class_pos']:.4f}) "
            f"AUROC={val_metrics['auc_roc']:.4f} "
            f"AUPRC={val_metrics['auc_pr']:.4f} "
            f"F1_bin@0.5={val_metrics['f1_bin_at_0p5']:.4f} "
            f"F1_bin@thr={val_metrics['f1_bin']:.4f} "
            f"thr={val_metrics['threshold_used']:.4f} "
            f"pred_pos_rate={val_metrics['pred_pos_rate']:.4f} "
            f"F1_class_macro_pos={val_metrics['f1_class_macro_pos']:.4f}"
        )

        # Checkpoint on best AUROC
        improvement = val_metrics["auc_roc"] - best_val_auc

        if improvement > args.min_delta:
            best_val_auc = val_metrics["auc_roc"]
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved (AUROC={best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No AUROC improvement for {epochs_no_improve} epoch(s).")

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered (patience={args.patience}).")
                break

    # ------------------ Final test evaluation ------------------
    print("\nLoading best checkpoint and evaluating on TEST set...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()


    # 1) Select best threshold on VAL using the best checkpoint (do not use TEST for tuning).
    val_metrics_best = evaluate(
        model,
        val_loader,
        device,
        lambda_class=args.lambda_class,
        tune_threshold=True,
        criterion_bin=criterion_bin_eval,
        criterion_class=criterion_class_eval,
    )
    best_thr = float(val_metrics_best.get("threshold_best_f1", val_metrics_best["threshold_used"]))
    print(f"Best VAL threshold for F1_bin: {best_thr:.4f} (VAL F1={val_metrics_best['f1_bin']:.4f})")

    # 2) Evaluate TEST using that fixed threshold.
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        lambda_class=args.lambda_class,
        threshold=best_thr,
        tune_threshold=False,
        criterion_bin=criterion_bin_eval,
        criterion_class=criterion_class_eval,
    )

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

