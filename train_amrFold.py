#!/usr/bin/env python3

import argparse
import os
from typing import Dict

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
)

from model import DBDataset, arg_collate_fn, AMRFoldModel


# -------------------------
# Evaluation
# -------------------------

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    lambda_class: float = 0.5,
) -> Dict[str, float]:
    """
    Run evaluation on a loader: compute losses + metrics (no attention regularisation).
    """
    model.eval()

    criterion_bin = nn.BCEWithLogitsLoss(reduction="sum")
    criterion_class = nn.CrossEntropyLoss(reduction="sum")

    total_bin_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0

    all_bin_labels = []
    all_bin_probs = []
    all_bin_preds = []

    all_class_labels = []
    all_class_preds = []

    with torch.no_grad():
        for batch in data_loader:
            plm = batch["plm"].to(device)
            di = batch["di"].to(device)
            conf = batch["conf"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            bin_labels = batch["bin_labels"].to(device)
            class_labels = batch["class_labels"].to(device)

            out = model(plm, di, conf, attn_mask, return_attn=False)
            logits_bin = out["logits_bin"]         # (B,)
            logits_class = out["logits_class"]     # (B, n_classes)

            B = bin_labels.size(0)
            total_samples += B

            bin_loss = criterion_bin(logits_bin, bin_labels)
            class_loss = criterion_class(logits_class, class_labels)

            total_bin_loss += bin_loss.item()
            total_class_loss += class_loss.item()

            probs_bin = torch.sigmoid(logits_bin)
            preds_bin = (probs_bin >= 0.5).long()

            all_bin_labels.append(bin_labels.cpu().numpy())
            all_bin_probs.append(probs_bin.cpu().numpy())
            all_bin_preds.append(preds_bin.cpu().numpy())

            preds_class = logits_class.argmax(dim=1)
            all_class_labels.append(class_labels.cpu().numpy())
            all_class_preds.append(preds_class.cpu().numpy())

    all_bin_labels = np.concatenate(all_bin_labels, axis=0)
    all_bin_probs = np.concatenate(all_bin_probs, axis=0)
    all_bin_preds = np.concatenate(all_bin_preds, axis=0)

    all_class_labels = np.concatenate(all_class_labels, axis=0)
    all_class_preds = np.concatenate(all_class_preds, axis=0)

    avg_bin_loss = total_bin_loss / total_samples
    avg_class_loss = total_class_loss / total_samples
    avg_total_loss = avg_bin_loss + lambda_class * avg_class_loss

    try:
        auc_roc = roc_auc_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = average_precision_score(all_bin_labels, all_bin_probs)
    except ValueError:
        auc_pr = float("nan")

    acc_bin = accuracy_score(all_bin_labels, all_bin_preds)
    f1_bin = f1_score(all_bin_labels, all_bin_preds)

    f1_class_macro = f1_score(all_class_labels, all_class_preds, average="macro")

    metrics = {
        "loss_total": avg_total_loss,
        "loss_bin": avg_bin_loss,
        "loss_class": avg_class_loss,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "acc_bin": acc_bin,
        "f1_bin": f1_bin,
        "f1_class_macro": f1_class_macro,
    }
    return metrics


# -------------------------
# Training one epoch
# -------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_class: float,
    lambda_ent: float,
    lambda_cont: float,
) -> Dict[str, float]:
    """
    One training epoch with attention regularisation.
    """
    model.train()

    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_bin_loss = 0.0
    total_class_loss = 0.0
    total_attn_loss = 0.0
    total_samples = 0

    eps = 1e-8

    for batch in data_loader:
        plm = batch["plm"].to(device)
        di = batch["di"].to(device)
        conf = batch["conf"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        bin_labels = batch["bin_labels"].to(device)
        class_labels = batch["class_labels"].to(device)

        optimizer.zero_grad()

        out = model(plm, di, conf, attn_mask, return_attn=True)
        logits_bin = out["logits_bin"]        # (B,)
        logits_class = out["logits_class"]    # (B, n_classes)
        attn_cls = out["attn_cls"]            # (B, L)

        loss_bin = criterion_bin(logits_bin, bin_labels)
        loss_class = criterion_class(logits_class, class_labels)

        # --- Attention regularisation ---
        # attn_cls currently has weights (may not sum to 1 over residues);
        # restrict to real tokens and renormalise.
        mask = attn_mask.float()              # (B, L)
        alpha = attn_cls * mask               # zero out pads
        alpha_sum = alpha.sum(dim=1, keepdim=True) + eps
        alpha = alpha / alpha_sum             # (B, L), per-sequence distribution

        # Entropy term (encourage low entropy -> focused attention)
        ent = -(alpha * (alpha + eps).log()).sum(dim=1).mean()
        loss_ent = lambda_ent * ent

        # Continuity term (encourage local smoothness)
        diff = alpha[:, 1:] - alpha[:, :-1]
        cont = (diff ** 2).sum(dim=1).mean()
        loss_cont = lambda_cont * cont

        loss_attn = loss_ent + loss_cont

        loss = loss_bin + lambda_class * loss_class + loss_attn

        B = bin_labels.size(0)
        total_samples += B

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_bin_loss += loss_bin.item() * B
        total_class_loss += loss_class.item() * B
        total_attn_loss += loss_attn.item() * B

    avg_loss = total_loss / total_samples
    avg_bin_loss = total_bin_loss / total_samples
    avg_class_loss = total_class_loss / total_samples
    avg_attn_loss = total_attn_loss / total_samples

    return {
        "loss_total": avg_loss,
        "loss_bin": avg_bin_loss,
        "loss_class": avg_class_loss,
        "loss_attn": avg_attn_loss,
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
        "--features_dir",
        required=True,
        help="Directory with *.plm.npy, *.3di.npy, *.conf.npy",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for checkpoints and logs",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lambda_class", type=float, default=0.5)
    p.add_argument("--lambda_ent", type=float, default=0.01)
    p.add_argument("--lambda_cont", type=float, default=0.01)
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------ Datasets & loaders ------------------
    print("Loading datasets...")

    train_ds = DBDataset(args.train_tsv, args.features_dir)
    val_ds = DBDataset(
        args.val_tsv,
        args.features_dir,
        class_to_idx=train_ds.class_to_idx,
    )
    test_ds = DBDataset(
        args.test_tsv,
        args.features_dir,
        class_to_idx=train_ds.class_to_idx,
    )

    n_classes = len(train_ds.class_to_idx)
    print(f"n_classes = {n_classes}")
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    collate = lambda b: arg_collate_fn(b, l_max=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
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
        verbose=True,
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
        )

        print(
            f"  Train: "
            f"loss={train_stats['loss_total']:.4f} "
            f"(bin={train_stats['loss_bin']:.4f}, "
            f"class={train_stats['loss_class']:.4f}, "
            f"attn={train_stats['loss_attn']:.4f})"
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            lambda_class=args.lambda_class,
        )

        # Step scheduler with validation AUROC
        scheduler.step(val_metrics["auc_roc"])

        print(
            f"  Val:   "
            f"loss={val_metrics['loss_total']:.4f} "
            f"(bin={val_metrics['loss_bin']:.4f}, "
            f"class={val_metrics['loss_class']:.4f}) "
            f"AUROC={val_metrics['auc_roc']:.4f} "
            f"AUPRC={val_metrics['auc_pr']:.4f} "
            f"F1_bin={val_metrics['f1_bin']:.4f} "
            f"F1_class_macro={val_metrics['f1_class_macro']:.4f}"
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
                    "class_to_idx": train_ds.class_to_idx,
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

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        lambda_class=args.lambda_class,
    )

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

