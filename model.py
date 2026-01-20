#!/usr/bin/env python3

import os
import csv
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DBDataset(Dataset):
    """
    Dataset for AMR-Fold.

    Expects:
      - tsv_path: tab-separated file with columns: ids, Ids, class, bin
      - feature_dir: directory with *.npy feature files:
          {Seq_id}.plm.npy   -> (L, 1024)
          {Seq_id}.3di_tokens.npy   -> (L,)
          {Seq_id}.3di_conf.npy  -> (L,)
    """

    def __init__(
        self,
        tsv_path: str,
        feature_dir: str,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.tsv_path = tsv_path
        self.feature_dir = feature_dir

        # Read TSV
        self.records: List[Dict[str, str]] = []
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.records.append(row)

        # Build or use class_to_idx
        if class_to_idx is None:
            classes = sorted({r["class"] for r in self.records})
            # Optional: force non_ARG to index 0 if present
            if "non_ARG" in classes:
                classes = ["non_ARG"] + [c for c in classes if c != "non_ARG"]
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        # For convenience, expose mapping
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        seq_id = rec["Ids"]  # canonical Seq_xxx

        # Labels
        bin_label = int(rec["bin"])
        class_label = self.class_to_idx[rec["class"]]

        # Load features
        plm_path = os.path.join(self.feature_dir, f"{seq_id}.plm.npy")
        di_path = os.path.join(self.feature_dir, f"{seq_id}.3di_tokens.npy")
        conf_path = os.path.join(self.feature_dir, f"{seq_id}.3di_conf.npy")

        plm = np.load(plm_path)        # (L, 1024)
        di = np.load(di_path)          # (L,)
        conf = np.load(conf_path)      # (L,)

        # Convert to tensors
        plm_t = torch.from_numpy(plm).float()      # (L, 1024)
        di_t = torch.from_numpy(di).long()         # (L,)
        conf_t = torch.from_numpy(conf).float()    # (L,)

        return {
            "seq_id": seq_id,
            "plm": plm_t,
            "di": di_t,
            "conf": conf_t,
            "bin": torch.tensor(bin_label, dtype=torch.float32),
            "class": torch.tensor(class_label, dtype=torch.long),
        }

def arg_collate_fn(
    batch: List[Dict[str, Any]],
    l_max: int = 1024,
    pad_token_id: int = 20,  # 0..19 are real 3Di tokens, 20 is pad
) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length proteins.

    Returns dict with:
      plm:  (B, L, 1024)
      di:   (B, L)         int64
      conf: (B, L)
      attention_mask: (B, L)  1 for real, 0 for pad
      bin_labels: (B,)
      class_labels: (B,)
    """

    batch_size = len(batch)
    lengths = [min(x["plm"].shape[0], l_max) for x in batch]
    max_len = max(lengths)

    d_plm = batch[0]["plm"].shape[1]  # 1024

    plm_batch = torch.zeros(batch_size, max_len, d_plm, dtype=torch.float32)
    di_batch = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    conf_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    bin_labels = torch.zeros(batch_size, dtype=torch.float32)
    class_labels = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        L = min(sample["plm"].shape[0], max_len)

        plm_batch[i, :L] = sample["plm"][:L]
        di_batch[i, :L] = sample["di"][:L]
        conf_batch[i, :L] = sample["conf"][:L]
        attn_mask[i, :L] = True  # real tokens

        bin_labels[i] = sample["bin"]
        class_labels[i] = sample["class"]

    return {
        "plm": plm_batch,
        "di": di_batch,
        "conf": conf_batch,
        "attention_mask": attn_mask,
        "bin_labels": bin_labels,
        "class_labels": class_labels,
    }
    
class TransformerEncoderLayerWithAttn(nn.Module):
    """
    Single Transformer encoder layer using MultiheadAttention, returning attention weights.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(
        self,
        src: torch.Tensor,  # (B, S, d_model)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool, True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # get per-head weights
        )  # attn_weights: (B, n_heads, S, S)

        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)

        # Feed-forward
        src2 = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = self.norm2(src + src2)

        return src, attn_weights


class SimpleTransformerEncoder(nn.Module):
    """
    Stacked Transformer encoder with LayerNorm at the end.
    Returns output + last layer attention weights.
    """

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttn(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        src: (B, S, d_model)
        src_key_padding_mask: (B, S) bool, True=pad
        """
        attn_last = None
        for layer in self.layers:
            src, attn_last = layer(src, src_key_padding_mask=src_key_padding_mask)
        src = self.norm(src)
        if return_attn:
            return src, attn_last
        return src, None

class AMRFoldModel(nn.Module):
    """
    AMR-Fold architecture.

    Inputs:
      plm:  (B, L, 1024)
      di:   (B, L) int64, 3Di tokens in [0..19] or pad_token_id
      conf: (B, L)
      attention_mask: (B, L) bool, True=real, False=pad

    Outputs:
      logits_bin:   (B,)    1 logit per sequence
      logits_class: (B, n_classes)
      attn_cls:     (B, S-1) attention distribution over residues for [CLS] (averaged over heads)
    """

    def __init__(
        self,
        n_classes: int,
        d_plm: int = 1024,
        d_model: int = 512,
        d_struct_tok: int = 64,
        n_3di_tokens: int = 20,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pad_token_id: int = 20,  # must match collate_fn
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_classes = n_classes
        self.pad_token_id = pad_token_id

        # Channel 1: PLM projection 1024 -> 512
        self.plm_proj = nn.Linear(d_plm, d_model)

        # Channel 2: 3Di + conf embedding
        # 3Di embedding: tokens 0..19, plus padding token 20
        self.di_embedding = nn.Embedding(
            num_embeddings=n_3di_tokens + 1,
            embedding_dim=d_struct_tok,
            padding_idx=pad_token_id,
        )
        # Project [3Di_emb (64) + conf (1)] -> 256
        self.struct_proj = nn.Linear(d_struct_tok + 1, d_model // 2)  # 65 -> 256

        # Fusion [512;256] -> 512
        self.fusion_proj = nn.Linear(d_model + d_model // 2, d_model)  # 768 -> 512

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        self.encoder = SimpleTransformerEncoder(
            num_layers=n_layers,
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
        )

        # Heads
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(d_model, 1)
        self.class_head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        plm: torch.Tensor,
        di: torch.Tensor,
        conf: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        plm:  (B, L, 1024)
        di:   (B, L)
        conf: (B, L)
        attention_mask: (B, L) bool, True=real, False=pad
        """
        B, L, _ = plm.shape
        device = plm.device

        # Truncate if necessary (safety)
        if L > self.max_len:
            plm = plm[:, : self.max_len, :]
            di = di[:, : self.max_len]
            conf = conf[:, : self.max_len]
            attention_mask = attention_mask[:, : self.max_len]
            L = self.max_len

        # --- Channel 1: PLM projection ---
        h_plm = self.plm_proj(plm)  # (B, L, 512)

        # --- Channel 2: 3Di + conf ---
        # di may contain pad_token_id; embedding handles this
        di_emb = self.di_embedding(di)  # (B, L, 64)
        conf_expanded = conf.unsqueeze(-1)  # (B, L, 1)
        struct_in = torch.cat([di_emb, conf_expanded], dim=-1)  # (B, L, 65)
        h_struct = self.struct_proj(struct_in)  # (B, L, 256)

        # --- Fusion ---
        h = torch.cat([h_plm, h_struct], dim=-1)  # (B, L, 768)
        h = self.fusion_proj(h)  # (B, L, 512)

        # --- Positional encoding ---
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        pos_emb = self.pos_embedding(pos_ids)  # (B, L, 512)
        x = h + pos_emb  # (B, L, 512)

        # --- Prepend CLS ---
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 512)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, L+1, 512)

        # Build key padding mask for encoder: True = PAD
        cls_mask = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, L+1)
        src_key_padding_mask = ~full_mask.bool()  # (B, L+1) True=pad

        # --- Transformer encoder ---
        enc_out, attn_weights = self.encoder(
            x, src_key_padding_mask=src_key_padding_mask, return_attn=return_attn
        )  # enc_out: (B, L+1, 512)

        # h_cls is first token
        h_cls = enc_out[:, 0, :]  # (B, 512)
        h_cls = self.dropout(h_cls)

        # --- Heads ---
        logits_bin = self.bin_head(h_cls).squeeze(-1)  # (B,)
        logits_class = self.class_head(h_cls)          # (B, n_classes)

        out = {
            "logits_bin": logits_bin,
            "logits_class": logits_class,
        }

        if return_attn and attn_weights is not None:
            # attn_weights: (B, n_heads, S, S) where S = L+1
            # We want CLS attention to residues (exclude CLS->CLS)
            # Take attention from CLS query position (index 0)
            # shape: (B, n_heads, S) -> then dropping CLS position
            alpha = attn_weights[:, :, 0, 1:]  # (B, n_heads, L)
            # Average over heads
            alpha_mean = alpha.mean(dim=1)     # (B, L)
            out["attn_cls"] = alpha_mean

        return out
