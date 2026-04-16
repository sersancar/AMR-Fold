#!/usr/bin/env python3
"""
ProstT5 feature extraction utilities.

Updated version:
- supports local/offline model loading
- supports batched inference
- writes features directly into LMDB
- keeps a single-sequence API for debugging/inspection
"""

import os
import re
import gzip
import io
import math
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Iterator, Optional

import lmdb
import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
from tqdm import tqdm

# Standard Foldseek 3Di alphabet (20 tokens)
THREEDi_ALPHABET = "abcdefghijklmnopqrst"
THREEDi2IDX = {c: i for i, c in enumerate(THREEDi_ALPHABET)}


@dataclass
class ProstT5Config:
    model_name: str = "Rostlab/ProstT5"
    device: str = "auto"             # "auto", "cpu", or "cuda"
    half_precision: bool = True       # use fp16 on GPU
    local_files_only: bool = False    # force local-only loading
    max_generated_extra: int = 5      # cushion for 3Di length vs AA length


class ProstT5FeatureExtractor:
    """
    High-level wrapper around ProstT5.

    Provides:
      - encode_sequence: AA sequence -> PLM embedding + 3Di tokens + confidence
      - encode_fasta_to_lmdb: bulk processing of a multi-FASTA file into an LMDB feature store
    """

    def __init__(self, config: ProstT5Config = ProstT5Config()):
        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            do_lower_case=False,
            use_fast=False,
            local_files_only=config.local_files_only,
        )

        self.encoder = T5EncoderModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        ).to(self.device)

        self.seq2seq = T5ForConditionalGeneration.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        ).to(self.device)

        if self.device.type == "cpu" or not config.half_precision:
            self.encoder.float()
            self.seq2seq.float()
        else:
            self.encoder.half()
            self.seq2seq.half()

        self.encoder.eval()
        self.seq2seq.eval()

    # ---------- Public API ----------

    def encode_sequence(self, seq_id: str, seq_aa: str) -> Dict[str, np.ndarray]:
        """Encode one sequence. Useful for debugging or small-scale use."""
        feats = self.encode_batch([(seq_id, seq_aa)])
        return feats[0]

    def encode_batch(self, records: List[Tuple[str, str]]) -> List[Dict[str, np.ndarray]]:
        """Encode a batch of sequences and return a feature dict per sequence."""
        cleaned: List[Tuple[str, str]] = []
        for seq_id, seq_aa in records:
            clean_seq = self._clean_aa_sequence(seq_aa)
            if not clean_seq:
                raise ValueError(f"Sequence {seq_id} is empty after cleaning.")
            cleaned.append((seq_id, clean_seq))

        seq_ids = [sid for sid, _ in cleaned]
        seqs = [seq for _, seq in cleaned]

        plm_list = self._get_plm_embedding_batch(seqs)
        di_list, conf_list = self._get_3di_and_conf_batch(seqs)

        out: List[Dict[str, np.ndarray]] = []
        for seq_id, seq, plm_emb, tokens_3di, conf_3di in zip(seq_ids, seqs, plm_list, di_list, conf_list):
            L = len(seq)
            L_eff = min(L, len(tokens_3di), len(conf_3di), plm_emb.shape[0])
            out.append({
                "id": seq_id,
                "plm": plm_emb[:L_eff].astype(np.float32),
                "tokens_3di": tokens_3di[:L_eff].astype(np.int16),
                "conf_3di": conf_3di[:L_eff].astype(np.float32),
            })
        return out

    def encode_fasta_to_lmdb(
        self,
        fasta_path: str,
        lmdb_dir: str,
        map_size_gb: float = 8.0,
        batch_size: int = 8,
        token_budget: int = 4096,
        commit_every: int = 100,
        overwrite: bool = True,
        progress: bool = True,
    ) -> None:
        """
        Bulk feature extraction from a multi-FASTA file directly into LMDB.

        LMDB value schema (same as the existing downstream loader expects):
          key   = sequence ID (utf-8)
          value = pickle.dumps((plm, di_tokens, di_conf))
        """
        os.makedirs(lmdb_dir, exist_ok=True)
        env = lmdb.open(
            lmdb_dir,
            map_size=int(map_size_gb * (1024 ** 3)),
            subdir=True,
            lock=True,
        )

        seq_iter = list(self._read_fasta(fasta_path))
        if not seq_iter:
            env.close()
            raise ValueError(f"No sequences found in FASTA: {fasta_path}")

        # Length-aware ordering improves padding efficiency.
        seq_iter.sort(key=lambda x: len(self._clean_aa_sequence(x[1])))
        batches = list(self._yield_batches(seq_iter, batch_size=batch_size, token_budget=token_budget))
        batch_iter: Iterable[List[Tuple[str, str]]] = batches
        if progress:
            batch_iter = tqdm(batch_iter, total=len(batches), desc="ProstT5 -> LMDB")

        written = 0
        txn = env.begin(write=True)
        try:
            if overwrite:
                # LMDB doesn't offer a cheap “clear all keys” for a subdir store.
                # In the pipeline we write a fresh shard LMDB each run, so this flag
                # mainly controls whether existing keys are overwritten.
                pass

            for batch in batch_iter:
                feats_list = self.encode_batch(batch)
                for feats in feats_list:
                    key = feats["id"].encode("utf-8")
                    value = pickle.dumps(
                        (feats["plm"], feats["tokens_3di"], feats["conf_3di"]),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    txn.put(key, value, overwrite=overwrite)
                    written += 1

                if written % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.commit()
            env.sync()
        finally:
            try:
                env.close()
            except Exception:
                pass

    # ---------- Internal helpers ----------

    @staticmethod
    def _open_fasta(path: str):
        if path.endswith(".gz"):
            return io.TextIOWrapper(gzip.open(path, "rb"))
        return open(path, "r")

    @classmethod
    def _read_fasta(cls, path: str) -> Iterable[Tuple[str, str]]:
        with cls._open_fasta(path) as f:
            header = None
            chunks: List[str] = []
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        yield header, "".join(chunks)
                    header = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line.strip())
            if header is not None:
                yield header, "".join(chunks)

    @staticmethod
    def _clean_aa_sequence(seq: str) -> str:
        seq = seq.strip().upper()
        seq = re.sub(r"[UZOB]", "X", seq)
        return seq

    @staticmethod
    def _prepare_embedding_input(seq_aa: str) -> str:
        return " ".join(list(seq_aa))

    @staticmethod
    def _prepare_3di_input(seq_aa: str) -> str:
        return "<AA2fold> " + " ".join(list(seq_aa))

    @staticmethod
    def _yield_batches(
        records: List[Tuple[str, str]],
        batch_size: int,
        token_budget: int,
    ) -> Iterator[List[Tuple[str, str]]]:
        current: List[Tuple[str, str]] = []
        current_tokens = 0
        for rec in records:
            seq_len = len(rec[1])
            if current and (len(current) >= batch_size or current_tokens + seq_len > token_budget):
                yield current
                current = []
                current_tokens = 0
            current.append(rec)
            current_tokens += seq_len
        if current:
            yield current

    def _get_plm_embedding_batch(self, seqs_aa: List[str]) -> List[np.ndarray]:
        lengths = [len(s) for s in seqs_aa]
        inputs = [self._prepare_embedding_input(s) for s in seqs_aa]

        batch = self.tokenizer(
            inputs,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            hidden = out.last_hidden_state  # (B, T, D)

        embs: List[np.ndarray] = []
        for i, L in enumerate(lengths):
            if hidden.size(1) < L + 1:
                raise RuntimeError(f"Unexpected token length {hidden.size(1)} for sequence length {L}")
            emb = hidden[i, 1 : L + 1].detach().cpu().float().numpy()
            embs.append(emb)
        return embs

    def _get_3di_and_conf_batch(self, seqs_aa: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        lengths = [len(s) for s in seqs_aa]
        inputs = [self._prepare_3di_input(s) for s in seqs_aa]
        max_len = max(lengths) + self.config.max_generated_extra
        min_len = min(lengths)

        batch = self.tokenizer(
            inputs,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gen = self.seq2seq.generate(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                max_length=max_len,
                min_length=min_len,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded_all = self.tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        decoded_all = [x.replace(" ", "") for x in decoded_all]

        # gen.scores is a list of length T_dec where each item is (B, vocab)
        scores = gen.scores
        seqs_ids = gen.sequences  # (B, T_seq)

        tokens_list: List[np.ndarray] = []
        confs_list: List[np.ndarray] = []

        for i, L in enumerate(lengths):
            tokens_str = decoded_all[i]
            tokens_int = [THREEDi2IDX.get(ch, -1) for ch in tokens_str]

            confs: List[float] = []
            for t, logits in enumerate(scores):
                if t + 1 >= seqs_ids.size(1):
                    break
                token_id = seqs_ids[i, t + 1]
                probs = torch.softmax(logits[i].float(), dim=-1)
                confs.append(float(probs[token_id].item()))

            L_eff = min(L, len(tokens_int), len(confs))
            tokens_arr = np.array(tokens_int[:L_eff], dtype=np.int16)
            confs_arr = np.array(confs[:L_eff], dtype=np.float32)
            tokens_list.append(tokens_arr)
            confs_list.append(confs_arr)

        return tokens_list, confs_list
