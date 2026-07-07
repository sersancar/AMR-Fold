#!/usr/bin/env python3
"""
ProstT5 feature extraction utilities.
"""

import os
import re
import gzip
import io
import gc
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Iterator, Optional

import lmdb
import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
from tqdm import tqdm

THREEDi_ALPHABET = "abcdefghijklmnopqrst"
THREEDi2IDX = {c: i for i, c in enumerate(THREEDi_ALPHABET)}

MAX_PROSTT5_LEN = 3000


@dataclass
class ProstT5Config:
    model_name: str = "Rostlab/ProstT5"
    device: str = "auto"
    half_precision: bool = True
    local_files_only: bool = False
    max_generated_extra: int = 5


class ProstT5FeatureExtractor:
    def __init__(self, config: ProstT5Config = ProstT5Config()):
        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        print(f"[ProstT5] device={self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"[ProstT5] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
            print(f"[ProstT5] torch.cuda.device_count={torch.cuda.device_count()}", flush=True)
            print(f"[ProstT5] torch.cuda.current_device={torch.cuda.current_device()}", flush=True)
            print(f"[ProstT5] torch.cuda.device_name={torch.cuda.get_device_name(0)}", flush=True)

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

    def encode_sequence(self, seq_id: str, seq_aa: str) -> Dict[str, np.ndarray]:
        feats = self.encode_batch([(seq_id, seq_aa)])
        return feats[0]

    def encode_batch(self, records: List[Tuple[str, str]]) -> List[Dict[str, np.ndarray]]:
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

        self._clear_cuda_cache()
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
        os.makedirs(lmdb_dir, exist_ok=True)

        env = lmdb.open(
            lmdb_dir,
            map_size=int(map_size_gb * (1024 ** 3)),
            subdir=True,
            lock=True,
        )

        raw_records = list(self._read_fasta(fasta_path))
        if not raw_records:
            env.close()
            raise ValueError(f"No sequences found in FASTA: {fasta_path}")

        seq_iter: List[Tuple[str, str]] = []
        n_cropped = 0
        max_raw_len = 0
        max_used_len = 0

        for seq_id, seq in raw_records:
            max_raw_len = max(max_raw_len, len(seq))
            clean_seq = self._clean_aa_sequence(seq)
            max_used_len = max(max_used_len, len(clean_seq))
            if len(seq.strip()) > len(clean_seq):
                n_cropped += 1
            seq_iter.append((seq_id, clean_seq))

        seq_iter.sort(key=lambda x: len(x[1]))

        quadratic_budget = max(int(token_budget) * 1024, 1)

        batches = list(
            self._yield_batches(
                seq_iter,
                batch_size=batch_size,
                token_budget=token_budget,
                quadratic_budget=quadratic_budget,
                generated_extra=self.config.max_generated_extra,
            )
        )

        print(
            f"[ProstT5] sequences={len(seq_iter)} batches={len(batches)} "
            f"batch_size={batch_size} token_budget={token_budget} "
            f"quadratic_budget={quadratic_budget} "
            f"max_raw_len={max_raw_len} max_used_len={max_used_len} "
            f"cropped_over_{MAX_PROSTT5_LEN}aa={n_cropped}",
            flush=True,
        )

        batch_iter: Iterable[List[Tuple[str, str]]] = batches
        if progress:
            batch_iter = tqdm(batch_iter, total=len(batches), desc="ProstT5 -> LMDB")

        written = 0
        txn = env.begin(write=True)

        try:
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
            self._clear_cuda_cache()

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

        if len(seq) > MAX_PROSTT5_LEN:
            start = (len(seq) - MAX_PROSTT5_LEN) // 2
            seq = seq[start:start + MAX_PROSTT5_LEN]

        return seq

    @staticmethod
    def _prepare_embedding_input(seq_aa: str) -> str:
        return " ".join(list(seq_aa))

    @staticmethod
    def _prepare_3di_input(seq_aa: str) -> str:
        return "<AA2fold> " + " ".join(list(seq_aa))

    @staticmethod
    def _batch_cost(records: List[Tuple[str, str]], generated_extra: int = 5) -> int:
        if not records:
            return 0

        max_len = max(len(seq) for _, seq in records) + int(generated_extra)
        return len(records) * max_len * max_len

    @classmethod
    def _yield_batches(
        cls,
        records: List[Tuple[str, str]],
        batch_size: int,
        token_budget: int,
        quadratic_budget: Optional[int] = None,
        generated_extra: int = 5,
    ) -> Iterator[List[Tuple[str, str]]]:
        current: List[Tuple[str, str]] = []
        current_tokens = 0

        token_budget = max(1, int(token_budget))
        batch_size = max(1, int(batch_size))
        quadratic_budget = int(quadratic_budget) if quadratic_budget else None

        for rec in records:
            seq_len = len(rec[1])
            candidate = current + [rec]
            candidate_tokens = current_tokens + seq_len
            candidate_quad = cls._batch_cost(candidate, generated_extra=generated_extra)

            would_exceed_count = len(candidate) > batch_size
            would_exceed_tokens = candidate_tokens > token_budget
            would_exceed_quad = quadratic_budget is not None and candidate_quad > quadratic_budget

            if current and (would_exceed_count or would_exceed_tokens or would_exceed_quad):
                yield current
                current = []
                current_tokens = 0

            current.append(rec)
            current_tokens += seq_len

        if current:
            yield current

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_plm_embedding_batch(self, seqs_aa: List[str]) -> List[np.ndarray]:
        lengths = [len(s) for s in seqs_aa]
        inputs = [self._prepare_embedding_input(s) for s in seqs_aa]

        batch = self.tokenizer(
            inputs,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            out = self.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            hidden = out.last_hidden_state

        embs: List[np.ndarray] = []

        for i, L in enumerate(lengths):
            if hidden.size(1) < L + 1:
                raise RuntimeError(f"Unexpected token length {hidden.size(1)} for sequence length {L}")

            emb = hidden[i, 1:L + 1].detach().cpu().float().numpy()
            embs.append(emb)

        del batch, out, hidden
        self._clear_cuda_cache()
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

        with torch.inference_mode():
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

        scores = gen.scores
        seqs_ids = gen.sequences

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

        del batch, gen, scores, seqs_ids
        self._clear_cuda_cache()

        return tokens_list, confs_list
