#!/usr/bin/env python3
"""
ProstT5 feature extraction utilities.

- Channel 1: ProstT5 encoder embeddings (T5EncoderModel)
- Channel 2: ProstT5 AA->3Di translation + per-residue confidence (AutoModelForSeq2SeqLM)

Designed to be reusable for:
    - Offline preprocessing of a FASTA database
    - Online inference in a deployed model
"""

import os
import re
import gzip
import io
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable
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
    device: str = "auto"       # "auto", "cpu", or "cuda"
    half_precision: bool = True  # use fp16 on GPU
    max_generated_extra: int = 5  # cushion for 3Di length vs AA length

class ProstT5FeatureExtractor:
    """
    High-level wrapper around ProstT5.

    Provides:
      - encode_sequence: AA sequence -> PLM embedding + 3Di tokens + confidence
      - encode_fasta: bulk processing of a multi-FASTA file into .npy feature files
    """

    def __init__(self, config: ProstT5Config = ProstT5Config()):
        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            do_lower_case=False,
	    use_fast=False
	)

        self.encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            config.model_name
        ).to(self.device)

        self.seq2seq: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
            config.model_name
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

    def encode_sequence(
        self,
        seq_id: str,
        seq_aa: str,
    ) -> Dict[str, np.ndarray]:
        """
        Encode a single amino acid sequence.

        Returns a dict with:
          - "id":           seq_id (string)
          - "plm":          (L, D) float32 ProstT5 embeddings
          - "tokens_3di":   (L,)   int16 3Di tokens in [0, 19] (or -1 for unknown)
          - "conf_3di":     (L,)   float32 confidence per residue in [0, 1]
        """
        clean_seq = self._clean_aa_sequence(seq_aa)
        L = len(clean_seq)
        if L == 0:
            raise ValueError(f"Sequence {seq_id} is empty after cleaning.")

        # Channel 1: encoder embeddings
        plm_emb = self._get_plm_embedding(clean_seq)  # (L, D)

        # Channel 2: 3Di tokens + confidence
        tokens_3di, conf_3di = self._get_3di_and_conf(clean_seq)  # (L,)

        # Ensure consistent length; in practice tokens_3di and conf_3di
        # should already be length L, but we truncate conservatively.
        L_eff = min(L, len(tokens_3di), len(conf_3di))
        plm_emb = plm_emb[:L_eff]
        tokens_3di = tokens_3di[:L_eff]
        conf_3di = conf_3di[:L_eff]

        return {
            "id": seq_id,
            "plm": plm_emb.astype(np.float32),
            "tokens_3di": tokens_3di.astype(np.int16),
            "conf_3di": conf_3di.astype(np.float32),
        }

    def encode_fasta(
        self,
        fasta_path: str,
        out_dir: str,
        overwrite: bool = False,
        progress: bool = True,
    ) -> None:
        """
        Bulk feature extraction from a multi-FASTA file.

        For each sequence with header '>Seq_xxx', writes:
          out_dir/Seq_xxx.plm.npy
          out_dir/Seq_xxx.3di_tokens.npy
          out_dir/Seq_xxx.3di_conf.npy
        """
        os.makedirs(out_dir, exist_ok=True)

        seq_iter = self._read_fasta(fasta_path)
        if progress:
            seq_iter = tqdm(seq_iter, desc="ProstT5 feature extraction")

        for seq_id, seq_aa in seq_iter:
            plm_path = os.path.join(out_dir, f"{seq_id}.plm.npy")
            tok_path = os.path.join(out_dir, f"{seq_id}.3di_tokens.npy")
            conf_path = os.path.join(out_dir, f"{seq_id}.3di_conf.npy")

            if not overwrite and all(
                os.path.exists(p) for p in (plm_path, tok_path, conf_path)
            ):
                continue

            feats = self.encode_sequence(seq_id, seq_aa)
            np.save(plm_path, feats["plm"])
            np.save(tok_path, feats["tokens_3di"])
            np.save(conf_path, feats["conf_3di"])

    # ---------- Internal helpers ----------

    @staticmethod
    def _open_fasta(path: str):
        """Open a text file that may be gzipped (.gz)."""
        if path.endswith(".gz"):
            # gzip.open returns a binary file; wrap in TextIOWrapper for text iteration
            return io.TextIOWrapper(gzip.open(path, "rb"))
        else:
            return open(path, "r")

    @classmethod
    def _read_fasta(cls, path: str) -> Iterable[Tuple[str, str]]:
        """
        Simple FASTA reader yielding (id, seq).

        - Supports both plain text (.fa/.fasta/.faa) and gzipped (.fa.gz, .faa.gz, etc.).
        - Uses the first token after '>' as the sequence ID (e.g. 'Seq_123').
        """
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
                    header = line[1:].split()[0]  # take first token as ID
                    chunks = []
                else:
                    chunks.append(line.strip())
            if header is not None:
                yield header, "".join(chunks)

    @staticmethod
    def _clean_aa_sequence(seq: str) -> str:
        """Uppercase AA sequence and replace rare/ambiguous residues by 'X'."""
        seq = seq.strip().upper()
        seq = re.sub(r"[UZOB]", "X", seq)
        return seq

    def _prepare_embedding_input(self, seq_aa: str) -> str:
        """
        Prepare input string for extracting embeddings:
          "A B C D ..."
        (No AA2fold prefix; we only need encoder states here.)
        """
        return " ".join(list(seq_aa))

    def _prepare_3di_input(self, seq_aa: str) -> str:
        """
        Prepare input string for AA -> 3Di translation:
          "<AA2fold> A B C D ..."
        """
        return "<AA2fold> " + " ".join(list(seq_aa))

    def _get_plm_embedding(self, seq_aa: str) -> np.ndarray:
        """
        Channel 1: compute ProstT5 encoder embeddings for one AA sequence.

        Returns:
          (L, D) numpy array (float32)
        """
        L = len(seq_aa)
        inp = self._prepare_embedding_input(seq_aa)

        batch = self.tokenizer(
            [inp],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            hidden = out.last_hidden_state[0]  # (T, D)

        # Token 0 is a special token; residues align with positions [1 : L+1]
        if hidden.size(0) < L + 1:
            raise RuntimeError(
                f"Unexpected token length {hidden.size(0)} for sequence length {L}"
            )

        emb = hidden[1 : L + 1].detach().cpu().float().numpy()
        return emb  # (L, D)

    def _get_3di_and_conf(
        self,
        seq_aa: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Channel 2: AA -> 3Di tokens + per-residue confidence.

        Returns:
          tokens_int: (L,) np.int16, values in [0, 19] or -1 if unknown symbol
          confs:      (L,) np.float32, probabilities in [0, 1]
        """
        L = len(seq_aa)
        inp = self._prepare_3di_input(seq_aa)

        batch = self.tokenizer(
            [inp],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        max_len = L + self.config.max_generated_extra
        min_len = L

        with torch.no_grad():
            gen = self.seq2seq.generate(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                max_length=max_len,
                min_length=min_len,
                num_beams=1,
                do_sample=False,  # greedy decoding
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode 3Di string, e.g. "a b c d ..." -> "abcd..."
        decoded = self.tokenizer.batch_decode(
            gen.sequences,
            skip_special_tokens=True,
        )[0]
        decoded = decoded.replace(" ", "")
        tokens_str = decoded

        tokens_int: List[int] = []
        for ch in tokens_str:
            tokens_int.append(THREEDi2IDX.get(ch, -1))

        scores = gen.scores                  # list length L_dec, each (batch, vocab)
        seq_ids = gen.sequences[0]           # generated token IDs

        confs: List[float] = []
        for t, logits in enumerate(scores):
            if t + 1 >= seq_ids.size(0):
                break
            probs = torch.softmax(logits[0].float(), dim=-1)
            token_id = seq_ids[t + 1]
            confs.append(float(probs[token_id].item()))

        # Truncate to AA length
        L_eff = min(L, len(tokens_int), len(confs))
        tokens_arr = np.array(tokens_int[:L_eff], dtype=np.int16)
        confs_arr = np.array(confs[:L_eff], dtype=np.float32)

        return tokens_arr, confs_arr

