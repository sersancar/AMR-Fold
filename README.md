# AMR-Fold

AMR-Fold is a supervised learning project to predict **antimicrobial resistance (AMR) classes** from **protein sequences**. The core idea is to combine (i) **frozen protein language model (pLM) residue embeddings** with (ii) **lightweight structure-aware tokens** predicted by ProstT5, then train a compact transformer classifier to output both **ARG vs non-ARG** and **antibiotic class** labels.

## Dataset (Zenodo)

The AMR-Fold dataset (including predefined splits) is hosted on Zenodo:

- https://doi.org/10.5281/zenodo.18234017

## Model architecture (high level)

**Input: two frozen ProstT5-based channels**
1. **Channel 1 — ProstT5 residue embeddings**
   - `E ∈ R^{L×1024}`

2. **Channel 2 — ProstT5-predicted structural tokens + confidence**
   - Structural tokens: `t ∈ {0,…,K−1}^L`, with `K = 20`
   - Token embedding: `z_tok = Emb_3Di(t) ∈ R^{L×64}`
   - Confidence: `c ∈ R^L` (appended as 1 scalar per residue)
   - Combined: `z_struct = [z_tok; c] ∈ R^{L×65}`

**Per-residue fusion**
- Project pLM: `1024 → 512`
- Project structure: `65 → 256`
- Concatenate and mix: `[512; 256] = 768 → 512`
- Add learned absolute positional embedding:
  - `P ∈ R^{L_max×512}`, `L_max = 1024`
  - If `L > L_max`, sequence is truncated to `L_max`

**Sequence encoder**
- Small Transformer encoder (3 layers)
  - `d_model = 512`
  - `n_heads = 8`
  - `FFN dim = 2048`
  - `dropout = 0.1`
- Prepend a `[CLS]` token and take the final hidden state at position 0 as the sequence representation `h_cls ∈ R^{512}`.

**Attention regularisation (on [CLS] attention over residues)**
Let `α ∈ R^L` be the [CLS]→residue attention distribution (averaged over heads).

- **Entropy minimisation (sparsity / focus)**
  - `H(α) = -∑_j α_j log(α_j + ε)`
  - `L_ent = λ_ent · E[H(α)]` (typical `λ_ent ≈ 0.01–0.1`)

- **Local continuity regularisation (contiguous stretches)**
  - `L_cont = λ_cont · E[∑_j (α_j - α_{j+1})^2]` (typical `λ_cont ≈ 0.01`)

- `L_attn = L_ent + L_cont`

**Multi-task classification head**
- **Binary head (ARG vs non-ARG)**:
  - `h_cls → Dropout → Linear(512→1)`; loss: `BCEWithLogitsLoss`
- **Multiclass head (antibiotic class + non_ARG)**:
  - `h_cls → Dropout → Linear(512→n_classes)`; loss: `CrossEntropyLoss`
- **Total loss**
  - `L_total = L_bin + λ · L_class + L_attn`, with `λ ≈ 0.5–1.0`

## Project status

AMR-Fold is **under active development**. Interfaces, directory structure, and training/evaluation scripts may change.

## Citation

If you use the dataset, please cite the Zenodo record:

- https://doi.org/10.5281/zenodo.18234017
