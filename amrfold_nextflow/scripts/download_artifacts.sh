#!/usr/bin/env bash
set -euo pipefail

USER="sersancar"
REPO="amr-fold-artifacts"
BASE_URL="https://huggingface.co/datasets/${USER}/${REPO}/resolve/main"

echo "Downloading model weights and containers from Hugging Face (${USER}/${REPO})..."

# Create target directories
mkdir -p models/ProstT5
mkdir -p containers

echo "-> ProstT5 model.safetensors"
curl -L "${BASE_URL}/ProstT5/model.safetensors" \
  -o "amrfold_nextflow/models/ProstT5/model.safetensors"

for SEED in 1 2 3 4 5; do
  echo "-> checkpoint seed ${SEED}"
  curl -L "${BASE_URL}/checkpoints/best_checkpoint_seed_${SEED}.pt" \
    -o "amrfold_nextflow/models/best_checkpoint_seed_${SEED}.pt"
done

echo "-> amrfold_cpu.sif"
curl -L "${BASE_URL}/containers/amrfold_cpu.sif" \
  -o "amrfold_nextflow/containers/amrfold_cpu.sif"

echo "-> amrfold_gpu.sif"
curl -L "${BASE_URL}/containers/amrfold_gpu.sif" \
  -o "amrfold_nextflow/containers/amrfold_gpu.sif"

echo "All artifacts downloaded succesfully."
