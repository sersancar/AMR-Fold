#!/bin/bash -l 
#SBATCH --job-name=train_AMRFold
#SBATCH --output=/home/users/u103609/scripts/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/users/u103609/scripts/amr_fold/logs//%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=p201140
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie

set -euo pipefail

# Conda
source /project/home/p201140/Miniforge3/etc/profile.d/conda.sh
conda activate /mnt/tier2/project/p201140/Miniforge3/envs/amrfold-cuda


echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Current working directory: $(pwd)"

# Directories
DATADIR=/project/home/p201140/amr_fold_data/tables
FEATURESDIR=/project/home/p201140/amr_fold_data/features
OUTPUTDIR=/project/home/p201140/amr_fold_data/models
CODEDIR=/home/users/u103609/scripts/amr_fold

# Create dirs if needed
mkdir -p "$OUTPUTDIR"
mkdir -p /home/users/u103609/scripts/amr_fold/logs

# Variables
DB_TRAIN=${DATADIR}/DB_train.tsv
DB_VAL=${DATADIR}/DB_val.tsv
DB_TEST=${DATADIR}/DB_test.tsv
MAX_LEN=1024
BATCH_SIZE=32
EPOCHS=1

# Export so they are visible inside Python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Make sure Python can find train_amrFold.py in CODEDIR (if it lives there)
export PYTHONPATH="$CODEDIR:$PYTHONPATH"

# Command
python train_amrFold.py \
  --train_tsv "$DB_TRAIN" \
  --val_tsv "$DB_VAL" \
  --test_tsv "$DB_TEST" \
  --features_dir "$FEATURESDIR" \
  --out_dir "$OUTPUTDIR" \
  --max_len "$MAX_LEN" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Job finished on $(date)"

# RUN:  sbatch train_AMRFold.sh

