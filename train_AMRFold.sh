#!/bin/bash
#SBATCH --job-name=train_AMRFold
#SBATCH --output=/home/ssanchez/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/ssanchez/amr_fold/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie

# Load modules
module load cuda/12.1.1
module load cudnn_for_cuda12/8.9.1

# Conda
source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

# CUDA
#CUDA_LAUNCH_BLOCKING=1
#export CUDA_LAUNCH_BLOCKING

# Logging
echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Current working directory: $(pwd)"

# Directories
DATADIR=/data3/ssanchez/amr_fold/data
FEATURESDIR=/data3/ssanchez/amr_fold/features
FEATURESLMDBDIR=/data3/ssanchez/amr_fold/features.lmdb
OUTPUTDIR=/data3/ssanchez/amr_fold/models
CODEDIR=/home/ssanchez/amr_fold


# Create dirs if needed
mkdir -p "$OUTPUTDIR"
mkdir -p /home/ssanchez/amr_fold/logs

# Variables
DB_TRAIN=${DATADIR}/DB_train.tsv
DB_VAL=${DATADIR}/DB_val.tsv
DB_TEST=${DATADIR}/DB_test.tsv
MAX_LEN=1024
BATCH_SIZE=32
EPOCHS=40

# Export so they are visible inside Python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Make sure Python can find train_amrFold.py in CODEDIR (if it lives there)
export PYTHONPATH="$CODEDIR:$PYTHONPATH"

# Command
python train_amrFold.py \
--train_tsv ${DB_TRAIN}	\
--val_tsv ${DB_VAL} \
--test_tsv ${DB_TEST} \
--features_lmdb ${FEATURESLMDBDIR} \
--out_dir ${OUTPUTDIR} \
--max_len ${MAX_LEN} \
--batch_size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--num_workers 6 

echo "Job finished on $(date)"

# RUN:  sbatch train_AMRFold.sh
