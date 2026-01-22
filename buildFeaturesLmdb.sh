#!/bin/bash
#SBATCH --job-name=buildFeaturesLmdb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ssanchez/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/ssanchez/amr_fold/logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie


set -euo pipefail

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Current working directory: $(pwd)"

# Conda
source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

#Command
python build_features_lmdb.py

echo "Job finished on $(date)"

# RUN:  sbatch buildFeaturesLmdb.sh
