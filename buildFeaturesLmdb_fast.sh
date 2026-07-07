#!/bin/bash
#SBATCH --job-name=buildFeaturesLmdbFast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/home/ssanchez/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/ssanchez/amr_fold/logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie

set -euo pipefail

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "CWD: $(pwd)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"

# Conda
source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

CODEDIR=/home/ssanchez/amr_fold
FEATURES_DIR=/data3/ssanchez/amr_fold/features
OUT_FINAL=/data3/ssanchez/amr_fold/features.lmdb

TSV1=/data3/ssanchez/amr_fold/data/DB_train.tsv
TSV2=/data3/ssanchez/amr_fold/data/DB_val.tsv
TSV3=/data3/ssanchez/amr_fold/data/DB_test.tsv

# Build on local job scratch if available (often MUCH faster than shared FS),
# then copy to /data3 at the end.
TMPBASE="${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}"
OUT_LOCAL="${TMPBASE}/features.lmdb"

mkdir -p "$TMPBASE"

echo "Local scratch: $TMPBASE"
echo "LMDB local:    $OUT_LOCAL"
echo "LMDB final:    $OUT_FINAL"

# If you want fully conservative durability, remove --fast-io (it will be slower).
python "$CODEDIR/build_features_lmdb_fast.py" \
  --tsvs "$TSV1" "$TSV2" "$TSV3" \
  --features-dir "$FEATURES_DIR" \
  --lmdb-dir "$OUT_LOCAL" \
  --commit-every 1000 \
  --fast-io \
  --overwrite

# Copy to persistent location
mkdir -p "$OUT_FINAL"
rsync -a --delete "$OUT_LOCAL/" "$OUT_FINAL/"

echo "Job finished on $(date)"
echo "Final LMDB: $OUT_FINAL"
