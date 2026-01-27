#!/bin/bash
#SBATCH --job-name=tune_AMRFold
#SBATCH --output=/home/ssanchez/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/ssanchez/amr_fold/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie

set -euo pipefail

# -------------------------
# Environment
# -------------------------
module load cuda/12.1.1
module load cudnn_for_cuda12/8.9.1

source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

# Ensure logs are written progressively
export PYTHONUNBUFFERED=1

echo "Job started: $(date)"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "PWD: $(pwd)"

# -------------------------
# Paths
# -------------------------
DATADIR=/data3/ssanchez/amr_fold/data
FEATURESLMDBDIR=/data3/ssanchez/amr_fold/features.lmdb
OUTPUTDIR=/data3/ssanchez/amr_fold/models
CODEDIR=/home/ssanchez/amr_fold
LOGDIR=/home/ssanchez/amr_fold/logs

mkdir -p "$OUTPUTDIR" "$LOGDIR"

DB_TRAIN=${DATADIR}/DB_train.tsv
DB_VAL=${DATADIR}/DB_val.tsv
DB_TEST=${DATADIR}/DB_test.tsv

# -------------------------
# Fast local Optuna DB + persistent checkpointing
# -------------------------
STUDY_NAME=amrfold_hpo
OPTUNA_DB_PERSIST="${OUTPUTDIR}/${STUDY_NAME}.db"

pick_fast_dir () {
  for d in "${SLURM_TMPDIR:-}" "/localscratch/${USER}" "/dev/shm/${USER}" "/tmp/${USER}"; do
    [[ -z "$d" ]] && continue
    mkdir -p "$d" 2>/dev/null || continue
    touch "$d/.writetest" 2>/dev/null || continue
    rm -f "$d/.writetest" 2>/dev/null || true
    echo "$d"
    return 0
  done
  return 1
}

FAST_BASE="$(pick_fast_dir)" || { echo "ERROR: No writable fast local directory found."; exit 1; }
JOBTAG="${SLURM_JOB_ID:-manual}"
OPTUNA_DB_DIR="${FAST_BASE}/optuna_${STUDY_NAME}_${JOBTAG}"
mkdir -p "$OPTUNA_DB_DIR"

OPTUNA_DB_LOCAL="${OPTUNA_DB_DIR}/${STUDY_NAME}.db"
OPTUNA_STORAGE="sqlite:////${OPTUNA_DB_LOCAL}"

# Encourage SQLite tmp files to stay local too
export SQLITE_TMPDIR="${OPTUNA_DB_DIR}"

echo "Optuna DB local:    ${OPTUNA_DB_LOCAL}"
echo "Optuna DB persist:  ${OPTUNA_DB_PERSIST}"
df -T "${OPTUNA_DB_DIR}" | tail -1
df -T "${OUTPUTDIR}" | tail -1

restore_optuna_db () {
  if [[ -f "${OPTUNA_DB_PERSIST}" ]]; then
    cp -f "${OPTUNA_DB_PERSIST}" "${OPTUNA_DB_LOCAL}"
    [[ -f "${OPTUNA_DB_PERSIST}-wal" ]] && cp -f "${OPTUNA_DB_PERSIST}-wal" "${OPTUNA_DB_LOCAL}-wal" || true
    [[ -f "${OPTUNA_DB_PERSIST}-shm" ]] && cp -f "${OPTUNA_DB_PERSIST}-shm" "${OPTUNA_DB_LOCAL}-shm" || true
  fi
}

backup_optuna_db () {
  local tmp="${OPTUNA_DB_PERSIST}.tmp"
  mkdir -p "$(dirname "${OPTUNA_DB_PERSIST}")"

  for attempt in 1 2 3 4 5; do
    if command -v sqlite3 >/dev/null 2>&1; then
      if sqlite3 "${OPTUNA_DB_LOCAL}" <<SQL
.timeout 60000
.backup '${tmp}'
SQL
      then
        mv -f "${tmp}" "${OPTUNA_DB_PERSIST}"
        return 0
      fi
    else
      cp -f "${OPTUNA_DB_LOCAL}" "${OPTUNA_DB_PERSIST}" && \
      ([[ -f "${OPTUNA_DB_LOCAL}-wal" ]] && cp -f "${OPTUNA_DB_LOCAL}-wal" "${OPTUNA_DB_PERSIST}-wal" || true) && \
      ([[ -f "${OPTUNA_DB_LOCAL}-shm" ]] && cp -f "${OPTUNA_DB_LOCAL}-shm" "${OPTUNA_DB_PERSIST}-shm" || true) && \
      return 0
    fi
    sleep $((attempt * 2))
  done
  return 1
}

restore_optuna_db || true

PID0=""
PID1=""
CKPT_PID=""

cleanup () {
  echo "Cleanup/exit handler: $(date)"
  [[ -n "${PID0}" ]] && kill "${PID0}" 2>/dev/null || true
  [[ -n "${PID1}" ]] && kill "${PID1}" 2>/dev/null || true
  [[ -n "${CKPT_PID}" ]] && kill "${CKPT_PID}" 2>/dev/null || true
  echo "Final Optuna DB backup..."
  backup_optuna_db || true
}
trap cleanup EXIT INT TERM

# Periodic checkpoint every 5 minutes to persistent storage
(
  while sleep 300; do
    backup_optuna_db >/dev/null 2>&1 || true
  done
) &
CKPT_PID="$!"

# -------------------------
# Stage dataset to fast local (TSVs + features.lmdb)
# -------------------------
STAGE_DATA=1

if [[ "${STAGE_DATA}" -eq 1 ]]; then
  # For 73G LMDB, prefer /localscratch specifically if writable; else fallback to FAST_BASE
  STAGE_BASE="/localscratch/${USER}"
  if ! (mkdir -p "${STAGE_BASE}" 2>/dev/null && touch "${STAGE_BASE}/.writetest" 2>/dev/null); then
    STAGE_BASE="${FAST_BASE}"
  else
    rm -f "${STAGE_BASE}/.writetest" 2>/dev/null || true
  fi

  STAGE_DIR="${STAGE_BASE}/amrfold_stage_${JOBTAG}"
  mkdir -p "${STAGE_DIR}/data"

  echo "Staging TSVs to: ${STAGE_DIR}/data"
  cp -f "${DB_TRAIN}" "${STAGE_DIR}/data/DB_train.tsv"
  cp -f "${DB_VAL}"   "${STAGE_DIR}/data/DB_val.tsv"
  cp -f "${DB_TEST}"  "${STAGE_DIR}/data/DB_test.tsv"

  echo "Staging features.lmdb (~73G) to: ${STAGE_DIR}/features.lmdb"
  if [[ -d "${FEATURESLMDBDIR}" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${FEATURESLMDBDIR}/" "${STAGE_DIR}/features.lmdb/"
    else
      cp -a "${FEATURESLMDBDIR}" "${STAGE_DIR}/features.lmdb"
    fi
  else
    cp -f "${FEATURESLMDBDIR}" "${STAGE_DIR}/features.lmdb"
  fi

  # Re-point paths to staged local copies (used by BOTH tuning + final retrain)
  DB_TRAIN="${STAGE_DIR}/data/DB_train.tsv"
  DB_VAL="${STAGE_DIR}/data/DB_val.tsv"
  DB_TEST="${STAGE_DIR}/data/DB_test.tsv"
  FEATURESLMDBDIR="${STAGE_DIR}/features.lmdb"

  echo "Using LOCAL staged dataset:"
  echo "  DB_TRAIN=${DB_TRAIN}"
  echo "  FEATURES=${FEATURESLMDBDIR}"
  df -T "${STAGE_BASE}" | tail -1
fi

# -------------------------
# Run config
# -------------------------
METRIC=f1_class_macro_pos

TRIALS_TOTAL=80
TRIALS_PER_WORKER=$((TRIALS_TOTAL / 2))
MAX_EPOCHS_PER_TRIAL=20

NUM_WORKERS=6

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${CODEDIR}:${PYTHONPATH:-}"

WORKER0_OUT=${LOGDIR}/${STUDY_NAME}_worker0_${JOBTAG}.out
WORKER0_ERR=${LOGDIR}/${STUDY_NAME}_worker0_${JOBTAG}.err
WORKER1_OUT=${LOGDIR}/${STUDY_NAME}_worker1_${JOBTAG}.out
WORKER1_ERR=${LOGDIR}/${STUDY_NAME}_worker1_${JOBTAG}.err
RETRAIN_OUT=${LOGDIR}/${STUDY_NAME}_retrain_${JOBTAG}.out
RETRAIN_ERR=${LOGDIR}/${STUDY_NAME}_retrain_${JOBTAG}.err

echo "Study: ${STUDY_NAME}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Metric: ${METRIC}"
echo "Trials total: ${TRIALS_TOTAL} (per worker: ${TRIALS_PER_WORKER})"
echo "Worker0 logs: ${WORKER0_OUT} / ${WORKER0_ERR}"
echo "Worker1 logs: ${WORKER1_OUT} / ${WORKER1_ERR}"

# -------------------------
# Optuna workers (2 GPUs, 2 processes)
# -------------------------
CUDA_VISIBLE_DEVICES=0 python -u ${CODEDIR}/tune_amrFold.py \
  --mode tune \
  --train_tsv ${DB_TRAIN} \
  --val_tsv   ${DB_VAL} \
  --test_tsv  ${DB_TEST} \
  --features_lmdb ${FEATURESLMDBDIR} \
  --out_dir ${OUTPUTDIR} \
  --study_name ${STUDY_NAME} \
  --storage ${OPTUNA_STORAGE} \
  --n_trials ${TRIALS_PER_WORKER} \
  --max_epochs ${MAX_EPOCHS_PER_TRIAL} \
  --num_workers ${NUM_WORKERS} \
  --metric ${METRIC} \
  --pruner median \
  --optuna_verbosity info \
  > ${WORKER0_OUT} 2> ${WORKER0_ERR} &

PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u ${CODEDIR}/tune_amrFold.py \
  --mode tune \
  --train_tsv ${DB_TRAIN} \
  --val_tsv   ${DB_VAL} \
  --test_tsv  ${DB_TEST} \
  --features_lmdb ${FEATURESLMDBDIR} \
  --out_dir ${OUTPUTDIR} \
  --study_name ${STUDY_NAME} \
  --storage ${OPTUNA_STORAGE} \
  --n_trials ${TRIALS_PER_WORKER} \
  --max_epochs ${MAX_EPOCHS_PER_TRIAL} \
  --num_workers ${NUM_WORKERS} \
  --metric ${METRIC} \
  --pruner median \
  --optuna_verbosity info \
  > ${WORKER1_OUT} 2> ${WORKER1_ERR} &

PID1=$!

wait ${PID0}
wait ${PID1}

echo "Tuning finished: $(date)"

# Checkpoint and archive DB (persistent + job snapshot)
backup_optuna_db || true
cp -f "${OPTUNA_DB_PERSIST}" "${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db" || true
echo "Saved Optuna DB: ${OPTUNA_DB_PERSIST} and ${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db"

# -------------------------
# Final retrain with best parameters (GPU 0)
# -------------------------
RETRAIN_TAG="retrain_best_${JOBTAG}"
echo "Final retrain (best params) tag=${RETRAIN_TAG} on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python -u ${CODEDIR}/tune_amrFold.py \
  --mode retrain_best \
  --train_tsv ${DB_TRAIN} \
  --val_tsv   ${DB_VAL} \
  --test_tsv  ${DB_TEST} \
  --features_lmdb ${FEATURESLMDBDIR} \
  --out_dir ${OUTPUTDIR} \
  --study_name ${STUDY_NAME} \
  --storage ${OPTUNA_STORAGE} \
  --num_workers ${NUM_WORKERS} \
  --metric ${METRIC} \
  --retrain_metric ${METRIC} \
  --retrain_epochs 40 \
  --retrain_patience 8 \
  --retrain_tag ${RETRAIN_TAG} \
  --optuna_verbosity info \
  > ${RETRAIN_OUT} 2> ${RETRAIN_ERR}

backup_optuna_db || true
cp -f "${OPTUNA_DB_PERSIST}" "${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db" || true

echo "Job finished: $(date)"
echo "Retrain outputs: ${OUTPUTDIR}/${RETRAIN_TAG}"
echo "Retrain logs: ${RETRAIN_OUT} / ${RETRAIN_ERR}"
