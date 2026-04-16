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
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie
#SBATCH --signal=B:TERM@120

set -euo pipefail

# -------------------------
# Environment
# -------------------------
module load cuda/12.1.1
module load cudnn_for_cuda12/8.9.1

source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

export PYTHONUNBUFFERED=1

echo "Job started: $(date)"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "HOSTNAME: $(hostname)"
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
# Helpers
# -------------------------
safe_rm_rf () {
  local p="${1:-}"
  [[ -z "$p" ]] && return 0
  [[ "$p" == "/" ]] && return 0
  rm -rf --one-file-system "$p" 2>/dev/null || rm -rf "$p" 2>/dev/null || true
}

writable_dir () {
  local d="$1"
  [[ -z "$d" ]] && return 1
  mkdir -p "$d" 2>/dev/null || return 1
  touch "$d/.writetest" 2>/dev/null || return 1
  rm -f "$d/.writetest" 2>/dev/null || true
  return 0
}

free_kib () { df -Pk "$1" 2>/dev/null | tail -1 | awk '{print $4}'; }
size_kib () { du -sk "$1" 2>/dev/null | awk '{print $1}'; }

# -------------------------
# Optuna: fast local DB + persistent checkpointing
# Minimal changes requested:
#   - force TMPDIR under OPTUNA_DB_DIR
#   - delete OPTUNA_DB_DIR on exit
# -------------------------
STUDY_NAME=amrfold_hpo
OPTUNA_DB_PERSIST="${OUTPUTDIR}/${STUDY_NAME}.db"

pick_fast_dir_db () {
  # DB is relatively small, so we can use:
  #   $SLURM_TMPDIR -> /dev/shm -> /localscratch -> /tmp
  for d in "${SLURM_TMPDIR:-}" "/dev/shm/${USER}" "/localscratch/${USER}" "/tmp/${USER}"; do
    [[ -z "$d" ]] && continue
    writable_dir "$d" || continue
    echo "$d"
    return 0
  done
  return 1
}

DB_BASE="$(pick_fast_dir_db)" || { echo "ERROR: No writable fast local directory found for Optuna DB."; exit 1; }
JOBTAG="${SLURM_JOB_ID:-manual}"
OPTUNA_DB_DIR="${DB_BASE}/optuna_${STUDY_NAME}_${JOBTAG}"
mkdir -p "$OPTUNA_DB_DIR"

OPTUNA_DB_LOCAL="${OPTUNA_DB_DIR}/${STUDY_NAME}.db"
OPTUNA_STORAGE="sqlite:////${OPTUNA_DB_LOCAL}"

# Keep SQLite tmp files local too
export SQLITE_TMPDIR="${OPTUNA_DB_DIR}"

# Minimal change: force *all* temp files to a job-scoped tmp we can delete
export TMPDIR="${OPTUNA_DB_DIR}/tmp"
mkdir -p "${TMPDIR}"

echo "Optuna DB local:    ${OPTUNA_DB_LOCAL}"
echo "Optuna DB persist:  ${OPTUNA_DB_PERSIST}"
echo "Optuna DB base:     ${DB_BASE}"
df -T "${OPTUNA_DB_DIR}" | tail -1 || true
df -T "${OUTPUTDIR}" | tail -1 || true

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

# Minimal change: track staged dataset dir globally so we can delete it on exit
STAGE_DIR=""

cleanup () {
  echo "Cleanup/exit handler: $(date)"

  # Stop workers/checkpointer first
  [[ -n "${PID0}" ]] && kill "${PID0}" 2>/dev/null || true
  [[ -n "${PID1}" ]] && kill "${PID1}" 2>/dev/null || true
  [[ -n "${CKPT_PID}" ]] && kill "${CKPT_PID}" 2>/dev/null || true

  echo "Final Optuna DB backup..."
  backup_optuna_db || true

  # Minimal change: delete what we wrote locally on the node
  echo "Removing job-local data:"
  echo "  STAGE_DIR=${STAGE_DIR:-<none>}"
  echo "  OPTUNA_DB_DIR=${OPTUNA_DB_DIR:-<none>}"
  safe_rm_rf "${STAGE_DIR}"
  safe_rm_rf "${OPTUNA_DB_DIR}"
}
trap cleanup EXIT INT TERM

# Periodic DB checkpoint (every 5 min)
(
  while sleep 300; do
    backup_optuna_db >/dev/null 2>&1 || true
  done
) &
CKPT_PID="$!"

# -------------------------
# Stage TSVs + features.lmdb locally, retrying multiple locations.
# Minimal changes requested:
#   - set STAGE_DIR on success so cleanup deletes it
#   - delete stage dir immediately if staging fails (space/copy/etc.)
# -------------------------
STAGE_DATA=1

stage_all_to_base () {
  local base="$1"
  local jobtag="$2"
  local src_lmdb="$3"
  local stage_dir="${base}/amrfold_stage_${jobtag}"
  local dst_lmdb="${stage_dir}/features.lmdb"

  writable_dir "$base" || return 1

  # Create stage dir and register it early (so even partial work is tracked),
  # but remove it on any failure before returning.
  mkdir -p "${stage_dir}/data" || return 1
  STAGE_DIR="${stage_dir}"

  # Stage TSVs (tiny)
  cp -f "${DB_TRAIN}" "${stage_dir}/data/DB_train.tsv" || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
  cp -f "${DB_VAL}"   "${stage_dir}/data/DB_val.tsv"   || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
  cp -f "${DB_TEST}"  "${stage_dir}/data/DB_test.tsv"  || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }

  # Space check for LMDB
  local lmdb_kb free_kb need_kb
  lmdb_kb="$(size_kib "$src_lmdb" || true)"
  free_kb="$(free_kib "$base" || true)"

  if [[ -z "${lmdb_kb}" || -z "${free_kb}" ]]; then
    echo "  Could not determine LMDB size or free space on ${base}"
    safe_rm_rf "${stage_dir}"
    STAGE_DIR=""
    return 4
  fi

  need_kb=$(( lmdb_kb + lmdb_kb / 10 ))  # +10% headroom

  echo "Trying LMDB staging on ${base}"
  echo "  features.lmdb ~ ${lmdb_kb} KiB"
  echo "  free on ${base}: ${free_kb} KiB"
  if (( free_kb < need_kb )); then
    echo "  Not enough space on ${base} (need ~${need_kb} KiB). Removing partial stage dir."
    safe_rm_rf "${stage_dir}"
    STAGE_DIR=""
    return 2
  fi

  echo "  Staging features.lmdb -> ${dst_lmdb}"
  if [[ -d "$src_lmdb" ]]; then
    mkdir -p "${dst_lmdb}" || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --info=progress2 "${src_lmdb}/" "${dst_lmdb}/" || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
    else
      cp -a "$src_lmdb" "$dst_lmdb" || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
    fi
  else
    cp -f "$src_lmdb" "$dst_lmdb" || { safe_rm_rf "${stage_dir}"; STAGE_DIR=""; return 3; }
  fi

  # Repoint to staged paths (only on full success)
  DB_TRAIN="${stage_dir}/data/DB_train.tsv"
  DB_VAL="${stage_dir}/data/DB_val.tsv"
  DB_TEST="${stage_dir}/data/DB_test.tsv"
  FEATURESLMDBDIR="${dst_lmdb}"

  echo "  SUCCESS: staged dataset on ${base}"
  echo "    DB_TRAIN=${DB_TRAIN}"
  echo "    FEATURES=${FEATURESLMDBDIR}"
  df -T "$base" | tail -1 || true
  return 0
}

CANDIDATES=(
  "/localscratch/${USER}"
  "${SLURM_TMPDIR:-}"
  "/dev/shm/${USER}"
  "/tmp/${USER}"
)

if [[ "${STAGE_DATA}" -eq 1 ]]; then
  echo "Staging TSVs + features.lmdb locally"
  STAGED=0
  for base in "${CANDIDATES[@]}"; do
    [[ -z "$base" ]] && continue
    if stage_all_to_base "$base" "$JOBTAG" "$FEATURESLMDBDIR"; then
      STAGED=1
      break
    else
      rc=$?
      echo "  staging attempt on ${base} failed (rc=${rc}), trying next..."
    fi
  done

  if [[ "${STAGED}" -ne 1 ]]; then
    echo "WARNING: Could not stage features.lmdb locally. Using BeeGFS LMDB: ${FEATURESLMDBDIR}"
    # Try at least to stage TSVs locally (still job-scoped, and will be deleted by cleanup)
    for tbase in "/localscratch/${USER}" "${SLURM_TMPDIR:-}" "/tmp/${USER}"; do
      [[ -z "$tbase" ]] && continue
      if writable_dir "$tbase"; then
        STAGE_DIR="${tbase}/amrfold_stage_${JOBTAG}"
        mkdir -p "${STAGE_DIR}/data"
        cp -f "${DB_TRAIN}" "${STAGE_DIR}/data/DB_train.tsv" || { safe_rm_rf "${STAGE_DIR}"; STAGE_DIR=""; break; }
        cp -f "${DB_VAL}"   "${STAGE_DIR}/data/DB_val.tsv"   || { safe_rm_rf "${STAGE_DIR}"; STAGE_DIR=""; break; }
        cp -f "${DB_TEST}"  "${STAGE_DIR}/data/DB_test.tsv"  || { safe_rm_rf "${STAGE_DIR}"; STAGE_DIR=""; break; }
        DB_TRAIN="${STAGE_DIR}/data/DB_train.tsv"
        DB_VAL="${STAGE_DIR}/data/DB_val.tsv"
        DB_TEST="${STAGE_DIR}/data/DB_test.tsv"
        echo "Staged TSVs on ${tbase}: ${STAGE_DIR}/data"
        break
      fi
    done
  fi
fi

# -------------------------
# Run config
# -------------------------
METRIC=f1_class_macro_pos
TRIALS_TOTAL=200
TRIALS_PER_WORKER=$((TRIALS_TOTAL / 2))
MAX_EPOCHS_PER_TRIAL=20
NUM_WORKERS=6

GATE_CLASSES="multidrug,others,sulfonamide,rifamycin,quinolone"
CONFIRM_TOP_K=3
CONFIRM_SEEDS="1,2,3"
RETRAIN_SEEDS="1,2,3,4,5"

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
echo "Dataset:"
echo "  DB_TRAIN=${DB_TRAIN}"
echo "  DB_VAL=${DB_VAL}"
echo "  DB_TEST=${DB_TEST}"
echo "  FEATURES=${FEATURESLMDBDIR}"
echo "Local dirs to delete on exit:"
echo "  STAGE_DIR=${STAGE_DIR:-<none>}"
echo "  OPTUNA_DB_DIR=${OPTUNA_DB_DIR:-<none>}"
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
  --gate_classes ${GATE_CLASSES} \
  --random_crop_train \
  --print_completed_trials \
  --seed 1234 \
  --class_label_smoothing 0.05 \
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
  --gate_classes ${GATE_CLASSES} \
  --random_crop_train \
  --print_completed_trials \
  --seed 4321 \
  --class_label_smoothing 0.05 \
  --optuna_verbosity info \
  > ${WORKER1_OUT} 2> ${WORKER1_ERR} &

PID1=$!

wait ${PID0}
wait ${PID1}

echo "Tuning finished: $(date)"

backup_optuna_db || true
cp -f "${OPTUNA_DB_PERSIST}" "${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db" || true
echo "Saved Optuna DB: ${OPTUNA_DB_PERSIST} and ${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db"

# -------------------------
# Final confirmation + retrain with both GPUs exposed
# -------------------------
RETRAIN_TAG="retrain_best_${JOBTAG}"
echo "Final confirmation + retrain tag=${RETRAIN_TAG} on GPUs 0,1..."

CUDA_VISIBLE_DEVICES=0,1 python -u ${CODEDIR}/tune_amrFold.py \
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
  --gate_classes ${GATE_CLASSES} \
  --confirm_top_k ${CONFIRM_TOP_K} \
  --confirm_seeds ${CONFIRM_SEEDS} \
  --retrain_seeds ${RETRAIN_SEEDS} \
  --random_crop_train \
  --parallel_gpus 0,1 \
  --max_parallel_jobs 2 \
  --class_label_smoothing 0.05 \
  --optuna_verbosity info \
  > ${RETRAIN_OUT} 2> ${RETRAIN_ERR}

backup_optuna_db || true
cp -f "${OPTUNA_DB_PERSIST}" "${OUTPUTDIR}/${STUDY_NAME}_${JOBTAG}.db" || true

echo "Job finished: $(date)"
echo "Retrain outputs: ${OUTPUTDIR}/${RETRAIN_TAG}"
echo "Retrain logs: ${RETRAIN_OUT} / ${RETRAIN_ERR}"
