#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: acquire_gpu.sh <command> [args...]" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU LOCK] nvidia-smi not found; running command directly." >&2
  exec "$@"
fi

USERNAME="${USER:-$(id -un)}"
LOCK_ROOT="${AMRFOLD_GPU_LOCK_DIR:-/tmp/amrfold_gpu_locks_${USERNAME}_$(hostname)}"
mkdir -p "${LOCK_ROOT}"

mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[GPU LOCK] No GPUs detected; running command directly." >&2
  exec "$@"
fi

cleanup() {
  if [[ -n "${AMRFOLD_GPU_LOCK_FILE:-}" && -f "${AMRFOLD_GPU_LOCK_FILE}" ]]; then
    rm -f "${AMRFOLD_GPU_LOCK_FILE}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "[GPU LOCK] Host: $(hostname)" >&2
echo "[GPU LOCK] Lock root: ${LOCK_ROOT}" >&2
echo "[GPU LOCK] Initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2

while true; do
  for gpu in "${GPU_IDS[@]}"; do
    lock_file="${LOCK_ROOT}/gpu_${gpu}.lock"

    if ( set -o noclobber; echo "$$" > "${lock_file}" ) 2>/dev/null; then
      export AMRFOLD_GPU_LOCK_FILE="${lock_file}"

      export CUDA_VISIBLE_DEVICES="${gpu}"
      export NVIDIA_VISIBLE_DEVICES="${gpu}"
      export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${gpu}"
      export APPTAINERENV_CUDA_VISIBLE_DEVICES="${gpu}"
      export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${gpu}"
      export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${gpu}"

      echo "[GPU LOCK] Acquired GPU ${gpu}" >&2
      echo "[GPU LOCK] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
      echo "[GPU LOCK] Running: $*" >&2

      "$@"
      exit_code=$?

      rm -f "${lock_file}" || true
      exit "${exit_code}"
    fi
  done

  sleep 2
done
