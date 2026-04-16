#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: acquire_gpu.sh <command> [args...]" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  exec "$@"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  exec "$@"
fi

lock_root="${TMPDIR:-/tmp}/amrfold_gpu_locks"
mkdir -p "$lock_root"

mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  exec "$@"
fi

cleanup() {
  if [[ -n "${_AMRFOLD_GPU_LOCK_FILE:-}" && -f "${_AMRFOLD_GPU_LOCK_FILE}" ]]; then
    rm -f "${_AMRFOLD_GPU_LOCK_FILE}" || true
  fi
}
trap cleanup EXIT INT TERM

while true; do
  for gpu in "${gpu_ids[@]}"; do
    lock_file="${lock_root}/gpu_${gpu}.lock"
    if ( set -o noclobber; : > "$lock_file" ) 2>/dev/null; then
      export _AMRFOLD_GPU_LOCK_FILE="$lock_file"
      export CUDA_VISIBLE_DEVICES="$gpu"
      exec "$@"
    fi
  done
  sleep 2
done
