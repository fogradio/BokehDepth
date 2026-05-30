#!/bin/bash
# Distributed training launcher for the UniDepth DSFA variant.
#
# Usage:
#   bash UniDepth/scripts/dist_train_DSFA.sh [config-file] [master-port]
#
# Positional arguments (optional):
#   config-file  - Path to a JSON config (default: UniDepth/configs/config_v2_vitl14_DSFA_nyuv2.json)
#   master-port  - Distributed master port (default: 29517)
#
# Environment variables (optional):
#   GPUS                  - Number of GPUs to use (default: 4)
#   SAVE_INTERVAL         - Checkpoint save interval in steps (default: 1000)
#   RESUME_CKPT           - Path to a training checkpoint to resume from
#   CONDA_ENV             - Conda environment name (default: bokehdiff)
#   NYUV2_MANIFEST_PATH   - NYUv2 manifest override; consumed by train_DSFA.py
#   HYPERSIM_MANIFEST_PATHS / HYPERSIM_MANIFEST_PATH - HyperSim overrides

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_CONFIG="${PROJECT_ROOT}/configs/config_v2_vitl14_DSFA_nyuv2.json"
DEFAULT_PORT=29517
DEFAULT_SAVE_INTERVAL=1000

CONFIG_FILE=${1:-$DEFAULT_CONFIG}
MASTER_PORT=${2:-$DEFAULT_PORT}
GPUS=${GPUS:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-$DEFAULT_SAVE_INTERVAL}
RESUME_CKPT=${RESUME_CKPT:-}

echo "========================================="
echo "UniDepth DSFA distributed training"
echo "========================================="
echo "Config file        : ${CONFIG_FILE}"
echo "Master port        : ${MASTER_PORT}"
echo "GPU count          : ${GPUS}"
echo "Checkpoint interval: every ${SAVE_INTERVAL} steps"
if [ -n "${RESUME_CKPT}" ]; then
  echo ">>> Resume checkpoint: ${RESUME_CKPT}"
fi
echo "========================================="

# Export NYUV2_MANIFEST_PATH if the caller set it; train_DSFA.py picks it up.
if [ -n "${NYUV2_MANIFEST_PATH:-}" ]; then
  export NYUV2_MANIFEST_PATH
  echo ">>> NYUV2_MANIFEST_PATH: ${NYUV2_MANIFEST_PATH}"
fi

# =========  Activate conda environment =========
conda activate bokehdepth

# Enable strict undefined-variable checks only after conda activate so that
# conda's own scripts (which sometimes touch unset vars) don't fail us.
set -u


# =========  Distributed environment tweaks =========
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========  Diagnostics =========
echo ">>> Using python: $(which python)"
echo ">>> CONDA_PREFIX : ${CONDA_PREFIX}"
echo ">>> LD_LIBRARY_PATH:" && echo "    ${LD_LIBRARY_PATH}" | tr ':' '
' | sed 's/^/    - /'

torch_python="${CONDA_PREFIX}/bin/python"

${torch_python} - <<'PYINFO'
import sys
print('>>> Python executable:', sys.executable)
try:
    import torch
    print(f'>>> torch: {torch.__version__}, cuda: {torch.version.cuda}')
    print('>>> CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('>>> GPU count:', torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print(f'    GPU {idx}:', torch.cuda.get_device_name(idx))
except Exception as exc:
    print('Torch self-check failed:', exc)
    raise
PYINFO

LOG_DIR=""
if [ -n "${RESUME_CKPT}" ]; then
  if [ -f "${RESUME_CKPT}" ]; then
    LOG_DIR=$(cd "$(dirname "${RESUME_CKPT}")" && pwd)
  else
    echo ">>> WARNING: RESUME_CKPT points to a non-existent file: ${RESUME_CKPT}"
    LOG_DIR=""
  fi
fi

if [ -z "${LOG_DIR}" ]; then
  RUN_ID=$(date +"%Y%m%d_%H%M%S")
  LOG_DIR="${PROJECT_ROOT}/exp/dsfa_${RUN_ID}"
fi

mkdir -p "${LOG_DIR}"

declare -a EXTRA_ARGS
if [ -n "${RESUME_CKPT}" ] && [ -f "${RESUME_CKPT}" ]; then
  EXTRA_ARGS+=("--resume-checkpoint" "${RESUME_CKPT}")
fi

echo ""
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> Starting training..."
echo ">>> Debug: GPUS=${GPUS}, MASTER_PORT=${MASTER_PORT}"
echo ">>> torchrun command: torchrun --nproc_per_node=${GPUS} --master_port=${MASTER_PORT}"
echo ""

torchrun --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
  "${SCRIPT_DIR}/train_DSFA.py" \
  --distributed \
  --config-file "${CONFIG_FILE}" \
  --master-port "${MASTER_PORT}" \
  --fusion-layers 0,1,2,3 \
  --save-interval ${SAVE_INTERVAL} \
  --save-dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/train.log"
