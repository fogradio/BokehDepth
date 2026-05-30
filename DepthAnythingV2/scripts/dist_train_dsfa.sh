#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-4}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/dsfa_train.json}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/outputs/train_dsfa}"
PRETRAINED_FROM="${PRETRAINED_FROM:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

CMD_ARGS=(
  "--config" "${CONFIG_PATH}"
  "--save-path" "${SAVE_PATH}"
)

if [[ -n "${MANIFEST_PATH}" ]]; then
  CMD_ARGS+=("--manifest-path" "${MANIFEST_PATH}")
fi

if [[ -n "${PRETRAINED_FROM}" ]]; then
  CMD_ARGS+=("--pretrained-from" "${PRETRAINED_FROM}")
fi

mkdir -p "${SAVE_PATH}"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes="${GPUS}" \
  --mixed_precision="${MIXED_PRECISION}" \
  "${SCRIPT_DIR}/train_dsfa.py" \
  "${CMD_ARGS[@]}"
