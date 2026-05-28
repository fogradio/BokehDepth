#!/usr/bin/env bash
# Inference script that relies only on code inside the BokehDepth repository:
# Stage-1 (bokeh diffusion generation) + Stage-2 (UniDepth DSFA depth estimation).
# Difference from run_pipeline.sh: all config / code paths point inside this
# repository; nothing under /mnt/slurm_home/hwzhang/UniDepth (or any external
# directory) is referenced. Model weights (FLUX base, LoRA, UniDepth DSFA, ...)
# still come from HuggingFace or BokehDepth/weights/.

set -eo pipefail
conda activate bokehdepth

# ---------------------------------------------------------------------------
# In-repository paths (no external code / config dependencies)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="${SCRIPT_DIR}/pipeline.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Unable to locate pipeline.py at ${PY_SCRIPT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Default inference hyper-parameters (all overridable via environment variables)
# ---------------------------------------------------------------------------
REF_IMAGE="${REF_IMAGE:-${SCRIPT_DIR}/examples/ref.png}"
REF_WIDTH="${REF_WIDTH:-512}"
REF_HEIGHT="${REF_HEIGHT:-512}"
K_VALUES="${K_VALUES:-10.0 20.0 30.0}"
SHORT_SIDE="${SHORT_SIDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/examples}"
SEED="${SEED:-42}"

# Stage-1: FLUX base + Bokeh LoRA adapter (weights come from HuggingFace or this repo's weights/ directory)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-black-forest-labs/FLUX.1-Kontext-dev}"
ADAPTER_CKPT="${ADAPTER_CKPT:-${SCRIPT_DIR}/weights/bokeh_lora.bin}"
BLOCK_IDS="${BLOCK_IDS:-0-56}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
UNFREEZE_Q="${UNFREEZE_Q:-1}"
UNFREEZE_K="${UNFREEZE_K:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_STEPS="${NUM_STEPS:-50}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-Set dof_cond = {value:.2f} (stronger background defocus); preserve subject sharpness; keep composition, lighting, and colors unchanged.}"
APPLY_COLOR_TRANSFER="${APPLY_COLOR_TRANSFER:-1}"

# Stage-2: UniDepth DSFA (config and weights both live inside the BokehDepth repository)
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/UniDepth/configs/config_v2_vitl14_DSFA_inference.json}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${SCRIPT_DIR}/weights/UDv2_dsfa_release.pth}"
RESOLUTION_LEVEL="${RESOLUTION_LEVEL:-}"
DEVICE="${DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# Required file existence checks
# ---------------------------------------------------------------------------
for required in "${REF_IMAGE}" "${ADAPTER_CKPT}" "${CONFIG_PATH}" "${WEIGHTS_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Required file not found: ${required}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Assemble pipeline.py command-line arguments
# ---------------------------------------------------------------------------
CMD_ARGS=(
  "--ref-image" "${REF_IMAGE}"
  "--output-root" "${OUTPUT_ROOT}"
  "--seed" "${SEED}"
  "--pretrained-model" "${PRETRAINED_MODEL}"
  "--adapter-ckpt" "${ADAPTER_CKPT}"
  "--block-ids" "${BLOCK_IDS}"
  "--mixed-precision" "${MIXED_PRECISION}"
  "--lora-rank" "${LORA_RANK}"
  "--lora-alpha" "${LORA_ALPHA}"
  "--guidance-scale" "${GUIDANCE_SCALE}"
  "--num-steps" "${NUM_STEPS}"
  "--prompt-template" "${PROMPT_TEMPLATE}"
  "--config" "${CONFIG_PATH}"
  "--weights" "${WEIGHTS_PATH}"
  "--device" "${DEVICE}"
)

if [[ "${APPLY_COLOR_TRANSFER}" == "1" ]]; then
  CMD_ARGS+=("--apply-color-transfer")
fi

if [[ -n "${REF_WIDTH}" ]]; then
  CMD_ARGS+=("--ref-width" "${REF_WIDTH}")
fi
if [[ -n "${REF_HEIGHT}" ]]; then
  CMD_ARGS+=("--ref-height" "${REF_HEIGHT}")
fi
if [[ -n "${SHORT_SIDE}" ]]; then
  CMD_ARGS+=("--short-side" "${SHORT_SIDE}")
fi
if [[ -n "${RESOLUTION_LEVEL}" ]]; then
  CMD_ARGS+=("--resolution-level" "${RESOLUTION_LEVEL}")
fi

if [[ "${UNFREEZE_Q}" == "1" ]]; then
  CMD_ARGS+=("--unfreeze-q")
fi
if [[ "${UNFREEZE_K}" == "1" ]]; then
  CMD_ARGS+=("--unfreeze-k")
fi

read -r -a K_ARRAY <<< "${K_VALUES}"
CMD_ARGS+=("--k-values")
CMD_ARGS+=("${K_ARRAY[@]}")

echo "[INFO] Running inference: ${PYTHON_BIN} ${PY_SCRIPT}" "${CMD_ARGS[@]}"
"${PYTHON_BIN}" "${PY_SCRIPT}" "${CMD_ARGS[@]}"
