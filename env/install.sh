#!/usr/bin/env bash
# One-shot installer for the BokehDepth runtime environment.
# Creates / updates the `bokehdepth` conda environment, installs every pip
# requirement and copies the activate.d / deactivate.d hooks so that a
# plain `conda activate bokehdepth` is enough to run env/run_inference.sh
# without any extra LD_LIBRARY_PATH patches.
#
# Usage:
#     bash env/install.sh                # default env name: bokehdepth
#     CONDA_ENV=myenv bash env/install.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-bokehdepth}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate conda.
if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[install] ERROR: cannot find conda. Install miniconda/anaconda first." >&2
  exit 1
fi

# Create or update the env from environment.yml.
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[install] updating existing conda env '${CONDA_ENV}' from env/environment.yml"
  conda env update -n "${CONDA_ENV}" -f "${SCRIPT_DIR}/environment.yml" --prune
else
  echo "[install] creating conda env '${CONDA_ENV}' from env/environment.yml"
  # environment.yml hard-codes the env name; honour --name to override.
  conda env create -n "${CONDA_ENV}" -f "${SCRIPT_DIR}/environment.yml"
fi

conda activate "${CONDA_ENV}"
ENV_PREFIX="${CONDA_PREFIX}"

# Install pip packages (CUDA wheels included).
echo "[install] installing pip requirements (this pulls in PyTorch 2.8 + CUDA 12.8 wheels)"
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Copy activate.d / deactivate.d hooks so the env auto-configures LD_LIBRARY_PATH.
echo "[install] wiring activate.d / deactivate.d hooks into ${ENV_PREFIX}"
mkdir -p "${ENV_PREFIX}/etc/conda/activate.d" "${ENV_PREFIX}/etc/conda/deactivate.d"
install -m 0755 "${SCRIPT_DIR}/activate.d/bokehdepth_cuda.sh"   "${ENV_PREFIX}/etc/conda/activate.d/bokehdepth_cuda.sh"
install -m 0755 "${SCRIPT_DIR}/deactivate.d/bokehdepth_cuda.sh" "${ENV_PREFIX}/etc/conda/deactivate.d/bokehdepth_cuda.sh"

# Re-activate to pick up the hook for the smoke test below.
conda deactivate
conda activate "${CONDA_ENV}"

# Smoke test: confirm CUDA is visible.
echo "[install] verifying torch + CUDA inside '${CONDA_ENV}'"
python - <<'PY'
import torch
print(f"  torch     : {torch.__version__}")
print(f"  CUDA build: {torch.version.cuda}")
print(f"  cuDNN     : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0  : {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "[install] done. To run the inference pipeline:"
echo "    conda activate ${CONDA_ENV}"
echo "    bash ${REPO_DIR}/run_inference.sh"
