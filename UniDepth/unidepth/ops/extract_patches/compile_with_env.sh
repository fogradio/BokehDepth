#!/usr/bin/env bash

# ========= 1) Activate conda environment =========
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
CONDA_ENV=${CONDA_ENV:-bokehdepth}
conda activate "${CONDA_ENV}"

# Auto-detect CUDA_HOME: prefer the conda environment's CUDA.
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/nvcc" ]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  echo ">>> Using conda CUDA: ${CUDA_HOME}"
else
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.7}
  echo ">>> Using system CUDA: ${CUDA_HOME}"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
export TORCH_CUDA_VERSION_CHECK=0  # skip CUDA version check


# ========= 2) Library paths =========
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
CUDA_LIB_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_RUNTIME_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
CUDA_CUPTI_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib"
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${CUDA_RUNTIME_PATH}:${CUDA_CUPTI_PATH}:${LD_LIBRARY_PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${CONDA_PREFIX}/lib/python3.10/site-packages:${PYTHONPATH:-}"

# ========= 3) Compiler selection =========
USE_SYSTEM_GCC=${USE_SYSTEM_GCC:-1}

if [[ "${USE_SYSTEM_GCC}" -eq 1 ]]; then
  export CC="/usr/bin/gcc"
  export CXX="/usr/bin/g++"
  export CUDAHOSTCXX="${CXX}"
  echo ">>> Using system GCC: ${CC}"
else
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="${CXX}"
  echo ">>> Using conda GCC: ${CC}"
fi

# Avoid using a sysroot pointing at a newer glibc.
unset CONDA_BUILD_SYSROOT

# ========= 4) Extra compile flags =========
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
fi

echo ">>> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo ">>> Using CC=${CC}"
echo ">>> Using CXX=${CXX}"
echo ">>> CUDA_HOME=${CUDA_HOME}"
echo ">>> nvcc location:"
which nvcc
echo ">>> nvcc version:"
nvcc --version | grep -E "release|Build"
echo ">>> gcc location:"
which gcc
echo ">>> gcc version:"
gcc --version | head -1

# ========= 5) Build & install =========
python setup.py build install
