#!/usr/bin/env bash

# ========= 1) Activate conda environment =========
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
CONDA_ENV=${CONDA_ENV:-bokehdepth}
conda activate "${CONDA_ENV}"

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

# ========= 4) Compile KNN =========
# Architectures supported by CUDA 11.7 (8.9 and 9.0 removed).
export TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6"
# Skip the CUDA version check.
export TORCH_CUDA_VERSION_CHECK=0

echo ">>> Using CC: ${CC}"
echo ">>> Using CXX: ${CXX}"
which gcc
gcc --version

python setup.py build install
