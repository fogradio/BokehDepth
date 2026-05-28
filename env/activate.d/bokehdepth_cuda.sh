#!/usr/bin/env bash
# Auto-installed by env/install.sh into
#     ${CONDA_PREFIX}/etc/conda/activate.d/bokehdepth_cuda.sh
# so that `conda activate bokehdepth` is enough to run the inference scripts
# without any extra LD_LIBRARY_PATH gymnastics.
#
# The PyTorch +cu128 wheels we install via pip ship the CUDA runtime, cuDNN,
# NCCL, ... as separate nvidia-*-cu12 wheels. Each wheel places its shared
# objects under site-packages/nvidia/<pkg>/lib/. Some downstream libraries
# (xformers, cupy, custom CUDA extensions, ...) look them up through the
# dynamic linker rather than through PyTorch's rpath, so we have to make
# sure those directories are on LD_LIBRARY_PATH before the env is used.

# Resolve the active env's site-packages directory.
_BOKEHDEPTH_SITE_PACKAGES="$(${CONDA_PREFIX}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
_BOKEHDEPTH_NVIDIA_ROOT="${_BOKEHDEPTH_SITE_PACKAGES}/nvidia"

# Build a colon-separated list of every nvidia/<pkg>/lib directory that exists.
_BOKEHDEPTH_CUDA_LIBS=""
if [[ -d "${_BOKEHDEPTH_NVIDIA_ROOT}" ]]; then
  while IFS= read -r _libdir; do
    if [[ -z "${_BOKEHDEPTH_CUDA_LIBS}" ]]; then
      _BOKEHDEPTH_CUDA_LIBS="${_libdir}"
    else
      _BOKEHDEPTH_CUDA_LIBS="${_BOKEHDEPTH_CUDA_LIBS}:${_libdir}"
    fi
  done < <(find "${_BOKEHDEPTH_NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | sort)
fi

# Preserve any previously exported LD_LIBRARY_PATH so deactivate can restore it.
export _BOKEHDEPTH_PREV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Prepend (in order): pip-installed CUDA libs, the conda env lib/, the
# conda env's GCC sysroot lib/. The last one only exists when the gcc_linux-64
# conda packages are installed; it is harmless when missing.
_BOKEHDEPTH_NEW_LD_PATH=""
for _candidate in \
  "${_BOKEHDEPTH_CUDA_LIBS}" \
  "${CONDA_PREFIX}/lib" \
  "${CONDA_PREFIX}/x86_64-conda-linux-gnu/lib"; do
  [[ -z "${_candidate}" ]] && continue
  if [[ -z "${_BOKEHDEPTH_NEW_LD_PATH}" ]]; then
    _BOKEHDEPTH_NEW_LD_PATH="${_candidate}"
  else
    _BOKEHDEPTH_NEW_LD_PATH="${_BOKEHDEPTH_NEW_LD_PATH}:${_candidate}"
  fi
done

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${_BOKEHDEPTH_NEW_LD_PATH}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${_BOKEHDEPTH_NEW_LD_PATH}"
fi

unset _BOKEHDEPTH_SITE_PACKAGES _BOKEHDEPTH_NVIDIA_ROOT _BOKEHDEPTH_CUDA_LIBS \
      _BOKEHDEPTH_NEW_LD_PATH _candidate _libdir
