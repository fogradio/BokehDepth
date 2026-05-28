#!/usr/bin/env bash
# Auto-installed by env/install.sh into
#     ${CONDA_PREFIX}/etc/conda/deactivate.d/bokehdepth_cuda.sh
# Restores LD_LIBRARY_PATH to whatever it was before the matching
# activate.d hook prepended the BokehDepth CUDA paths.

if [[ -n "${_BOKEHDEPTH_PREV_LD_LIBRARY_PATH+x}" ]]; then
  if [[ -z "${_BOKEHDEPTH_PREV_LD_LIBRARY_PATH}" ]]; then
    unset LD_LIBRARY_PATH
  else
    export LD_LIBRARY_PATH="${_BOKEHDEPTH_PREV_LD_LIBRARY_PATH}"
  fi
  unset _BOKEHDEPTH_PREV_LD_LIBRARY_PATH
fi
