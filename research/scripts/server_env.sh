#!/usr/bin/env bash
# Source inside a Slurm allocation. Override paths without editing tracked code.
export CUDA_HOME="${RESEARCH_CUDA_HOME:-$HOME/hejy/toolchains/cuda-12.9}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export NVCC="$CUDACXX"
export PATH="${RESEARCH_PYTHON_ENV:-$HOME/hejy/envs/cohere-megakernel}/bin:$CUDA_HOME/bin:$HOME/hejy/cmake-3.29.0-linux-x86_64/bin:$HOME/hejy/miniconda3/envs/flashmoe/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LITTLE_JIT_CACHE_DIR="$PWD/.research-cache/jit"
