# Cluster workflow

Use `ln301`, with a checkout at `~/hejy/cohere-megakernel`. GPU commands run
inside an existing allocation, never on the login node. Pass the current job ID
explicitly; `281111` was inspected on 2026-10-25 and may expire or change.

```bash
python3 research/scripts/cluster.py push
python3 research/scripts/cluster.py --job 281111 status
python3 research/scripts/cluster.py --job 281111 run -- python --version
python3 research/scripts/cluster.py pull-figures
git push -u origin codex/bandwidth-characterization
```

`push` transfers committed Git history using a bundle and only fast-forwards
the remote branch. It refuses dirty local source. It does not overwrite remote
uncommitted changes. Submodules must be populated separately at their pinned
commits; if using GitHub API tarballs offline, record URL and SHA256 remotely.

Raw installation/build/test logs and profiler traces stay under
`research/results/<experiment-id>/` on the server. They are ignored by Git.
Only reviewed, conclusive performance figures go in `research/figures/`, and
`pull-figures` downloads only PNG/PDF/SVG files. Conclusions and compact metrics
belong in the tracked log/state. Do not upload diagnostic dumps or checkpoints.

The independent Python environment uses the existing read-only FlashMoE
site-packages (Python 3.12, torch 2.5.1+cu121); extra packages are installed only
in `~/hejy/envs/cohere-megakernel`. Freeze package versions for experiments.
`server_env.sh` selects CUDA 12.9 and the project JIT cache. Environment variables
`RESEARCH_CUDA_HOME` and `RESEARCH_PYTHON_ENV` override these paths.

## Driver constraint

Observed: H100 80GB HBM3, driver 550.54.15, CUDA compiler 12.9.86. The
`nvidia-smi` CUDA 12.4 label is driver capability, not installed toolkit version.
CUDA 13 requires a newer driver or a supported forward-compatibility path.
The current NVIDIA matrix does not list R550 as a supported CUDA 13 forward
compatibility target. Use CUDA 12.x first; validate actual SM90a compilation,
module loading, tests, and decode rather than assuming compatibility from the
toolkit label. CUDA 12.9 PTX/new driver APIs can still fail on this driver; a
12.4 fallback must be isolated and recorded if needed.

Sources checked 2026-10-25:
- https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html
