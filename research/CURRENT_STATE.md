# Current Research State

Last updated: 2026-10-26

## Status

**Synthetic G1 sensitivity observed; CONTINUE characterization, no adaptive policy**

No research hypothesis has been confirmed.

## Observed environment and baseline

- `ln301`, checkout `~/hejy/cohere-megakernel`, existing job `281111` on `hn43`.
  Job ID is an execution parameter, not a permanent resource identifier.
- One H100 80GB HBM3, driver 550.54.15. CUDA 12.9.86 successfully built
  upstream release libraries and JIT SM90a kernel and executed synthetic decode.
- Independent project environment: Python 3.12.12 / torch 2.5.1+cu121,
  referencing existing FlashMoE packages read-only. No driver changes.
- Token callback tests: 15/15 passed. KV binding tests: 15/15 passed.
- Synthetic BS1, 512 requested tokens / 511 decode steps, five independent
  repetitions with a full same-process warmup before each timed run:
  - 8K: median TPOT 3.373669 ms; min–max 3.373215–3.374736 ms;
    sample SD 0.000634 ms; median throughput 296.413 tok/s.
  - 128K: median TPOT 4.460984 ms; min–max 4.459497–4.477135 ms;
    sample SD 0.007390 ms; median throughput 224.166 tok/s.
- 4K baseline Perfetto trace parsed: 32,246 positive-duration complete events.
  Same-state instrumented CUDA timing was +6.72% versus uninstrumented timing.
- These initial baselines do not establish interference, crossover, correctness
  of real model outputs, or optimization gain.

## Observed mock-control results (2026-10-26)

- Optional global/per-opcode input-wait controls are implemented, defaulting to
  upstream behavior. Build and both binding suites passed again (C004).
- Exact random-fixture routing checks were not stable even between baselines
  (C005/C007). Failures and snapshots are retained on the server. Stable-router
  weights are used only in a directed correctness fixture, not in timing.
- Shared-fixture validation C010 passed 56/56 checks across baseline and five
  control modes: selected expert sets/token exact; finite floating outputs
  within the predeclared baseline-repeat-noise threshold. Top-k slot order is
  not treated as a semantic invariant. Real checkpoint tests remain unrun.
- C011 used original random weights/KV, six AB/BA pairs each at 4K/128K,
  with full same-process warmup and 511 measured steps. All 12 paired fixture
  fingerprints matched. Baseline/after-input median TPOT:
  - 4K: 3.350752 / 3.478339 ms; median paired savings 128.10 µs (3.8237%).
  - 128K: 4.458137 / 4.584589 ms; savings 123.59 µs (2.7725%).
- Every C011 pair favors early prefetch. Absolute savings remain similar;
  declining relative percentage alone is not evidence of bandwidth interference.
- C012 QKV-only control (six pairs per length) gives median paired benefit
  0.7175% at 4K and1.0926% at 128K, with substantially larger noise than C011.
  One 4K pair reverses sign. It does not identify a long-context harmful QKV path.
- G1 sensitivity is observed in synthetic data. G2/G3/G4 and G0-R remain open;
  H1 bandwidth interference is unconfirmed. No final policy is implemented.

## Established repository facts

- Target implementation: Cohere North Mini Code decode megakernel.
- Primary hardware: single H100 / SM90a.
- Primary regime: BS=1 long-context decode.
- Upstream supports context up to 256K.
- Upstream decode megakernel uses early immutable-weight prefetch in some GEMM paths.
- Upstream profiling support includes per-SM/Perfetto traces.
- At BS1, zero explicit prefetch stages does not eliminate dynamic MoE's first
  early weight load. Static and dynamic paths require distinct controls; see
  `results/prefetch-map.md`.
- Upstream displayed throughput includes a bootstrap token. Research metrics
  use actual decode steps and exclude it; C++ per-step timing is wall/n,
  not a distribution of individually measured token latencies.

## Current hypothesis

**Unverified:** long-context KV-cache traffic may contend with early future-weight prefetch for HBM bandwidth.

## Current decision

Proceed only with characterization.

Do not implement an adaptive bandwidth policy yet.

## Next actions

1. Complete context coverage and remaining per-opcode attribution, retaining
   absolute microseconds as well as relative deltas. Investigate QKV timing
   dispersion before making fine-grained claims; do not discard outliers.
2. Download/access North Mini Code checkpoint later; existing decode-layer tests,
   real-prompt sanity and real-weight 8K reproduction remain required for G0-R.
3. Capture selected matched mechanism profiles after attribution; separate KV
   attention duration changes from loss of latency hiding and split transitions.
4. GitHub destination confirmed by user: `JingyiHe268/cohere-megakernel`;
   research branch pushed successfully. Keep raw diagnostics on server.

Raw diagnostics and profiler output remain on the server under
`research/results/<experiment-id>/`, including failures C005/C007/C009. Only
reviewed performance figures are published in `research/figures/` (C011).
See `EXPERIMENT_LOG.md` for all repeats and `CLUSTER.md` for transport commands.

## Open questions

- Which early requests actually overlap KV traffic on the measured schedule?
- How much data is prefetched per task/opcode?
- Does attention split count change at the context boundaries we will sweep?
- Can early prefetch be disabled without materially restructuring the GEMM pipeline?
- Which profiling signal best separates KV traffic from weight traffic?
- Does finite-prefix traffic have enough headroom to matter at token scale?
