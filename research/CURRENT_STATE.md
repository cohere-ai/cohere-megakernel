# Current Research State

Last updated: 2026-10-26

## Status

**Phase 0 — partial reproduction; CONTINUE after checkpoint access**

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
- No prefetch behavior has been modified. These baselines do not establish
  interference, crossover, correctness of real model outputs, or optimization gain.

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

1. Obtain North Mini Code checkpoint path; run existing decode-layer tests,
   real-prompt sanity and real-weight 8K reproduction to complete G0.
2. Then add true global/per-opcode after-input controls with baseline preserved.
3. Compare short/long regimes with matched seeds, warmed repetitions and
   unchanged tiling/routing/split policy; examine mechanism before any policy.
4. Confirm the GitHub upload destination. Local/server commit-based transport
   works; remote GitHub push awaits destination confirmation after auto-review.

Raw diagnostics and profiler output remain on the server under
`research/results/C000_environment`, `C001_baseline_s8k`,
`C002_baseline_profile_s4k`, and `C003_baseline_s128k`. Only final performance
figures will be retrieved/published. See `CLUSTER.md` for transport commands.

## Open questions

- Which early requests actually overlap KV traffic on the measured schedule?
- How much data is prefetched per task/opcode?
- Does attention split count change at the context boundaries we will sweep?
- Can early prefetch be disabled without materially restructuring the GEMM pipeline?
- Which profiling signal best separates KV traffic from weight traffic?
- Does finite-prefix traffic have enough headroom to matter at token scale?
