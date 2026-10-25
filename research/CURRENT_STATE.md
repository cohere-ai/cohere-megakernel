# Current Research State

Last updated: INITIAL

## Status

**Phase 0 — upstream reproduction**

No research hypothesis has been confirmed.

## Established repository facts

- Target implementation: Cohere North Mini Code decode megakernel.
- Primary hardware: single H100 / SM90a.
- Primary regime: BS=1 long-context decode.
- Upstream supports context up to 256K.
- Upstream decode megakernel uses early immutable-weight prefetch in some GEMM paths.
- Upstream profiling support includes per-SM/Perfetto traces.

## Current hypothesis

**Unverified:** long-context KV-cache traffic may contend with early future-weight prefetch for HBM bandwidth.

## Current decision

Proceed only with characterization.

Do not implement an adaptive bandwidth policy yet.

## Next actions

1. Build upstream.
2. Run existing tests.
3. Reproduce BS=1 / 8K decode.
4. Produce one valid profiler trace.
5. Create `research/results/prefetch-map.md`.
6. Add minimally invasive research controls for early prefetch.

## Open questions

- Which exact opcodes currently prefetch before activation readiness?
- How much data is prefetched per task/opcode?
- Does attention split count change at the context boundaries we will sweep?
- Can early prefetch be disabled without materially restructuring the GEMM pipeline?
- Which profiling signal best separates KV traffic from weight traffic?
