# sm-profiler (vendored)

SM-level profiling for the decode megakernel, exported as a Perfetto-compatible
Chrome trace. Adapted from [leepoly/sm-profiler](https://github.com/leepoly/sm-profiler).

Only `sm_profiler.h` and `sm_profiler.cu` are vendored here. Upstream's demos,
Makefile, SASS decoder and Triton/Python bindings are not used by this project.

## What changed

The goal was to cut the cost of a profiler call to the point where it can be
placed around individual megakernel instructions without distorting what it
measures. A call now costs tens of nanoseconds rather than hundreds; see the
comment at the top of `sm_profiler.h` for the measured numbers.

- **Per-warp context.** `SmProfilerCtx` resolves the buffer header, SM id, warp
  slot, counter address and event-array base once per kernel. Each event then
  costs a counter bump and two stores, with none of the index arithmetic
  upstream repeats on every call.
- **Handle-based ranges.** `sm_profiler_start` returns a handle that
  `sm_profiler_end` writes through directly, so ending a range needs no lookup
  or event-number matching.
- **`sm_profiler_swap`.** Ends one range and starts the next on a single
  timestamp read. This removes the visible gap between back-to-back events and
  halves the overhead at instruction transitions, which is the common case in a
  megakernel.
- **L1-friendly loads.** Profiler state is read with
  `ld.global.L1::evict_last`, in an attempt to keep the profiler state in L1 to 
  reduce access latency.
- **Indexing by SM id.** `sm_profiler_init_ctx_by_smid` slots events by SM
  rather than block index, matching a persistent kernel that runs one block per
  SM.
- **External buffer allocation** through `sm_profiler_create_buffer_external`,
  so the caller can place the buffer in memory it already owns.

## Known limitation

Events cannot be measured accurately immediately after `__syncthreads`; the
reported duration comes out shorter than the true one. This is a property of the
hardware timer, not of the changes above.
