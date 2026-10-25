# Research Plan: Bandwidth-Orchestrated Long-Context Decode Megakernel

## 1. Working title

**Bandwidth-Orchestrated Megakernel for Long-Context Decode**

Alternative neutral title during characterization:

**Characterizing Weight-Prefetch and KV-Traffic Interference in Persistent LLM Decode**

Do not lock the final paper title before the characterization phase succeeds.

---

## 2. Problem statement

Modern decode megakernels remove kernel-launch gaps and overlap work across operator boundaries. A particularly effective mechanism is to prefetch immutable model weights from HBM before activation dependencies are satisfied.

For short-context decode, this is attractive because otherwise-unused memory-bandwidth slack can be used to hide future weight loads.

Long-context decode changes the traffic composition:

- model-weight traffic is approximately context-independent per token;
- KV-cache traffic grows with effective context length;
- attention and future-weight prefetch may therefore become concurrent consumers of HBM bandwidth.

The research question is:

> **Does aggressive cross-task weight prefetch remain beneficial when long-context KV traffic consumes an increasing share of HBM bandwidth, or does it create bandwidth interference that should be explicitly orchestrated?**

This is intentionally different from:

- FMK: program/cell specialization;
- Weave: SM-resource partitioning;
- MegaFlux: inter-GPU expert/work placement;
- generic tile/autotuning work.

The resource of interest here is **HBM bandwidth over time**, not SM ownership.

---

## 3. Source implementation facts

The current upstream Cohere megakernel is a single-H100, BF16, decode-only serving engine for North Mini Code.

Relevant upstream files:

- `src/decode/megakernel.cuh`
  - persistent controller/worker pipeline;
  - GEMMs, paged attention, MoE;
  - producer warp can issue weight-tile loads before cross-SM activation waits.
- `src/decode/schedule.py`
  - task schedule construction and descriptor generation.
- `src/decode/runtime.cu`
  - host decode runtime.
- `src/runner.py`
  - standalone benchmark and profiling entry point.
- `src/sm-profiler/`
  - per-SM tracing support.

The release supports context lengths up to 256K and batch sizes 1–8. The main characterization target is BS=1.

Important implementation detail motivating this study:

`prefetch_weight_tiles(...)` can execute before `wait_input_bars(...)`, allowing a task blocked on activation readiness to still consume HBM bandwidth.

---

## 4. Research hypothesis

### H1 — Interference hypothesis

As context length increases, attention KV traffic increasingly overlaps with speculative/cross-task weight prefetch. At some regimes, aggressive prefetch causes measurable contention:

- attention duration increases;
- achieved bandwidth becomes less usefully allocated;
- critical-path decode latency stops improving or regresses.

### H2 — Slack hypothesis

Weight prefetch is not uniformly harmful. There exist periods where HBM bandwidth is underutilized and prefetch remains beneficial.

Therefore the useful optimization is not "disable prefetch", but:

> **admit prefetch only when it consumes bandwidth slack without delaying more critical traffic.**

### H3 — Regime hypothesis

The effect is workload-dependent and should be strongest in:

- BS=1;
- long-context decode;
- attention phases with substantial KV traffic;
- task transitions where prefetch currently crosses activation dependencies.

Short-context decode is expected to be the control regime where aggressive prefetch remains beneficial.

These are hypotheses, not conclusions.

---

## 5. Intended contribution if characterization succeeds

### Design A: Phase-Aware Bandwidth Pressure Estimation

Represent decode execution as phases/tasks with different memory-pressure profiles.

Estimate whether the current execution window is:

- KV-pressure dominated;
- weight-streaming dominated;
- bandwidth-slack.

The first implementation should be deliberately simple. Candidate signals, in increasing complexity:

1. context-length bucket;
2. current attention split count / KV work size;
3. static expected byte count for active attention work;
4. lightweight runtime progress signal from attention queue.

Do **not** begin with a learned predictor.

### Design B: Criticality-Aware Prefetch Admission

Use the pressure estimate to control speculative future-weight traffic.

Possible actions:

- allow normal early prefetch;
- delay prefetch until closer to activation readiness;
- restrict the number of concurrent speculative prefetching tasks;
- prioritize weights on the critical path;
- adapt prefetch distance.

The conceptual contribution is **bandwidth admission**, not merely a context-dependent constant.

The simplest successful policy is preferred.

---

## 6. Characterization comes before Design A/B

The characterization phase must determine whether the premise is real.

We require evidence for all three:

### E1. Context-dependent effect

The performance delta between aggressive and restricted prefetch changes materially with context length.

### E2. Mechanistic evidence

The delta correlates with evidence of memory-system contention or altered attention/weight-load overlap, not only unexplained timing noise.

### E3. Recoverable slack

A policy between "always early" and "never early" can plausibly recover short-context benefit without long-context interference.

If E1 or E2 fails, do not continue to the proposed design.

---

## 7. Evidence gates

### Gate G0 — Reproduction

Before changing prefetch behavior:

- reproduce upstream BS=1 decode performance within a reasonable range;
- pass relevant tests;
- obtain a valid profiler trace.

If G0 fails, stop and debug the environment.

### Gate G1 — Causal prefetch sensitivity

Create controlled variants:

- `BASELINE`: upstream behavior;
- `NO_EARLY_PREFETCH`: no speculative prefetch before activation dependency;
- `SELECTIVE_OFF`: disable early prefetch for one candidate opcode/category at a time.

A useful signal requires a stable latency/TPOT difference larger than run-to-run noise for at least one context regime.

### Gate G2 — Long-context crossover or divergence

A strong result is one of:

- early prefetch helps short context but hurts long context;
- early prefetch remains positive but its benefit collapses with context;
- different opcodes have opposite behavior as context grows.

A perfectly flat effect across all contexts weakens the idea substantially.

### Gate G3 — Mechanism

At least one profiler-level observation should support bandwidth interference, for example:

- increased attention task duration when speculative weight traffic overlaps;
- changed HBM-throughput timeline;
- reduced useful overlap despite similar SM activity;
- queue/barrier waiting shifts consistent with attention being delayed.

Do not infer mechanism from TPOT alone.

### Gate G4 — Prototype headroom

A simple intermediate policy should beat both extremes in at least one meaningful long-context regime while avoiding a short-context regression.

Only after G4 should the project transition from characterization to optimization.

---

## 8. Kill criteria

Stop or pivot if any of the following holds after careful measurement:

1. Disabling/delaying prefetch changes TPOT by less than measurement noise across the full context sweep.
2. Prefetch remains consistently beneficial with no meaningful long-context degradation and no measurable interference.
3. Long-context slowdown is dominated by another factor that cannot be influenced locally.
4. Required instrumentation changes perturb the schedule enough to invalidate measurements.
5. The effect exists only under synthetic routing but disappears with real weights/real routing.

A negative result is valuable if documented cleanly.

---

## 9. Primary workload

### Hardware

- 1× NVIDIA H100.
- Keep clocks/power settings stable if cluster policy permits.
- Record exact driver/CUDA versions.

### Model

- North Mini Code checkpoint.
- BF16.
- Real weights are required for the final characterization.
- Synthetic/uniform routing is useful only for controlled decomposition.

### Primary regime

- Batch size: **1**
- Context lengths:
  - 1K
  - 4K
  - 8K
  - 16K
  - 32K
  - 64K
  - 128K
  - 256K
- Decode output length:
  - use enough tokens for stable steady-state timing, initially 512 or 1K if memory/runtime permits.

### Secondary controls

- BS=2 and BS=4 at a reduced context set if time permits.
- Uniform-routing synthetic mode to isolate MoE variability.
- Real-routing mode for final claims.

---

## 10. Metrics

### Primary

- TPOT / decode throughput.
- Median across repeated runs.
- Dispersion: min/max or p25/p75; preferably standard deviation too.

### Timeline / mechanism

Collect whatever is available without excessive perturbation:

- per-SM task durations;
- attention task duration;
- GEMM/prefetch task duration;
- barrier waiting;
- attention queue progress;
- overlap intervals between attention and speculative weight prefetch.

### Hardware counters, if practical

Use Nsight Compute selectively; do not profile every sweep point.

Candidate metrics:

- DRAM bytes read/write;
- DRAM throughput / utilization;
- L2 bytes / hit behavior;
- memory-pipeline utilization;
- SM active/occupancy only as supporting metrics.

Exact metric names may vary by Nsight Compute version. Inspect available metrics rather than hard-coding unsupported names.

### Correctness

Characterization variants must not change numerical outputs except for nondeterminism already present upstream.

At minimum:

- existing decode-layer tests;
- one real-prompt sanity run;
- compare generated token/output behavior where applicable.

---

## 11. Experimental phases

### Phase 0 — Environment and upstream reproduction

Goal:
- build upstream;
- pass tests;
- reproduce BS=1 8K ballpark;
- produce a Perfetto trace.

No research modification.

### Phase 1 — Locate and classify early prefetch sites

Goal:
- identify every early-weight-prefetch path;
- classify by opcode:
  - QKV;
  - router;
  - MoE up/gate;
  - MoE down;
  - other GEMMs.

Deliverable:
`research/results/prefetch-map.md`

No performance claims yet.

### Phase 2 — Add minimally invasive research controls

Add switches that preserve baseline by default.

Desired controls:

- global enable/disable early-prefetch;
- per-opcode enable/disable;
- optional coarse prefetch timing mode:
  - `EARLY`
  - `AFTER_INPUT_READY`

Avoid implementing dynamic policy yet.

### Phase 3 — Context sweep

Run baseline and controlled variants across context lengths.

First use synthetic mode for speed and variance control.
Then repeat important points using real weights.

Question:
Does the relative value of early prefetch change with S?

### Phase 4 — Mechanism profiling

Choose only representative points, e.g.:

- 4K: short;
- 32K: medium;
- 128K: long;
- 256K: extreme.

Profile baseline vs one or two controlled variants.

Question:
Where does the lost/gained time move?

### Phase 5 — Per-opcode attribution

Disable early prefetch one category at a time.

Goal:
identify whether interference comes primarily from:

- QKV;
- router;
- MoE down;
- another path.

This phase is critical for keeping the eventual mechanism small.

### Phase 6 — Decision

Classify the result:

#### Outcome A — Strong crossover

Short S: early prefetch helps.
Long S: early prefetch hurts.

Proceed to adaptive bandwidth admission.

#### Outcome B — Benefit decays but never becomes negative

Still viable:
optimize prefetch aggressiveness to preserve diminishing slack.

#### Outcome C — No S dependence

Stop this idea unless profiling reveals a different clear mechanism.

#### Outcome D — Only one opcode causes the issue

Narrow the paper:
focus on criticality-aware prefetch for that producer/consumer chain.

### Phase 7 — Minimal prototype

Only after Gate G3.

Implement the simplest policy justified by the data.

Do not over-engineer.

---

## 12. Baseline benchmark commands

Follow upstream `BUILD_AND_RUN.md`.

Illustrative uniform-routing decode benchmark:

```bash
python src/runner.py --mk --fast \
  --lib-path "$PWD/build/libmk_release.so" \
  --batch-size 1 \
  --max-new-tokens 512 \
  --fake-prompt-len 8192 \
  --num-sms 132 \
  --frac-vram-utilization 0.9 \
  --cpp-decode-runtime
```

Real expert traffic:

```bash
python src/runner.py --mk --fast --real-weight \
  --checkpoint <checkpoint-path> \
  --lib-path "$PWD/build/libmk_release.so" \
  --batch-size 1 \
  --max-new-tokens 512 \
  --fake-prompt-len 8192 \
  --num-sms 132 \
  --frac-vram-utilization 0.9 \
  --cpp-decode-runtime
```

Perfetto profiling uses the Python decode loop, not `--cpp-decode-runtime`:

```bash
mkdir -p dump
python src/runner.py \
  --mk --fast \
  --lib-path "$PWD/build/libmk_release.so" \
  --fake-prompt-len 4096 \
  --batch-size 1 \
  --max-new-tokens 512 \
  --profile dump/profile-seq4k-bs1.json
```

Always re-check upstream arguments in the local checkout before relying on these examples.

---

## 13. Initial experiment matrix

Run at least 5 timed repeats after warmup.

| Exp | Routing | Prefetch | BS | Context | Purpose |
|---|---|---|---:|---:|---|
| C01 | synthetic | baseline | 1 | 4K | short control |
| C02 | synthetic | no-early | 1 | 4K | short sensitivity |
| C03 | synthetic | baseline | 1 | 32K | medium |
| C04 | synthetic | no-early | 1 | 32K | medium sensitivity |
| C05 | synthetic | baseline | 1 | 128K | long |
| C06 | synthetic | no-early | 1 | 128K | long sensitivity |
| C07 | synthetic | baseline | 1 | 256K | extreme |
| C08 | synthetic | no-early | 1 | 256K | extreme sensitivity |
| C09 | real | baseline | 1 | 8K | reproduce upstream |
| C10 | real | no-early | 1 | 8K | real-routing sensitivity |
| C11 | real | baseline | 1 | 128K | long real routing |
| C12 | real | no-early | 1 | 128K | long real routing |

Do not immediately run the entire matrix if C01/C02 fail basic correctness.

---

## 14. Analysis quantities

For each context S:

\[
\Delta_{\text{prefetch}}(S)
=
\frac{TPOT_{\text{no-early}}(S)-TPOT_{\text{baseline}}(S)}
     {TPOT_{\text{baseline}}(S)}
\]

Interpretation:

- positive: early prefetch is beneficial;
- approximately zero: early prefetch has little effect;
- negative: early prefetch is harmful.

The important result is not a single value but the **shape of**
\(\Delta_{\text{prefetch}}(S)\) versus context length.

Also compute normalized throughput:

\[
R(S)=
\frac{\text{throughput variant}(S)}
     {\text{throughput baseline}(S)}
\]

Plotting can happen later; raw CSV/JSON should be saved first.

---

## 15. Threats to validity

Actively test these alternatives:

- routing changes rather than bandwidth contention;
- profiler perturbation;
- thermal/clock drift;
- KV-cache allocation effects;
- context bucket changes causing schedule changes;
- attention split-count changes;
- synthetic vs real expert distributions;
- different number of generated tokens;
- compilation/JIT differences between variants.

A context-dependent performance change is not automatically bandwidth interference.

---

## 16. Paper narrative if successful

Potential narrative:

1. Decode megakernels aggressively exploit idle HBM periods by prefetching immutable weights.
2. Long-context agent workloads fundamentally change memory-pressure composition because KV traffic grows with context.
3. Existing schedulers reason about task readiness and SM utilization, but not explicitly about **which ready memory traffic should consume HBM bandwidth now**.
4. We characterize a context-dependent prefetch/attention interference.
5. We introduce lightweight bandwidth-pressure awareness.
6. We use it to admit/prioritize speculative prefetch without changing the megakernel programming model.

Keep the claim narrow:
**long-context, low-batch decode on a persistent megakernel**.

Do not claim general GPU scheduling unless the evidence supports it.
