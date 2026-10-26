# Characterization Experiment Protocol

## Purpose

This file defines how experiments are executed and recorded. It is procedural; research hypotheses live in `RESEARCH_PLAN.md`.

---

## 1. Before every experiment

Read:

- `RESEARCH_PLAN.md`
- `CURRENT_STATE.md`
- latest `EXPERIMENT_LOG.md`

Then record:

```bash
git rev-parse HEAD
git status --short
nvidia-smi
nvcc --version
python --version
```

If possible, also record GPU clocks/power state.

Create:

```text
research/results/<EXP_ID>/
```

Each directory should contain:

```text
metadata.md
commands.sh
stdout/
raw/
summary.json
```

Profiler outputs go under `raw/`.

---

## 2. Naming

Use:

```text
C<NNN>_<short-slug>
```

Examples:

```text
C001_baseline_s4k
C002_noearly_s4k
C015_noearly_qkv_s128k
```

Never reuse an experiment ID.

---

## 3. Baseline preservation

Research switches must default to exact upstream behavior.

Preferred pattern:

```text
--research-prefetch-mode baseline
--research-prefetch-mode after-input
```

or a compile-time/JIT key if runtime control is impractical.

Do not change default production behavior merely to simplify experiments.

---

## 4. Warmup and timing

Unless upstream benchmark semantics require otherwise:

1. Run one untimed sanity pass.
2. Run enough warmup tokens to stabilize JIT/cache state.
3. Run at least 5 independent timed repetitions.
4. Use the same output token count across compared variants.
5. Report all repetitions.

If runtime is expensive at 256K, reduce repetitions only after documenting the reason.

Never report only the best run.

---

## 5. Characterization order

### Step A — Correctness

Authorized mock-first path (2026-10-26): before checkpoint download, run the
existing binding suites and differential synthetic tensor/output checks, with
baseline-versus-baseline repeats to estimate numerical nondeterminism. Disclose
that this is not a substitute for the checkpoint decode-layer reference suite.
Real-prompt/model correctness remains a separate required gate for final claims.

For each new research switch:

- build;
- run relevant tests;
- run one 4K/BS1 benchmark;
- check for hangs, NaNs, invalid memory access, or output drift.

### Step B — Fast sensitivity test

Compare:

- baseline
- all-early-prefetch-disabled

at:

- 4K
- 128K

If the two variants are indistinguishable beyond noise, investigate instrumentation before running the full matrix.

### Step C — Full context sweep

Contexts:

```text
1K 4K 8K 16K 32K 64K 128K 256K
```

Primary BS:

```text
1
```

### Step D — Real weights

Repeat representative points with real checkpoint/routing.

### Step E — Mechanism profiling

Use only selected contexts to reduce profiler cost.

---

## 6. Required per-experiment metadata

`metadata.md` must include:

```markdown
# <EXP_ID>

## Goal
...

## Hypothesis
...

## Git
commit:
dirty files:

## Environment
GPU:
driver:
CUDA:
Python:
checkpoint:
routing:
BS:
context:
output tokens:
prefetch mode:

## Code change
...

## Expected observation
...

## Actual observation
...
```

---

## 7. Summary JSON schema

Use:

```json
{
  "experiment_id": "C001_baseline_s4k",
  "git_commit": "",
  "variant": "baseline",
  "routing": "synthetic",
  "batch_size": 1,
  "context_length": 4096,
  "output_tokens": 512,
  "repetitions": [
    {
      "tpot_ms": null,
      "throughput_tps": null
    }
  ],
  "median_tpot_ms": null,
  "median_throughput_tps": null,
  "notes": ""
}
```

Add fields rather than changing meanings.

---

## 8. Profiling protocol

### Perfetto / sm-profiler

Use the upstream Python decode loop because upstream states profiling is incompatible with `--cpp-decode-runtime`.

Capture:

- baseline short context;
- modified short context;
- baseline long context;
- modified long context.

Annotate important task/opcode intervals if possible.

### Nsight Compute

Use NCU only on a reduced set of experiments because replay/profiling overhead can distort timing.

NCU is for hardware-counter evidence, not latency reporting.

Record:

- command;
- metric set;
- kernel selector;
- NCU version.

Do not compare NCU-profiled wall-clock timing to normal benchmark timing.

---

## 9. What to inspect in code first

Start read-only.

Search for:

```text
prefetch_weight_tiles
wait_input_bars
QKV_PROJ
ROUTER
FFN_UPGATE_ACT
FFN_DOWN
ATTN_DECODE
dispatch_instruction
```

Map:

- which opcodes issue early weight traffic;
- whether prefetch occurs before or after input barrier;
- how many tiles are prefetched;
- whether behavior depends on context bucket;
- whether attention task count/splits change with context.

Write the map to:

`research/results/prefetch-map.md`

before changing behavior.

---

## 10. Minimum viable research controls

Implement in this order:

### R0 — Baseline

No code behavior change.

### R1 — Global no-early-prefetch

Move or gate speculative weight loads so they begin only after required input dependency is ready.

This is a characterization control, not a proposed optimization.

### R2 — Per-opcode control

Enable/disable early prefetch independently for:

- QKV;
- router;
- MoE up/gate;
- MoE down;
- any additional identified op.

### R3 — Optional bounded prefetch

Only if R1/R2 show a meaningful effect.

Examples:

- at most N speculative tasks per SM;
- prefetch only within a limited distance;
- prefetch only outside attention-heavy intervals.

Do not build R3 before R1/R2 data exists.

---

## 11. Interpretation rules

### Positive signal

Example:

```text
4K: baseline 4% faster than no-early
128K: baseline 1% slower than no-early
```

This strongly supports a context-dependent policy.

### Weaker but usable signal

```text
4K: +5%
128K: +1%
256K: ~0%
```

This indicates shrinking bandwidth slack. A conservative admission policy may still have value.

### Negative result

```text
4K: +4%
128K: +4%
256K: +5%
```

No evidence of context-dependent interference. Do not invent a policy.

### Suspicious result

Large performance shifts with:

- different routing distribution;
- different JIT code;
- schedule-bucket transition;
- unstable clocks.

Treat as confounded until isolated.

---

## 12. Decision update

After every group of 2–4 experiments:

1. append experiment entries;
2. update `CURRENT_STATE.md`;
3. explicitly state one of:
   - CONTINUE;
   - NARROW;
   - PIVOT;
   - STOP.

This prevents accumulating experiments without a research decision.
