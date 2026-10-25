# Experiment Log

Append entries chronologically. Do not rewrite old observations to match later conclusions.

---

### C000_environment — fixed-driver upstream build and sanity

**Date:** 2026-10-25 (Asia/Shanghai)
**Phase:** 0
**Commit:** `7447a889947dbf70773d7c4c7b87aa209398de03`, clean source;
upstream kernel unchanged from `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`.

**Hypothesis:** upstream can build/load/run on the fixed H100 driver using an
available CUDA 12.x toolkit. This does not test bandwidth interference.

**Configuration:** `ln301` → job 281111 → hn43; H100 80GB HBM3;
driver 550.54.15; nvcc 12.9.86; Python 3.12.12; torch 2.5.1+cu121;
no clock/power changes. New environment references existing packages read-only.
Pinned dependency tarballs transferred offline; URL/commit/SHA256 saved remotely.

**Commands:** inside the allocation, source `research/scripts/server_env.sh`;
`cmake -S . -B build -G Ninja -DPython_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release`;
`cmake --build build -j 8`; `python src/tests/test_token_callback.py`;
`python src/tests/test_kv_bindings.py`;
`python src/runner.py --mk --fast --lib-path "$PWD/build/libmk_release.so" --batch-size 1 --max-new-tokens 64 --fake-prompt-len 4096 --num-sms 132 --frac-vram-utilization 0.9 --cpp-decode-runtime`.
Complete setup commands: server artifact `commands.sh`.

**Raw artifacts:** server `~/hejy/cohere-megakernel/research/results/C000_environment/`:
`metadata.md`, `commands.sh`, `summary.json`, `stdout/{environment,configure,build,test-token-callback,test-kv-bindings,baseline-s4k-sanity,install-torch}.log`,
`raw/dependency-{sources.json,sha256.txt}`.

**Observed:** initial PyTorch index installation failed DNS (failure retained).
Fallback torch CUDA tensor operation succeeded. Build succeeded; both binding
suites passed 15/15; synthetic S4096/BS1/63 decode steps completed at
3.492684 ms/step. This sanity was not process-locally warmed and is not a
repeat-based performance result. No checkpoint located in checked model/cache
directories; real decode-layer tests and real-prompt sanity remain NOT RUN.

**Derived:** available CUDA 12.9 is viable for the exercised build/JIT/decode
paths with this fixed driver. This does not validate every CUDA API or real model.

**Interpretation:** no interference inference.

**Decision:** KEEP environment; full G0 INCONCLUSIVE pending real checkpoint.

**Next:** warmed synthetic baseline, profiler sanity, then checkpoint validation.

### C001_baseline_s8k — warmed synthetic short baseline

**Date:** 2026-10-25
**Phase:** 0
**Commit:** `769074e861c0f33f3ee52c23ed4745067d17bed8`, clean source.

**Hypothesis:** unmodified synthetic decode has a reproducible timing baseline.
**Change/config:** no kernel changes; same hardware/software as C000, BS1,
S8192, synthetic weights/KV, emergent expert routing (no forced uniform routing),
seed 42; 512 requested tokens = 511 measured decode steps. Five fresh workers,
each runs a full same-process warmup followed by timed decode with reset seed.

**Command:** `python3 research/scripts/cluster.py --job 281111 run -- python research/scripts/baseline.py --id C001_baseline_s8k --context 8192 --tokens 512 --repeats 5`.

**Raw artifacts:** server `research/results/C001_baseline_s8k/`:
`metadata.md`, `commands.sh`, `summary.json`, `stdout/environment.log`,
`stdout/repeat-{01..05}.log` (includes warmup and timed outputs).

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| TPOT ms | 3.373669 | 3.373499 | 3.374373 | 3.374736 | 3.373215 | 3.373669 |
| tok/s | 296.4132 | 296.4282 | 296.3513 | 296.3195 | 296.4531 | 296.4132 |

**Observed:** all five runs completed. No manually controlled clocks.
**Derived:** sample SD 0.000634 ms; p25/p75 3.373357/3.374554 ms
(Python statistics exclusive quartiles). Native decode metrics exclude bootstrap;
the upstream display includes it. C++ step samples are wall/n, not individual
latencies, so dispersion is computed across repetitions only.
**Interpretation:** useful synthetic baseline; no comparison to real-weight
upstream throughput, no causal interference evidence.
**Decision:** KEEP baseline; G0 still partial. **Next:** real-weight reproduction.

### C002_baseline_profile_s4k — compact trace usability

**Date:** 2026-10-25
**Phase:** 0
**Commit:** `769074e861c0f33f3ee52c23ed4745067d17bed8`, clean source.

**Hypothesis:** upstream instrumentation yields a parseable task timeline.
**Config:** unchanged kernel; synthetic weights/KV, emergent routing, BS1/S4096,
64 requested output tokens, Python loop; normal and instrumented kernel warmups
and same-state CUDA calibration are performed by upstream.

**Command:** `python src/runner.py --mk --fast --lib-path "$PWD/build/libmk_release.so" --batch-size 1 --max-new-tokens 64 --fake-prompt-len 4096 --num-sms 132 --frac-vram-utilization 0.9 --profile research/results/C002_baseline_profile_s4k/raw/trace.json`.
Executed in job 281111 with `server_env.sh`.

**Artifacts:** server `research/results/C002_baseline_profile_s4k/`:
`metadata.md`, `commands.sh`, `summary.json`, `stdout/profile.log`,
`raw/{trace.json,commit.txt,dirty.txt,gpu.txt,nvcc.txt}`.

**Observed:** JSON parsed; 32,246 complete events, all with positive durations,
including attention/QKV/router/MoE ranges. Same-state CUDA baseline 3.327104 ms;
profiled 3.550560 ms; trace span 3.523712 ms.
**Derived:** instrumentation overhead +6.72% for this calibration.
**Interpretation:** task trace usable, but task intervals do not directly identify
HBM requests; no mechanistic inference or statistical latency claim.
**Decision:** KEEP profiler path; G3 INCONCLUSIVE. **Next:** matched variants only
after completing G0 and correctness validation.

### C003_baseline_s128k — long-context environment and noise control

**Date:** 2026-10-25
**Phase:** 0
**Commit:** `769074e861c0f33f3ee52c23ed4745067d17bed8`, clean source.

**Hypothesis:** unmodified synthetic decode executes at 128K and yields a usable
repeat baseline under the fixed environment.
**Config:** same as C001 except S131072; five fresh workers, one full
same-process warmup before each measured 511-step decode. No concurrent profile.

**Command:** `python3 research/scripts/cluster.py --job 281111 run -- python research/scripts/baseline.py --id C003_baseline_s128k --context 131072 --tokens 512 --repeats 5`.

**Artifacts:** server `research/results/C003_baseline_s128k/`:
`metadata.md`, `commands.sh`, `summary.json`, `stdout/environment.log`,
`stdout/repeat-{01..05}.log`.

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| TPOT ms | 4.477135 | 4.459497 | 4.460257 | 4.466532 | 4.460984 | 4.460984 |
| tok/s | 223.3571 | 224.2405 | 224.2023 | 223.8874 | 224.1658 | 224.1658 |

**Observed:** all five runs completed; all repetitions retained.
**Derived:** sample SD 0.007390 ms; p25/p75 4.459877/4.471833 ms
(exclusive quartiles). Timing variation exceeds the 8K baseline's variation.
**Interpretation:** 128K is feasible in this synthetic environment. Increasing
baseline latency with context alone does not establish prefetch interference.
**Decision:** KEEP long baseline; CONTINUE reproduction, no adaptive policy.
**Next:** obtain checkpoint, complete G0, then matched baseline/after-input pairs.

**Group decision (C000–C003):** CONTINUE with a narrowed measurement plan.
Synthetic environment, binding tests and profiler paths work. Real-model
correctness/performance gate remains open; no G1–G4 or paper novelty claim.

---

## Template

### <EXP_ID> — <short title>

**Date:**
**Research phase:**
**Commit:**

**Hypothesis**

What specific statement is this experiment testing?

**Change / configuration**

What differs from baseline?

**Command**

```bash
...
```

**Artifacts**

- `research/results/<EXP_ID>/...`

**Results**

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| TPOT (ms) | | | | | | |
| Throughput (tok/s) | | | | | | |

**Observed**

Only direct observations.

**Derived**

Calculations/comparisons from the observations.

**Hypothesized explanation**

Mechanistic interpretation, clearly marked as tentative.

**Decision**

`KEEP / REJECT / INCONCLUSIVE`

**Next**

One concrete next experiment or action.

---
