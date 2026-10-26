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

### C004_controls_build — research control build and binding regression

**Date:** 2026-10-26. **Commit:** `6c5d87462ded0f29a1bc90d8e0ee79403aca36c3`, clean.
**Hypothesis:** minimal input-wait controls can build while preserving baseline
configuration. User authorized mock-first progress and confirmed GitHub origin.
**Change:** optional JIT opcode mask; default zero. Selected producers wait for
inputs before weight issues; stage counts, load counts, arithmetic, routing,
tiling, split policy and cache hints remain unchanged. Nonzero masks enter JIT
JSON/source hashes. Dynamic MoE's legacy first-weight issue is covered.
**Commands:** in job281111/hn43 with `server_env.sh`, `cmake --build build -j 8`;
`python src/tests/test_token_callback.py`; `python src/tests/test_kv_bindings.py`.
H100/driver550.54.15/CUDA12.9.86/Python3.12.12/torch2.5.1+cu121, clocks uncontrolled.
**Artifacts:** server `research/results/C004_controls_build/{metadata.md,commands.sh,summary.json,stdout/}`.
**Observed:** build passed; callback15/15 and KV15/15. First mock differential
driver failed (C005); stdout retained. No real checkpoint tests run.
**Decision:** KEEP build/control implementation; numeric verdict handled below.
**Next:** differential mock validation, then paired timing only if valid.

### C005_mock_differential — initial strict differential check fails

**Date:** 2026-10-26. **Commit:** `6c5d87462ded0f29a1bc90d8e0ee79403aca36c3`, clean.
**Hypothesis:** resetting random seed and recreating fixtures suffices for exact
mock expert/token comparison. **Config:** BS1/S4096, one decode step, random
weights/KV, seed42; two baselines plus all/QKV/router/MoE-up/MoE-down controls.
**Command:** `python research/scripts/validate_prefetch.py --id C005_mock_differential`.
**Artifacts:** server `research/results/C005_mock_differential/` including all
`raw/*.pt`, summary and logs; native JIT output in C004 `stdout/validation-driver.log`.
**Observed:** strict check failed in baseline-versus-baseline too; baseline
max logits error0.002068, raw residual0.003456; expert sets and sampled token
differed. All floating checks met the initially defined baseline-noise threshold.
**Interpretation:** invalid basis for attributing differences to prefetch.
Recreated physical KV mappings and random top-k boundary ties are alternatives;
their individual contributions were not isolated by this run.
**Decision:** REJECT fixture-equivalence assumption; kernel-correctness INCONCLUSIVE.
**Next:** reuse one weights/KV/bootstrap/page-mapping fixture.

### C007_mock_shared_fixture — shared random fixture exposes router instability

**Date:** 2026-10-26. **Commit:** `159eb4f99e4b73d003dfe9da5596269d4dc7bdb3`, clean.
**Hypothesis/config:** C005 with one persistent fixture and decode page reserved
once. Exact page-table stability is checked; all modes run the same one-step input.
**Command:** `python research/scripts/validate_prefetch.py --id C007_mock_shared_fixture`.
**Artifacts:** server `research/results/C007_mock_shared_fixture/`; failed summary,
all snapshots and post-hoc `raw/audit.json` retained.
**Observed:** tokens match for all modes and all floating comparisons pass;
baseline-repeat logits max error0.000122 and residual0.000244. Expert sets still
vary in baseline repeat, so the strict routing check fails.
**Interpretation:** random mock routing is not a stable exact-reference fixture.
Existing upstream layer tests already use expert-set overlap for boundary ties.
**Decision:** INCONCLUSIVE for exact routing correctness; KEEP shared-input control.
**Next:** separate correctness fixture with distinct router projection rows.

### C009_mock_stable_router — strict top-k slot assertion is invalid

**Date:** 2026-10-26. **Commit:** `e22b8bc5872fa7df31a1a68e80bdd840bb9e3a49`, clean.
**Change/config:** correctness-only mock router weights use eight distinct
positive and eight negative multiples of a random direction, remaining rows
zero. The kernel routing algorithm is unchanged. Timing fixtures do not use this.
**Command:** `python research/scripts/validate_prefetch.py --id C009_mock_stable_router --stable-router`.
**Artifacts:** server `research/results/C009_mock_stable_router/`, failed summary
and snapshots preserved; post-hoc audit in `raw/audit.json`.
**Observed:** all tokens and floating checks pass. Strict slot-array equality
fails for repeated baseline/all-off (max ID difference1). Post-hoc canonical
sorting shows selected expert sets exactly match for every mode.
**Interpretation:** top-k slot order is not a semantic invariant; compare sets,
as the existing decode-layer suite does, without loosening membership checks.
**Decision:** REJECT slot-order assertion; KEEP stable-router correctness fixture.
**Next:** require exact expert sets and token equality.

### C010_mock_expert_sets — directed mock scheduling validation passes

**Date:** 2026-10-26. **Commit:** `5f23d01a5f7a706f406eef84ac3c917faca7431b`, clean.
**Hypothesis/config:** C009, now comparing exact sorted expert sets; two baseline
runs plus five control modes. This tests scheduling regressions, not model quality.
**Command:** `python research/scripts/validate_prefetch.py --id C010_mock_expert_sets --stable-router`.
**Artifacts:** server `research/results/C010_mock_expert_sets/`, all logs/tensors
and full per-tensor tolerances/results in summary.json.
**Observed:** 56/56 checks passed. Every output finite, tokens and selected expert
sets exactly equal. Floating tolerance is max(1e-5,4×measured baseline-repeat
max absolute error), defined before comparison. Baseline-repeat max errors:
logits0.000122, raw residual0.000244, normalized residual0.000488,
Q0.000122, attention output9.54e-7. Real checkpoint suite remains NOT RUN.
**Decision:** KEEP controls for mock characterization; G0-R remains open.
**Next:** paired random-weight timing with explicit input fingerprints.

### C011_mock_sensitivity — early prefetch helps at both tested lengths

**Date:** 2026-10-26. **Commit:** `5f23d01a5f7a706f406eef84ac3c917faca7431b`, clean.
**Hypothesis:** delaying all early paths has a context-dependent causal effect;
the sign is not assumed. **Config:** original random synthetic weights/KV,
emergent routing (no stable-router fixture), BS1, S4096/S131072, 512 requested
tokens/511 decode steps, six pairs each, seed42. AB/BA order alternates;
context order shuffled each round. Every child performs a full same-process
warmup. No concurrent profiling; clocks/power unchanged and inventory recorded.
**Command:** `python3 research/scripts/cluster.py --job 281111 run -- python research/scripts/compare_prefetch.py --id C011_mock_sensitivity --contexts 4096 131072 --pairs 6 --tokens 512`.
**Artifacts:** server `research/results/C011_mock_sensitivity/` contains complete
summary, metadata, commands, environment and all24 warmup/timed logs.
Only final PNG/SVG/PDF figure and explanation are published in `research/figures/`.

| Pair | 4K baseline ms | 4K after-input ms | 128K baseline ms | 128K after-input ms |
|---|---:|---:|---:|---:|
| 1 | 3.344062 | 3.472826 | 4.461256 | 4.582282 |
| 2 | 3.348790 | 3.477370 | 4.453782 | 4.575200 |
| 3 | 3.350807 | 3.476839 | 4.464149 | 4.578325 |
| 4 | 3.350698 | 3.491082 | 4.461742 | 4.587509 |
| 5 | 3.351682 | 3.479308 | 4.455018 | 4.595957 |
| 6 | 3.360755 | 3.480602 | 4.451530 | 4.586896 |
| Median | 3.350752 | 3.478339 | 4.458137 | 4.584589 |

**Observed:** all24 runs complete; every one of12 matched pairs has identical
recorded pool shape, full page-table hash, bootstrap token, generator-state hash,
weight sample and sampled-KV fingerprint. All pairs favor baseline early prefetch.
**Derived:** median paired benefit relative to baseline: 4K3.8237%,128K2.7725%.
Median absolute savings128.10 and123.59 µs/token, respectively. Baseline sample
SD0.005453/0.005117 ms; after-input SD0.006187/0.007423 ms. Thus relative benefit
falls while absolute savings remain close; denominator dilution is an alternative
to bandwidth interference. Figure includes both measures and all repetitions.
**Interpretation:** synthetic G1 sensitivity is observed. No crossover and no
HBM mechanism identified; G2/G3/G4 are not satisfied by this result.
**Decision:** KEEP characterization control; INCONCLUSIVE for H1. CONTINUE
attribution, do not implement an adaptive policy.
**Next:** QKV-only paired control (C012), then remaining context/opcode coverage
and selected mechanism profiling before a policy decision.

### C012_mock_qkv_attribution — QKV-only delay has a smaller noisy effect

**Date:** 2026-10-26. **Commit:** `5f23d01a5f7a706f406eef84ac3c917faca7431b`, clean.
**Hypothesis:** the global sensitivity may be concentrated in QKV; alternatively,
other early paths contribute materially. No additive decomposition is assumed.
**Config:** identical to C011 except only QKV is input-gated. Random synthetic
weights/KV and natural routing retained; six AB/BA pairs per context, warmed511
steps, same fixed-driver environment and no concurrent GPU experiment.
**Command:** `python3 research/scripts/cluster.py --job 281111 run -- python research/scripts/compare_prefetch.py --id C012_mock_qkv_attribution --contexts 4096 131072 --mode qkv --pairs 6 --tokens 512`.
**Artifacts:** server `research/results/C012_mock_qkv_attribution/`, all24
warmup/timed logs, environment, commands, metadata and complete summary. No
diagnostic dumps or exploratory noisy plot uploaded.

| Pair | 4K baseline ms | 4K QKV-delayed ms | 128K baseline ms | 128K QKV-delayed ms |
|---|---:|---:|---:|---:|
| 1 | 3.352784 | 3.366166 | 4.492756 | 4.500947 |
| 2 | 3.350112 | 3.376168 | 4.452663 | 4.500647 |
| 3 | 3.354611 | 3.409772 | 4.451336 | 4.514079 |
| 4 | 3.351482 | 3.382439 | 4.453130 | 4.502306 |
| 5 | 3.425964 | 3.371511 | 4.454089 | 4.502229 |
| 6 | 3.350185 | 3.372204 | 4.461786 | 4.560033 |
| Median | 3.352133 | 3.374186 | 4.453610 | 4.502268 |

**Observed:** all12 paired fixture fingerprints match. Five of six short and
all six long pairs favor early QKV; one short pair reverses. Large deviations
are retained, including4K pair5 and128K pairs1/6.
**Derived:** median paired benefit0.7175% short and1.0926% long; median absolute
savings24.04/48.66 µs. Baseline sample SD0.030311/0.016010 ms; variant
SD0.015689/0.023412 ms. These effects are smaller/noisier than C011 global delay.
**Interpretation:** QKV-only results do not identify a long-context harmful path.
Global and QKV effects are not assumed additive; interaction and other opcodes
remain unmeasured. Short-context fine-grained attribution remains uncertain.
**Decision:** INCONCLUSIVE for H1/context dependence; KEEP per-opcode control.
**Next:** remaining opcode/context coverage, repeat noisy QKV points with clock
telemetry, then selected matched profiling. G2/G3/G4 remain open.

**Group decision (C004–C012):** CONTINUE characterization. Mock-first workflow
is operational and all failures retained. Synthetic sensitivity exists, but the
current global result is similar absolute prefetch savings at both lengths, and
QKV does not show long-context harm. No adaptive policy or paper-level bandwidth
interference claim is justified yet; G0-R remains mandatory for final claims.
