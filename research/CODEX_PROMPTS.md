# Reusable Codex Prompts

These are entry-point prompts. `AGENTS.md` should remain the durable instruction layer.

## Prompt A — Begin characterization

Read `AGENTS.md`, `research/RESEARCH_PLAN.md`, `research/CURRENT_STATE.md`, and `research/EXPERIMENT_PROTOCOL.md`.

Work only on the current characterization phase. First inspect the repository and identify the exact early weight-prefetch paths. Do not modify runtime behavior yet.

Create `research/results/prefetch-map.md` containing:
- opcode/path;
- relevant source file/function;
- where weight loads occur relative to activation dependency waits;
- whether the path is context-dependent;
- what minimal switch could disable only the early/speculative part.

Update `research/EXPERIMENT_LOG.md` only if you run experiments.
Update `research/CURRENT_STATE.md` only with durable observations.
Stop after presenting the map and the minimal instrumentation plan.

## Prompt B — Implement characterization controls

Read the research guidance and latest log/state first.

Implement the smallest reversible controls needed for:
1. exact upstream baseline;
2. global "no early prefetch before activation readiness";
3. per-opcode enable/disable if feasible without invasive restructuring.

Baseline must remain the default.

Build and run existing relevant tests. Run one BS=1, 4K sanity benchmark for baseline and modified mode.

Store raw outputs under a new `research/results/<EXP_ID>/` directory, append the experiment log, and update current state only if evidence is durable.

Do not implement an adaptive policy.

## Prompt C — Run a context-sweep batch

Read all research instructions and current state.

Run the next approved context-sweep experiments from `RESEARCH_PLAN.md`.
Do not change code unless needed to fix a reproducibility bug.

For each experiment:
- save raw output;
- record environment and git state;
- run repeated timings;
- write `summary.json`;
- append `EXPERIMENT_LOG.md`.

After the batch, compute the prefetch sensitivity versus context and update `CURRENT_STATE.md` with:
- observed trend;
- uncertainty/confounders;
- CONTINUE/NARROW/PIVOT/STOP decision.

Do not implement the final policy.

## Prompt D — Mechanism profiling

Read the existing characterization results first.

Select the minimum representative contexts needed to explain the observed crossover/decay.

Profile baseline vs the relevant prefetch-control variant.
Use Perfetto/sm-profiler for timeline evidence and Nsight Compute only where hardware counters materially help.

Do not use profiler wall-clock time as benchmark latency.

Record:
- which task intervals changed;
- attention duration;
- waiting/overlap changes;
- memory-system evidence available;
- alternative explanations that remain.

Update the experiment log and current state.

## Prompt E — Re-plan after unexpected data

Read the complete research log and current state.

Do not assume the original idea is correct.

Summarize:
1. what has been established;
2. what was falsified;
3. strongest remaining explanation;
4. smallest next experiment that distinguishes competing explanations.

Propose an update to `RESEARCH_PLAN.md` only where evidence requires it.
Preserve rejected hypotheses/history in `EXPERIMENT_LOG.md`.

Stop before coding the new direction.
