# Repository Research Instructions

This repository is being used for a research characterization project on **bandwidth interference inside a decode megakernel**.

## Mandatory reading order

Before changing code or running a new experiment, read:

1. `research/RESEARCH_PLAN.md`
2. `research/CURRENT_STATE.md`
3. `research/EXPERIMENT_PROTOCOL.md`
4. The most recent entries in `research/EXPERIMENT_LOG.md`

Treat those files as the source of truth for research intent and prior evidence.

## Core research rule

Do **not** assume the hypothesis is true.

The current hypothesis is that, in long-context decode, aggressive immutable-weight prefetch can contend with KV-cache traffic for HBM bandwidth and can therefore stop helping or become harmful.

Every characterization change must be designed to **falsify or quantify** this hypothesis, not to force a positive result.

## Scope

Initial scope:

- Single NVIDIA H100 / SM90a.
- Cohere North Mini Code decode megakernel.
- Batch size 1 is the primary regime.
- Decode only.
- Context sweep from short to very long contexts.
- Characterization before optimization.
- Preserve model numerics and task dependencies.

Do not introduce a new compiler, serving framework, model architecture, or multi-GPU runtime during characterization.

## Modification discipline

During characterization:

- Prefer the smallest reversible code change.
- Expose research knobs through compile-time constants or runtime flags where practical.
- Do not silently change unrelated scheduling, tiling, routing, quantization, sampling, or numerical precision.
- Keep baseline behavior available in the same commit.
- Never overwrite upstream behavior without an explicit switch or clearly isolated patch.
- Do not optimize for one context length before the characterization matrix is complete.

## Experimental integrity

For every experiment:

- Record exact git commit and dirty status.
- Record GPU model, CUDA/driver versions, clocks if controlled, and command line.
- Record whether weights are real or synthetic.
- Record routing mode.
- Run warmups before timing.
- Repeat timed runs; report median and dispersion.
- Preserve raw stdout/stderr and profiler outputs under `research/results/<experiment-id>/`.
- Do not copy only the "best" run.
- If a run fails, record the failure instead of deleting it.

## Required logging

After each meaningful experiment, append one entry to `research/EXPERIMENT_LOG.md` containing:

- Experiment ID
- Hypothesis
- Code/config change
- Commands
- Raw artifact paths
- Metrics
- Interpretation
- Decision: KEEP / REJECT / INCONCLUSIVE
- Next experiment

Then update `research/CURRENT_STATE.md` with only durable conclusions.

Do not put speculative claims in `CURRENT_STATE.md`.

## Approval boundary

Characterization and instrumentation may proceed autonomously.

Do **not** implement the final adaptive policy until the evidence gate in `research/RESEARCH_PLAN.md` is satisfied.

If the evidence gate fails, stop policy development and document why.

## Validation

After code changes:

- Build the repository.
- Run the relevant existing unit/decode tests.
- Run at least one short-context baseline sanity case.
- Run `git diff --check`.

Do not claim success without observed output.

## Research writing style

When recording conclusions, distinguish:

- **Observed**: directly measured.
- **Derived**: computed from measurements.
- **Hypothesized**: plausible explanation not yet established.

Avoid phrases such as "proves" unless the experiment truly rules out alternatives.
