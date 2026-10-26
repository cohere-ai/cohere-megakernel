# Synthetic characterization figures

`C011_mock_sensitivity.{png,svg,pdf}` reports six paired repetitions for each
of 4K and 128K on the allocated H100, BS1. Baseline is upstream early prefetch;
after-input delays QKV/router/dynamic-MoE weight loads until their input
dependency. All pipeline stage counts, arithmetic and routing algorithms are
unchanged. Random synthetic weights and KV are used, not a trained checkpoint.

Each child performs a full same-process warmup before its 511 timed decode
steps (512 requested tokens includes bootstrap). AB/BA order alternates, context
order is shuffled by round. All 12 pairs passed fixture-fingerprint checks.
Dots show every observation; diamonds show medians and 95% resampled-pair
bootstrap intervals (20,000 resamples, seed42). These small-sample intervals
are descriptive. Absolute and relative benefits are shown together to avoid
mistaking denominator dilution for loss of memory-bandwidth slack.

Source commit for the measurements: `5f23d01a5f7a706f406eef84ac3c917faca7431b`.
Authoritative full repetitions, hardware/package inventory, commands and seed
metadata remain on the server in
`~/hejy/cohere-megakernel/research/results/C011_mock_sensitivity/`.
The tracked experiment log contains compact metrics. No diagnostic traces or
raw tensor snapshots are published here.

Reproduce the plot with `research/scripts/plot_sensitivity.py <summary.json>
--output research/figures/C011_mock_sensitivity`. The plotting environment used
Python 3.12.10, matplotlib 3.11.2 and numpy 2.5.4. The GPU environment is recorded
separately in the experiment metadata.

Observation: baseline is faster at both measured lengths. Median paired savings
are 128.10 and 123.59 µs/token; relative savings are 3.824% and 2.772% of baseline
TPOT. This is not evidence that early prefetch harms long-context decode or that
HBM interference has been identified. No final adaptive policy is justified.
