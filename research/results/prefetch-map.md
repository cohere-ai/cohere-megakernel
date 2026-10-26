# Prefetch map at upstream 67d0b9ca22ea3652796b715d1d1863459e0e2c3c

Source inspection only; no performance or HBM-interference result.

The plan's `prefetch_weight_tiles` helper does not exist in this revision.
Loads are inline in `gemm_n8_op_scheduled` and `qkv_n8_op`.

| Path | BS1 prefix | Tile BN × BK | Weight traffic issued before input wait |
|---|---:|---|---|
| QKV | 10 | 64 × 64 | min(10, K tiles) weight tiles, then layer-input barrier |
| Router | 10 | 64 × 64 | min(10, split-local K tiles), then layer-input barrier |
| Dense up/gate | 0 | 128 × 64 | none; static scheduler waits once before tile loop |
| Dense down | 0 | 64 × 64 | none; static scheduler waits once |
| O projection | 0 | 64 × 64 | none; static scheduler waits on attention |
| LM head | 0 | 64 × 64 | none; static scheduler waits once |
| MoE up/gate drain | 0 | 128 × 64 | **one weight tile per claimed task**, then gather-input wait |
| MoE down drain | 0 | 128 × 64 | **one weight tile per claimed task**, then up/gate-input wait |

All listed weights are BF16. `_op` fixes BK=64. A 64×64 weight tile is
8192 bytes and a 128×64 tile is 16384 bytes. These are TMA request sizes,
not measured HBM bytes (cache hits/reuse can change physical traffic).
For BS2/4/8, the explicit MoE prefix is 1 instead of 0.

`InstructionTileScheduler::WAIT_INPUT_ONCE=true` and
`MoeQueueTileScheduler::WAIT_INPUT_ONCE=false` distinguish the paths.
Consequently setting all `prefetch_stages=0` alone is **not** a global
no-early-prefetch control. A valid control must also move the dynamic MoE
first-weight issue behind its per-task input dependency. Preserve stage count,
split-K, routing, task ordering, cache hints, and numeric work.

Relevant code: `src/decode/schedule.py:default_nmc_tiling_for_bs`,
`src/decode/megakernel.cuh:gemm_n8_op_scheduled`, `qkv_n8_op`, and policy
`wait_input_bars` functions.

Attention splitting is already context dependent (`_choose_attn_splits` and
`_attn_drain_splits_by_row`): a <=64K effective-context cap restricts splits to
16, with additional chunk/SM limits. Sliding-window attention caps effective
span separately. Record the realized split/schedule signatures when comparing
contexts; a latency kink alone does not establish contention.

Candidate scope refinement (hypothesized): quantify the maximum issued early
bytes and actual overlap first. Finite prefixes may have too little traffic
to explain whole-token changes. If so, reject or narrow the interference
premise rather than increasing speculative traffic to manufacture it.
