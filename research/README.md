# Research Harness

This directory keeps research intent and evidence separate from implementation code.

## Files

- `RESEARCH_PLAN.md` — hypothesis, contribution boundary, phases, gates, kill criteria.
- `EXPERIMENT_PROTOCOL.md` — how experiments must be executed.
- `CURRENT_STATE.md` — compact current truth; no experiment diary.
- `EXPERIMENT_LOG.md` — append-only experiment history.
- `CODEX_PROMPTS.md` — reusable prompts for Codex sessions.
- `results/` — raw artifacts and per-experiment metadata.
- `scripts/` — research-only automation scripts.

## Rule of ownership

If information answers:

- **Why are we doing this?** → `RESEARCH_PLAN.md`
- **How must experiments be run?** → `EXPERIMENT_PROTOCOL.md`
- **What do we currently believe based on evidence?** → `CURRENT_STATE.md`
- **What happened in each run?** → `EXPERIMENT_LOG.md`
- **How do I start a Codex task?** → `CODEX_PROMPTS.md`
- **Where are raw measurements?** → `results/`

Do not duplicate the same evolving state across several files.
