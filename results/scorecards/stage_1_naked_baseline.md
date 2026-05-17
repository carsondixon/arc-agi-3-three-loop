# Stage 1 — Naked Claude Sonnet Baseline

**Date:** 2026-05-17
**Stage:** 1 (Naked Baseline)
**Agent:** `src/agent.py` — Claude Sonnet 4.6 picks the next action from a hex-text rendering of the 64×64 grid. No hypothesis tracking, no memory, no diff highlighting. One short externalized prompt (`prompts/baseline_action_selector.txt`).
**Purpose:** Establish the floor that the three loops (falsification, memory, self-patching) must beat.

## Results

| Game | Tags | Steps | Levels Completed | Cost (USD) | Scorecard |
|---|---|---|---|---|---|
| `ls20` | keyboard | 60 | 0 / 7 | $0.3912 | [9464cc3b](https://three.arcprize.org/scorecards/9464cc3b-221d-4250-af1a-5a49272fb83b) |
| `tu93` | keyboard_click | 60 | 0 / 9 | $0.4788 | [2854b483](https://three.arcprize.org/scorecards/2854b483-6126-40f7-9344-9a4f654c2f7a) |
| `tn36` | click | 60 | 0 / 7 | $0.3748 | [846d21ab](https://three.arcprize.org/scorecards/846d21ab-1b65-4df4-8504-40405c032f75) |
| **Total** | | **180** | **0 / 23** | **$1.25** | |

(Plus a 10-action smoke test on `ls20` at $0.06 and a 1-action aborted run on `tn36` at $0.006. Total Stage 1 spend: **$1.31**.)

## Cost-efficiency snapshot

- **$0.006 per action** (Sonnet 4.6, ~1100 input tokens + ~100 output tokens per step)
- **$0.42 per game** at 60 actions (vs. OpenClaw's reported $2,912 per run — a 50× per-action efficiency advantage even before any scaffolding)

## What the naked baseline *can* do

Claude verbalizes exploration intent fluently. Sample thoughts (from `9464cc3b-...ls20.json`):

> "I need to explore the game to understand its rules, so I'll start with ACTION1 to see what happens."
> "I've cycled through all 4 actions without apparent progress; let me try ACTION2 again to see if the grid responds differently."
> "The grid shows complex colored patterns; I'll try ACTION3 to explore different outcomes."

## What the naked baseline *cannot* do

Despite verbalizing exploration, Claude cycles ACTION1→2→3→4 indefinitely for 60 steps without:

1. **Registering whether the grid changed between actions** (no diff observation)
2. **Forming any hypothesis** about what each action does
3. **Selecting actions on the basis of expected information gain**
4. **Carrying state across turns** — every turn the prompt is fresh and the only history is the bare action names

This is *exactly* the failure mode the three-loop architecture is designed to address:

- **Loop 1 (falsification)** will make Claude register an expected outcome before each action and observe whether it held.
- **Loop 2 (memory)** will preserve priors from earlier games for the next game.
- **Loop 3 (self-patching)** will let the harness rewrite its own prompts when it notices recurring failure modes.

## Gate disposition

The original Stage 1 gate (`≥1% on 3 games`) is mis-calibrated against the actual scoring rule (`efficiency² × level-weighted sum`): completing even a single level on a single game would not necessarily clear 1%. The meaningful signal is binary — **does scaffolding cause Claude to complete at least one level on at least one game?**

**Stage 2 revised gate:** complete ≥1 level on ≥1 game out of the same 3-game test set, holding all other variables equal. This is the cleanest possible "scaffold works / doesn't work" signal.

## Reproduce

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ARC_API_KEY=...
for g in ls20 tu93 tn36; do
  uv run python -m src.agent --game="$g" --max-actions=60 --tag="stage-1-gate"
done
```

Trajectories saved to `data/trajectories/*.json`.

## Next: Stage 2 — Loop 1 (Formal Falsification)

`src/hypothesis.py` + `src/selector.py`. Every action must register a falsifying expectation. Action selector picks the action that maximally falsifies open hypotheses. Same 3-game test set, identical budget.
