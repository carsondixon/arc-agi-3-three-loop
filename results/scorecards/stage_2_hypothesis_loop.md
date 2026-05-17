# Stage 2 — Loop 1 (Popperian Falsification)

**Date:** 2026-05-17
**Stage:** 2 (Loop 1: within-game hypothesis-and-falsification)
**Agent:** `src/agent.py --mode=hypothesis-loop` — Claude Sonnet 4.6 maintains a structured hypothesis graph (`data/hypotheses/{scorecard_id}_{game_id}.json`). Every action commits to an `expected_outcome` AND `falsifying_observation` before stepping; next turn, Claude verifies the prediction against the new frame and updates rule confidences.
**Purpose:** Test whether within-game falsification alone produces a measurable score improvement over the naked baseline.

## Results

| Game | Tags | Steps | Levels | Cost (USD) | Rules tracked | Scorecard |
|---|---|---|---|---|---|---|
| `ls20` | keyboard | 60 | 0 / 7 | $1.92 | 64 | [fb9df0d2](https://three.arcprize.org/scorecards/fb9df0d2-6b26-4e9a-95ec-120bc0fe102a) |
| `tu93` | keyboard_click | 59 | 0 / 9 | $1.74 | 38 | [7fc9b366](https://three.arcprize.org/scorecards/7fc9b366-25c5-4ed6-b18d-5478fa77453e) |
| `tn36` | click | 60 | 0 / 7 | $1.89 | 38 | [94a0ce46](https://three.arcprize.org/scorecards/94a0ce46-cfdf-426e-b1ae-cb5cbe4c1a1b) |
| **Total** | | **179** | **0 / 23** | **$5.55** | | |

(Plus a 5-action smoke test on `ls20` at $0.14.)

## Comparison: naked vs. hypothesis-loop

| Mode | Total Levels | Total $ | $/action | Rules/game |
|---|---|---|---|---|
| Stage 1 (naked) | 0 / 23 | $1.25 | $0.0070 | 0 (no graph) |
| Stage 2 (hypothesis-loop) | 0 / 23 | $5.55 | $0.0310 | 38-64 |

Same score. **5× the cost.** But the difference in what's happening inside the agent is enormous (see below).

## What hypothesis-loop *can* do

After **just 5 actions** on `ls20`, the agent had:
- Identified the player avatar (`OBJ_player_candidate`, 0.85 confidence)
- Hypothesized the goal (`reach the magenta block at rows 45-46`)
- Mapped 8 typed objects (walls, corridor, goal, hazards)
- Formed 6 falsifiable action rules
- Falsified `ACTION2 = down` after one observation (confidence 0.50 → 0.15)
- Registered a precise next-action prediction (`expected: player shifts +y by ~3 cells`)

The naked baseline produced **none** of this in 60 actions.

## What hypothesis-loop *cannot* do: the lock-in trap

All three games exhibited the same systematic failure: **hypothesis lock-in**. Claude formed a wrong-but-falsifiable theory, then burned the action budget testing it instead of exploring alternatives.

### Three observed lock-in variants

| Game | Theory Claude got stuck on | Cost of lock-in (actions wasted) |
|---|---|---|
| `ls20` | "The game auto-advances at step 64" → spammed ACTION1 to accumulate steps | ~15 actions |
| `tu93` | "Pressing ACTION2 exactly 55 times triggers a transition" (matched a 55-cell fuchsia row) | ~25 actions |
| `tn36` | "No player input possible — this is an autonomous animation" (ACTION6 was being rejected because random click coords missed targets) | ~30 actions |

### Why this is the **right** failure mode to find

Each lock-in is a *technically legitimate* falsifiable hypothesis. Loop 1's design literally incentivizes this: commit to a rule, register a prediction, test the prediction. When the rule is "do X repeatedly to trigger Y", that's a well-formed Popperian hypothesis — it just happens to be wrong, and verifying its falseness requires waiting out the full sequence.

This is **the empirical motivation for Loops 2 and 3**:

- **Loop 2 (cross-game memory)** should provide priors like "previous games' 'do X N times' theories were all wrong; reject this hypothesis shape." Memory entries from past failed games carry exactly this information.
- **Loop 3 (self-patching prompts)** should detect repeated identical-action stretches and inject "you appear to be stuck on a bad hypothesis; switch tracks" guidance into the next-turn prompt.

If both Loop 2 and Loop 3 prove necessary, the paper's central claim is established: **novel-environment agents need feedback loops at multiple time scales because each loop introduces a failure mode the others address.** Loop 1 alone improves *observability* of reasoning but not *score*.

## Cost-efficiency context

Even at 5× the per-action cost of the naked baseline, Stage 2 hypothesis-loop remains **~110× more cost-efficient per action than OpenClaw**:

| Approach | $/action |
|---|---|
| OpenClaw (current SOTA non-human harness, 5.2%) | ~$36 |
| Our Stage 2 hypothesis-loop | $0.031 |
| Our Stage 1 naked baseline | $0.007 |

The remaining question is whether scaffolding can drive score up without driving cost up. Stage 3 (memory) and Stage 4 (self-patching) are the next data points.

## Reproduce

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ARC_API_KEY=...
for g in ls20 tu93 tn36; do
  uv run python -m src.agent --mode=hypothesis-loop --game="$g" --max-actions=60 --tag="stage-2-gate"
done
```

Trajectories saved to `data/trajectories/*.json`. Hypothesis graphs saved per-step to `data/hypotheses/*.json`.

## Gate disposition

Stage 2 gate (`≥1 level on ≥1 game`) **not met** — same outcome as Stage 1. But the *mechanism* of failure is qualitatively different and informative. **Proceeding to Stage 3** with the expectation that cross-game memory priors will break the lock-in trap.

## Next: Stage 3 — Loop 2 (cross-game memory)

`src/memory.py` + `prompts/memory_distillation.txt` + `--mode=memory-augmented` are already wired and pushed. Stage 3 plan:

1. `uv run python scripts/backfill_memory.py` — distill the 3 Stage 2 trajectories into `data/memory.db` (~$0.15).
2. Run `--mode=memory-augmented` on the same 3 games. Same scorecards/budget controls.
3. **Stage 3 gate**: at least one of (a) ≥1 level completed on ≥1 game, OR (b) measurable reduction in lock-in steps (longer effective exploration before commitment).

Probe-first plan: run Stage 3 on `ls20` only first (~$1), then decide whether to fund the full 3-game gate.
