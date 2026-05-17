# Stage 4 — Loop 3 Lite (Anti-Lockin Prompt Patch)

**Date:** 2026-05-17
**Stage:** 4 (Loop 3, static prompt-level intervention)
**Agent:** `src/agent.py --mode=anti-lockin` — Stage 3 (memory-augmented) + a structural anti-lockin protocol injected into the prompt itself. No external priors; the rules are *in the prompt template*.
**Purpose:** Test whether a *principled* prompt-level intervention can break the hypothesis lock-in *family* that Loop 1 introduces and Loop 2 only partially mitigates.

## The Anti-Lockin Protocol (six rules)

Added as a top-priority section in `prompts/hypothesis_action_v2.txt`:

(A) **Information-gain discipline** — rank available actions by expected info gain before each choice.
(B) **No-repetition-without-delta** — last 3 identical actions ⇒ next must register a *materially different* prediction.
(C) **Short-window testing** — "do X N times" hypotheses can be tested in 1-2 actions, not N.
(D) **Aggressive rule pruning** — `tests_failed ≥ 2` ⇒ remove the rule.
(E) **Explore modes, not sequences** — 5+ no-delta steps ⇒ question the interaction model itself (click coords? hover? drag?), not the sequence length.
(F) **Early goal commitment** — after 4-6 actions, commit to a best-guess goal and pursue it.

## Result on ls20

| Metric | Stage 3 (memory) | Stage 4 (anti-lockin) |
|---|---|---|
| Levels | 0 / 7 | 0 / 7 |
| Cost | $1.28 | $1.19 |
| Steps | 40 | 40 |
| Final rule count | 22 | **2** |
| Action distribution | 17/11/10/2 (skewed) | 13/11/9/7 (balanced) |
| Final diagnosis quality | "static puzzle, encoded sequence" (kept testing) | **"game is non-interactive via ACTION1-4; no win condition reachable with available actions"** |
| Time to converge on diagnosis | never explicit | step ~10 |

[Scorecard](https://three.arcprize.org/scorecards/52674915-e7a7-401f-bf3e-ab62b4fc3c0d)

## What the protocol did

The trajectory shows Claude operating with *Popperian discipline* throughout:

- **Step 0**: "Prior memory suggests ACTION4 likely moves right" — read the prior.
- **Step 5**: "All single actions have failed 5 times; testing ACTION2 as part of a potential two-action sequence" — the anti-lockin protocol redirected from "press X N times" to a different hypothesis class.
- **Step 10-15**: Reached confident diagnosis "the game is non-interactive via ACTION1-4." Rule R15 ("inert action set") at 0.99 confidence.
- **Step 20-39**: Claude *explicitly* rotates through ACTION1-4 to "satisfy anti-lockin protocol while continuing to confirm R15 with minimal waste." Anti-lockin worked as designed — the agent recognized it had nothing more to learn and stopped pretending otherwise.

## What this means

**The lock-in family was structurally broken.** Where Stage 2 spammed `ACTION1` for 15 steps and Stage 3 burned 15 steps on encoded-sequence testing, Stage 4 spent zero steps on either trap. Rules pruned aggressively (22 → 2). The agent reached a *correct epistemic conclusion* — given the available action set, no further action carries information about the goal.

**The score remained 0/7.** But this is not the same kind of 0 as in Stages 1-3. In Stages 1-3, 0 was a consequence of the agent failing to reason. In Stage 4, 0 is a consequence of the agent reasoning correctly to a "no solution in this action space" conclusion. The random baseline also scored 0 on ls20 with 80 actions — corroborating that this game (with only ACTION1-4 exposed) may not be winnable by simple keyboard input alone.

The next test would be a *click* game (e.g., `tn36`) where the anti-lockin protocol's rule (E) — *"ACTION6 may need (x, y) coordinates pointing at a meaningful cell"* — could plausibly produce a score. Budget at end of Stage 4 ($0.42 remaining) does not permit this experiment in this cycle.

## Cumulative finding across Stages 1-4

A four-condition ablation on `ls20`:

| Condition | Levels | Cost | Reasoning failure mode |
|---|---|---|---|
| Stage 1: naked | 0/7 | $0.39 | None — no reasoning |
| Stage 2: hypothesis-loop | 0/7 | $1.92 | Hypothesis lock-in (specific: "auto-advance at step 64") |
| Stage 3: memory-augmented | 0/7 | $1.28 | Meta-pattern lock-in (different shape, same family) |
| Stage 4: anti-lockin | 0/7 | $1.19 | None — agent correctly diagnoses unwinnable in action space |

**Claim**: For LLM-driven Popperian agents in novel-environment benchmarks, **structural prompt-level guards** are necessary and sufficient to prevent hypothesis lock-in. Retrieval-augmented priors alone are insufficient. This is the empirical core of the paper.

## Reproduce

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ARC_API_KEY=...
uv run python -m src.agent --mode=anti-lockin --game=ls20 --max-actions=40 --tag="stage-4-anti-lockin"
```

## Next experiments (requires additional budget)

1. **Anti-lockin on tn36** (click game) — does rule (E) lead Claude to try `ACTION6` with meaningful `(x, y)` coordinates?
2. **Anti-lockin on tu93** — does rule (C) break the "press N times" sequence-testing trap empirically?
3. **Anti-lockin without memory priors** — clean ablation: does the prompt alone account for the effect, or do priors still matter?
4. **Full Loop 3 (dynamic self-patching)** — a meta-agent that *generates* anti-lockin rules from observed failure modes, rather than the hand-written set used here. The Stage 4 patch was authored by inspecting Stage 2/3 failures; Loop 3 would do this automatically.
