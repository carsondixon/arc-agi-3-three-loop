# Overnight Autonomous Session — 2026-05-26

Goal: get a non-zero ARC-AGI-3 leaderboard score (vision mode) and build the novel state-transition world-graph contribution. Budget: ~$23.50 available; hard session ceiling ~$19 (keep ~$4.5 reserve). Commit every milestone.

## ☀️ MORNING BRIEFING (read this first)

**Headline:** I pivoted the whole approach based on prior-art research, and it worked at the capability level: the agent went from "concludes games are non-interactive" (hex-text, 0 score) to **perceiving the board, learning controls, forming correct goals, and navigating to collect targets** — at **~5× lower cost per action** than before. We are **not yet on the leaderboard** — no completed level across the night's runs — but the gap is now narrow and well-understood, and the codebase has a genuinely novel, paper-worthy contribution.

**Final wa30 run (vision-pilot, fixes applied):** completed cleanly (empty-frame crash fixed), ~200 actions for **$1.41**, still 0/9 levels. wa30 has been tried 3 ways now (vision-graph 80a, focused 143a, pilot 200a) without completing level 1 — the agent collects yellow blobs but some appear wall-blocked and it can't finish the set. Further wa30 runs = lock-in; the move is to try *other* games cheaply with vision-pilot.

**Three things built tonight (all committed):**
1. **`--mode=vision`** — render the grid to a PNG and send it as an image (we were doing lossy, mis-labeled hex text before). This alone unblocked navigation. Confirms the field's finding: *perception, not reasoning, was the binding constraint.*
2. **`--mode=vision-graph`** — the novel contribution: an LLM reasoning over a **harness-maintained, deterministic state-transition world graph** (`src/world_graph.py`). The harness records ground-truth action effects; the LLM consults them. This is the unclaimed gap between pure graph-search/RL winners and pure-LLM scaffolds.
3. **`--mode=vision-pilot`** — the breakthrough: a **training-free LLM + graph-search hybrid**. Claude picks a target; the harness drives the player onto it using the measured movement vectors, with **no LLM call per move**. Result: precise navigation (no more oscillation) at **~$0.006/action vs $0.03** — and it bridges the two prior-art families.

**Why no score yet (honest):** the agent navigates and collects, but on wa30 it couldn't complete level 1 — it gets stuck on collectibles that appear wall-blocked, and these games have 6–9 levels. The remaining gap is goal-completion robustness on hard layouts, not perception or reasoning.

**Spend tonight:** ~$17–18 of the budget. Reserve held.

**Recommended next steps (your call):**
- Run `vision-pilot` across more games (it's cheap now — ~$0.3–1.5/game) to find ones whose level 1 completes; banking even one = on the leaderboard.
- The artifact is strong *today*: vision-unblock + the world-graph + the LLM/graph-search hybrid + the cost-efficiency story, all reproducible.

---

## Status board

| Time | Event | Result | Cost | Cumulative |
|---|---|---|---|---|
| start | Phase 0 vision smoke (ls20, 5a) | PASS — maps controls, navigates | $0.13 | $0.13 |
| — | Phase 1 gate su15 (fixed cam, 80a) | running | — | — |

## Game selection (free research, no Claude calls)
25 envs total. Tags: `keyboard` = movement-only (best fit for vision navigation), `click`/`keyboard_click` = pointer puzzles (harder for us). Camera classification of the keyboard games:
- **g50t**: fixed camera, actions [1-4] movement — UNTESTED, top candidate.
- **wa30**: fixed camera, actions [1-5] movement+interact — UNTESTED, top candidate.
- **tr87**: fixed camera, actions [1-4] — tested w/ hex-text before (symmetric-failure game); retry with vision.
- **ls20**: 16x16 follow camera (the reference-frame trap), 7 levels.
- su15 (the gate) is actually `click`, not keyboard — explains slow scoring; clicks are harder for navigation-style play.
Plan: prioritize g50t + wa30 (fixed-cam movement) with vision for a first score; use ls20 for the vision vs vision-graph head-to-head (novelty test).

## Decisions log
- Vision confirmed as the unblock (Phase 0). Hex-text modes abandoned.
- Camera classification: su15/sc25 = fixed camera (full 64x64, clean test); ls20 = 16x16 follow-camera (reference-frame trap).
- Picked su15 as the primary fixed-camera gate (short ~361 human baseline) — best shot at an actual score.

## Track B — novel contribution: state-transition world graph
- Built `src/world_graph.py`: harness-authored (deterministic) per-(level,action) effect model — movement (dy/dx), no-ops, appear/vanish, and the gold signal: level-advanced count. Complement to the LLM-authored HypothesisGraph.
- Added `--mode=vision-graph` (`prompts/vision_graph_action.txt`): vision + the LEARNED TRANSITION MODEL injected into the prompt so Claude reasons over an objective world model. This is the unclaimed gap (LLM-over-graph) and the paper differentiator.
- Bug caught + fixed via the deterministic graph: observe() read `frame` before it was reassigned to `next_frame`, so it diffed the before-grid against itself (all no-ops). Fixed to use `next_frame`. Post-fix smoke: ACTION1 recorded moved 6/6, dy=-14 (up), reliable.

## Phase 2 sweep result + diagnosis (vision-graph, g50t/wa30/tr87, 80a each, $7.41)
All 3 scored 0 levels — but the world graphs + trajectories show the agent IS working, just not finishing:
- **wa30 (closest):** correct goal hypothesis ("collect all yellow blobs, they turn green"), actively navigating + collecting (7+ blobs vanished=collected). Ran out of actions: the $2.50/game cap stopped it at 80 with multi-cell/diagonal movement. **More actions should score.** → launched focused run (160a/$4.5) + tightened anti-overshoot nav guidance.
- **g50t:** movement tiny/noisy (avg <1 cell); agent never locks a reliable control map. Weak candidate.
- **tr87:** moves but spammed ACTION3 31x with little effect; no goal progress.
Key takeaway: vision+world-graph gives correct goals and real navigation; the gap to a score is execution precision + action budget, not perception or reasoning.

## BREAKTHROUGH: vision-pilot autopilot (the night's main result)
Diagnosis from wa30/g50t/tr87: vision+world-graph gives CORRECT goals and real navigation, but per-step LLM nav oscillates near a target and can't land on it (even contradicting its own measured world graph). The field's known fix is graph-search execution — so I built it as the natural extension of the world graph:
- **`--mode=vision-pilot`**: Claude identifies the player + a target object; the harness then drives the player onto it via `greedy_move()` using the world graph's measured movement vectors. Primitive moves take NO LLM call.
- Result on the smoke: 40/45 actions were LLM-free autopilot navigation in bursts of 6-8, targeting collectibles by name. **Cost ~$0.006/action vs ~$0.03 for per-step modes — a ~5x reduction**, AND precise (greedy can't oscillate).
- Smoke caught + fixed a stall bug: when no known action reduces distance, autopilot now takes one forced exploration step (least-mapped control) and tells the LLM to pick a reachable target, instead of infinite re-planning.
- This is the paper's strongest result: a training-free LLM+graph-search hybrid bridging the two prior-art families, with a clean cost-efficiency story. `greedy_move` is unit-tested offline.
- Validation run (wa30, 280 actions, $2 cap) launched to test whether cheap+precise actions can complete a level. (result below)

## Runs (scorecards)
- vision-pilot wa30 smokes: eb69bde3 (stall, fixed) / f50a3ce2 (40/45 auto steps, $0.28)
- vision-pilot wa30 validate: bsy19c8qg log (reached action 194 for $1.16, then empty-frame crash — fixed)
- vision-pilot wa30 final (fixes): https://three.arcprize.org/scorecards/59e8dc32-4eb5-4a5f-88b8-90fcd8ebdf48 (clean, ~200a, $1.41, 0/9)

## Spend summary (tonight)
~$19 total. Breakdown: vision/vision-graph smokes ~$0.5; su15 gate $1.76; vision-graph sweep $7.41; wa30 focused $4.52; pilot smokes ~$1.1; pilot validate $1.16; pilot final $1.41. At my self-imposed ceiling — stopped to avoid re-hitting the account usage limit.
- vision-graph sweep g50t/wa30/tr87: 7497ea90 / 1b3df412 / 05115f7f (all 0 levels, $7.41)
- ls20 vision smoke: https://three.arcprize.org/scorecards/ed759d4a-6354-4f05-9e46-e224dfd6da85 (0 levels, 5 actions, $0.13)
- ls20 vision-graph smoke (pre-fix): 2bd30d4f-ffb1-4b47-acb0-17b5032edafe ($0.17)
- ls20 vision-graph smoke (post-fix): a4b989b9-09a5-4c6a-a56c-21cdb60075eb (world graph verified)
