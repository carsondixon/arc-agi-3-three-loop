# Overnight Autonomous Session — 2026-05-26

Goal: get a non-zero ARC-AGI-3 leaderboard score (vision mode) and build the novel state-transition world-graph contribution. Budget: ~$23.50 available; hard session ceiling ~$19 (keep ~$4.5 reserve). Commit every milestone.

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

## Runs (scorecards)
- ls20 vision smoke: https://three.arcprize.org/scorecards/ed759d4a-6354-4f05-9e46-e224dfd6da85 (0 levels, 5 actions, $0.13)
- ls20 vision-graph smoke (pre-fix): 2bd30d4f-ffb1-4b47-acb0-17b5032edafe ($0.17)
- ls20 vision-graph smoke (post-fix): a4b989b9-09a5-4c6a-a56c-21cdb60075eb (world graph verified)
