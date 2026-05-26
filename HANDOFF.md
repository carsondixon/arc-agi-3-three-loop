# ARC-AGI-3 Project — Handoff / Compaction Doc

**Last updated:** 2026-05-25
**Repo:** https://github.com/carsondixon/arc-agi-3-three-loop
**Purpose of this doc:** Let a fresh Claude Code session (or a human) resume without the bloated multi-week conversation context. Read this first.

---

## 1. Goal

Compete on the ARC-AGI-3 **Community Leaderboard** + **Paper Track** (NOT the Kaggle prize — that's sandboxed, no Claude API at runtime). Primary purpose: a recruiting/research artifact for top AI labs (Anthropic). Beat OpenClaw's published 5.2% / $2,912 baseline, ideally far more cheaply.

## 2. Current status (blunt)

- **Score: 0 levels completed across 10 unique games, 30+ runs, 7 agent modes.**
- **API spend: ~$46.5 of $50 topped up. ~$3.50 left.**
- We tie the random baseline on the leaderboard metric.
- BUT: capability progressed from "agent concludes games are non-interactive" → "agent learns controls and navigates toward goals" (just in the wrong reference frame on camera games).

## 3. The single most important lesson

**We did not study prior art before building.** The official reference repo (`references/ARC-AGI-3-Agents`, gitignored) ships an `agents/templates/multimodal.py` that already:
- Renders the 64×64 grid to a 256×256 PNG and sends it as a **vision image**
- Has an **`image_diff()`** function for frame differencing
- Prompts the model to reason about x,y locations across **multiple frames**

We instead built hex-text perception + object-diffing from scratch — a worse reimplementation. **Any future work should start by reading and building on `multimodal.py`, not our hex-text pipeline.**

## 4. Failure modes discovered (the real research value)

Across 7 agent modes, each revealed a distinct, named failure mode:

| Mode | Failure mode |
|---|---|
| `naked` (Stage 1) | No discipline → blind action cycling |
| `hypothesis-loop` (Stage 2) | Falsification discipline → **hypothesis lock-in** (over-commits to one wrong rule; e.g. "auto-advance at step 64") |
| `memory-augmented` (Stage 3) | Cross-game priors → lock-in shifts shape but persists |
| `anti-lockin` (Stage 4) | Rotation discipline → **commitment failure** (under-commits; can't pick a single answer) |
| `perception-aware` (Path 1) | Object-targeted clicks land correctly but score stays 0 → click accuracy wasn't the bottleneck |
| `perception-loose` | Removing anti-lockin doesn't help on games lacking the right action exposure |
| `perception-diff` (latest) | Frame-diff makes Claude navigate — but **camera-follow games invert the reference frame**: player appears stationary, world scrolls, Claude chases scrolling background |

**Architectural claim for the paper:** LLM-agent discipline mechanisms have *symmetric* failure modes (exploration discipline prevents commitment; commitment discipline prevents exploration), and turn-based perception has *reference-frame* failure modes on camera games. Cost efficiency: ~$0.03/action vs OpenClaw's ~$36/action (≈1000× cheaper).

## 5. Key discovery from reading game source

The SDK downloads each game's Python source to `environment_files/{game}/{hash}/{game}.py` (gitignored). Reading `ls20.py` revealed:
- ls20 is a **collect-all-targets navigation game**: ACTION1-4 = up/down/left/right; win = player visits every collectible position (`pbznecvnfr`, line 2042).
- It uses a `Camera(width=16, height=16)` — the 64×64 frame is a **16×16 viewport upscaled 4×**, and it follows the player. This is why frame-diff showed the whole world scrolling.

**Reading game source is legitimate for understanding/debugging perception. Do NOT hardcode per-game solutions — submissions must be general-purpose.**

## 6. Repo architecture

```
src/
  agent.py           # 7 modes via --mode=. Main loop, JSON parsing, budget caps.
  perception.py      # hex grid + extract_objects (scipy connected components) + diff_objects/render_delta (frame diff)
  hypothesis.py      # HypothesisGraph (rules/objects/goal), JSON persistence
  memory.py          # SQLite + sentence-transformer episodic memory, distill_and_store
  claude_client.py   # Anthropic SDK wrapper + per-call cost logging
prompts/
  baseline_action_selector.txt      # Stage 1
  hypothesis_action.txt             # v1 Stage 2
  hypothesis_action_v2.txt          # + anti-lockin
  hypothesis_action_v3.txt          # + object inventory
  hypothesis_action_v4.txt          # loose commitment
  hypothesis_action_v5.txt          # + frame-diff CHANGES block  (latest)
  memory_distillation.txt
scripts/
  run_multi_game.py   # multi-game runner with cost ceiling
  backfill_memory.py  # distill trajectories into memory.db
  aggregate_results.py
config/models.yaml    # pinned model ids, per-game budget cap ($2.50)
results/scorecards/   # per-stage writeups (stage_0..4, path_1)
data/trajectories/    # every run's full log
data/hypotheses/      # per-run hypothesis graphs
```

Run example: `uv run python -m src.agent --mode=perception-diff --game=ls20 --max-actions=80`

## 7. Recommended next steps (in priority order)

1. **PIVOT TO VISION.** Port `references/ARC-AGI-3-Agents/agents/templates/multimodal.py` approach: send rendered images (not hex text), use its `image_diff`, multi-frame reasoning. This likely fixes the camera trap (model sees the centered player). This is the highest-EV move and the one we should have done first.
2. **Keep the three-loop scaffolding** (falsification + memory + anti-lockin-toggle) on top of vision perception — that's still the novel contribution.
3. **Test on fixed-camera games first** to validate frame-diff navigation works when the reference frame isn't inverted (not all games have cameras).
4. **Budget**: vision calls cost more (images ≈ 1-1.5k tokens each, multi-frame). A serious vision-based push needs a top-up (~$30-50). Do NOT spend the last $3.50 on another hex-text experiment.

## 8. Honest assessment of "is it worth continuing"

- **For a leaderboard score**: maybe, but only via the vision pivot, and only with more budget. The hex-text path is exhausted.
- **For the recruiting artifact**: we already have a strong story (systematic failure-mode taxonomy + source-driven diagnosis + cost-efficiency + the honest "we should have read the template" lesson). A writeup is viable *today* with zero further spend.
- **The cheapest high-value move right now is FREE**: read `multimodal.py` fully, write the paper/blog around what we learned, and decide on the vision pivot separately.

## 9. Credentials / environment

- `.env` (gitignored) holds `ARC_API_KEY` and `ANTHROPIC_API_KEY`.
- `uv` for deps; Python 3.12; sentence-transformers installed (local embeddings, free).
- GitHub: `carsondixon/arc-agi-3-three-loop`, public, MIT-0. gh CLI authenticated as `carsondixon`.
