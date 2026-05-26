# Writeup Outline — "Seven Harnesses, Zero Points: A Failure-Mode Taxonomy for LLM Agents on ARC-AGI-3"

**Status:** outline for the next session to expand into a blog post + Paper Track submission.
**Format target:** ~2,500-word technical blog post (primary), convertible to a short paper.
**Audience:** ARC Prize community + AI lab recruiters (Anthropic especially). Tone: rigorous, honest, self-aware.

> **UPDATE 2026-05-26 — the story has evolved past "zero points."** After the prior-art research we (a) pivoted to **vision perception** (render the grid to an image), which immediately unblocked navigation where every hex-text mode scored 0 — confirming the field's finding that perception, not reasoning, is the binding constraint; and (b) built the **novel contribution**: an LLM reasoning over a *harness-maintained, deterministic state-transition world graph* (`src/world_graph.py`, `--mode=vision-graph`) — the unclaimed gap between pure graph-search/RL winners and pure-LLM scaffolds. The final framing depends on the overnight score runs (TODO: fill leaderboard result). Two candidate spines: **success** ("vision + a training-free world graph gets a single agent onto the board cheaply") or **honest-progress** ("from non-interactive to navigating; here's the remaining gap"). Either way the prior-art-landscape + perception-bottleneck + world-graph sections below are the load-bearing content.

### New prior-art landscape section (insert as Section 1.5)
- Frontier LLMs score <1% on ARC-AGI-3; CNN/RL and graph-search agents lead (an order of magnitude higher). Pure-LLM scaffolds (e.g. OpenClaw, ~5% at ~$2,912) cap low.
- ~80% of agent failures are *perception* errors, not reasoning (multiple sources). Motivates the vision pivot directly.
- The official reference repo already ships vision + frame-diff + think/observe/memory — so those are table stakes, not novelty. **Our novelty must sit above perception**: the harness-maintained world graph.
- (Verify all external numbers before publishing — some came from web search.)

### New contribution section (replaces/augments old "future work")
- **Harness-authored world graph vs LLM-authored hypothesis graph.** Two complementary state objects: the LLM writes its *beliefs* (HypothesisGraph); the *harness* writes *ground-truth* action effects observed from frame deltas (WorldGraph) — movement vectors, no-ops, appear/vanish, and the gold signal: which action advanced a level. The LLM then reasons over an objective model it did not author.
- **Why it matters:** bridges the two prior-art families; training-free (unlike the RL/CNN winners); debuggable (a determinism mismatch caught a real bug the model's narration had masked).
- Ablation: `--mode=vision` (no graph) vs `--mode=vision-graph` (with graph) on the same game — TODO fill.

---

## Working titles (pick one in next session)
- "Seven Harnesses, Zero Points: What Breaks LLM Agents on ARC-AGI-3"
- "Harness Engineering Is a Perception Problem: A Failure-Mode Taxonomy for ARC-AGI-3"
- "I Built 7 Agent Harnesses for ARC-AGI-3 and Scored 0%. Here's Everything That Broke."

## The hook (lead)
Frontier models score 2-8% on ARC-AGI-3; humans score 100%. I spent two weeks and ~$46 building seven progressively more sophisticated Claude harnesses to close that gap. I scored 0%. This is the map of exactly *why* — five distinct failure modes, each one only visible after fixing the previous — and the embarrassing prior-art lesson that frames the whole thing.

## Section 1 — The bet (why harness, not model)
- ARC-AGI-3 = interactive reasoning; no instructions, no goals, pixel grids + discrete actions.
- ARC explicitly created a community leaderboard for *harness* research. The thesis: scaffold beats raw capability.
- Our novel idea: three feedback loops at different time scales — within-game (falsification), across-game (memory), across-experiment (self-patching).
- Cost philosophy: minimize $/action vs OpenClaw's $2,912 run.

## Section 2 — The failure-mode taxonomy (the core contribution)
Present as a progression. Each mode fixes the prior's failure and reveals a new one. **This is the figure that carries the piece** — a table + per-mode trajectory excerpt.

1. **Naked baseline** → blind cycling. Claude narrates exploration but never observes consequences. (cite trajectory: "I've cycled through all 4 actions...")
2. **Hypothesis loop (Popperian falsification)** → *hypothesis lock-in*. Commits to a falsifiable-but-wrong rule and exhausts the budget. Three variants: "auto-advance at step 64" (ls20), "press ACTION2 ×55" (tu93), "no input possible" (tn36).
3. **Cross-game memory** → lock-in shifts shape but survives; priors break specific traps, not the family.
4. **Anti-lockin discipline** → *commitment failure* (the symmetric twin): rotation rule prevents committing to a single correct answer. → **Architectural claim: discipline mechanisms have symmetric failure modes.**
5. **Perception upgrade (object inventory + click-by-id)** → click accuracy fixed (0 rejected clicks) but score still 0 → coordinate-blindness was *a* failure, not *the* one.
6. **Frame differencing** → Claude finally learns controls and navigates ("ACTION1=up; avatar at (40,36); move toward goal") → but *camera-follow reference-frame trap*: player appears stationary, world scrolls, Claude chases the background.

## Section 3 — Reading the source (the debugging depth flex)
- The SDK downloads each game's Python source locally.
- ls20 decompiled: collect-all-targets navigation game; `Camera(16×16)` viewport upscaled 4× and follows the player.
- This explained the frame-diff trap precisely. Show the win-condition function and the camera constructor.
- Lesson: when an agent says "nothing works," check whether your *perception* can even represent what's happening.

## Section 4 — The prior-art lesson (the honesty that builds credibility)
- We cloned the official `ARC-AGI-3-Agents` repo on day one and never read `multimodal.py`.
- It already does vision + `image_diff` + multi-frame reasoning — exactly what we reinvented worse in hex text.
- Own it plainly. This is the section that signals judgment and self-awareness — disproportionately valuable for recruiting.
- Generalize: "optimize "informed" before "novel.""

## Section 5 — Cost efficiency (the quantitative win)
- Table: OpenClaw ~$36/action @ 5.2% vs our $0.03/action @ 0%.
- Honest framing: we have the efficiency, not the score. But the per-action reasoning quality (world models, falsifiable rules, navigation) is high — the bottleneck was perception representation, not reasoning cost.

## Section 6 — What I'd do next (vision pivot)
- Port the multimodal vision approach; keep the three-loop scaffolding on top.
- Hypothesis: vision dissolves the camera trap (model sees the centered player).
- Test on fixed-camera games first to isolate variables.
- Frame as falsifiable predictions, not promises.

## Section 7 — Takeaways
- Harness engineering for novel environments is dominated by *perception representation* and *discipline calibration*, not raw model reasoning.
- Discipline mechanisms have symmetric failure modes; static prompts can't escape both — motivates dynamic, game-class-aware policy.
- The cheap-research-before-expensive-building discipline matters more than any single clever idea.

---

## Assets already in the repo to use
- `results/scorecards/stage_0..4_*.md`, `path_1_results_and_diagnosis.md` — per-stage data + scorecard URLs.
- `data/trajectories/*.json` — Claude's own verbatim thoughts for excerpts (the trajectory quotes are the most compelling evidence).
- 18+ public scorecard URLs on three.arcprize.org — embed as proof.
- Cost data in `results/cost_report.jsonl`.
- `HANDOFF.md` — the project state summary.

## Figures/tables to produce (next session, free to generate)
1. The failure-mode progression table (modes × failure mode × representative quote).
2. Cost-per-action bar: OpenClaw vs our modes.
3. The ls20 camera diagram (16×16 viewport, player-centered, world scroll) — explains the trap visually.
4. A before/after trajectory excerpt: naked ("nothing happens") vs frame-diff ("avatar at (40,36), move toward goal").

## Distribution plan (when drafted)
- Blog post on personal site + the repo README.
- Tweet thread tagging François Chollet, Mike Knoop, Greg Kamradt; the failure-mode taxonomy + camera GIF as the hook.
- Paper Track submission (deadline Nov 8 2026) — same content, formalized.
- Community leaderboard entry referencing the repo (even at 0%, the harness is reproducible + the writeup is the artifact).
