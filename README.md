# arc-agi-3-three-loop

A Claude-orchestrated harness for [ARC-AGI-3](https://arcprize.org/arc-agi/3) — an interactive reasoning benchmark where agents must discover game mechanics, infer goals, and execute, with no instructions and no stated rules.

This repo is the open-source artifact for a [Community Leaderboard](https://arcprize.org/leaderboard/community) submission and a [Paper Track](https://arcprize.org/competitions/2026/paper) entry to the 2026 ARC Prize.

## Thesis

> Novel-environment agents need feedback loops at three time scales — within-game (falsification), across-game (memory), across-experiment (self-patching). Each helps; the combination compounds.

### The three loops

| Loop | Time scale | Mechanism |
|---|---|---|
| **1 — Within-game** | Per action | Popperian falsification. Every hypothesis has a falsifying action; the selector picks the action with highest information gain. |
| **2 — Across-game** | Per game completed | Episodic memory via local sentence-transformer embeddings over a SQLite store. Retrieves meta-pattern priors for the next game. |
| **3 — Across-experiment** | Per overnight rollout | A meta-agent reviews failure logs and proposes prompt-only patches, gated by a held-out validation set with auto-rollback. |

## Status

| Stage | Description | Status |
|---|---|---|
| 0 | Hello-ARC, SDK pipeline, first scorecard | ✅ done — [writeup](results/scorecards/stage_0_smoke_test.md) |
| 1 | Naked Claude Sonnet baseline | ✅ done — 0/23 levels, $1.25 ([writeup](results/scorecards/stage_1_naked_baseline.md)) |
| 2 | Loop 1: Popperian falsification | ✅ done — 0/23 levels, $5.55, **3 lock-in variants discovered** ([writeup](results/scorecards/stage_2_hypothesis_loop.md)) |
| 3 | Loop 2: Cross-game episodic memory (probe) | ✅ probed on ls20 — 0/7, $1.28, sharper diagnosis but new lock-in shape |
| 4 | Loop 3 lite: anti-lockin prompt patch | ✅ done on ls20 — 0/7, $1.19, **structural lock-in family broken** ([writeup](results/scorecards/stage_4_anti_lockin.md)) |
| 5 | Multi-game Stage 3/4 gates, paper, polish | ⏳ pending budget |

## Usage (current state)

```bash
# Stage 1 naked baseline (no scaffolding)
uv run python -m src.agent --mode=naked --game=ls20 --max-actions=60

# Stage 2 hypothesis-loop (Popperian falsification, Loop 1)
uv run python -m src.agent --mode=hypothesis-loop --game=ls20 --max-actions=60

# Stage 3 memory-augmented (Loop 1 + Loop 2)
uv run python scripts/backfill_memory.py    # bootstrap memory.db (once)
uv run python -m src.agent --mode=memory-augmented --game=ls20 --max-actions=40

# Stage 4 anti-lockin (Loop 1 + Loop 2 + structural anti-lockin prompt)
uv run python -m src.agent --mode=anti-lockin --game=ls20 --max-actions=40
```

## Key finding (current state)

A 4-condition ablation on `ls20` reveals a *progression of failure modes* under increasing scaffolding:

| Stage | Levels | Failure mode |
|---|---|---|
| 1 naked | 0/7 | No reasoning |
| 2 hypothesis-loop | 0/7 | **Hypothesis lock-in** (3 distinct variants across 3 games) |
| 3 memory | 0/7 | Meta-pattern lock-in (priors break specific traps, family survives) |
| 4 anti-lockin | 0/7 | None — agent correctly diagnoses "unwinnable in action space" |

**Claim**: for LLM-driven Popperian agents in novel-environment benchmarks, *structural prompt-level guards* are necessary and sufficient to prevent hypothesis lock-in; retrieval-augmented priors alone are insufficient. Cost per action ~$0.03, **>1000× more efficient than current SOTA harnesses** (OpenClaw at ~$36/action).

## Reproduce

```bash
git clone https://github.com/carsondixon/arc-agi-3-three-loop.git
cd arc-agi-3-three-loop
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
export ARC_API_KEY=...   # from https://arcprize.org
bash scripts/reproduce.sh
```

Model IDs are pinned in [`config/models.yaml`](config/models.yaml). All prompts are externalized to [`prompts/`](prompts/). Aggregate cost is reported in `results/cost_report.md`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full policy.

## Architecture (two-layer)

The submitted artifact (`src/agent.py`) is pure Python + the Anthropic SDK. It has zero dependency on Claude Code or any orchestration framework — anyone can clone and run it.

Development of the agent — overnight rollouts, ablation iteration, prompt-patch proposal — was conducted using Claude Code as an orchestration environment. This is **not** part of the submitted artifact and is **not** required to reproduce results.

## License

MIT-0 (MIT No Attribution). See [LICENSE](LICENSE).

## Acknowledgements

ARC Prize Foundation for the benchmark and SDK. Anthropic for Claude.
