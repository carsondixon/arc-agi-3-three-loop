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
| 0 | Hello-ARC, SDK pipeline, first scorecard | ✅ done — [stage_0_smoke_test.md](results/scorecards/stage_0_smoke_test.md) |
| 1 | Naked Claude Sonnet baseline | ✅ done — 0/23 levels, $1.25 ([writeup](results/scorecards/stage_1_naked_baseline.md)) |
| 2 | Loop 1: Popperian falsification | 🔧 running gate |
| 3 | Loop 2: Cross-game episodic memory | ⏳ pending |
| 4 | Loop 3: Self-patching prompts | ⏳ pending |
| 5 | Reproducibility, polish, paper | ⏳ pending |

## Usage (current state)

```bash
# Stage 1 naked baseline (no scaffolding)
uv run python -m src.agent --mode=naked --game=ls20 --max-actions=60

# Stage 2 hypothesis-loop (Popperian falsification)
uv run python -m src.agent --mode=hypothesis-loop --game=ls20 --max-actions=60
```

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
