# Reproducibility

## Policy

No scorecard reported in this repo or its accompanying paper has been submitted to the [ARC-AGI Community Leaderboard](https://arcprize.org/leaderboard/community) until it has been reproduced from a fresh clone of this repo, on a different machine, using only the Anthropic SDK (no Claude Code), via `scripts/reproduce.sh`.

If the public repo cannot reproduce internal numbers within stochastic variance, the public repo is broken — we fix it before submitting, even if it delays the deadline.

## Pinned components

- **Models:** see [`config/models.yaml`](config/models.yaml). Sonnet, Haiku, Opus model IDs are pinned explicitly.
- **Dependencies:** [`uv.lock`](uv.lock) pins exact transitive dependency versions.
- **Prompts:** all prompts live in [`prompts/`](prompts/) as plain text files, version-controlled.
- **Memory:** [`data/memory.db`](data/) (the cross-game embedding store) is committed in a curated form.

## Reproducing a scorecard

```bash
git clone https://github.com/carsondixon/arc-agi-3-three-loop.git
cd arc-agi-3-three-loop
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
export ARC_API_KEY=...   # from https://arcprize.org
bash scripts/reproduce.sh
```

Expected output: a scorecard URL on `https://arcprize.org/scorecards/...` matching (within stochastic variance) the score reported in [`results/scorecards/`](results/scorecards/).

## Cost transparency

Every API call's cost is logged. See `results/cost_report.md` for the aggregate cost of each published scorecard run.

## Dev environment vs. submitted runtime

The agent's runtime (`src/agent.py`) is pure Python + the Anthropic SDK. It has no dependency on Claude Code, MCP servers, or any orchestration framework.

Development of the agent — including overnight rollouts, ablation iteration, and prompt-patch proposal — was conducted using Claude Code as an orchestration environment. This development infrastructure is **not** part of the submitted artifact and is **not** required to reproduce results. See [`.claude/`](.claude/) for the (optional) Claude Code orchestration.
