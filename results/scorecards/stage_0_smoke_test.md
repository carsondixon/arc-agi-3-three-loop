# Stage 0 — Smoke Test Scorecard

**Date:** 2026-05-17
**Stage:** 0 (Hello-ARC)
**Agent:** `src/baseline_random.py` — uniform-random action selection
**Purpose:** Confirm the SDK pipeline works end-to-end and produces a publicly resolvable scorecard.

## Scorecard

- **URL:** https://three.arcprize.org/scorecards/55dc2a5b-beeb-4f61-a4d7-b2337de32931
- **Game:** `ls20`
- **Max actions:** 20
- **Seed:** 0
- **Score:** 0 (random agent, expected)

## Reproduce

```bash
export ARC_API_KEY=...   # from https://three.arcprize.org/
uv run python -m src.baseline_random --game=ls20 --max-actions=20 --seed=0
```

## Notes

- The SDK auto-downloads `environment_files/ls20/<hash>/` on first run. This directory is gitignored.
- Game `ls20` has 7 win-levels with human baseline action counts `[22, 123, 73, 84, 96, 192, 186]`.
- The action space exposed per-state via `frame.available_actions` is much smaller than the full `GameAction` enum.
- This run confirms Stage 0 gate: the pipeline produces a publicly resolvable scorecard URL.

## Next: Stage 1

Replace random selection with Claude Sonnet 4.6 driven action selection. Gate: ≥1% score on 3 games.
