"""Summarize all trajectories under data/trajectories/ into a comparison table.

Run:
    uv run python scripts/aggregate_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
HYP_DIR = PROJECT_ROOT / "data" / "hypotheses"


def main() -> None:
    rows = []
    by_mode_game = defaultdict(list)
    for p in sorted(TRAJ_DIR.glob("*.json")):
        t = json.loads(p.read_text())
        mode = t.get("mode", "naked")
        game = t["game_id"]
        steps = len(t["steps"])
        usd = t["total_usd"]
        levels = t["levels_completed"]
        win_levels = t["win_levels"]
        rows.append((mode, game, steps, usd, levels, win_levels, p.name, t["scorecard_id"]))
        by_mode_game[(mode, game)].append((steps, usd, levels, win_levels))

    # Per-run table
    print("=" * 100)
    print(f"{'mode':18s} {'game':6s} {'steps':>6s} {'usd':>8s} {'levels':>10s}  scorecard")
    print("-" * 100)
    for mode, game, steps, usd, levels, win_levels, fname, sid in rows:
        print(f"{mode:18s} {game:6s} {steps:>6d} ${usd:>7.4f} {levels:>5d}/{win_levels:<4d}  {sid}")

    # Aggregate by mode
    print()
    print("=" * 100)
    print(f"{'mode':18s} {'game':6s} {'runs':>5s} {'total_usd':>10s} {'best_levels':>12s} {'rules_in_graph':>16s}")
    print("-" * 100)
    for (mode, game), runs in sorted(by_mode_game.items()):
        total_usd = sum(r[1] for r in runs)
        best_levels = max(r[2] for r in runs)
        # find latest hypothesis graph for this mode+game (if any)
        rule_count = ""
        for hyp_path in sorted(HYP_DIR.glob(f"*_{game}.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                h = json.loads(hyp_path.read_text())
                if h.get("game_id") == game:
                    rule_count = f"{len(h.get('rules', {}))}"
                    break
            except Exception:
                continue
        print(f"{mode:18s} {game:6s} {len(runs):>5d} ${total_usd:>9.4f} {best_levels:>12d} {rule_count:>16s}")


if __name__ == "__main__":
    main()
