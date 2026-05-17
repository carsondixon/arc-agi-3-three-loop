"""Run an agent mode across many games, with a global cost ceiling.

Designed for Phase 1 (strategic subset) and Phase 2 (full sweep) of the
leaderboard run. Each game executes via subprocess for clean isolation
-- a crash in one game does not kill the rest.

Usage:
    # Phase 1: strategic subset
    uv run python scripts/run_multi_game.py \
        --mode=anti-lockin \
        --games=tn36,tu93,re86 \
        --max-actions=80 \
        --max-total-usd=15

    # Phase 2: auto-discover all 25 games and run remaining ones
    uv run python scripts/run_multi_game.py \
        --mode=anti-lockin \
        --all-games \
        --skip-already-run \
        --max-actions=80 \
        --max-total-usd=30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
COST_PATH = PROJECT_ROOT / "results" / "cost_report.jsonl"
SUMMARY_DIR = PROJECT_ROOT / "results" / "multi_game_runs"

logger = logging.getLogger(__name__)


def fetch_all_game_ids(arc_api_key: str) -> list[tuple[str, list[str]]]:
    """Return [(short_game_id, tags), ...] for all available environments."""
    from arc_agi import Arcade  # lazy import
    arcade = Arcade(arc_api_key=arc_api_key)
    envs = arcade.get_environments()
    out = []
    for e in envs:
        short = e.game_id.split("-")[0]
        out.append((short, list(e.tags or [])))
    return out


def previously_run_games(mode: str) -> set[str]:
    """Set of (short) game_ids we've already played under the given mode."""
    out: set[str] = set()
    if not TRAJ_DIR.exists():
        return out
    for p in TRAJ_DIR.glob("*.json"):
        try:
            t = json.loads(p.read_text())
            if t.get("mode") == mode:
                out.add(t.get("game_id", ""))
        except Exception:
            continue
    return out


def current_total_usd() -> float:
    if not COST_PATH.exists():
        return 0.0
    total = 0.0
    for line in COST_PATH.read_text().splitlines():
        try:
            total += json.loads(line).get("usd", 0.0)
        except Exception:
            continue
    return total


def run_one_game(mode: str, game: str, max_actions: int, tag: str) -> dict:
    """Run a single game as a subprocess. Returns a summary dict."""
    cmd = [
        "uv", "run", "python", "-m", "src.agent",
        f"--mode={mode}",
        f"--game={game}",
        f"--max-actions={max_actions}",
        f"--tag={tag}",
    ]
    t0 = time.time()
    started_usd = current_total_usd()
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=2400,  # 40-min hard timeout per game
    )
    elapsed = time.time() - t0
    ended_usd = current_total_usd()
    delta_usd = ended_usd - started_usd

    scorecard_url = ""
    levels = 0
    win_levels = 0
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "scorecards/" in line and not scorecard_url:
            scorecard_url = line.strip().split()[-1]
        # parse "Done: ... levels=X/Y"
        if "Done:" in line and "levels=" in line:
            try:
                levels_part = line.split("levels=")[1].split()[0]
                levels = int(levels_part.split("/")[0])
                win_levels = int(levels_part.split("/")[1])
            except Exception:
                pass

    return {
        "game": game,
        "mode": mode,
        "elapsed_s": round(elapsed, 1),
        "usd": round(delta_usd, 4),
        "levels": levels,
        "win_levels": win_levels,
        "scorecard_url": scorecard_url,
        "returncode": result.returncode,
        "stderr_tail": result.stderr.splitlines()[-3:] if result.returncode != 0 else [],
    }


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-game runner with cost ceiling")
    parser.add_argument("--mode", default="anti-lockin")
    parser.add_argument("--games", help="comma-separated game ids (short)")
    parser.add_argument("--all-games", action="store_true", help="auto-discover all 25 games")
    parser.add_argument("--prefer-tags", help="comma-separated; prefer games with these tags (e.g. click,keyboard_click)")
    parser.add_argument("--skip-already-run", action="store_true", help="skip games already run in this mode")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-total-usd", type=float, default=15.0)
    parser.add_argument("--tag", default="phase-1")
    args = parser.parse_args()

    arc_key = os.environ.get("ARC_API_KEY", "")
    if not arc_key:
        raise SystemExit("ARC_API_KEY not set")

    # Resolve game list
    if args.games:
        game_list = [g.strip() for g in args.games.split(",") if g.strip()]
        # No tag filtering when explicit
    elif args.all_games:
        all_envs = fetch_all_game_ids(arc_key)
        if args.prefer_tags:
            preferred = set(t.strip() for t in args.prefer_tags.split(","))
            tagged = [(g, t) for g, t in all_envs if any(tag in preferred for tag in t)]
            untagged = [(g, t) for g, t in all_envs if not any(tag in preferred for tag in t)]
            ordered = tagged + untagged
        else:
            ordered = all_envs
        game_list = [g for g, _ in ordered]
        tag_map = dict(all_envs)
    else:
        raise SystemExit("Must specify --games=... or --all-games")

    if args.skip_already_run:
        already = previously_run_games(args.mode)
        original = list(game_list)
        game_list = [g for g in game_list if g not in already]
        logger.info("Skipping already-run games: %s", sorted(already & set(original)))

    if not game_list:
        logger.warning("No games to run after filtering. Exiting.")
        return 0

    start_total_usd = current_total_usd()
    logger.info("Multi-game run starting. games=%d mode=%s start_total_usd=$%.2f ceiling=$%.2f",
                len(game_list), args.mode, start_total_usd, args.max_total_usd)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary_path = SUMMARY_DIR / f"run_{run_id}.json"
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "max_actions": args.max_actions,
        "max_total_usd": args.max_total_usd,
        "tag": args.tag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "games": [],
    }

    for i, game in enumerate(game_list, 1):
        cumulative_added = current_total_usd() - start_total_usd
        remaining = args.max_total_usd - cumulative_added
        if remaining <= 0:
            logger.warning("Cost ceiling hit ($%.2f used of $%.2f). Stopping.", cumulative_added, args.max_total_usd)
            break

        logger.info("[%d/%d] %s | cumulative_added=$%.2f remaining=$%.2f",
                    i, len(game_list), game, cumulative_added, remaining)
        try:
            row = run_one_game(args.mode, game, args.max_actions, args.tag)
        except subprocess.TimeoutExpired:
            logger.warning("Game %s timed out", game)
            row = {"game": game, "mode": args.mode, "timeout": True}
        except Exception as e:
            logger.exception("Game %s crashed: %s", game, e)
            row = {"game": game, "mode": args.mode, "error": str(e)}
        summary["games"].append(row)
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("  -> %s", row)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["total_added_usd"] = round(current_total_usd() - start_total_usd, 4)
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print final summary
    print()
    print("=" * 80)
    print(f"FINAL SUMMARY  ({summary['mode']})")
    print("=" * 80)
    print(f"{'game':10s} {'levels':>10s} {'usd':>8s} {'time':>7s}  scorecard")
    print("-" * 80)
    total_levels = 0
    for r in summary["games"]:
        levels = r.get("levels", 0)
        win = r.get("win_levels", 0)
        usd = r.get("usd", 0.0)
        t = r.get("elapsed_s", 0.0)
        url = r.get("scorecard_url", "")
        total_levels += levels
        print(f"{r['game']:10s} {levels:>3d}/{win:<4d}{'':3s} ${usd:>6.2f} {t:>6.0f}s  {url}")
    print("-" * 80)
    print(f"TOTAL: {total_levels} levels completed | ${summary['total_added_usd']} spent")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
