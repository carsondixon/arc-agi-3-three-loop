"""Stage 0 baseline: random-action agent against ARC-AGI-3.

This is purely a smoke test for the SDK and the scorecard pipeline.
It is NOT the submitted agent. It exists to confirm that:
  1. arc_agi.Arcade can authenticate against three.arcprize.org
  2. We can open a scorecard, play a game, and close the scorecard
  3. A scorecard URL gets generated that resolves to a public page

Stage 1 will replace this with src/agent.py (the real Claude-based baseline).

Usage:
    export ARC_API_KEY=...   # from https://three.arcprize.org/
    uv run python -m src.baseline_random --game=ls20 --max-actions=20
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

from arc_agi import Arcade
from arcengine import GameAction, GameState
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/carsondixon/arc-agi-3-three-loop"
SCORECARD_HOST = "https://three.arcprize.org"


def choose_action(frame, rng: random.Random) -> GameAction:
    """Pick a random valid action (or RESET if game isn't in play)."""
    if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        return GameAction.RESET

    available_ids = set(frame.available_actions)
    candidates = [a for a in GameAction if a.value in available_ids and a is not GameAction.RESET]
    if not candidates:
        return GameAction.RESET

    action = rng.choice(candidates)
    if action.is_complex():
        action.set_data({"x": rng.randint(0, 63), "y": rng.randint(0, 63)})
        action.reasoning = {"my_reason": "stage-0 random baseline"}
    else:
        action.reasoning = "stage-0 random baseline"
    return action


def run_random(game_id: str, max_actions: int = 80, seed: int = 0) -> str:
    """Run a single game with random action selection. Returns scorecard URL."""
    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "ARC_API_KEY not set. Sign up at https://three.arcprize.org/ "
            "and export ARC_API_KEY=... (or put it in .env)."
        )

    arcade = Arcade(arc_api_key=api_key)

    scorecard_id = arcade.open_scorecard(
        source_url=SOURCE_URL,
        tags=["stage-0", "baseline-random", "smoke-test"],
    )
    logger.info("Opened scorecard: %s", scorecard_id)

    env = arcade.make(game_id, seed=seed, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"Failed to make environment for game_id={game_id}")

    frame = env.reset()
    rng = random.Random(seed)

    for step in range(max_actions):
        if frame is None:
            logger.warning("Got None frame; aborting")
            break
        action = choose_action(frame, rng)
        frame = env.step(action)
        logger.info(
            "step=%d action=%s state=%s levels_completed=%s win_levels=%s",
            step,
            action.name,
            getattr(frame, "state", "?"),
            getattr(frame, "levels_completed", "?"),
            getattr(frame, "win_levels", "?"),
        )
        if frame is not None and frame.state is GameState.WIN:
            logger.info("WIN at step %d", step)
            break

    arcade.close_scorecard(scorecard_id)
    scorecard_url = f"{SCORECARD_HOST}/scorecards/{scorecard_id}"
    logger.info("Scorecard URL: %s", scorecard_url)
    return scorecard_url


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 0 random-baseline runner")
    parser.add_argument("--game", default="ls20", help="ARC game id (default: ls20)")
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    url = run_random(args.game, args.max_actions, args.seed)
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
