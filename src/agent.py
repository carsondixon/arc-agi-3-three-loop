"""Stage 1: naked Claude Sonnet baseline.

The single-file submission entry point. No scaffolding beyond:
  - hex-text grid perception (src/perception.py)
  - a single externalized prompt (prompts/baseline_action_selector.txt)
  - JSON parsing of the model's response
  - per-game USD budget cap from config/models.yaml

Stage 2 will add the hypothesis graph (src/hypothesis.py + src/selector.py).
Stage 3 will add episodic memory. Stage 4 will add self-patching prompts.

Usage:
    export ANTHROPIC_API_KEY=...
    export ARC_API_KEY=...
    uv run python -m src.agent --game=ls20 --max-actions=40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from arc_agi import Arcade
from arcengine import GameAction, GameState
from dotenv import load_dotenv

from src.claude_client import ClaudeClient
from src.perception import color_legend, grid_to_hex

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "baseline_action_selector.txt"
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"

SOURCE_URL = "https://github.com/carsondixon/arc-agi-3-three-loop"
SCORECARD_HOST = "https://three.arcprize.org"

HISTORY_LEN = 8  # last N actions shown in the prompt


@dataclass
class Step:
    step_index: int
    action: str
    thought: str
    state_before: str
    state_after: str
    levels_completed_after: int
    usd: float


@dataclass
class Trajectory:
    game_id: str
    scorecard_id: str
    seed: int
    started_at: str
    finished_at: str | None = None
    total_usd: float = 0.0
    steps: list[Step] = field(default_factory=list)
    final_state: str = ""
    levels_completed: int = 0
    win_levels: int = 0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def _parse_action(text: str, available: list[GameAction]) -> tuple[GameAction, str]:
    """Pull JSON out of the model response, return (action, thought).

    Falls back to the first available action if parsing fails (so we never crash).
    """
    available_names = {a.name: a for a in available}
    # find first {...} block
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            name = obj.get("action", "").strip().upper()
            thought = obj.get("thought", "")
            if name in available_names:
                return available_names[name], thought
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse action from model output; falling back to first available")
    return available[0], "<parse_error>"


def play_game(
    arcade: Arcade,
    client: ClaudeClient,
    game_id: str,
    scorecard_id: str,
    seed: int,
    max_actions: int,
    per_game_budget_usd: float,
) -> Trajectory:
    prompt_template = _load_prompt()

    env = arcade.make(game_id, seed=seed, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"Failed to make environment for game_id={game_id}")

    frame = env.reset()
    rng = random.Random(seed)
    history: deque[tuple[str, str]] = deque(maxlen=HISTORY_LEN)
    traj = Trajectory(
        game_id=game_id,
        scorecard_id=scorecard_id,
        seed=seed,
        started_at=datetime.now(timezone.utc).isoformat(),
        win_levels=frame.win_levels,
    )

    for step_index in range(max_actions):
        if frame is None:
            logger.warning("Got None frame at step %d; aborting game", step_index)
            break

        if frame.state is GameState.WIN:
            logger.info("WIN reached at step %d", step_index)
            break

        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            thought = "<auto-reset>"
            state_before = str(frame.state)
            frame = env.step(action)
            traj.steps.append(Step(
                step_index=step_index,
                action=action.name,
                thought=thought,
                state_before=state_before,
                state_after=str(frame.state) if frame else "?",
                levels_completed_after=frame.levels_completed if frame else 0,
                usd=0.0,
            ))
            continue

        # Build the prompt
        available = [a for a in GameAction if a.value in frame.available_actions and a is not GameAction.RESET]
        if not available:
            # nothing legal: reset and continue
            action = GameAction.RESET
            frame = env.step(action)
            continue

        grid = frame.frame[0]  # 64x64 int8 ndarray
        history_str = "\n".join(f"  step {i}: {a} ({t[:60]})" for i, (a, t) in enumerate(history)) or "  (none yet)"
        prompt = prompt_template.format(
            game_id=game_id,
            win_levels_remaining=traj.win_levels - frame.levels_completed,
            win_levels_total=traj.win_levels,
            game_state=str(frame.state),
            color_legend=color_legend(grid),
            grid_hex=grid_to_hex(grid),
            available_actions=", ".join(a.name for a in available),
            history_len=len(history),
            action_history=history_str,
            available_action_names=" | ".join(a.name for a in available),
        )

        # Ask Claude
        text, cost = client.reason(
            prompt=prompt,
            role="reasoner",
            tags={"game_id": game_id, "step": step_index, "scorecard_id": scorecard_id},
        )
        traj.total_usd += cost.usd

        action, thought = _parse_action(text, available)

        if action.is_complex():
            # Stage 1 doesn't have spatial reasoning; pick random coords (matches
            # the reference random_agent behavior). Stage 2 will let Claude pick (x, y).
            action.set_data({"x": rng.randint(0, 63), "y": rng.randint(0, 63)})
        action.reasoning = thought[:200]

        state_before = str(frame.state)
        levels_before = frame.levels_completed
        next_frame = env.step(action)
        if next_frame is None:
            # Action was rejected (e.g. invalid coords). Log and keep current frame
            # so we don't abort the game on one bad step.
            logger.warning("env.step returned None for %s; retrying next step", action.name)
            traj.steps.append(Step(
                step_index=step_index,
                action=action.name,
                thought=thought + " [REJECTED]",
                state_before=state_before,
                state_after=state_before,
                levels_completed_after=levels_before,
                usd=cost.usd,
            ))
            history.append((action.name + "(rejected)", thought))
            continue
        frame = next_frame
        levels_after = frame.levels_completed

        traj.steps.append(Step(
            step_index=step_index,
            action=action.name,
            thought=thought,
            state_before=state_before,
            state_after=str(frame.state) if frame else "?",
            levels_completed_after=levels_after,
            usd=cost.usd,
        ))
        history.append((action.name, thought))

        logger.info(
            "step=%d action=%s levels=%d/%d state=%s usd=$%.4f (game total $%.4f)",
            step_index, action.name, levels_after, traj.win_levels,
            frame.state if frame else "?", cost.usd, traj.total_usd,
        )

        if traj.total_usd > per_game_budget_usd:
            logger.warning("Per-game budget $%.2f exceeded at step %d; halting game",
                           per_game_budget_usd, step_index)
            break

    traj.finished_at = datetime.now(timezone.utc).isoformat()
    traj.final_state = str(frame.state) if frame else "?"
    traj.levels_completed = frame.levels_completed if frame else 0
    return traj


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 1 naked Claude baseline")
    parser.add_argument("--game", default="ls20", help="ARC game id (default: ls20)")
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--tag", action="append", default=[], help="extra scorecard tag")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    arc_key = os.environ.get("ARC_API_KEY", "")
    if not arc_key:
        raise SystemExit("ARC_API_KEY not set. See .env.example.")

    config = _load_config()
    per_game_budget = float(config["budget"]["per_game_usd"])

    arcade = Arcade(arc_api_key=arc_key)
    client = ClaudeClient()

    scorecard_id = arcade.open_scorecard(
        source_url=SOURCE_URL,
        tags=["stage-1", "naked-claude-baseline", *args.tag],
    )
    logger.info("Opened scorecard: %s", scorecard_id)

    traj = play_game(
        arcade=arcade,
        client=client,
        game_id=args.game,
        scorecard_id=scorecard_id,
        seed=args.seed,
        max_actions=args.max_actions,
        per_game_budget_usd=per_game_budget,
    )

    arcade.close_scorecard(scorecard_id)
    scorecard_url = f"{SCORECARD_HOST}/scorecards/{scorecard_id}"

    # Persist trajectory
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"{scorecard_id}_{args.game}.json"
    out_path.write_text(traj.to_json())

    logger.info("Game finished: levels_completed=%d/%d total_usd=$%.4f",
                traj.levels_completed, traj.win_levels, traj.total_usd)
    logger.info("Trajectory saved: %s", out_path)
    logger.info("Scorecard URL: %s", scorecard_url)
    print(scorecard_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
