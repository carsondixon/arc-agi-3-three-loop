"""Three-loop ARC-AGI-3 agent.

Two modes, switchable via --mode:

  naked            (Stage 1) -- plain Claude Sonnet picks an action from a
                                hex-text grid. No state, no scaffolding.
                                The floor we beat.

  hypothesis-loop  (Stage 2) -- Loop 1 active. Every action commits to an
                                expected outcome before stepping; next turn,
                                Claude verifies the prediction and updates a
                                Popperian hypothesis graph. Stored in
                                data/hypotheses/{scorecard_id}_{game_id}.json.

Stage 3 (memory) and Stage 4 (self-patching) will introduce additional modes.

Usage:
    export ANTHROPIC_API_KEY=...
    export ARC_API_KEY=...
    uv run python -m src.agent --mode=naked --game=ls20 --max-actions=60
    uv run python -m src.agent --mode=hypothesis-loop --game=ls20 --max-actions=60
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
from typing import Any

import yaml
from arc_agi import Arcade
from arcengine import GameAction, GameState
from dotenv import load_dotenv

from src.claude_client import ClaudeClient
from src.hypothesis import HypothesisGraph, Prediction
from src.world_graph import WorldGraph
from src.perception import (
    color_legend,
    color_legend_visual,
    diff_image,
    diff_objects,
    extract_objects,
    grid_to_hex,
    grid_to_image,
    object_index,
    render_delta,
    render_object_inventory,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NAKED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "baseline_action_selector.txt"
HYPOTHESIS_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hypothesis_action.txt"
HYPOTHESIS_V2_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hypothesis_action_v2.txt"
HYPOTHESIS_V3_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hypothesis_action_v3.txt"
HYPOTHESIS_V4_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hypothesis_action_v4.txt"
HYPOTHESIS_V5_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hypothesis_action_v5.txt"
VISION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "vision_action.txt"
VISION_GRAPH_PROMPT_PATH = PROJECT_ROOT / "prompts" / "vision_graph_action.txt"
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"
HYP_DIR = PROJECT_ROOT / "data" / "hypotheses"
WORLD_GRAPH_DIR = PROJECT_ROOT / "data" / "world_graphs"

SOURCE_URL = "https://github.com/carsondixon/arc-agi-3-three-loop"
SCORECARD_HOST = "https://three.arcprize.org"

HISTORY_LEN = 8


# --------------------------------------------------------------------------- #
# Trajectory data model (shared across modes)
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    step_index: int
    action: str
    thought: str
    state_before: str
    state_after: str
    levels_completed_after: int
    usd: float
    # Stage 2+ fields (optional)
    expected_outcome: str | None = None
    falsifying_observation: str | None = None
    rule_id: str | None = None
    verification: str | None = None


@dataclass
class Trajectory:
    game_id: str
    scorecard_id: str
    seed: int
    mode: str
    started_at: str
    finished_at: str | None = None
    total_usd: float = 0.0
    steps: list[Step] = field(default_factory=list)
    final_state: str = ""
    levels_completed: int = 0
    win_levels: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model response.

    Looks for the first balanced { ... } block. Fences are stripped.
    Returns None on failure.
    """
    # strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # find the outermost JSON object via brace matching
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def _resolve_action(
    name: str, available: list[GameAction]
) -> GameAction | None:
    name = name.strip().upper()
    for a in available:
        if a.name == name:
            return a
    return None


def _apply_action_data(action: GameAction, payload: dict[str, Any] | None, rng: random.Random) -> None:
    """Set click coordinates on complex actions.

    Stage 1 uses random coords. Stage 2 lets Claude pick (x, y) if it provides them.
    """
    if not action.is_complex():
        return
    x = y = None
    if payload:
        try:
            x = int(payload.get("x"))
            y = int(payload.get("y"))
        except (TypeError, ValueError):
            x = y = None
    if x is None or y is None:
        x = rng.randint(0, 63)
        y = rng.randint(0, 63)
    x = max(0, min(63, x))
    y = max(0, min(63, y))
    action.set_data({"x": x, "y": y})


def _setup_env(arcade: Arcade, game_id: str, seed: int, scorecard_id: str):
    env = arcade.make(game_id, seed=seed, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"Failed to make environment for game_id={game_id}")
    frame = env.reset()
    return env, frame


def _available_actions(frame) -> list[GameAction]:
    return [
        a
        for a in GameAction
        if a.value in frame.available_actions and a is not GameAction.RESET
    ]


# --------------------------------------------------------------------------- #
# Stage 1: naked baseline
# --------------------------------------------------------------------------- #


def play_game_naked(
    arcade: Arcade,
    client: ClaudeClient,
    game_id: str,
    scorecard_id: str,
    seed: int,
    max_actions: int,
    per_game_budget_usd: float,
) -> Trajectory:
    prompt_template = NAKED_PROMPT_PATH.read_text()
    env, frame = _setup_env(arcade, game_id, seed, scorecard_id)

    rng = random.Random(seed)
    history: deque[tuple[str, str]] = deque(maxlen=HISTORY_LEN)
    traj = Trajectory(
        game_id=game_id,
        scorecard_id=scorecard_id,
        seed=seed,
        mode="naked",
        started_at=datetime.now(timezone.utc).isoformat(),
        win_levels=frame.win_levels if frame else 0,
    )

    for step_index in range(max_actions):
        if frame is None:
            logger.warning("None frame at step %d; aborting", step_index)
            break
        if frame.state is GameState.WIN:
            logger.info("WIN at step %d", step_index)
            break
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            frame = env.step(GameAction.RESET)
            continue

        available = _available_actions(frame)
        if not available:
            frame = env.step(GameAction.RESET)
            continue

        grid = frame.frame[0]
        history_str = "\n".join(
            f"  step {i}: {a} ({t[:60]})" for i, (a, t) in enumerate(history)
        ) or "  (none yet)"
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

        text, cost = client.reason(
            prompt=prompt,
            role="reasoner",
            tags={"game_id": game_id, "step": step_index, "scorecard_id": scorecard_id, "mode": "naked"},
        )
        traj.total_usd += cost.usd

        parsed = _extract_json(text) or {}
        action_name = parsed.get("action", "").strip().upper()
        thought = parsed.get("thought", "")
        action = _resolve_action(action_name, available) or available[0]
        _apply_action_data(action, None, rng)
        action.reasoning = thought[:200]

        state_before = str(frame.state)
        levels_before = frame.levels_completed
        next_frame = env.step(action)
        if next_frame is None:
            logger.warning("env.step rejected %s; continuing", action.name)
            traj.steps.append(Step(
                step_index=step_index, action=action.name,
                thought=thought + " [REJECTED]",
                state_before=state_before, state_after=state_before,
                levels_completed_after=levels_before, usd=cost.usd,
            ))
            history.append((action.name + "(rejected)", thought))
            continue

        frame = next_frame
        traj.steps.append(Step(
            step_index=step_index, action=action.name, thought=thought,
            state_before=state_before, state_after=str(frame.state),
            levels_completed_after=frame.levels_completed, usd=cost.usd,
        ))
        history.append((action.name, thought))

        logger.info(
            "[naked] step=%d action=%s levels=%d/%d state=%s usd=$%.4f total=$%.4f",
            step_index, action.name, frame.levels_completed, traj.win_levels,
            frame.state, cost.usd, traj.total_usd,
        )
        if traj.total_usd > per_game_budget_usd:
            logger.warning("Budget $%.2f exceeded at step %d; halting", per_game_budget_usd, step_index)
            break

    traj.finished_at = datetime.now(timezone.utc).isoformat()
    traj.final_state = str(frame.state) if frame else "?"
    traj.levels_completed = frame.levels_completed if frame else 0
    return traj


# --------------------------------------------------------------------------- #
# Stage 2: hypothesis loop (Loop 1)
# --------------------------------------------------------------------------- #


def play_game_hypothesis_loop(
    arcade: Arcade,
    client: ClaudeClient,
    game_id: str,
    scorecard_id: str,
    seed: int,
    max_actions: int,
    per_game_budget_usd: float,
    memory_priors: list[Any] | None = None,  # list of (MemoryEntry, similarity) tuples
    mode_label: str = "hypothesis-loop",
    use_anti_lockin_prompt: bool = False,
    use_object_perception: bool = False,
    use_loose_commitment_prompt: bool = False,
    use_frame_diff: bool = False,
    use_vision: bool = False,
    use_world_graph: bool = False,
) -> tuple[Trajectory, HypothesisGraph]:
    if use_world_graph:
        prompt_template = VISION_GRAPH_PROMPT_PATH.read_text()
        use_vision = True
    elif use_vision:
        prompt_template = VISION_PROMPT_PATH.read_text()
    elif use_frame_diff:
        prompt_template = HYPOTHESIS_V5_PROMPT_PATH.read_text()
    elif use_loose_commitment_prompt:
        prompt_template = HYPOTHESIS_V4_PROMPT_PATH.read_text()
    elif use_object_perception:
        prompt_template = HYPOTHESIS_V3_PROMPT_PATH.read_text()
    elif use_anti_lockin_prompt:
        prompt_template = HYPOTHESIS_V2_PROMPT_PATH.read_text()
    else:
        prompt_template = HYPOTHESIS_PROMPT_PATH.read_text()
    # frame-diff and vision modes both use object perception + a CHANGES block
    if use_frame_diff or use_vision:
        use_object_perception = True
    # vision renders prev/current/diff images and shows the change block as text too
    show_delta = use_frame_diff or use_vision
    priors_section = _format_priors(memory_priors or [])
    env, frame = _setup_env(arcade, game_id, seed, scorecard_id)

    rng = random.Random(seed)
    history: deque[tuple[str, str]] = deque(maxlen=HISTORY_LEN)
    graph = HypothesisGraph(game_id=game_id, scorecard_id=scorecard_id)
    world_graph = WorldGraph(game_id=game_id, scorecard_id=scorecard_id) if use_world_graph else None
    prev_grid = None
    prev_objects: list = []
    last_action_name = "(none)"
    traj = Trajectory(
        game_id=game_id,
        scorecard_id=scorecard_id,
        seed=seed,
        mode=mode_label,
        started_at=datetime.now(timezone.utc).isoformat(),
        win_levels=frame.win_levels if frame else 0,
    )

    for step_index in range(max_actions):
        if frame is None:
            logger.warning("None frame at step %d; aborting", step_index)
            break
        if frame.state is GameState.WIN:
            logger.info("WIN at step %d", step_index)
            break
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            frame = env.step(GameAction.RESET)
            continue

        available = _available_actions(frame)
        if not available:
            frame = env.step(GameAction.RESET)
            continue

        grid = frame.frame[0]
        history_str = "\n".join(
            f"  step {i}: {a} ({t[:60]})" for i, (a, t) in enumerate(history)
        ) or "  (none yet)"

        # Object perception (Path 1) -- only when v3/v5 prompt is active
        obj_inventory_str = ""
        obj_lookup: dict = {}
        detected: list = []
        if use_object_perception:
            detected = extract_objects(grid)
            obj_lookup = object_index(detected)
            obj_inventory_str = render_object_inventory(detected)

        # Frame differencing (Stage 4.6) -- show Claude what its last action did
        delta_section = ""
        if show_delta:
            if prev_grid is not None:
                delta = diff_objects(prev_objects, detected, prev_grid, grid)
                delta_section = render_delta(delta, last_action_name)
            else:
                delta_section = "CHANGES SINCE YOUR LAST ACTION: (first action -- no prior frame yet)"

        # Vision perception -- render images now, using prev_grid before it is
        # overwritten below. First action: just the current frame. Otherwise:
        # previous frame, current frame, and a changes-highlighted image.
        vision_images: list[bytes] = []
        if use_vision:
            if prev_grid is not None:
                vision_images.append(grid_to_image(prev_grid))
            vision_images.append(grid_to_image(grid))
            if prev_grid is not None:
                vision_images.append(diff_image(prev_grid, grid))

        fmt_kwargs = dict(
            game_id=game_id,
            win_levels_remaining=traj.win_levels - frame.levels_completed,
            win_levels_total=traj.win_levels,
            game_state=str(frame.state),
            step_index=step_index,
            color_legend=color_legend_visual(grid) if use_vision else color_legend(grid),
            grid_hex=grid_to_hex(grid),
            hypothesis_graph=graph.render_for_prompt(),
            available_actions=", ".join(a.name for a in available),
            history_len=len(history),
            action_history=history_str,
            available_action_names=" | ".join(a.name for a in available),
        )
        if use_object_perception:
            fmt_kwargs["object_inventory"] = obj_inventory_str
            fmt_kwargs["n_objects"] = len(obj_lookup)
        if show_delta:
            fmt_kwargs["delta_section"] = delta_section
        if use_world_graph and world_graph is not None:
            fmt_kwargs["transition_model"] = world_graph.render_for_prompt(
                frame.levels_completed, [a.name for a in available]
            )

        prompt = priors_section + prompt_template.format(**fmt_kwargs)

        # Remember this frame for next-step differencing (before we step)
        if show_delta:
            prev_grid = grid
            prev_objects = detected

        call_tags = {"game_id": game_id, "step": step_index, "scorecard_id": scorecard_id, "mode": mode_label}
        if use_vision:
            text, cost = client.reason_vision(prompt, vision_images, role="reasoner", tags=call_tags)
        else:
            text, cost = client.reason(prompt=prompt, role="reasoner", tags=call_tags)
        traj.total_usd += cost.usd

        parsed = _extract_json(text)
        if parsed is None:
            logger.warning("Failed to parse JSON at step %d; falling back to random action", step_index)
            action = rng.choice(available)
            thought = "<parse_error>"
            expected_outcome = None
            falsifying_observation = None
            rule_id = None
            verification = None
        else:
            verification = parsed.get("verification") or None
            graph_update = parsed.get("graph_update") or {}
            graph.apply_update(graph_update, step_index)

            chosen = parsed.get("chosen_action") or {}
            action_name = chosen.get("name", "")
            action = _resolve_action(action_name, available) or rng.choice(available)
            thought = parsed.get("thought") or ""
            expected_outcome = chosen.get("expected_outcome") or None
            falsifying_observation = chosen.get("falsifying_observation") or None
            rule_id = chosen.get("rule_id") or None

            graph.last_prediction = Prediction(
                step_index=step_index,
                action=action.name,
                rule_id=rule_id,
                expected_outcome=expected_outcome or "",
                falsifying_observation=falsifying_observation or "",
            )

        # Coordinates: resolve target_object_id, else use explicit (x,y), else random
        chosen_data = (parsed or {}).get("chosen_action", {}).get("data") if parsed else None
        click_x = click_y = None
        click_target_id = None
        if action.is_complex():
            if chosen_data and "target_object_id" in chosen_data and obj_lookup:
                tid = str(chosen_data["target_object_id"])
                obj = obj_lookup.get(tid)
                if obj is not None:
                    click_x, click_y = int(obj.centroid_x), int(obj.centroid_y)
                    click_target_id = tid
            if click_x is None and chosen_data and "x" in chosen_data:
                try:
                    click_x = int(chosen_data["x"])
                    click_y = int(chosen_data.get("y", 0))
                except (TypeError, ValueError):
                    click_x = click_y = None
            if click_x is None:
                click_x = rng.randint(0, 63)
                click_y = rng.randint(0, 63)
            click_x = max(0, min(63, click_x))
            click_y = max(0, min(63, click_y))
            action.set_data({"x": click_x, "y": click_y})
        action.reasoning = (thought or "")[:200]

        # Action label for history (includes coords + target id for clicks)
        if action.is_complex():
            tag = f"@({click_x},{click_y})"
            if click_target_id:
                tag += f"=>{click_target_id}"
            action_label = f"{action.name}{tag}"
        else:
            action_label = action.name
        last_action_name = action_label

        state_before = str(frame.state)
        levels_before = frame.levels_completed
        next_frame = env.step(action)
        if next_frame is None:
            logger.warning("env.step rejected %s; continuing", action_label)
            traj.steps.append(Step(
                step_index=step_index, action=action_label,
                thought=thought + " [REJECTED]",
                state_before=state_before, state_after=state_before,
                levels_completed_after=levels_before, usd=cost.usd,
                expected_outcome=expected_outcome, falsifying_observation=falsifying_observation,
                rule_id=rule_id, verification=verification,
            ))
            history.append((action_label + "[REJECTED]", thought))
            graph.save(HYP_DIR)
            continue

        # World graph: record what this action objectively did (before vs after).
        # NOTE: use next_frame here -- `frame` is not reassigned until below.
        if use_world_graph and world_graph is not None:
            after_grid = next_frame.frame[0]
            after_objs = extract_objects(after_grid)
            step_delta = diff_objects(detected, after_objs, grid, after_grid)
            world_graph.observe(
                level=levels_before,
                action=action.name,
                delta=step_delta,
                level_advanced=next_frame.levels_completed > levels_before,
                step=step_index,
            )
            world_graph.save(WORLD_GRAPH_DIR)

        frame = next_frame
        traj.steps.append(Step(
            step_index=step_index, action=action_label, thought=thought,
            state_before=state_before, state_after=str(frame.state),
            levels_completed_after=frame.levels_completed, usd=cost.usd,
            expected_outcome=expected_outcome, falsifying_observation=falsifying_observation,
            rule_id=rule_id, verification=verification,
        ))
        history.append((action_label, thought))
        graph.save(HYP_DIR)

        logger.info(
            "[hyp] step=%d action=%s levels=%d/%d rules=%d objs=%d usd=$%.4f total=$%.4f",
            step_index, action.name, frame.levels_completed, traj.win_levels,
            len(graph.rules), len(graph.objects), cost.usd, traj.total_usd,
        )
        if traj.total_usd > per_game_budget_usd:
            logger.warning("Budget $%.2f exceeded at step %d; halting", per_game_budget_usd, step_index)
            break

    traj.finished_at = datetime.now(timezone.utc).isoformat()
    traj.final_state = str(frame.state) if frame else "?"
    traj.levels_completed = frame.levels_completed if frame else 0
    return traj, graph


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


MODES = ("naked", "hypothesis-loop", "memory-augmented", "anti-lockin", "perception-aware", "perception-loose", "perception-diff", "vision", "vision-graph")


def _format_priors(priors: list[tuple[Any, float]]) -> str:
    """Render retrieved MemoryEntries as a PRIORS section for the prompt."""
    if not priors:
        return ""
    lines = ["PRIORS FROM PAST GAMES (top similar memories; treat as hints, not facts):"]
    for entry, sim in priors:
        lines.append(f"  [sim={sim:.2f}] game={entry.game_id} levels={entry.levels_completed}/{entry.win_levels}")
        lines.append(f"    summary: {entry.summary}")
        if entry.meta_patterns:
            lines.append(f"    meta-patterns: {'; '.join(entry.meta_patterns[:4])}")
        if entry.failure_modes:
            lines.append(f"    failure-modes: {'; '.join(entry.failure_modes[:3])}")
    return "\n".join(lines) + "\n\n"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Three-loop ARC-AGI-3 agent")
    parser.add_argument("--mode", choices=MODES, default="naked")
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--max-actions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--tag", action="append", default=[])
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

    scorecard_tags = [f"stage-{1 if args.mode == 'naked' else 2}", f"mode-{args.mode}", *args.tag]
    scorecard_id = arcade.open_scorecard(source_url=SOURCE_URL, tags=scorecard_tags)
    logger.info("Opened scorecard: %s (mode=%s)", scorecard_id, args.mode)

    if args.mode == "naked":
        traj = play_game_naked(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
        )
    elif args.mode == "hypothesis-loop":
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=None, mode_label="hypothesis-loop",
        )
        graph.save(HYP_DIR)
    elif args.mode == "memory-augmented":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="memory-augmented",
        )
        graph.save(HYP_DIR)
    elif args.mode == "anti-lockin":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (anti-lockin mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="anti-lockin",
            use_anti_lockin_prompt=True,
        )
        graph.save(HYP_DIR)
    elif args.mode == "perception-aware":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (perception-aware mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="perception-aware",
            use_anti_lockin_prompt=False,
            use_object_perception=True,
        )
        graph.save(HYP_DIR)
    elif args.mode == "perception-loose":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (perception-loose mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="perception-loose",
            use_anti_lockin_prompt=False,
            use_object_perception=True,
            use_loose_commitment_prompt=True,
        )
        graph.save(HYP_DIR)
    elif args.mode == "perception-diff":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (perception-diff mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="perception-diff",
            use_frame_diff=True,
        )
        graph.save(HYP_DIR)
    elif args.mode == "vision":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (vision mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="vision",
            use_vision=True,
        )
        graph.save(HYP_DIR)
    elif args.mode == "vision-graph":
        from src.memory import MemoryStore
        store = MemoryStore()
        query = f"Starting game {args.game}. What general lessons from past games apply?"
        priors = store.retrieve(query, k=3)
        logger.info("Retrieved %d priors from memory.db (vision-graph mode)", len(priors))
        traj, graph = play_game_hypothesis_loop(
            arcade=arcade, client=client, game_id=args.game,
            scorecard_id=scorecard_id, seed=args.seed,
            max_actions=args.max_actions, per_game_budget_usd=per_game_budget,
            memory_priors=priors, mode_label="vision-graph",
            use_world_graph=True,
        )
        graph.save(HYP_DIR)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    arcade.close_scorecard(scorecard_id)
    scorecard_url = f"{SCORECARD_HOST}/scorecards/{scorecard_id}"

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"{scorecard_id}_{args.game}.json"
    out_path.write_text(traj.to_json())

    logger.info(
        "Done: mode=%s game=%s levels=%d/%d usd=$%.4f",
        args.mode, args.game, traj.levels_completed, traj.win_levels, traj.total_usd,
    )
    logger.info("Trajectory: %s", out_path)
    logger.info("Scorecard:  %s", scorecard_url)
    print(scorecard_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
