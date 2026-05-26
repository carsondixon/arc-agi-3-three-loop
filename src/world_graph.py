"""Harness-maintained state-transition world model.

This is the project's novel contribution and the complement to
src/hypothesis.py. Where the HypothesisGraph is *authored by Claude* (the
model writes its beliefs each turn), the WorldGraph is *authored by the
harness*: the code deterministically records what each action actually did,
observed from frame deltas, with no model involvement. The model then
*reasons over* this objective transition model.

This bridges the two families that dominate ARC-AGI-3 prior art:
pure graph-search / RL agents (which win by building explicit state models
but cannot reason abstractly) and pure-LLM scaffolds (which reason but have
no persistent objective world model). Here the LLM reasons over a
lightweight, training-free state-transition graph the harness maintains.

Effects are bucketed per level, because each ARC-AGI-3 level can have
different mechanics. For each (level, action) we accumulate: how often it
was tried, how often it was a no-op, the representative movement it caused
(from the largest moved object), appearance/disappearance counts, and --
the gold signal -- how often it advanced a level.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.perception import FrameDelta


def greedy_move(
    player_yx: tuple[int, int],
    target_yx: tuple[int, int],
    vectors: dict[str, tuple[float, float]],
    available: list[str],
    blocked: set[str] | None = None,
) -> str | None:
    """Pick the available, non-blocked movement action that most reduces the
    Manhattan distance from player to target, given measured action vectors.

    Returns the action name, or None if no action makes progress (arrived,
    blocked, or no useful vector). Pure function -- unit-testable offline.
    """
    blocked = blocked or set()
    gy, gx = target_yx[0] - player_yx[0], target_yx[1] - player_yx[1]
    cur = abs(gy) + abs(gx)
    if cur == 0:
        return None
    best: str | None = None
    best_reduction = 0.0
    for name in available:
        if name in blocked:
            continue
        v = vectors.get(name)
        if v is None:
            continue
        dy, dx = v
        new_gap = abs(gy - dy) + abs(gx - dx)
        reduction = cur - new_gap
        if reduction > best_reduction + 1e-9:
            best_reduction = reduction
            best = name
    return best


@dataclass
class ActionEffect:
    action: str
    count: int = 0
    noop_count: int = 0          # action observed to change nothing
    moved_count: int = 0         # at least one object moved
    sum_dy: int = 0              # cumulative movement of the representative object
    sum_dx: int = 0
    appeared_count: int = 0      # turns where new objects appeared
    disappeared_count: int = 0   # turns where objects disappeared
    level_advanced_count: int = 0
    last_step: int = -1

    @property
    def avg_dy(self) -> float:
        return self.sum_dy / self.moved_count if self.moved_count else 0.0

    @property
    def avg_dx(self) -> float:
        return self.sum_dx / self.moved_count if self.moved_count else 0.0

    def summary(self) -> str:
        if self.count == 0:
            return f"{self.action}: untried"
        parts: list[str] = []
        # Movement direction, if consistent
        if self.moved_count:
            dy, dx = self.avg_dy, self.avg_dx
            dir_parts = []
            if abs(dy) >= 0.5:
                dir_parts.append("up" if dy < 0 else "down")
            if abs(dx) >= 0.5:
                dir_parts.append("left" if dx < 0 else "right")
            move_desc = "+".join(dir_parts) if dir_parts else "shifts"
            parts.append(f"moves player {move_desc} (~dy{dy:+.0f},dx{dx:+.0f})")
        if self.appeared_count:
            parts.append(f"objects appear x{self.appeared_count}")
        if self.disappeared_count:
            parts.append(f"objects vanish x{self.disappeared_count}")
        if self.noop_count and not parts:
            parts.append("NO observed effect -- likely useless/blocked here")
        elif self.noop_count:
            parts.append(f"no-op {self.noop_count}x")
        reliability = ""
        if self.moved_count and self.count:
            frac = self.moved_count / self.count
            reliability = " [reliable]" if frac >= 0.8 else f" [{self.moved_count}/{self.count}]"
        line = f"{self.action}: {'; '.join(parts) or 'effect unclear'} (tried {self.count}x){reliability}"
        if self.level_advanced_count:
            line += f"  *** ADVANCED A LEVEL {self.level_advanced_count}x -- HIGH VALUE ***"
        return line


@dataclass
class WorldGraph:
    game_id: str
    scorecard_id: str
    # level -> action_name -> ActionEffect
    effects: dict[str, dict[str, ActionEffect]] = field(default_factory=dict)
    transitions: int = 0

    # ---- I/O ----------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "WorldGraph":
        d = json.loads(text)
        effects = {
            lvl: {a: ActionEffect(**ev) for a, ev in actions.items()}
            for lvl, actions in d.pop("effects", {}).items()
        }
        return cls(effects=effects, **d)

    def save(self, dir_path: Path) -> Path:
        dir_path.mkdir(parents=True, exist_ok=True)
        out = dir_path / f"{self.scorecard_id}_{self.game_id}.json"
        out.write_text(self.to_json())
        return out

    # ---- Observation (harness-authored, deterministic) ----------------------

    def observe(
        self,
        level: int,
        action: str,
        delta: FrameDelta,
        level_advanced: bool,
        step: int,
    ) -> None:
        """Record what `action` objectively did, from the observed frame delta."""
        level_key = str(level)
        bucket = self.effects.setdefault(level_key, {})
        eff = bucket.get(action)
        if eff is None:
            eff = ActionEffect(action=action)
            bucket[action] = eff

        eff.count += 1
        eff.last_step = step

        changed = bool(delta.moved or delta.appeared or delta.disappeared or delta.changed_cells)
        if not changed:
            eff.noop_count += 1
        if delta.moved:
            eff.moved_count += 1
            # Representative motion = the largest moved object (by current size).
            _, curr, dy, dx = max(delta.moved, key=lambda m: m[1].size)
            eff.sum_dy += int(dy)
            eff.sum_dx += int(dx)
        if delta.appeared:
            eff.appeared_count += 1
        if delta.disappeared:
            eff.disappeared_count += 1
        if level_advanced:
            eff.level_advanced_count += 1

        self.transitions += 1

    # ---- Navigation (used by the autopilot) ---------------------------------

    def movement_vectors(self, level: int, min_reliability: float = 0.5) -> dict[str, tuple[float, float]]:
        """Return {action: (avg_dy, avg_dx)} for actions that reliably move things.

        Only actions whose moved-fraction >= min_reliability and whose mean
        displacement is non-trivial are returned -- these are the ones the
        autopilot can steer with.
        """
        out: dict[str, tuple[float, float]] = {}
        for action, eff in self.effects.get(str(level), {}).items():
            if eff.count == 0 or eff.moved_count == 0:
                continue
            if eff.moved_count / eff.count < min_reliability:
                continue
            dy, dx = eff.avg_dy, eff.avg_dx
            if abs(dy) < 0.5 and abs(dx) < 0.5:
                continue
            out[action] = (dy, dx)
        return out

    # ---- Prompt-facing render -----------------------------------------------

    def render_for_prompt(self, level: int, available_action_names: list[str]) -> str:
        """Compact objective transition model for the current level."""
        level_key = str(level)
        bucket = self.effects.get(level_key, {})
        if not bucket and self.transitions == 0:
            return "(no transitions observed yet -- this model fills in as you act)"

        lines = [f"LEARNED TRANSITION MODEL (level {level}, {self.transitions} transitions observed):"]
        # Known actions first, ordered by value (level-advancing, then most reliable movement)
        def _rank(e: ActionEffect) -> tuple:
            return (-e.level_advanced_count, -(e.moved_count / e.count if e.count else 0), -e.count)

        for eff in sorted(bucket.values(), key=_rank):
            lines.append(f"  {eff.summary()}")

        # Unexplored actions available right now
        tried = set(bucket.keys())
        unexplored = [a for a in available_action_names if a not in tried]
        if unexplored:
            lines.append(f"  UNEXPLORED here (try to learn their effect): {', '.join(unexplored)}")
        return "\n".join(lines)
