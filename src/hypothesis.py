"""Loop 1: hypothesis graph + Popperian falsification state.

The graph is plain-old-data (Pydantic for serialization) and is updated
exclusively by the Claude reasoner (see prompts/hypothesis_action.txt).
This module owns the data shape, the JSON I/O, and convenience accessors
used by src/agent.py and src/selector.py.

Design notes:
  - Every rule has tests_passed / tests_failed counters. A "test" is when
    we asked Claude to predict an outcome under that rule, executed the
    falsifying action, and observed whether the prediction held.
  - last_prediction is the prediction the agent committed to on the
    previous turn. It is compared against the new observed frame on the
    current turn; this comparison is what generates new evidence.
  - Hypotheses can also be falsified by direct contradiction with the
    new frame even if no explicit prediction was registered.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Rule:
    id: str
    text: str
    confidence: float = 0.5  # 0..1
    tests_passed: int = 0
    tests_failed: int = 0
    last_updated_step: int = -1


@dataclass
class ObjectHypothesis:
    id: str
    description: str  # e.g. "blue 3x3 blob at (32, 18)"
    role_hypothesis: str  # e.g. "player_avatar", "goal", "wall", "hazard", "collectible"
    confidence: float = 0.5


@dataclass
class Prediction:
    """The agent's commitment before an action: what should happen if rule X holds."""
    step_index: int
    action: str  # e.g. "ACTION3"
    rule_id: str | None  # the rule being tested, if any
    expected_outcome: str  # plain English, e.g. "blue blob moves +1 in y"
    falsifying_observation: str  # plain English, e.g. "blue blob unchanged OR moves differently"


@dataclass
class HypothesisGraph:
    game_id: str
    scorecard_id: str
    rules: dict[str, Rule] = field(default_factory=dict)
    objects: dict[str, ObjectHypothesis] = field(default_factory=dict)
    goal_hypothesis: str = "<unknown>"
    open_questions: list[str] = field(default_factory=list)
    last_prediction: Prediction | None = None
    step_count: int = 0

    # ---- I/O ----------------------------------------------------------------

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "HypothesisGraph":
        d = json.loads(text)
        rules = {k: Rule(**v) for k, v in d.pop("rules", {}).items()}
        objects = {k: ObjectHypothesis(**v) for k, v in d.pop("objects", {}).items()}
        lp = d.pop("last_prediction", None)
        prediction = Prediction(**lp) if lp else None
        return cls(rules=rules, objects=objects, last_prediction=prediction, **d)

    def save(self, dir_path: Path) -> Path:
        dir_path.mkdir(parents=True, exist_ok=True)
        out = dir_path / f"{self.scorecard_id}_{self.game_id}.json"
        out.write_text(self.to_json())
        return out

    # ---- Prompt-facing render ------------------------------------------------

    def render_for_prompt(self) -> str:
        """Compact human-readable rendering injected into the prompt each turn."""
        lines = []
        lines.append(f"GOAL HYPOTHESIS: {self.goal_hypothesis}")
        if self.objects:
            lines.append("\nOBJECTS:")
            for o in self.objects.values():
                lines.append(f"  - [{o.id}] {o.description} -- role: {o.role_hypothesis} (conf {o.confidence:.2f})")
        if self.rules:
            # Cap what we show: a long stale rule list bloats the prompt and
            # drowns the model. Show the most-confident, most-recent rules only.
            RULE_CAP = 15
            ranked = sorted(
                self.rules.values(),
                key=lambda r: (-r.confidence, -r.last_updated_step),
            )
            shown = ranked[:RULE_CAP]
            lines.append(f"\nRULES (top {len(shown)} of {len(self.rules)} by confidence):")
            for r in shown:
                lines.append(
                    f"  - [{r.id}] {r.text} (conf {r.confidence:.2f}, "
                    f"passed {r.tests_passed}, failed {r.tests_failed})"
                )
        if self.open_questions:
            lines.append("\nOPEN QUESTIONS:")
            for q in self.open_questions:
                lines.append(f"  - {q}")
        if self.last_prediction:
            p = self.last_prediction
            lines.append("\nLAST PREDICTION (verify against current frame):")
            lines.append(f"  Action taken: {p.action}")
            if p.rule_id:
                lines.append(f"  Rule being tested: {p.rule_id}")
            lines.append(f"  Expected: {p.expected_outcome}")
            lines.append(f"  Would falsify: {p.falsifying_observation}")
        return "\n".join(lines) if lines else "(empty graph -- no hypotheses yet)"

    # ---- Mutation helpers (used by the prompt response parser) --------------

    def apply_update(self, update: dict[str, Any], step_index: int) -> None:
        """Apply a structured update payload from the Claude reasoner.

        Expected shape (any subset):
        {
          "goal_hypothesis": "...",
          "rules": { "r1": {"text": "...", "confidence": 0.7, "tests_passed_delta": 1, "tests_failed_delta": 0}, ... },
          "objects": { "obj_1": {"description": "...", "role_hypothesis": "...", "confidence": 0.6}, ... },
          "open_questions": ["..."],
          "remove_rule_ids": ["rN"],
          "remove_object_ids": ["objN"]
        }
        """
        if "goal_hypothesis" in update and update["goal_hypothesis"]:
            self.goal_hypothesis = update["goal_hypothesis"]

        for rid in update.get("remove_rule_ids", []) or []:
            self.rules.pop(rid, None)
        for oid in update.get("remove_object_ids", []) or []:
            self.objects.pop(oid, None)

        for rid, payload in (update.get("rules") or {}).items():
            existing = self.rules.get(rid)
            if existing is None:
                existing = Rule(id=rid, text=payload.get("text", ""))
                self.rules[rid] = existing
            if "text" in payload:
                existing.text = payload["text"]
            if "confidence" in payload:
                existing.confidence = max(0.0, min(1.0, float(payload["confidence"])))
            existing.tests_passed += int(payload.get("tests_passed_delta", 0) or 0)
            existing.tests_failed += int(payload.get("tests_failed_delta", 0) or 0)
            existing.last_updated_step = step_index

        for oid, payload in (update.get("objects") or {}).items():
            existing = self.objects.get(oid)
            if existing is None:
                existing = ObjectHypothesis(
                    id=oid,
                    description=payload.get("description", ""),
                    role_hypothesis=payload.get("role_hypothesis", "?"),
                )
                self.objects[oid] = existing
            if "description" in payload:
                existing.description = payload["description"]
            if "role_hypothesis" in payload:
                existing.role_hypothesis = payload["role_hypothesis"]
            if "confidence" in payload:
                existing.confidence = max(0.0, min(1.0, float(payload["confidence"])))

        if "open_questions" in update and isinstance(update["open_questions"], list):
            # replace wholesale -- Claude curates the list each turn
            self.open_questions = [str(q) for q in update["open_questions"]]

        self.step_count = max(self.step_count, step_index)
