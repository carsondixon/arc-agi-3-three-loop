"""Loop 2: cross-game episodic memory.

After each game completes, the trajectory + final hypothesis graph are
distilled (via Claude) into a short memory entry. The entry is embedded
locally (sentence-transformers MiniLM, no API cost) and stored in
SQLite at data/memory.db. On a new game start, we retrieve top-K
similar past entries and inject them as priors into the hypothesis-loop
prompt.

This module is Stage 3. The actual integration into the agent loop
(injection of retrieved memories into the prompt) lives in src/agent.py
once Stage 2 is validated.

Key design choices:
  - Embeddings are local. Zero per-game embedding cost.
  - The memory entry is a structured summary (game tags, confirmed rules,
    falsified rules, meta-patterns) PLUS a free-text passage that the
    embedder sees. This gives semantic retrieval while keeping the
    retrieved entry interpretable when injected into a prompt.
  - We store the embedding as raw bytes (float32) in SQLite. Cosine
    similarity is computed in numpy at query time. For <10K entries this
    is plenty fast; we will revisit if memory.db grows beyond that.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.db"

EMBED_DIM = 384  # MiniLM all-MiniLM-L6-v2 output dim


@dataclass
class MemoryEntry:
    """One distilled memory of a completed game run."""
    game_id: str
    scorecard_id: str
    levels_completed: int
    win_levels: int
    game_tags: list[str]  # from EnvironmentInfo.tags, e.g. ["keyboard"]
    summary: str  # 3-5 sentences (Claude-generated)
    confirmed_rules: list[str]  # high-confidence rules at end of game
    falsified_rules: list[str]  # rules dropped due to tests_failed
    meta_patterns: list[str]  # cross-game-applicable observations
    failure_modes: list[str]  # what went wrong / blocked progress
    cost_usd: float
    step_count: int

    def passage(self) -> str:
        """The free-text representation embedded for retrieval.

        Includes everything semantically meaningful. Order matters less than
        coverage -- the embedder reads tokens, not structure.
        """
        return "\n".join([
            f"Game: {self.game_id}",
            f"Tags: {', '.join(self.game_tags)}",
            f"Outcome: {self.levels_completed}/{self.win_levels} levels in {self.step_count} steps (${self.cost_usd:.2f})",
            f"Summary: {self.summary}",
            f"Confirmed rules: {'; '.join(self.confirmed_rules)}" if self.confirmed_rules else "",
            f"Falsified rules: {'; '.join(self.falsified_rules)}" if self.falsified_rules else "",
            f"Meta-patterns: {'; '.join(self.meta_patterns)}" if self.meta_patterns else "",
            f"Failure modes: {'; '.join(self.failure_modes)}" if self.failure_modes else "",
        ]).strip()


# --------------------------------------------------------------------------- #
# Embedding (lazy import to keep Stage 0/1/2 startup fast)
# --------------------------------------------------------------------------- #


_model = None


def _get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    """Return a (EMBED_DIM,) float32 numpy vector. L2-normalized."""
    model = _get_embedder()
    v = model.encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


# --------------------------------------------------------------------------- #
# SQLite-backed store
# --------------------------------------------------------------------------- #


class MemoryStore:
    """SQLite-backed episodic memory with cosine retrieval."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scorecard_id TEXT NOT NULL,
        game_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_entries_game_id ON entries(game_id);
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> int:
        from datetime import datetime, timezone
        emb = embed(entry.passage())
        cur = self._conn.execute(
            "INSERT INTO entries (scorecard_id, game_id, ts, payload_json, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                entry.scorecard_id,
                entry.game_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(asdict(entry)),
                emb.tobytes(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def all_entries(self) -> list[tuple[int, MemoryEntry, np.ndarray]]:
        rows = self._conn.execute(
            "SELECT id, payload_json, embedding FROM entries"
        ).fetchall()
        out = []
        for row_id, payload, emb_bytes in rows:
            payload_obj = json.loads(payload)
            entry = MemoryEntry(**payload_obj)
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            out.append((row_id, entry, emb))
        return out

    def retrieve(self, query: str, k: int = 3, exclude_scorecard_ids: list[str] | None = None) -> list[tuple[MemoryEntry, float]]:
        """Return top-k MemoryEntries by cosine similarity to the query.

        exclude_scorecard_ids lets us avoid retrieving the in-progress game's
        own entries (relevant if we ever store mid-game snapshots).
        """
        all_entries = self.all_entries()
        if not all_entries:
            return []
        exclude = set(exclude_scorecard_ids or [])
        candidates = [
            (entry, emb) for _, entry, emb in all_entries
            if entry.scorecard_id not in exclude
        ]
        if not candidates:
            return []
        q = embed(query)
        # all entries are L2-normalized -> cosine = dot product
        sims = np.array([float(np.dot(q, emb)) for _, emb in candidates])
        order = np.argsort(-sims)[:k]
        return [(candidates[i][0], float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# Trajectory -> MemoryEntry distillation (Claude-driven; see prompts/)
# --------------------------------------------------------------------------- #


def trajectory_to_distillation_prompt_payload(traj: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    """Build the structured payload the distillation prompt consumes.

    We pull out everything potentially useful for retrieval, but keep token
    count down -- raw step-by-step is too verbose. Pass: outcome stats,
    final hypothesis graph, last 10 steps verbatim, plus a 'first N actions'
    sketch.
    """
    steps = traj.get("steps", [])
    sketch_first = [
        {"i": s["step_index"], "a": s["action"], "thought": (s.get("thought") or "")[:120]}
        for s in steps[:5]
    ]
    sketch_last = [
        {"i": s["step_index"], "a": s["action"], "thought": (s.get("thought") or "")[:120]}
        for s in steps[-10:]
    ]
    # Trim the rules dict: keep only high-conf or strongly-falsified ones
    rules = graph.get("rules", {}) or {}
    interesting_rules = {
        k: v for k, v in rules.items()
        if v.get("confidence", 0) >= 0.7 or v.get("tests_failed", 0) >= 1
    }
    return {
        "game_id": traj.get("game_id"),
        "levels_completed": traj.get("levels_completed"),
        "win_levels": traj.get("win_levels"),
        "step_count": len(steps),
        "cost_usd": traj.get("total_usd"),
        "final_state": traj.get("final_state"),
        "goal_hypothesis": graph.get("goal_hypothesis"),
        "rules": interesting_rules,
        "objects": graph.get("objects", {}),
        "open_questions": graph.get("open_questions", []),
        "first_steps": sketch_first,
        "last_steps": sketch_last,
    }


def distill_and_store(
    scorecard_id: str,
    game_id: str,
    client,  # ClaudeClient -- avoids circular import
    store: MemoryStore | None = None,
    game_tags: list[str] | None = None,
) -> MemoryEntry | None:
    """Read a completed trajectory + final hypothesis graph, distill via Claude,
    and persist the resulting MemoryEntry. Returns the stored entry (or None
    on failure)."""
    import json as _json
    import logging
    import re

    logger = logging.getLogger(__name__)

    traj_path = PROJECT_ROOT / "data" / "trajectories" / f"{scorecard_id}_{game_id}.json"
    graph_path = PROJECT_ROOT / "data" / "hypotheses" / f"{scorecard_id}_{game_id}.json"
    prompt_path = PROJECT_ROOT / "prompts" / "memory_distillation.txt"

    if not traj_path.exists():
        logger.warning("No trajectory file at %s", traj_path)
        return None

    traj = _json.loads(traj_path.read_text())
    graph = _json.loads(graph_path.read_text()) if graph_path.exists() else {}
    payload = trajectory_to_distillation_prompt_payload(traj, graph)

    prompt = prompt_path.read_text().format(payload_json=_json.dumps(payload, indent=2))
    text, cost = client.reason(
        prompt=prompt,
        role="reasoner",
        tags={"phase": "distillation", "scorecard_id": scorecard_id, "game_id": game_id},
    )

    # Extract JSON from response
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        logger.warning("Distillation produced no JSON for %s/%s", scorecard_id, game_id)
        return None
    try:
        parsed = _json.loads(m.group(0))
    except _json.JSONDecodeError as e:
        logger.warning("Distillation JSON parse failed for %s/%s: %s", scorecard_id, game_id, e)
        return None

    entry = MemoryEntry(
        game_id=game_id,
        scorecard_id=scorecard_id,
        levels_completed=int(traj.get("levels_completed", 0)),
        win_levels=int(traj.get("win_levels", 0)),
        game_tags=game_tags or [],
        summary=parsed.get("summary", ""),
        confirmed_rules=list(parsed.get("confirmed_rules", []) or []),
        falsified_rules=list(parsed.get("falsified_rules", []) or []),
        meta_patterns=list(parsed.get("meta_patterns", []) or []),
        failure_modes=list(parsed.get("failure_modes", []) or []),
        cost_usd=float(traj.get("total_usd", 0.0)) + cost.usd,
        step_count=int(len(traj.get("steps", []))),
    )

    if store is None:
        store = MemoryStore()
    store.add(entry)
    logger.info("Distilled + stored memory entry for %s/%s (distill cost $%.4f)",
                scorecard_id, game_id, cost.usd)
    return entry
