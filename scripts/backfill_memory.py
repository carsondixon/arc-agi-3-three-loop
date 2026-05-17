"""Bulk-distill all existing hypothesis-loop trajectories into memory.db.

Useful before running Stage 3 (memory-augmented) so the memory store has
priors ready. Skips trajectories that have already been distilled.

Run:
    uv run python scripts/backfill_memory.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from arc_agi import Arcade

from src.claude_client import ClaudeClient
from src.memory import MemoryStore, distill_and_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = PROJECT_ROOT / "data" / "trajectories"

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    from dotenv import load_dotenv
    load_dotenv()

    import os
    arc_key = os.environ.get("ARC_API_KEY", "")

    client = ClaudeClient()
    store = MemoryStore()

    # Pull game metadata (for tags) -- one shot, then cache locally
    game_tags: dict[str, list[str]] = {}
    if arc_key:
        arcade = Arcade(arc_api_key=arc_key)
        try:
            envs = arcade.get_environments()
            for e in envs:
                # game_id format: "ls20-9607627b" -> we store under short name "ls20"
                short = e.game_id.split("-")[0]
                game_tags[short] = list(e.tags or [])
        except Exception as e:
            logger.warning("Could not fetch env metadata for tags: %s", e)

    # Already-distilled scorecards (avoid duplicates)
    existing = {(row[1].scorecard_id, row[1].game_id) for row in store.all_entries()}
    logger.info("Memory store already has %d entries", len(existing))

    distilled = 0
    skipped = 0
    failed = 0

    for traj_path in sorted(TRAJ_DIR.glob("*.json")):
        traj = json.loads(traj_path.read_text())
        if traj.get("mode") != "hypothesis-loop":
            continue
        scorecard_id = traj["scorecard_id"]
        game_id = traj["game_id"]
        if (scorecard_id, game_id) in existing:
            logger.info("Skipping already-distilled %s/%s", scorecard_id, game_id)
            skipped += 1
            continue
        logger.info("Distilling %s/%s ...", scorecard_id, game_id)
        try:
            entry = distill_and_store(
                scorecard_id=scorecard_id,
                game_id=game_id,
                client=client,
                store=store,
                game_tags=game_tags.get(game_id, []),
            )
            if entry:
                distilled += 1
            else:
                failed += 1
        except Exception as e:
            logger.exception("Failed to distill %s/%s: %s", scorecard_id, game_id, e)
            failed += 1

    logger.info("Backfill done: %d distilled, %d skipped, %d failed", distilled, skipped, failed)
    print(f"Memory store now has {len(store.all_entries())} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
