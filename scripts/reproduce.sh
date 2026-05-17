#!/usr/bin/env bash
# scripts/reproduce.sh — reproduce the latest published scorecard.
#
# This is the legitimacy guarantee. Anyone with $ANTHROPIC_API_KEY and $ARC_API_KEY
# can run this script from a fresh clone and produce a scorecard URL comparable to
# the latest entry in results/scorecards/.
#
# Stage 0: stub. Will be implemented in Stage 1 once src/agent.py exists.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running reproduce.sh}"
: "${ARC_API_KEY:?Set ARC_API_KEY (from https://arcprize.org) before running reproduce.sh}"

echo "[reproduce] Stage 0 stub. Run after Stage 1 (Naked Baseline) lands."
echo "[reproduce] Will execute: uv run python -m src.agent --games=all --commit-scorecard"
exit 1
