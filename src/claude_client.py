"""Anthropic SDK wrapper with cost tracking.

Reads pinned model IDs from config/models.yaml. Logs each call's token
counts + cost to results/cost_report.jsonl. Stage 1 keeps this minimal:
no caching, no streaming, no retry beyond what the SDK does. Stage 3+ will
add prompt caching once the prompts stabilize.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
COST_LOG_PATH = Path(__file__).resolve().parent.parent / "results" / "cost_report.jsonl"

# Per-1M-token USD pricing for Claude (as of 2026-05). Update if Anthropic changes pricing.
PRICING_PER_1M = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
}


@dataclass
class CallCost:
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    usd: float


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _compute_cost(model: str, usage: anthropic.types.Usage) -> CallCost:
    pricing = PRICING_PER_1M.get(model)
    if pricing is None:
        # Unknown model: log zero so we don't crash; user should update PRICING_PER_1M.
        return CallCost(model, usage.input_tokens, usage.output_tokens, 0, 0, 0.0)

    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    usd = (
        usage.input_tokens * pricing["input"] / 1_000_000
        + usage.output_tokens * pricing["output"] / 1_000_000
        + cache_write * pricing["cache_write"] / 1_000_000
        + cache_read * pricing["cache_read"] / 1_000_000
    )
    return CallCost(model, usage.input_tokens, usage.output_tokens, cache_write, cache_read, usd)


def _log_cost(cost: CallCost, tags: dict[str, Any] | None = None) -> None:
    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": cost.model,
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "cache_write_tokens": cost.cache_write_tokens,
        "cache_read_tokens": cost.cache_read_tokens,
        "usd": cost.usd,
        "tags": tags or {},
    }
    with COST_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


class ClaudeClient:
    """Thin wrapper over anthropic.Anthropic with cost logging."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. Add it to .env "
                "(generate at https://console.anthropic.com/settings/keys)."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._config = _load_config()

    def _resolve_params(self, role: str) -> tuple[str, int, float]:
        model = self._config["models"][f"primary_{role}"] if role == "reasoner" else (
            self._config["models"].get(role) or self._config["models"]["primary_reasoner"]
        )
        max_tokens = self._config["max_tokens"].get(role, 4096)
        temperature = self._config["temperature"].get(role, 0.5)
        return model, max_tokens, temperature

    def reason(
        self,
        prompt: str,
        role: str = "reasoner",
        system: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> tuple[str, CallCost]:
        """Send a single-turn text prompt. Returns (text, cost)."""
        model, max_tokens, temperature = self._resolve_params(role)

        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        cost = _compute_cost(model, response.usage)
        _log_cost(cost, tags=tags)
        return text, cost

    def reason_vision(
        self,
        prompt: str,
        images: list[bytes],
        role: str = "reasoner",
        system: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> tuple[str, CallCost]:
        """Send a single-turn prompt with one or more PNG images.

        `images` are raw PNG bytes (rendered by perception.grid_to_image /
        diff_image). They are sent as Anthropic image content blocks, followed
        by the text prompt, in a single user message -- one call per turn.
        """
        import base64

        model, max_tokens, temperature = self._resolve_params(role)

        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(png).decode("ascii"),
                },
            }
            for png in images
        ]
        content.append({"type": "text", "text": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        cost = _compute_cost(model, response.usage)
        _log_cost(cost, tags=tags)
        return text, cost
