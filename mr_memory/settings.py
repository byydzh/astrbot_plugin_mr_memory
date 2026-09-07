"""Normalize supported settings without overwriting explicitly configured values."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


_SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8-sig"))
DEFAULTS = {name: specification["default"] for name, specification in _SCHEMA.items()}

# Original public names remain authoritative whenever both are supplied.
# background_interval_seconds is a polling interval, not a rename of the
# message accumulation deadline maintenance_interval_seconds.
_RENAMED = {
    "memory_timeout_seconds": "local_serving_timeout_seconds",
    "memory_max_turns": "max_loop_steps",
    "background_min_messages": "auto_distillation_min_pending",
    "background_batch_size": "distillation_max_messages",
    "background_timeout_seconds": "maintenance_llm_timeout_seconds",
    "background_max_tokens": "distillation_max_output_tokens",
    "background_daily_tokens": "private_daily_token_budget",
}


def normalize_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return supported canonical fields; never mutate the caller's config.

    Public defaults fill only missing values. The old minutes/hours controls
    retain their original precedence over their hidden seconds counterparts.
    A token-limit alias migrates its value, not the old usage ledger: callers
    must retain the distinct distillation and feedback accounting histories.
    """
    result = deepcopy(DEFAULTS)
    for newer, original in _RENAMED.items():
        if original not in raw and newer in raw:
            result[original] = deepcopy(raw[newer])
    result.update({key: deepcopy(value) for key, value in raw.items() if key in DEFAULTS})
    if "maintenance_interval_minutes" in raw:
        result["maintenance_interval_seconds"] = int(float(raw["maintenance_interval_minutes"]) * 60)
    if "feedback_window_hours" in raw:
        result["feedback_window_seconds"] = int(float(raw["feedback_window_hours"]) * 3600)
    if str(result["embedding_query_prompt_name"]).casefold() == "auto":
        result["embedding_query_prompt_name"] = (
            "web_search_query" if "harrier" in str(result["embedding_model_name"]).casefold() else ""
        )
    return result
