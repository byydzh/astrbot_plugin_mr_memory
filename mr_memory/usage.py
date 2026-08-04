from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsageRecord:
    input_other: int = 0
    input_cached: int = 0
    output: int = 0

    @property
    def input(self) -> int:
        return self.input_other + self.input_cached

    @property
    def total(self) -> int:
        return self.input + self.output

    def as_dict(self) -> dict[str, int]:
        return {
            "input_other": self.input_other,
            "input_cached": self.input_cached,
            "output": self.output,
            "input": self.input,
            "total": self.total,
        }

    @classmethod
    def from_value(cls, value: Any) -> "TokenUsageRecord":
        """Normalize AstrBot, OpenAI SDK, or dict-shaped usage objects."""

        if value is None:
            return cls()
        if isinstance(value, dict):
            cached = int(
                value.get("input_cached", value.get("cached_tokens", 0)) or 0
            )
            if "input_other" in value:
                other = int(value.get("input_other", 0) or 0)
            else:
                prompt = int(
                    value.get("prompt_tokens", value.get("input_tokens", 0)) or 0
                )
                other = max(0, prompt - cached)
            output = int(
                value.get(
                    "output",
                    value.get("completion_tokens", value.get("output_tokens", 0)),
                )
                or 0
            )
            return cls(input_other=other, input_cached=cached, output=output)

        cached = int(getattr(value, "input_cached", 0) or 0)
        if hasattr(value, "input_other"):
            other = int(getattr(value, "input_other", 0) or 0)
        else:
            details = getattr(value, "prompt_tokens_details", None)
            if details is not None:
                cached = int(getattr(details, "cached_tokens", cached) or 0)
            prompt = int(
                getattr(value, "prompt_tokens", getattr(value, "input_tokens", 0))
                or 0
            )
            other = max(0, prompt - cached)
        output = int(
            getattr(
                value,
                "output",
                getattr(
                    value,
                    "completion_tokens",
                    getattr(value, "output_tokens", 0),
                ),
            )
            or 0
        )
        return cls(input_other=other, input_cached=cached, output=output)
