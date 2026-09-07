from __future__ import annotations

from typing import Any

SENSITIVE_KEYS = {
    "label", "gold", "gold_label", "gold_native_label", "answer", "target",
    "expected", "expected_result", "ground_truth",
}


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: redact_sensitive(item)
            for key, item in value.items()
            if str(key).casefold() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value
