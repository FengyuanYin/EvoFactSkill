from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {"label", "gold", "api_key", "secret", "held_out", "meta_test", "final_test"}


def _sanitize(value: Any, *, max_text_chars: int) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in SENSITIVE_KEYS):
                continue
            result[str(key)] = _sanitize(item, max_text_chars=max_text_chars)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, max_text_chars=max_text_chars) for item in value]
    if isinstance(value, str):
        if Path(value).is_absolute():
            return "<redacted-path>"
        return value[:max_text_chars]
    return value


def build_optimizer_context(
    *,
    target,
    reports=(),
    traces=(),
    audits=(),
    max_reports: int = 12,
    max_traces: int = 12,
    max_audits: int = 50,
    max_text_chars: int = 8000,
) -> dict[str, Any]:
    return {
        "target": _sanitize(target, max_text_chars=max_text_chars),
        "reports": _sanitize(tuple(reports)[:max_reports], max_text_chars=max_text_chars),
        "traces": _sanitize(tuple(traces)[:max_traces], max_text_chars=max_text_chars),
        "generation_audits": _sanitize(tuple(audits)[:max_audits], max_text_chars=max_text_chars),
    }
