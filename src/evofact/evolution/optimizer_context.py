from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# Ground truth and dataset label governance never reach the optimizer. Matching is by
# substring so an unexpected key such as ``expected_label`` stays covered.
SENSITIVE_TOKENS = ("gold", "label", "api_key", "secret", "held_out", "meta_test", "final_test")

# …except the model's own prediction, which is the signal an error-driven optimizer needs:
# knowing whether the system answered REAL, FAKE or ABSTAIN is not ground-truth leakage.
PREDICTION_KEYS = frozenset({"label", "labels", "predicted", "predicted_label", "prediction"})


def _redacted(key: object) -> bool:
    lowered = str(key).casefold()
    if lowered in PREDICTION_KEYS:
        return False
    return any(token in lowered for token in SENSITIVE_TOKENS)


def _clip(text: object, limit: int) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: max(limit - 3, 0)] + "..."


def _sanitize(value: Any, *, max_text_chars: int) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item, max_text_chars=max_text_chars)
            for key, item in value.items()
            if not _redacted(key)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, max_text_chars=max_text_chars) for item in value]
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "content": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, str):
        if Path(value).is_absolute():
            return "<redacted-path>"
        return value[:max_text_chars]
    return value


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass instance or an already-serialized dict."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _trace_view(trace: Any, *, max_text_chars: int) -> dict[str, Any]:
    """Project one trace onto the fields that explain a decision.

    Cost, tokens, execution plans and node bookkeeping are omitted: they dominate the
    payload without telling the optimizer why an answer was wrong.
    """
    public = _field(trace, "sample_public") or {}
    decision = _field(trace, "decision")
    routing = _field(trace, "routing")
    origin = _field(decision, "origin")
    return {
        "trace_id": _field(trace, "trace_id"),
        "sample": {
            "text": _clip(_field(public, "text", ""), max_text_chars),
            "domain": _field(public, "domain"),
        },
        "routing": {
            "selected_skill_ids": [
                str(item) for item in (_field(routing, "selected_skill_ids") or ())
            ],
            "reasons": {
                str(key): _clip(value, max_text_chars)
                for key, value in sorted((_field(routing, "reasons") or {}).items())
            },
            "confidence": _field(routing, "confidence"),
            "fallback_used": bool(_field(routing, "fallback_used", False)),
        },
        "decision": {
            "label": _field(decision, "label"),
            "origin": str(getattr(origin, "value", origin)),
            "confidence": _field(decision, "confidence"),
            "rationale": _clip(_field(decision, "rationale", ""), max_text_chars),
        },
        "specialist_reports": _sanitize(
            tuple(_field(trace, "specialist_reports") or ()),
            max_text_chars=max_text_chars,
        ),
        "errors": [_clip(item, max_text_chars) for item in list(_field(trace, "errors") or ())[:5]],
    }


def _context_chars(context: dict[str, Any]) -> int:
    return len(json.dumps(context, ensure_ascii=False, default=str))


def build_optimizer_context(
    *,
    target,
    reports=(),
    traces=(),
    distillation: dict[str, Any] | None = None,
    prior_proposals=(),
    audits=(),
    max_reports: int = 12,
    max_traces: int = 12,
    max_text_chars: int = 8000,
    max_total_chars: int = 300000,
) -> dict[str, Any]:
    """Build the optimizer prompt payload: the target package plus the evidence for it."""
    context: dict[str, Any] = {
        "target": _sanitize(target, max_text_chars=max_text_chars),
        "reports": _sanitize(tuple(reports)[:max_reports], max_text_chars=max_text_chars),
        "traces": [
            _trace_view(trace, max_text_chars=max_text_chars)
            for trace in tuple(traces)[:max_traces]
        ],
    }
    if distillation:
        context["distillation"] = _sanitize(distillation, max_text_chars=max_text_chars)
    if prior_proposals:
        context["prior_proposals"] = _sanitize(
            tuple(prior_proposals), max_text_chars=max_text_chars
        )
    if audits:
        context["generation_audits"] = _sanitize(tuple(audits)[:50], max_text_chars=max_text_chars)

    # Drop the least load-bearing evidence first; the target Package is never trimmed.
    while _context_chars(context) > max_total_chars:
        if context["traces"]:
            context["traces"].pop()
        elif context["reports"]:
            context["reports"].pop()
        elif context.get("prior_proposals"):
            context["prior_proposals"].pop()
        elif context.get("generation_audits"):
            context.pop("generation_audits")
        elif context.get("distillation"):
            context.pop("distillation")
        else:
            raise ValueError("optimizer context cannot fit max_total_chars")

    return context


def build_skillbank_optimizer_context(
    *,
    packages,
    traces,
    max_text_chars=8000,
    max_instructions_chars=60000,
    max_total_chars=300000,
    feedback=None,
    prior_proposals=(),
):
    """Supply training traces and the current bank without precomputed diagnoses."""
    skills = []
    for package in packages:
        view = asdict(package)
        for item in view["files"]:
            try:
                item["content"] = item["content"].decode("utf-8")
            except UnicodeDecodeError:
                pass
        skills.append(_sanitize(view, max_text_chars=max_instructions_chars))
    context = {
        "skills": skills,
        "traces": [_sanitize(trace, max_text_chars=max_text_chars) for trace in traces],
    }
    if prior_proposals:
        # Newest first, so context trimming preserves the most recent decisions.
        context["prior_proposals"] = _sanitize(
            list(reversed(tuple(prior_proposals)[-12:])), max_text_chars=max_text_chars
        )
    if feedback is not None:
        context["validation_feedback"] = _sanitize(feedback, max_text_chars=max_text_chars)
    for trace in context["traces"]:
        public = trace.get("sample_public", {})
        for key in list(public):
            if str(key).casefold() in PREDICTION_KEYS:
                public.pop(key)
    if feedback is not None:
        # Keep the latest failed attempt; prune older feedback first.
        while _context_chars(context) > max_total_chars and len(context["validation_feedback"]) > 1:
            context["validation_feedback"].pop()
        if context["validation_feedback"]:
            latest = context["validation_feedback"][0]
            while _context_chars(context) > max_total_chars:
                groups = [latest.get("baseline_traces", []), latest.get("candidate_traces", [])]
                group = max(groups, key=len)
                if len(group) <= 1:
                    break
                group.pop()
    while _context_chars(context) > max_total_chars and len(context["traces"]) > 1:
        context["traces"].pop()
    while _context_chars(context) > max_total_chars and context.get("prior_proposals"):
        context["prior_proposals"].pop()
    if _context_chars(context) > max_total_chars:
        raise ValueError("current Skill bank and training trace cannot fit max_total_chars")
    return context


__all__ = ["build_optimizer_context", "build_skillbank_optimizer_context"]
