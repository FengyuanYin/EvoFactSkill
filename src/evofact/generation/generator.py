"""One LLM call returns strategy decisions and complete dataset-style samples."""
from dataclasses import asdict
import json

from evofact.experiments.runner import _gold
from .prompts import GENERATOR_SYSTEM


def json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class ChallengeGenerator:
    def build_request(self, samples, traces, attributions, *, config, episode_id):
        by_id = {s.sample_id: s for s in samples}
        reports = {a.trace_id: a for a in attributions}
        success, failure = [], []
        for trace in sorted(traces, key=lambda t: t.trace_id):
            source = by_id.get(trace.sample_public.get("sample_id"))
            if source is None:
                raise ValueError("generator trace outside construction")
            gold = _gold(source.label)
            source_row = asdict(source)
            source_row["metadata"] = {}
            source_row["label"] = gold
            row = {"source_sample": source_row, "trace": asdict(trace),
                   "gold_label": gold, "outcome": "success" if trace.decision.label == gold else "failure",
                   "attribution": asdict(reports[trace.trace_id])}
            row["trace"]["sample_public"] = source.public_view()
            row["trace"]["sample_public"]["metadata"] = {}
            (success if row["outcome"] == "success" else failure).append(row)
        chosen = []
        # Preserve both outcomes when available; never invent a successful trace.
        for i in range(max(len(success), len(failure))):
            for group in (failure, success):
                if i < len(group):
                    chosen.append(group[i])
        return json_value({"batch_size": config.batch_size, "id_prefix": f"gen-{episode_id}-",
                           "trace_counts": {"success": len(success), "failure": len(failure)},
                           "examples": chosen[:config.max_trace_examples]})

    async def generate(self, backend, request):
        # No proposal stage, deterministic fallback, text renderer or repair call.
        return await backend._call(GENERATOR_SYSTEM, request)
