"""Structural provenance checks followed by one blind evidence review batch."""

import json
from dataclasses import asdict
from datetime import datetime

from evofact.core.models import Evidence, Sample

from .generator import json_value
from .prompts import STRATEGIES, VERIFIER_SYSTEM

SAMPLE_FIELDS = set(Sample.__dataclass_fields__)
DECISION_FIELDS = {"sample_id", "source_sample_id", "strategy", "reason", "source_trace_ids"}


def normalized(text):
    """函数作用：负责当前模块中的 `normalized` 处理，封装调用方需要复用的业务步骤。
    输入要求：`text`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return " ".join(text.casefold().split())


class ChallengeVerifier:
    def validate(self, response, request, *, config, forbidden=(), existing=()):
        """函数作用：校验生成响应的结构、来源、策略、重复内容和留出数据隔离。
        输入要求：`self` 应为已初始化的 `ChallengeVerifier` 实例；`response`（未显式标注）需符合函数签名约定；`request`（未显式标注）需符合函数签名约定；`config`（未显式标注）需以关键字传入并符合签名约定；`forbidden`（未显式标注，默认 `()`）需以关键字传入并符合签名约定；`existing`（未显式标注，默认 `()`）需以关键字传入并符合签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if not isinstance(response, dict) or set(response) != {"samples", "decisions"}:
            raise ValueError("generator must return samples and decisions")
        rows, decisions = response["samples"], response["decisions"]
        if not isinstance(rows, list) or not isinstance(decisions, list):
            raise ValueError("generator samples/decisions must be lists")
        if len(rows) > config.batch_size or len(rows) != len(decisions):
            raise ValueError("generator batch budget or decision count mismatch")
        if any(not isinstance(d, dict) or set(d) != DECISION_FIELDS for d in decisions):
            raise ValueError("invalid strategy decision schema")
        if any(not isinstance(d["sample_id"], str) for d in decisions):
            raise ValueError("invalid decision sample ID")
        by_decision = {d["sample_id"]: d for d in decisions}
        if len(by_decision) != len(decisions):
            raise ValueError("duplicate strategy decision IDs")
        examples = {e["source_sample"]["sample_id"]: e for e in request["examples"]}
        trace_ids = {e["trace"]["trace_id"] for e in request["examples"]}
        seen_ids = {s.sample_id for s in existing}
        seen_texts = {normalized(s.text) for s in existing}
        accepted, rejected = [], []
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, dict) or set(row) != SAMPLE_FIELDS:
                    raise ValueError("invalid Sample fields")
                sid = row["sample_id"]
                if (
                    not isinstance(sid, str)
                    or not sid.startswith(request["id_prefix"])
                    or sid in seen_ids
                ):
                    raise ValueError("invalid or duplicate generated sample ID")
                seen_ids.add(sid)
                decision = by_decision.get(sid)
                if decision is None:
                    raise ValueError("missing strategy decision")
                if not isinstance(decision["source_sample_id"], str):
                    raise ValueError("invalid source ID")
                example = examples.get(decision["source_sample_id"])
                if example is None:
                    raise ValueError("source is outside supplied construction traces")
                refs = decision["source_trace_ids"]
                if (
                    not isinstance(refs, list)
                    or not all(isinstance(t, str) for t in refs)
                    or not set(refs) <= trace_ids
                    or example["trace"]["trace_id"] not in refs
                ):
                    raise ValueError("invalid source trace provenance")
                if (
                    decision["strategy"] not in STRATEGIES
                    or not isinstance(decision["reason"], str)
                    or not decision["reason"].strip()
                ):
                    raise ValueError("invalid strategy or missing rationale")
                source = example["source_sample"]
                for field in ("dataset", "domain", "event_id", "published_at", "evidence"):
                    if row[field] != source[field]:
                        raise ValueError(f"source {field} was changed or fabricated")
                if row["metadata"] != {} or row["label"] not in ("REAL", "FAKE"):
                    raise ValueError("invalid label or audit metadata leak")
                if not row["evidence"] or any(not e["text"].strip() for e in row["evidence"]):
                    raise ValueError("no source evidence for independent verification")
                text = row["text"]
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or len(text) > config.max_text_chars
                ):
                    raise ValueError("invalid generated text length")
                if normalized(text) in seen_texts:
                    raise ValueError("duplicate generated or construction text")
                blob = normalized(
                    json.dumps({"sample": row, "decision": decision}, ensure_ascii=False)
                )
                for heldout in forbidden:
                    if heldout.sample_id.casefold() in blob or (
                        len(normalized(heldout.text)) >= 12 and normalized(heldout.text) in blob
                    ):
                        raise ValueError("generated content contains held-out material")
                evidence = []
                for e in row["evidence"]:
                    e = dict(e)
                    e["published_at"] = (
                        datetime.fromisoformat(e["published_at"]) if e["published_at"] else None
                    )
                    # Stance is claim-relative and must not leak the source claim's verdict.
                    e["stance"] = "unknown"
                    evidence.append(Evidence(**e))
                data = dict(row)
                data["evidence"] = tuple(evidence)
                data["published_at"] = (
                    datetime.fromisoformat(data["published_at"]) if data["published_at"] else None
                )
                accepted.append(Sample(**data))
                seen_texts.add(normalized(text))
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append({"index": index, "reason": str(exc)})
        return accepted, rejected

    async def verify(self, samples, backend):
        """函数作用：使用看不到生成标签的独立审核模型核对样本正文与证据是否一致。
        输入要求：`self` 应为已初始化的 `ChallengeVerifier` 实例；`samples`（未显式标注）需符合函数签名约定；`backend`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if not samples:
            return [], [], []
        # Blind review excludes generated labels, decisions and trace outcomes.
        payload = {
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "text": s.text,
                    "evidence": [asdict(e) for e in s.evidence],
                }
                for s in samples
            ]
        }
        response = await backend._call(VERIFIER_SYSTEM, json_value(payload))
        if (
            not isinstance(response, dict)
            or set(response) != {"reviews"}
            or not isinstance(response["reviews"], list)
        ):
            raise ValueError("invalid verifier response")
        reviews = response["reviews"]
        by_id = {}
        for row in reviews:
            if (
                not isinstance(row, dict)
                or set(row) != {"sample_id", "label", "reason"}
                or not isinstance(row["sample_id"], str)
                or row["sample_id"] in by_id
                or row["label"] not in ("REAL", "FAKE", "UNKNOWN")
                or not isinstance(row["reason"], str)
                or not row["reason"].strip()
            ):
                raise ValueError("invalid or duplicate verifier review")
            by_id[row["sample_id"]] = row
        if set(by_id) != {s.sample_id for s in samples}:
            raise ValueError("verifier must review every sample exactly once")
        accepted, rejected = [], []
        for sample in samples:
            review = by_id[sample.sample_id]
            if review["label"] == sample.label:
                accepted.append(sample)
            else:
                rejected.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "independent evidence verdict disagrees or is UNKNOWN",
                        "review": review,
                    }
                )
        return accepted, rejected, reviews
