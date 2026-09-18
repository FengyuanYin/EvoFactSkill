"""Structural provenance checks followed by one blind evidence review batch."""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from evofact.core.budget_models import BudgetRequest, UsageDetails
from evofact.core.generation_models import VerificationDecision, VerificationStatus
from evofact.core.models import Evidence, Sample
from evofact.data.label_registry import LabelContractRegistry, fixture_binary_contract
from evofact.runtime.backend import call_json_with_usage

from .generator import json_value
from .prompts import STRATEGIES, VERIFIER_SYSTEM

SAMPLE_FIELDS = set(Sample.__dataclass_fields__)
DECISION_REQUIRED_FIELDS = {
    "sample_id",
    "source_sample_id",
    "strategy",
    "reason",
    "source_trace_ids",
}
DECISION_OPTIONAL_FIELDS = {"source_label", "target_label"}


@dataclass(frozen=True)
class VerificationResult:
    accepted: list
    rejected: list
    reviews: list
    usage: UsageDetails

    def __iter__(self):
        """Keep compatibility with historical three-value unpacking."""
        yield self.accepted
        yield self.rejected
        yield self.reviews


def normalized(text):
    """函数作用：负责当前模块中的 `normalized` 处理，封装调用方需要复用的业务步骤。
    输入要求：`text`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return " ".join(text.casefold().split())


class ChallengeVerifier:
    def validate(
        self,
        response,
        request,
        *,
        config,
        forbidden=(),
        existing=(),
        label_contract_registry: LabelContractRegistry | None = None,
        allowed_strategies=None,
    ):
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
        registry = label_contract_registry or LabelContractRegistry((fixture_binary_contract(),))
        allowed_strategies = set(allowed_strategies or STRATEGIES)
        if any(
            not isinstance(d, dict)
            or not DECISION_REQUIRED_FIELDS <= set(d)
            or not set(d) <= DECISION_REQUIRED_FIELDS | DECISION_OPTIONAL_FIELDS
            for d in decisions
        ):
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
                    decision["strategy"] not in allowed_strategies
                    or not isinstance(decision["reason"], str)
                    or not decision["reason"].strip()
                ):
                    raise ValueError("invalid strategy or missing rationale")
                source = example["source_sample"]
                for field in (
                    "dataset",
                    "domain",
                    "event_id",
                    "published_at",
                    "evidence",
                    "label_schema_id",
                ):
                    if row[field] != source[field]:
                        raise ValueError(f"source {field} was changed or fabricated")
                probe = Sample(
                    sample_id=str(row["sample_id"]),
                    dataset=str(row["dataset"]),
                    text=str(row["text"]),
                    label=row["label"],
                    label_schema_id=str(row["label_schema_id"]),
                )
                contract = registry.resolve_sample(probe)
                source_label = contract.normalize(source["label"])
                target_label = contract.require_label(row["label"])
                if decision.get("source_label", source_label) != source_label:
                    raise ValueError("decision source_label does not match source gold")
                if decision.get("target_label", target_label) != target_label:
                    raise ValueError("decision target_label does not match generated label")
                if row["metadata"] != {}:
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

    async def verify(
        self,
        samples,
        backend,
        *,
        budget_manager=None,
        budget_sample_id="verifier",
        label_contract_registry: LabelContractRegistry | None = None,
    ):
        """函数作用：使用看不到生成标签的独立审核模型核对样本正文与证据是否一致。
        输入要求：`self` 应为已初始化的 `ChallengeVerifier` 实例；`samples`（未显式标注）需符合函数签名约定；`backend`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if not samples:
            return VerificationResult([], [], [], UsageDetails())
        registry = label_contract_registry or LabelContractRegistry((fixture_binary_contract(),))
        # Blind review excludes generated labels, decisions and trace outcomes.
        payload_rows = []
        for sample in samples:
            row = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "evidence": [asdict(e) for e in sample.evidence],
            }
            contract = registry.resolve_sample(sample)
            if contract.dataset_id != "fixture":
                row["label_contract"] = contract.prompt_view()
            payload_rows.append(row)
        payload = {"samples": payload_rows}
        reservation = None
        try:
            if budget_manager is not None:
                async with budget_manager.concurrency(budget_sample_id):
                    reservation = await budget_manager.reserve(
                        budget_sample_id,
                        BudgetRequest(calls=1, tokens=3000, purpose="frozen_verifier"),
                    )
                    response, usage = await asyncio.wait_for(
                        call_json_with_usage(backend, VERIFIER_SYSTEM, json_value(payload)),
                        budget_manager.limits.call_timeout_ms / 1000,
                    )
                await budget_manager.reconcile(reservation, usage)
                reservation = None
            else:
                response, usage = await call_json_with_usage(
                    backend, VERIFIER_SYSTEM, json_value(payload)
                )
        finally:
            if reservation is not None:
                await budget_manager.release(reservation)
        if (
            not isinstance(response, dict)
            or set(response) != {"reviews"}
            or not isinstance(response["reviews"], list)
        ):
            raise ValueError("invalid verifier response")
        reviews = response["reviews"]
        by_id: dict[str, VerificationDecision] = {}
        for row in reviews:
            if not isinstance(row, dict):
                raise ValueError("invalid or duplicate verifier review")
            sample_id = row.get("sample_id")
            reason = row.get("reason")
            if (
                not isinstance(sample_id, str)
                or sample_id in by_id
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("invalid or duplicate verifier review")
            sample = next((item for item in samples if item.sample_id == sample_id), None)
            if sample is None:
                raise ValueError("verifier reviewed an unknown sample")
            contract = registry.resolve_sample(sample)
            if set(row) == {"sample_id", "label", "reason"}:
                raw_label = row["label"]
                status = (
                    VerificationStatus.INDETERMINATE
                    if raw_label == "UNKNOWN"
                    else VerificationStatus.CLASSIFIED
                )
                predicted = None if status == VerificationStatus.INDETERMINATE else raw_label
            elif set(row) == {"sample_id", "status", "predicted_label", "reason"}:
                try:
                    status = VerificationStatus(str(row["status"]).casefold())
                except ValueError as exc:
                    raise ValueError("invalid verifier status") from exc
                predicted = row["predicted_label"]
            else:
                raise ValueError("invalid verifier review schema")
            if status == VerificationStatus.CLASSIFIED:
                predicted = contract.require_label(predicted)
            elif predicted is not None:
                raise ValueError("indeterminate review must not include a predicted label")
            by_id[sample_id] = VerificationDecision(
                sample_id,
                status,
                predicted,
                status == VerificationStatus.CLASSIFIED and predicted == sample.label,
                reason,
                "frozen-verifier-v2",
                contract.digest,
            )
        if set(by_id) != {s.sample_id for s in samples}:
            raise ValueError("verifier must review every sample exactly once")
        accepted, rejected = [], []
        for sample in samples:
            review = by_id[sample.sample_id]
            if review.accepted:
                accepted.append(sample)
            else:
                rejected.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "independent evidence verdict disagrees or is UNKNOWN",
                        "review": review.model_dump(),
                    }
                )
        return VerificationResult(
            accepted,
            rejected,
            [review.model_dump() for review in by_id.values()],
            usage,
        )
