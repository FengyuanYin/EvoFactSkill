"""One LLM call returns strategy decisions and complete dataset-style samples."""

import json
from dataclasses import asdict

from evofact.data.label_registry import LabelContractRegistry, fixture_binary_contract

from .prompts import GENERATOR_SYSTEM


def json_value(value):
    """函数作用：负责当前模块中的 `json_value` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class ChallengeGenerator:
    def build_request(
        self,
        samples,
        traces,
        attributions,
        *,
        config,
        episode_id,
        label_contract_registry: LabelContractRegistry | None = None,
    ):
        """函数作用：把 construction 样本、成功/失败轨迹和归因整理成单次生成请求。
        输入要求：`self` 应为已初始化的 `ChallengeGenerator` 实例；`samples`（未显式标注）需符合函数签名约定；`traces`（未显式标注）需符合函数签名约定；`attributions`（未显式标注）需符合函数签名约定；`config`（未显式标注）需以关键字传入并符合签名约定；`episode_id`（未显式标注）需以关键字传入并符合签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        registry = label_contract_registry or LabelContractRegistry((fixture_binary_contract(),))
        by_id = {s.sample_id: s for s in samples}
        reports = {a.trace_id: a for a in attributions}
        success, failure = [], []
        for trace in sorted(traces, key=lambda t: t.trace_id):
            source = by_id.get(trace.sample_id)
            if source is None:
                raise ValueError("generator trace outside construction")
            contract = registry.resolve_sample(source)
            gold = contract.normalize(source.label)
            source_row = asdict(source)
            source_row["metadata"] = {}
            source_row["label"] = gold
            row = {
                "source_sample": source_row,
                "trace": asdict(trace),
                "gold_label": gold,
                "label_contract": contract.prompt_view(),
                "label_contract_digest": contract.digest,
                "outcome": "success" if trace.decision.label == gold else "failure",
                "attribution": asdict(reports[trace.trace_id]),
            }
            row["trace"]["sample_public"] = source.public_view()
            (success if row["outcome"] == "success" else failure).append(row)
        chosen = []
        # Preserve both outcomes when available; never invent a successful trace.
        for i in range(max(len(success), len(failure))):
            for group in (failure, success):
                if i < len(group):
                    chosen.append(group[i])
        contracts = {
            registry.resolve_sample(source).key: registry.resolve_sample(source)
            for source in samples
        }
        return json_value(
            {
                "batch_size": config.batch_size,
                "id_prefix": f"gen-{episode_id}-",
                "trace_counts": {"success": len(success), "failure": len(failure)},
                "label_contracts": [
                    {
                        **contract.prompt_view(),
                        "generation_guidance": {
                            label.name: label.generation_guidance for label in contract.labels
                        },
                        "digest": contract.digest,
                    }
                    for contract in (contracts[key] for key in sorted(contracts))
                ],
                "examples": chosen[: config.max_trace_examples],
            }
        )

    async def generate(self, backend, request):
        """函数作用：调用 LLM 一次，在同一响应中取得完整样本和策略决策。
        输入要求：`self` 应为已初始化的 `ChallengeGenerator` 实例；`backend`（未显式标注）需符合函数签名约定；`request`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        # No proposal stage, deterministic fallback, text renderer or repair call.
        return await backend._call(GENERATOR_SYSTEM, request)
