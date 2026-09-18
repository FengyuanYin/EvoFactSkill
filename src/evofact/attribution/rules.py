from evofact.core.label_models import DatasetLabelContract, DecisionOrigin
from evofact.core.models import AttributionReport, ErrorType, InferenceTrace


def attribute_trace(
    trace: InferenceTrace,
    gold: str | int | None,
    contract: DatasetLabelContract | None = None,
) -> AttributionReport:
    """函数作用：对一次推理结果进行规则化错误归因，并返回结构化归因报告。
    输入要求：`trace`（InferenceTrace）需符合函数签名约定；`gold`（str | int | None）需符合函数签名约定。
    输出：返回 `AttributionReport` 类型结果；校验或下游调用失败时异常向上传递。"""

    # 分别收集错误类型、支持归因的说明，以及可能需要负责的技能 ID。
    errors = []
    evidence = []
    responsible = []

    normalized = str(gold) if contract is not None else str(gold).upper()
    allowed = contract.allowed_labels if contract is not None else ("REAL", "FAKE")
    if normalized not in allowed:
        # gold 未预先映射成标准标签，无法继续判断预测是否正确。
        errors.append(ErrorType.LABEL_MAPPING_ERROR)
        evidence.append("gold label is not canonical")
    elif trace.decision.origin == DecisionOrigin.RUNTIME or (
        contract is None and trace.decision.label == "ABSTAIN"
    ):
        # 对有明确真实标签的样本弃权，视为一次错误弃权。
        errors.append(ErrorType.ABSTENTION_ERROR)
        evidence.append("decision abstained on a labeled item")
    elif trace.decision.label != normalized:
        # 最终预测错误时，根据路由结果和专家报告进一步定位错误环节。
        if not trace.routing.selected_skill_ids:
            # 路由器没有选择任何专家技能。
            errors.append(ErrorType.ROUTING_MISS)
        elif not trace.specialist_reports:
            # 已选择专家，但没有得到任何有效的专家报告。
            errors.append(ErrorType.EVIDENCE_MISS)
        else:
            if contract is None:
                majority = max(
                    ("real", "fake"),
                    key=lambda value: sum(
                        report.assessment == value for report in trace.specialist_reports
                    ),
                )
                errors.append(
                    ErrorType.JUDGE_AGGREGATION_ERROR
                    if majority.upper() == normalized
                    else ErrorType.REASONING_ERROR
                )
            else:
                weak_findings = not trace.aggregated_evidence or any(
                    any(
                        token in finding.finding_type.casefold()
                        for token in ("missing", "insufficient", "unresolved", "unknown")
                    )
                    for report in trace.specialist_reports
                    for finding in report.findings
                )
                errors.append(
                    ErrorType.REASONING_ERROR
                    if weak_findings
                    else ErrorType.JUDGE_AGGREGATION_ERROR
                )
            # 当前是粗粒度归因：将所有参与分析的专家都列为可能责任方。
            responsible.extend(r.skill_id for r in trace.specialist_reports)

    # 从运行期错误文本中识别证据幻觉和使用未来证据的问题。
    if any("hallucin" in e.casefold() for e in trace.errors):
        errors.append(ErrorType.EVIDENCE_HALLUCINATION)
    if any("future evidence" in e.casefold() for e in trace.errors):
        errors.append(ErrorType.TEMPORAL_LEAKAGE)

    # 如果依赖回退技能后仍预测错误，也将其标记为路由遗漏。
    if trace.routing.fallback_used and normalized in allowed and trace.decision.label != normalized:
        errors.append(ErrorType.ROUTING_MISS)

    # dict.fromkeys 在保持首次出现顺序的同时去除重复错误和技能 ID。
    return AttributionReport(
        trace.trace_id,
        tuple(dict.fromkeys(errors)),
        tuple(dict.fromkeys(responsible)),
        0.9 if errors else 1.0,
        tuple(evidence),
    )
