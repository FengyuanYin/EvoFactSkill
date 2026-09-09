from evofact.core.models import AttributionReport, ErrorType, InferenceTrace


def attribute_trace(trace: InferenceTrace, gold: str | int | None) -> AttributionReport:
    errors = []
    evidence = []
    responsible = []
    normalized = str(gold).upper()
    if normalized not in {"REAL", "FAKE"}:
        errors.append(ErrorType.LABEL_MAPPING_ERROR)
        evidence.append("gold label is not canonical")
    elif trace.decision.label == "ABSTAIN":
        errors.append(ErrorType.ABSTENTION_ERROR)
        evidence.append("decision abstained on a labeled item")
    elif trace.decision.label != normalized:
        if not trace.routing.selected_skill_ids:
            errors.append(ErrorType.ROUTING_MISS)
        elif not trace.specialist_reports:
            errors.append(ErrorType.EVIDENCE_MISS)
        else:
            majority = max(
                ("real", "fake"),
                key=lambda x: sum(r.assessment == x for r in trace.specialist_reports),
            )
            errors.append(
                ErrorType.JUDGE_AGGREGATION_ERROR
                if majority.upper() == normalized
                else ErrorType.REASONING_ERROR
            )
            responsible.extend(r.skill_id for r in trace.specialist_reports)
    if any("hallucin" in e.casefold() for e in trace.errors):
        errors.append(ErrorType.EVIDENCE_HALLUCINATION)
    if any("future evidence" in e.casefold() for e in trace.errors):
        errors.append(ErrorType.TEMPORAL_LEAKAGE)
    if (
        trace.routing.fallback_used
        and normalized in {"REAL", "FAKE"}
        and trace.decision.label != normalized
    ):
        errors.append(ErrorType.ROUTING_MISS)
    return AttributionReport(
        trace.trace_id,
        tuple(dict.fromkeys(errors)),
        tuple(dict.fromkeys(responsible)),
        0.9 if errors else 1.0,
        tuple(evidence),
    )
