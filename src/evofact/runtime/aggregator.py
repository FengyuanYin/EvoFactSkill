from evofact.core.models import Evidence, SpecialistReport


def aggregate_evidence(reports: tuple[SpecialistReport, ...]) -> tuple[Evidence, ...]:
    """Deterministically deduplicate evidence while preserving specialist order."""
    seen = set()
    result = []
    for report in reports:
        for evidence in report.evidence:
            key = (
                " ".join(evidence.text.casefold().split()),
                evidence.source,
                evidence.published_at,
            )
            if key not in seen:
                seen.add(key)
                result.append(evidence)
    return tuple(result)
