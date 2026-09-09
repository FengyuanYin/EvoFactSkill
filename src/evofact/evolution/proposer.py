import hashlib
from dataclasses import replace

from evofact.core.models import (
    AttributionReport,
    EvolutionOperation,
    EvolutionProposal,
    SkillKind,
    SkillSpec,
    SkillStatus,
)


def propose_from_cluster(
    cluster_id: str, reports: list[AttributionReport], bank: list[SkillSpec]
) -> EvolutionProposal:
    responsible = [s for r in reports for s in r.responsible_skill_ids]
    target = (
        next(
            (
                sid
                for sid in responsible
                if responsible.count(sid) == max(map(responsible.count, responsible))
            ),
            None,
        )
        if responsible
        else None
    )
    trace_ids = tuple(r.trace_id for r in reports)
    error = next((e.value for r in reports for e in r.error_types), "unclassified")
    if target and any(s.skill_id == target for s in bank):
        old = next(s for s in bank if s.skill_id == target)
        candidate = replace(
            old,
            skill_id=hashlib.sha256(f"{old.skill_id}:{cluster_id}".encode()).hexdigest()[:20],
            version=_bump(old.version),
            status=SkillStatus.CANDIDATE,
            instructions=old.instructions
            + f"\nHandle recurring {error} cases by explicitly checking the missing condition.",
            parent_ids=(old.skill_id,),
        )
        operation = EvolutionOperation.EDIT
        targets = (old.name,)
    else:
        name = f"discovered_{error}"[:63]
        candidate = SkillSpec(
            hashlib.sha256(f"{name}:{cluster_id}".encode()).hexdigest()[:20],
            name,
            SkillKind.SPECIALIST,
            "0.1.0",
            SkillStatus.CANDIDATE,
            f"Detect and mitigate {error} without using labels or dataset identity.",
        )
        operation = EvolutionOperation.ADD
        targets = ()
    return EvolutionProposal(
        f"proposal-{hashlib.sha1(cluster_id.encode()).hexdigest()[:10]}",
        operation,
        f"Recurring error cluster: {error}",
        targets,
        (candidate,),
        trace_ids,
        cluster_id,
    )


def _bump(version: str) -> str:
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except ValueError:
        return version + ".1"
