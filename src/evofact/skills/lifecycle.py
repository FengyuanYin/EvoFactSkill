from evofact.core.models import EvolutionOperation, EvolutionProposal, SkillStatus

from .repository import SkillRepository

ARITY = {
    EvolutionOperation.ADD: (0, 1),
    EvolutionOperation.EDIT: (1, 1),
    EvolutionOperation.SPLIT: (1, 2),
    EvolutionOperation.MERGE: (2, 1),
    EvolutionOperation.GENERALIZE: (1, 1),
    EvolutionOperation.SPECIALIZE: (1, 1),
    EvolutionOperation.RETIRE: (1, 0),
    EvolutionOperation.ROLLBACK: (1, 0),
}


def execute(
    repo: SkillRepository, proposal: EvolutionProposal, *, rollback_snapshot: str | None = None
) -> list[str]:
    targets, candidates = len(proposal.target_skill_ids), len(proposal.candidate_skills)
    min_targets, min_candidates = ARITY[proposal.operation]
    if targets < min_targets or candidates < min_candidates:
        raise ValueError(f"invalid arity for {proposal.operation}")
    if proposal.operation == EvolutionOperation.RETIRE:
        return [repo.retire(proposal.target_skill_ids[0], proposal.rationale)]
    if proposal.operation == EvolutionOperation.ROLLBACK:
        if not rollback_snapshot:
            raise ValueError("rollback snapshot required")
        repo.rollback(proposal.target_skill_ids[0], rollback_snapshot)
        return [rollback_snapshot]
    return [
        repo.save(
            skill
            if skill.status == SkillStatus.CANDIDATE
            else __import__("dataclasses").replace(skill, status=SkillStatus.CANDIDATE),
            proposal.operation.value,
        )
        for skill in proposal.candidate_skills
    ]
