"""Pure lifecycle transitions shared by evaluation and repository commits."""

from dataclasses import replace

from evofact.core.models import EvolutionOperation as Op
from evofact.core.models import EvolutionProposal, SkillSpec, SkillStatus


def apply_candidate(skills: list[SkillSpec], proposal: EvolutionProposal) -> list[SkillSpec]:
    targets = proposal.target_skill_ids
    candidates = proposal.candidate_skills
    count = (len(targets), len(candidates))
    exact = {Op.ADD: (0, 1), Op.EDIT: (1, 1), Op.GENERALIZE: (1, 1),
             Op.SPECIALIZE: (1, 1), Op.RETIRE: (1, 0)}
    valid = count == exact.get(proposal.operation)
    if proposal.operation == Op.SPLIT:
        valid = count[0] == 1 and count[1] >= 2
    elif proposal.operation == Op.MERGE:
        valid = count[0] >= 2 and count[1] == 1
    if not valid or len(set(targets)) != len(targets):
        raise ValueError(f"invalid candidate operation/arity: {proposal.operation}")
    selected = [s for s in skills if s.name in targets or s.skill_id in targets]
    if len(selected) != len(targets):
        raise ValueError("candidate targets must resolve to existing skills")
    if any(s.status == SkillStatus.FROZEN for s in selected):
        raise ValueError("cannot evolve frozen skills")
    result = [s for s in skills if s not in selected]
    result.extend(replace(s, status=SkillStatus.ACTIVE) for s in candidates)
    if len({s.name for s in result}) != len(result):
        raise ValueError("candidate introduces duplicate skill names")
    if len({s.skill_id for s in result}) != len(result):
        raise ValueError("candidate introduces duplicate skill ids")
    return result
