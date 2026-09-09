import random
from evofact.core.models import RoutingDecision, RunBudget, SkillKind, SkillSpec, SkillStatus, SkillUtility

class SkillRouter:
    def __init__(self,strategy:str="utility-aware",seed:int=42): self.strategy=strategy; self.seed=seed
    def select(self,sample:dict,skills:list[SkillSpec],utilities:dict[str,SkillUtility],budget:RunBudget)->RoutingDecision:
        candidates=[s for s in skills if s.kind==SkillKind.SPECIALIST and s.status in {SkillStatus.ACTIVE,SkillStatus.FROZEN} and matches_scope(s, sample)]
        if self.strategy=="all-experts": chosen=candidates[:budget.max_skills]
        elif self.strategy=="random": chosen=random.Random(f"{self.seed}:{sample.get('sample_id')}").sample(candidates,min(len(candidates),budget.max_skills))
        elif self.strategy=="static": chosen=[s for s in candidates if s.name in {"claim_decomposition","evidence_assessment","temporal_reasoning"}][:budget.max_skills]
        else:
            text=str(sample.get("text","")).casefold(); domain=str(sample.get("domain") or "")
            def score(skill):
                scope=1 if not skill.scope.domains or domain in skill.scope.domains else -10
                trigger=sum(t.weight for t in skill.triggers if t.pattern.casefold() in text)
                u=utilities.get(skill.skill_id,SkillUtility(skill.skill_id)); return scope+trigger+u.marginal_utility-.1*u.mean_cost-.2*u.negative_transfer_count
            chosen=sorted(candidates,key=lambda s:(-score(s),s.name))[:budget.max_skills]
        fallback=False
        if not chosen:
            chosen=[s for s in candidates if s.name=="claim_decomposition"][:1]; fallback=True
        selected=tuple(s.skill_id for s in chosen); rejected=tuple(s.skill_id for s in candidates if s.skill_id not in selected)
        return RoutingDecision(selected,rejected,{s.skill_id:self.strategy for s in chosen},1.0 if chosen else 0.0,budget,fallback)


def matches_scope(skill: SkillSpec, sample: dict) -> bool:
    scope = skill.scope
    return (
        (not scope.domains or sample.get("domain") in scope.domains)
        and (not scope.datasets or sample.get("dataset") in scope.datasets)
        and (not scope.temporal_windows or str(sample.get("metadata", {}).get("temporal_window")) in scope.temporal_windows)
    )
