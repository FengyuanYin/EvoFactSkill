from dataclasses import replace

from evofact.core.models import SkillUtility


class UtilityTracker:
    def __init__(self, retire_after: int = 3):
        self.values: dict[str, SkillUtility] = {}
        self.retire_after = retire_after

    def update(
        self,
        skill_id: str,
        *,
        success: bool,
        delta: float,
        cost: float,
        domain: str | None = None,
        window: str | None = None,
    ) -> SkillUtility:
        old = self.values.get(skill_id, SkillUtility(skill_id))
        uses = old.uses + 1
        du = dict(old.domain_utility)
        tu = dict(old.temporal_utility)
        if domain:
            du[domain] = (du.get(domain, 0) * (uses - 1) + delta) / uses
        if window:
            tu[window] = (tu.get(window, 0) * (uses - 1) + delta) / uses
        value = replace(
            old,
            uses=uses,
            successes=old.successes + int(success),
            marginal_utility=(old.marginal_utility * (uses - 1) + delta) / uses,
            mean_cost=(old.mean_cost * (uses - 1) + cost) / uses,
            domain_utility=du,
            temporal_utility=tu,
            negative_transfer_count=old.negative_transfer_count + int(delta < 0),
        )
        self.values[skill_id] = value
        return value

    def recommendation(self, skill_id: str) -> str:
        value = self.values[skill_id]
        return (
            "retire"
            if value.negative_transfer_count >= self.retire_after and value.marginal_utility < 0
            else ("downweight" if value.marginal_utility < 0 else "keep")
        )
