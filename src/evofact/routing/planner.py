from __future__ import annotations

from typing import Protocol

from evofact.core.models import RunBudget, SkillSpec, SkillUtility
from evofact.runtime.backend import BackendResult


class SkillPlanner(Protocol):
    async def plan(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult: ...
