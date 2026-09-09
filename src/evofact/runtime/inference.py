import hashlib

from evofact.core.models import InferenceTrace, RunBudget, Sample, SkillKind, SkillSpec, UsageRecord
from evofact.routing.router import SkillRouter, matches_scope

from .aggregator import aggregate_evidence


class InferenceRuntime:
    def __init__(self, backend, router: SkillRouter, skills: list[SkillSpec], utilities=None):
        self.backend = backend
        self.router = router
        self.skills = skills
        self.utilities = utilities or {}

    async def infer(self, sample: Sample, budget: RunBudget = RunBudget()) -> InferenceTrace:
        public = sample.public_view()
        decision = self.router.select(public, self.skills, self.utilities, budget)
        by_id = {s.skill_id: s for s in self.skills}
        reports = []
        usage = UsageRecord()
        errors = []
        for sid in decision.selected_skill_ids:
            try:
                result = await self.backend.analyze(public, by_id[sid])
                reports.append(result.value)
                usage = _merge(usage, result.usage)
            except Exception as exc:
                errors.append(f"specialist {sid}: {type(exc).__name__}: {exc}")
        judges = [
            s
            for s in self.skills
            if s.kind == SkillKind.JUDGE
            and s.status.value in {"active", "frozen"}
            and matches_scope(s, public)
        ]
        if not judges:
            raise RuntimeError("no active judge skill")
        result = await self.backend.judge(public, tuple(reports), judges[0])
        usage = _merge(usage, result.usage)
        trace_id = hashlib.sha256(
            f"{sample.sample_id}:{','.join(decision.selected_skill_ids)}".encode()
        ).hexdigest()[:20]
        return InferenceTrace(
            trace_id,
            public,
            decision,
            tuple(reports),
            result.value,
            {s.skill_id: s.version for s in self.skills},
            aggregated_evidence=aggregate_evidence(tuple(reports)),
            usage=usage,
            errors=tuple(errors),
        )


def _merge(a: UsageRecord, b: UsageRecord) -> UsageRecord:
    return UsageRecord(
        a.calls + b.calls,
        a.prompt_tokens + b.prompt_tokens,
        a.completion_tokens + b.completion_tokens,
        a.latency_ms + b.latency_ms,
        a.estimated_cost + b.estimated_cost,
    )
