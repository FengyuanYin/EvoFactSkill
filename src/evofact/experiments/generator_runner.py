from __future__ import annotations

from dataclasses import dataclass

from evofact.core.generation_models import PairedGeneratorEvaluation
from evofact.evolution.package_candidate import PackageCandidate
from evofact.skills.repository import SkillRepository
from evofact.validation.generator_gate import GeneratorGate, GeneratorGateDecision


@dataclass(frozen=True)
class GeneratorEvolutionOutcome:
    candidate_digest: str
    evaluation: PairedGeneratorEvaluation
    decision: GeneratorGateDecision
    committed: bool


class GeneratorEvolutionRunner:
    def __init__(self, repository: SkillRepository | None, gate: GeneratorGate | None = None):
        self.repository = repository
        self.gate = gate or GeneratorGate()

    def finalize(
        self,
        candidate: PackageCandidate,
        evaluation: PairedGeneratorEvaluation,
        *,
        run_id: str,
        evaluation_only: bool = False,
    ) -> GeneratorEvolutionOutcome:
        decision = self.gate.decide(evaluation, safety_level=candidate.safety_level)
        committed = False
        if decision.accepted and not evaluation_only and self.repository is not None:
            self.repository.promote_package(candidate.package, run_id=run_id)
            committed = True
        return GeneratorEvolutionOutcome(
            candidate.package.package_digest, evaluation, decision, committed
        )
