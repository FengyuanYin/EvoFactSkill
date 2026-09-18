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
        if evaluation.challenger_digest != candidate.package.package_digest:
            raise ValueError("paired evaluation challenger digest does not match candidate Package")
        if (
            candidate.base is not None
            and evaluation.champion_digest != candidate.base.package_digest
        ):
            raise ValueError("paired evaluation champion digest does not match active Package")
        if candidate.package.manifest.name != "generation_agent":
            raise ValueError("generator-evolve may only promote the generation_agent Package")
        decision = self.gate.decide(evaluation, safety_level=candidate.safety_level)
        committed = False
        if decision.accepted and not evaluation_only and self.repository is not None:
            self.repository.promote_package(candidate.package, run_id=run_id)
            committed = True
        return GeneratorEvolutionOutcome(
            candidate.package.package_digest, evaluation, decision, committed
        )


def validate_evaluation_provenance(
    evaluation: PairedGeneratorEvaluation,
    *,
    verifier_fingerprint: str,
    pricing_identity: str,
    label_contract_digest: str,
) -> None:
    """Fail closed when an offline paired evaluation no longer matches runtime policy."""
    if evaluation.label_contract_digest != label_contract_digest:
        raise ValueError("paired evaluation label contract identity has drifted")
    required = {
        "evaluation_protocol": "paired-generator-v1",
        "champion_generator_digest": evaluation.champion_digest,
        "challenger_generator_digest": evaluation.challenger_digest,
        "verifier_fingerprint": verifier_fingerprint,
        "pricing_identity": pricing_identity,
        "label_contract_digest": label_contract_digest,
    }
    missing = sorted(key for key in required if key not in evaluation.metadata)
    if missing:
        raise ValueError("paired evaluation provenance is incomplete: " + ", ".join(missing))
    drift = sorted(
        key for key, expected in required.items() if evaluation.metadata.get(key) != expected
    )
    if drift:
        raise ValueError("paired evaluation provenance has drifted: " + ", ".join(drift))
    source_digest = evaluation.metadata.get("source_data_digest")
    evaluation_digest = evaluation.metadata.get("evaluation_data_digest")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("paired evaluation source_data_digest is invalid")
    if not isinstance(evaluation_digest, str) or len(evaluation_digest) != 64:
        raise ValueError("paired evaluation evaluation_data_digest is invalid")
    if source_digest == evaluation_digest:
        raise ValueError("paired evaluation source and evaluation data must be independent")
