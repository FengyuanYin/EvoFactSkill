from __future__ import annotations

from dataclasses import dataclass

from evofact.core.generation_models import PairedGeneratorEvaluation


@dataclass(frozen=True)
class GeneratorGateConfig:
    min_validity_delta: float = 0.0
    min_agreement_delta: float = 0.0
    min_evidence_delta: float = 0.0
    min_diversity_delta: float = 0.0
    min_difficulty_delta: float = 0.0
    min_teaching_gain_delta: float = 0.0
    max_duplicate_increase: float = 0.0
    max_leakage_rate: float = 0.0
    max_safety_rejection_rate: float = 0.0
    max_cost_ratio: float = 1.5
    min_label_coverage: float = 0.0
    min_worst_label_agreement: float = 0.0
    max_worst_label_probe_drop: float = 1.0


@dataclass(frozen=True)
class GeneratorGateDecision:
    accepted: bool
    disposition: str
    failures: tuple[str, ...]
    reason: str


class GeneratorGate:
    def __init__(self, config: GeneratorGateConfig = GeneratorGateConfig()):
        self.config = config

    def decide(
        self, evaluation: PairedGeneratorEvaluation, *, safety_level: str = "safe"
    ) -> GeneratorGateDecision:
        champion, challenger = evaluation.champion, evaluation.challenger
        failures = []
        if (
            evaluation.label_contract_digest
            and evaluation.metadata.get("champion_label_contract_digest")
            and evaluation.metadata.get("champion_label_contract_digest")
            != evaluation.label_contract_digest
        ):
            failures.append("champion label contract mismatch")
        if (
            evaluation.label_contract_digest
            and evaluation.metadata.get("challenger_label_contract_digest")
            and evaluation.metadata.get("challenger_label_contract_digest")
            != evaluation.label_contract_digest
        ):
            failures.append("challenger label contract mismatch")
        comparisons = (
            (
                challenger.validity_rate - champion.validity_rate,
                self.config.min_validity_delta,
                "validity",
            ),
            (
                challenger.agreement_rate - champion.agreement_rate,
                self.config.min_agreement_delta,
                "agreement",
            ),
            (
                challenger.evidence_rate - champion.evidence_rate,
                self.config.min_evidence_delta,
                "evidence",
            ),
            (
                challenger.diversity - champion.diversity,
                self.config.min_diversity_delta,
                "diversity",
            ),
            (
                challenger.difficulty - champion.difficulty,
                self.config.min_difficulty_delta,
                "difficulty",
            ),
            (
                challenger.teaching_gain - champion.teaching_gain,
                self.config.min_teaching_gain_delta,
                "teaching",
            ),
        )
        for delta, minimum, name in comparisons:
            if delta < minimum:
                failures.append(f"{name} delta below threshold")
        if challenger.duplicate_rate - champion.duplicate_rate > self.config.max_duplicate_increase:
            failures.append("duplicate rate increased")
        if challenger.leakage_rate > self.config.max_leakage_rate:
            failures.append("leakage threshold exceeded")
        if challenger.safety_rejection_rate > self.config.max_safety_rejection_rate:
            failures.append("safety rejection threshold exceeded")
        if challenger.label_coverage_rate < self.config.min_label_coverage:
            failures.append("label coverage below threshold")
        if challenger.worst_label_agreement < self.config.min_worst_label_agreement:
            failures.append("worst-label agreement below threshold")
        if challenger.worst_label_probe_drop > self.config.max_worst_label_probe_drop:
            failures.append("worst-label probe drop exceeded")
        if champion.cost is None or challenger.cost is None:
            failures.append("cost unavailable")
        elif challenger.cost > max(champion.cost, type(champion.cost)("0.000000001")) * type(
            champion.cost
        )(str(self.config.max_cost_ratio)):
            failures.append("cost ratio exceeded")
        if safety_level == "blocked":
            failures.append("candidate blocked by safety scan")
        if safety_level == "review_required":
            return GeneratorGateDecision(
                False, "review_required", tuple(failures), "human review required"
            )
        if evaluation.source_episode_id == evaluation.evaluation_episode_id:
            failures.append("candidate source and evaluation episodes must differ")
        return GeneratorGateDecision(
            not failures,
            "active" if not failures else "rejected",
            tuple(failures),
            "all Generator constraints passed" if not failures else "; ".join(failures),
        )
