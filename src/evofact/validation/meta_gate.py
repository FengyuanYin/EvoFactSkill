from __future__ import annotations

from evofact.config import MetaLearningConfig
from evofact.core.models import MetaGateDecision, TransferUtility


class MetaValidationGate:
    def __init__(self, config: MetaLearningConfig):
        self.config = config

    def decide(self, utility: TransferUtility, *, safety_level: str = "safe", retirement_candidate: bool = False) -> MetaGateDecision:
        if safety_level == "blocked":
            return self._decision(utility, False, "rejected", (), ("security scan blocked candidate",))
        if safety_level == "review_required":
            return self._decision(utility, False, "review_required", (), ("human review required",))
        failures = []
        if utility.episode_count < self.config.min_valid_episodes:
            failures.append("insufficient valid episodes")
        if utility.mean_gain < self.config.min_mean_gain:
            failures.append("mean transfer gain below minimum")
        if utility.confidence_interval[0] <= 0:
            failures.append("transfer confidence interval does not exclude zero")
        if utility.negative_transfer_rate > self.config.max_negative_transfer_rate:
            failures.append("negative transfer rate exceeded")
        if self.config.enforce_worst_domain and utility.worst_domain_drop < -self.config.max_worst_domain_drop:
            failures.append("worst-domain regression exceeded")
        if utility.mean_coverage < self.config.min_coverage:
            failures.append("coverage below minimum")
        if utility.cost_ratio > self.config.max_cost_ratio:
            failures.append("cost ratio exceeded")
        if utility.calibration_delta > self.config.max_calibration_increase:
            failures.append("calibration regression exceeded")
        if not failures:
            return self._decision(utility, True, "retired" if retirement_candidate else "generalized", utility.tested_domains, ())
        positive = tuple(sorted(domain for domain, gain in utility.domain_gains.items() if gain > 0))
        hard_constraints_pass = (
            utility.mean_coverage >= self.config.min_coverage
            and utility.cost_ratio <= self.config.max_cost_ratio
            and utility.calibration_delta <= self.config.max_calibration_increase
        )
        if not retirement_candidate and self.config.allow_specialization and hard_constraints_pass and positive and len(positive) < len(utility.tested_domains) and utility.episode_count >= self.config.min_valid_episodes:
            severe = utility.negative_transfer_rate > .5 or utility.worst_domain_drop < -2 * self.config.max_worst_domain_drop
            if not severe:
                return self._decision(utility, True, "specialized", positive, tuple(failures))
        if utility.mean_gain > 0 and utility.mean_coverage >= self.config.min_coverage:
            return self._decision(utility, False, "pareto", (), tuple(failures))
        return self._decision(utility, False, "rejected", (), tuple(failures))

    @staticmethod
    def _decision(utility: TransferUtility, accepted: bool, disposition: str, domains: tuple[str, ...], failures: tuple[str, ...]) -> MetaGateDecision:
        reason = "all cross-domain promotion constraints passed" if not failures else "; ".join(failures)
        return MetaGateDecision(utility.candidate_fingerprint, accepted, disposition, domains, utility, failures, reason)
