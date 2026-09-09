from evofact.config import GateConfig
from evofact.core.models import EvaluationResult, GateDecision, StatisticalTestResult

from .objectives import regression_failures


class ValidationGate:
    def __init__(self, config: GateConfig):
        self.config = config

    def decide(
        self,
        baseline: EvaluationResult,
        candidate: EvaluationResult,
        paired: StatisticalTestResult,
        *,
        safety: str = "safe",
    ) -> GateDecision:
        b = baseline.aggregate_metrics
        c = candidate.aggregate_metrics
        failures = list(
            regression_failures(
                baseline.domain_metrics,
                candidate.domain_metrics,
                self.config.max_protected_domain_drop,
            )
        )
        if c.get("coverage", 0) < self.config.min_coverage:
            failures.append("coverage below minimum")
        if c.get("mean_cost", 0) > max(1e-12, b.get("mean_cost", 0)) * self.config.max_cost_ratio:
            failures.append("cost ratio exceeded")
        gain = c.get("macro_f1_all", 0) - b.get("macro_f1_all", 0)
        if gain < self.config.min_macro_f1_gain:
            failures.append("macro-F1 gain below minimum")
        if safety == "blocked":
            failures.append("security scan blocked candidate")
        if safety == "review_required":
            return GateDecision(
                False,
                "review_required",
                baseline,
                candidate,
                paired,
                tuple(failures),
                "human review required",
            )
        # Statistical significance is required only when discordant pairs exist; repeated
        # evaluation may instead provide a confidence interval excluding zero.
        ci = candidate.confidence_intervals.get("paired_accuracy_delta")
        statistically_supported = paired.significant or bool(ci and ci[0] > 0)
        if not statistically_supported:
            failures.append("paired improvement not statistically supported")
        if failures:
            # Useful non-dominated candidates remain auditable but never become active.
            disposition = (
                "pareto" if gain > 0 and not any("security" in x for x in failures) else "rejected"
            )
            return GateDecision(
                False,
                disposition,
                baseline,
                candidate,
                paired,
                tuple(failures),
                "; ".join(failures),
            )
        return GateDecision(
            True, "active", baseline, candidate, paired, (), "all promotion constraints passed"
        )
