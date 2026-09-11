from evofact.config import GateConfig
from evofact.core.models import EvaluationResult, GateDecision, StatisticalTestResult

from .objectives import regression_failures


class ValidationGate:
    def __init__(self, config: GateConfig):
        """函数作用：创建并初始化 `ValidationGate` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `ValidationGate` 实例；`config`（GateConfig）需符合函数签名约定。
        输出：返回 `None`；初始化 `ValidationGate` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config

    def decide(
        self,
        baseline: EvaluationResult,
        candidate: EvaluationResult,
        paired: StatisticalTestResult,
        *,
        safety: str = "safe",
    ) -> GateDecision:
        """函数作用：根据当前指标、安全结果和门控阈值产生候选处置决策。
        输入要求：`self` 应为已初始化的 `ValidationGate` 实例；`baseline`（EvaluationResult）需符合函数签名约定；`candidate`（EvaluationResult）需符合函数签名约定；`paired`（StatisticalTestResult）需符合函数签名约定；`safety`（str，默认 `'safe'`）需以关键字传入并符合签名约定。
        输出：返回 `GateDecision` 类型结果；校验或下游调用失败时异常向上传递。"""
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
