from __future__ import annotations

from evofact.config import MetaLearningConfig
from evofact.core.models import MetaGateDecision, TransferUtility


class MetaValidationGate:
    def __init__(self, config: MetaLearningConfig):
        """函数作用：创建并初始化 `MetaValidationGate` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `MetaValidationGate` 实例；`config`（MetaLearningConfig）需符合函数签名约定。
        输出：返回 `None`；初始化 `MetaValidationGate` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config

    def decide(
        self,
        utility: TransferUtility,
        *,
        safety_level: str = "safe",
        retirement_candidate: bool = False,
    ) -> MetaGateDecision:
        """函数作用：根据当前指标、安全结果和门控阈值产生候选处置决策。
        输入要求：`self` 应为已初始化的 `MetaValidationGate` 实例；`utility`（TransferUtility）需符合函数签名约定；`safety_level`（str，默认 `'safe'`）需以关键字传入并符合签名约定；`retirement_candidate`（bool，默认 `False`）需以关键字传入并符合签名约定。
        输出：返回 `MetaGateDecision` 类型结果；校验或下游调用失败时异常向上传递。"""
        if safety_level == "blocked":
            return self._decision(
                utility, False, "rejected", (), ("security scan blocked candidate",)
            )
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
        if (
            self.config.enforce_worst_domain
            and utility.worst_domain_drop < -self.config.max_worst_domain_drop
        ):
            failures.append("worst-domain regression exceeded")
        if utility.mean_coverage < self.config.min_coverage:
            failures.append("coverage below minimum")
        if utility.cost_ratio > self.config.max_cost_ratio:
            failures.append("cost ratio exceeded")
        if utility.calibration_delta > self.config.max_calibration_increase:
            failures.append("calibration regression exceeded")
        if not failures:
            return self._decision(
                utility,
                True,
                "retired" if retirement_candidate else "generalized",
                utility.tested_domains,
                (),
            )
        positive = tuple(
            sorted(domain for domain, gain in utility.domain_gains.items() if gain > 0)
        )
        hard_constraints_pass = (
            utility.mean_coverage >= self.config.min_coverage
            and utility.cost_ratio <= self.config.max_cost_ratio
            and utility.calibration_delta <= self.config.max_calibration_increase
        )
        if (
            not retirement_candidate
            and self.config.allow_specialization
            and hard_constraints_pass
            and positive
            and len(positive) < len(utility.tested_domains)
            and utility.episode_count >= self.config.min_valid_episodes
        ):
            severe = (
                utility.negative_transfer_rate > 0.5
                or utility.worst_domain_drop < -2 * self.config.max_worst_domain_drop
            )
            if not severe:
                return self._decision(utility, True, "specialized", positive, tuple(failures))
        if utility.mean_gain > 0 and utility.mean_coverage >= self.config.min_coverage:
            return self._decision(utility, False, "pareto", (), tuple(failures))
        return self._decision(utility, False, "rejected", (), tuple(failures))

    @staticmethod
    def _decision(
        utility: TransferUtility,
        accepted: bool,
        disposition: str,
        domains: tuple[str, ...],
        failures: tuple[str, ...],
    ) -> MetaGateDecision:
        """函数作用：负责`MetaValidationGate` 中的 `_decision` 处理，封装调用方需要复用的业务步骤。
        输入要求：`utility`（TransferUtility）需符合函数签名约定；`accepted`（bool）需符合函数签名约定；`disposition`（str）需符合函数签名约定；`domains`（tuple[str, ...]）需符合函数签名约定；`failures`（tuple[str, ...]）需符合函数签名约定。
        输出：返回 `MetaGateDecision` 类型结果；校验或下游调用失败时异常向上传递。"""
        reason = (
            "all cross-domain promotion constraints passed" if not failures else "; ".join(failures)
        )
        return MetaGateDecision(
            utility.candidate_fingerprint, accepted, disposition, domains, utility, failures, reason
        )
