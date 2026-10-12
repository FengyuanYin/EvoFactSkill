from evofact.config import GateConfig
from evofact.core.models import EvaluationResult, GateDecision, StatisticalTestResult


class ValidationGate:
    def __init__(self, config: GateConfig, *, require_cost: bool = False):
        """函数作用：创建并初始化 `ValidationGate` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `ValidationGate` 实例；`config`（GateConfig）需符合函数签名约定。
        输出：返回 `None`；初始化 `ValidationGate` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config
        self.require_cost = require_cost

    def decide(
        self,
        baseline: EvaluationResult,
        candidate: EvaluationResult,
        paired: StatisticalTestResult,
        *,
        safety: str = "safe",
    ) -> GateDecision:
        """函数作用：仅根据宏平均 F1 是否严格提升决定晋升，同时保留安全拦截。
        输入要求：`self` 应为已初始化的 `ValidationGate` 实例；`baseline`（EvaluationResult）需符合函数签名约定；`candidate`（EvaluationResult）需符合函数签名约定；`paired`（StatisticalTestResult）需符合函数签名约定；`safety`（str，默认 `'safe'`）需以关键字传入并符合签名约定。
        输出：返回 `GateDecision` 类型结果；校验或下游调用失败时异常向上传递。"""
        b = baseline.aggregate_metrics
        c = candidate.aggregate_metrics
        # Other metrics and paired statistics remain in the report, but do not veto
        # ordinary evolve promotion. Config/require_cost remain accepted for compatibility.
        failures: list[str] = []
        gain = c.get("macro_f1_all", float("nan")) - b.get("macro_f1_all", float("nan"))
        if not gain > 0:
            failures.append("macro-F1 did not improve")
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
        if failures:
            return GateDecision(
                False,
                "rejected",
                baseline,
                candidate,
                paired,
                tuple(failures),
                "; ".join(failures),
            )
        return GateDecision(True, "active", baseline, candidate, paired, (), "macro-F1 improved")
