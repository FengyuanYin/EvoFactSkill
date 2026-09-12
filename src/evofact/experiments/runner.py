from dataclasses import replace
from pathlib import Path

from evofact.attribution.clustering import cluster_reports
from evofact.attribution.rules import attribute_trace
from evofact.config import AppConfig
from evofact.core.models import RunBudget, Sample, SampleEvaluation, SkillStatus
from evofact.evolution.distiller import distill
from evofact.evolution.proposer import propose_from_cluster
from evofact.routing.router import LLMSkillRouter, SkillRouter
from evofact.runtime.inference import InferenceRuntime
from evofact.runtime.mock_backend import MockBackend
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.security.scanner import scan_resources
from evofact.skills.candidates import apply_candidate
from evofact.skills.loader import load_skill_package
from evofact.skills.utility import UtilityTracker
from evofact.validation.evaluator import evaluate
from evofact.validation.gate import ValidationGate
from evofact.validation.statistics import mcnemar, paired_bootstrap


def fixture_samples() -> list[Sample]:
    """函数作用：构造离线测试夹具 `fixture_samples` 所表示的数据，供当前模块后续流程使用。
    输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
    输出：返回 `list[Sample]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [
        Sample("fx-1", "fixture", "官方通报：道路已经恢复通行。", "REAL", "social"),
        Sample("fx-2", "fixture", "网传假消息：饮用盐水可以治愈所有疾病。", "FAKE", "health"),
        Sample("fx-3", "fixture", "未经证实的普通陈述。", "FAKE", "social"),
        Sample("fx-4", "fixture", "研究机构发布可复核的数据报告。", "REAL", "science"),
    ]


def fixture_validation_samples() -> list[Sample]:
    """函数作用：构造离线测试夹具 `fixture_validation_samples` 所表示的数据，供当前模块后续流程使用。
    输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
    输出：返回 `list[Sample]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [
        Sample("fv-1", "fixture", "权威机构公开了可复核记录。", "REAL", "social"),
        Sample("fv-2", "fixture", "这是已被辟谣的假消息。", "FAKE", "health"),
        Sample("fv-3", "fixture", "未经证实的另一项陈述。", "FAKE", "social"),
        Sample("fv-4", "fixture", "多个独立来源确认该报告。", "REAL", "science"),
    ]


def load_seed_skills(root: Path) -> list:
    """函数作用：读取并转换 `load_seed_skills` 所表示的数据，供当前模块后续流程使用。
    输入要求：`root`（Path）需符合函数签名约定。
    输出：返回 `list` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [
        load_skill_package(p, status=SkillStatus.ACTIVE)
        for p in sorted(root.iterdir())
        if (p / "SKILL.md").is_file()
    ]


class ExperimentRunner:
    def __init__(self, config: AppConfig, project_root: Path):
        """函数作用：创建并初始化 `ExperimentRunner` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`config`（AppConfig）需符合函数签名约定；`project_root`（Path）需符合函数签名约定。
        输出：返回 `None`；初始化 `ExperimentRunner` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config
        self.root = Path(project_root)
        self.skills = load_seed_skills(self.root / "skills" / "seeds")

    def _backend(self):
        """函数作用：负责`ExperimentRunner` 中的 `_backend` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        # 制作openai-compatible 后端llm
        if self.config.backend == "mock":
            return MockBackend()
        return OpenAICompatibleBackend(
            self.config.base_url, self.config.resolved_api_key(), self.config.model
        )

    async def run(
        self, samples: list[Sample] | None = None, strategy: str = "utility-aware", skills=None
    ):
        """函数作用：执行当前对象负责的主运行流程，并汇总本轮结果。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`samples`（list[Sample] | None，默认 `None`）需符合函数签名约定；`strategy`（str，默认 `'utility-aware'`）需符合函数签名约定；`skills`（未显式标注，默认 `None`）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        rows = (
            fixture_samples() if samples is None else samples
        )  # 这里做一个samples判断 如果没有samples则使用固定的测试用例

        backend = self._backend()
        if strategy == "llm":
            router = LLMSkillRouter(
                backend=backend,
                fallback=SkillRouter(
                    strategy="utility-aware",
                    seed = self.config.seed,
                )
            )
        else:
            router = SkillRouter(
                strategy=strategy,
                seed=self.config.seed,
            )

        runtime = InferenceRuntime(
            backend,
            router,
            self.skills if skills is None else skills,
        )
        traces = []
        evaluations = []
        for sample in rows:
            trace = await runtime.infer(sample, RunBudget(self.config.max_skills_per_item))
            traces.append(trace)
            gold = _gold(sample.label)
            evaluations.append(
                SampleEvaluation(
                    sample.sample_id,
                    gold,
                    trace.decision.label,
                    trace.decision.confidence,
                    sample.domain,
                    str(sample.metadata.get("temporal_window"))
                    if sample.metadata.get("temporal_window") is not None
                    else None,
                    trace.usage.estimated_cost,
                )
            )
        return traces, evaluate(evaluations)

    async def evolve_once(self, samples: list[Sample] | None = None, *, generation_guard=None):
        """函数作用：根据一批样本完成推理、归因、经验蒸馏和检测技能候选生成。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`samples`（list[Sample] | None，默认 `None`）需符合函数签名约定；`generation_guard`（未显式标注，默认 `None`）需以关键字传入并符合签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        rows = fixture_samples() if samples is None else samples
        traces, result = await self.run(rows)
        reports = [attribute_trace(t, _gold(s.label)) for t, s in zip(traces, rows)]
        tracker = UtilityTracker()
        credited = []
        for trace, sample, report in zip(traces, rows, reports):
            baseline_ok = trace.decision.label == _gold(sample.label)
            deltas = {}
            for sid in trace.routing.selected_skill_ids:
                reduced = [skill for skill in self.skills if skill.skill_id != sid]
                cf_trace = (await self.run([sample], skills=reduced))[0][0]
                delta = float(baseline_ok) - float(cf_trace.decision.label == _gold(sample.label))
                deltas[sid] = delta
                tracker.update(
                    sid,
                    success=baseline_ok,
                    delta=delta,
                    cost=trace.usage.estimated_cost,
                    domain=sample.domain,
                    window=str(sample.metadata.get("temporal_window"))
                    if sample.metadata.get("temporal_window") is not None
                    else None,
                )
            credited.append(replace(report, counterfactual_deltas=deltas))
        reports = credited
        if generation_guard is not None:
            generation_guard(traces, reports)
        clusters = cluster_reports([r for r in reports if r.error_types])
        proposals = [
            propose_from_cluster(cid, group, self.skills) for cid, group in clusters.items()
        ]
        return {
            "traces": traces,
            "evaluation": result,
            "attributions": reports,
            "utilities": tracker.values,
            "distillation": distill(traces, reports),
            "proposals": proposals,
        }

    async def closed_loop(
        self, samples: list[Sample] | None = None, validation_samples: list[Sample] | None = None
    ):
        """函数作用：运行候选生成与独立验证闭环，为每个候选生成门控决策。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`samples`（list[Sample] | None，默认 `None`）需符合函数签名约定；`validation_samples`（list[Sample] | None，默认 `None`）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        rows = fixture_samples() if samples is None else samples
        validation_rows = (
            fixture_validation_samples() if validation_samples is None else validation_samples
        )
        if not rows or not validation_rows:
            raise ValueError("evolution requires non-empty training and validation samples")
        from evofact.core.models import DataManifest
        from evofact.data.leakage import detect_leakage

        manifest = DataManifest(
            "validation",
            {},
            tuple(s.sample_id for s in rows),
            tuple(s.sample_id for s in validation_rows),
        )
        errors = detect_leakage(rows + validation_rows, manifest)
        if errors:
            raise ValueError("validation data leakage: " + "; ".join(errors))
        outcome = await self.evolve_once(rows)
        decisions = []
        for proposal in outcome["proposals"]:
            candidate_skills = apply_candidate(self.skills, proposal)
            repeated_baseline = []
            repeated_candidate = []
            for repeat_index in range(self.config.gate.repeats):
                _, base_run = await self.run(validation_rows)
                _, candidate_run = await self.run(validation_rows, skills=candidate_skills)
                repeated_baseline.extend(
                    replace(x, sample_id=f"{x.sample_id}:r{repeat_index}")
                    for x in base_run.per_sample
                )
                repeated_candidate.extend(
                    replace(x, sample_id=f"{x.sample_id}:r{repeat_index}")
                    for x in candidate_run.per_sample
                )
            baseline_result = evaluate(repeated_baseline)
            candidate_result = evaluate(repeated_candidate)
            ci = paired_bootstrap(repeated_baseline, repeated_candidate, seed=self.config.seed)
            candidate_result = replace(
                candidate_result, confidence_intervals={"paired_accuracy_delta": ci}
            )
            test = mcnemar(repeated_baseline, repeated_candidate, self.config.gate.alpha)
            levels = [
                scan_resources(candidate.resources).level for candidate in proposal.candidate_skills
            ]
            safety = (
                "blocked"
                if "blocked" in levels
                else ("review_required" if "review_required" in levels else "safe")
            )
            decisions.append(
                ValidationGate(self.config.gate).decide(
                    baseline_result, candidate_result, test, safety=safety
                )
            )
        outcome["gate_decisions"] = decisions
        return outcome


def _gold(value) -> str:
    """函数作用：负责当前模块中的 `_gold` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    text = str(value).upper()
    if text in {"1", "FAKE", "FALSE", "FALSO"}:
        return "FAKE"
    if text in {"0", "REAL", "TRUE", "LEGIT"}:
        return "REAL"
    raise ValueError(f"unsupported fixture label: {value}")
