import hashlib
import inspect
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from evofact.attribution.clustering import cluster_reports
from evofact.attribution.rules import attribute_trace
from evofact.config import AppConfig
from evofact.core.budget_models import BudgetLimits
from evofact.core.label_models import RUNTIME_ABSTAIN_LABEL, DecisionOrigin
from evofact.core.models import (
    InferenceTrace,
    Prediction,
    RoutingDecision,
    RunBudget,
    Sample,
    SampleEvaluation,
    SkillKind,
    SkillStatus,
)
from evofact.data.label_registry import fixture_binary_contract
from evofact.data.registry import DataRegistry
from evofact.evolution.distiller import distill
from evofact.evolution.package_candidate import (
    build_package_candidate,
    package_candidate_to_proposal,
)
from evofact.evolution.package_optimizer import (
    ALLOWED_TARGET_KINDS,
    FROZEN_TARGET_NAMES,
    PackageOptimizerAgent,
    legacy_instruction_edit_to_patch,
)
from evofact.evolution.proposer import propose_from_cluster
from evofact.governance.pricing_policy import load_pricing_table
from evofact.routing.router import LLMSkillRouter, SkillRouter
from evofact.runtime.batching import ordered_batched_map
from evofact.runtime.budget import BudgetExceeded, BudgetManager
from evofact.runtime.inference import InferenceRuntime
from evofact.runtime.mock_backend import MockBackend
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.runtime.progress import ProgressEvent, ProgressSink
from evofact.runtime.trace_store import TraceStore
from evofact.security.scanner import scan_resources
from evofact.skills.candidates import apply_candidate
from evofact.skills.loader import load_skill_package
from evofact.skills.package_adapter import (
    package_bank_digest,
    package_to_skill_spec,
    project_skill_bank_to_packages,
)
from evofact.skills.package_loader import load_package
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


def fixture_test_samples() -> list[Sample]:
    """Disjoint offline final-test fixture used by the ablation runner."""
    return [
        Sample("ft-1", "fixture", "公开档案支持这项陈述。", "REAL", "social"),
        Sample("ft-2", "fixture", "已核实这是虚假传言。", "FAKE", "health"),
        Sample("ft-3", "fixture", "独立来源发布一致记录。", "REAL", "science"),
        Sample("ft-4", "fixture", "未经证实的假消息再次传播。", "FAKE", "social"),
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
        self.packages = [
            load_package(path, status=SkillStatus.ACTIVE)
            for path in sorted((self.root / "skills" / "seeds").iterdir())
            if (path / "SKILL.md").is_file()
        ]
        self.skills = [package_to_skill_spec(package) for package in self.packages]
        self.label_contract_registry = DataRegistry().label_contracts
        self.label_contract_registry.register(fixture_binary_contract())
        self._operation_budget_manager: BudgetManager | None = None
        self._trace_store: TraceStore | None = None
        self._trace_run_id: str | None = None

    def enable_trace_logging(
        self,
        path: Path,
        *,
        run_id: str,
        resume: bool = False,
    ) -> Path:
        """Persist every inference trace produced by this training runner to one JSONL file."""
        store = TraceStore(path)
        if resume:
            if not store.path.is_file():
                raise ValueError(f"training trace log does not exist for resume: {store.path}")
        else:
            store.reset()
        self._trace_store = store
        self._trace_run_id = run_id
        return store.path

    @property
    def trace_log_path(self) -> Path | None:
        return self._trace_store.path if self._trace_store is not None else None

    def _backend(self):
        """函数作用：负责`ExperimentRunner` 中的 `_backend` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        # 制作openai-compatible 后端llm
        if self.config.backend == "mock":
            return MockBackend()
        pricing_table = None
        if self.config.pricing.table_path is not None:
            path = self.config.pricing.table_path
            if not path.is_absolute():
                path = self.root / path
            pricing_table = load_pricing_table(path)
            if pricing_table.provider != self.config.pricing.provider:
                raise ValueError("pricing provider does not match configured provider")
        return OpenAICompatibleBackend(
            self.config.base_url,
            self.config.resolved_api_key(),
            self.config.model,
            provider=self.config.pricing.provider,
            pricing_table=pricing_table,
        )

    def _optimizer_backend(self):
        """Build the training-only optimizer backend without changing inference."""
        config = self.config.optimizer_backend
        if not config.enabled:
            return self._backend()
        if config.backend == "mock":
            return MockBackend()
        base_url = config.resolved_base_url()
        if config.require_distinct_base_url and base_url.rstrip("/") == self.config.base_url.rstrip(
            "/"
        ):
            raise ValueError("optimizer and forward base_url must be different")
        pricing_table = None
        pricing_path = config.resolved_pricing_table_path()
        if pricing_path is not None:
            if not pricing_path.is_absolute():
                pricing_path = self.root / pricing_path
            pricing_table = load_pricing_table(pricing_path)
            if pricing_table.provider != config.provider:
                raise ValueError("optimizer pricing provider does not match configured provider")
        elif self.config.pricing.require_cost_for_promotion:
            raise ValueError(
                "optimizer pricing table is required when promotion requires complete cost data"
            )
        return OpenAICompatibleBackend(
            base_url,
            config.resolved_api_key(),
            config.resolved_model(),
            provider=config.provider,
            pricing_table=pricing_table,
            temperature=config.temperature,
        )

    def pricing_identity(self) -> str:
        """Fingerprint actual pricing contents, not only the configured path."""
        if self.config.pricing.table_path is None:
            forward_identity = hashlib.sha256(
                f"unpriced:{self.config.pricing.provider}".encode("utf-8")
            ).hexdigest()
        else:
            path = self.config.pricing.table_path
            if not path.is_absolute():
                path = self.root / path
            table = load_pricing_table(path)
            if table.provider != self.config.pricing.provider:
                raise ValueError("pricing provider does not match configured provider")
            forward_identity = table.identity
        optimizer = self.config.optimizer_backend
        if not optimizer.enabled:
            return forward_identity
        optimizer_path = optimizer.resolved_pricing_table_path()
        optimizer_pricing_identity = f"unpriced:{optimizer.provider}"
        if optimizer_path is not None:
            if not optimizer_path.is_absolute():
                optimizer_path = self.root / optimizer_path
            optimizer_table = load_pricing_table(optimizer_path)
            if optimizer_table.provider != optimizer.provider:
                raise ValueError("optimizer pricing provider does not match configured provider")
            optimizer_pricing_identity = optimizer_table.identity
        payload = {
            "forward": forward_identity,
            "optimizer": optimizer_pricing_identity,
            "optimizer_provider": optimizer.provider,
            "optimizer_model": optimizer.resolved_model(),
            "optimizer_base_url": optimizer.resolved_base_url(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def model_identity(self) -> dict:
        """Return credential-free model provenance for checkpoints and Skill Banks."""
        optimizer = self.config.optimizer_backend
        return {
            "forward": {
                "backend": self.config.backend,
                "provider": self.config.pricing.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "temperature": 0,
            },
            "optimizer": (
                {
                    "backend": optimizer.backend,
                    "provider": optimizer.provider,
                    "model": optimizer.resolved_model(),
                    "base_url": optimizer.resolved_base_url(),
                    "temperature": optimizer.temperature,
                }
                if optimizer.enabled
                else {"inherits_forward": True}
            ),
        }

    def _budget_manager(self) -> BudgetManager:
        if self._operation_budget_manager is None:
            self._operation_budget_manager = BudgetManager(
                BudgetLimits(
                    max_nodes=self.config.dag.max_nodes,
                    max_depth=self.config.dag.max_depth,
                    max_calls_per_sample=self.config.budget.max_calls_per_sample,
                    max_tokens_per_sample=self.config.budget.max_tokens_per_sample,
                    max_cost_per_sample=self.config.budget.max_cost_per_sample,
                    max_calls_per_run=self.config.budget.max_calls_per_run,
                    max_tokens_per_run=self.config.budget.max_tokens_per_run,
                    max_cost_per_run=self.config.budget.max_cost_per_run,
                    max_sample_concurrency=self.config.budget.max_sample_concurrency,
                    max_global_concurrency=self.config.budget.max_global_concurrency,
                    call_timeout_ms=self.config.dag.node_timeout_ms,
                    sample_timeout_ms=self.config.dag.sample_timeout_ms,
                    judge_reserved_calls=self.config.budget.judge_reserved_calls,
                    judge_reserved_tokens=self.config.budget.judge_reserved_tokens,
                    judge_reserved_cost=self.config.budget.judge_reserved_cost,
                )
            )
        return self._operation_budget_manager

    async def run(
        self,
        samples: list[Sample] | None = None,
        strategy: str | None = None,
        skills=None,
        *,
        progress: ProgressSink | None = None,
        task_name: str = "run",
        phase: str = "inference",
    ):
        """函数作用：执行当前对象负责的主运行流程，并汇总本轮结果。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`samples`（list[Sample] | None，默认 `None`）需符合函数签名约定；`strategy`（str，默认 `'utility-aware'`）需符合函数签名约定；`skills`（未显式标注，默认 `None`）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""

        effective_strategy = self.config.routing_strategy if strategy is None else strategy

        rows = (
            fixture_samples() if samples is None else samples
        )  # 这里做一个samples判断 如果没有samples则使用固定的测试用例

        backend = self._backend()
        plan_limits = self._budget_manager().limits
        if effective_strategy == "llm":
            router = LLMSkillRouter(
                backend=backend,
                fallback=SkillRouter(
                    strategy="utility-aware",
                    seed=self.config.seed,
                    plan_limits=plan_limits,
                ),
                strict_serial=self.config.dag.strict_legacy_order,
                plan_limits=plan_limits,
            )
        else:
            router = SkillRouter(
                strategy=effective_strategy,
                seed=self.config.seed,
                plan_limits=plan_limits,
            )
            if self.config.dag.strict_legacy_order:
                from evofact.routing.rule_planner import RulePlanner

                router = RulePlanner(router, strict_serial=True, limits=plan_limits)

        budget_manager = self._budget_manager()
        skill_snapshot = list(self.skills if skills is None else skills)
        runtime = InferenceRuntime(
            backend,
            router,
            skill_snapshot,
            budget_manager=budget_manager,
            label_contract_registry=self.label_contract_registry,
        )

        async def infer_sample(sample):
            contract = self.label_contract_registry.resolve_sample(sample)
            gold = contract.normalize(sample.label)
            try:
                trace = await runtime.infer(sample, RunBudget(self.config.max_skills_per_item))
            except BudgetExceeded as exc:
                # 兜底：单条样本的预算耗尽只降级为 ABSTAIN，不能让整轮实验失败（否则 955 条结果全丢）。
                trace = _budget_abstention_trace(sample, skill_snapshot, str(exc), contract)
            return trace, SampleEvaluation(
                sample.sample_id,
                gold,
                trace.decision.label,
                trace.decision.confidence,
                sample.domain,
                str(sample.metadata.get("temporal_window"))
                if sample.metadata.get("temporal_window") is not None
                else None,
                float(trace.usage.cost or 0),
                contract.schema_id,
                contract.allowed_labels,
                contract.positive_label,
                trace.decision.origin.value,
                trace.usage.cost_status.value,
                bool(sample.evidence),
            )

        def item_done(completed, total):
            if progress is not None:
                progress.update(ProgressEvent(task_name, phase, completed, total, "samples"))

        pairs = await ordered_batched_map(
            rows,
            infer_sample,
            batch_size=self.config.execution.batch_size,
            concurrency=self.config.execution.max_concurrent_samples,
            on_item_done=item_done,
        )
        traces = [trace for trace, _ in pairs]
        evaluations = [evaluation for _, evaluation in pairs]
        if self._trace_store is not None:
            for sample_index, trace in enumerate(traces):
                self._trace_store.append(
                    trace,
                    context={
                        "run_id": self._trace_run_id,
                        "task": task_name,
                        "phase": phase,
                        "sample_index": sample_index,
                        "skill_ids": [skill.skill_id for skill in skill_snapshot],
                    },
                )
        return traces, evaluate(evaluations)

    async def evolve_once(
        self,
        samples: list[Sample] | None = None,
        *,
        generation_guard=None,
        progress: ProgressSink | None = None,
        task_name: str = "evolve",
    ):
        """函数作用：根据一批样本完成推理、归因、经验蒸馏和检测技能候选生成。
        输入要求：`self` 应为已初始化的 `ExperimentRunner` 实例；`samples`（list[Sample] | None，默认 `None`）需符合函数签名约定；`generation_guard`（未显式标注，默认 `None`）需以关键字传入并符合签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        rows = fixture_samples() if samples is None else samples
        traces, result = await self.run(
            rows, progress=progress, task_name=task_name, phase="train inference"
        )
        reports = []
        for trace, sample in zip(traces, rows):
            contract = self.label_contract_registry.resolve_sample(sample)
            reports.append(attribute_trace(trace, contract.normalize(sample.label), contract))
        if not self.config.evolution.enabled:
            if generation_guard is not None:
                generation_guard(traces, reports)
            return {
                "traces": traces,
                "evaluation": result,
                "attributions": reports,
                "utilities": {},
                "distillation": distill(traces, reports),
                "proposals": [],
                "package_candidates": {},
            }
        tracker = UtilityTracker()
        jobs = []
        attribution_limit = self.config.evolution.max_attribution_samples_per_update
        if attribution_limit:
            attributed_indexes = [
                index for index, report in enumerate(reports) if report.error_types
            ][:attribution_limit]
        else:
            attributed_indexes = list(range(len(reports)))
        attributed = set(attributed_indexes)
        for trace_index, (trace, sample) in enumerate(zip(traces, rows)):
            if trace_index not in attributed:
                continue
            selected_skill_ids = trace.routing.selected_skill_ids
            counterfactual_limit = self.config.evolution.max_counterfactuals_per_sample
            if counterfactual_limit:
                selected_skill_ids = selected_skill_ids[:counterfactual_limit]
            for skill_index, sid in enumerate(selected_skill_ids):
                reduced = [skill for skill in self.skills if skill.skill_id != sid]
                jobs.append((trace_index, skill_index, sid, sample, reduced))

        async def counterfactual(job):
            trace_index, skill_index, sid, sample, reduced = job
            cf_trace = (
                await self.run(
                    [sample],
                    skills=reduced,
                    task_name=task_name,
                    phase=f"counterfactual without {sid}",
                )
            )[0][0]
            return trace_index, skill_index, sid, cf_trace

        def counterfactual_done(completed, total):
            if progress is not None:
                progress.update(
                    ProgressEvent(
                        task_name,
                        "counterfactual",
                        completed,
                        total,
                        "evaluations",
                    )
                )

        counterfactuals = await ordered_batched_map(
            jobs,
            counterfactual,
            batch_size=self.config.execution.batch_size,
            concurrency=self.config.execution.max_concurrent_samples,
            on_item_done=counterfactual_done,
        )
        grouped = {index: [] for index in range(len(traces))}
        for trace_index, skill_index, sid, cf_trace in counterfactuals:
            grouped[trace_index].append((skill_index, sid, cf_trace))

        credited = []
        for trace_index, (trace, sample, report) in enumerate(zip(traces, rows, reports)):
            gold = self.label_contract_registry.resolve_sample(sample).normalize(sample.label)
            baseline_ok = trace.decision.label == gold
            deltas = {}
            for _, sid, cf_trace in sorted(grouped[trace_index]):
                delta = float(baseline_ok) - float(cf_trace.decision.label == gold)
                deltas[sid] = delta
                tracker.update(
                    sid,
                    success=baseline_ok,
                    delta=delta,
                    cost=float(trace.usage.cost or 0),
                    domain=sample.domain,
                    window=str(sample.metadata.get("temporal_window"))
                    if sample.metadata.get("temporal_window") is not None
                    else None,
                )
            credited.append(replace(report, counterfactual_deltas=deltas))
        reports = credited
        if generation_guard is not None:
            generation_guard(traces, reports)
        clusters = cluster_reports([report for report in reports if report.error_types])
        package_candidates = {}

        if self.config.evolution.proposer == "llm":
            optimizer_packages = [
                package
                for package in self.packages
                if package.manifest.name == self.config.evolution.optimizer_skill
                and package.manifest.kind == SkillKind.META
            ]
            if len(optimizer_packages) != 1:
                raise ValueError("optimizer Package must resolve to exactly one META Package")
            optimizer = PackageOptimizerAgent(
                backend=self._optimizer_backend(),
                optimizer_package=optimizer_packages[0],
                budget_manager=self._budget_manager(),
            )

            cluster_items = sorted(clusters.items())

            async def optimize_cluster(item):
                cluster_id, group = item
                responsible = {
                    skill_id for report in group for skill_id in report.responsible_skill_ids
                }
                editable = [
                    package
                    for package in self.packages
                    if package.manifest.kind in ALLOWED_TARGET_KINDS
                    and package.manifest.name not in FROZEN_TARGET_NAMES
                    and package.manifest.name != "generation_agent"
                ]
                target = next(
                    (package for package in editable if package.skill_id in responsible),
                    next(
                        (
                            package
                            for package in editable
                            if package.manifest.kind in {SkillKind.ROUTER, SkillKind.JUDGE}
                        ),
                        editable[0] if editable else None,
                    ),
                )
                if target is None:
                    return None
                candidate = await optimizer.propose(
                    target,
                    self.packages,
                    reports=group,
                    traces=traces,
                )
                if candidate is None:
                    return None
                if candidate.base is None and not self.config.evolution.discovery:
                    return None
                if candidate.base is not None and self.config.evolution.scope == "instructions":
                    instructions = package_to_skill_spec(candidate.package).instructions
                    candidate = build_package_candidate(
                        target,
                        legacy_instruction_edit_to_patch(
                            target,
                            instructions,
                            rationale=(
                                candidate.patch.rationale
                                if candidate.patch is not None
                                else "instruction-only ablation"
                            ),
                            source_trace_ids=tuple(report.trace_id for report in group),
                        ),
                    )
                proposal = package_candidate_to_proposal(
                    candidate,
                    cluster_id=cluster_id,
                    source_trace_ids=tuple(report.trace_id for report in group),
                )
                return proposal, candidate

            def optimizer_done(completed, total):
                if progress is not None:
                    progress.update(
                        ProgressEvent(
                            task_name,
                            "optimizer",
                            completed,
                            total,
                            "clusters",
                        )
                    )

            proposed = await ordered_batched_map(
                cluster_items,
                optimize_cluster,
                batch_size=self.config.execution.batch_size,
                concurrency=self.config.execution.max_concurrent_samples,
                on_item_done=optimizer_done,
            )
            # NO_CHANGE 返回 None，不进入后续评估和 Gate。
            pairs = [item for item in proposed if item is not None]
            proposals = [proposal for proposal, _ in pairs]
            package_candidates = {proposal.proposal_id: candidate for proposal, candidate in pairs}

        else:
            # 保留原来的规则 proposer，保证向后兼容。
            proposals = [
                propose_from_cluster(
                    cluster_id,
                    group,
                    self.skills,
                )
                for cluster_id, group in clusters.items()
            ]
            if not self.config.evolution.discovery:
                proposals = [
                    proposal for proposal in proposals if proposal.operation.value != "add"
                ]

        return {
            "traces": traces,
            "evaluation": result,
            "attributions": reports,
            "utilities": tracker.values,
            "distillation": distill(traces, reports),
            "proposals": proposals,
            "package_candidates": package_candidates,
        }

    async def closed_loop(
        self,
        samples: list[Sample] | None = None,
        validation_samples: list[Sample] | None = None,
        *,
        progress: ProgressSink | None = None,
        task_name: str = "evolve",
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
            label_contract_digest=self.label_contract_registry.digest,
        )
        errors = detect_leakage(rows + validation_rows, manifest)
        if errors:
            raise ValueError("validation data leakage: " + "; ".join(errors))
        outcome = await self.evolve_once(rows, progress=progress, task_name=task_name)
        decisions = []
        for proposal in outcome["proposals"]:
            candidate_skills = apply_candidate(self.skills, proposal)
            repeated_baseline = []
            repeated_candidate = []
            for repeat_index in range(self.config.gate.repeats):
                _, base_run = await self.run(
                    validation_rows,
                    progress=progress,
                    task_name=task_name,
                    phase=f"validation baseline r{repeat_index + 1}",
                )
                _, candidate_run = await self.run(
                    validation_rows,
                    skills=candidate_skills,
                    progress=progress,
                    task_name=task_name,
                    phase=f"validation candidate r{repeat_index + 1}",
                )
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
            package_candidate = outcome["package_candidates"].get(proposal.proposal_id)
            levels = (
                [package_candidate.safety_level]
                if package_candidate is not None
                else [
                    scan_resources(candidate.resources).level
                    for candidate in proposal.candidate_skills
                ]
            )
            safety = (
                "blocked"
                if "blocked" in levels
                else ("review_required" if "review_required" in levels else "safe")
            )
            decisions.append(
                ValidationGate(
                    self.config.gate,
                    require_cost=(
                        self.config.pricing.require_cost_for_promotion
                        and self.config.backend != "mock"
                    ),
                ).decide(baseline_result, candidate_result, test, safety=safety)
            )
        outcome["gate_decisions"] = decisions
        return outcome

    async def closed_loop_batched(
        self,
        samples: list[Sample] | None,
        validation_samples: list[Sample] | None,
        *,
        repository=None,
        evaluation_only: bool = False,
        progress: ProgressSink | None = None,
        start_batch: int = 0,
        prior_batch_audit: list[dict] | None = None,
        prior_state: dict | None = None,
        checkpoint_callback=None,
        training_run_id: str | None = None,
        provenance: dict | None = None,
    ):
        """Evolve sequential training batches with one atomic bank update per batch."""
        from evofact.data.domains import skillbank_fingerprint

        rows = fixture_samples() if samples is None else samples
        validation_rows = (
            fixture_validation_samples() if validation_samples is None else validation_samples
        )
        if not rows or not validation_rows:
            raise ValueError("evolution requires non-empty training and validation samples")
        batch_size = self.config.execution.batch_size
        batch_count = (len(rows) + batch_size - 1) // batch_size
        if start_batch < 0 or start_batch > batch_count:
            raise ValueError("resume batch is outside the configured training range")
        if evaluation_only:
            if start_batch or checkpoint_callback is not None:
                raise ValueError("evaluation-only evolution cannot be resumed")
            from evofact.skills.repository import SkillRepository

            with tempfile.TemporaryDirectory(prefix="evofact-evaluation-only-") as temp:
                result = await self.closed_loop_batched(
                    rows,
                    validation_rows,
                    repository=SkillRepository(Path(temp) / "skill-store"),
                    evaluation_only=False,
                    progress=progress,
                    provenance=provenance,
                )
            result["evaluation_only"] = True
            return result
        if repository is None:
            raise ValueError("repository is required unless evaluation_only is enabled")
        active_packages = repository.active_packages() if repository is not None else {}
        if repository is not None and not evaluation_only and not active_packages:
            if repository.active():
                legacy_packages = project_skill_bank_to_packages(
                    self.packages, list(repository.active().values())
                )
                repository.commit_package_bank(
                    legacy_packages,
                    run_id="migrate-active-v2",
                    expected_active={},
                    audit={"source": "legacy-active-v2", "provenance": provenance or {}},
                )
            else:
                repository.commit_package_bank(
                    self.packages,
                    run_id="initialize-package-bank-" + package_bank_digest(self.packages)[:16],
                    expected_active={},
                    audit={"source": "seed-packages", "provenance": provenance or {}},
                )
            active_packages = repository.active_packages()
        if active_packages:
            self.packages = list(active_packages.values())
            self.skills = [package_to_skill_spec(package) for package in self.packages]

        restored = prior_state or {}
        all_traces = list(restored.get("traces", []))
        all_rows = [
            item
            if isinstance(item, SampleEvaluation)
            else SampleEvaluation(
                **{
                    **item,
                    "allowed_labels": tuple(item.get("allowed_labels", ("REAL", "FAKE"))),
                }
            )
            for item in restored.get("sample_evaluations", [])
        ]
        all_attributions = list(restored.get("attributions", []))
        all_proposals = list(restored.get("proposals", []))
        all_decisions = list(restored.get("gate_decisions", []))
        batch_audit = list(prior_batch_audit or [])
        utilities = dict(restored.get("utilities", {}))
        distillation = list(restored.get("distillation", []))
        for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
            if batch_index <= start_batch:
                continue
            batch = rows[start : start + batch_size]
            baseline = list(self.skills)
            baseline_id = skillbank_fingerprint(baseline)
            if progress is not None:
                progress.update(
                    ProgressEvent(
                        "evolve",
                        "batch",
                        batch_index - 1,
                        batch_count,
                        "batches",
                        f"batch {batch_index}/{batch_count}: infer",
                    )
                )
            outcome = await self.closed_loop(
                batch,
                validation_rows,
                progress=progress,
                task_name=f"evolve batch {batch_index}/{batch_count}",
            )
            pairs = sorted(
                zip(outcome["proposals"], outcome["gate_decisions"]),
                key=lambda pair: (pair[0].proposal_id, pair[0].target_skill_ids),
            )
            candidate_bank = list(baseline)
            accepted = []
            for proposal, decision in pairs:
                if decision.disposition == "active":
                    candidate_bank = apply_candidate(candidate_bank, proposal)
                    accepted.append(proposal)
                    continue
                package_candidate = outcome.get("package_candidates", {}).get(proposal.proposal_id)
                if package_candidate is not None and repository is not None and not evaluation_only:
                    repository.save_package(
                        replace(
                            package_candidate.package,
                            status=(
                                SkillStatus.PARETO
                                if decision.disposition == "pareto"
                                else SkillStatus.CANDIDATE
                            ),
                        )
                    )
                for skill in proposal.candidate_skills:
                    if repository is None or evaluation_only:
                        continue
                    repository.save(
                        replace(
                            skill,
                            status=SkillStatus.PARETO
                            if decision.disposition == "pareto"
                            else SkillStatus.CANDIDATE,
                        ),
                        decision.disposition,
                    )
            snapshots = ()
            if accepted and repository is not None and not evaluation_only:
                proposal_ids = tuple(item.proposal_id for item in accepted)
                run_id = (
                    (f"{training_run_id}-" if training_run_id else "")
                    + f"batch-{batch_index}-"
                    + skillbank_fingerprint(baseline)[:12]
                    + "-"
                    + skillbank_fingerprint(candidate_bank)[:12]
                )
                exact_candidates = [
                    outcome.get("package_candidates", {}).get(proposal.proposal_id)
                    for proposal in accepted
                ]
                if all(candidate is not None for candidate in exact_candidates):
                    candidate_packages = list(self.packages)
                    for candidate in exact_candidates:
                        if candidate.base is not None:
                            candidate_packages = [
                                package
                                for package in candidate_packages
                                if package.skill_id != candidate.base.skill_id
                            ]
                        candidate_packages.append(
                            replace(candidate.package, status=SkillStatus.ACTIVE)
                        )
                else:
                    candidate_packages = project_skill_bank_to_packages(
                        self.packages, candidate_bank
                    )
                expected_packages = {
                    package.manifest.name: package.package_digest for package in self.packages
                }
                snapshots = repository.commit_package_bank(
                    candidate_packages,
                    run_id=run_id,
                    expected_active=expected_packages,
                    audit={
                        "batch_index": batch_index,
                        "proposal_ids": proposal_ids,
                        "decisions": [asdict(decision) for _, decision in pairs],
                        "provenance": provenance or {},
                    },
                )
                self.packages = list(repository.active_packages().values())
                self.skills = [package_to_skill_spec(package) for package in self.packages]
            if skillbank_fingerprint(baseline) != baseline_id:
                raise RuntimeError("evolution baseline mutated during batch evaluation")

            all_traces.extend(outcome["traces"])
            all_rows.extend(outcome["evaluation"].per_sample)
            all_attributions.extend(outcome["attributions"])
            all_proposals.extend(outcome["proposals"])
            all_decisions.extend(outcome["gate_decisions"])
            utilities.update(outcome["utilities"])
            distillation.append(outcome["distillation"])
            batch_audit.append(
                {
                    "batch_index": batch_index,
                    "sample_ids": [sample.sample_id for sample in batch],
                    "baseline_fingerprint": baseline_id,
                    "result_fingerprint": skillbank_fingerprint(self.skills),
                    "package_bank_digest": package_bank_digest(self.packages),
                    "accepted_proposal_ids": [item.proposal_id for item in accepted],
                    "committed_snapshots": list(snapshots),
                }
            )
            if progress is not None:
                progress.update(
                    ProgressEvent(
                        "evolve",
                        "batch",
                        batch_index,
                        batch_count,
                        "batches",
                        f"batch {batch_index}/{batch_count}: complete",
                    )
                )
            if checkpoint_callback is not None:
                checkpoint_result = checkpoint_callback(
                    {
                        "completed_batches": batch_index,
                        "total_batches": batch_count,
                        "batch_audit": batch_audit,
                        "active_packages": {
                            package.manifest.name: package.package_digest
                            for package in self.packages
                        },
                        "budget_state": self._budget_manager().export_state(),
                        "accumulated": {
                            "traces": all_traces,
                            "sample_evaluations": all_rows,
                            "attributions": all_attributions,
                            "proposals": all_proposals,
                            "gate_decisions": all_decisions,
                            "utilities": utilities,
                            "distillation": distillation,
                        },
                    }
                )
                if inspect.isawaitable(checkpoint_result):
                    await checkpoint_result

        return {
            "traces": all_traces,
            "evaluation": evaluate(all_rows),
            "attributions": all_attributions,
            "utilities": utilities,
            "distillation": distillation,
            "proposals": all_proposals,
            "gate_decisions": all_decisions,
            "budget_snapshot": self._budget_manager().snapshot(),
            "batches": batch_audit,
            "evaluation_only": evaluation_only,
            "resumed_from_batch": start_batch,
            "training_run_id": training_run_id,
            "provenance": provenance or {},
        }


def _budget_abstention_trace(sample: Sample, skills, reason: str, contract=None) -> InferenceTrace:
    """预算耗尽时构造一条 ABSTAIN 轨迹，使整轮实验可以继续推进而不是整体失败。
    输入要求：`sample`（Sample）需符合函数签名约定；`skills` 为当前技能库快照；`reason`（str）为预算拒绝原因。
    输出：返回 `InferenceTrace` 类型结果；本函数不发起任何模型调用。"""
    return InferenceTrace(
        trace_id=hashlib.sha256(f"{sample.sample_id}:budget".encode()).hexdigest()[:20],
        sample_id=sample.sample_id,
        sample_public=sample.public_view(),
        routing=RoutingDecision((), (), {"budget": reason}, 0.0, RunBudget(), True),
        specialist_reports=(),
        decision=Prediction(
            RUNTIME_ABSTAIN_LABEL,
            0.0,
            f"Budget exhausted: {reason}",
            DecisionOrigin.RUNTIME,
        ),
        skill_versions={s.skill_id: s.package_digest or s.version for s in skills},
        errors=(f"budget exhausted: {reason}",),
        label_schema_id=contract.schema_id if contract is not None else None,
        label_contract_digest=contract.digest if contract is not None else None,
    )


def _gold(value) -> str:
    """Legacy fixture-only adapter; production paths resolve the dataset contract."""
    return fixture_binary_contract().normalize(value)
