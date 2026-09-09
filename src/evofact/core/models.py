"""
数据层面：Evidence、Sample、DataManifest

技能层面：SkillSpec、SkillStatus、SkillKind、SkillScope、SkillUtility 等

推理层面：RoutingDecision、SpecialistReport、Prediction、InferenceTrace

错误分析：ErrorType、AttributionReport

进化机制：EvolutionOperation、EvolutionProposal、CandidateIdentity

评估与门控：EvaluationResult、GateDecision、MetaGateDecision

元学习流程：DomainEpisode、MetaTrainView、EpisodeEvaluation、TransferUtility、MetaCheckpoint、MetaEvolutionOutcome
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal


class ModelMixin:
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return asdict(self)


@dataclass(frozen=True)
class Evidence(ModelMixin):
    text: str  # 证据文本
    source: str | None = None  # 来源（如网址、数据库名）
    published_at: datetime | None = None  # 发布时间
    stance: Literal["support", "refute", "neutral", "unknown"] = (
        "unknown"  # 立场（支持、反驳、中立、未知）
    )


@dataclass(frozen=True)
class Sample(ModelMixin):
    sample_id: str  # 样本唯一标识
    dataset: str  # 所属数据集名称
    text: str  # 新闻文本内容
    label: str | int | None = None  # 标签（如 REAL/FAKE，也可能是其他类型）
    domain: str | None = None  # 领域（例如政治、健康）
    event_id: str | None = None  # 关联事件 ID
    published_at: datetime | None = None  # 发布时间
    evidence: tuple[Evidence, ...] = ()  # 该样本关联的证据元组
    metadata: dict[str, Any] = field(default_factory=dict)  # 其他元数据

    def public_view(
        self,
    ) -> dict[str, Any]:  # 返回经过敏感信息脱敏（redact_sensitive）后的字典视图，用于展示或日志
        from .redaction import redact_sensitive

        return redact_sensitive(self.model_dump())


@dataclass(frozen=True)
class DataManifest(
    ModelMixin
):  # 数据清单，记录数据集分割信息（训练、验证、测试等），并确保各个分割集合互不相交。
    manifest_id: str
    dataset_fingerprints: dict[str, str]
    train_ids: tuple[str, ...] = ()
    evolution_validation_ids: tuple[str, ...] = ()
    protected_validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    split_policy: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "data_manifest_v1"

    def __post_init__(self) -> None:
        groups = [
            set(self.train_ids),
            set(self.evolution_validation_ids),
            set(self.protected_validation_ids),
            set(self.test_ids),
        ]
        if sum(map(len, groups)) != len(set().union(*groups)):
            raise ValueError("manifest splits must be disjoint")


class SkillStatus(StrEnum):  # 技能状态（候选、活跃、帕累托最优、冻结、退役）
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PARETO = "pareto"
    FROZEN = "frozen"
    RETIRED = "retired"


class SkillKind(StrEnum):  # 技能类型（路由、专家、裁判、工作流、元技能）
    ROUTER = "router"
    SPECIALIST = "specialist"
    JUDGE = "judge"
    WORKFLOW = "workflow"
    META = "meta"


@dataclass(frozen=True)  # 技能适用范围，限定技能在哪些领域、数据集、时间窗口或标签下工作
class SkillScope(ModelMixin):
    domains: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    temporal_windows: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)  # 触发条件，根据输入特征匹配模式来决定是否激活某个技能，权重表示重要性
class Trigger(ModelMixin):
    feature: str
    pattern: str
    weight: float = 1.0


@dataclass(
    frozen=True
)  # 技能的完整定义，包括标识、类型、版本、状态、指令文本、资源引用、适用范围、触发条件、父级技能（用于进化谱系）以及安全级别（纯文本、需要审查、可执行）
class SkillSpec(ModelMixin):
    skill_id: str
    name: str
    kind: SkillKind
    version: str
    status: SkillStatus
    instructions: str
    resources: dict[str, str] = field(default_factory=dict)
    scope: SkillScope = field(default_factory=SkillScope)
    triggers: tuple[Trigger, ...] = ()
    parent_ids: tuple[str, ...] = ()
    safety_level: Literal["text_only", "review_required", "executable"] = "text_only"
    schema_version: str = "skill_spec_v1"


@dataclass(frozen=True)
class SkillUtility(ModelMixin):  # 记录技能的使用情况和效用指标，用于评估技能表现和决定是否保留/进化
    skill_id: str
    uses: int = 0
    successes: int = 0
    marginal_utility: float = 0.0
    mean_cost: float = 0.0
    domain_utility: dict[str, float] = field(default_factory=dict)
    temporal_utility: dict[str, float] = field(default_factory=dict)
    negative_transfer_count: int = 0


@dataclass(frozen=True)
class RunBudget(ModelMixin):  # 资源限制，控制一次推理中最多调用多少技能、多少次、消耗多少 token
    max_skills: int = 3
    max_calls: int = 6
    max_tokens: int = 8000


@dataclass(frozen=True)
class RoutingDecision(
    ModelMixin
):  # 路由决策结果，记录选中的技能 ID、拒绝的技能 ID、决策原因、置信度、使用的预算以及是否使用了回退策略
    selected_skill_ids: tuple[str, ...]
    rejected_skill_ids: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    budget: RunBudget = field(default_factory=RunBudget)
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class SpecialistReport(
    ModelMixin
):  # 专家技能的产出报告，包含其提出的声明、使用的证据、评估结论和局限性
    skill_id: str
    claims: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    assessment: str = "unknown"
    confidence: float = 0.0
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction(ModelMixin):  # 最终预测结果，标签为 REAL / FAKE / ABSTAIN（弃权），附置信度和理由
    label: Literal["REAL", "FAKE", "ABSTAIN"]
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class UsageRecord(ModelMixin):  # 资源使用记录，统计调用次数、token 消耗、延迟和预估成本
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0


@dataclass(frozen=True)
class InferenceTrace(
    ModelMixin
):  # 完整推理过程记录，包括样本公共视图、路由决策、专家报告、最终决策、技能版本、聚合证据、资源使用和错误列表
    trace_id: str
    sample_public: dict[str, Any]
    routing: RoutingDecision
    specialist_reports: tuple[SpecialistReport, ...]
    decision: Prediction
    skill_versions: dict[str, str]
    aggregated_evidence: tuple[Evidence, ...] = ()
    usage: UsageRecord = field(default_factory=UsageRecord)
    errors: tuple[str, ...] = ()
    schema_version: str = "inference_trace_v1"


class ErrorType(
    StrEnum
):  # 错误类型枚举，用于标识系统在推理过程中可能出现的各种问题（路由失败、证据缺失、证据幻觉、时间泄漏、推理错误等）
    ROUTING_MISS = "routing_miss"
    EVIDENCE_MISS = "evidence_miss"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    REASONING_ERROR = "reasoning_error"
    JUDGE_AGGREGATION_ERROR = "judge_aggregation_error"
    LABEL_MAPPING_ERROR = "label_mapping_error"
    ABSTENTION_ERROR = "abstention_error"


@dataclass(frozen=True)
class AttributionReport(
    ModelMixin
):  # 错误归因报告，指出哪些技能应对哪些错误负责，并提供反事实增量（移除某技能后的性能变化）
    trace_id: str
    error_types: tuple[ErrorType, ...]
    responsible_skill_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    counterfactual_deltas: dict[str, float] = field(default_factory=dict)


class EvolutionOperation(
    StrEnum
):  # 进化操作枚举，定义技能演化的方式（添加、编辑、拆分、合并、泛化、特化、退役、回滚）
    ADD = "add"
    EDIT = "edit"
    SPLIT = "split"
    MERGE = "merge"
    GENERALIZE = "generalize"
    SPECIALIZE = "specialize"
    RETIRE = "retire"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class EvolutionProposal(
    ModelMixin
):  # 进化提案，描述一次技能修改的意图、操作类型、理由、涉及的目标技能、候选新技能、依据的推理追踪等
    proposal_id: str
    operation: EvolutionOperation
    rationale: str
    target_skill_ids: tuple[str, ...] = ()
    candidate_skills: tuple[SkillSpec, ...] = ()
    source_trace_ids: tuple[str, ...] = ()
    error_cluster_id: str | None = None
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleEvaluation(
    ModelMixin
):  # 单个样本的评估结果，包含真实标签、预测标签、置信度、领域和时间窗口等信息
    sample_id: str
    gold: Literal["REAL", "FAKE"]
    predicted: Literal["REAL", "FAKE", "ABSTAIN"]
    confidence: float
    domain: str | None = None
    temporal_window: str | None = None
    cost: float = 0.0


@dataclass(frozen=True)
class EvaluationResult(
    ModelMixin
):  # 整体评估结果，包括逐样本结果、聚合指标、按领域/时间划分的指标、置信区间以及资源使用
    per_sample: tuple[SampleEvaluation, ...]
    aggregate_metrics: dict[str, float]
    domain_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    temporal_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    usage: UsageRecord = field(default_factory=UsageRecord)


@dataclass(frozen=True)
class StatisticalTestResult(ModelMixin):  # 统计检验结果，包含检验名称、统计量、p 值和是否显著
    name: str
    statistic: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class GateDecision(
    ModelMixin
):  # 门控决策，决定是否接受一个进化候选技能。包含基线评估、候选评估、配对检验结果以及回归失败信息
    accepted: bool
    disposition: Literal["active", "pareto", "rejected", "review_required"]
    baseline_result: EvaluationResult
    candidate_result: EvaluationResult
    paired_test: StatisticalTestResult
    regression_failures: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class DomainEpisode(
    ModelMixin
):  # 定义一次元学习片段（episode），将领域划分为元训练域和元测试域，以及最终测试域。__post_init__ 校验这些域之间互斥，且样本集不相交。这是元进化过程中的基本单元
    episode_id: str
    seed: int
    strategy: Literal["repeated_holdout", "leave_one_domain_out"]
    meta_train_domains: tuple[str, ...]
    meta_test_domains: tuple[str, ...]
    meta_train_sample_ids: tuple[str, ...]
    meta_test_sample_ids: tuple[str, ...]
    final_test_domains: tuple[str, ...] = ()
    data_fingerprint: str = ""
    skillbank_snapshot_id: str = ""
    schema_version: str = "domain_episode_v1"

    def __post_init__(self) -> None:
        train, test, final = (
            set(self.meta_train_domains),
            set(self.meta_test_domains),
            set(self.final_test_domains),
        )
        if not train or not test:
            raise ValueError("meta-train and meta-test domains must be non-empty")
        if train & test or (train | test) & final:
            raise ValueError("episode domains must be disjoint from each other and final test")
        if set(self.meta_train_sample_ids) & set(self.meta_test_sample_ids):
            raise ValueError("episode sample sets must be disjoint")


@dataclass(frozen=True)
class CandidateIdentity(
    ModelMixin
):  # 候选技能的唯一标识，由指纹、操作类型、父技能、内容哈希和作用域签名组成，用于跨片段追踪同一候选技能
    fingerprint: str
    operation: EvolutionOperation
    parent_skill_ids: tuple[str, ...]
    normalized_content_hash: str
    scope_signature: str
    schema_version: str = "candidate_identity_v1"


@dataclass(frozen=True)
class MetaTrainView(
    ModelMixin
):  # 元训练视图，聚合了一个片段内的推理追踪 ID、样本视图和归因摘要，供元学习器使用
    episode_id: str
    trace_ids: tuple[str, ...]
    sample_views: tuple[dict[str, Any], ...]
    attribution_summaries: tuple[dict[str, Any], ...]
    schema_version: str = "meta_train_view_v1"


@dataclass(frozen=True)
class EpisodeEvaluation(
    ModelMixin
):  # 片段评估结果，比较基线和候选技能在特定片段上的表现，记录总体和分领域的性能差异
    episode_id: str
    candidate_fingerprint: str
    proposal_id: str
    baseline_result: EvaluationResult
    candidate_result: EvaluationResult
    aggregate_deltas: dict[str, float]
    domain_deltas: dict[str, dict[str, float]]
    safety_level: str = "safe"
    schema_version: str = "episode_evaluation_v1"


@dataclass(frozen=True)
class TransferUtility(
    ModelMixin
):  # 候选技能在多个片段上的综合迁移效用，包括平均增益、标准差、置信区间、成功率、负迁移率、最差领域下降等指标
    candidate_fingerprint: str
    episode_count: int
    tested_domains: tuple[str, ...]
    mean_gain: float
    std_gain: float
    confidence_interval: tuple[float, float]
    success_rate: float
    negative_transfer_rate: float
    worst_domain_drop: float
    mean_coverage: float
    coverage_delta: float
    calibration_delta: float
    cost_ratio: float
    domain_gains: dict[str, float]
    schema_version: str = "transfer_utility_v1"

    def __post_init__(self) -> None:
        if self.episode_count < 1:
            raise ValueError("transfer utility requires at least one episode")
        if not 0 <= self.success_rate <= 1 or not 0 <= self.negative_transfer_rate <= 1:
            raise ValueError("invalid transfer rate")


@dataclass(frozen=True)
class MetaGateDecision(
    ModelMixin
):  # 元门控决策，决定一个候选技能是否被接受以及其后续处置（泛化、特化、帕累托、拒绝、退役等）
    candidate_fingerprint: str
    accepted: bool
    disposition: Literal[
        "generalized", "specialized", "pareto", "rejected", "retired", "review_required"
    ]
    target_domains: tuple[str, ...]
    utility: TransferUtility
    failures: tuple[str, ...] = ()
    reason: str = ""
    schema_version: str = "meta_gate_decision_v1"


@dataclass(frozen=True)
class MetaCheckpoint(
    ModelMixin
):  # 元进化过程的检查点，保存运行状态，包括配置指纹、数据指纹、技能库快照、计划与已完成的片段 ID 等，用于断点续跑
    run_id: str
    config_fingerprint: str
    data_fingerprint: str
    skillbank_snapshot_id: str
    source_domains: tuple[str, ...]
    final_test_domains: tuple[str, ...]
    planned_episode_ids: tuple[str, ...]
    completed_episode_ids: tuple[str, ...] = ()
    episode_results: tuple[dict[str, Any], ...] = ()
    committed: bool = False
    schema_version: str = "meta_checkpoint_v1"

    def __post_init__(self) -> None:
        if not set(self.completed_episode_ids) <= set(self.planned_episode_ids):
            raise ValueError("completed episodes must be planned")


@dataclass(frozen=True)
class MetaEvolutionOutcome(
    ModelMixin
):  # 元进化运行的最终结果，汇总所有片段、评估、效用、决策和已提交的技能库快照
    run_id: str
    episodes: tuple[DomainEpisode, ...]
    episode_results: tuple[EpisodeEvaluation, ...]
    utilities: dict[str, TransferUtility]
    decisions: tuple[MetaGateDecision, ...]
    committed_snapshots: tuple[str, ...] = ()
    mock_results: bool = False
    schema_version: str = "meta_evolution_outcome_v1"
