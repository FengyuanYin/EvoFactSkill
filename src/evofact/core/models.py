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
    text: str
    source: str | None = None
    published_at: datetime | None = None
    stance: Literal["support", "refute", "neutral", "unknown"] = "unknown"


@dataclass(frozen=True)
class Sample(ModelMixin):
    sample_id: str
    dataset: str
    text: str
    label: str | int | None = None
    domain: str | None = None
    event_id: str | None = None
    published_at: datetime | None = None
    evidence: tuple[Evidence, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_view(self) -> dict[str, Any]:
        from .redaction import redact_sensitive

        return redact_sensitive(self.model_dump())


@dataclass(frozen=True)
class DataManifest(ModelMixin):
    manifest_id: str
    dataset_fingerprints: dict[str, str]
    train_ids: tuple[str, ...] = ()
    evolution_validation_ids: tuple[str, ...] = ()
    protected_validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    split_policy: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "data_manifest_v1"

    def __post_init__(self) -> None:
        groups = [set(self.train_ids), set(self.evolution_validation_ids),
                  set(self.protected_validation_ids), set(self.test_ids)]
        if sum(map(len, groups)) != len(set().union(*groups)):
            raise ValueError("manifest splits must be disjoint")


class SkillStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PARETO = "pareto"
    FROZEN = "frozen"
    RETIRED = "retired"


class SkillKind(StrEnum):
    ROUTER = "router"
    SPECIALIST = "specialist"
    JUDGE = "judge"
    WORKFLOW = "workflow"
    META = "meta"


@dataclass(frozen=True)
class SkillScope(ModelMixin):
    domains: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    temporal_windows: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trigger(ModelMixin):
    feature: str
    pattern: str
    weight: float = 1.0


@dataclass(frozen=True)
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
class SkillUtility(ModelMixin):
    skill_id: str
    uses: int = 0
    successes: int = 0
    marginal_utility: float = 0.0
    mean_cost: float = 0.0
    domain_utility: dict[str, float] = field(default_factory=dict)
    temporal_utility: dict[str, float] = field(default_factory=dict)
    negative_transfer_count: int = 0


@dataclass(frozen=True)
class RunBudget(ModelMixin):
    max_skills: int = 3
    max_calls: int = 6
    max_tokens: int = 8000


@dataclass(frozen=True)
class RoutingDecision(ModelMixin):
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
class SpecialistReport(ModelMixin):
    skill_id: str
    claims: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    assessment: str = "unknown"
    confidence: float = 0.0
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction(ModelMixin):
    label: Literal["REAL", "FAKE", "ABSTAIN"]
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class UsageRecord(ModelMixin):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0


@dataclass(frozen=True)
class InferenceTrace(ModelMixin):
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


class ErrorType(StrEnum):
    ROUTING_MISS = "routing_miss"
    EVIDENCE_MISS = "evidence_miss"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    REASONING_ERROR = "reasoning_error"
    JUDGE_AGGREGATION_ERROR = "judge_aggregation_error"
    LABEL_MAPPING_ERROR = "label_mapping_error"
    ABSTENTION_ERROR = "abstention_error"


@dataclass(frozen=True)
class AttributionReport(ModelMixin):
    trace_id: str
    error_types: tuple[ErrorType, ...]
    responsible_skill_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    counterfactual_deltas: dict[str, float] = field(default_factory=dict)


class EvolutionOperation(StrEnum):
    ADD = "add"
    EDIT = "edit"
    SPLIT = "split"
    MERGE = "merge"
    GENERALIZE = "generalize"
    SPECIALIZE = "specialize"
    RETIRE = "retire"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class EvolutionProposal(ModelMixin):
    proposal_id: str
    operation: EvolutionOperation
    rationale: str
    target_skill_ids: tuple[str, ...] = ()
    candidate_skills: tuple[SkillSpec, ...] = ()
    source_trace_ids: tuple[str, ...] = ()
    error_cluster_id: str | None = None
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleEvaluation(ModelMixin):
    sample_id: str
    gold: Literal["REAL", "FAKE"]
    predicted: Literal["REAL", "FAKE", "ABSTAIN"]
    confidence: float
    domain: str | None = None
    temporal_window: str | None = None
    cost: float = 0.0


@dataclass(frozen=True)
class EvaluationResult(ModelMixin):
    per_sample: tuple[SampleEvaluation, ...]
    aggregate_metrics: dict[str, float]
    domain_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    temporal_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    usage: UsageRecord = field(default_factory=UsageRecord)


@dataclass(frozen=True)
class StatisticalTestResult(ModelMixin):
    name: str
    statistic: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class GateDecision(ModelMixin):
    accepted: bool
    disposition: Literal["active", "pareto", "rejected", "review_required"]
    baseline_result: EvaluationResult
    candidate_result: EvaluationResult
    paired_test: StatisticalTestResult
    regression_failures: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class DomainEpisode(ModelMixin):
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
        train, test, final = set(self.meta_train_domains), set(self.meta_test_domains), set(self.final_test_domains)
        if not train or not test:
            raise ValueError("meta-train and meta-test domains must be non-empty")
        if train & test or (train | test) & final:
            raise ValueError("episode domains must be disjoint from each other and final test")
        if set(self.meta_train_sample_ids) & set(self.meta_test_sample_ids):
            raise ValueError("episode sample sets must be disjoint")


@dataclass(frozen=True)
class CandidateIdentity(ModelMixin):
    fingerprint: str
    operation: EvolutionOperation
    parent_skill_ids: tuple[str, ...]
    normalized_content_hash: str
    scope_signature: str
    schema_version: str = "candidate_identity_v1"


@dataclass(frozen=True)
class MetaTrainView(ModelMixin):
    episode_id: str
    trace_ids: tuple[str, ...]
    sample_views: tuple[dict[str, Any], ...]
    attribution_summaries: tuple[dict[str, Any], ...]
    schema_version: str = "meta_train_view_v1"


@dataclass(frozen=True)
class EpisodeEvaluation(ModelMixin):
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
class TransferUtility(ModelMixin):
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
class MetaGateDecision(ModelMixin):
    candidate_fingerprint: str
    accepted: bool
    disposition: Literal["generalized", "specialized", "pareto", "rejected", "retired", "review_required"]
    target_domains: tuple[str, ...]
    utility: TransferUtility
    failures: tuple[str, ...] = ()
    reason: str = ""
    schema_version: str = "meta_gate_decision_v1"


@dataclass(frozen=True)
class MetaCheckpoint(ModelMixin):
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
class MetaEvolutionOutcome(ModelMixin):
    run_id: str
    episodes: tuple[DomainEpisode, ...]
    episode_results: tuple[EpisodeEvaluation, ...]
    utilities: dict[str, TransferUtility]
    decisions: tuple[MetaGateDecision, ...]
    committed_snapshots: tuple[str, ...] = ()
    mock_results: bool = False
    schema_version: str = "meta_evolution_outcome_v1"
