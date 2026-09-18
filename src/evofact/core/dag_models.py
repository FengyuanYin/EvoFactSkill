from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from evofact.governance import EXECUTION_PLAN_SCHEMA_VERSION

from .models import ModelMixin, Prediction, RunBudget, SpecialistReport, UsageRecord


@dataclass(frozen=True)
class PlanNode(ModelMixin):
    node_id: str
    skill_id: str
    depends_on: tuple[str, ...] = ()
    required: bool = True
    upstream_outputs: tuple[str, ...] = ()
    priority: int = 0
    timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.skill_id.strip():
            raise ValueError("plan node IDs must not be empty")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("plan node dependencies must be unique")
        if self.node_id in self.depends_on:
            raise ValueError("plan node cannot depend on itself")
        if self.timeout_ms is not None and self.timeout_ms < 1:
            raise ValueError("node timeout must be positive")


@dataclass(frozen=True)
class ExecutionPlan(ModelMixin):
    plan_id: str
    nodes: tuple[PlanNode, ...]
    reasons: dict[str, str]
    confidence: float
    budget: RunBudget
    fallback_used: bool = False
    schema_version: str = EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("plan confidence must be in [0, 1]")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("plan node IDs must be unique")
        if self.schema_version != EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported execution plan schema: {self.schema_version}")


@dataclass(frozen=True)
class PlanFinding(ModelMixin):
    code: str
    message: str
    node_id: str | None = None


@dataclass(frozen=True)
class PlanValidationReport(ModelMixin):
    valid: bool
    plan: ExecutionPlan | None
    topological_order: tuple[str, ...] = ()
    levels: tuple[tuple[str, ...], ...] = ()
    depth: int = 0
    findings: tuple[PlanFinding, ...] = ()
    warnings: tuple[PlanFinding, ...] = ()


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED_DEPENDENCY = "skipped_dependency"
    SKIPPED_BUDGET = "skipped_budget"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class ResourceSelection(ModelMixin):
    references: tuple[str, ...] = ()
    template: str | None = None
    script: str | None = None
    digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeExecution(ModelMixin):
    node_id: str
    skill_id: str
    status: NodeStatus
    upstream_node_ids: tuple[str, ...] = ()
    report: SpecialistReport | None = None
    usage: UsageRecord = field(default_factory=UsageRecord)
    resources: ResourceSelection = field(default_factory=ResourceSelection)
    started_at: str | None = None
    finished_at: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    usage_details: Any | None = None
    started_sequence: int | None = None
    completed_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.status == NodeStatus.SUCCEEDED and self.report is None:
            raise ValueError("successful node execution requires a report")
        if self.status != NodeStatus.SUCCEEDED and self.report is not None:
            raise ValueError("non-successful node execution cannot contain a report")
        if self.status in {NodeStatus.FAILED, NodeStatus.TIMED_OUT} and not self.error:
            raise ValueError("failed/timed-out node execution requires an error")


@dataclass(frozen=True)
class ExecutionSummary(ModelMixin):
    required_complete: bool
    succeeded: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DAGExecutionResult(ModelMixin):
    plan: ExecutionPlan
    nodes: tuple[NodeExecution, ...]
    summary: ExecutionSummary
    usage: UsageRecord = field(default_factory=UsageRecord)
    prediction: Prediction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage_details: Any | None = None
