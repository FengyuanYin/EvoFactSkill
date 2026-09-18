from __future__ import annotations

import asyncio
from dataclasses import replace

from evofact.core.budget_models import BudgetLimits
from evofact.core.dag_models import (
    DAGExecutionResult,
    ExecutionPlan,
    ExecutionSummary,
    NodeExecution,
    NodeStatus,
)
from evofact.core.models import UsageRecord
from evofact.governance.dag_policy import require_valid_plan

from .node_runner import NodeRunner, merge_usage_details


def _merge_usage(items: list[UsageRecord]) -> UsageRecord:
    return UsageRecord(
        calls=sum(item.calls for item in items),
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
        latency_ms=sum(item.latency_ms for item in items),
        estimated_cost=sum(item.estimated_cost for item in items),
    )


class DAGExecutor:
    def __init__(
        self,
        backend,
        skills,
        *,
        resource_runtime=None,
        sample_timeout_ms: int | None = None,
        budget_manager=None,
        plan_limits: BudgetLimits = BudgetLimits(),
        available_capabilities: tuple[str, ...] = ("llm", "references", "template"),
    ):
        self.skills = {item.skill_id: item for item in skills}
        self.runner = NodeRunner(
            backend,
            self.skills,
            resource_runtime=resource_runtime,
            budget_manager=budget_manager,
        )
        self.sample_timeout_ms = sample_timeout_ms
        self.plan_limits = plan_limits
        self.available_capabilities = available_capabilities

    async def execute(
        self, plan: ExecutionPlan, sample: dict, *, sample_id: str = "sample"
    ) -> DAGExecutionResult:
        validation = require_valid_plan(
            plan,
            self.skills,
            limits=self.plan_limits,
            available_capabilities=self.available_capabilities,
        )

        async def run_all() -> tuple[NodeExecution, ...]:
            completed: dict[str, NodeExecution] = {}
            by_node = {item.node_id: item for item in plan.nodes}
            sequence = 0
            sequence_lock = asyncio.Lock()

            async def run_observed(node):
                nonlocal sequence
                async with sequence_lock:
                    sequence += 1
                    started_sequence = sequence
                result = await self.runner.run(
                    node,
                    sample,
                    completed,
                    sample_id=sample_id,
                )
                async with sequence_lock:
                    sequence += 1
                    completed_sequence = sequence
                return replace(
                    result,
                    started_sequence=started_sequence,
                    completed_sequence=completed_sequence,
                )

            for level in validation.levels:
                runnable = []
                for node_id in level:
                    node = by_node[node_id]
                    dependency_states = [completed[item].status for item in node.depends_on]
                    if any(state != NodeStatus.SUCCEEDED for state in dependency_states):
                        completed[node_id] = NodeExecution(
                            node_id,
                            node.skill_id,
                            NodeStatus.SKIPPED_DEPENDENCY,
                            tuple(sorted(node.depends_on)),
                        )
                    else:
                        runnable.append(node)
                results = await asyncio.gather(*(run_observed(node) for node in runnable))
                completed.update((item.node_id, item) for item in results)
            return tuple(completed[node_id] for node_id in validation.topological_order)

        try:
            nodes = await asyncio.wait_for(
                run_all(),
                timeout=self.sample_timeout_ms / 1000 if self.sample_timeout_ms else None,
            )
        except asyncio.TimeoutError:
            raise TimeoutError("sample DAG execution timed out") from None
        required_ids = {item.node_id for item in plan.nodes if item.required}
        succeeded = tuple(item.node_id for item in nodes if item.status == NodeStatus.SUCCEEDED)
        failed = tuple(
            item.node_id
            for item in nodes
            if item.status in {NodeStatus.FAILED, NodeStatus.TIMED_OUT}
        )
        skipped = tuple(
            item.node_id
            for item in nodes
            if item.status in {NodeStatus.SKIPPED_DEPENDENCY, NodeStatus.SKIPPED_BUDGET}
        )
        summary = ExecutionSummary(
            required_complete=required_ids <= set(succeeded),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            errors=tuple(item.error for item in nodes if item.error),
        )
        return DAGExecutionResult(
            plan,
            nodes,
            summary,
            _merge_usage([item.usage for item in nodes]),
            usage_details=merge_usage_details(item.usage_details for item in nodes),
        )
