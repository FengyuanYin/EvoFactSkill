from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal

from evofact.core.budget_models import BudgetRequest, CostStatus, UsageDetails
from evofact.core.dag_models import NodeExecution, NodeStatus, PlanNode, ResourceSelection
from evofact.core.models import SpecialistReport, UsageRecord

from .budget import BudgetExceeded


def _usage_details(result) -> UsageDetails:
    if result.usage_details is not None:
        return result.usage_details
    usage = result.usage
    return UsageDetails(
        calls=usage.calls,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=usage.latency_ms,
        cost=Decimal(str(usage.estimated_cost)) if usage.estimated_cost else None,
        cost_status=CostStatus.ESTIMATED if usage.estimated_cost else CostStatus.UNAVAILABLE,
    )


class NodeRunner:
    def __init__(self, backend, skills: dict, *, resource_runtime=None, budget_manager=None):
        self.backend = backend
        self.skills = skills
        self.resource_runtime = resource_runtime
        self.budget_manager = budget_manager

    async def run(
        self,
        node: PlanNode,
        sample: dict,
        completed: dict[str, NodeExecution],
        *,
        sample_id: str = "sample",
    ) -> NodeExecution:
        started = datetime.now(timezone.utc).isoformat()
        clock = time.perf_counter()
        allowed = set(node.upstream_outputs or node.depends_on)
        upstream = tuple(
            completed[node_id].report
            for node_id in sorted(allowed)
            if node_id in completed and completed[node_id].report is not None
        )
        resources = ResourceSelection()
        payload = sample
        if self.resource_runtime is not None:
            prepared = await self.resource_runtime.prepare(
                self.skills[node.skill_id], sample, upstream
            )
            payload = prepared.payload
            resources = prepared.selection
        reservation = None
        try:
            if self.budget_manager is not None:
                reservation = await self.budget_manager.reserve(
                    sample_id,
                    BudgetRequest(
                        calls=1,
                        tokens=1000,
                        required=node.required,
                        purpose=f"specialist:{node.skill_id}",
                    ),
                )

            async def invoke():
                try:
                    call = self.backend.analyze(
                        payload,
                        self.skills[node.skill_id],
                        upstream=upstream,
                        resources=resources,
                    )
                except TypeError:
                    call = self.backend.analyze(payload, self.skills[node.skill_id])
                return await asyncio.wait_for(
                    call,
                    timeout=node.timeout_ms / 1000 if node.timeout_ms else None,
                )

            if self.budget_manager is not None:
                async with self.budget_manager.concurrency(sample_id):
                    result = await invoke()
                await self.budget_manager.reconcile(reservation, _usage_details(result))
                reservation = None
            else:
                result = await invoke()
            if not isinstance(result.value, SpecialistReport):
                raise TypeError("backend analyze must return SpecialistReport")
            return NodeExecution(
                node.node_id,
                node.skill_id,
                NodeStatus.SUCCEEDED,
                tuple(sorted(allowed)),
                result.value,
                result.usage,
                resources,
                started,
                datetime.now(timezone.utc).isoformat(),
                (time.perf_counter() - clock) * 1000,
            )
        except BudgetExceeded:
            return NodeExecution(
                node.node_id,
                node.skill_id,
                NodeStatus.SKIPPED_BUDGET,
                tuple(sorted(allowed)),
            )
        except asyncio.TimeoutError:
            return NodeExecution(
                node.node_id,
                node.skill_id,
                NodeStatus.TIMED_OUT,
                tuple(sorted(allowed)),
                None,
                UsageRecord(),
                resources,
                started,
                datetime.now(timezone.utc).isoformat(),
                (time.perf_counter() - clock) * 1000,
                "node timeout",
            )
        except Exception as exc:
            return NodeExecution(
                node.node_id,
                node.skill_id,
                NodeStatus.FAILED,
                tuple(sorted(allowed)),
                None,
                UsageRecord(),
                resources,
                started,
                datetime.now(timezone.utc).isoformat(),
                (time.perf_counter() - clock) * 1000,
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            if reservation is not None and self.budget_manager is not None:
                await self.budget_manager.release(reservation)
