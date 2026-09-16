from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

from evofact.core.budget_models import BudgetRequest, CostStatus, UsageDetails
from evofact.core.models import SkillKind
from evofact.core.package_models import SkillPackage
from evofact.skills.package_adapter import package_to_skill_spec

from .generator import ChallengeGenerator


@dataclass(frozen=True)
class GenerationWorkflowResult:
    response: dict
    request: dict
    request_digest: str
    response_digest: str
    generator_package_digest: str


class GenerationWorkflow:
    def __init__(self, package: SkillPackage):
        if package.manifest.kind != SkillKind.WORKFLOW:
            raise ValueError("Generator must be a workflow Skill Package")
        self.package = package
        self.runtime_skill = package_to_skill_spec(package)
        self.generator = ChallengeGenerator()

    async def run(
        self,
        backend,
        samples,
        traces,
        attributions,
        *,
        config,
        episode_id,
        budget_manager=None,
    ):
        request = self.generator.build_request(
            samples,
            traces,
            attributions,
            config=config,
            episode_id=episode_id,
        )
        request["generator_package_digest"] = self.package.package_digest
        raw_instructions = self.package.file(
            self.package.manifest.entrypoints.instructions
        ).content.decode("utf-8")
        if raw_instructions.startswith("---\n"):
            _, marker, system = raw_instructions[4:].partition("\n---\n")
            if not marker:
                raise ValueError("Generator SKILL.md frontmatter is invalid")
        else:
            system = raw_instructions
        references = [
            {"path": path, "content": value}
            for path, value in sorted(self.runtime_skill.resources.items())
            if path.startswith("references/")
        ]
        template_path = self.package.manifest.entrypoints.template
        payload = dict(request)
        payload["workflow_references"] = references
        if template_path:
            template = self.package.file(template_path).content.decode("utf-8")
            payload["workflow_template"] = template
        reservation = None
        try:
            if budget_manager is not None:
                reservation = await budget_manager.reserve(
                    episode_id,
                    BudgetRequest(calls=1, tokens=4000, purpose="generator"),
                )
                async with budget_manager.concurrency(episode_id):
                    response = await asyncio.wait_for(
                        backend._call(system, payload),
                        budget_manager.limits.call_timeout_ms / 1000,
                    )
                usage = getattr(backend, "_last_usage_details", None) or UsageDetails(
                    calls=1,
                    cost_status=CostStatus.UNAVAILABLE,
                )
                await budget_manager.reconcile(reservation, usage)
                reservation = None
            else:
                response = await backend._call(system, payload)
        finally:
            if reservation is not None:
                await budget_manager.release(reservation)

        def encode(value):
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode()

        return GenerationWorkflowResult(
            response,
            request,
            hashlib.sha256(encode(request)).hexdigest(),
            hashlib.sha256(encode(response)).hexdigest(),
            self.package.package_digest,
        )
