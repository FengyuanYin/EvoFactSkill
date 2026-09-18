from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

from evofact.core.budget_models import BudgetRequest, UsageDetails
from evofact.core.models import SkillKind
from evofact.core.package_models import SkillPackage
from evofact.evolution.firewall import validate_generation_request
from evofact.runtime.backend import call_json_with_usage
from evofact.skills.package_adapter import package_to_skill_spec

from .generator import ChallengeGenerator


@dataclass(frozen=True)
class GenerationWorkflowResult:
    response: dict
    request: dict
    request_digest: str
    response_digest: str
    generator_package_digest: str
    usage: UsageDetails


class GenerationWorkflow:
    def __init__(self, package: SkillPackage):
        if package.manifest.kind != SkillKind.WORKFLOW:
            raise ValueError("Generator must be a workflow Skill Package")
        self.package = package
        self.runtime_skill = package_to_skill_spec(package)
        self.generator = ChallengeGenerator()
        strategy_path = "references/strategies.json"
        try:
            raw_strategies = json.loads(package.file(strategy_path).content.decode("utf-8"))
        except KeyError as exc:
            raise ValueError("Generator Package must declare references/strategies.json") from exc
        strategies = raw_strategies.get("strategies")
        if (
            not isinstance(strategies, list)
            or not strategies
            or not all(isinstance(item, str) and item.strip() for item in strategies)
            or len(set(strategies)) != len(strategies)
        ):
            raise ValueError("Generator Package strategy registry is invalid")
        self.strategies = frozenset(strategies)

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
        label_contract_registry=None,
        forbidden=(),
    ):
        request = self.generator.build_request(
            samples,
            traces,
            attributions,
            config=config,
            episode_id=episode_id,
            label_contract_registry=label_contract_registry,
        )
        request["generator_package_digest"] = self.package.package_digest
        validate_generation_request(request, forbidden)
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
                async with budget_manager.concurrency(episode_id):
                    reservation = await budget_manager.reserve(
                        episode_id,
                        BudgetRequest(calls=1, tokens=4000, purpose="generator"),
                    )
                    response, usage = await asyncio.wait_for(
                        call_json_with_usage(backend, system, payload),
                        budget_manager.limits.call_timeout_ms / 1000,
                    )
                await budget_manager.reconcile(reservation, usage)
                reservation = None
            else:
                response, usage = await call_json_with_usage(backend, system, payload)
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
            usage,
        )
