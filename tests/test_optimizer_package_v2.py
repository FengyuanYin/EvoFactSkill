from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from evofact.config import load_config
from evofact.core.budget_models import BudgetLimits, UsageDetails
from evofact.core.models import UsageRecord
from evofact.evolution.package_candidate import (
    build_package_candidate,
    package_candidate_to_proposal,
)
from evofact.evolution.package_optimizer import (
    PackageOptimizerAgent,
    legacy_instruction_edit_to_patch,
    parse_package_optimizer_response,
)
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.runtime.backend import BackendResult
from evofact.runtime.budget import BudgetManager
from evofact.runtime.mock_backend import MockBackend
from evofact.skills.package_loader import load_package

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_optimizer_can_change_complete_package():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    reference = package.file("references/rewriting_rules.md")
    patch = parse_package_optimizer_response(
        {
            "action": "edit",
            "target_skill_id": package.skill_id,
            "rationale": "improve grounded rewriting",
            "source_audit_ids": ["audit-1"],
            "file_operations": [
                {
                    "operation": "update",
                    "path": reference.path,
                    "content": "Use only facts present in the source evidence.\n",
                    "expected_digest": reference.digest,
                    "media_type": reference.media_type,
                }
            ],
        },
        [package],
    )
    candidate = build_package_candidate(package, patch)
    assert candidate.package.package_digest != package.package_digest
    assert candidate.package.file("SKILL.md").content == package.file("SKILL.md").content
    assert candidate.package.file(reference.path).digest != reference.digest
    assert patch.source_audit_ids == ("audit-1",)
    proposal = package_candidate_to_proposal(
        candidate,
        cluster_id="grounding",
        source_trace_ids=("trace-1",),
    )
    assert proposal.candidate_skills[0].package_digest == candidate.package.package_digest
    assert proposal.target_skill_ids == (package.skill_id,)


def test_legacy_edit_only_updates_instruction_entrypoint():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    patch = legacy_instruction_edit_to_patch(package, "New instructions.", rationale="legacy")
    assert len(patch.file_operations) == 1
    assert patch.file_operations[0].path == "SKILL.md"
    assert patch.file_operations[0].expected_digest == package.file("SKILL.md").digest


def test_optimizer_cannot_target_itself_or_frozen_governance():
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")
    with pytest.raises(ValueError, match="frozen"):
        parse_package_optimizer_response(
            {
                "action": "edit",
                "target_skill_id": optimizer.skill_id,
                "rationale": "recursive self improvement",
                "file_operations": [],
            },
            [optimizer],
        )


def test_script_patch_requires_human_review():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    patch = parse_package_optimizer_response(
        {
            "action": "edit",
            "target_skill_id": package.skill_id,
            "rationale": "add helper",
            "file_operations": [
                {
                    "operation": "add",
                    "path": "scripts/helper.py",
                    "content": "print('unsafe host execution must not occur')\n",
                    "media_type": "text/x-python",
                    "executable": True,
                }
            ],
        },
        [package],
    )
    assert {"script_change", "human_review_required"} <= set(patch.risk_flags)


def test_optimizer_cannot_embed_runtime_owned_label_contract_files():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    with pytest.raises(ValueError, match="Runtime-owned dataset label contract"):
        parse_package_optimizer_response(
            {
                "action": "edit",
                "target_skill_id": package.skill_id,
                "rationale": "override dataset labels",
                "file_operations": [
                    {
                        "operation": "add",
                        "path": "references/dataset-label-contract.json",
                        "content": '{"allowed_labels": ["yes", "no"]}',
                        "media_type": "application/json",
                    }
                ],
            },
            [package],
        )


def test_mismatched_base_digest_is_rejected():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    patch = parse_package_optimizer_response(
        {
            "action": "edit",
            "target_skill_id": package.skill_id,
            "rationale": "edit",
            "file_operations": [],
            "manifest_patch": {"version": "0.1.1"},
        },
        [package],
    )
    bad = replace(patch, base_package_digest=hashlib.sha256(b"wrong").hexdigest())
    with pytest.raises(ValueError, match="base package digest"):
        build_package_candidate(package, bad)


def test_manifest_version_edit_updates_package_version_fields():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    patch = parse_package_optimizer_response(
        {
            "action": "edit",
            "target_skill_id": package.skill_id,
            "rationale": "release a revised specialist",
            "file_operations": [],
            "manifest_patch": {"version": "0.2.1"},
        },
        [package],
    )
    candidate = build_package_candidate(package, patch)
    assert candidate.package.manifest.version == "0.2.1"
    assert b"version: 0.2.1" in candidate.package.file("SKILL.md").content
    assert b'"version":"0.2.1"' in candidate.package.file("metadata.json").content
    assert candidate.validation.valid


def test_unchanged_manifest_name_and_kind_are_ignored():
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    patch = parse_package_optimizer_response(
        {
            "action": "edit",
            "target_skill_id": package.skill_id,
            "rationale": "release a revised specialist",
            "file_operations": [],
            "manifest_patch": {
                "name": package.manifest.name,
                "kind": package.manifest.kind.value,
                "version": "0.2.1",
            },
        },
        [package],
    )
    assert build_package_candidate(package, patch).package.manifest.version == "0.2.1"


@pytest.mark.parametrize("field,value", [("name", "other_skill"), ("kind", "judge")])
def test_manifest_identity_changes_remain_forbidden(field, value):
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    with pytest.raises(ValueError, match=f"{field} cannot be changed"):
        parse_package_optimizer_response(
            {
                "action": "edit",
                "target_skill_id": package.skill_id,
                "rationale": "invalid identity change",
                "file_operations": [],
                "manifest_patch": {field: value, "version": "0.2.1"},
            },
            [package],
        )


def test_invalid_optimizer_proposal_is_skipped_without_failing_batch(caplog):
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")

    class InvalidBackend(MockBackend):
        async def optimize_package(self, context, optimizer_skill):
            del context, optimizer_skill
            return BackendResult(
                {
                    "action": "edit",
                    "target_skill_id": target.skill_id,
                    "rationale": "invalid identity change",
                    "file_operations": [],
                    "manifest_patch": {"name": "other_skill"},
                }
            )

    agent = PackageOptimizerAgent(InvalidBackend(), optimizer)
    assert asyncio.run(agent.propose(target, [target, optimizer])) is None
    assert "Skipping invalid optimizer proposal" in caplog.text


def test_optimizer_proposals_for_same_target_use_separate_sample_budgets():
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")

    class NoChangeBackend(MockBackend):
        async def optimize_package(self, context, optimizer_skill):
            del context, optimizer_skill
            return BackendResult({"action": "no_change"}, usage_details=UsageDetails(calls=1))

    budget = BudgetManager(BudgetLimits(max_calls_per_sample=1))
    agent = PackageOptimizerAgent(NoChangeBackend(), optimizer, budget_manager=budget)

    async def propose_twice():
        assert await agent.propose(target, [target, optimizer]) is None
        assert await agent.propose(target, [target, optimizer]) is None

    asyncio.run(propose_twice())
    ledgers = [
        (sample_id, usage)
        for sample_id, usage in budget.export_state()["samples"].items()
        if sample_id.startswith(f"package-optimizer:{target.skill_id}:")
    ]
    assert len(ledgers) == 2
    assert all(usage["calls_used"] == 1 for _, usage in ledgers)
    assert budget.snapshot().calls_used == 2


def test_package_optimizer_agent_uses_meta_prompt_but_does_not_evolve_itself():
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")
    agent = PackageOptimizerAgent(MockBackend(), optimizer)
    assert asyncio.run(agent.propose(target, [target, optimizer])) is None
    with pytest.raises(ValueError, match="itself"):
        asyncio.run(agent.propose(optimizer, [target, optimizer]))


def test_ordinary_llm_evolution_emits_exact_package_candidate(monkeypatch):
    config = load_config(PROJECT_ROOT / "configs" / "dry_run.yaml")
    config = replace(config, evolution=replace(config.evolution, proposer="llm"))
    runner = ExperimentRunner(config, PROJECT_ROOT)

    class EditingBackend(MockBackend):
        async def optimize_package(self, context, optimizer_skill):
            del optimizer_skill
            target_id = context["target"]["skill_id"]
            target = next(package for package in runner.packages if package.skill_id == target_id)
            source = target.file(target.manifest.entrypoints.instructions)
            content = source.content.decode("utf-8") + "\nUse an auditable package-level edit.\n"
            return BackendResult(
                {
                    "action": "edit",
                    "target_skill_id": target_id,
                    "rationale": "exercise the production package path",
                    "file_operations": [
                        {
                            "operation": "update",
                            "path": source.path,
                            "content": content,
                            "expected_digest": source.digest,
                            "media_type": source.media_type,
                        }
                    ],
                },
                UsageRecord(calls=1),
            )

    backend = EditingBackend()
    monkeypatch.setattr(runner, "_backend", lambda: backend)
    monkeypatch.setattr(
        "evofact.experiments.runner.cluster_reports",
        lambda reports: {"forced-package-audit": reports[:1]},
    )
    outcome = asyncio.run(runner.evolve_once(fixture_samples()))
    assert len(outcome["proposals"]) == 1
    proposal = outcome["proposals"][0]
    candidate = outcome["package_candidates"][proposal.proposal_id]
    assert candidate.package.package_digest == proposal.candidate_skills[0].package_digest
    assert candidate.package.file("SKILL.md").content.endswith(
        b"Use an auditable package-level edit.\n"
    )
