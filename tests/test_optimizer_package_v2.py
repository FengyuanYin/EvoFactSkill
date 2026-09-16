from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from evofact.evolution.package_candidate import build_package_candidate
from evofact.evolution.package_optimizer import (
    PackageOptimizerAgent,
    legacy_instruction_edit_to_patch,
    parse_package_optimizer_response,
)
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


def test_package_optimizer_agent_uses_meta_prompt_but_does_not_evolve_itself():
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "generation_agent")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")
    agent = PackageOptimizerAgent(MockBackend(), optimizer)
    assert asyncio.run(agent.propose(target, [target, optimizer])) is None
    with pytest.raises(ValueError, match="itself"):
        asyncio.run(agent.propose(optimizer, [target, optimizer]))
