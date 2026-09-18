from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evofact.core.models import SkillKind, SkillSpec, SkillStatus
from evofact.core.package_models import SpecialistReportContract
from evofact.evolution.package_candidate import build_package_addition_candidate
from evofact.evolution.package_optimizer import parse_package_optimizer_response
from evofact.governance.package_policy import validate_package
from evofact.runtime.openai_backend import _specialist_json_contract
from evofact.skills.package_adapter import package_to_skill_spec, skill_spec_to_package
from evofact.skills.package_loader import load_package
from evofact.skills.repository import SkillRepository


def _addition_response(name: str = "geospatial_context") -> dict:
    contract_path = "schemas/specialist_report.json"
    report_contract = {
        "schema_version": "specialist_report_contract_v1",
        "report_type": name,
        "finding_types": ["location_match", "location_conflict"],
        "guidance": "Compare only the locations stated in claims and supplied evidence.",
    }
    metadata = {
        "name": name,
        "kind": "specialist",
        "version": "0.1.0",
        "safety_level": "text_only",
        "contract": {
            "consumes": ["atomic_claims"],
            "produces": ["specialist_report"],
            "requires_capabilities": ["llm"],
            "optional_capabilities": [],
            "allow_root": False,
            "allow_parallel": True,
            "output_schema": "specialist_report_v2",
        },
        "entrypoints": {
            "instructions": "SKILL.md",
            "template": None,
            "script": None,
            "output_schema": contract_path,
            "tests": [],
        },
    }
    return {
        "action": "add",
        "target_skill_id": None,
        "rationale": "A recurring location error needs a dedicated analysis role.",
        "file_operations": [],
        "manifest_patch": None,
        "new_package": {
            "skill_id": None,
            "manifest": metadata,
            "files": [
                {
                    "path": "SKILL.md",
                    "content": (
                        f"---\nname: {name}\nkind: specialist\nversion: 0.1.0\n---\n"
                        "Analyze geographic consistency without issuing a verdict.\n"
                    ),
                    "media_type": "text/markdown",
                    "executable": False,
                },
                {
                    "path": "metadata.json",
                    "content": json.dumps(metadata),
                    "media_type": "application/json",
                    "executable": False,
                },
                {
                    "path": contract_path,
                    "content": json.dumps(report_contract),
                    "media_type": "application/json",
                    "executable": False,
                },
            ],
        },
        "source_trace_ids": ["trace-1"],
        "source_audit_ids": [],
        "risk_flags": [],
    }


def test_runtime_prefers_package_owned_contract_for_unknown_specialist() -> None:
    skill = SkillSpec(
        "geo-id",
        "geospatial_context",
        SkillKind.SPECIALIST,
        "0.1.0",
        SkillStatus.ACTIVE,
        "Analyze locations.",
        report_contract=SpecialistReportContract(
            "geospatial_context",
            ("location_match", "location_conflict"),
            "Compare locations only.",
        ),
    )

    prompt = _specialist_json_contract(skill)

    assert "location_match | location_conflict" in prompt
    assert "Compare locations only" in prompt
    assert "domain_observation" not in prompt


def test_seed_packages_load_their_report_contract() -> None:
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    skill = package_to_skill_spec(package)

    assert package.manifest.entrypoints.output_schema == "schemas/specialist_report.json"
    assert package.file("schemas/specialist_report.json")
    assert validate_package(package).valid
    assert skill.report_contract.report_type == "source_credibility"
    assert "source_missing" in skill.report_contract.finding_types


def test_invalid_final_verdict_contract_fails_package_validation() -> None:
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    schema = package.file("schemas/specialist_report.json")
    bad_content = json.dumps(
        {
            "schema_version": "specialist_report_contract_v1",
            "report_type": "source_credibility",
            "finding_types": ["fake"],
            "guidance": "Return a final verdict.",
        }
    ).encode()
    bad_file = replace(
        schema,
        content=bad_content,
        digest=__import__("hashlib").sha256(bad_content).hexdigest(),
    )
    bad_package = replace(
        package,
        files=tuple(bad_file if item.path == schema.path else item for item in package.files),
        package_digest="",
    )

    report = validate_package(bad_package)

    assert not report.valid
    assert any(item.code == "report_contract_invalid" for item in report.findings)


def test_optimizer_add_builds_complete_new_package_and_repository_round_trips(tmp_path) -> None:
    existing = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    addition = parse_package_optimizer_response(_addition_response(), [existing])
    candidate = build_package_addition_candidate(addition)

    assert candidate.base is None
    assert candidate.patch is None
    assert candidate.package.manifest.kind == SkillKind.SPECIALIST
    assert candidate.package.status == SkillStatus.CANDIDATE
    assert candidate.validation.valid

    repository = SkillRepository(tmp_path / "bank")
    repository.promote_package(candidate.package, run_id="add-geospatial")
    restored = repository.active_packages()["geospatial_context"]

    assert restored.package_digest == candidate.package.package_digest
    assert restored.file("schemas/specialist_report.json").content


def test_optimizer_add_rejects_existing_name_and_non_specialist() -> None:
    existing = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    with pytest.raises(ValueError, match="already exists"):
        parse_package_optimizer_response(_addition_response("source_credibility"), [existing])

    response = _addition_response()
    response["new_package"]["manifest"]["kind"] = "router"
    with pytest.raises(ValueError, match="only add specialist"):
        parse_package_optimizer_response(response, [existing])


def test_legacy_add_skill_can_be_converted_to_complete_package() -> None:
    from evofact.core.models import OptimizerAction, SkillOptimizerDecision
    from evofact.evolution.optimizer import decision_to_proposal

    proposal = decision_to_proposal(
        SkillOptimizerDecision(
            OptimizerAction.ADD,
            "Add a specialist for a recurring domain error.",
            0.8,
            skill_name="geospatial_context",
            skill_kind=SkillKind.SPECIALIST,
            instructions="Analyze location consistency without issuing a final verdict.",
        ),
        [],
        cluster_id="geo-errors",
        source_trace_ids=("trace-1",),
    )

    package = skill_spec_to_package(proposal.candidate_skills[0])
    restored = package_to_skill_spec(package)

    assert validate_package(package).valid
    assert package.file("metadata.json")
    assert package.file("schemas/specialist_report.json")
    assert restored.report_contract.report_type == "geospatial_context"
