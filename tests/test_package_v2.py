from __future__ import annotations

import json
from pathlib import Path

from evofact.core.package_models import FileOperation, FileOperationKind, SkillPackagePatch
from evofact.skills.package_diff import diff_packages
from evofact.skills.package_loader import load_package
from evofact.skills.package_patch import apply_package_patch
from evofact.skills.package_serializer import dumps_package, loads_package
from evofact.skills.repository import SkillRepository

ROOT = Path(__file__).resolve().parents[1]


def test_generator_package_round_trip_and_digest() -> None:
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    restored = loads_package(dumps_package(package))

    assert restored == package
    assert restored.package_digest == package.package_digest
    assert {item.path for item in restored.files} >= {
        "SKILL.md",
        "metadata.json",
        "references/rewriting_rules.md",
        "templates/request.json",
        "tests/cases.json",
    }


def test_patch_preserves_unmentioned_files_and_diff_is_precise() -> None:
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    reference = package.file("references/rewriting_rules.md")
    patch = SkillPackagePatch(
        package.skill_id,
        package.package_digest,
        (
            FileOperation(
                FileOperationKind.UPDATE,
                reference.path,
                reference.content + b"\n- Keep numbers auditable.\n",
                expected_digest=reference.digest,
            ),
        ),
        None,
        "clarify numeric rewriting",
    )

    candidate = apply_package_patch(package, patch)
    difference = diff_packages(package, candidate)

    assert candidate.package_digest != package.package_digest
    assert len(difference.file_changes) == 1
    assert difference.file_changes[0].path == reference.path
    assert candidate.file("SKILL.md").content == package.file("SKILL.md").content


def test_package_repository_commit_and_rollback(tmp_path) -> None:
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    reference = package.file("references/rewriting_rules.md")
    candidate = apply_package_patch(
        package,
        SkillPackagePatch(
            package.skill_id,
            package.package_digest,
            (
                FileOperation(
                    FileOperationKind.UPDATE,
                    reference.path,
                    reference.content + b"\nrevision\n",
                    expected_digest=reference.digest,
                ),
            ),
            None,
            "test revision",
        ),
    )
    repository = SkillRepository(tmp_path)
    repository.promote_package(package, run_id="initial")
    repository.promote_package(candidate, run_id="candidate")
    assert (
        repository.active_packages()[package.manifest.name].package_digest
        == candidate.package_digest
    )

    repository.rollback_package(package.manifest.name, package.package_digest, run_id="rollback")
    restored = repository.active_packages()[package.manifest.name]
    assert restored.package_digest == package.package_digest
    assert restored.files == package.files


def test_package_bank_can_be_selected_by_run_digest_or_lock(tmp_path) -> None:
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    repository = SkillRepository(tmp_path / "store")
    repository.commit_package_bank(
        [package],
        run_id="training-run-1",
        expected_active={},
        audit={"provenance": {"config_digest": "config-a", "manifest_id": "manifest-a"}},
    )
    info = repository.package_bank_info("training-run-1")
    assert info["packages"] == {package.manifest.name: package.package_digest}
    assert info["provenance"]["config_digest"] == "config-a"
    assert repository.resolve_package_bank(info["bank_digest"]) == {package.manifest.name: package}

    lock = tmp_path / "bank-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "skill_bank_lock_v1",
                "packages": info["packages"],
            }
        ),
        encoding="utf-8",
    )
    assert repository.resolve_package_bank(str(lock)) == {package.manifest.name: package}
