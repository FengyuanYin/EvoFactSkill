from __future__ import annotations

import hashlib
from pathlib import Path

from evofact.core.models import SkillSpec, SkillStatus
from evofact.core.package_models import (
    SkillContract,
    SkillEntrypoints,
    SkillFile,
    SkillManifest,
    SkillPackage,
)

from .package_loader import load_package


def legacy_directory_to_package(
    directory: str | Path,
    *,
    status: SkillStatus = SkillStatus.ACTIVE,
) -> SkillPackage:
    return load_package(directory, status=status)


def package_to_skill_spec(package: SkillPackage) -> SkillSpec:
    instructions = package.file(package.manifest.entrypoints.instructions).content.decode("utf-8")
    if instructions.startswith("---\n"):
        _, marker, body = instructions[4:].partition("\n---\n")
        if marker:
            instructions = body.strip()
    resources = {
        item.path: item.content.decode("utf-8")
        for item in package.files
        if item.path != package.manifest.entrypoints.instructions
        and item.media_type.startswith("text/")
    }
    return SkillSpec(
        skill_id=package.skill_id,
        name=package.manifest.name,
        kind=package.manifest.kind,
        version=package.manifest.version,
        status=package.status,
        instructions=instructions,
        resources=resources,
        scope=package.manifest.scope,
        triggers=package.manifest.triggers,
        parent_ids=package.parent_ids,
        safety_level=package.manifest.safety_level,
        package_digest=package.package_digest,
        contract=package.manifest.contract,
        entrypoints=package.manifest.entrypoints,
    )


def skill_spec_to_package(skill: SkillSpec) -> SkillPackage:
    contract = skill.contract if isinstance(skill.contract, SkillContract) else SkillContract()
    entrypoints = (
        skill.entrypoints if isinstance(skill.entrypoints, SkillEntrypoints) else SkillEntrypoints()
    )
    header = f"---\nname: {skill.name}\nkind: {skill.kind.value}\nversion: {skill.version}\n---\n"
    content = (header + skill.instructions.strip() + "\n").encode("utf-8")
    files = [
        SkillFile(
            "SKILL.md",
            "text/markdown",
            content,
            hashlib.sha256(content).hexdigest(),
        )
    ]
    for path, value in sorted(skill.resources.items()):
        raw = value.encode("utf-8")
        media_type = "text/x-python" if path.endswith(".py") else "text/plain"
        files.append(
            SkillFile(
                path,
                media_type,
                raw,
                hashlib.sha256(raw).hexdigest(),
                path == entrypoints.script,
            )
        )
    manifest = SkillManifest(
        skill.name,
        skill.kind,
        skill.version,
        skill.scope,
        skill.triggers,
        contract,
        entrypoints,
        skill.safety_level,
    )
    return SkillPackage(
        skill.skill_id,
        manifest,
        tuple(files),
        skill.status,
        skill.parent_ids,
    )
