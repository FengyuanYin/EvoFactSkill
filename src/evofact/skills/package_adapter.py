from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evofact.core.frontmatter import normalize_newlines, parse_frontmatter, render_frontmatter
from evofact.core.models import SkillSpec, SkillStatus
from evofact.core.package_models import (
    SkillContract,
    SkillEntrypoints,
    SkillFile,
    SkillManifest,
    SkillPackage,
)
from evofact.core.report_contract import load_specialist_report_contract

from .package_loader import load_package


def legacy_directory_to_package(
    directory: str | Path,
    *,
    status: SkillStatus = SkillStatus.ACTIVE,
) -> SkillPackage:
    return load_package(directory, status=status)


def package_to_skill_spec(package: SkillPackage) -> SkillSpec:
    instructions = package.file(package.manifest.entrypoints.instructions).content.decode("utf-8")
    _, instructions = parse_frontmatter(instructions, required=False)
    resources = {
        item.path: item.content.decode("utf-8")
        for item in package.files
        if item.path != package.manifest.entrypoints.instructions
        and (
            item.media_type.startswith("text/")
            or item.media_type in {"application/json", "application/yaml"}
        )
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
        report_contract=load_specialist_report_contract(package),
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


def update_package_from_skill(base: SkillPackage, skill: SkillSpec) -> SkillPackage:
    """Project a legacy SkillSpec edit onto a Package without dropping untouched files."""
    if base.manifest.name != skill.name:
        raise ValueError("SkillSpec does not match the base Package")
    contract = (
        skill.contract if isinstance(skill.contract, SkillContract) else base.manifest.contract
    )
    entrypoints = (
        skill.entrypoints
        if isinstance(skill.entrypoints, SkillEntrypoints)
        else base.manifest.entrypoints
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
    files = {item.path: item for item in base.files}
    instruction_file = files[entrypoints.instructions]
    raw_instructions = normalize_newlines(instruction_file.content.decode("utf-8"))
    if raw_instructions.startswith("---\n"):
        frontmatter, _ = parse_frontmatter(raw_instructions)
        retained = [
            (key, value)
            for key, value in frontmatter.items()
            if key not in {"name", "kind", "version"}
        ]
        header = {
            "name": skill.name,
            "kind": skill.kind.value,
            "version": skill.version,
            **dict(retained),
        }
        instruction_content = render_frontmatter(header, skill.instructions).encode("utf-8")
    else:
        instruction_content = (skill.instructions.strip() + "\n").encode("utf-8")
    files[entrypoints.instructions] = SkillFile(
        instruction_file.path,
        instruction_file.media_type,
        instruction_content,
        hashlib.sha256(instruction_content).hexdigest(),
        instruction_file.executable,
    )
    for path, value in skill.resources.items():
        if path == "metadata.json":
            continue
        content = value.encode("utf-8")
        old = files.get(path)
        files[path] = SkillFile(
            path,
            old.media_type if old else ("text/x-python" if path.endswith(".py") else "text/plain"),
            content,
            hashlib.sha256(content).hexdigest(),
            bool(old.executable if old else path == entrypoints.script),
        )
    metadata = files.get("metadata.json")
    if metadata is not None:
        raw = json.loads(metadata.content.decode("utf-8"))
        raw.update(
            {
                "name": manifest.name,
                "kind": manifest.kind.value,
                "version": manifest.version,
                "scope": manifest.scope.model_dump(),
                "triggers": [item.model_dump() for item in manifest.triggers],
                "contract": manifest.contract.model_dump(),
                "entrypoints": manifest.entrypoints.model_dump(),
                "safety_level": manifest.safety_level,
            }
        )
        content = json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        files["metadata.json"] = SkillFile(
            "metadata.json",
            metadata.media_type,
            content,
            hashlib.sha256(content).hexdigest(),
            metadata.executable,
        )
    return SkillPackage(
        skill.skill_id,
        manifest,
        tuple(sorted(files.values(), key=lambda item: item.path)),
        skill.status,
        skill.parent_ids,
    )


def package_bank_digest(packages) -> str:
    payload = "\n".join(
        f"{package.manifest.name}:{package.package_digest}"
        for package in sorted(packages, key=lambda item: item.manifest.name)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_skill_bank_to_packages(
    packages: list[SkillPackage] | tuple[SkillPackage, ...], skills: list[SkillSpec]
) -> list[SkillPackage]:
    by_name = {package.manifest.name: package for package in packages}
    projected = []
    for skill in skills:
        base = by_name.get(skill.name)
        projected.append(
            skill_spec_to_package(skill)
            if base is None
            else (
                base
                if package_to_skill_spec(base) == skill
                else update_package_from_skill(base, skill)
            )
        )
    return sorted(projected, key=lambda item: item.manifest.name)
