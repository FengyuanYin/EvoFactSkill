from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import replace

from evofact.core.frontmatter import parse_frontmatter, render_frontmatter
from evofact.core.package_models import (
    FileOperationKind,
    SkillFile,
    SkillPackage,
    SkillPackagePatch,
)
from evofact.governance.package_policy import require_valid_package


def _with_content(file: SkillFile, content: bytes) -> SkillFile:
    return replace(file, content=content, digest=hashlib.sha256(content).hexdigest())


def _sync_manifest_version(files: dict[str, SkillFile], version: str) -> None:
    """Keep the two descriptive version fields aligned with the manifest."""
    skill_file = files.get("SKILL.md")
    if skill_file is not None:
        raw = skill_file.content.decode("utf-8")
        frontmatter, body = parse_frontmatter(raw)
        if "version" in frontmatter and frontmatter["version"] != version:
            frontmatter["version"] = version
            files["SKILL.md"] = _with_content(
                skill_file, render_frontmatter(frontmatter, body).encode("utf-8")
            )

    metadata = files.get("metadata.json")
    if metadata is not None:
        raw = json.loads(metadata.content.decode("utf-8"))
        if isinstance(raw, dict) and "version" in raw and raw["version"] != version:
            raw["version"] = version
            files["metadata.json"] = _with_content(
                metadata, json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )


def apply_package_patch(base: SkillPackage, patch: SkillPackagePatch) -> SkillPackage:
    if patch.target_skill_id != base.skill_id:
        raise ValueError("patch targets another skill")
    if patch.base_package_digest != base.package_digest:
        raise ValueError("base package digest mismatch")
    files = {item.path: item for item in base.files}
    for operation in patch.file_operations:
        current = files.get(operation.path)
        if operation.operation == FileOperationKind.ADD:
            if current is not None:
                raise ValueError(f"add target already exists: {operation.path}")
            media_type = (
                operation.media_type
                or mimetypes.guess_type(operation.path)[0]
                or "application/octet-stream"
            )
            content = operation.content or b""
            files[operation.path] = SkillFile(
                operation.path,
                media_type,
                content,
                hashlib.sha256(content).hexdigest(),
                bool(operation.executable),
            )
            continue
        if current is None:
            raise ValueError(f"patch source does not exist: {operation.path}")
        if current.digest != operation.expected_digest:
            raise ValueError(f"expected digest mismatch: {operation.path}")
        if operation.operation == FileOperationKind.DELETE:
            del files[operation.path]
        elif operation.operation == FileOperationKind.UPDATE:
            content = operation.content or b""
            files[operation.path] = SkillFile(
                operation.path,
                operation.media_type or current.media_type,
                content,
                hashlib.sha256(content).hexdigest(),
                current.executable if operation.executable is None else operation.executable,
            )
        elif operation.operation == FileOperationKind.RENAME:
            destination = operation.destination or ""
            if destination in files:
                raise ValueError(f"rename destination already exists: {destination}")
            del files[operation.path]
            files[destination] = replace(current, path=destination)

    manifest = base.manifest
    if patch.manifest_patch is not None:
        changes = {
            key: getattr(patch.manifest_patch, key)
            for key in (
                "version",
                "scope",
                "triggers",
                "contract",
                "entrypoints",
                "safety_level",
            )
            if getattr(patch.manifest_patch, key) is not None
        }
        manifest = replace(manifest, **changes)
    if manifest.version != base.manifest.version:
        _sync_manifest_version(files, manifest.version)
    candidate = SkillPackage(
        skill_id=base.skill_id,
        manifest=manifest,
        files=tuple(files.values()),
        status=base.status,
        parent_ids=tuple(dict.fromkeys((*base.parent_ids, base.skill_id))),
    )
    require_valid_package(candidate)
    return candidate
