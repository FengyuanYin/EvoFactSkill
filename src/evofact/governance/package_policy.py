from __future__ import annotations

import json
import re
from dataclasses import dataclass

from evofact.core.package_models import (
    PackageFinding,
    PackageValidationReport,
    SkillPackage,
    _validate_safe_relative_path,
)


@dataclass(frozen=True)
class PackageLimits:
    max_files: int = 128
    max_file_bytes: int = 1_000_000
    max_package_bytes: int = 8_000_000
    allowed_media_prefixes: tuple[str, ...] = (
        "text/",
        "application/json",
        "application/octet-stream",
        "image/",
    )


TEXT_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".jinja", ".j2"}


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    head, marker, _ = text[4:].partition("\n---\n")
    if not marker:
        return {}
    result: dict[str, str] = {}
    for line in head.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip().strip("\"'")
    return result


def validate_package(
    package: SkillPackage,
    *,
    limits: PackageLimits = PackageLimits(),
) -> PackageValidationReport:
    findings: list[PackageFinding] = []
    files = {item.path: item for item in package.files}
    if len(files) > limits.max_files:
        findings.append(PackageFinding("too_many_files", "package exceeds file count limit"))
    total = sum(len(item.content) for item in package.files)
    if total > limits.max_package_bytes:
        findings.append(PackageFinding("package_too_large", "package exceeds total size limit"))
    for item in package.files:
        try:
            _validate_safe_relative_path(item.path)
        except ValueError as exc:
            findings.append(PackageFinding("unsafe_path", str(exc), item.path))
        if len(item.content) > limits.max_file_bytes:
            findings.append(PackageFinding("file_too_large", "file exceeds size limit", item.path))
        if not any(
            item.media_type == allowed or item.media_type.startswith(allowed)
            for allowed in limits.allowed_media_prefixes
        ):
            findings.append(PackageFinding("media_type", "unsupported media type", item.path))
        extension = "." + item.path.rsplit(".", 1)[-1].lower() if "." in item.path else ""
        if extension in TEXT_EXTENSIONS:
            try:
                item.content.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                findings.append(
                    PackageFinding("encoding", "text file is not valid UTF-8", item.path)
                )

    entrypoints = package.manifest.entrypoints
    required = [entrypoints.instructions]
    optional = [
        entrypoints.template,
        entrypoints.script,
        entrypoints.output_schema,
        *entrypoints.tests,
    ]
    for path in required:
        if path not in files:
            findings.append(
                PackageFinding("missing_entrypoint", "required entrypoint is missing", path)
            )
    for path in optional:
        if path is not None and path not in files:
            findings.append(
                PackageFinding("dangling_entrypoint", "entrypoint target is missing", path)
            )
    declared_scripts = {entrypoints.script} if entrypoints.script else set()
    for item in package.files:
        if (
            item.path.startswith("scripts/")
            and item.executable
            and item.path not in declared_scripts
        ):
            findings.append(
                PackageFinding("undeclared_script", "executable script is not declared", item.path)
            )

    skill_file = files.get("SKILL.md")
    if skill_file:
        try:
            frontmatter = _frontmatter(skill_file.content.decode("utf-8"))
            expected = {
                "name": package.manifest.name,
                "kind": package.manifest.kind.value,
                "version": package.manifest.version,
            }
            for key, value in expected.items():
                if frontmatter.get(key) and frontmatter[key] != value:
                    findings.append(
                        PackageFinding(
                            "frontmatter_mismatch", f"{key} conflicts with manifest", "SKILL.md"
                        )
                    )
        except UnicodeDecodeError:
            pass

    metadata = files.get("metadata.json")
    if metadata:
        try:
            raw = json.loads(metadata.content.decode("utf-8"))
            for key, value in (
                ("name", package.manifest.name),
                ("kind", package.manifest.kind.value),
                ("version", package.manifest.version),
            ):
                if key in raw and raw[key] != value:
                    findings.append(
                        PackageFinding(
                            "metadata_mismatch", f"{key} conflicts with manifest", "metadata.json"
                        )
                    )
        except (UnicodeDecodeError, json.JSONDecodeError):
            findings.append(
                PackageFinding(
                    "metadata_invalid", "metadata.json must be valid UTF-8 JSON", "metadata.json"
                )
            )

    for collection_name, values in (
        ("consumes", package.manifest.contract.consumes),
        ("produces", package.manifest.contract.produces),
        ("requires_capabilities", package.manifest.contract.requires_capabilities),
        ("optional_capabilities", package.manifest.contract.optional_capabilities),
    ):
        if len(values) != len(set(values)):
            findings.append(
                PackageFinding("contract_duplicate", f"duplicate value in {collection_name}")
            )
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", value) for value in values):
            findings.append(PackageFinding("contract_value", f"invalid value in {collection_name}"))

    requires_review = package.manifest.safety_level == "review_required" or any(
        item.severity == "review_required" for item in findings
    )
    return PackageValidationReport(
        valid=not any(item.severity == "error" for item in findings),
        findings=tuple(findings),
        requires_review=requires_review,
    )


def require_valid_package(
    package: SkillPackage, *, limits: PackageLimits = PackageLimits()
) -> None:
    report = validate_package(package, limits=limits)
    if not report.valid:
        details = "; ".join(f"{item.code}: {item.message}" for item in report.findings)
        raise ValueError(f"invalid skill package: {details}")
