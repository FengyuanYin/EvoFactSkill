"""Data models for complete, evolvable skill packages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from evofact.governance import (
    SKILL_FILE_SCHEMA_VERSION,
    SKILL_MANIFEST_SCHEMA_VERSION,
    SKILL_PACKAGE_PATCH_SCHEMA_VERSION,
    SKILL_PACKAGE_SCHEMA_VERSION,
)

from .models import ModelMixin, SkillKind, SkillScope, SkillStatus, Trigger

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def _validate_safe_relative_path(path: str) -> None:
    """Validate a portable path contained inside a skill package."""
    if not isinstance(path, str) or not path:
        raise ValueError("skill file path must not be empty")

    if path != path.strip():
        raise ValueError("skill file path must not contain surrounding whitespace")

    if path.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(path):
        raise ValueError("skill file path must be relative")

    if "\\" in path:
        raise ValueError("skill file path must use forward slashes")

    if "\x00" in path:
        raise ValueError("skill file path must not contain null bytes")

    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("skill file path contains an unsafe segment")


def _validate_sha256_digest(digest: str) -> None:
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError("digest must be a lowercase SHA-256 hexadecimal digest")


def _validate_contract_items(field_name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")

    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain empty values")


@dataclass(frozen=True)
class SkillFile(ModelMixin):
    """One immutable file stored inside a skill package."""

    path: str
    media_type: str
    content: bytes
    digest: str
    executable: bool = False
    schema_version: str = SKILL_FILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_safe_relative_path(self.path)

        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("media_type must not be empty")

        if not isinstance(self.content, bytes):
            raise ValueError("content must be bytes")

        _validate_sha256_digest(self.digest)

        actual_digest = hashlib.sha256(self.content).hexdigest()
        if self.digest != actual_digest:
            raise ValueError("digest does not match skill file content")

        if self.schema_version != SKILL_FILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported skill file schema version: {self.schema_version}")


@dataclass(frozen=True)
class SkillContract(ModelMixin):
    """Inputs, outputs and runtime capabilities declared by a skill."""

    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    requires_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    allow_root: bool = True
    allow_parallel: bool = True
    output_schema: str | None = None

    def __post_init__(self) -> None:
        _validate_contract_items("consumes", self.consumes)
        _validate_contract_items("produces", self.produces)
        _validate_contract_items(
            "requires_capabilities",
            self.requires_capabilities,
        )
        _validate_contract_items(
            "optional_capabilities",
            self.optional_capabilities,
        )

        if self.output_schema is not None and not self.output_schema.strip():
            raise ValueError("output_schema must not be empty when provided")


@dataclass(frozen=True)
class SkillEntrypoints(ModelMixin):
    instructions: str = "SKILL.md"
    template: str | None = None
    script: str | None = None
    output_schema: str | None = None
    tests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        paths = [self.instructions, self.template, self.script, self.output_schema, *self.tests]
        for path in paths:
            if path is not None:
                _validate_safe_relative_path(path)


@dataclass(frozen=True)
class SkillManifest(ModelMixin):
    name: str
    kind: SkillKind
    version: str
    scope: SkillScope
    triggers: tuple[Trigger, ...]
    contract: SkillContract
    entrypoints: SkillEntrypoints
    safety_level: Literal["text_only", "review_required", "executable"]
    schema_version: str = SKILL_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.name):
            raise ValueError(f"invalid skill name: {self.name}")
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", self.version):
            raise ValueError(f"invalid skill version: {self.version}")
        if self.schema_version != SKILL_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def compute_package_digest(manifest: SkillManifest, files: tuple[SkillFile, ...]) -> str:
    payload = {
        "manifest": _jsonable(manifest),
        "files": [
            {
                "path": item.path,
                "media_type": item.media_type,
                "digest": item.digest,
                "size": len(item.content),
                "executable": item.executable,
            }
            for item in files
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillPackage(ModelMixin):
    skill_id: str
    manifest: SkillManifest
    files: tuple[SkillFile, ...]
    status: SkillStatus
    parent_ids: tuple[str, ...] = ()
    package_digest: str = ""
    schema_version: str = SKILL_PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must not be empty")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("skill package file paths must be unique")
        if paths.count("SKILL.md") != 1:
            raise ValueError("skill package must contain exactly one SKILL.md")
        ordered = tuple(sorted(self.files, key=lambda item: item.path))
        if self.files != ordered:
            object.__setattr__(self, "files", ordered)
        expected = compute_package_digest(self.manifest, ordered)
        if self.package_digest and self.package_digest != expected:
            raise ValueError("package digest does not match manifest and files")
        object.__setattr__(self, "package_digest", expected)
        if self.schema_version != SKILL_PACKAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported package schema: {self.schema_version}")

    def file(self, path: str) -> SkillFile:
        for item in self.files:
            if item.path == path:
                return item
        raise KeyError(path)


class FileOperationKind(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


@dataclass(frozen=True)
class FileOperation(ModelMixin):
    operation: FileOperationKind
    path: str
    content: bytes | None = None
    destination: str | None = None
    expected_digest: str | None = None
    media_type: str | None = None
    executable: bool | None = None

    def __post_init__(self) -> None:
        _validate_safe_relative_path(self.path)
        if self.destination is not None:
            _validate_safe_relative_path(self.destination)
        if self.expected_digest is not None:
            _validate_sha256_digest(self.expected_digest)
        if self.operation == FileOperationKind.ADD:
            if (
                self.content is None
                or self.destination is not None
                or self.expected_digest is not None
            ):
                raise ValueError("add requires content and forbids destination/expected_digest")
        elif self.operation == FileOperationKind.UPDATE:
            if self.content is None or self.expected_digest is None or self.destination is not None:
                raise ValueError("update requires content and expected_digest")
        elif self.operation == FileOperationKind.DELETE:
            if (
                self.content is not None
                or self.expected_digest is None
                or self.destination is not None
            ):
                raise ValueError("delete requires expected_digest only")
        elif self.operation == FileOperationKind.RENAME:
            if self.content is not None or self.expected_digest is None or self.destination is None:
                raise ValueError("rename requires destination and expected_digest")
            if self.destination == self.path:
                raise ValueError("rename destination must differ from source")


@dataclass(frozen=True)
class ManifestPatch(ModelMixin):
    version: str | None = None
    scope: SkillScope | None = None
    triggers: tuple[Trigger, ...] | None = None
    contract: SkillContract | None = None
    entrypoints: SkillEntrypoints | None = None
    safety_level: str | None = None


@dataclass(frozen=True)
class SkillPackagePatch(ModelMixin):
    target_skill_id: str
    base_package_digest: str
    file_operations: tuple[FileOperation, ...]
    manifest_patch: ManifestPatch | None
    rationale: str
    source_trace_ids: tuple[str, ...] = ()
    source_audit_ids: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    schema_version: str = SKILL_PACKAGE_PATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.target_skill_id.strip() or not self.rationale.strip():
            raise ValueError("patch target and rationale must not be empty")
        _validate_sha256_digest(self.base_package_digest)
        if not self.file_operations and self.manifest_patch is None:
            raise ValueError("package patch must contain a change")
        if self.schema_version != SKILL_PACKAGE_PATCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported patch schema: {self.schema_version}")


@dataclass(frozen=True)
class PackageFinding(ModelMixin):
    code: str
    message: str
    path: str | None = None
    severity: Literal["error", "warning", "review_required"] = "error"


@dataclass(frozen=True)
class PackageValidationReport(ModelMixin):
    valid: bool
    findings: tuple[PackageFinding, ...] = ()
    requires_review: bool = False


@dataclass(frozen=True)
class PackageFileDiff(ModelMixin):
    operation: FileOperationKind
    path: str
    destination: str | None = None
    old_digest: str | None = None
    new_digest: str | None = None
    old_size: int | None = None
    new_size: int | None = None
    text_diff: str | None = None


@dataclass(frozen=True)
class PackageDiff(ModelMixin):
    before_digest: str
    after_digest: str
    manifest_changes: dict[str, tuple[Any, Any]]
    file_changes: tuple[PackageFileDiff, ...]


__all__ = [
    "FileOperation",
    "FileOperationKind",
    "ManifestPatch",
    "PackageDiff",
    "PackageFileDiff",
    "PackageFinding",
    "PackageValidationReport",
    "SkillContract",
    "SkillEntrypoints",
    "SkillFile",
    "SkillManifest",
    "SkillPackage",
    "SkillPackagePatch",
    "compute_package_digest",
]
