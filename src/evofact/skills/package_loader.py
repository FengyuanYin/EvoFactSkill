from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path

from evofact.core.models import SkillKind, SkillScope, SkillStatus, Trigger
from evofact.core.package_models import (
    SkillContract,
    SkillEntrypoints,
    SkillFile,
    SkillManifest,
    SkillPackage,
)
from evofact.governance.package_policy import require_valid_package

SUPPORTED_ROOT_FILES = {"SKILL.md", "metadata.json"}
SUPPORTED_DIRECTORIES = {"references", "templates", "assets", "schemas", "scripts", "tests"}


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    head, marker, body = text[4:].partition("\n---\n")
    if not marker:
        raise ValueError("unterminated frontmatter")
    metadata: dict[str, str] = {}
    for line in head.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body.strip()


def _media_type(path: str) -> str:
    explicit = {
        ".md": "text/markdown",
        ".py": "text/x-python",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".jinja": "text/plain",
        ".j2": "text/plain",
    }
    suffix = Path(path).suffix.lower()
    return explicit.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def _read_files(directory: Path) -> tuple[SkillFile, ...]:
    paths: list[Path] = []
    for name in SUPPORTED_ROOT_FILES:
        candidate = directory / name
        if candidate.is_file():
            paths.append(candidate)
    for name in SUPPORTED_DIRECTORIES:
        root = directory / name
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    result = []
    for path in sorted(paths, key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        content = path.read_bytes()
        result.append(
            SkillFile(
                path=relative,
                media_type=_media_type(relative),
                content=content,
                digest=hashlib.sha256(content).hexdigest(),
                executable=relative.startswith("scripts/") and path.suffix == ".py",
            )
        )
    return tuple(result)


def load_package(
    directory: str | Path, *, status: SkillStatus = SkillStatus.ACTIVE
) -> SkillPackage:
    directory = Path(directory)
    frontmatter, _ = _frontmatter((directory / "SKILL.md").read_text(encoding="utf-8"))
    metadata: dict = {}
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    name = str(metadata.get("name", frontmatter.get("name", "")))
    kind = SkillKind(metadata.get("kind", frontmatter.get("kind", "specialist")))
    version = str(metadata.get("version", frontmatter.get("version", "0.1.0")))
    scope_raw = metadata.get("scope", metadata)
    scope = SkillScope(
        tuple(scope_raw.get("domains", ())),
        tuple(scope_raw.get("datasets", ())),
        tuple(scope_raw.get("temporal_windows", ())),
        tuple(scope_raw.get("tags", ())),
    )
    contract_raw = metadata.get("contract", {})
    contract = SkillContract(
        consumes=tuple(contract_raw.get("consumes", ())),
        produces=tuple(contract_raw.get("produces", ())),
        requires_capabilities=tuple(contract_raw.get("requires_capabilities", ())),
        optional_capabilities=tuple(contract_raw.get("optional_capabilities", ())),
        allow_root=bool(contract_raw.get("allow_root", True)),
        allow_parallel=bool(contract_raw.get("allow_parallel", True)),
        output_schema=contract_raw.get("output_schema"),
    )
    entry_raw = metadata.get("entrypoints", {})
    entrypoints = SkillEntrypoints(
        instructions=entry_raw.get("instructions", "SKILL.md"),
        template=entry_raw.get("template"),
        script=entry_raw.get("script"),
        output_schema=entry_raw.get("output_schema"),
        tests=tuple(entry_raw.get("tests", ())),
    )
    manifest = SkillManifest(
        name=name,
        kind=kind,
        version=version,
        scope=scope,
        triggers=tuple(Trigger(**row) for row in metadata.get("triggers", ())),
        contract=contract,
        entrypoints=entrypoints,
        safety_level=metadata.get("safety_level", "text_only"),
    )
    files = _read_files(directory)
    stable_id = hashlib.sha256(f"{name}:{kind.value}".encode()).hexdigest()[:20]
    package = SkillPackage(
        skill_id=str(metadata.get("skill_id", stable_id)),
        manifest=manifest,
        files=files,
        status=status,
        parent_ids=tuple(metadata.get("parent_ids", ())),
    )
    require_valid_package(package)
    return package
