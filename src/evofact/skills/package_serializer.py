from __future__ import annotations

import base64
import json
from typing import Any

from evofact.core.models import SkillKind, SkillScope, SkillStatus, Trigger
from evofact.core.package_models import (
    SkillContract,
    SkillEntrypoints,
    SkillFile,
    SkillManifest,
    SkillPackage,
)
from evofact.governance import SKILL_PACKAGE_SCHEMA_VERSION


def package_to_dict(package: SkillPackage) -> dict[str, Any]:
    return {
        "schema_version": package.schema_version,
        "skill_id": package.skill_id,
        "status": package.status.value,
        "parent_ids": list(package.parent_ids),
        "package_digest": package.package_digest,
        "manifest": {
            "schema_version": package.manifest.schema_version,
            "name": package.manifest.name,
            "kind": package.manifest.kind.value,
            "version": package.manifest.version,
            "scope": package.manifest.scope.model_dump(),
            "triggers": [item.model_dump() for item in package.manifest.triggers],
            "contract": package.manifest.contract.model_dump(),
            "entrypoints": package.manifest.entrypoints.model_dump(),
            "safety_level": package.manifest.safety_level,
        },
        "files": [
            {
                "schema_version": item.schema_version,
                "path": item.path,
                "media_type": item.media_type,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
                "digest": item.digest,
                "executable": item.executable,
            }
            for item in package.files
        ],
    }


def package_from_dict(data: dict[str, Any]) -> SkillPackage:
    if data.get("schema_version") != SKILL_PACKAGE_SCHEMA_VERSION:
        raise ValueError(f"unsupported package schema: {data.get('schema_version')}")
    raw = data["manifest"]
    scope = SkillScope(**{key: tuple(value) for key, value in raw["scope"].items()})
    contract_raw = raw["contract"]
    contract = SkillContract(
        **{
            **contract_raw,
            "consumes": tuple(contract_raw.get("consumes", ())),
            "produces": tuple(contract_raw.get("produces", ())),
            "requires_capabilities": tuple(contract_raw.get("requires_capabilities", ())),
            "optional_capabilities": tuple(contract_raw.get("optional_capabilities", ())),
        }
    )
    entry_raw = raw["entrypoints"]
    entrypoints = SkillEntrypoints(**{**entry_raw, "tests": tuple(entry_raw.get("tests", ()))})
    manifest = SkillManifest(
        name=raw["name"],
        kind=SkillKind(raw["kind"]),
        version=raw["version"],
        scope=scope,
        triggers=tuple(Trigger(**item) for item in raw.get("triggers", ())),
        contract=contract,
        entrypoints=entrypoints,
        safety_level=raw["safety_level"],
        schema_version=raw["schema_version"],
    )
    files = tuple(
        SkillFile(
            path=item["path"],
            media_type=item["media_type"],
            content=base64.b64decode(item["content_base64"], validate=True),
            digest=item["digest"],
            executable=bool(item.get("executable", False)),
            schema_version=item["schema_version"],
        )
        for item in data["files"]
    )
    return SkillPackage(
        skill_id=data["skill_id"],
        manifest=manifest,
        files=files,
        status=SkillStatus(data["status"]),
        parent_ids=tuple(data.get("parent_ids", ())),
        package_digest=data["package_digest"],
        schema_version=data["schema_version"],
    )


def dumps_package(package: SkillPackage) -> str:
    return json.dumps(
        package_to_dict(package), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def loads_package(value: str | bytes) -> SkillPackage:
    return package_from_dict(json.loads(value))
