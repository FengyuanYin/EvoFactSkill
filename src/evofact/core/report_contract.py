from __future__ import annotations

import json

from evofact.core.models import SkillKind
from evofact.core.package_models import SkillPackage, SpecialistReportContract

MAX_REPORT_CONTRACT_BYTES = 32_000
_ALLOWED_FIELDS = {"schema_version", "report_type", "finding_types", "guidance"}


def parse_specialist_report_contract(content: bytes | str) -> SpecialistReportContract:
    raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
    if len(raw_bytes) > MAX_REPORT_CONTRACT_BYTES:
        raise ValueError("specialist report contract is too large")
    try:
        data = json.loads(raw_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("specialist report contract must be valid UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("specialist report contract must be an object")
    unknown = set(data) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"unknown specialist report contract fields: {sorted(unknown)}")
    finding_types = data.get("finding_types")
    if not isinstance(finding_types, list) or any(
        not isinstance(item, str) for item in finding_types
    ):
        raise ValueError("specialist finding_types must be an array of strings")
    return SpecialistReportContract(
        report_type=data.get("report_type", ""),
        finding_types=tuple(finding_types),
        guidance=data.get("guidance", ""),
        schema_version=data.get("schema_version", ""),
    )


def load_specialist_report_contract(
    package: SkillPackage,
) -> SpecialistReportContract | None:
    if package.manifest.kind != SkillKind.SPECIALIST:
        return None
    path = package.manifest.entrypoints.output_schema
    if path is None:
        return None
    return parse_specialist_report_contract(package.file(path).content)
