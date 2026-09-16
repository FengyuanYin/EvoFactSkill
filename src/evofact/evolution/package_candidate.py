from __future__ import annotations

from dataclasses import dataclass

from evofact.core.package_models import PackageValidationReport, SkillPackage, SkillPackagePatch
from evofact.governance.package_policy import validate_package
from evofact.security.package_scanner import scan_package
from evofact.skills.package_patch import apply_package_patch


@dataclass(frozen=True)
class PackageCandidate:
    base: SkillPackage
    patch: SkillPackagePatch
    package: SkillPackage
    validation: PackageValidationReport
    safety_level: str
    safety_findings: tuple[str, ...]


def build_package_candidate(base: SkillPackage, patch: SkillPackagePatch) -> PackageCandidate:
    package = apply_package_patch(base, patch)
    validation = validate_package(package)
    if not validation.valid:
        raise ValueError("candidate Package failed fixed validation")
    safety = scan_package(package)
    if safety.level == "blocked":
        raise ValueError("candidate Package failed safety scan: " + "; ".join(safety.findings))
    return PackageCandidate(base, patch, package, validation, safety.level, safety.findings)
