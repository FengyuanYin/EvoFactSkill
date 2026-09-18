from __future__ import annotations

import hashlib
from dataclasses import dataclass

from evofact.core.models import EvolutionOperation, EvolutionProposal
from evofact.core.package_models import (
    PackageValidationReport,
    SkillPackage,
    SkillPackageAddition,
    SkillPackagePatch,
)
from evofact.governance.package_policy import validate_package
from evofact.security.package_scanner import scan_package
from evofact.skills.package_adapter import package_to_skill_spec
from evofact.skills.package_patch import apply_package_patch


@dataclass(frozen=True)
class PackageCandidate:
    base: SkillPackage | None
    patch: SkillPackagePatch | None
    package: SkillPackage
    validation: PackageValidationReport
    safety_level: str
    safety_findings: tuple[str, ...]


def package_candidate_to_proposal(
    candidate: PackageCandidate,
    *,
    cluster_id: str,
    source_trace_ids: tuple[str, ...],
) -> EvolutionProposal:
    """Expose an exact Package candidate through the legacy evaluation contract.

    The proposal is only a runtime projection. Persistence must use the paired
    ``PackageCandidate`` so files that are not representable by ``SkillSpec``
    (including binary assets) can never be reconstructed or lost.
    """
    runtime_skill = package_to_skill_spec(candidate.package)
    operation = EvolutionOperation.ADD if candidate.base is None else EvolutionOperation.EDIT
    targets = () if candidate.base is None else (candidate.base.skill_id,)
    proposal_id = (
        "package-"
        + hashlib.sha256(
            f"{cluster_id}:{candidate.package.package_digest}".encode("utf-8")
        ).hexdigest()[:16]
    )
    rationale = (
        candidate.patch.rationale
        if candidate.patch is not None
        else "complete Package addition proposed by optimizer"
    )
    return EvolutionProposal(
        proposal_id=proposal_id,
        operation=operation,
        rationale=rationale,
        target_skill_ids=targets,
        candidate_skills=(runtime_skill,),
        source_trace_ids=tuple(dict.fromkeys(source_trace_ids)),
        error_cluster_id=cluster_id,
    )


def build_package_candidate(base: SkillPackage, patch: SkillPackagePatch) -> PackageCandidate:
    package = apply_package_patch(base, patch)
    validation = validate_package(package)
    if not validation.valid:
        raise ValueError("candidate Package failed fixed validation")
    safety = scan_package(package)
    if safety.level == "blocked":
        raise ValueError("candidate Package failed safety scan: " + "; ".join(safety.findings))
    return PackageCandidate(base, patch, package, validation, safety.level, safety.findings)


def build_package_addition_candidate(addition: SkillPackageAddition) -> PackageCandidate:
    package = addition.package
    validation = validate_package(package)
    if not validation.valid:
        raise ValueError("candidate Package failed fixed validation")
    safety = scan_package(package)
    if safety.level == "blocked":
        raise ValueError("candidate Package failed safety scan: " + "; ".join(safety.findings))
    return PackageCandidate(None, None, package, validation, safety.level, safety.findings)
