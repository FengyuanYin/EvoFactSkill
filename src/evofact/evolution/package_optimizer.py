from __future__ import annotations

import asyncio
import base64

from evofact.core.budget_models import BudgetRequest
from evofact.core.models import SkillKind, SkillScope, Trigger
from evofact.core.package_models import (
    FileOperation,
    FileOperationKind,
    ManifestPatch,
    SkillContract,
    SkillEntrypoints,
    SkillPackage,
    SkillPackagePatch,
)
from evofact.runtime.node_runner import _usage_details
from evofact.skills.package_adapter import package_to_skill_spec

from .optimizer_context import build_optimizer_context
from .package_candidate import build_package_candidate

ALLOWED_TARGET_KINDS = {SkillKind.ROUTER, SkillKind.SPECIALIST, SkillKind.JUDGE, SkillKind.WORKFLOW}
FROZEN_TARGET_NAMES = {
    "generation_verifier",
    "verifier",
    "package_validator",
    "dag_policy",
    "budget_policy",
    "pricing_policy",
    "data_firewall",
    "skill_optimizer",
}


def parse_package_optimizer_response(
    data: object, packages: list[SkillPackage] | tuple[SkillPackage, ...]
):
    if not isinstance(data, dict):
        raise ValueError("optimizer response must be an object")
    allowed = {
        "action",
        "target_skill_id",
        "rationale",
        "file_operations",
        "manifest_patch",
        "source_trace_ids",
        "source_audit_ids",
        "risk_flags",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown optimizer fields: {sorted(unknown)}")
    action = data.get("action")
    if action == "no_change":
        return None
    if action not in {"add", "edit"}:
        raise ValueError("optimizer action must be add, edit, or no_change")
    by_id = {item.skill_id: item for item in packages}
    target = by_id.get(data.get("target_skill_id"))
    if target is None:
        raise ValueError("optimizer target must be an existing Package")
    if target.manifest.kind not in ALLOWED_TARGET_KINDS:
        raise ValueError("optimizer target kind is frozen")
    if target.manifest.name in FROZEN_TARGET_NAMES:
        raise ValueError("optimizer target is a frozen governance component")
    operations = []
    for raw in data.get("file_operations", ()):
        if not isinstance(raw, dict):
            raise ValueError("file operation must be an object")
        fields = {
            "operation",
            "path",
            "content",
            "content_base64",
            "destination",
            "expected_digest",
            "media_type",
            "executable",
        }
        if set(raw) - fields:
            raise ValueError("unknown file operation fields")
        if raw.get("content") is not None and raw.get("content_base64") is not None:
            raise ValueError("file operation content encoding is ambiguous")
        content = None
        if raw.get("content") is not None:
            content = str(raw["content"]).encode("utf-8")
        elif raw.get("content_base64") is not None:
            content = base64.b64decode(raw["content_base64"], validate=True)
        operations.append(
            FileOperation(
                FileOperationKind(raw["operation"]),
                raw["path"],
                content,
                raw.get("destination"),
                raw.get("expected_digest"),
                raw.get("media_type"),
                raw.get("executable"),
            )
        )
    manifest_raw = data.get("manifest_patch")
    manifest_patch = None
    if manifest_raw is not None:
        if not isinstance(manifest_raw, dict):
            raise ValueError("manifest_patch must be an object")
        allowed_manifest = {
            "version",
            "scope",
            "triggers",
            "contract",
            "entrypoints",
            "safety_level",
        }
        if set(manifest_raw) - allowed_manifest:
            raise ValueError("name and kind cannot be changed by optimizer")
        converted = dict(manifest_raw)
        if "scope" in converted:
            raw = converted["scope"]
            if not isinstance(raw, dict):
                raise ValueError("manifest scope must be an object")
            converted["scope"] = SkillScope(**{key: tuple(value) for key, value in raw.items()})
        if "triggers" in converted:
            raw = converted["triggers"]
            if not isinstance(raw, list):
                raise ValueError("manifest triggers must be an array")
            converted["triggers"] = tuple(Trigger(**item) for item in raw)
        if "contract" in converted:
            raw = converted["contract"]
            if not isinstance(raw, dict):
                raise ValueError("manifest contract must be an object")
            converted["contract"] = SkillContract(
                **{
                    **raw,
                    "consumes": tuple(raw.get("consumes", ())),
                    "produces": tuple(raw.get("produces", ())),
                    "requires_capabilities": tuple(raw.get("requires_capabilities", ())),
                    "optional_capabilities": tuple(raw.get("optional_capabilities", ())),
                }
            )
        if "entrypoints" in converted:
            raw = converted["entrypoints"]
            if not isinstance(raw, dict):
                raise ValueError("manifest entrypoints must be an object")
            converted["entrypoints"] = SkillEntrypoints(
                **{**raw, "tests": tuple(raw.get("tests", ()))}
            )
        manifest_patch = ManifestPatch(**converted)
    risk_flags = tuple(data.get("risk_flags", ()))
    if any(item.path.startswith("scripts/") for item in operations):
        risk_flags = tuple(dict.fromkeys((*risk_flags, "script_change", "human_review_required")))
    return SkillPackagePatch(
        target.skill_id,
        target.package_digest,
        tuple(operations),
        manifest_patch,
        str(data.get("rationale", "")),
        tuple(data.get("source_trace_ids", ())),
        tuple(data.get("source_audit_ids", ())),
        risk_flags,
    )


def legacy_instruction_edit_to_patch(
    target: SkillPackage,
    instructions: str,
    *,
    rationale: str,
    source_trace_ids: tuple[str, ...] = (),
) -> SkillPackagePatch:
    if target.manifest.kind not in ALLOWED_TARGET_KINDS:
        raise ValueError("legacy edit target is frozen")
    current = target.file(target.manifest.entrypoints.instructions)
    raw = current.content.decode("utf-8")
    if raw.startswith("---\n"):
        head, marker, _ = raw[4:].partition("\n---\n")
        if not marker:
            raise ValueError("target SKILL.md frontmatter is invalid")
        replacement = f"---\n{head}\n---\n{instructions.strip()}\n".encode()
    else:
        replacement = (instructions.strip() + "\n").encode()
    operation = FileOperation(
        FileOperationKind.UPDATE,
        current.path,
        replacement,
        expected_digest=current.digest,
        media_type=current.media_type,
    )
    return SkillPackagePatch(
        target.skill_id,
        target.package_digest,
        (operation,),
        None,
        rationale,
        source_trace_ids,
    )


class PackageOptimizerAgent:
    """Model-facing optimizer for complete Skill Packages.

    The optimizer package itself is only used as a prompt and is never an
    eligible target. Fixed parsing, validation and safety scanning remain
    outside the model-controlled boundary.
    """

    def __init__(self, backend, optimizer_package: SkillPackage, *, budget_manager=None):
        if optimizer_package.manifest.kind != SkillKind.META:
            raise ValueError("optimizer_package must be a META Package")
        self.backend = backend
        self.optimizer_package = optimizer_package
        self.optimizer_skill = package_to_skill_spec(optimizer_package)
        self.budget_manager = budget_manager

    async def propose(
        self,
        target: SkillPackage,
        packages,
        *,
        reports=(),
        traces=(),
        audits=(),
    ):
        if target.skill_id == self.optimizer_package.skill_id:
            raise ValueError("optimizer cannot optimize itself")
        context = build_optimizer_context(
            target=target,
            reports=reports,
            traces=traces,
            audits=audits,
        )
        reservation = None
        sample_id = f"package-optimizer:{target.skill_id}"
        try:
            if self.budget_manager is not None:
                reservation = await self.budget_manager.reserve(
                    sample_id,
                    BudgetRequest(calls=1, tokens=4000, purpose="package_optimizer"),
                )
                async with self.budget_manager.concurrency(sample_id):
                    result = await asyncio.wait_for(
                        self.backend.optimize_package(context, self.optimizer_skill),
                        self.budget_manager.limits.call_timeout_ms / 1000,
                    )
            else:
                result = await self.backend.optimize_package(context, self.optimizer_skill)
            if reservation is not None:
                await self.budget_manager.reconcile(reservation, _usage_details(result))
                reservation = None
            patch = parse_package_optimizer_response(result.value, tuple(packages))
            if patch is None:
                return None
            if patch.target_skill_id != target.skill_id:
                raise ValueError("optimizer response targeted a Package outside this evaluation")
            return build_package_candidate(target, patch)
        finally:
            if reservation is not None:
                await self.budget_manager.release(reservation)
