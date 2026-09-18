from __future__ import annotations

import asyncio
import base64
import hashlib

from evofact.core.budget_models import BudgetRequest
from evofact.core.models import SkillKind, SkillScope, SkillStatus, Trigger
from evofact.core.package_models import (
    FileOperation,
    FileOperationKind,
    ManifestPatch,
    SkillContract,
    SkillEntrypoints,
    SkillFile,
    SkillManifest,
    SkillPackage,
    SkillPackageAddition,
    SkillPackagePatch,
)
from evofact.governance.package_policy import is_reserved_label_contract_path
from evofact.runtime.node_runner import _usage_details
from evofact.skills.package_adapter import package_to_skill_spec

from .optimizer_context import build_optimizer_context
from .package_candidate import build_package_addition_candidate, build_package_candidate

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


def _decode_file_content(raw: dict) -> bytes | None:
    if raw.get("content") is not None and raw.get("content_base64") is not None:
        raise ValueError("file content encoding is ambiguous")
    if raw.get("content") is not None:
        return str(raw["content"]).encode("utf-8")
    if raw.get("content_base64") is not None:
        return base64.b64decode(raw["content_base64"], validate=True)
    return None


def _parse_scope(raw: object) -> SkillScope:
    if not isinstance(raw, dict):
        raise ValueError("manifest scope must be an object")
    allowed = {"domains", "datasets", "temporal_windows", "tags"}
    if set(raw) - allowed:
        raise ValueError("unknown manifest scope fields")
    return SkillScope(**{key: tuple(value) for key, value in raw.items()})


def _parse_contract(raw: object) -> SkillContract:
    if not isinstance(raw, dict):
        raise ValueError("manifest contract must be an object")
    return SkillContract(
        **{
            **raw,
            "consumes": tuple(raw.get("consumes", ())),
            "produces": tuple(raw.get("produces", ())),
            "requires_capabilities": tuple(raw.get("requires_capabilities", ())),
            "optional_capabilities": tuple(raw.get("optional_capabilities", ())),
        }
    )


def _parse_entrypoints(raw: object) -> SkillEntrypoints:
    if not isinstance(raw, dict):
        raise ValueError("manifest entrypoints must be an object")
    return SkillEntrypoints(**{**raw, "tests": tuple(raw.get("tests", ()))})


def _parse_package_addition(data: dict, packages: tuple[SkillPackage, ...]):
    if data.get("target_skill_id") is not None:
        raise ValueError("add must not specify target_skill_id")
    raw_package = data.get("new_package")
    if not isinstance(raw_package, dict):
        raise ValueError("add must provide new_package")
    if set(raw_package) - {"skill_id", "manifest", "files"}:
        raise ValueError("unknown new_package fields")
    raw_manifest = raw_package.get("manifest")
    if not isinstance(raw_manifest, dict):
        raise ValueError("new_package manifest must be an object")
    allowed_manifest = {
        "name",
        "kind",
        "version",
        "scope",
        "triggers",
        "contract",
        "entrypoints",
        "safety_level",
    }
    if set(raw_manifest) - allowed_manifest:
        raise ValueError("unknown new_package manifest fields")
    kind = SkillKind(raw_manifest.get("kind"))
    if kind != SkillKind.SPECIALIST:
        raise ValueError("optimizer may only add specialist Packages")
    manifest = SkillManifest(
        name=raw_manifest.get("name", ""),
        kind=kind,
        version=raw_manifest.get("version", "0.1.0"),
        scope=_parse_scope(raw_manifest.get("scope", {})),
        triggers=tuple(Trigger(**item) for item in raw_manifest.get("triggers", ())),
        contract=_parse_contract(raw_manifest.get("contract", {})),
        entrypoints=_parse_entrypoints(raw_manifest.get("entrypoints", {})),
        safety_level=raw_manifest.get("safety_level", "text_only"),
    )
    raw_files = raw_package.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("new_package files must be a non-empty array")
    files = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("new package file must be an object")
        if set(raw) - {"path", "content", "content_base64", "media_type", "executable"}:
            raise ValueError("unknown new package file fields")
        content = _decode_file_content(raw)
        if content is None:
            raise ValueError("new package file content is required")
        path = raw.get("path", "")
        if is_reserved_label_contract_path(str(path)):
            raise ValueError("optimizer cannot add Runtime-owned dataset label contract files")
        media_type = raw.get("media_type") or (
            "application/json" if str(path).endswith(".json") else "text/plain"
        )
        files.append(
            SkillFile(
                path,
                media_type,
                content,
                hashlib.sha256(content).hexdigest(),
                bool(raw.get("executable", False)),
            )
        )
    stable_id = hashlib.sha256(f"{manifest.name}:{kind.value}".encode()).hexdigest()[:20]
    skill_id = str(raw_package.get("skill_id") or stable_id)
    if any(item.skill_id == skill_id or item.manifest.name == manifest.name for item in packages):
        raise ValueError("new Package name or skill_id already exists")
    package = SkillPackage(
        skill_id=skill_id,
        manifest=manifest,
        files=tuple(files),
        status=SkillStatus.CANDIDATE,
    )
    paths = {item.path for item in package.files}
    required_paths = {"SKILL.md", "metadata.json"}
    if package.manifest.entrypoints.output_schema is None:
        raise ValueError("new specialist Package must declare an output schema entrypoint")
    required_paths.add(package.manifest.entrypoints.output_schema)
    if not required_paths <= paths:
        raise ValueError("new specialist Package is missing required package files")
    risk_flags = tuple(data.get("risk_flags", ()))
    if any(item.path.startswith("scripts/") for item in package.files):
        risk_flags = tuple(dict.fromkeys((*risk_flags, "script_change", "human_review_required")))
    return SkillPackageAddition(
        package,
        str(data.get("rationale", "")),
        tuple(data.get("source_trace_ids", ())),
        tuple(data.get("source_audit_ids", ())),
        risk_flags,
    )


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
        "new_package",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown optimizer fields: {sorted(unknown)}")
    action = data.get("action")
    if action == "no_change":
        return None
    if action not in {"add", "edit"}:
        raise ValueError("optimizer action must be add, edit, or no_change")
    if action == "add":
        if data.get("file_operations") not in (None, []):
            raise ValueError("add must not contain file_operations")
        if data.get("manifest_patch") is not None:
            raise ValueError("add must not contain manifest_patch")
        return _parse_package_addition(data, tuple(packages))
    if data.get("new_package") is not None:
        raise ValueError("edit must not contain new_package")
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
        if is_reserved_label_contract_path(str(raw.get("path", ""))) or (
            raw.get("destination") is not None
            and is_reserved_label_contract_path(str(raw["destination"]))
        ):
            raise ValueError("optimizer cannot modify Runtime-owned dataset label contract files")
        content = _decode_file_content(raw)
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
            converted["scope"] = _parse_scope(raw)
        if "triggers" in converted:
            raw = converted["triggers"]
            if not isinstance(raw, list):
                raise ValueError("manifest triggers must be an array")
            converted["triggers"] = tuple(Trigger(**item) for item in raw)
        if "contract" in converted:
            raw = converted["contract"]
            if not isinstance(raw, dict):
                raise ValueError("manifest contract must be an object")
            converted["contract"] = _parse_contract(raw)
        if "entrypoints" in converted:
            raw = converted["entrypoints"]
            if not isinstance(raw, dict):
                raise ValueError("manifest entrypoints must be an object")
            converted["entrypoints"] = _parse_entrypoints(raw)
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
    if (
        target.manifest.kind not in ALLOWED_TARGET_KINDS
        or target.manifest.name in FROZEN_TARGET_NAMES
    ):
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
                async with self.budget_manager.concurrency(sample_id):
                    reservation = await self.budget_manager.reserve(
                        sample_id,
                        BudgetRequest(calls=1, tokens=4000, purpose="package_optimizer"),
                    )
                    result = await asyncio.wait_for(
                        self.backend.optimize_package(context, self.optimizer_skill),
                        self.budget_manager.limits.call_timeout_ms / 1000,
                    )
            else:
                result = await self.backend.optimize_package(context, self.optimizer_skill)
            if reservation is not None:
                await self.budget_manager.reconcile(reservation, _usage_details(result))
                reservation = None
            proposal = parse_package_optimizer_response(result.value, tuple(packages))
            if proposal is None:
                return None
            if isinstance(proposal, SkillPackageAddition):
                return build_package_addition_candidate(proposal)
            if proposal.target_skill_id != target.skill_id:
                raise ValueError("optimizer response targeted a Package outside this evaluation")
            return build_package_candidate(target, proposal)
        finally:
            if reservation is not None:
                await self.budget_manager.release(reservation)
