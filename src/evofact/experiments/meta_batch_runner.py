"""Batch-wise cross-domain evolution with rejected-validation trace feedback.

This runner deliberately does not call or modify the ordinary evolution loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from evofact.core.models import SkillKind, SkillStatus, StatisticalTestResult
from evofact.data.domains import data_fingerprint, effective_domain, split_source_and_final
from evofact.evolution.firewall import validate_generation_request
from evofact.evolution.package_optimizer import PackageOptimizerAgent
from evofact.skills.candidates import apply_candidate
from evofact.skills.package_adapter import package_bank_digest, package_to_skill_spec
from evofact.skills.package_serializer import package_from_dict, package_to_dict
from evofact.validation.gate import ValidationGate

from .meta_runner import MetaEvolutionRunner, _evaluation_from, _proposal_from


def _json(value):
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


class MetaBatchRunner(MetaEvolutionRunner):
    mode = "meta-evolve"
    checkpoint_schema = "meta_batch_checkpoint_v1"
    run_prefix = "meta-batch-"

    def _identity_extra(self):
        return {}

    async def _optimizer_traces(self, state, batch, forbidden, save, *, progress, task):
        return state["train_traces"]

    def _reset_batch(self, state):
        pass

    def _extra_outcome(self, state):
        return {}

    async def run(
        self,
        samples,
        *,
        final_test_domains=(),
        train_domains=None,
        validation_domains=None,
        max_attempts_per_batch=5,
        train_limit=None,
        resume=False,
        evaluation_only=False,
        progress=None,
        trace_log_path=None,
    ):
        if not self.config.meta_learning.enabled or not self.config.evolution.enabled:
            raise ValueError("meta_learning.enabled and evolution.enabled must be true")
        if max_attempts_per_batch < 1 or (train_limit is not None and train_limit < 1):
            raise ValueError("attempt limit and training limit must be positive")
        source, final = split_source_and_final(samples, final_test_domains, train_domains)
        event_domains = {}
        for sample in samples:
            domain = effective_domain(sample)
            if domain in source and (
                sample.metadata.get("robustness_only") or sample.metadata.get("split") == "test"
            ):
                raise ValueError("test-only samples cannot enter meta source domains")
            if sample.event_id:
                if event_domains.setdefault(sample.event_id, domain) != domain:
                    raise ValueError("an event crosses meta train, validation, or final domains")
        domains = sorted(source)
        heldout = tuple(
            validation_domains or domains[-self.config.meta_learning.meta_test_domain_count :]
        )
        if not heldout or not set(heldout) <= set(domains) or set(heldout) == set(domains):
            raise ValueError(
                "validation domains must be a non-empty proper subset of source domains"
            )
        train = [s for s in samples if effective_domain(s) in set(source) - set(heldout)]
        valid = [s for s in samples if effective_domain(s) in set(heldout)]
        random.Random(self.config.seed).shuffle(train)
        random.Random(self.config.seed + 1).shuffle(valid)
        if train_limit is not None:
            train = train[:train_limit]
        if not train or not valid:
            raise ValueError("meta evolution requires training and validation samples")
        batch_size = self.config.execution.batch_size
        valid_size = self.config.data.evolution_validation_samples
        checkpoint = self.root / self.config.meta_learning.checkpoint_path
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "data": data_fingerprint(samples),
                    "train": [s.sample_id for s in train],
                    "valid": [s.sample_id for s in valid],
                    "evaluation_only": evaluation_only,
                    **self._identity_extra(),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if resume:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                state.get("schema_version") != self.checkpoint_schema
                or state.get("identity") != identity
            ):
                raise ValueError(
                    "incompatible meta batch checkpoint; use a separate checkpoint path"
                )
            if state.get("trace_log") != (
                str(Path(trace_log_path).resolve()) if trace_log_path is not None else None
            ):
                raise ValueError("resume must use the checkpoint-recorded trace log")
        else:
            if checkpoint.exists():
                raise ValueError(
                    "meta checkpoint already exists; use --resume or a new checkpoint path"
                )
            state = {
                "schema_version": self.checkpoint_schema,
                "identity": identity,
                "run_id": self.run_prefix + uuid.uuid4().hex,
                "completed_batches": 0,
                "validation_cursor": 0,
                "attempt": 0,
                "train_traces": None,
                "feedback": [],
                "attempts": [],
                "pending": None,
                "packages": [package_to_dict(p) for p in self.base.packages],
                "committed_snapshots": [],
                "status": "running",
            }
            state["trace_log"] = (
                str(Path(trace_log_path).resolve()) if trace_log_path is not None else None
            )
        budget = self.base._budget_manager()
        if resume and state.get("budget"):
            budget.restore(state["budget"])

        def save():
            state["budget"] = budget.export_state()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            os.replace(temporary, checkpoint)

        def sync_bank(packages):
            self.base.packages = packages
            self.base.skills = [package_to_skill_spec(p) for p in packages]

        # A saved commit intent closes the repository/checkpoint crash window.
        if state.get("commit_intent"):
            intent = state["commit_intent"]
            desired = [package_from_dict(p) for p in intent["packages"]]
            old = [package_from_dict(p) for p in state["packages"]]
            if self.repository is not None and not evaluation_only:
                active = list(self.repository.active_packages().values())
                if package_bank_digest(active) == package_bank_digest(old):
                    snapshots = self.repository.commit_package_bank(
                        desired,
                        run_id=intent["run_id"],
                        expected_active={p.manifest.name: p.package_digest for p in old},
                        audit=intent["audit"],
                    )
                    state["committed_snapshots"].extend(str(p) for p in snapshots)
                elif package_bank_digest(active) != package_bank_digest(desired):
                    raise ValueError("active bank changed outside this meta run")
            state["packages"] = intent["packages"]
            state["completed_batches"] += 1
            self._reset_batch(state)
            state.update(
                commit_intent=None, pending=None, train_traces=None, feedback=[], attempt=0
            )
            save()
        packages = [package_from_dict(p) for p in state["packages"]]
        if self.repository is not None and not evaluation_only:
            if package_bank_digest(
                list(self.repository.active_packages().values())
            ) != package_bank_digest(packages):
                raise ValueError("active bank changed outside this meta run")
        sync_bank(packages)
        if trace_log_path is not None:
            self.base.enable_trace_logging(trace_log_path, run_id=state["run_id"], resume=resume)
        total = (len(train) + batch_size - 1) // batch_size
        state["status"] = "running"
        save()
        try:
            while state["completed_batches"] < total:
                index = state["completed_batches"]
                task = f"{self.mode} batch {index + 1}/{total}"
                if state["train_traces"] is None:
                    traces, _ = await self.base.run(
                        train[index * batch_size : (index + 1) * batch_size],
                        progress=progress,
                        task_name=task,
                        phase="train inference",
                    )
                    state["train_traces"] = _json([asdict(t) for t in traces])
                    save()
                if state["pending"] is None:
                    if state["attempt"] >= max_attempts_per_batch:
                        state["status"] = "attempt_limit"
                        break
                    if len(valid) - state["validation_cursor"] < valid_size:
                        state["status"] = "validation_pool_exhausted"
                        break
                    forbidden = valid[state["validation_cursor"] :] + [
                        s for s in samples if effective_domain(s) in final
                    ]
                    optimizer_traces = await self._optimizer_traces(
                        state,
                        train[index * batch_size : (index + 1) * batch_size],
                        forbidden,
                        save,
                        progress=progress,
                        task=task,
                    )
                    optimizer_packages = [
                        p
                        for p in self.base.packages
                        if p.manifest.name == self.config.evolution.optimizer_skill
                        and p.manifest.kind == SkillKind.META
                    ]
                    if len(optimizer_packages) != 1:
                        raise ValueError(
                            "optimizer Package must resolve to exactly one META Package"
                        )
                    optimizer = PackageOptimizerAgent(
                        backend=self.base._optimizer_backend(),
                        optimizer_package=optimizer_packages[0],
                        config=self.config.evolution,
                        budget_manager=budget,
                    )
                    state["attempt"] += 1
                    save()
                    # Most recent rejected attempt gets context priority over older feedback.
                    optimized = await optimizer.propose(
                        None,
                        self.base.packages,
                        traces=optimizer_traces,
                        feedback=list(reversed(state["feedback"])) or None,
                    )
                    candidate = optimized.candidate
                    if candidate is None or (
                        candidate.base is None and not self.config.evolution.discovery
                    ):
                        state["attempts"].append(
                            {
                                "batch": index + 1,
                                "attempt": state["attempt"],
                                "status": optimized.status,
                                "reason": optimized.reason,
                            }
                        )
                        save()
                        continue
                    if candidate.base is not None and self.config.evolution.scope == "instructions":
                        edits = {op.path for op in candidate.patch.file_operations}
                        manifest = replace(
                            candidate.package.manifest, version=candidate.base.manifest.version
                        )
                        if (
                            edits - {candidate.base.manifest.entrypoints.instructions}
                            or manifest != candidate.base.manifest
                        ):
                            state["attempts"].append(
                                {
                                    "batch": index + 1,
                                    "attempt": state["attempt"],
                                    "status": "scope_violation",
                                }
                            )
                            save()
                            continue
                    from evofact.evolution.package_candidate import package_candidate_to_proposal

                    proposal = package_candidate_to_proposal(
                        candidate,
                        cluster_id=task,
                        source_trace_ids=tuple(t["trace_id"] for t in optimizer_traces),
                    )
                    forbidden = valid[state["validation_cursor"] :] + [
                        s for s in samples if effective_domain(s) in final
                    ]
                    validate_generation_request(asdict(proposal), forbidden)
                    start = state["validation_cursor"]
                    state["pending"] = {
                        "proposal": _json(asdict(proposal)),
                        "package": package_to_dict(candidate.package),
                        "base_id": candidate.base.skill_id if candidate.base else None,
                        "safety": candidate.safety_level,
                        "validation_start": start,
                        "baseline": None,
                        "candidate": None,
                    }
                    state["validation_cursor"] += valid_size
                    save()
                pending = state["pending"]
                proposal = _proposal_from(pending["proposal"])
                candidate_skills = apply_candidate(self.base.skills, proposal)
                rows = valid[pending["validation_start"] : pending["validation_start"] + valid_size]
                for role, skills in (
                    ("baseline", self.base.skills),
                    ("candidate", candidate_skills),
                ):
                    if pending[role] is None:
                        traces, result = await self.base.run(
                            rows,
                            skills=skills,
                            progress=progress,
                            task_name=task,
                            phase="validation " + role,
                            isolate_sample_budget=True,
                        )
                        pending[role] = {
                            "traces": _json([asdict(t) for t in traces]),
                            "result": _json(asdict(result)),
                        }
                        save()
                decision = ValidationGate(self.config.gate).decide(
                    _evaluation_from(pending["baseline"]["result"]),
                    _evaluation_from(pending["candidate"]["result"]),
                    StatisticalTestResult("not_used_for_promotion", 0, 1, False),
                    safety=pending["safety"],
                )
                audit = {
                    "batch": index + 1,
                    "attempt": state["attempt"],
                    "validation_sample_ids": [s.sample_id for s in rows],
                    "proposal": pending["proposal"],
                    "decision": _json(asdict(decision)),
                }
                state["attempts"].append(audit)
                if not decision.accepted:
                    state["feedback"].append(
                        {
                            "attempt": state["attempt"],
                            "rejected_package": pending["package"],
                            "gate": audit["decision"],
                            "baseline_traces": pending["baseline"]["traces"],
                            "candidate_traces": pending["candidate"]["traces"],
                        }
                    )
                    state["pending"] = None
                    save()
                    continue
                desired = [p for p in self.base.packages if p.skill_id != pending["base_id"]]
                desired.append(
                    replace(package_from_dict(pending["package"]), status=SkillStatus.ACTIVE)
                )
                intent = {
                    "packages": [package_to_dict(p) for p in desired],
                    "run_id": f"{state['run_id']}-batch-{index + 1}",
                    "audit": audit,
                }
                state["commit_intent"] = intent
                save()
                if self.repository is not None and not evaluation_only:
                    snapshots = self.repository.commit_package_bank(
                        desired,
                        run_id=intent["run_id"],
                        expected_active={
                            p.manifest.name: p.package_digest for p in self.base.packages
                        },
                        audit=audit,
                    )
                    state["committed_snapshots"].extend(str(p) for p in snapshots)
                sync_bank(desired)
                state["packages"] = intent["packages"]
                state["completed_batches"] += 1
                self._reset_batch(state)
                state.update(
                    commit_intent=None, pending=None, train_traces=None, feedback=[], attempt=0
                )
                save()
            if state["completed_batches"] == total:
                state["status"] = "completed"
        finally:
            save()
        return {
            "mode": self.mode,
            "run_id": state["run_id"],
            "status": state["status"],
            "completed_batches": state["completed_batches"],
            "total_batches": total,
            "train_domains": sorted(set(source) - set(heldout)),
            "validation_domains": list(heldout),
            "validation_samples_used": state["validation_cursor"],
            "attempts": state["attempts"],
            "committed_snapshots": state["committed_snapshots"],
            "checkpoint": str(checkpoint),
            "mock_results": self.config.backend == "mock",
            **self._extra_outcome(state),
        }
