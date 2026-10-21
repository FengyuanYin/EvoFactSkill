"""Batch-wise cross-domain evolution with rejected-validation trace feedback.

The durable retry loop is shared by ordinary, meta, and adversarial evolution;
each route supplies its own sample planning policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from evofact.config import config_identity
from evofact.core.models import SampleEvaluation, SkillKind, StatisticalTestResult
from evofact.data.domains import data_fingerprint, effective_domain, split_source_and_final
from evofact.evolution.firewall import validate_generation_request
from evofact.evolution.joint_optimizer import pending_bank
from evofact.evolution.optimizer_memory import ManagedOptimizerBackend, remember
from evofact.evolution.package_optimizer import PackageOptimizerAgent
from evofact.evolution.source_feedback import (
    SOURCE_FEEDBACK_POLICY,
    balanced_source_rows,
    source_recall_failures,
    source_validation_feedback,
    validate_source_feedback_manifest,
)
from evofact.evolution.training_feedback import (
    FEEDBACK_POLICY_VERSION,
    build_training_supervision,
    information_fingerprint,
    record_no_change,
)
from evofact.generation.memory import GeneratorContextFailure
from evofact.skills.candidates import apply_candidate
from evofact.skills.package_adapter import package_bank_digest, package_to_skill_spec
from evofact.skills.package_serializer import package_from_dict, package_to_dict
from evofact.validation.evaluator import evaluate
from evofact.validation.gate import ValidationGate
from evofact.validation.independent import (
    InferenceIncomplete,
    discovery_without_pool,
    guard_failures,
    independent_pool,
    independent_rows,
)
from evofact.validation.rounds import AGGREGATION_POLICY, discovery_rounds, independent_rounds

from .meta_runner import MetaEvolutionRunner, _evaluation_from, _proposal_from
from .meta_sampling import matched_meta_plan


def _json(value):
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


class MetaBatchRunner(MetaEvolutionRunner):
    mode = "meta-evolve"
    checkpoint_schema = "meta_batch_checkpoint_v1"
    run_prefix = "meta-batch-"

    def _identity_extra(self):
        return {}

    def _new_route_state(self):
        return {}

    def _after_optimizer_outcome(self, state, *, status, stop_unchanged):
        return stop_unchanged

    async def _optimizer_traces(self, state, batch, forbidden, save, *, progress, task):
        return state["train_traces"]

    def _reset_batch(self, state):
        state.pop("optimizer_no_change", None)

    def _optimizer_training_sources(self, state, batch):
        return [(sample, "real_training") for sample in batch]

    def _skip_batch(self, state, index, reason):
        state["skipped_batches"].append(
            {
                "batch": index + 1,
                "attempts": state["attempt"],
                "reason": reason,
                "bank_digest": package_bank_digest(self.base.packages),
                **(
                    {"no_change": dict(state["optimizer_no_change"])}
                    if reason == "no_new_information"
                    else {}
                ),
            }
        )
        state["completed_batches"] += 1
        self._reset_batch(state)
        state.update(pending=None, train_traces=None, feedback=[], attempt=0, commit_intent=None)

    def _extra_outcome(self, state):
        return {}

    async def _prepare_route_runtime(self, state, save, budget):
        pass

    def _check_enabled(self):
        if not self.config.meta_learning.enabled or not self.config.evolution.enabled:
            raise ValueError("meta_learning.enabled and evolution.enabled must be true")

    def _prepare_plan(
        self,
        samples,
        *,
        final_test_domains,
        train_domains,
        validation_domains,
        train_limit,
        match_evolve_training,
    ):
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
        batch_size = self.config.execution.batch_size
        sampling = None
        if match_evolve_training:
            if validation_domains:
                raise ValueError(
                    "matched training chooses validation domains per batch; omit --validation-domains"
                )
            domain_order = tuple(train_domains or self.config.data.train_domains or source)
            train, batches, valid, sampling = matched_meta_plan(
                samples,
                source_domains=domain_order,
                final_domains=final,
                seed=self.config.seed,
                batch_size=batch_size,
                train_limit=train_limit,
                label_contract_digest=self.base.label_contract_registry.digest,
            )
            heldout = ()
        else:
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
            if train_limit is not None:
                train = train[:train_limit]
            batches = [
                train[start : start + batch_size] for start in range(0, len(train), batch_size)
            ]
        random.Random(self.config.seed + 1).shuffle(valid)
        if not train or not valid:
            raise ValueError("meta evolution requires training and validation samples")
        return source, final, train, valid, batches, heldout, sampling

    def _after_save(self, checkpoint):
        pass

    def _proposal_history(self, state):
        return ()

    async def _clean_validation(
        self, pending, role, rows, skills, save, *, progress, task, phase=None
    ):
        """Persist successful samples; retry only runtime failures, never wrong labels."""
        cache = pending.setdefault(role + "_progress", {"completed": {}, "failures": []})

        def completed(sample, trace, evaluation):
            if evaluation.decision_origin == "runtime":
                failed(sample, OSError("Validation runtime abstention or budget exhaustion"))
                return
            cache["completed"][sample.sample_id] = {
                "trace": _json(asdict(trace)),
                "evaluation": _json(asdict(evaluation)),
            }
            save()

        def failed(sample, exc):
            cache["failures"].append(
                {
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trace": _json(asdict(exc.trace)) if getattr(exc, "trace", None) else None,
                }
            )
            save()

        def runtime_failure(sample, exc, attempt):
            failed(sample, exc)
            cache["failures"][-1]["attempt"] = attempt
            save()

        missing = [s for s in rows if s.sample_id not in cache["completed"]]
        if missing:
            await self.base.run(
                missing,
                skills=skills,
                progress=progress,
                task_name=task,
                phase=phase or "validation " + role,
                isolate_sample_budget=True,
                sample_retries=self.config.independent_validation.sample_retries,
                strict_runtime_failures=True,
                on_runtime_failure=runtime_failure,
                on_sample_complete=completed,
                on_sample_error=lambda sample, exc: None,
            )
        if any(s.sample_id not in cache["completed"] for s in rows):
            return False
        ordered = [cache["completed"][s.sample_id] for s in rows]
        result = evaluate(
            [
                SampleEvaluation(
                    **{
                        **r["evaluation"],
                        "allowed_labels": tuple(r["evaluation"]["allowed_labels"]),
                    }
                )
                for r in ordered
            ]
        )
        pending[role] = {"traces": [r["trace"] for r in ordered], "result": _json(asdict(result))}
        save()
        return True

    async def _paired_round_validation(
        self,
        pending,
        stage,
        plans,
        samples,
        baseline_skills,
        candidate_skills,
        save,
        *,
        progress,
        task,
    ):
        """Finish every fixed paired round, then recompute metrics from the union."""
        by_id = {s.sample_id: s for s in samples}
        ids = [sid for group in plans for sid in group]
        if len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("invalid fixed validation sample plan")
        records = pending.setdefault(stage + "_rounds", [{} for _ in plans])
        if len(records) != len(plans):
            raise ValueError("saved validation rounds do not match the fixed plan")
        for index, (record, group) in enumerate(zip(records, plans, strict=True)):
            if record.setdefault("sample_ids", group) != group:
                raise ValueError("saved paired validation sample IDs changed")
            rows = [by_id[sid] for sid in group]
            for role, skills in (("baseline", baseline_skills), ("candidate", candidate_skills)):
                if record.get(role) is not None:
                    continue
                clean = await self._clean_validation(
                    record,
                    role,
                    rows,
                    skills,
                    save,
                    progress=progress,
                    task=task,
                    phase=f"validation {stage} round {index + 1}/{len(plans)} {role}",
                )
                if not clean:
                    return False
            save()
        for role in ("baseline", "candidate"):
            evaluations = [
                row
                for record in records
                for row in _evaluation_from(record[role]["result"]).per_sample
            ]
            if [row.sample_id for row in evaluations] != ids:
                raise ValueError("paired validation results differ from the fixed sample plan")
            key = role if stage == "discovery" else "independent_" + role
            pending[key] = {
                "traces": [t for record in records for t in record[role]["traces"]],
                "result": _json(asdict(evaluate(evaluations))),
            }
        save()
        return True

    @staticmethod
    def _round_audit(pending):
        return {
            "policy": AGGREGATION_POLICY,
            "stages": {
                stage: [
                    {
                        "sample_ids": record["sample_ids"],
                        **{
                            role: record[role]["result"]["aggregate_metrics"]
                            for role in ("baseline", "candidate")
                        },
                        "runtime_failures": {
                            role: record.get(role + "_progress", {}).get("failures", [])
                            for role in ("baseline", "candidate")
                        },
                    }
                    for record in pending.get(stage + "_rounds", [])
                ]
                for stage in ("discovery", "independent")
            },
        }

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
        match_evolve_training=False,
        validation_rounds=1,
    ):
        self._check_enabled()
        if type(validation_rounds) is not int or validation_rounds < 1:
            raise ValueError("validation_rounds must be a positive integer")
        if max_attempts_per_batch < 1 or (train_limit is not None and train_limit < 1):
            raise ValueError("attempt limit and training limit must be positive")
        source, final, train, valid, batches, heldout, sampling = self._prepare_plan(
            samples,
            final_test_domains=final_test_domains,
            train_domains=train_domains,
            validation_domains=validation_domains,
            train_limit=train_limit,
            match_evolve_training=match_evolve_training,
        )
        policy = self.config.independent_validation
        source_feedback = policy.enabled and policy.source_feedback
        memory_policy = self.config.optimizer_memory
        independent = []
        if policy.enabled and not source_feedback:
            pool_candidates = getattr(self, "independent_validation_samples", None) or valid
            if any(effective_domain(s) not in source for s in pool_candidates):
                raise ValueError("independent validation candidates must stay in source domains")
            independent, _ = independent_pool(
                pool_candidates,
                train,
                [s for s in samples if effective_domain(s) in final],
                registry=self.base.label_contract_registry,
                per_domain=policy.samples_per_domain,
                seed=self.config.seed,
            )
            valid = discovery_without_pool(valid, independent)
            if policy.manifest_path is not None:
                expected = json.loads(
                    (self.root / policy.manifest_path).read_text(encoding="utf-8")
                )
                actual_pool = [
                    {
                        "sample_id": s.sample_id,
                        "domain": effective_domain(s),
                        "label": self.base.label_contract_registry.resolve_sample(s).normalize(
                            s.label
                        ),
                    }
                    for s in independent
                ]
                if (
                    expected["training_sample_ids"] != [s.sample_id for s in train]
                    or expected["independent_samples"] != actual_pool
                ):
                    raise ValueError(
                        "independent validation manifest does not match the training/data plan"
                    )
            if match_evolve_training:
                for batch in batches:
                    independent_rows(independent, batch, cross_domain=True)
            sampling = {
                **(sampling or {}),
                "independent_validation": {
                    "policy": "independent_balanced_source_domains_v1",
                    "pool_sample_ids": [s.sample_id for s in independent],
                    "samples_per_domain": policy.samples_per_domain,
                    "cross_domain_only": match_evolve_training,
                },
            }
            sampling["validation_sample_ids"] = [s.sample_id for s in valid]
        valid_size = self.config.data.evolution_validation_samples
        if source_feedback:
            if policy.manifest_path is not None:
                expected = json.loads(
                    (self.root / policy.manifest_path).read_text(encoding="utf-8")
                )
                validate_source_feedback_manifest(expected, train, valid)
            sampling = {
                **(sampling or {}),
                "validation_feedback_policy": SOURCE_FEEDBACK_POLICY,
                "validation_sample_ids": [s.sample_id for s in valid],
            }
            sampling.pop("protected_sample_ids", None)
        checkpoint = self.root / self.config.meta_learning.checkpoint_path
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": config_identity(self.config),
                    "data": data_fingerprint(samples),
                    "train": [s.sample_id for s in train],
                    "valid": [s.sample_id for s in valid],
                    "evaluation_only": evaluation_only,
                    **({"training_plan": sampling} if sampling is not None else {}),
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
            aggregation = state.get("validation_aggregation")
            if aggregation and (
                aggregation.get("policy") != AGGREGATION_POLICY
                or aggregation.get("rounds") != validation_rounds
            ):
                raise ValueError("resume must keep the checkpoint-recorded validation rounds")
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
                "total_batches": len(batches),
                "joint_optimization": True,
                "optimizer_feedback_policy": FEEDBACK_POLICY_VERSION,
                **self._new_route_state(),
                "validation_aggregation": {
                    "policy": AGGREGATION_POLICY,
                    "rounds": validation_rounds,
                },
                "pending": None,
                "packages": [package_to_dict(p) for p in self.base.packages],
                "committed_snapshots": [],
                "status": "running",
                **(
                    {"training_plan": sampling, "used_validation_ids": []}
                    if sampling is not None
                    else {}
                ),
            }
            state["trace_log"] = (
                str(Path(trace_log_path).resolve()) if trace_log_path is not None else None
            )
        # A missing marker is deliberately legacy, including a pending candidate.
        rounds = state.get("validation_aggregation", {}).get("rounds", 1)
        discovery_size = valid_size * rounds
        if rounds > 1 and policy.enabled and not source_feedback:
            # Fail before model calls if a complete balanced, disjoint plan cannot fit.
            for batch in batches:
                independent_rounds(
                    independent,
                    batch,
                    rounds=rounds,
                    per_domain=policy.evaluation_samples_per_domain,
                    registry=self.base.label_contract_registry,
                    seed=self.config.seed,
                    cross_domain=match_evolve_training,
                )
        if policy.enabled and not source_feedback:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            (checkpoint.parent / "independent-validation-manifest.json").write_text(
                json.dumps(
                    {
                        "policy": sampling["independent_validation"],
                        "samples": [
                            {
                                "sample_id": s.sample_id,
                                "domain": effective_domain(s),
                                "label": self.base.label_contract_registry.resolve_sample(
                                    s
                                ).normalize(s.label),
                            }
                            for s in independent
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        budget = self.base._budget_manager()
        if resume and state.get("budget"):
            budget.restore(state["budget"])

        def save():
            # Per-sample callbacks may save while sibling requests are in flight.
            # Persist settled usage only; reservations remain live in memory and
            # must not be restored as charges or stale locks after a restart.
            state["budget"] = budget.export_reconciled_state()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            os.replace(temporary, checkpoint)
            self._after_save(checkpoint)

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
        total = len(batches)
        # Older checkpoints stopped at attempt_limit; resume them with the same
        # schema and advance the exhausted batch without changing the active bank.
        state.setdefault("skipped_batches", [])
        state["status"] = "running"
        save()
        try:
            if memory_policy.enabled:
                try:
                    await ManagedOptimizerBackend(
                        self.base._optimizer_backend(), memory_policy, state, save, budget=budget
                    ).capabilities()
                except ValueError as exc:
                    state["status"] = "optimizer_context_failure"
                    state["optimizer_context_error"] = str(exc)
            if state["status"] == "running":
                try:
                    await self._prepare_route_runtime(state, save, budget)
                except GeneratorContextFailure as exc:
                    state["status"] = "generator_context_failure"
                    state["generator_context_error"] = str(exc)
            while state["status"] == "running" and state["completed_batches"] < total:
                index = state["completed_batches"]
                batch = batches[index]
                task = f"{self.mode} batch {index + 1}/{total}"
                if state["train_traces"] is None:
                    if policy.enabled:
                        inference = state.setdefault("training_inference", {})
                        clean = await self._clean_validation(
                            inference,
                            "training",
                            batch,
                            self.base.skills,
                            save,
                            progress=progress,
                            task=task,
                            phase="train inference",
                        )
                        if not clean:
                            state["status"] = "training_runtime_failure"
                            break
                        state["train_traces"] = inference["training"]["traces"]
                        state.setdefault("training_runtime_failures", []).append(
                            {
                                "batch": index + 1,
                                "failures": inference["training_progress"]["failures"],
                            }
                        )
                        state.pop("training_inference")
                        save()
                    else:
                        traces, _ = await self.base.run(
                            batch,
                            progress=progress,
                            task_name=task,
                            phase="train inference",
                        )
                        state["train_traces"] = _json([asdict(t) for t in traces])
                        save()
                if state["pending"] is None:
                    if state["attempt"] >= max_attempts_per_batch:
                        self._skip_batch(state, index, "attempt_limit")
                        save()
                        continue
                    if match_evolve_training or source_feedback:
                        used = set(state["used_validation_ids"])
                        batch_domains = {effective_domain(s) for s in batch}
                        available = [
                            s
                            for s in valid
                            if s.sample_id not in used and effective_domain(s) not in batch_domains
                        ]
                        observed_training_ids = {
                            s.sample_id for previous in batches[: index + 1] for s in previous
                        }
                        # Only other-domain samples can be held out for this
                        # batch. Same-domain siblings may share training evidence.
                        forbidden_validation = [
                            s for s in available if s.sample_id not in observed_training_ids
                        ]
                    else:
                        available = valid[state["validation_cursor"] :]
                        forbidden_validation = available
                    if source_feedback:
                        available = balanced_source_rows(
                            available,
                            size=valid_size,
                            rounds=rounds,
                            registry=self.base.label_contract_registry,
                        )
                    if len(available) < discovery_size:
                        state["status"] = "validation_pool_exhausted"
                        break
                    forbidden = (
                        independent
                        + forbidden_validation
                        + [s for s in samples if effective_domain(s) in final]
                    )
                    if source_feedback:
                        forbidden = [s for s in samples if effective_domain(s) in final]
                    try:
                        optimizer_traces = await self._optimizer_traces(
                            state, batch, forbidden, save, progress=progress, task=task
                        )
                    except GeneratorContextFailure as exc:
                        state["status"] = "generator_context_failure"
                        state["generator_context_error"] = str(exc)
                        break
                    except InferenceIncomplete:
                        state["status"] = "challenge_runtime_failure"
                        break
                    supervised = state.get("optimizer_feedback_policy") == FEEDBACK_POLICY_VERSION
                    supervision = None
                    fingerprint = None
                    if supervised:
                        supervision = build_training_supervision(
                            optimizer_traces,
                            self._optimizer_training_sources(state, batch),
                            self.base.label_contract_registry,
                        )
                        if source_feedback:
                            supervision["source_validation_feedback"] = list(
                                reversed(state.get("source_validation_feedback", []))
                            )
                        validate_generation_request(supervision, forbidden)
                        fingerprint = information_fingerprint(
                            package_bank_digest(self.base.packages),
                            optimizer_traces,
                            state["feedback"],
                            supervision,
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
                    backend = self.base._optimizer_backend()
                    if memory_policy.enabled:
                        backend = ManagedOptimizerBackend(
                            backend, memory_policy, state, save, budget=budget, forbidden=forbidden
                        )
                    optimizer = PackageOptimizerAgent(
                        backend=backend,
                        optimizer_package=optimizer_packages[0],
                        config=self.config.evolution,
                        budget_manager=None if memory_policy.enabled else budget,
                    )
                    if not (memory_policy.enabled and state.get("optimizer_request")):
                        state["attempt"] += 1
                    save()
                    # Most recent rejected attempt gets context priority over older feedback.
                    optimized = await optimizer.propose(
                        None,
                        self.base.packages,
                        traces=optimizer_traces,
                        feedback=list(reversed(state["feedback"])) or None,
                        prior_proposals=self._proposal_history(state),
                        feedback_enabled=True,
                        joint_enabled=state.get("joint_optimization", False),
                        training_supervision=supervision,
                    )
                    if memory_policy.enabled:
                        request = state.get("optimizer_request")
                        if optimized.reason == "call:OptimizerContextFailure":
                            state["status"] = "optimizer_context_failure"
                            save()
                            break
                        if request:
                            request["outcome"] = {
                                "status": optimized.status,
                                "reason": optimized.reason,
                            }
                            state.setdefault("optimizer_requests", []).append(request)
                            state.pop("optimizer_request", None)
                    candidate = optimized.candidate
                    candidates = optimized.candidates or ((candidate,) if candidate else ())
                    stop_unchanged = False
                    if supervised:
                        stop_unchanged = record_no_change(
                            state, fingerprint, no_change=optimized.status == "no_change"
                        )
                    stop_unchanged = self._after_optimizer_outcome(
                        state, status=optimized.status, stop_unchanged=stop_unchanged
                    )
                    if candidate is None or (
                        any(c.base is None for c in candidates)
                        and not self.config.evolution.discovery
                    ):
                        state["attempts"].append(
                            {
                                "batch": index + 1,
                                "attempt": state["attempt"],
                                "status": optimized.status,
                                "reason": optimized.reason,
                                **({"information_fingerprint": fingerprint} if supervised else {}),
                            }
                        )
                        if memory_policy.enabled:
                            remember(state, state["attempts"][-1], training=state["train_traces"])
                        if stop_unchanged:
                            self._skip_batch(state, index, "no_new_information")
                        save()
                        continue
                    if self.config.evolution.scope == "instructions":

                        def outside_scope(c):
                            if c.base is None:
                                return False
                            edits = {op.path for op in c.patch.file_operations}
                            manifest = replace(c.package.manifest, version=c.base.manifest.version)
                            return bool(
                                edits - {c.base.manifest.entrypoints.instructions}
                                or manifest != c.base.manifest
                            )

                        if any(outside_scope(c) for c in candidates):
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

                    proposals = [
                        package_candidate_to_proposal(
                            c,
                            cluster_id=task,
                            source_trace_ids=tuple(t["trace_id"] for t in optimizer_traces),
                        )
                        for c in candidates
                    ]
                    proposal = proposals[0]
                    forbidden = (
                        independent
                        + forbidden_validation
                        + [s for s in samples if effective_domain(s) in final]
                    )
                    if source_feedback:
                        forbidden = [s for s in samples if effective_domain(s) in final]
                    for item in proposals:
                        validate_generation_request(asdict(item), forbidden)
                    for c in candidates:
                        validate_generation_request(package_to_dict(c.package), forbidden)
                    start = state["validation_cursor"]
                    state["pending"] = {
                        "proposal": _json(asdict(proposal)),
                        "package": package_to_dict(candidate.package),
                        "base_id": candidate.base.skill_id if candidate.base else None,
                        "safety": "review_required"
                        if any(c.safety_level == "review_required" for c in candidates)
                        else candidate.safety_level,
                        "validation_start": start,
                        "baseline": None,
                        "candidate": None,
                        **(
                            {"validation_ids": [s.sample_id for s in available[:discovery_size]]}
                            if match_evolve_training or source_feedback
                            else {}
                        ),
                    }
                    if rounds > 1:
                        # Freeze both stages' complete sample plans before any evaluation.
                        state["pending"]["discovery_round_ids"] = discovery_rounds(
                            available, size=valid_size, rounds=rounds
                        )
                        if policy.enabled and not source_feedback:
                            guard_ids = independent_rounds(
                                independent,
                                batch,
                                rounds=rounds,
                                per_domain=policy.evaluation_samples_per_domain,
                                registry=self.base.label_contract_registry,
                                seed=self.config.seed + start,
                                cross_domain=match_evolve_training,
                            )
                            state["pending"]["independent_round_ids"] = guard_ids
                            state["pending"]["independent_sample_ids"] = [
                                sid for group in guard_ids for sid in group
                            ]
                    if state.get("joint_optimization"):
                        state["pending"].update(
                            proposals=[_json(asdict(p)) for p in proposals],
                            packages=[package_to_dict(c.package) for c in candidates],
                            base_ids=[c.base.skill_id for c in candidates if c.base is not None],
                            joint_rationale=optimized.reason,
                        )
                    if match_evolve_training or source_feedback:
                        state["used_validation_ids"].extend(state["pending"]["validation_ids"])
                    state["validation_cursor"] += discovery_size
                    save()
                pending = state["pending"]
                proposal = _proposal_from(pending["proposal"])
                candidate_skills = (
                    [package_to_skill_spec(p) for p in pending_bank(self.base.packages, pending)]
                    if "packages" in pending
                    else apply_candidate(self.base.skills, proposal)
                )
                if rounds > 1:
                    by_id = {s.sample_id: s for s in valid}
                    rows = [by_id[sid] for group in pending["discovery_round_ids"] for sid in group]
                elif match_evolve_training or source_feedback:
                    by_id = {s.sample_id: s for s in valid}
                    rows = [by_id[sid] for sid in pending["validation_ids"]]
                else:
                    rows = valid[
                        pending["validation_start"] : pending["validation_start"] + valid_size
                    ]
                if rounds > 1:
                    clean = await self._paired_round_validation(
                        pending,
                        "discovery",
                        pending["discovery_round_ids"],
                        valid,
                        self.base.skills,
                        candidate_skills,
                        save,
                        progress=progress,
                        task=task,
                    )
                    if not clean:
                        state["status"] = "validation_runtime_failure"
                for role, skills in (
                    ()
                    if rounds > 1
                    else (
                        ("baseline", self.base.skills),
                        ("candidate", candidate_skills),
                    )
                ):
                    if pending[role] is None:
                        if policy.enabled:
                            clean = await self._clean_validation(
                                pending, role, rows, skills, save, progress=progress, task=task
                            )
                            if not clean:
                                state["status"] = "validation_runtime_failure"
                                break
                            continue
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
                if state["status"] == "validation_runtime_failure":
                    break
                decision = ValidationGate(self.config.gate).decide(
                    _evaluation_from(pending["baseline"]["result"]),
                    _evaluation_from(pending["candidate"]["result"]),
                    StatisticalTestResult("not_used_for_promotion", 0, 1, False),
                    safety=pending["safety"],
                )
                if source_feedback and decision.accepted:
                    failures = source_recall_failures(
                        _evaluation_from(pending["baseline"]["result"]),
                        _evaluation_from(pending["candidate"]["result"]),
                        max_drop=policy.max_positive_recall_drop,
                    )
                    if failures:
                        decision = replace(
                            decision,
                            accepted=False,
                            disposition="rejected",
                            regression_failures=tuple(failures),
                            reason="; ".join(failures),
                        )
                if policy.enabled and not source_feedback and decision.accepted:
                    guard_rows = (
                        []
                        if rounds > 1
                        else independent_rows(
                            independent,
                            batch,
                            cross_domain=match_evolve_training,
                            per_domain=policy.evaluation_samples_per_domain,
                            registry=self.base.label_contract_registry,
                            seed=self.config.seed + pending["validation_start"],
                        )
                    )
                    pending.setdefault("independent_sample_ids", [s.sample_id for s in guard_rows])
                    by_id = {s.sample_id: s for s in independent}
                    guard_rows = [by_id[sid] for sid in pending["independent_sample_ids"]]
                    if rounds > 1:
                        clean = await self._paired_round_validation(
                            pending,
                            "independent",
                            pending["independent_round_ids"],
                            independent,
                            self.base.skills,
                            candidate_skills,
                            save,
                            progress=progress,
                            task=task,
                        )
                        if not clean:
                            state["status"] = "validation_runtime_failure"
                    for role, skills in (
                        ()
                        if rounds > 1
                        else (
                            ("independent_baseline", self.base.skills),
                            ("independent_candidate", candidate_skills),
                        )
                    ):
                        if pending.get(role) is None:
                            clean = await self._clean_validation(
                                pending,
                                role,
                                guard_rows,
                                skills,
                                save,
                                progress=progress,
                                task=task,
                            )
                            if not clean:
                                state["status"] = "validation_runtime_failure"
                                break
                    if state["status"] == "validation_runtime_failure":
                        break
                    failures = guard_failures(
                        _evaluation_from(pending["independent_baseline"]["result"]),
                        _evaluation_from(pending["independent_candidate"]["result"]),
                        policy,
                    )
                    if failures:
                        decision = replace(
                            decision,
                            accepted=False,
                            disposition="rejected",
                            regression_failures=tuple(failures),
                            reason="; ".join(failures),
                        )
                audit = {
                    "batch": index + 1,
                    "attempt": state["attempt"],
                    "validation_sample_ids": [s.sample_id for s in rows],
                    "proposal": pending["proposal"],
                    **(
                        {
                            "proposals": pending["proposals"],
                            "joint_rationale": pending.get("joint_rationale", ""),
                        }
                        if "proposals" in pending
                        else {}
                    ),
                    "decision": _json(asdict(decision)),
                    **(
                        {"validation_aggregation": self._round_audit(pending)} if rounds > 1 else {}
                    ),
                    **(
                        {
                            "train_domains": sorted({effective_domain(s) for s in batch}),
                            "validation_domains": sorted({effective_domain(s) for s in rows}),
                        }
                        if match_evolve_training
                        else {}
                    ),
                }
                if policy.enabled:
                    audit["runtime_failures"] = {
                        k: v["failures"] for k, v in pending.items() if k.endswith("_progress")
                    }
                    if pending.get("independent_baseline"):
                        audit["independent_validation"] = {
                            "sample_ids": pending["independent_sample_ids"],
                            "baseline": pending["independent_baseline"]["result"],
                            "candidate": pending["independent_candidate"]["result"],
                        }
                state["attempts"].append(audit)
                if source_feedback:
                    state.setdefault("source_validation_feedback", []).append(
                        source_validation_feedback(
                            pending,
                            valid,
                            self.base.label_contract_registry,
                            batch=index + 1,
                            attempt=state["attempt"],
                            decision=decision,
                        )
                    )
                if memory_policy.enabled:
                    remember(state, audit, pending=pending, training=state["train_traces"])
                if not decision.accepted:
                    state["feedback"].append(
                        {
                            "attempt": state["attempt"],
                            "rejected_package": pending["package"],
                            **(
                                {"rejected_packages": pending["packages"]}
                                if "packages" in pending
                                else {}
                            ),
                            "gate": audit["decision"],
                            "baseline_traces": pending["baseline"]["traces"],
                            "candidate_traces": pending["candidate"]["traces"],
                        }
                    )
                    state["pending"] = None
                    save()
                    continue
                desired = pending_bank(self.base.packages, pending)
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
            "validation_domains": list(source) if match_evolve_training else list(heldout),
            **(
                {
                    "training_sampling": sampling,
                    "validation_policy": "different_domain_from_current_batch",
                }
                if sampling is not None
                else {}
            ),
            "validation_samples_used": state["validation_cursor"],
            "validation_aggregation": state.get(
                "validation_aggregation",
                {
                    "policy": "legacy_single_round",
                    "rounds": 1,
                },
            ),
            "attempts": state["attempts"],
            "skipped_batches": state["skipped_batches"],
            "committed_snapshots": state["committed_snapshots"],
            "checkpoint": str(checkpoint),
            "mock_results": self.config.backend == "mock",
            **(
                {
                    "optimizer_memory": {
                        "records": len(state.get("optimizer_memories", [])),
                        "requests": len(state.get("optimizer_requests", [])),
                        "error": state.get("optimizer_context_error"),
                    }
                }
                if memory_policy.enabled
                else {}
            ),
            **self._extra_outcome(state),
        }
