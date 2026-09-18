"""Trace-conditioned, one-shot LLM sample generation within DEMSE."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal

from evofact.attribution.rules import attribute_trace
from evofact.core.budget_models import CostStatus, UsageDetails
from evofact.core.generation_models import (
    GeneratedSample,
    GenerationAuditEntry,
    GenerationLineage,
    VerificationDecision,
)
from evofact.core.models import DomainEpisode
from evofact.data.domains import skillbank_fingerprint
from evofact.evolution.firewall import CandidateFirewall, validate_generation_request
from evofact.generation.attribution import attribute_generation
from evofact.generation.data import validate_fact_links
from evofact.generation.generator import ChallengeGenerator, json_value
from evofact.generation.prompts import VERIFIER_SYSTEM
from evofact.generation.robustness import build_robustness_set
from evofact.generation.split import split_construction_probe
from evofact.generation.verifier import ChallengeVerifier, normalized
from evofact.generation.workflow import GenerationWorkflow
from evofact.governance import GENERATION_AUDIT_SCHEMA_VERSION
from evofact.runtime.progress import ProgressEvent
from evofact.security.scanner import scan_resources
from evofact.skills.candidates import apply_candidate

from .checkpoint import MetaCheckpointStore
from .meta_runner import MetaEvolutionRunner


def fingerprint(value):
    """函数作用：负责当前模块中的 `fingerprint` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def per_label_error_rates(result):
    """函数作用：负责当前模块中的 `false_positive_rate` 处理，封装调用方需要复用的业务步骤。
    输入要求：`result`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    rates = {}
    labels = dict.fromkeys(label for row in result.per_sample for label in row.allowed_labels)
    for label in labels:
        selected = [row for row in result.per_sample if row.gold == label]
        rates[label] = (
            sum(row.predicted != label for row in selected) / len(selected) if selected else 0.0
        )
    return rates


class AdversarialEvolutionRunner(MetaEvolutionRunner):
    def __init__(self, config, project_root, facts=(), repository=None, *, generation_backend=None):
        """函数作用：创建并初始化 `AdversarialEvolutionRunner` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`config`（未显式标注）需符合函数签名约定；`project_root`（未显式标注）需符合函数签名约定；`facts`（未显式标注，默认 `()`）需符合函数签名约定；`repository`（未显式标注，默认 `None`）需符合函数签名约定；`generation_backend`（未显式标注，默认 `None`）需以关键字传入并符合签名约定。
        输出：返回 `None`；初始化 `AdversarialEvolutionRunner` 的实例状态，构造参数非法时可能抛出异常。"""
        super().__init__(config, project_root, repository)
        self.progress_task_name = "adversarial-evolve"
        self.facts = list(facts)
        self.audit = {}
        self.generator = ChallengeGenerator()
        self.generator_package = next(
            package for package in self.base.packages if package.manifest.name == "generation_agent"
        )
        self.generation_workflow = GenerationWorkflow(self.generator_package)
        self.verifier = ChallengeVerifier()
        # Dependency injection is for tests; CLI has no deterministic generation path.
        self.generation_backend = generation_backend
        self.resume_generation = False
        self.generation_budget = self.base._budget_manager()

    async def run(
        self,
        samples,
        *,
        final_test_domains=(),
        resume=False,
        evaluation_only=False,
        progress=None,
    ):
        """函数作用：执行当前对象负责的主运行流程，并汇总本轮结果。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`samples`（未显式标注）需符合函数签名约定；`final_test_domains`（未显式标注，默认 `()`）需以关键字传入并符合签名约定；`resume`（未显式标注，默认 `False`）需以关键字传入并符合签名约定；`evaluation_only`（未显式标注，默认 `False`）需以关键字传入并符合签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if not self.config.generation.enabled:
            raise ValueError("generation.enabled must be true for adversarial-evolve")
        if not final_test_domains:
            raise ValueError("adversarial evolution requires explicit final-test domains")
        if self.config.backend == "mock" and self.generation_backend is None:
            raise ValueError(
                "LLM generation requires a real backend; mock is only allowed with an injected test backend"
            )
        validate_fact_links(samples, self.facts)
        evidence_domains = {}
        for sample in samples:
            domain = sample.domain or sample.dataset
            for evidence in sample.evidence:
                keys = [("text", normalized(evidence.text))]
                if evidence.source:
                    keys.append(("source", evidence.source))
                for key in keys:
                    if key in evidence_domains and evidence_domains[key] != domain:
                        raise ValueError("evidence source/content crosses adversarial domains")
                    evidence_domains[key] = domain
            if (sample.domain or sample.dataset) not in final_test_domains:
                if (
                    sample.metadata.get("robustness_only")
                    or str(sample.metadata.get("official_split", "")).casefold() == "test"
                    or sample.metadata.get("split_role") == "test"
                ):
                    raise ValueError("test-only samples cannot enter adversarial source domains")
            if any(
                sample.published_at and e.published_at and e.published_at > sample.published_at
                for e in sample.evidence
            ):
                raise ValueError("future evidence cannot support adversarial samples")
        store_path = self.root / self.config.generation.store_path
        checkpoint_path = self.root / self.config.meta_learning.checkpoint_path
        if store_path.resolve() == checkpoint_path.resolve():
            raise ValueError("generation store and checkpoint must be separate paths")
        if store_path.exists():
            stored = json.loads(store_path.read_text(encoding="utf-8"))
            if stored.get("schema_version") != GENERATION_AUDIT_SCHEMA_VERSION:
                raise ValueError(
                    "legacy generation bank is incompatible; choose a new generation.store_path"
                )
        self.audit = {}
        self.resume_generation = resume
        outcome = await super().run(
            samples,
            final_test_domains=final_test_domains,
            resume=resume,
            evaluation_only=evaluation_only,
            progress=progress,
        )
        if not evaluation_only:
            self._commit_generation(store_path, outcome.run_id)
        return outcome

    def _config_fingerprint(self):
        """函数作用：计算当前运行配置的稳定标识 `_config_fingerprint` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return fingerprint(
            {
                "schema": "generation_workflow_v4",
                "config": asdict(self.config),
                "generator_package_digest": self.generator_package.package_digest,
                "verifier_system": VERIFIER_SYSTEM,
                "facts": sorted((asdict(f) for f in self.facts), key=lambda f: f["fact_id"]),
                "label_contract_digest": self.base.label_contract_registry.digest,
            }
        )

    def _checkpoint_extra(self):
        """函数作用：负责`AdversarialEvolutionRunner` 中的 `_checkpoint_extra` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return {
            **super()._checkpoint_extra(),
            "generation_schema": "generation_workflow_v4",
            "generator_package_digest": self.generator_package.package_digest,
            "verifier_fingerprint": fingerprint(
                {
                    "system": VERIFIER_SYSTEM,
                    "backend": self.config.backend,
                    "model": self.config.model,
                }
            ),
            "generation_episodes": self.audit,
            "label_contract_digest": self.base.label_contract_registry.digest,
        }

    def _restore_extra(self, saved):
        """函数作用：从检查点恢复 `_restore_extra` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`saved`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
        super()._restore_extra(saved)
        self.audit = saved.get("generation_episodes", {})
        if saved and (
            saved.get("generation_schema") != "generation_workflow_v4"
            or saved.get("generator_package_digest") != self.generator_package.package_digest
            or set(self.audit) != set(saved.get("completed_episode_ids", ()))
            or saved.get("label_contract_digest") != self.base.label_contract_registry.digest
        ):
            raise ValueError("generation checkpoint episode audit is incompatible or incomplete")

    def _commit_generation(self, path, run_id):
        """函数作用：原子提交 `_commit_generation` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`path`（未显式标注）需符合函数签名约定；`run_id`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
        current = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"schema_version": GENERATION_AUDIT_SCHEMA_VERSION, "runs": {}}
        )
        record = json_value(
            {
                "generator_package_digest": self.generator_package.package_digest,
                "verifier_version": fingerprint(VERIFIER_SYSTEM),
                "label_contract_digest": self.base.label_contract_registry.digest,
                "episodes": self.audit,
            }
        )
        if run_id in current["runs"]:
            if current["runs"][run_id] != record:
                raise ValueError(
                    "generation audit differs for an existing run; use --resume or a new output path"
                )
            return
        current["runs"][run_id] = record
        MetaCheckpointStore(path).save(current)

    async def _evolve_episode(self, train_rows, episode, firewall, *, progress=None):
        """函数作用：负责`AdversarialEvolutionRunner` 中的 `_evolve_episode` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`train_rows`（未显式标注）需符合函数签名约定；`episode`（未显式标注）需符合函数签名约定；`firewall`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        cfg = self.config.generation
        task_name = f"adversarial-evolve {episode.episode_id}"

        def phase(name, completed=0):
            if progress is not None:
                progress.update(ProgressEvent(task_name, name, completed, 1, "stages"))

        allowed = {s.sample_id for s in train_rows}
        train_facts = [f for f in self.facts if f.sample_id in allowed]
        construction, probe, facts = split_construction_probe(
            train_rows,
            train_facts,
            fraction=cfg.probe_fraction,
            seed=episode.episode_id,
            label_contract_registry=self.base.label_contract_registry,
        )
        probe_contracts = {
            self.base.label_contract_registry.resolve_sample(sample).key for sample in probe
        }
        for key in probe_contracts:
            contract = self.base.label_contract_registry.resolve(*key)
            present = {
                contract.normalize(sample.label)
                for sample in probe
                if (sample.dataset, sample.label_schema_id) == key
            }
            missing = set(contract.allowed_labels) - present
            if missing:
                raise ValueError(
                    f"probe lacks label coverage for {contract.dataset_id}/{contract.schema_id}: "
                    f"{sorted(missing)}"
                )
        frozen_id = skillbank_fingerprint(self.base.skills)
        backend = self.generation_backend or self.base._backend()
        # A persisted response can be reused after interruption during verification/evolution.
        key = fingerprint(
            {
                "config": self._config_fingerprint(),
                "episode": asdict(episode),
                "construction": [asdict(s) for s in construction],
                "bank": frozen_id,
                "label_contract_digest": self.base.label_contract_registry.digest,
            }
        )
        cache_path = (
            (self.root / self.config.meta_learning.checkpoint_path).parent
            / "generation-cache"
            / f"{key}.json"
        )
        forbidden = probe + [
            s
            for s in firewall.samples.values()
            if s.sample_id in episode.meta_test_sample_ids
            or (s.domain or s.dataset) in episode.final_test_domains
        ]
        if self.resume_generation and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("input_fingerprint") != key:
                raise ValueError("generation response cache mismatch")
            if "budget_state" not in cached:
                raise ValueError("generation response cache lacks budget accounting state")
            self.generation_budget.restore(cached["budget_state"], allow_forward=True)
            request, response = cached["request"], cached["response"]
            request_digest = fingerprint(request)
            response_digest = fingerprint(response)
            usage_raw = dict(cached.get("generation_usage", {}))
            if usage_raw.get("cost") is not None:
                usage_raw["cost"] = Decimal(str(usage_raw["cost"]))
            if usage_raw.get("cost_status") is not None:
                usage_raw["cost_status"] = CostStatus(usage_raw["cost_status"])
            generation_usage = UsageDetails(**usage_raw)
        else:
            traces, _ = await self.base.run(
                construction,
                progress=progress,
                task_name=task_name,
                phase="construction inference",
            )
            by_id = {s.sample_id: s for s in construction}
            reports = []
            for trace in traces:
                source = by_id[trace.sample_id]
                contract = self.base.label_contract_registry.resolve_sample(source)
                reports.append(attribute_trace(trace, contract.normalize(source.label), contract))
            firewall.build_generation_view(episode, traces, reports)
            phase("generate")
            workflow_result = await self.generation_workflow.run(
                backend,
                construction,
                traces,
                reports,
                config=cfg,
                episode_id=episode.episode_id,
                budget_manager=self.generation_budget,
                label_contract_registry=self.base.label_contract_registry,
                forbidden=forbidden,
            )
            request = workflow_result.request
            response = workflow_result.response
            request_digest = workflow_result.request_digest
            response_digest = workflow_result.response_digest
            generation_usage = workflow_result.usage
            phase("generate", 1)
            MetaCheckpointStore(cache_path).save(
                {
                    "input_fingerprint": key,
                    "request": request,
                    "response": response,
                    "generation_usage": asdict(generation_usage),
                    "budget_state": self.generation_budget.export_state(),
                }
            )
        validate_generation_request(request, forbidden)
        structured, rejected = self.verifier.validate(
            response,
            request,
            config=cfg,
            forbidden=forbidden,
            existing=construction,
            label_contract_registry=self.base.label_contract_registry,
            allowed_strategies=self.generation_workflow.strategies,
        )
        phase("verify")
        verification = await self.verifier.verify(
            structured,
            backend,
            budget_manager=self.generation_budget,
            budget_sample_id=episode.episode_id,
            label_contract_registry=self.base.label_contract_registry,
        )
        synthetic, review_rejections, reviews = verification
        verifier_usage = verification.usage
        phase("verify", 1)
        rejected.extend(review_rejections)
        decisions = {row["sample_id"]: row for row in response["decisions"]}
        construction_by_id = {sample.sample_id: sample for sample in construction}
        reviews_by_id = {
            row["sample_id"]: VerificationDecision(
                sample_id=row["sample_id"],
                status=row["status"],
                predicted_label=row["predicted_label"],
                accepted=row["accepted"],
                reason=row["reason"],
                verifier_version=row["verifier_version"],
                label_contract_digest=row.get("label_contract_digest", ""),
            )
            for row in reviews
        }
        accepted_ids = {sample.sample_id for sample in synthetic}
        generated_samples = []
        audit_entries = []
        config_digest = self._config_fingerprint()
        created_at = datetime.fromtimestamp(int(key[:8], 16), tz=timezone.utc).isoformat()
        for sample in structured:
            decision = decisions[sample.sample_id]
            source = construction_by_id[decision["source_sample_id"]]
            contract = self.base.label_contract_registry.resolve_sample(sample)
            evidence_digests = tuple(fingerprint(asdict(evidence)) for evidence in sample.evidence)
            lineage = GenerationLineage(
                generated_sample_id=sample.sample_id,
                source_sample_id=source.sample_id,
                lineage_id=fingerprint(
                    {
                        "episode_id": episode.episode_id,
                        "generated_sample_id": sample.sample_id,
                        "source_sample_id": source.sample_id,
                        "strategy": decision["strategy"],
                        "generator_package_digest": self.generator_package.package_digest,
                    }
                ),
                episode_id=episode.episode_id,
                strategy=decision["strategy"],
                event_id=sample.event_id,
                evidence_digests=evidence_digests,
                generator_package_digest=self.generator_package.package_digest,
                model=str(getattr(backend, "model", type(backend).__name__)),
                config_digest=config_digest,
                created_at=created_at,
                label_schema_id=sample.label_schema_id,
                label_contract_digest=contract.digest,
            )
            generated = GeneratedSample(sample=sample, lineage=lineage)
            if sample.sample_id in accepted_ids:
                generated_samples.append(generated)
            review = reviews_by_id.get(sample.sample_id)
            reasons = (
                ()
                if review and review.accepted
                else (review.reason if review else "verification result missing",)
            )
            audit_entries.append(
                GenerationAuditEntry(
                    audit_id=fingerprint(
                        {"lineage_id": lineage.lineage_id, "response": response_digest}
                    ),
                    lineage=lineage,
                    accepted=sample.sample_id in accepted_ids,
                    reasons=reasons,
                    request_digest=request_digest,
                    response_digest=response_digest,
                    verifier=review,
                    # Generator and Verifier calls are batch-level charges. Per-sample
                    # entries keep detailed identity without duplicating that cost.
                    usage=UsageDetails(),
                )
            )
        robustness = build_robustness_set(tuple(generated_samples))
        generation_attribution = attribute_generation(tuple(audit_entries))
        rows = construction + synthetic
        inner_episode = DomainEpisode(
            episode.episode_id,
            episode.seed,
            episode.strategy,
            episode.meta_train_domains,
            episode.meta_test_domains,
            tuple(s.sample_id for s in rows),
            episode.meta_test_sample_ids,
            episode.final_test_domains,
        )
        inner_firewall = CandidateFirewall(rows, episode.final_test_domains)
        phase("evolve")
        inner = await self.base.evolve_once(
            rows,
            generation_guard=lambda traces, reports: inner_firewall.build_generation_view(
                inner_episode, traces, reports
            ),
            progress=progress,
            task_name=task_name,
        )
        phase("evolve", 1)
        # Probe stays diagnostic. Neither its texts nor its metrics return to the generator.
        _, baseline = await self.base.run(
            probe, progress=progress, task_name=task_name, phase="probe baseline"
        )
        evaluations = []
        for proposal in inner["proposals"]:
            firewall.validate_candidate(proposal, episode)
            blob = json.dumps(asdict(proposal), ensure_ascii=False)
            if any(s.sample_id in blob or s.text in blob for s in probe):
                raise ValueError("candidate contains probe content")
            if any(scan_resources(s.resources).level != "safe" for s in proposal.candidate_skills):
                continue
            _, result = await self.base.run(
                probe,
                skills=apply_candidate(self.base.skills, proposal),
                progress=progress,
                task_name=task_name,
                phase="probe candidate",
            )
            evaluations.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "metrics": result.aggregate_metrics,
                    "teaching_gain": result.aggregate_metrics["macro_f1_all"]
                    - baseline.aggregate_metrics["macro_f1_all"],
                    "per_label_error_rate": per_label_error_rates(result),
                    "worst_label_error_rate": max(
                        per_label_error_rates(result).values(), default=0.0
                    ),
                }
            )
        challenge_ids = {s.sample_id for s in synthetic}
        challenge_rows = [r for r in inner["evaluation"].per_sample if r.sample_id in challenge_ids]
        if skillbank_fingerprint(self.base.skills) != frozen_id:
            raise RuntimeError("detector baseline mutated during generation evaluation")
        generated_count = len(response["samples"])
        all_labels = tuple(
            dict.fromkeys(
                label
                for sample in construction
                for label in self.base.label_contract_registry.resolve_sample(sample).allowed_labels
            )
        )
        accepted_by_label = {
            label: [sample for sample in synthetic if sample.label == label] for label in all_labels
        }
        reviewed_by_label = {
            label: [
                sample
                for sample in structured
                if sample.label == label and sample.sample_id in reviews_by_id
            ]
            for label in all_labels
        }
        rejection_blob = " ".join(str(item.get("reason", "")) for item in rejected).casefold()
        teaching_gain = max((float(item["teaching_gain"]) for item in evaluations), default=0.0)
        worst_probe_drop = max(
            (
                max(
                    0.0,
                    float(baseline.aggregate_metrics.get("macro_f1_all", 0.0))
                    - float(item["metrics"].get("macro_f1_all", 0.0)),
                )
                for item in evaluations
            ),
            default=0.0,
        )
        self.audit[episode.episode_id] = json_value(
            {
                "construction_ids": [s.sample_id for s in construction],
                "probe_ids": [s.sample_id for s in probe],
                "fact_ids": [f.fact_id for f in facts],
                "detector_baseline": frozen_id,
                "generator_calls": 1,
                "verifier_calls": int(bool(structured)),
                "generator_package_digest": self.generator_package.package_digest,
                "verifier_version": fingerprint(VERIFIER_SYSTEM),
                "request": request,
                "response": response,
                "samples": [asdict(s) for s in synthetic],
                "lineages": [asdict(item.lineage) for item in generated_samples],
                "generation_audit_entries": [asdict(item) for item in audit_entries],
                "generation_attribution": [asdict(item) for item in generation_attribution],
                "generated_robustness": asdict(robustness),
                "generation_usage": asdict(generation_usage),
                "verifier_usage": asdict(verifier_usage),
                "rejected": rejected,
                "reviews": reviews,
                "metrics": {
                    "generated": generated_count,
                    "accepted": len(synthetic),
                    "validity": len(synthetic) / max(1, generated_count),
                    "agreement": len(synthetic) / max(1, len(structured)),
                    "evidence_rate": sum(bool(sample.evidence) for sample in structured)
                    / max(1, generated_count),
                    "diversity": len({normalized(sample.text) for sample in structured})
                    / max(1, len(structured)),
                    "duplicate_rate": sum(
                        "duplicate" in str(item.get("reason", "")).casefold() for item in rejected
                    )
                    / max(1, generated_count),
                    "leakage_rate": sum(
                        "held-out" in str(item.get("reason", "")).casefold() for item in rejected
                    )
                    / max(1, generated_count),
                    "safety_rejection_rate": sum(
                        token in rejection_blob for token in ("safety", "security", "blocked")
                    )
                    / max(1, generated_count),
                    "difficulty": sum(r.predicted != r.gold for r in challenge_rows)
                    / max(1, len(challenge_rows)),
                    "teaching_gain": teaching_gain,
                    "label_counts": {label: len(accepted_by_label[label]) for label in all_labels},
                    "label_coverage_rate": sum(
                        bool(accepted_by_label[label]) for label in all_labels
                    )
                    / max(1, len(all_labels)),
                    "worst_label_agreement": min(
                        (
                            sum(
                                reviews_by_id[sample.sample_id].accepted
                                for sample in reviewed_by_label[label]
                            )
                            / len(reviewed_by_label[label])
                            for label in all_labels
                            if reviewed_by_label[label]
                        ),
                        default=0.0,
                    ),
                    "worst_label_probe_drop": worst_probe_drop,
                    "cost": (
                        generation_usage.cost + verifier_usage.cost
                        if generation_usage.cost is not None and verifier_usage.cost is not None
                        else None
                    ),
                    "cost_status": (
                        CostStatus.UNAVAILABLE.value
                        if generation_usage.cost is None or verifier_usage.cost is None
                        else (
                            CostStatus.ESTIMATED.value
                            if CostStatus.ESTIMATED
                            in {generation_usage.cost_status, verifier_usage.cost_status}
                            else CostStatus.ACTUAL.value
                        )
                    ),
                    "runtime_abstain": sum(
                        row.decision_origin == "runtime" for row in challenge_rows
                    ),
                },
                "probe_baseline": baseline.aggregate_metrics,
                "probe_evaluations": evaluations,
            }
        )
        return inner
