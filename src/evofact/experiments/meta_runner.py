from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from evofact.config import AppConfig
from evofact.core.models import (
    EpisodeEvaluation,
    EvaluationResult,
    EvolutionOperation,
    EvolutionProposal,
    MetaEvolutionOutcome,
    Sample,
    SampleEvaluation,
    SkillKind,
    SkillScope,
    SkillSpec,
    SkillStatus,
    Trigger,
    UsageRecord,
)
from evofact.data.domains import data_fingerprint, skillbank_fingerprint, split_source_and_final
from evofact.data.episodes import DomainEpisodeSampler
from evofact.evolution.firewall import CandidateFirewall
from evofact.evolution.identity import candidate_identity
from evofact.security.scanner import scan_resources
from evofact.skills.candidates import apply_candidate
from evofact.skills.repository import SkillRepository, _skill_from
from evofact.validation.evaluator import evaluate
from evofact.validation.meta_gate import MetaValidationGate
from evofact.validation.transfer import CrossEpisodeAggregator

from .checkpoint import MetaCheckpointStore
from .runner import ExperimentRunner


def fixture_meta_samples() -> list[Sample]:
    """函数作用：构造离线测试夹具 `fixture_meta_samples` 所表示的数据，供当前模块后续流程使用。
    输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
    输出：返回 `list[Sample]` 类型结果；校验或下游调用失败时异常向上传递。"""
    domains = ("health", "politics", "science", "finance", "disaster", "outer_holdout")
    rows = []
    for index, domain in enumerate(domains):
        rows.extend(
            [
                Sample(
                    f"{domain}-real",
                    "fixture",
                    f"{domain} 官方机构发布了可复核记录。",
                    "REAL",
                    domain,
                ),
                Sample(
                    f"{domain}-fake-cue",
                    "fixture",
                    f"{domain} 网传假消息已被辟谣。",
                    "FAKE",
                    domain,
                ),
                Sample(
                    f"{domain}-unverified",
                    "fixture",
                    f"{domain} 未经证实的高影响陈述 {index}。",
                    "FAKE",
                    domain,
                ),
            ]
        )
    return rows


class MetaEvolutionRunner:
    def __init__(
        self, config: AppConfig, project_root: Path, repository: SkillRepository | None = None
    ):
        """函数作用：创建并初始化 `MetaEvolutionRunner` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；`config`（AppConfig）需符合函数签名约定；`project_root`（Path）需符合函数签名约定；`repository`（SkillRepository | None，默认 `None`）需符合函数签名约定。
        输出：返回 `None`；初始化 `MetaEvolutionRunner` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config
        self.root = Path(project_root)
        self.base = ExperimentRunner(config, self.root)
        self.repository = repository
        if repository is not None:
            active = repository.active()
            if active:
                self.base.skills = list(active.values())

    async def run(
        self,
        samples: list[Sample],
        *,
        final_test_domains: tuple[str, ...] = (),
        resume: bool = False,
        evaluation_only: bool = False,
    ) -> MetaEvolutionOutcome:
        """函数作用：执行当前对象负责的主运行流程，并汇总本轮结果。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；`samples`（list[Sample]）需符合函数签名约定；`final_test_domains`（tuple[str, ...]，默认 `()`）需以关键字传入并符合签名约定；`resume`（bool，默认 `False`）需以关键字传入并符合签名约定；`evaluation_only`（bool，默认 `False`）需以关键字传入并符合签名约定。
        输出：异步返回 `MetaEvolutionOutcome` 类型结果；校验或下游调用失败时异常向上传递。"""
        if not self.config.meta_learning.enabled:  # 检查元学习功能是否开启
            raise ValueError("meta_learning.enabled must be true for DEMSE")
        source_domains, final_domains = split_source_and_final(samples, final_test_domains)
        from evofact.data.domains import effective_domain
        from evofact.data.manifests import sample_fingerprint

        owners = {}
        for sample in samples:
            domain = effective_domain(sample)
            if domain in source_domains and (
                sample.metadata.get("robustness_only") or sample.metadata.get("split") == "test"
            ):
                raise ValueError("test-only samples cannot enter DEMSE source domains")
            for key in [("content", sample_fingerprint(sample))] + (
                [("event", sample.event_id)] if sample.event_id else []
            ):
                if key in owners and owners[key] != domain:
                    raise ValueError("content or event crosses DEMSE domains")
                owners[key] = domain
        checkpoint_path = self.config.meta_learning.checkpoint_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.root / checkpoint_path
        if resume and checkpoint_path.exists() and self.repository is not None:
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            transaction = self.repository.transaction(raw.get("run_id", ""))
            if transaction:
                if (
                    skillbank_fingerprint(list(self.repository.active().values()))
                    != transaction["after"]
                ):
                    raise ValueError("repository changed after completed run")
                self.base.skills = [_skill_from(item) for item in raw["baseline_skills"]]
        data_id = data_fingerprint(samples)
        bank_id = skillbank_fingerprint(self.base.skills)
        episodes = DomainEpisodeSampler(self.config.meta_learning, self.config.seed).build(
            samples, source_domains, final_domains, bank_id
        )
        config_id = self._config_fingerprint()
        run_id = (
            "demse-"
            + hashlib.sha256(
                f"{config_id}:{data_id}:{bank_id}:{final_domains}".encode()
            ).hexdigest()[:16]
        )
        checkpoint_path = self.config.meta_learning.checkpoint_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.root / checkpoint_path
        store = MetaCheckpointStore(checkpoint_path)
        saved = (
            store.load(
                config_fingerprint=config_id,
                data_fingerprint=data_id,
                skillbank_snapshot_id=bank_id,
            )
            if resume
            else {}
        )
        self._restore_extra(saved)
        planned = [episode.episode_id for episode in episodes]
        if saved and (
            saved.get("planned_episode_ids") != planned
            or tuple(saved.get("final_test_domains", ())) != final_domains
        ):
            raise ValueError("checkpoint episode plan or final-test policy mismatch")
        completed = set(saved.get("completed_episode_ids", ()))
        if not completed <= set(planned):
            raise ValueError("checkpoint contains unplanned completed episodes")
        results = [_episode_result_from(item) for item in saved.get("episode_results", ())]
        proposals = {
            key: _proposal_from(value) for key, value in saved.get("candidates", {}).items()
        }
        by_id = {sample.sample_id: sample for sample in samples}
        firewall = CandidateFirewall(samples, final_domains)
        for episode in episodes:
            if episode.episode_id in completed:
                continue
            train_rows = [by_id[sample_id] for sample_id in episode.meta_train_sample_ids]
            test_rows = [by_id[sample_id] for sample_id in episode.meta_test_sample_ids]
            inner = await self._evolve_episode(train_rows, episode, firewall)
            allowed_traces = {trace.trace_id for trace in inner["traces"]}
            episode_candidates = set()
            for proposal in inner["proposals"]:
                if not set(proposal.source_trace_ids) <= allowed_traces:
                    raise ValueError("proposal contains a non-meta-train source trace")
                firewall.validate_candidate(proposal, episode)
                identity = candidate_identity(proposal)
                if identity.fingerprint in episode_candidates:
                    continue
                episode_candidates.add(identity.fingerprint)
                proposals.setdefault(identity.fingerprint, proposal)
                candidate_skills = _apply_candidate(self.base.skills, proposal)
                baseline_rows, candidate_rows = [], []
                for repeat_index in range(self.config.gate.repeats):
                    _, baseline = await self.base.run(test_rows)
                    _, candidate = await self.base.run(test_rows, skills=candidate_skills)
                    baseline_rows.extend(
                        replace(row, sample_id=f"{row.sample_id}:r{repeat_index}")
                        for row in baseline.per_sample
                    )
                    candidate_rows.extend(
                        replace(row, sample_id=f"{row.sample_id}:r{repeat_index}")
                        for row in candidate.per_sample
                    )
                baseline_result, candidate_result = (
                    evaluate(baseline_rows),
                    evaluate(candidate_rows),
                )
                aggregate_deltas = _metric_deltas(
                    baseline_result.aggregate_metrics, candidate_result.aggregate_metrics
                )
                domain_deltas = {
                    domain: _metric_deltas(
                        baseline_result.domain_metrics.get(domain, {}),
                        candidate_result.domain_metrics.get(domain, {}),
                    )
                    for domain in sorted(
                        set(baseline_result.domain_metrics) | set(candidate_result.domain_metrics)
                    )
                }
                levels = [
                    scan_resources(skill.resources).level for skill in proposal.candidate_skills
                ]
                safety = (
                    "blocked"
                    if "blocked" in levels
                    else ("review_required" if "review_required" in levels else "safe")
                )
                results.append(
                    EpisodeEvaluation(
                        episode.episode_id,
                        identity.fingerprint,
                        proposal.proposal_id,
                        baseline_result,
                        candidate_result,
                        aggregate_deltas,
                        domain_deltas,
                        safety,
                    )
                )
            completed.add(episode.episode_id)
            store.save(
                {
                    **self._checkpoint_extra(),
                    "schema_version": "meta_checkpoint_v1",
                    "baseline_skills": [asdict(s) for s in self.base.skills],
                    "run_id": run_id,
                    "config_fingerprint": config_id,
                    "data_fingerprint": data_id,
                    "skillbank_snapshot_id": bank_id,
                    "source_domains": source_domains,
                    "final_test_domains": final_domains,
                    "planned_episode_ids": [item.episode_id for item in episodes],
                    "completed_episode_ids": sorted(completed),
                    "episode_results": [asdict(item) for item in results],
                    "candidates": {key: asdict(value) for key, value in proposals.items()},
                    "committed": False,
                }
            )
        utilities = CrossEpisodeAggregator(
            self.config.seed,
            self.config.meta_learning.confidence_level,
            self.config.meta_learning.aggregate_across_episodes,
        ).aggregate(results)
        decisions = []
        for fingerprint, utility in utilities.items():
            safety_levels = {
                row.safety_level for row in results if row.candidate_fingerprint == fingerprint
            }
            safety = (
                "blocked"
                if "blocked" in safety_levels
                else ("review_required" if "review_required" in safety_levels else "safe")
            )
            proposal = proposals[fingerprint]
            decisions.append(
                MetaValidationGate(self.config.meta_learning).decide(
                    utility,
                    safety_level=safety,
                    retirement_candidate=proposal.operation == EvolutionOperation.RETIRE,
                )
            )
        committed = self._commit(tuple(decisions), proposals, run_id) if not evaluation_only else ()
        outcome = MetaEvolutionOutcome(
            run_id,
            episodes,
            tuple(results),
            utilities,
            tuple(decisions),
            committed,
            self.config.backend == "mock",
        )
        store.save(
            {
                **self._checkpoint_extra(),
                "schema_version": "meta_checkpoint_v1",
                "baseline_skills": [asdict(s) for s in self.base.skills],
                "run_id": run_id,
                "config_fingerprint": config_id,
                "data_fingerprint": data_id,
                "skillbank_snapshot_id": bank_id,
                "source_domains": source_domains,
                "final_test_domains": final_domains,
                "planned_episode_ids": [item.episode_id for item in episodes],
                "completed_episode_ids": sorted(completed),
                "episode_results": [asdict(item) for item in results],
                "candidates": {key: asdict(value) for key, value in proposals.items()},
                "committed": bool(committed),
            }
        )
        return outcome

    async def _evolve_episode(self, train_rows, episode, firewall):
        """函数作用：负责`MetaEvolutionRunner` 中的 `_evolve_episode` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；`train_rows`（未显式标注）需符合函数签名约定；`episode`（未显式标注）需符合函数签名约定；`firewall`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return await self.base.evolve_once(
            train_rows,
            generation_guard=lambda traces, reports: firewall.build_generation_view(
                episode, traces, reports
            ),
        )

    def _checkpoint_extra(self):
        """函数作用：负责`MetaEvolutionRunner` 中的 `_checkpoint_extra` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return {}

    def _restore_extra(self, saved):
        """函数作用：从检查点恢复 `_restore_extra` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；`saved`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
        pass

    def _commit(
        self, decisions, proposals: dict[str, EvolutionProposal], run_id: str
    ) -> tuple[str, ...]:
        """函数作用：原子提交 `_commit` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；`decisions`（未显式标注）需符合函数签名约定；`proposals`（dict[str, EvolutionProposal]）需符合函数签名约定；`run_id`（str）需符合函数签名约定。
        输出：返回 `tuple[str, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
        if self.repository is None:
            return ()
        previous = self.repository.transaction(run_id)
        if previous:
            return tuple(previous["snapshots"])
        bank = list(self.base.skills)
        changed = False
        touched = set()
        for decision in decisions:
            proposal = proposals[decision.candidate_fingerprint]
            if decision.disposition not in {"generalized", "specialized", "retired"}:
                for skill in proposal.candidate_skills:
                    if decision.disposition == "pareto":
                        self.repository.save(
                            replace(skill, status=SkillStatus.PARETO), "meta_pareto"
                        )
                continue
            conflict = set(proposal.target_skill_ids) | {s.name for s in proposal.candidate_skills}
            if touched & conflict:
                raise ValueError("accepted candidates conflict; joint evaluation required")
            touched.update(conflict)
            if decision.disposition == "specialized":
                proposal = replace(
                    proposal,
                    candidate_skills=tuple(
                        replace(s, scope=replace(s.scope, domains=decision.target_domains))
                        for s in proposal.candidate_skills
                    ),
                )
            bank = apply_candidate(bank, proposal)
            changed = True
        if not changed:
            return ()
        return self.repository.commit_bank(
            bank,
            run_id=run_id,
            baseline=self.base.skills,
            audit={
                "decisions": [asdict(d) for d in decisions],
                "proposals": {key: asdict(value) for key, value in proposals.items()},
            },
        )

    def _config_fingerprint(self) -> str:
        """函数作用：计算当前运行配置的稳定标识 `_config_fingerprint` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `MetaEvolutionRunner` 实例；无其他显式输入。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        payload = json.dumps(asdict(self.config), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


def _apply_candidate(skills: list[SkillSpec], proposal: EvolutionProposal) -> list[SkillSpec]:
    """函数作用：负责当前模块中的 `_apply_candidate` 处理，封装调用方需要复用的业务步骤。
    输入要求：`skills`（list[SkillSpec]）需符合函数签名约定；`proposal`（EvolutionProposal）需符合函数签名约定。
    输出：返回 `list[SkillSpec]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return apply_candidate(skills, proposal)


def _metric_deltas(old: dict[str, float], new: dict[str, float]) -> dict[str, float]:
    """函数作用：负责当前模块中的 `_metric_deltas` 处理，封装调用方需要复用的业务步骤。
    输入要求：`old`（dict[str, float]）需符合函数签名约定；`new`（dict[str, float]）需符合函数签名约定。
    输出：返回 `dict[str, float]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return {
        key: float(new.get(key, 0.0)) - float(old.get(key, 0.0))
        for key in sorted(set(old) | set(new))
    }


def _evaluation_from(data: dict) -> EvaluationResult:
    """函数作用：负责当前模块中的 `_evaluation_from` 处理，封装调用方需要复用的业务步骤。
    输入要求：`data`（dict）需符合函数签名约定。
    输出：返回 `EvaluationResult` 类型结果；校验或下游调用失败时异常向上传递。"""
    rows = tuple(SampleEvaluation(**item) for item in data.get("per_sample", ()))
    usage = UsageRecord(**data.get("usage", {}))
    cis = {key: tuple(value) for key, value in data.get("confidence_intervals", {}).items()}
    return EvaluationResult(
        rows,
        data.get("aggregate_metrics", {}),
        data.get("domain_metrics", {}),
        data.get("temporal_metrics", {}),
        cis,
        usage,
    )


def _episode_result_from(data: dict) -> EpisodeEvaluation:
    """函数作用：负责当前模块中的 `_episode_result_from` 处理，封装调用方需要复用的业务步骤。
    输入要求：`data`（dict）需符合函数签名约定。
    输出：返回 `EpisodeEvaluation` 类型结果；校验或下游调用失败时异常向上传递。"""
    return EpisodeEvaluation(
        data["episode_id"],
        data["candidate_fingerprint"],
        data["proposal_id"],
        _evaluation_from(data["baseline_result"]),
        _evaluation_from(data["candidate_result"]),
        data.get("aggregate_deltas", {}),
        data.get("domain_deltas", {}),
        data.get("safety_level", "safe"),
        data.get("schema_version", "episode_evaluation_v1"),
    )


def _proposal_from(data: dict) -> EvolutionProposal:
    """函数作用：负责当前模块中的 `_proposal_from` 处理，封装调用方需要复用的业务步骤。
    输入要求：`data`（dict）需符合函数签名约定。
    输出：返回 `EvolutionProposal` 类型结果；校验或下游调用失败时异常向上传递。"""
    candidates = []
    for raw in data.get("candidate_skills", ()):
        scope = SkillScope(**{key: tuple(value) for key, value in raw.get("scope", {}).items()})
        triggers = tuple(Trigger(**item) for item in raw.get("triggers", ()))
        candidates.append(
            SkillSpec(
                raw["skill_id"],
                raw["name"],
                SkillKind(raw["kind"]),
                raw["version"],
                SkillStatus(raw["status"]),
                raw["instructions"],
                raw.get("resources", {}),
                scope,
                triggers,
                tuple(raw.get("parent_ids", ())),
                raw.get("safety_level", "text_only"),
                raw.get("schema_version", "skill_spec_v1"),
            )
        )
    return EvolutionProposal(
        data["proposal_id"],
        EvolutionOperation(data["operation"]),
        data["rationale"],
        tuple(data.get("target_skill_ids", ())),
        tuple(candidates),
        tuple(data.get("source_trace_ids", ())),
        data.get("error_cluster_id"),
        tuple(data.get("risk_flags", ())),
    )
