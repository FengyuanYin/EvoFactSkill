"""Trace-conditioned, one-shot LLM sample generation within DEMSE."""

import hashlib
import json
from dataclasses import asdict

from evofact.attribution.rules import attribute_trace
from evofact.core.models import DomainEpisode
from evofact.data.domains import skillbank_fingerprint
from evofact.evolution.firewall import CandidateFirewall
from evofact.generation.data import validate_fact_links
from evofact.generation.generator import ChallengeGenerator, json_value
from evofact.generation.prompts import GENERATOR_SYSTEM, VERIFIER_SYSTEM
from evofact.generation.split import split_construction_probe
from evofact.generation.verifier import ChallengeVerifier, normalized
from evofact.security.scanner import scan_resources
from evofact.skills.candidates import apply_candidate

from .checkpoint import MetaCheckpointStore
from .meta_runner import MetaEvolutionRunner
from .runner import _gold


def fingerprint(value):
    """函数作用：负责当前模块中的 `fingerprint` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def false_positive_rate(result):
    """函数作用：负责当前模块中的 `false_positive_rate` 处理，封装调用方需要复用的业务步骤。
    输入要求：`result`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    real = [row for row in result.per_sample if row.gold == "REAL"]
    if not real:
        raise ValueError("probe must contain true-news examples to measure false positives")
    return sum(row.predicted == "FAKE" for row in real) / len(real)


class AdversarialEvolutionRunner(MetaEvolutionRunner):
    def __init__(self, config, project_root, facts=(), repository=None, *, generation_backend=None):
        """函数作用：创建并初始化 `AdversarialEvolutionRunner` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`config`（未显式标注）需符合函数签名约定；`project_root`（未显式标注）需符合函数签名约定；`facts`（未显式标注，默认 `()`）需符合函数签名约定；`repository`（未显式标注，默认 `None`）需符合函数签名约定；`generation_backend`（未显式标注，默认 `None`）需以关键字传入并符合签名约定。
        输出：返回 `None`；初始化 `AdversarialEvolutionRunner` 的实例状态，构造参数非法时可能抛出异常。"""
        super().__init__(config, project_root, repository)
        self.facts = list(facts)
        self.audit = {}
        self.generator = ChallengeGenerator()
        self.verifier = ChallengeVerifier()
        # Dependency injection is for tests; CLI has no deterministic generation path.
        self.generation_backend = generation_backend
        self.resume_generation = False

    async def run(self, samples, *, final_test_domains=(), resume=False, evaluation_only=False):
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
            if stored.get("schema_version") != "generation_audit_v2":
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
                "schema": "one_shot_llm_v2",
                "config": asdict(self.config),
                "generator_system": GENERATOR_SYSTEM,
                "verifier_system": VERIFIER_SYSTEM,
                "facts": sorted((asdict(f) for f in self.facts), key=lambda f: f["fact_id"]),
            }
        )

    def _checkpoint_extra(self):
        """函数作用：负责`AdversarialEvolutionRunner` 中的 `_checkpoint_extra` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return {"generation_schema": "one_shot_llm_v2", "generation_episodes": self.audit}

    def _restore_extra(self, saved):
        """函数作用：从检查点恢复 `_restore_extra` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`saved`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
        self.audit = saved.get("generation_episodes", {})
        if saved and (
            saved.get("generation_schema") != "one_shot_llm_v2"
            or set(self.audit) != set(saved.get("completed_episode_ids", ()))
        ):
            raise ValueError("generation checkpoint episode audit is incompatible or incomplete")

    def _commit_generation(self, path, run_id):
        """函数作用：原子提交 `_commit_generation` 所表示的数据，供当前模块后续流程使用。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`path`（未显式标注）需符合函数签名约定；`run_id`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
        current = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"schema_version": "generation_audit_v2", "runs": {}}
        )
        record = json_value({"prompt_hash": fingerprint(GENERATOR_SYSTEM), "episodes": self.audit})
        if run_id in current["runs"]:
            if current["runs"][run_id] != record:
                raise ValueError(
                    "generation audit differs for an existing run; use --resume or a new output path"
                )
            return
        current["runs"][run_id] = record
        MetaCheckpointStore(path).save(current)

    async def _evolve_episode(self, train_rows, episode, firewall):
        """函数作用：负责`AdversarialEvolutionRunner` 中的 `_evolve_episode` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `AdversarialEvolutionRunner` 实例；`train_rows`（未显式标注）需符合函数签名约定；`episode`（未显式标注）需符合函数签名约定；`firewall`（未显式标注）需符合函数签名约定。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        cfg = self.config.generation
        allowed = {s.sample_id for s in train_rows}
        train_facts = [f for f in self.facts if f.sample_id in allowed]
        construction, probe, facts = split_construction_probe(
            train_rows, train_facts, fraction=cfg.probe_fraction, seed=episode.episode_id
        )
        if not {"REAL", "FAKE"} <= {_gold(s.label) for s in probe}:
            raise ValueError("probe needs both REAL and FAKE labels")
        frozen_id = skillbank_fingerprint(self.base.skills)
        backend = self.generation_backend or self.base._backend()
        # A persisted response can be reused after interruption during verification/evolution.
        key = fingerprint(
            {
                "config": self._config_fingerprint(),
                "episode": asdict(episode),
                "construction": [asdict(s) for s in construction],
                "bank": frozen_id,
            }
        )
        cache_path = (
            (self.root / self.config.meta_learning.checkpoint_path).parent
            / "generation-cache"
            / f"{key}.json"
        )
        if self.resume_generation and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("input_fingerprint") != key:
                raise ValueError("generation response cache mismatch")
            request, response = cached["request"], cached["response"]
        else:
            traces, _ = await self.base.run(construction)
            by_id = {s.sample_id: s for s in construction}
            reports = [
                attribute_trace(t, _gold(by_id[t.sample_public["sample_id"]].label)) for t in traces
            ]
            firewall.build_generation_view(episode, traces, reports)
            request = self.generator.build_request(
                construction, traces, reports, config=cfg, episode_id=episode.episode_id
            )
            response = await self.generator.generate(backend, request)
            MetaCheckpointStore(cache_path).save(
                {"input_fingerprint": key, "request": request, "response": response}
            )
        forbidden = probe + [
            s
            for s in firewall.samples.values()
            if s.sample_id in episode.meta_test_sample_ids
            or (s.domain or s.dataset) in episode.final_test_domains
        ]
        structured, rejected = self.verifier.validate(
            response, request, config=cfg, forbidden=forbidden, existing=construction
        )
        synthetic, review_rejections, reviews = await self.verifier.verify(structured, backend)
        rejected.extend(review_rejections)
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
        inner = await self.base.evolve_once(
            rows,
            generation_guard=lambda traces, reports: inner_firewall.build_generation_view(
                inner_episode, traces, reports
            ),
        )
        # Probe stays diagnostic. Neither its texts nor its metrics return to the generator.
        _, baseline = await self.base.run(probe)
        evaluations = []
        for proposal in inner["proposals"]:
            firewall.validate_candidate(proposal, episode)
            blob = json.dumps(asdict(proposal), ensure_ascii=False)
            if any(s.sample_id in blob or s.text in blob for s in probe):
                raise ValueError("candidate contains probe content")
            if any(scan_resources(s.resources).level != "safe" for s in proposal.candidate_skills):
                continue
            _, result = await self.base.run(
                probe, skills=apply_candidate(self.base.skills, proposal)
            )
            evaluations.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "metrics": result.aggregate_metrics,
                    "teaching_gain": result.aggregate_metrics["macro_f1_all"]
                    - baseline.aggregate_metrics["macro_f1_all"],
                    "false_positive_rate": false_positive_rate(result),
                }
            )
        challenge_ids = {s.sample_id for s in synthetic}
        challenge_rows = [r for r in inner["evaluation"].per_sample if r.sample_id in challenge_ids]
        if skillbank_fingerprint(self.base.skills) != frozen_id:
            raise RuntimeError("detector baseline mutated during generation evaluation")
        self.audit[episode.episode_id] = json_value(
            {
                "construction_ids": [s.sample_id for s in construction],
                "probe_ids": [s.sample_id for s in probe],
                "fact_ids": [f.fact_id for f in facts],
                "detector_baseline": frozen_id,
                "generator_calls": 1,
                "verifier_calls": int(bool(structured)),
                "prompt_hash": fingerprint(GENERATOR_SYSTEM),
                "request": request,
                "response": response,
                "samples": [asdict(s) for s in synthetic],
                "rejected": rejected,
                "reviews": reviews,
                "metrics": {
                    "generated": len(response["samples"]),
                    "accepted": len(synthetic),
                    "validity": len(synthetic) / max(1, len(response["samples"])),
                    "difficulty": sum(r.predicted != r.gold for r in challenge_rows)
                    / max(1, len(challenge_rows)),
                    "label_counts": {
                        label: sum(s.label == label for s in synthetic)
                        for label in ("REAL", "FAKE")
                    },
                },
                "probe_baseline": baseline.aggregate_metrics,
                "probe_evaluations": evaluations,
            }
        )
        return inner
