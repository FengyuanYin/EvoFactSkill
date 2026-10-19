"""Independent adversarial route: initial challenges and bounded failure-driven supplements."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime

from evofact.core.models import Evidence, Sample
from evofact.evolution.firewall import validate_generation_request
from evofact.evolution.optimizer_context import build_skillbank_optimizer_context
from evofact.generation.data import validate_fact_links
from evofact.generation.memory import (
    GENERATOR_MEMORY_POLICY,
    GeneratorMemoryBackend,
    bind_memory,
    build_memory_view,
    historical_samples,
    record_round,
)
from evofact.generation.prompts import KNOWLEDGE_GENERATOR_POLICY, KNOWLEDGE_VERIFIER_SYSTEM
from evofact.generation.verifier import ChallengeVerifier, normalized
from evofact.generation.workflow import GenerationWorkflow
from evofact.skills.package_adapter import package_bank_digest
from evofact.validation.independent import InferenceIncomplete

from .meta_batch_runner import MetaBatchRunner, _json

LOGGER = logging.getLogger(__name__)


def sample_from_dict(raw):
    data = dict(raw)
    evidence = []
    for item in data.pop("evidence", []):
        item = dict(item)
        if item.get("published_at"):
            item["published_at"] = datetime.fromisoformat(item["published_at"])
        evidence.append(Evidence(**item))
    if data.get("published_at"):
        data["published_at"] = datetime.fromisoformat(data["published_at"])
    return Sample(**data, evidence=tuple(evidence))


class AdversarialBatchRunner(MetaBatchRunner):
    mode = "adversarial-evolve"
    checkpoint_schema = "adversarial_batch_checkpoint_v1"
    run_prefix = "adversarial-batch-"

    def __init__(
        self,
        config,
        project_root,
        facts=(),
        repository=None,
        *,
        generation_backend=None,
        max_supplement_rounds=2,
    ):
        if not config.generation.enabled or not config.generation.use_generated_in_training:
            raise ValueError(
                "adversarial-evolve requires generation and generated training enabled"
            )
        if config.generation.proposer != "llm":
            raise ValueError("batch adversarial-evolve requires an LLM generator")
        if max_supplement_rounds < 0:
            raise ValueError("max_supplement_rounds must be non-negative")
        if config.backend == "mock" and generation_backend is None:
            raise ValueError(
                "adversarial verification requires a real backend or an injected test backend"
            )
        super().__init__(config, project_root, repository)
        self.facts = list(facts)
        self.generation_backend = generation_backend
        self.max_supplement_rounds = max_supplement_rounds
        generator = next(p for p in self.base.packages if p.manifest.name == "generation_agent")
        self.workflow = GenerationWorkflow(generator)
        self.verifier = ChallengeVerifier(mode="knowledge")

    async def run(self, samples, **kwargs):
        if not kwargs.get("final_test_domains"):
            raise ValueError("adversarial evolution requires explicit final-test domains")
        validate_fact_links(samples, self.facts)
        if any(not f.verified for f in self.facts):
            raise ValueError("generation facts must be verified")
        owners = {}
        for sample in samples:
            domain = sample.domain or sample.dataset
            for evidence in sample.evidence:
                keys = [("text", normalized(evidence.text))]
                if evidence.source:
                    keys.append(("source", evidence.source))
                for key in keys:
                    if owners.setdefault(key, domain) != domain:
                        raise ValueError("evidence source/content crosses adversarial domains")
                if (
                    sample.published_at
                    and evidence.published_at
                    and evidence.published_at > sample.published_at
                ):
                    raise ValueError("future evidence cannot support adversarial samples")
            if domain not in kwargs["final_test_domains"] and (
                str(sample.metadata.get("official_split", "")).casefold() == "test"
                or sample.metadata.get("split_role") == "test"
            ):
                raise ValueError("test-only samples cannot enter adversarial source domains")
        return await super().run(samples, **kwargs)

    def _identity_extra(self):
        return {
            "route": self.mode,
            "generation_policy": "endogenous-knowledge-v1",
            "generation_policy_digest": hashlib.sha256(
                (KNOWLEDGE_GENERATOR_POLICY + KNOWLEDGE_VERIFIER_SYSTEM).encode()
            ).hexdigest(),
            "generator_digest": self.workflow.package.package_digest,
            "max_supplement_rounds": self.max_supplement_rounds,
            "facts": _json([asdict(f) for f in self.facts]),
            **(
                {
                    "generator_memory_policy_digest": hashlib.sha256(
                        GENERATOR_MEMORY_POLICY.encode()
                    ).hexdigest()
                }
                if self.config.generator_memory.enabled
                else {}
            ),
        }

    def _reset_batch(self, state):
        state.pop("adversarial_batch", None)

    async def _prepare_route_runtime(self, state, save, budget):
        if self.config.generator_memory.enabled:
            bind_memory(state, self.workflow.package.package_digest)
            await GeneratorMemoryBackend(
                self.generation_backend or self.base._backend(),
                self.config.generator_memory,
                state,
                save,
                round_id=state["run_id"] + ":preflight",
                budget=budget,
            ).prepare()

    def _validate_response(self, current, forbidden, existing):
        response = current["response"]
        overflow = []
        limit = self.config.generation.batch_size
        if isinstance(response, dict) and set(response) == {"samples", "decisions"}:
            rows, decisions = response["samples"], response["decisions"]
            if isinstance(rows, list) and isinstance(decisions, list) and len(rows) > limit:
                sample_ids = [r.get("sample_id") for r in rows if isinstance(r, dict)]
                decision_ids = [d.get("sample_id") for d in decisions if isinstance(d, dict)]
                # Truncate only complete, unique, one-to-one pairs. The strict
                # verifier still checks provenance, labels and every selected row.
                if (
                    len(sample_ids) == len(rows) == len(decisions) == len(decision_ids)
                    and all(isinstance(s, str) for s in sample_ids + decision_ids)
                    and len(set(sample_ids)) == len(rows)
                    and len(set(decision_ids)) == len(decisions)
                    and set(sample_ids) == set(decision_ids)
                ):
                    selected_ids = set(sample_ids[:limit])
                    response = {
                        "samples": rows[:limit],
                        "decisions": [d for d in decisions if d["sample_id"] in selected_ids],
                    }
                    overflow = [
                        {"index": i, "reason": "generator batch budget exceeded"}
                        for i in range(limit, len(rows))
                    ]
                    LOGGER.warning(
                        "Generator returned %s samples; validating first %s paired candidates.",
                        len(rows),
                        limit,
                    )
        try:
            structured, rejected = self.verifier.validate(
                response,
                current["request"],
                config=self.config.generation,
                forbidden=forbidden,
                existing=existing,
                label_contract_registry=self.base.label_contract_registry,
                allowed_strategies=self.workflow.strategies,
            )
        except ValueError as exc:
            LOGGER.warning("Rejecting malformed generator response: %s", exc)
            return [], [{"index": None, "reason": f"invalid generator response: {exc}"}]
        return structured, rejected + overflow

    def _request(self, state, batch, round_id, *, forbidden=()):
        cfg = self.config.generation
        registry = self.base.label_contract_registry
        by_id = {s.sample_id: s for s in batch}
        examples, contracts = [], {}
        for trace in state["train_traces"][: cfg.max_trace_examples]:
            source = by_id[trace["sample_id"]]
            contract = registry.resolve_sample(source)
            contracts[contract.key] = contract
            row = _json(asdict(source))
            row.update(label=contract.normalize(source.label), metadata={})
            examples.append(
                {
                    "source_sample": row,
                    "trace": trace,
                    "label_contract": contract.prompt_view(),
                    "label_contract_digest": contract.digest,
                }
            )
        # This context contains no rule attribution, clustering, or counterfactual replay.
        context = build_skillbank_optimizer_context(
            packages=self.base.packages,
            traces=state["train_traces"],
            feedback=list(reversed(state["feedback"])) or None,
            max_text_chars=self.config.evolution.max_text_chars,
            max_instructions_chars=self.config.evolution.max_instructions_chars,
            max_total_chars=self.config.evolution.max_total_chars,
        )
        feedback = []
        removed_traces = removed_entries = 0
        for item in context.get("validation_feedback", []):
            # Consumed feedback can contain near-duplicates of still-held-out
            # samples. Remove those traces, not the held-out protection itself.
            cleaned = dict(item)
            for key in ("baseline_traces", "candidate_traces"):
                if key not in cleaned:
                    continue
                safe_traces = []
                for trace in cleaned[key]:
                    try:
                        validate_generation_request(trace, forbidden)
                    except ValueError:
                        removed_traces += 1
                    else:
                        safe_traces.append(trace)
                cleaned[key] = safe_traces
            try:
                validate_generation_request(cleaned, forbidden)
            except ValueError:
                removed_entries += 1
            else:
                feedback.append(cleaned)
        request = {
            "generation_policy": "endogenous-knowledge-v1",
            "batch_size": cfg.batch_size,
            "id_prefix": f"gen-{round_id}-",
            "examples": examples,
            "skills": context["skills"],
            "validation_feedback": feedback,
            "feedback_filter": {
                "removed_traces": removed_traces,
                "removed_entries": removed_entries,
            },
            "label_contracts": [
                {
                    **c.prompt_view(),
                    "digest": c.digest,
                    "generation_guidance": {
                        label.name: label.generation_guidance for label in c.labels
                    },
                }
                for c in contracts.values()
            ],
            "construction_facts": [asdict(f) for f in self.facts if f.sample_id in by_id],
        }
        if self.config.generator_memory.enabled:
            request["generation_memory"] = build_memory_view(
                state,
                batch,
                bank_digest=package_bank_digest(self.base.packages),
                forbidden=forbidden,
            )
            return request  # Token management includes workflow instructions/resources below.
        while (
            len(json.dumps(request, ensure_ascii=False, default=str))
            > self.config.evolution.max_total_chars
            and len(request["examples"]) > 1
        ):
            request["examples"].pop()
        if (
            len(json.dumps(request, ensure_ascii=False, default=str))
            > self.config.evolution.max_total_chars
        ):
            raise ValueError("generator request cannot fit evolution.max_total_chars")
        return request

    async def _optimizer_traces(self, state, batch, forbidden, save, *, progress, task):
        rounds = state.setdefault("adversarial_batch", [])
        failures = len(state["feedback"])
        # One initial round, then at most one supplement per failed validation.
        target = 1 + min(failures, self.max_supplement_rounds)
        while len(rounds) < target or (rounds and not rounds[-1].get("complete")):
            if not rounds or rounds[-1].get("complete"):
                rounds.append(
                    {
                        "round_id": f"{state['run_id']}-b{state['completed_batches'] + 1}-r{len(rounds)}",
                        "batch": state["completed_batches"] + 1,
                        "kind": "initial" if not rounds else "supplement",
                        "after_failures": failures,
                        "response": None,
                        "verification": None,
                        "traces": None,
                        "complete": False,
                    }
                )
                save()
            current = rounds[-1]
            backend = self.generation_backend or self.base._backend()
            if current["response"] is None:
                if self.config.generator_memory.enabled and current.get("request") is not None:
                    request = current["request"]
                else:
                    request = self._request(state, batch, current["round_id"], forbidden=forbidden)
                    if self.config.generator_memory.enabled:
                        current["request"] = request
                        save()
                managed = None
                if self.config.generator_memory.enabled:
                    managed = GeneratorMemoryBackend(
                        backend,
                        self.config.generator_memory,
                        state,
                        save,
                        round_id=current["round_id"],
                        budget=self.base._budget_manager(),
                        forbidden=forbidden,
                    )
                result = await self.workflow.run(
                    backend,
                    (),
                    (),
                    (),
                    config=self.config.generation,
                    episode_id=current["round_id"],
                    budget_manager=self.base._budget_manager(),
                    label_contract_registry=self.base.label_contract_registry,
                    forbidden=forbidden,
                    request=request,
                    system_addendum=(
                        "\nFor this batch route, diagnose directly from supplied traces and skills; "
                        "there are no precomputed attributions. Consumed validation feedback is allowed "
                        "only to identify capabilities to challenge. Construct every sample from "
                        "labeled training examples and endogenous knowledge. Never reproduce feedback samples "
                        "or access unused validation/final-test data.\n"
                        + KNOWLEDGE_GENERATOR_POLICY
                        + (GENERATOR_MEMORY_POLICY if managed else "")
                    ),
                    request_executor=managed.execute if managed else None,
                )
                current.update(
                    request=result.request,
                    response=result.response,
                    request_digest=result.request_digest,
                    response_digest=result.response_digest,
                    generator_usage=_json(asdict(result.usage)),
                )
                save()
            if current["verification"] is None:
                existing = batch + [
                    sample_from_dict(s)
                    for r in rounds[:-1]
                    for s in r.get("verification", {}).get("accepted", [])
                ]
                if self.config.generator_memory.enabled:
                    existing += [sample_from_dict(s) for s in historical_samples(state)]
                structured, rejected = self._validate_response(current, forbidden, existing)
                if structured:
                    verified = await self.verifier.verify(
                        structured,
                        backend,
                        budget_manager=self.base._budget_manager(),
                        budget_sample_id=current["round_id"] + ":verify",
                        label_contract_registry=self.base.label_contract_registry,
                        construction_request={
                            "examples": current["request"]["examples"],
                            "decisions": current["response"]["decisions"],
                        },
                    )
                    current["verification"] = {
                        "accepted": _json([asdict(s) for s in verified.accepted]),
                        "rejected": _json(rejected + verified.rejected),
                        "reviews": _json(verified.reviews),
                        "usage": _json(asdict(verified.usage)),
                    }
                else:
                    current["verification"] = {
                        "accepted": [],
                        "rejected": rejected,
                        "reviews": [],
                        "usage": {},
                    }
                save()
            if self.config.generator_memory.enabled:
                record_round(
                    state,
                    current,
                    bank_digest=package_bank_digest(self.base.packages),
                    generator_digest=self.workflow.package.package_digest,
                )
                save()
            if current["traces"] is None:
                accepted = [sample_from_dict(s) for s in current["verification"]["accepted"]]
                if accepted:
                    if self.config.independent_validation.enabled:
                        clean = await self._clean_validation(
                            current,
                            "challenge",
                            accepted,
                            self.base.skills,
                            save,
                            progress=progress,
                            task=task,
                            phase="adversarial " + current["kind"],
                        )
                        if not clean:
                            raise InferenceIncomplete("challenge inference has runtime failures")
                        current["traces"] = current["challenge"]["traces"]
                        current["metrics"] = current["challenge"]["result"]["aggregate_metrics"]
                    else:
                        traces, evaluation = await self.base.run(
                            accepted,
                            progress=progress,
                            task_name=task,
                            phase="adversarial " + current["kind"],
                            isolate_sample_budget=True,
                        )
                        current["traces"] = _json([asdict(t) for t in traces])
                        current["metrics"] = _json(evaluation.aggregate_metrics)
                else:
                    current["traces"] = []
                    current["metrics"] = {}
                save()
            current["complete"] = True
            if self.config.generator_memory.enabled:
                record_round(
                    state,
                    current,
                    bank_digest=package_bank_digest(self.base.packages),
                    generator_digest=self.workflow.package.package_digest,
                )
            state.setdefault("generation_audit", []).append(_json(current))
            save()
        # Prioritize fresh challenge traces within the existing context size limits.
        return (
            state["train_traces"][:1]
            + [t for r in reversed(rounds) for t in r["traces"]]
            + state["train_traces"][1:]
        )

    def _extra_outcome(self, state):
        audit = state.get("generation_audit", [])
        return {
            "generation_policy": "endogenous-knowledge-v1",
            "generation_rounds": audit,
            "accepted_challenge_samples": sum(len(r["verification"]["accepted"]) for r in audit),
            "max_supplement_rounds": self.max_supplement_rounds,
            **(
                {
                    "generator_memory": {
                        "rounds": len(state.get("generator_memory", {}).get("rounds", [])),
                        "requests": len(state.get("generator_requests", {})),
                        "error": state.get("generator_context_error"),
                    }
                }
                if self.config.generator_memory.enabled
                else {}
            ),
        }
