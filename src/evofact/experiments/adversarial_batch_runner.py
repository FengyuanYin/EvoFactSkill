"""Independent adversarial route: initial challenges and bounded failure-driven supplements."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from evofact.core.models import Evidence, Sample
from evofact.evolution.optimizer_context import build_skillbank_optimizer_context
from evofact.generation.data import validate_fact_links
from evofact.generation.verifier import ChallengeVerifier, normalized
from evofact.generation.workflow import GenerationWorkflow

from .meta_batch_runner import MetaBatchRunner, _json


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
        self.verifier = ChallengeVerifier()

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
            "generator_digest": self.workflow.package.package_digest,
            "max_supplement_rounds": self.max_supplement_rounds,
            "facts": _json([asdict(f) for f in self.facts]),
        }

    def _reset_batch(self, state):
        state.pop("adversarial_batch", None)

    def _request(self, state, batch, round_id):
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
        request = {
            "batch_size": cfg.batch_size,
            "id_prefix": f"gen-{round_id}-",
            "examples": examples,
            "skills": context["skills"],
            "validation_feedback": context.get("validation_feedback", []),
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
                request = self._request(state, batch, current["round_id"])
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
                        "examples and their training-domain evidence. Never reproduce feedback samples "
                        "or access unused validation/final-test data. Return fewer samples when evidence is insufficient."
                    ),
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
                structured, rejected = self.verifier.validate(
                    current["response"],
                    current["request"],
                    config=self.config.generation,
                    forbidden=forbidden,
                    existing=existing,
                    label_contract_registry=self.base.label_contract_registry,
                    allowed_strategies=self.workflow.strategies,
                )
                if structured:
                    verified = await self.verifier.verify(
                        structured,
                        backend,
                        budget_manager=self.base._budget_manager(),
                        budget_sample_id=current["round_id"] + ":verify",
                        label_contract_registry=self.base.label_contract_registry,
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
            if current["traces"] is None:
                accepted = [sample_from_dict(s) for s in current["verification"]["accepted"]]
                if accepted:
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
            "generation_rounds": audit,
            "accepted_challenge_samples": sum(len(r["verification"]["accepted"]) for r in audit),
            "max_supplement_rounds": self.max_supplement_rounds,
        }
