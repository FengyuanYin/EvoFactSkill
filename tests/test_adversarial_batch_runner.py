import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from optimizer_stub import EditingOptimizerBackend

from evofact.config import load_config
from evofact.core.models import Evidence, Sample
from evofact.evolution.firewall import validate_generation_request
from evofact.experiments.adversarial_batch_runner import AdversarialBatchRunner
from evofact.experiments.meta_batch_runner import MetaBatchRunner
from evofact.experiments.route_isolation import evolution_route_config, route_artifact_path
from evofact.generation.prompts import KNOWLEDGE_VERIFIER_SYSTEM, VERIFIER_SYSTEM
from evofact.skills.package_adapter import package_bank_digest
from evofact.skills.repository import SkillRepository

ROOT = Path(__file__).resolve().parents[1]


class Generator:
    def __init__(self, *, reject=False):
        self.requests = []
        self.verifications = []
        self.reject = reject

    async def _call(self, system, payload):
        if system in {VERIFIER_SYSTEM, KNOWLEDGE_VERIFIER_SYSTEM}:
            self.verifications.append(deepcopy(payload))
            return {
                "reviews": [
                    {
                        "sample_id": s["sample_id"],
                        "label": ("FAKE" if self.reject else "REAL")
                        if " REAL " in s["text"]
                        else ("REAL" if self.reject else "FAKE"),
                        "reason": "fixture independent review",
                    }
                    for s in payload["samples"]
                ]
            }
        self.requests.append(deepcopy(payload))
        rows, decisions = [], []
        for i in range(payload["batch_size"]):
            example = payload["examples"][i % len(payload["examples"])]
            if "generation_plan" in payload:
                source_id = payload["generation_plan"]["slots"][i]["source_sample_id"]
                example = next(
                    e for e in payload["examples"] if e["source_sample"]["sample_id"] == source_id
                )
            source = example["source_sample"]
            row = deepcopy(source)
            row.update(
                sample_id=payload["id_prefix"] + str(i),
                text=f"Challenge {source['label']} case {payload['id_prefix']}{i}",
            )
            rows.append(row)
            decisions.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sample_id": source["sample_id"],
                    "strategy": "label_preserving_rewrite",
                    "reason": "fixture challenge",
                    "source_trace_ids": [example["trace"]["trace_id"]],
                }
            )
        return {"samples": rows, "decisions": decisions}


def setup(tmp_path, monkeypatch, *, max_supplement_rounds=2, reject=False, repository=None):
    config = load_config(ROOT / "configs/adversarial_llm.yaml")
    config = replace(
        config,
        backend="mock",
        model="mock-v1",
        output_dir=tmp_path / "reports",
        skill_store=tmp_path / "skills",
        data=replace(config.data, evolution_validation_samples=2),
        execution=replace(config.execution, batch_size=2),
        generation=replace(config.generation, batch_size=2),
        meta_learning=replace(config.meta_learning, checkpoint_path=tmp_path / "checkpoint.json"),
    )
    backend = Generator(reject=reject)
    runner = AdversarialBatchRunner(
        config,
        ROOT,
        repository=repository,
        generation_backend=backend,
        max_supplement_rounds=max_supplement_rounds,
    )
    contexts, calls = [], []

    class Optimizer(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(deepcopy(context))
            result = await super().optimize_package(context, skill)
            result.value["file_operations"][0]["content"] += f"Revision {len(contexts)}.\n"
            return result

    monkeypatch.setattr(runner.base, "_optimizer_backend", Optimizer)
    samples = [
        Sample(
            f"{d}-{i}",
            "fixture",
            f"Original sample {d} {i}.",
            "REAL" if i % 2 else "FAKE",
            d,
            evidence=(Evidence(f"Evidence from {d} {i}.", f"fixture://{d}/{i}"),),
        )
        for d, n in (("train", 4), ("valid", 8), ("final", 2))
        for i in range(n)
    ]
    original = runner.base.run

    async def run(rows, **kwargs):
        calls.append((kwargs["phase"], [s.sample_id for s in rows]))
        traces, result = await original(rows, **kwargs)
        if kwargs["phase"].startswith("validation"):
            score = (
                0.75 if kwargs["phase"] == "validation candidate" and len(contexts) >= 2 else 0.5
            )
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": score}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", run)
    return runner, backend, contexts, calls, samples


def invoke(runner, samples, **kwargs):
    return asyncio.run(
        runner.run(samples, final_test_domains=("final",), validation_domains=("valid",), **kwargs)
    )


def test_initial_supplement_feedback_and_reset(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    result = invoke(runner, samples)
    assert result["status"] == "completed" and result["mode"] == "adversarial-evolve"
    assert [r["kind"] for r in result["generation_rounds"]] == ["initial", "supplement", "initial"]
    assert result["accepted_challenge_samples"] == 6
    assert [r["batch"] for r in result["generation_rounds"]] == [1, 1, 2]
    assert len(contexts[0]["traces"]) == 4
    assert len(contexts[1]["traces"]) == 6
    assert len(contexts[2]["traces"]) == 4
    assert all(context["feedback_enabled"] is True for context in contexts)
    assert contexts[1]["validation_feedback"]
    assert not contexts[2].get("validation_feedback")
    assert backend.requests[1]["validation_feedback"]
    assert not backend.requests[2]["validation_feedback"]
    assert all("attribution" not in e for r in backend.requests for e in r["examples"])
    assert all(
        e["source_sample"]["domain"] == "train" for r in backend.requests for e in r["examples"]
    )
    assert not any(s.startswith("final-") for _, ids in calls for s in ids)
    baseline = [ids for phase, ids in calls if phase == "validation baseline"]
    assert len(set(sum(baseline, []))) == 6


def test_generator_overflow_keeps_only_bounded_paired_candidates(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    original = backend._call

    async def oversized(system, payload):
        response = await original(system, payload)
        if "examples" in payload:
            row = deepcopy(response["samples"][-1])
            decision = deepcopy(response["decisions"][-1])
            row["sample_id"] += "-overflow"
            row["text"] += " additional case"
            decision["sample_id"] = row["sample_id"]
            response["samples"].append(row)
            response["decisions"].insert(0, decision)
        return response

    monkeypatch.setattr(backend, "_call", oversized)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert result["accepted_challenge_samples"] == 6
    for round_ in result["generation_rounds"]:
        assert len(round_["response"]["samples"]) == 3  # Preserve raw response for audit.
        verification = round_["verification"]
        assert len(verification["accepted"]) == 2
        assert verification["rejected"] == [
            {"index": 2, "reason": "generator batch budget exceeded"}
        ]
    assert all(len(p["samples"]) == 2 for p in backend.verifications)


@pytest.mark.parametrize("malformation", ["count", "duplicate", "top_level"])
def test_malformed_generator_round_is_rejected_without_stopping_training(
    tmp_path, monkeypatch, malformation
):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    original = backend._call

    async def malformed(system, payload):
        response = await original(system, payload)
        if "examples" in payload:
            if malformation == "count":
                response["decisions"].pop()
            elif malformation == "duplicate":
                response["decisions"][1]["sample_id"] = response["decisions"][0]["sample_id"]
            else:
                response["extra"] = "not allowed"
        return response

    monkeypatch.setattr(backend, "_call", malformed)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert result["accepted_challenge_samples"] == 0
    assert not backend.verifications
    assert contexts  # Optimization still receives real training traces.
    assert all(
        r["verification"]["rejected"][0]["reason"].startswith("invalid generator response:")
        for r in result["generation_rounds"]
    )


def test_supplement_filters_feedback_overlapping_unused_holdout(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    invoke(runner, samples)
    heldout = Sample("unused", "fixture", "A shared news statement with exact text.", "REAL")
    overlap = {"sample_id": "consumed", "sample_public": {"text": heldout.text + " City"}}
    safe = {"sample_id": "safe", "sample_public": {"text": "Unrelated feedback text."}}
    state = {
        "train_traces": [e["trace"] for e in backend.requests[0]["examples"]],
        "feedback": [
            {"attempt": 1, "baseline_traces": [overlap, safe], "candidate_traces": [overlap]}
        ],
    }
    before = deepcopy(state)
    batch = [s for s in samples if s.sample_id in {t["sample_id"] for t in state["train_traces"]}]
    request = runner._request(state, batch, "filtered", forbidden=[heldout])
    validate_generation_request(request, [heldout])
    assert request["feedback_filter"] == {"removed_traces": 2, "removed_entries": 0}
    feedback = request["validation_feedback"][0]
    assert feedback["attempt"] == 1
    assert feedback["baseline_traces"] == [safe]
    assert feedback["candidate_traces"] == []
    assert state == before
    # Construction examples remain subject to the original firewall.
    with pytest.raises(ValueError, match="leaks held-out sample"):
        validate_generation_request(request, [batch[0]])


def test_endogenous_challenges_without_evidence_reach_optimizer(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    samples = [replace(s, evidence=()) for s in samples]
    result = invoke(runner, samples)
    assert result["generation_policy"] == "endogenous-knowledge-v1"
    assert result["accepted_challenge_samples"] == 6
    assert all(
        e["source_sample"]["evidence"] == [] for r in backend.requests for e in r["examples"]
    )
    roles = set()
    for request in backend.requests:
        assert request["construction_facts"] == []
        for example in request["examples"]:
            source = example["source_sample"]
            assert source["domain"] == "train"
            expected_role = {
                "REAL": "real_training_anchor",
                "FAKE": "fake_training_reference",
            }[source["label"]]
            assert example["source_role"] == expected_role
            roles.add(example["source_role"])
    assert roles == {"real_training_anchor", "fake_training_reference"}
    assert any(phase.startswith("adversarial") for phase, _ in calls)
    assert any(t["sample_id"].startswith("gen-") for c in contexts for t in c["traces"])
    for payload in backend.verifications:
        for sample in payload["samples"]:
            assert sample["evidence"] == []
            assert sample["source_training_sample"]["gold_label"] in {"REAL", "FAKE"}
            assert set(sample["source_training_sample"]) == {"text", "gold_label"}
            assert "source_role" not in sample
            assert not {"label", "target_label", "reason", "strategy", "decisions"} & set(sample)
    assert "generation_policy" in runner._identity_extra()


def test_endogenous_review_still_rejects_wrong_labels_without_evidence(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch, reject=True)
    result = invoke(runner, [replace(s, evidence=()) for s in samples])
    assert backend.verifications
    assert result["accepted_challenge_samples"] == 0
    assert not any(phase.startswith("adversarial") for phase, _ in calls)
    assert all(
        "knowledge verdict disagrees" in rejection["reason"]
        for r in result["generation_rounds"]
        for rejection in r["verification"]["rejected"]
    )


def test_endogenous_review_can_decline_uncertain_labels(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    original = backend._call

    async def uncertain(system, payload):
        if system == KNOWLEDGE_VERIFIER_SYSTEM:
            return {
                "reviews": [
                    {
                        "sample_id": s["sample_id"],
                        "status": "INDETERMINATE",
                        "predicted_label": None,
                        "reason": "The transformed factual claim is not known.",
                    }
                    for s in payload["samples"]
                ]
            }
        return await original(system, payload)

    monkeypatch.setattr(backend, "_call", uncertain)
    result = invoke(runner, [replace(s, evidence=()) for s in samples])
    assert result["accepted_challenge_samples"] == 0
    assert not any(phase.startswith("adversarial") for phase, _ in calls)


def test_evidence_mode_checkpoint_cannot_resume_as_endogenous(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    identity = runner._identity_extra

    def legacy_identity():
        extra = identity()
        extra.pop("generation_policy")
        extra.pop("generation_policy_digest")
        return extra

    monkeypatch.setattr(runner, "_identity_extra", legacy_identity)
    invoke(runner, samples)
    monkeypatch.setattr(runner, "_identity_extra", identity)
    requests_before = len(backend.requests)
    with pytest.raises(ValueError, match="incompatible meta batch checkpoint"):
        invoke(runner, samples, resume=True)
    assert len(backend.requests) == requests_before


def test_supplement_cap_and_cached_traces(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(
        tmp_path, monkeypatch, max_supplement_rounds=1
    )
    original = runner.base.run

    async def fail_gate(rows, **kwargs):
        traces, result = await original(rows, **kwargs)
        if kwargs["phase"].startswith("validation"):
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": 0.5}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", fail_gate)
    result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=4)
    assert result["status"] == "completed" and result["completed_batches"] == 1
    assert [b["batch"] for b in result["skipped_batches"]] == [1]
    assert len(backend.requests) == 2
    assert len([c for c in calls if c[0].startswith("adversarial")]) == 2
    assert len(contexts) == 4


def test_attempt_limit_resets_adversarial_batch_before_next_batch(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    result = invoke(runner, samples, max_attempts_per_batch=1)
    assert result["status"] == "completed" and result["completed_batches"] == 2
    assert [b["batch"] for b in result["skipped_batches"]] == [1]
    assert len(backend.requests) == 2
    assert not backend.requests[1]["validation_feedback"]
    assert not contexts[1].get("validation_feedback")
    assert contexts[0]["skills"] == contexts[1]["skills"]


def test_resume_after_generation_reuses_response(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    verify = runner.verifier.verify
    failed = False

    async def interrupt(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("interrupt verification")
        return await verify(*args, **kwargs)

    monkeypatch.setattr(runner.verifier, "verify", interrupt)
    with pytest.raises(RuntimeError, match="interrupt verification"):
        invoke(runner, samples)
    assert len(backend.requests) == 1
    result = invoke(runner, samples, resume=True)
    assert result["status"] == "completed"
    assert len(backend.requests) == 3
    assert len([c for c in calls if c[0] == "train inference"]) == 2


def test_verifier_rejection_never_reaches_optimizer(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch, reject=True)
    result = invoke(runner, samples)
    assert result["accepted_challenge_samples"] == 0
    assert not any(p.startswith("adversarial") for p, _ in calls)
    assert all(not t["sample_id"].startswith("gen-") for c in contexts for t in c["traces"])


def test_same_config_routes_have_independent_artifacts_and_banks(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    ordinary = runner.config
    meta = evolution_route_config(ordinary, "meta-evolve")
    adversarial = evolution_route_config(ordinary, "adversarial-evolve")
    assert evolution_route_config(ordinary, "evolve") is ordinary
    assert len({ordinary.skill_store, meta.skill_store, adversarial.skill_store}) == 3
    assert (
        len(
            {
                ordinary.meta_learning.checkpoint_path,
                meta.meta_learning.checkpoint_path,
                adversarial.meta_learning.checkpoint_path,
            }
        )
        == 3
    )
    assert evolution_route_config(meta, "meta-evolve") == meta
    repo_ordinary, repo_meta, repo_adv = [
        SkillRepository(c.skill_store) for c in (ordinary, meta, adversarial)
    ]
    for repo in (repo_ordinary, repo_meta, repo_adv):
        repo.commit_package_bank(
            runner.base.packages, run_id="initialize", expected_active={}, audit={}
        )
    before = package_bank_digest(list(repo_ordinary.active_packages().values()))
    runner.config = adversarial
    runner.repository = repo_adv
    result = invoke(runner, samples, train_limit=2)
    assert result["status"] == "completed"
    assert package_bank_digest(list(repo_adv.active_packages().values())) != before
    assert package_bank_digest(list(repo_ordinary.active_packages().values())) == before
    assert package_bank_digest(list(repo_meta.active_packages().values())) == before
    assert route_artifact_path(tmp_path / "trace.jsonl", "meta-evolve") != route_artifact_path(
        tmp_path / "trace.jsonl", "adversarial-evolve"
    )


def test_meta_does_not_generate_even_when_generation_enabled(tmp_path, monkeypatch):
    adv, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    config = evolution_route_config(adv.config, "meta-evolve")
    meta = MetaBatchRunner(config, ROOT)
    monkeypatch.setattr(meta.base, "_optimizer_backend", EditingOptimizerBackend)
    result = invoke(meta, samples, max_attempts_per_batch=1, train_limit=2)
    assert result["mode"] == "meta-evolve" and result["status"] == "completed"
    assert "generation_rounds" not in result
    assert not backend.requests


def test_route_checkpoint_cannot_resume_other_mode(tmp_path, monkeypatch):
    adv, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    invoke(adv, samples, max_attempts_per_batch=1, train_limit=2)
    meta = MetaBatchRunner(adv.config, ROOT)
    with pytest.raises(ValueError, match="incompatible"):
        invoke(meta, samples, resume=True, train_limit=2)


def test_cli_meta_default_limit_and_isolated_output(tmp_path, monkeypatch):
    from evofact.cli import _run, build_parser

    adv, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    observed = {}

    async def run(instance, rows, **kwargs):
        observed.update(config=instance.config, **kwargs)
        return {"mode": "meta-evolve", "run_id": "test-run"}

    monkeypatch.setattr("evofact.cli.load_config", lambda _: adv.config)
    monkeypatch.setattr(MetaBatchRunner, "run", run)
    args = build_parser().parse_args(
        ["meta-evolve", "--evaluation-only", "--output", str(tmp_path / "result.json")]
    )
    result = asyncio.run(_run(args))
    assert observed["train_limit"] is None
    assert observed["config"].skill_store == adv.config.skill_store / "meta-evolve"
    assert Path(args.output).parent.name == "meta-evolve"
    assert result["mode"] == "meta-evolve"
    assert not backend.requests


def test_cli_test_selects_route_bank(tmp_path, monkeypatch):
    from evofact.cli import _run, build_parser
    from evofact.experiments.runner import ExperimentRunner

    adv, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    observed = []
    original = ExperimentRunner.__init__

    def initialize(instance, config, root):
        observed.append(config.skill_store)
        original(instance, config, root)

    monkeypatch.setattr("evofact.cli.load_config", lambda _: adv.config)
    monkeypatch.setattr(ExperimentRunner, "__init__", initialize)
    args = build_parser().parse_args(
        ["--evolution-route", "adversarial-evolve", "--limit", "1", "test"]
    )
    result = asyncio.run(_run(args))
    assert observed == [adv.config.skill_store / "adversarial-evolve"]
    assert result
    assert not backend.requests
