"""Joint changes are validated, evaluated and committed together, without API calls."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_dynamic_specialist_packages import _addition_response
from test_meta_batch_runner import ROOT, invoke, setup_runner
from test_optimizer_memory import Backend, context, setup

from evofact.evolution.joint_optimizer import build_joint_candidates, candidate_bank
from evofact.evolution.optimizer_memory import (
    OptimizerContextFailure,
    OptimizerProtocolFailure,
    reduce_context,
    remember,
)
from evofact.evolution.package_optimizer import PackageOptimizerAgent
from evofact.experiments.meta_batch_runner import MetaBatchRunner
from evofact.runtime.backend import BackendResult
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.skills.package_adapter import package_bank_digest, package_to_skill_spec
from evofact.skills.package_serializer import package_to_dict
from evofact.skills.repository import SkillRepository


def joint_response(packages):
    selected = [p for p in packages if p.manifest.name in {"source_credibility", "judge_decision"}]
    assert len(selected) == 2
    return {
        "action": "joint",
        "rationale": "Coordinate evidence provenance with verdict aggregation.",
        "changes": [
            {
                "action": "edit",
                "target_skill_id": p.skill_id,
                "rationale": "Keep provenance limitations visible in the final assessment.",
                "file_operations": [
                    {
                        "operation": "update",
                        "path": "SKILL.md",
                        "expected_digest": p.file("SKILL.md").digest,
                        "content": p.file("SKILL.md").content.decode()
                        + "\nExplain provenance limitations.\n",
                    }
                ],
            }
            for p in selected
        ],
    }


def install_joint(runner, contexts, monkeypatch, *, addition=False):
    class JointBackend:
        async def optimize_package(self, context, skill):
            contexts.append(deepcopy(context))
            assert context["joint_optimization"] is True
            response = joint_response(runner.base.packages)
            if addition:
                response["changes"].append(_addition_response())
            return BackendResult(response)

    monkeypatch.setattr(runner.base, "_optimizer_backend", JointBackend)


def test_joint_edit_and_add_preserves_unmodified_packages(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    original = list(runner.base.packages)
    response = joint_response(original)
    response["changes"].append(_addition_response())
    candidates = build_joint_candidates(response, original)
    desired = candidate_bank(original, candidates)
    assert len(candidates) == 3
    assert len(desired) == len(original) + 1
    targets = {c.base.skill_id for c in candidates if c.base}
    assert all(p in desired for p in original if p.skill_id not in targets)
    assert package_bank_digest(original) == package_bank_digest(runner.base.packages)


@pytest.mark.parametrize("failure", ["duplicate", "digest", "frozen", "collision"])
def test_invalid_member_rejects_entire_group(tmp_path, monkeypatch, failure):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    packages = list(runner.base.packages)
    before = package_bank_digest(packages)
    response = joint_response(packages)
    if failure == "duplicate":
        response["changes"].append(deepcopy(response["changes"][0]))
    elif failure == "digest":
        response["changes"][1]["file_operations"][0]["expected_digest"] = "0" * 64
    elif failure == "frozen":
        frozen = next(p for p in packages if p.manifest.name == "generation_agent")
        response["changes"][1]["target_skill_id"] = frozen.skill_id
    else:
        response["changes"] += [_addition_response(), _addition_response()]
    with pytest.raises(ValueError):
        build_joint_candidates(response, packages)
    assert package_bank_digest(packages) == before


def test_joint_protocol_is_opt_in_for_agent_and_prompt(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    packages = runner.base.packages
    response = joint_response(packages)

    class Backend:
        async def optimize_package(self, context, skill):
            return BackendResult(response)

    package = next(p for p in packages if p.manifest.name == "skill_optimizer")
    agent = PackageOptimizerAgent(Backend(), package)
    legacy = asyncio.run(agent.propose(None, packages))
    assert legacy.status == "skipped"
    joint = asyncio.run(agent.propose(None, packages, joint_enabled=True))
    assert len(joint.candidates) == 2
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "unused", "test")
    skill = package_to_skill_spec(package)
    assert "Joint optimization policy" not in backend.optimizer_system({"skills": []}, skill)
    assert "Joint optimization policy" in backend.optimizer_system(
        {"skills": [], "joint_optimization": True}, skill
    )


@pytest.mark.parametrize("accept", [False, True])
def test_group_validation_and_commit_are_all_or_nothing(tmp_path, monkeypatch, accept):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_joint(runner, contexts, monkeypatch, addition=True)
    original = list(runner.base.packages)
    repo = SkillRepository(tmp_path / "skills")
    repo.commit_package_bank(original, run_id="initialize", expected_active={}, audit={})
    runner.repository = repo
    run = runner.base.run
    candidate_banks = []

    async def scored(rows, **kwargs):
        if kwargs["phase"] == "validation candidate":
            candidate_banks.append(kwargs["skills"])
        traces, result = await run(rows, **kwargs)
        if kwargs["phase"].startswith("validation"):
            score = 0.75 if accept and kwargs["phase"] == "validation candidate" else 0.5
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": score}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", scored)
    result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=1)
    assert result["status"] == "completed"
    assert len(result["attempts"][0]["proposals"]) == 3
    assert len(candidate_banks) == 1
    assert len(candidate_banks[0]) == len(original) + 1
    active = list(repo.active_packages().values())
    assert (package_bank_digest(active) != package_bank_digest(original)) is accept
    assert package_bank_digest(active) == package_bank_digest(runner.base.packages)
    if accept:
        changed = {p.manifest.name for p in active if p not in original}
        assert changed == {"source_credibility", "judge_decision", "geospatial_context"}
    else:
        saved = json.loads((tmp_path / "checkpoint.json").read_text())
        assert len(saved["attempts"][0]["proposals"]) == 3


@pytest.mark.parametrize("after_write", [False, True])
def test_resume_recovers_entire_joint_commit(tmp_path, monkeypatch, after_write):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_joint(runner, contexts, monkeypatch)
    repo = SkillRepository(tmp_path / "skills")
    repo.commit_package_bank(
        runner.base.packages, run_id="initialize", expected_active={}, audit={}
    )
    runner.repository = repo
    commit = repo.commit_package_bank

    def interrupted(*args, **kwargs):
        if after_write:
            commit(*args, **kwargs)
        raise RuntimeError("joint commit interruption")

    monkeypatch.setattr(repo, "commit_package_bank", interrupted)
    with pytest.raises(RuntimeError, match="joint commit interruption"):
        invoke(runner, samples, train_limit=2)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    desired = state["commit_intent"]["packages"]
    assert len(state["pending"]["packages"]) == 2
    monkeypatch.setattr(repo, "commit_package_bank", commit)
    restored = MetaBatchRunner(runner.config, ROOT, repo)
    result = invoke(restored, samples, train_limit=2, resume=True)
    assert result["completed_batches"] == 1
    assert {p.package_digest for p in restored.base.packages} == {
        p["package_digest"] for p in desired
    }


class JointSelectionBackend(Backend):
    async def select_optimizer_target(self, context, skill):
        self.calls.append(("selection", deepcopy(context)))
        return BackendResult(
            {"action": "joint", "target_skill_ids": ["s0", "s1"], "allow_additions": False}
        )


def test_two_stage_keeps_all_selected_packages_and_caches_selection():
    manager, backend, state, skill = setup(JointSelectionBackend(), window=30000)
    payload = context(size=6000, skills=5)
    payload["joint_optimization"] = True
    asyncio.run(manager.optimize_package(payload, skill))
    proposal = [c for name, c in backend.calls if name == "proposal"][0]
    assert {p["skill_id"] for p in proposal["skills"]} == {"s0", "s1"}
    before = len(backend.calls)
    asyncio.run(manager.optimize_package(payload, skill))
    assert len(backend.calls) == before


def test_selected_group_too_large_fails_without_silently_dropping_target():
    manager, backend, _, skill = setup(JointSelectionBackend(), window=14000)
    payload = context(size=6000, skills=5)
    payload["joint_optimization"] = True
    with pytest.raises(OptimizerContextFailure, match="does not fit"):
        asyncio.run(manager.optimize_package(payload, skill))
    assert not any(name == "proposal" for name, _ in backend.calls)


def test_joint_selection_rejects_target_drift():
    class DriftingBackend(JointSelectionBackend):
        async def optimize_package(self, context, skill):
            return BackendResult(
                {"action": "joint", "changes": [{"action": "edit", "target_skill_id": "s4"}]}
            )

    manager, _, _, skill = setup(DriftingBackend(), window=30000)
    payload = context(size=6000, skills=5)
    payload["joint_optimization"] = True
    with pytest.raises(OptimizerProtocolFailure, match="selected targets"):
        asyncio.run(manager.optimize_package(payload, skill))


@pytest.mark.parametrize("route", ["feedback", "adversarial"])
def test_joint_changes_use_shared_loop_in_both_routes(tmp_path, monkeypatch, route):
    if route == "feedback":
        from test_feedback_evolve import invoke as route_invoke
        from test_feedback_evolve import setup as route_setup

        runner, train, valid, contexts, _ = route_setup(tmp_path, monkeypatch)
        install_joint(runner, contexts, monkeypatch)
        result = route_invoke(runner, train, valid)
        assert all(len(g) == 2 for g in result["proposal_groups"])
        assert len(contexts[1]["prior_proposals"]) == 2
    else:
        from test_adversarial_batch_runner import invoke as route_invoke
        from test_adversarial_batch_runner import setup as route_setup

        runner, backend, contexts, _, samples = route_setup(tmp_path, monkeypatch)
        install_joint(runner, contexts, monkeypatch)
        result = route_invoke(runner, samples)
        assert backend.requests
        assert any(t["sample_id"].startswith("gen-") for c in contexts for t in c["traces"])
    assert result["status"] == "completed"
    assert all(len(a["proposals"]) == 2 for a in result["attempts"] if "decision" in a)
    assert len(contexts[1]["validation_feedback"][0]["rejected_packages"]) == 2


def test_legacy_checkpoint_resumes_with_single_package_protocol(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("before proposal")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="before proposal"):
        invoke(runner, samples, train_limit=2)
    checkpoint = tmp_path / "checkpoint.json"
    state = json.loads(checkpoint.read_text())
    state.pop("joint_optimization")
    state.pop("optimizer_feedback_policy")
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    result = invoke(runner, samples, train_limit=2, resume=True)
    assert result["status"] == "completed"
    assert all("joint_optimization" not in c for c in contexts)
    assert all("proposals" not in a for a in result["attempts"])


def test_joint_memory_retains_all_versions_and_reduces_all_rejected_files(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    packages = runner.base.packages
    candidates = build_joint_candidates(joint_response(packages), packages)
    serialized = [package_to_dict(c.package) for c in candidates]
    state = {"packages": [package_to_dict(p) for p in packages]}
    pending = {
        "package": serialized[0],
        "packages": serialized,
        "base_id": candidates[0].base.skill_id,
        "base_ids": [c.base.skill_id for c in candidates],
        "joint_rationale": "coordinate provenance and aggregation",
        "baseline": {"result": {}, "traces": []},
        "candidate": {"result": {}, "traces": []},
    }
    remember(state, {"batch": 1, "attempt": 1, "decision": {"accepted": False}}, pending=pending)
    summary = state["optimizer_memories"][0]["summary"]
    assert summary["joint_outcome"] is True
    assert len(summary["changes"]) == 2
    assert {c["base"]["skill_id"] for c in summary["changes"]} == set(pending["base_ids"])
    context = {
        "skills": deepcopy(serialized),
        "validation_feedback": [
            {
                "rejected_package": deepcopy(serialized[0]),
                "rejected_packages": deepcopy(serialized),
            }
        ],
    }
    while reduce_context(context) is not None:
        pass
    feedback = context["validation_feedback"][0]
    assert all("content" not in f for p in feedback["rejected_packages"] for f in p["files"])
    assert context["skills"] == serialized  # Full current targets are protected.
