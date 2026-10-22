import json
from types import SimpleNamespace

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.config import OptimizerMemoryConfig
from evofact.evolution.optimizer_memory import ManagedOptimizerBackend
from evofact.evolution.regression_policy import (
    REGRESSION_REVIEW_POLICY,
    regression_policy_identity,
)
from evofact.runtime.openai_backend import OpenAICompatibleBackend


def test_review_reaches_selection_and_proposal_but_not_legacy_prompts():
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "unused", "test")
    skill = SimpleNamespace(instructions="Frozen optimizer seed instructions.")
    legacy = {"skills": [], "feedback_enabled": True, "training_supervision": {"records": []}}
    original = backend.optimizer_system(legacy, skill)
    current = {**legacy, "optimizer_regression_policy": regression_policy_identity()}
    assert backend.optimizer_system(current, skill) == original + "\n\n" + REGRESSION_REVIEW_POLICY
    assert backend.optimizer_system(legacy, skill) == original
    assert REGRESSION_REVIEW_POLICY not in backend.optimizer_system({"skills": []}, skill)
    manager = ManagedOptimizerBackend(
        backend, OptimizerMemoryConfig(enabled=True, context_window=1000000), {}, lambda: None
    )
    assert manager.system(current, skill, selection=True).endswith(REGRESSION_REVIEW_POLICY)
    assert manager.system(current, skill).count(REGRESSION_REVIEW_POLICY) == 1
    assert REGRESSION_REVIEW_POLICY not in manager.system(legacy, skill, selection=True)
    with pytest.raises(ValueError, match="feedback mode"):
        backend.optimizer_system({**current, "feedback_enabled": False}, skill)


def test_new_batch_run_records_review_version_and_passes_it_to_optimizer(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=1)
    assert result["status"] == "completed"
    assert contexts[0]["optimizer_regression_policy"] == regression_policy_identity()
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["optimizer_regression_policy"] == regression_policy_identity()


@pytest.mark.parametrize("marker", [None, {"version": "different", "prompt_digest": "changed"}])
def test_resume_does_not_upgrade_legacy_or_accept_changed_review(tmp_path, monkeypatch, marker):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("pause before optimizer")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause before optimizer"):
        invoke(runner, samples, train_limit=2)
    path = tmp_path / "checkpoint.json"
    state = json.loads(path.read_text())
    if marker is None:
        state.pop("optimizer_regression_policy")
    else:
        state["optimizer_regression_policy"] = marker
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    if marker is None:
        result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=1, resume=True)
        assert result["status"] == "completed"
        assert all("optimizer_regression_policy" not in context for context in contexts)
        assert "optimizer_regression_policy" not in json.loads(path.read_text())
    else:
        unchanged = path.read_bytes()
        with pytest.raises(ValueError, match="regression prompt changed"):
            invoke(runner, samples, train_limit=2, resume=True)
        assert path.read_bytes() == unchanged and not contexts
