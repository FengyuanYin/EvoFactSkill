from __future__ import annotations

from dataclasses import fields, replace

import pytest

from evofact.experiments.checkpoint import CheckpointIdentity, CheckpointV2Store


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        config_digest="config",
        real_data_digest="real",
        split_digest="split",
        lineage_digest="lineage",
        package_bank_digest="bank",
        generator_package_digest="generator",
        governance_digest="governance",
        dag_policy_digest="dag",
        budget_policy_digest="budget",
        pricing_version="pricing-v1",
        episode_plan_digest="episodes",
    )


def test_checkpoint_v2_round_trip_and_each_identity_field_is_enforced(tmp_path):
    store = CheckpointV2Store(tmp_path / "checkpoint.json")
    identity = _identity()
    store.save(identity, {"completed_episode_ids": ["episode-1"]})
    assert store.load(identity) == {"completed_episode_ids": ["episode-1"]}

    for field in fields(identity):
        changed = replace(identity, **{field.name: getattr(identity, field.name) + "-changed"})
        with pytest.raises(ValueError, match=field.name):
            store.load(changed)


def test_checkpoint_v1_cannot_resume_as_v2(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema_version":"meta_checkpoint_v1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="cannot resume"):
        CheckpointV2Store(path).load(_identity())
