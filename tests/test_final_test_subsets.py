from types import SimpleNamespace

import pytest

from scripts.recompute_final_test_subsets import ordered_records, prefix_by_domain


def test_resumed_records_use_manifest_order_not_completion_order():
    records = [
        {"kind": "manifest", "sample_ids": ["z", "a", "m"]},
        {"kind": "sample", "sample_id": "m", "status": "failed"},
        {"kind": "sample", "sample_id": "a", "status": "complete", "trace": {"sample_id": "a"}},
        {"kind": "sample", "sample_id": "m", "status": "complete", "trace": {"sample_id": "m"}},
        {"kind": "sample", "sample_id": "z", "status": "complete", "trace": {"sample_id": "z"}},
    ]
    assert [t["sample_id"] for t in ordered_records(records)] == ["z", "a", "m"]
    with pytest.raises(ValueError, match="membership"):
        ordered_records(records[:-1])


def test_prefixes_are_per_domain_nested_and_preserve_interleaving():
    traces = [{"sample_id": f"{domain}-{i}"} for i in range(100) for domain in ("a", "b")]
    index = {t["sample_id"]: SimpleNamespace(domain=t["sample_id"][0]) for t in traces}
    fifty = prefix_by_domain(traces, index, 50)
    twenty_five = prefix_by_domain(traces, index, 25)
    assert fifty == traces[:100]
    assert twenty_five == fifty[:50]
    assert len(fifty) == 100 and len(twenty_five) == 50
    with pytest.raises(ValueError, match="exactly 100"):
        prefix_by_domain(traces[:-1], index, 25)
    with pytest.raises(ValueError, match="duplicate"):
        prefix_by_domain(traces + [traces[0]], index, 25)
