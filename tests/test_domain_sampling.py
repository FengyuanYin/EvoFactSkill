from __future__ import annotations

import json
from pathlib import Path

import pytest

from evofact.core.models import Sample
from evofact.data.sampling import balanced_domain_sample, select_final_test_samples


def _rows(domains: int, rows_per_domain: int) -> list[Sample]:
    return [
        Sample(
            sample_id=f"sample-{domain}-{index:03d}",
            dataset="fixture",
            text=f"claim {domain} {index}",
            label="REAL" if index % 2 else "FAKE",
            domain=f"domain-{domain}",
        )
        for domain in range(domains)
        for index in range(rows_per_domain)
    ]


def test_balanced_domain_limit_is_total_and_reproducible() -> None:
    rows = _rows(8, 70)
    domains = tuple(f"domain-{index}" for index in range(8))
    first, audit = balanced_domain_sample(rows, domains, 400, seed=42)
    second, second_audit = balanced_domain_sample(rows, domains, 400, seed=42)

    assert len(first) == 400
    assert audit["selected_by_domain"] == {domain: 50 for domain in domains}
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert audit["sample_ids_digest"] == second_audit["sample_ids_digest"]


def test_balanced_domain_limit_reallocates_only_when_a_domain_is_short() -> None:
    rows = _rows(2, 10) + [
        Sample(f"extra-{index}", "fixture", f"extra {index}", "REAL", "domain-2")
        for index in range(30)
    ]
    selected, audit = balanced_domain_sample(
        rows,
        ("domain-0", "domain-1", "domain-2"),
        45,
        seed=7,
    )
    assert len(selected) == 45
    assert audit["selected_by_domain"] == {
        "domain-0": 10,
        "domain-1": 10,
        "domain-2": 25,
    }


def test_static_final_test_uses_first_ids_in_recorded_order(tmp_path: Path) -> None:
    rows = [
        Sample(f"final-{index:03d}", "fixture", f"claim {index}", "REAL", "final")
        for index in range(120)
    ]
    recorded = list(reversed(rows))
    static = tmp_path / "fixture-test-final.json"
    static.write_text(
        json.dumps({"traces": [{"sample_id": sample.sample_id} for sample in recorded]}),
        encoding="utf-8",
    )

    selected, audit = select_final_test_samples(
        rows,
        ("final",),
        dataset="fixture",
        samples_per_domain=100,
        seed=42,
        repository_root=tmp_path,
        static_pattern="{dataset}-test-{domain}.json",
        require_static=True,
    )

    assert [sample.sample_id for sample in selected] == [
        sample.sample_id for sample in recorded[:100]
    ]
    assert audit["requested_per_domain"] == 100
    assert audit["selected_total"] == 100
    assert audit["selected_by_domain"] == {"final": 100}


def test_static_final_test_fails_closed_on_dataset_drift(tmp_path: Path) -> None:
    rows = [Sample("known", "fixture", "claim", "REAL", "final")]
    static = tmp_path / "fixture-test-final.json"
    static.write_text(json.dumps({"traces": [{"sample_id": "unknown"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the loaded dataset"):
        select_final_test_samples(
            rows,
            ("final",),
            dataset="fixture",
            samples_per_domain=100,
            seed=42,
            repository_root=tmp_path,
            static_pattern="{dataset}-test-{domain}.json",
            require_static=True,
        )


def test_static_final_test_requires_full_quota_for_every_domain(tmp_path: Path) -> None:
    rows = [
        Sample(f"a-{index:03d}", "fixture", f"a {index}", "REAL", "a") for index in range(100)
    ] + [Sample(f"b-{index:03d}", "fixture", f"b {index}", "REAL", "b") for index in range(99)]
    for domain in ("a", "b"):
        domain_rows = [sample for sample in rows if sample.domain == domain]
        (tmp_path / f"fixture-test-{domain}.json").write_text(
            json.dumps({"traces": [{"sample_id": sample.sample_id} for sample in domain_rows]}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="undersized domains.*'b': 99"):
        select_final_test_samples(
            rows,
            ("a", "b"),
            dataset="fixture",
            samples_per_domain=100,
            seed=42,
            repository_root=tmp_path,
            static_pattern="{dataset}-test-{domain}.json",
            require_static=True,
        )
