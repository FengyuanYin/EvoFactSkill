from __future__ import annotations

import csv
from pathlib import Path

from evofact.reporting.validation_curve import (
    active_validation_rows,
    gate_validation_rows,
    write_validation_curve,
)


def test_gate_validation_rows_join_proposal_to_batch():
    checkpoint = {
        "total_batches": 2,
        "batch_audit": [
            {"batch_index": 1, "sample_ids": ["sample-1"]},
            {"batch_index": 2, "sample_ids": ["sample-2"]},
        ],
        "accumulated": {
            "traces": [{"trace_id": "trace-2", "sample_id": "sample-2"}],
            "proposals": [{"proposal_id": "proposal-1", "source_trace_ids": ["trace-2"]}],
            "gate_decisions": [
                {
                    "disposition": "rejected",
                    "baseline_result": {
                        "aggregate_metrics": {"brier": 0.2, "macro_f1_all": 0.7, "n": 150.0}
                    },
                    "candidate_result": {
                        "aggregate_metrics": {"brier": 0.3, "macro_f1_all": 0.6}
                    },
                }
            ],
        },
    }
    assert gate_validation_rows(checkpoint) == [
        {
            "batch_index": 2,
            "proposal_id": "proposal-1",
            "disposition": "rejected",
            "baseline_brier": 0.2,
            "candidate_brier": 0.3,
            "baseline_macro_f1": 0.7,
            "candidate_macro_f1": 0.6,
            "validation_evaluations": 150.0,
            "baseline_cost_available": None,
            "candidate_cost_available": None,
        }
    ]


def test_validation_curve_exports_csv_and_svg(tmp_path: Path):
    import json

    checkpoint = {
        "total_batches": 2,
        "batch_audit": [
            {
                "batch_index": 1,
                "sample_ids": ["sample-1"],
                "validation_metrics": {"brier": 0.18, "n": 50.0, "cost_available": 1.0},
            }
        ],
        "accumulated": {
            "traces": [{"trace_id": "trace-1", "sample_id": "sample-1"}],
            "proposals": [{"proposal_id": "proposal-1", "source_trace_ids": ["trace-1"]}],
            "gate_decisions": [
                {
                    "disposition": "active",
                    "baseline_result": {"aggregate_metrics": {"brier": 0.25}},
                    "candidate_result": {"aggregate_metrics": {"brier": 0.2}},
                }
            ],
        },
    }
    source = tmp_path / "checkpoint.json"
    source.write_text(json.dumps(checkpoint), encoding="utf-8")
    paths = write_validation_curve(source, tmp_path / "plots")
    with Path(paths["csv"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["batch_index"] == "1"
    assert rows[0]["candidate_brier"] == "0.2"
    assert active_validation_rows(checkpoint)[0]["brier"] == 0.18
    with Path(paths["active_csv"]).open(encoding="utf-8", newline="") as stream:
        active = list(csv.DictReader(stream))
    assert active[0]["brier"] == "0.18"
    assert "Validation Brier score" in Path(paths["svg"]).read_text(encoding="utf-8")
