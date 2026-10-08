from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recompute_static_100.py"
SPEC = importlib.util.spec_from_file_location("recompute_static_100", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cost": 0, "input_tokens": 1}, (Decimal(0), "estimated")),
        ({"cost": None, "input_tokens": 1}, (Decimal(0), "unavailable")),
        ({"cost": "0.12", "cost_status": "actual"}, (Decimal("0.12"), "actual")),
        ({"estimated_cost": "0.12"}, (Decimal("0.12"), "estimated")),
    ],
)
def test_static_trace_cost_distinguishes_zero_from_missing(usage, expected):
    assert script.trace_cost(usage) == expected


@pytest.mark.parametrize("trace_count", [0, 1])
def test_static_recomputation_refuses_missing_or_undersized_inputs(
    monkeypatch, tmp_path, trace_count
):
    monkeypatch.setattr(script, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(script, "STATIC_DIR", tmp_path)
    monkeypatch.setattr(script, "REFERENCE_RESULT", tmp_path / "reference.json")
    monkeypatch.setattr(script.DataRegistry, "load", lambda *args, **kwargs: [])
    monkeypatch.setattr(script.sys, "argv", [str(SCRIPT_PATH)])
    if trace_count:
        (tmp_path / "weibo21-test-example.json").write_text(
            json.dumps({"traces": [{"sample_id": "example"}]}), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="no static result files|traces; 100 are required"):
        script.main()
    assert not list(tmp_path.glob("*-100.json"))
