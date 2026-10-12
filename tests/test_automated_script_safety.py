from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from evofact.config import GateConfig, MetaLearningConfig
from evofact.core.models import EvaluationResult, StatisticalTestResult, TransferUtility
from evofact.core.package_models import SkillFile
from evofact.security.package_scanner import scan_package
from evofact.security.scanner import scan_resources
from evofact.skills.package_loader import load_package
from evofact.validation.gate import ValidationGate
from evofact.validation.meta_gate import MetaValidationGate


def script_package(content: bytes, *, declared: bool = True):
    root = Path(__file__).resolve().parents[1]
    package = load_package(root / "skills/seeds/generation_agent")
    script = SkillFile("scripts/helper.py", "text/x-python", content, sha256(content).hexdigest())
    manifest = replace(
        package.manifest,
        entrypoints=replace(package.manifest.entrypoints, script=script.path if declared else None),
    )
    return replace(package, manifest=manifest, files=(*package.files, script), package_digest="")


@pytest.mark.parametrize(
    "content,declared,expected",
    [
        (b"def helper(x): return x\n", True, "safe"),
        (b"import subprocess\n", True, "blocked"),
        (b"open('file')\n", True, "blocked"),
        (b"def broken(\n", True, "blocked"),
        (b"\xff", True, "blocked"),
        (b"def helper(x): return x\n", False, "blocked"),
    ],
)
def test_package_script_automated_safety(content, declared, expected):
    report = scan_package(script_package(content, declared=declared))
    assert report.level == expected
    assert bool(report.findings) == (expected == "blocked")


@pytest.mark.parametrize(
    "source,accepted", [("def helper(x): return x", True), ("import subprocess", False)]
)
def test_script_scan_result_controls_both_promotion_gates(source, accepted):
    level = scan_resources({"scripts/helper.py": source}).level
    baseline = EvaluationResult((), {"macro_f1_all": 0.7})
    candidate = EvaluationResult((), {"macro_f1_all": 0.71})
    ordinary = ValidationGate(GateConfig()).decide(
        baseline, candidate, StatisticalTestResult("paired", 0, 1, False), safety=level
    )
    utility = TransferUtility(
        candidate_fingerprint="candidate",
        episode_count=3,
        tested_domains=("a", "b"),
        mean_gain=0.01,
        std_gain=0.0,
        confidence_interval=(0.005, 0.015),
        success_rate=1.0,
        negative_transfer_rate=0.0,
        worst_domain_drop=0.0,
        mean_coverage=1.0,
        coverage_delta=0.0,
        calibration_delta=0.0,
        cost_ratio=1.0,
        domain_gains={"a": 0.01, "b": 0.01},
    )
    meta = MetaValidationGate(MetaLearningConfig()).decide(utility, safety_level=level)
    assert ordinary.accepted == accepted
    assert meta.accepted == accepted
    assert ordinary.disposition == ("active" if accepted else "rejected")
    assert meta.disposition == ("generalized" if accepted else "rejected")
