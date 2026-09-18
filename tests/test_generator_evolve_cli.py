import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from evofact.cli import _run_impl, build_parser
from evofact.config import load_config
from evofact.experiments.runner import ExperimentRunner
from evofact.generation.prompts import VERIFIER_SYSTEM
from evofact.runtime.progress import NullProgressSink
from evofact.skills.package_loader import load_package

ROOT = Path(__file__).resolve().parents[1]


def test_generator_evolve_evaluation_only_runs_real_gate(tmp_path):
    package_path = ROOT / "skills" / "seeds" / "generation_agent"
    package = load_package(package_path)
    config = load_config(ROOT / "configs" / "dry_run.yaml")
    runner = ExperimentRunner(config, ROOT)
    pricing_identity = runner.pricing_identity()
    metrics = {
        "validity_rate": 1.0,
        "agreement_rate": 1.0,
        "evidence_rate": 1.0,
        "diversity": 1.0,
        "duplicate_rate": 0.0,
        "difficulty": 0.5,
        "teaching_gain": 0.1,
        "leakage_rate": 0.0,
        "safety_rejection_rate": 0.0,
        "cost": "0.01",
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "source_episode_id": "construction-1",
                "evaluation_episode_id": "evaluation-2",
                "champion_digest": package.package_digest,
                "challenger_digest": package.package_digest,
                "label_contract_digest": runner.label_contract_registry.digest,
                "metadata": {
                    "evaluation_protocol": "paired-generator-v1",
                    "champion_generator_digest": package.package_digest,
                    "challenger_generator_digest": package.package_digest,
                    "verifier_fingerprint": hashlib.sha256(
                        f"{VERIFIER_SYSTEM}\0{config.backend}\0{config.model}".encode("utf-8")
                    ).hexdigest(),
                    "pricing_identity": pricing_identity,
                    "label_contract_digest": runner.label_contract_registry.digest,
                    "source_data_digest": "a" * 64,
                    "evaluation_data_digest": "b" * 64,
                },
                "champion": metrics,
                "challenger": metrics,
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dry_run.yaml",
            "generator-evolve",
            "--evaluation-only",
            "--candidate",
            str(package_path),
            "--evaluation",
            str(evaluation_path),
        ]
    )

    result = asyncio.run(_run_impl(args, NullProgressSink()))

    assert result["decision"]["accepted"] is True
    assert result["active_bank_updated"] is False
    assert result["candidate_digest"] == package.package_digest


def test_generator_evolve_rejects_unbound_evaluation(tmp_path):
    package_path = ROOT / "skills" / "seeds" / "generation_agent"
    package = load_package(package_path)
    metrics = {
        "validity_rate": 1.0,
        "agreement_rate": 1.0,
        "evidence_rate": 1.0,
        "diversity": 1.0,
        "duplicate_rate": 0.0,
        "difficulty": 0.5,
        "teaching_gain": 0.1,
        "leakage_rate": 0.0,
        "safety_rejection_rate": 0.0,
        "cost": "0.01",
    }
    evaluation_path = tmp_path / "unbound.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "source_episode_id": "construction-1",
                "evaluation_episode_id": "evaluation-2",
                "champion_digest": package.package_digest,
                "challenger_digest": package.package_digest,
                "champion": metrics,
                "challenger": metrics,
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dry_run.yaml",
            "generator-evolve",
            "--evaluation-only",
            "--candidate",
            str(package_path),
            "--evaluation",
            str(evaluation_path),
        ]
    )
    with pytest.raises(ValueError, match="label contract identity"):
        asyncio.run(_run_impl(args, NullProgressSink()))


def test_generator_propose_only_emits_complete_package():
    package_path = ROOT / "skills" / "seeds" / "generation_agent"
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dry_run.yaml",
            "generator-evolve",
            "--propose-only",
            "--candidate",
            str(package_path),
        ]
    )
    result = asyncio.run(_run_impl(args, NullProgressSink()))
    assert result["mode"] == "generator-propose"
    assert {row["path"] for row in result["candidate_package"]["files"]} >= {
        "SKILL.md",
        "metadata.json",
        "references/strategies.json",
    }


def test_generator_audit_path_invokes_package_optimizer(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"entries": [{"audit_id": "a-1"}]}), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dry_run.yaml",
            "generator-evolve",
            "--propose-only",
            "--audit",
            str(audit_path),
        ]
    )
    with pytest.raises(ValueError, match="Package Optimizer proposed no Generator change"):
        asyncio.run(_run_impl(args, NullProgressSink()))
