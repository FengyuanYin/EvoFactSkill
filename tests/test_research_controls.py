import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from evofact.cli import _run, build_parser
from evofact.config import EvolutionConfig, MetaLearningConfig, load_config
from evofact.core.frontmatter import parse_frontmatter
from evofact.core.models import TransferUtility
from evofact.evaluation.ablations import apply_evolution_ablation
from evofact.experiments.runner import ExperimentRunner
from evofact.governance.package_policy import validate_package
from evofact.skills.package_adapter import package_to_skill_spec
from evofact.skills.package_loader import load_package
from evofact.validation.meta_gate import MetaValidationGate

ROOT = Path(__file__).resolve().parents[1]


def test_crlf_frontmatter_is_parsed_and_removed_from_runtime_prompt():
    raw = "---\r\nname: example\r\nkind: specialist\r\n---\r\nDo the work.\r\n"
    metadata, body = parse_frontmatter(raw)
    assert metadata == {"name": "example", "kind": "specialist"}
    assert body == "Do the work."

    for directory in (ROOT / "skills/seeds").iterdir():
        if directory.is_dir():
            package = load_package(directory)
            assert not package_to_skill_spec(package).instructions.startswith("---")


def test_malformed_or_mismatched_frontmatter_fails_validation():
    with pytest.raises(ValueError, match="unterminated"):
        parse_frontmatter("---\r\nname: broken\r\n")

    package = load_package(ROOT / "skills/seeds/generation_agent")
    files = list(package.files)
    index = next(i for i, item in enumerate(files) if item.path == "SKILL.md")
    original = files[index]
    content = original.content.replace(b"name: generation_agent", b"name: wrong_name")
    files[index] = replace(original, content=content, digest=hashlib.sha256(content).hexdigest())
    changed = replace(package, files=tuple(files), package_digest="")
    findings = validate_package(changed).findings
    assert any(item.code == "frontmatter_mismatch" for item in findings)


def test_evolution_ablation_switches_are_independent():
    config = load_config(ROOT / "configs/dry_run.yaml")
    frozen = apply_evolution_ablation(config, "no-evolution")
    instruction = apply_evolution_ablation(config, "instructions-only")
    discovery = apply_evolution_ablation(config, "no-discovery")
    assert not frozen.evolution.enabled
    assert instruction.evolution.enabled
    assert instruction.evolution.scope == "instructions"
    assert not instruction.evolution.discovery
    assert discovery.evolution.scope == "package"
    assert not discovery.evolution.discovery
    with pytest.raises(ValueError, match="identical"):
        apply_evolution_ablation(config, "rule-proposer")
    with pytest.raises(ValueError, match="scope"):
        EvolutionConfig(scope="invalid")


def test_negative_transfer_constraint_has_a_real_switch():
    utility = TransferUtility(
        "candidate",
        3,
        ("a", "b"),
        0.1,
        0.01,
        (0.01, 0.2),
        0.5,
        0.8,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        {"a": 0.1, "b": 0.1},
    )
    enforced = MetaValidationGate(
        MetaLearningConfig(min_valid_episodes=1, enforce_negative_transfer=True)
    ).decide(utility)
    disabled = MetaValidationGate(
        MetaLearningConfig(min_valid_episodes=1, enforce_negative_transfer=False)
    ).decide(utility)
    assert "negative transfer rate exceeded" in enforced.failures
    assert "negative transfer rate exceeded" not in disabled.failures
    assert disabled.accepted


def test_disabled_evolution_produces_no_candidates():
    config = load_config(ROOT / "configs/dry_run.yaml")
    runner = ExperimentRunner(
        replace(config, evolution=replace(config.evolution, enabled=False)), ROOT
    )
    outcome = asyncio.run(runner.evolve_once())
    assert outcome["proposals"] == []
    assert outcome["package_candidates"] == {}


def test_evolve_evaluation_only_does_not_create_package_store(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(config, skill_store=tmp_path / "store", output_dir=tmp_path / "outputs")
    monkeypatch.setattr("evofact.cli.load_config", lambda _: config)
    args = build_parser().parse_args(["evolve", "--evaluation-only"])
    outcome = asyncio.run(_run(args))
    assert outcome["evaluation_only"] is True
    assert not config.skill_store.exists()
