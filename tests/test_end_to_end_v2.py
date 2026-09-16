from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from evofact.config import load_config
from evofact.core.generation_models import GenerationMetrics, PairedGeneratorEvaluation
from evofact.core.package_models import FileOperation, FileOperationKind, SkillPackagePatch
from evofact.evolution.package_candidate import build_package_candidate
from evofact.experiments.generator_runner import GeneratorEvolutionRunner
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.generation.split import split_real_only
from evofact.skills.package_loader import load_package
from evofact.skills.package_patch import apply_package_patch
from evofact.skills.repository import SkillRepository

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_dry_run_now_emits_trace_v2_with_execution_plan():
    runner = ExperimentRunner(load_config(ROOT / "configs" / "dry_run.yaml"), ROOT)
    traces, result = asyncio.run(runner.run(fixture_samples()[:1]))
    assert result.aggregate_metrics["n"] == 1
    assert traces[0].schema_version == "inference_trace_v2"
    assert traces[0].execution_plan is not None
    assert traces[0].node_executions


def test_complete_package_candidate_commit_and_rollback(tmp_path):
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    reference = package.file("references/rewriting_rules.md")
    candidate = apply_package_patch(
        package,
        SkillPackagePatch(
            package.skill_id,
            package.package_digest,
            (
                FileOperation(
                    FileOperationKind.UPDATE,
                    reference.path,
                    reference.content + b"\n- Preserve source dates exactly.\n",
                    expected_digest=reference.digest,
                ),
            ),
            None,
            "strengthen temporal fidelity",
        ),
    )
    repository = SkillRepository(tmp_path / "bank")
    repository.promote_package(package, run_id="baseline")
    repository.promote_package(candidate, run_id="candidate")
    repository.rollback_package(package.manifest.name, package.package_digest, run_id="rollback")
    assert repository.active_packages()[package.manifest.name] == package


def test_generator_challenger_is_gated_on_an_independent_episode():
    package = load_package(ROOT / "skills" / "seeds" / "generation_agent")
    reference = package.file("references/rewriting_rules.md")
    patch = SkillPackagePatch(
        package.skill_id,
        package.package_digest,
        (
            FileOperation(
                FileOperationKind.UPDATE,
                reference.path,
                reference.content + b"\n- Keep evidence identifiers unchanged.\n",
                expected_digest=reference.digest,
            ),
        ),
        None,
        "grounding improvement",
    )
    candidate = build_package_candidate(package, patch)
    champion = GenerationMetrics(0.8, 0.8, 0.8, 0.7, 0.1, 0.6, 0.1, 0, 0, Decimal("1"))
    challenger = GenerationMetrics(0.9, 0.9, 0.9, 0.8, 0.1, 0.7, 0.2, 0, 0, Decimal("1"))
    evaluation = PairedGeneratorEvaluation(
        "construction-episode",
        "independent-evaluation-episode",
        package.package_digest,
        candidate.package.package_digest,
        champion,
        challenger,
    )
    outcome = GeneratorEvolutionRunner(None).finalize(
        candidate,
        evaluation,
        run_id="evaluation-only",
        evaluation_only=True,
    )
    assert outcome.decision.accepted
    assert not outcome.committed


def test_real_only_split_is_fixed_before_augmentation():
    real = fixture_samples()
    before = split_real_only(real, seed="e2e")
    after = split_real_only(real, seed="e2e")
    assert before == after
    assert not set(before.meta_train_ids) & set(before.meta_test_ids)
    assert not set(before.meta_train_ids) & set(before.final_test_ids)
