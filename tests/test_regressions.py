"""Behavior regressions from the spec-to-code audit (offline only)."""

import asyncio
import sys
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_demse

from evofact.cli import build_parser
from evofact.config import AppConfig, MetaLearningConfig, load_config
from evofact.core.models import EvolutionOperation as Op
from evofact.core.models import (
    EvolutionProposal,
    RunBudget,
    SampleEvaluation,
    SkillScope,
    SkillStatus,
    TransferUtility,
    Trigger,
)
from evofact.data.domains import data_fingerprint, skillbank_fingerprint
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.routing.router import SkillRouter
from evofact.skills.candidates import apply_candidate
from evofact.skills.repository import SkillRepository
from evofact.validation.meta_gate import MetaValidationGate
from evofact.validation.statistics import mcnemar, paired_bootstrap
from evofact.validation.transfer import CrossEpisodeAggregator


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.runner = ExperimentRunner(AppConfig(), ROOT)

    def test_console_entry_exists(self):
        import evofact.cli

        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(config["project"]["scripts"]["evofact"], "evofact.cli:main")
        self.assertTrue(callable(evofact.cli.main))

    def test_empty_samples_do_not_use_fixtures(self):
        traces, result = asyncio.run(self.runner.run([]))
        self.assertEqual(traces, [])
        self.assertEqual(result.per_sample, ())

    def test_empty_bank_does_not_use_seed_bank(self):
        with self.assertRaisesRegex(RuntimeError, "judge"):
            asyncio.run(self.runner.run(fixture_samples()[:1], skills=[]))

    def test_validation_rejects_training_overlap(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.runner.closed_loop(fixture_samples(), fixture_samples()))

    def test_scopes_are_hard_filters_for_all_strategies(self):
        skill = next(s for s in self.runner.skills if s.name == "claim_decomposition")
        skill = replace(skill, scope=SkillScope(domains=("health",)))
        for strategy in ("static", "random", "all-experts", "utility-aware"):
            with self.subTest(strategy=strategy):
                route = SkillRouter(strategy).select(
                    {"domain": "finance"}, [skill], {}, RunBudget()
                )
                self.assertEqual(route.selected_skill_ids, ())

    def test_rollback_requires_snapshot(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["skills", "rollback", "claim_decomposition"])


class LifecycleRegressionTests(unittest.TestCase):
    def setUp(self):
        self.bank = ExperimentRunner(AppConfig(), ROOT).skills
        self.old = self.bank[0]
        self.a = replace(self.old, name="child_a", skill_id="child_a")
        self.b = replace(self.old, name="child_b", skill_id="child_b")

    def test_split_and_merge_replace_all_targets(self):
        split = EvolutionProposal("s", Op.SPLIT, "split", (self.old.name,), (self.a, self.b))
        bank = apply_candidate(self.bank, split)
        self.assertNotIn(self.old.name, [s.name for s in bank])
        self.assertEqual(len(bank), len(self.bank) + 1)
        merge = EvolutionProposal("m", Op.MERGE, "merge", (self.a.name, self.b.name), (self.old,))
        self.assertEqual(
            set(s.name for s in apply_candidate(bank, merge)), set(s.name for s in self.bank)
        )

    def test_retire_removes_target(self):
        proposal = EvolutionProposal("r", Op.RETIRE, "remove", (self.old.name,))
        self.assertEqual(len(apply_candidate(self.bank, proposal)), len(self.bank) - 1)

    def test_missing_frozen_duplicate_targets_rejected(self):
        proposals = [
            EvolutionProposal("x", Op.EDIT, "bad", ("missing",), (self.a,)),
            EvolutionProposal("x", Op.ADD, "bad", (), (self.old,)),
        ]
        for proposal in proposals:
            with self.assertRaises(ValueError):
                apply_candidate(self.bank, proposal)
        with self.assertRaisesRegex(ValueError, "frozen"):
            apply_candidate(
                [replace(self.old, status=SkillStatus.FROZEN)],
                EvolutionProposal("x", Op.EDIT, "bad", (self.old.name,), (self.a,)),
            )

    def test_atomic_failure_and_idempotent_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SkillRepository(Path(directory))
            repo.promote(self.old)
            before = repo.active_file.read_bytes()
            original = repo._atomic_json

            def fail_active(path, value):
                if path == repo.active_file:
                    raise OSError("injected pointer failure")
                return original(path, value)

            with patch.object(repo, "_atomic_json", side_effect=fail_active):
                with self.assertRaises(OSError):
                    repo.commit_bank([self.a], run_id="txn", baseline=[self.old], audit={})
            self.assertEqual(before, repo.active_file.read_bytes())
            first = repo.commit_bank([self.a], run_id="txn", baseline=[self.old], audit={})
            before = repo.active_file.read_bytes()
            second = repo.commit_bank([self.a], run_id="txn", baseline=[self.old], audit={})
            self.assertEqual(first, second)
            self.assertEqual(before, repo.active_file.read_bytes())

    def test_diff_and_snapshot_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SkillRepository(Path(directory))
            old = repo.promote(self.old)
            repo.promote(replace(self.old, instructions="changed instructions"))
            self.assertIn("+", repo.diff(self.old.name, old))
            self.assertIn("changed instructions", repo.diff(self.old.name, old))
            with self.assertRaises(ValueError):
                repo.get_snapshot("../../outside")


class ResumeRegressionTests(unittest.TestCase):
    def test_fingerprints_cover_labels_resources_and_triggers(self):
        rows = fixture_meta_samples()
        self.assertNotEqual(
            data_fingerprint(rows), data_fingerprint([replace(rows[0], label="FAKE"), *rows[1:]])
        )
        bank = ExperimentRunner(AppConfig(), ROOT).skills
        for change in (
            {"resources": {"references/a.md": "new"}},
            {"triggers": (Trigger("text", "cue"),)},
        ):
            self.assertNotEqual(
                skillbank_fingerprint(bank),
                skillbank_fingerprint([replace(bank[0], **change), *bank[1:]]),
            )

    def test_resume_rejects_changed_budget_or_domain_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "configs/demse_dry_run.yaml")
            config = replace(
                config,
                meta_learning=replace(
                    config.meta_learning, checkpoint_path=Path(directory) / "checkpoint.json"
                ),
            )
            asyncio.run(
                MetaEvolutionRunner(config, ROOT).run(
                    fixture_meta_samples(),
                    final_test_domains=("outer_holdout",),
                    evaluation_only=True,
                )
            )
            for changed, final in [
                (replace(config, max_skills_per_item=1), ("outer_holdout",)),
                (config, ("finance",)),
            ]:
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    asyncio.run(
                        MetaEvolutionRunner(changed, ROOT).run(
                            fixture_meta_samples(),
                            final_test_domains=final,
                            resume=True,
                            evaluation_only=True,
                        )
                    )

    def test_committed_run_resumes_without_recommit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "configs/demse_dry_run.yaml")
            config = replace(
                config,
                meta_learning=replace(
                    config.meta_learning, checkpoint_path=Path(directory) / "checkpoint.json"
                ),
            )
            repo = SkillRepository(Path(directory) / "bank")
            first = asyncio.run(
                MetaEvolutionRunner(config, ROOT, repo).run(
                    fixture_meta_samples(), final_test_domains=("outer_holdout",)
                )
            )
            before = repo.active_file.read_bytes()
            second = asyncio.run(
                MetaEvolutionRunner(config, ROOT, repo).run(
                    fixture_meta_samples(), final_test_domains=("outer_holdout",), resume=True
                )
            )
            self.assertEqual(first.episode_results, second.episode_results)
            self.assertEqual(first.committed_snapshots, second.committed_snapshots)
            self.assertEqual(before, repo.active_file.read_bytes())


class StatisticalRegressionTests(unittest.TestCase):
    def test_invalid_pairs_fail(self):
        row = SampleEvaluation("a", "REAL", "REAL", 0.8)
        for other in ([], [row, row], [replace(row, gold="FAKE")]):
            for function in (paired_bootstrap, mcnemar):
                with self.assertRaises(ValueError):
                    function([row], other)

    def test_duplicate_episode_does_not_inflate_evidence(self):
        row = test_demse.TransferGateTests.result("health", False, True)
        with self.assertRaises(ValueError):
            CrossEpisodeAggregator().aggregate([row, row])

    def test_specialization_cannot_bypass_hard_constraints(self):
        base = TransferUtility(
            "c",
            3,
            ("a", "b"),
            0.01,
            0.2,
            (-0.1, 0.1),
            0.5,
            0.5,
            -0.05,
            1,
            0,
            0,
            1,
            {"a": 0.2, "b": -0.05},
        )
        gate = MetaValidationGate(MetaLearningConfig(enabled=True))
        for changes in (
            {"mean_coverage": 0.1},
            {"cost_ratio": 100},
            {"calibration_delta": 1},
            {"episode_count": 2},
        ):
            self.assertFalse(gate.decide(replace(base, **changes)).accepted)


if __name__ == "__main__":
    unittest.main()
