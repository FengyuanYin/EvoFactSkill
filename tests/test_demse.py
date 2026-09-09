import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from evofact.config import MetaLearningConfig, load_config
from evofact.core.models import (
    DomainEpisode,
    EpisodeEvaluation,
    EvolutionOperation,
    EvolutionProposal,
    SampleEvaluation,
    SkillStatus,
    TransferUtility,
)
from evofact.data.domains import split_source_and_final
from evofact.data.episodes import DomainEpisodeSampler
from evofact.evolution.firewall import CandidateFirewall
from evofact.evolution.identity import candidate_identity
from evofact.experiments.checkpoint import MetaCheckpointStore
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.skills.loader import load_skill_package
from evofact.skills.repository import SkillRepository
from evofact.validation.evaluator import evaluate
from evofact.validation.meta_gate import MetaValidationGate
from evofact.validation.transfer import CrossEpisodeAggregator


class DomainEpisodeTests(unittest.TestCase):
    def test_deterministic_disjoint_and_covering(self):
        samples = fixture_meta_samples()
        source, final = split_source_and_final(samples, ("outer_holdout",))
        config = MetaLearningConfig(enabled=True, episodes=2)
        first = DomainEpisodeSampler(config, 7).build(samples, source, final, "bank")
        second = DomainEpisodeSampler(config, 7).build(
            list(reversed(samples)), source, final, "bank"
        )
        self.assertEqual(first, second)
        self.assertEqual(set().union(*(set(x.meta_test_domains) for x in first)), set(source))
        self.assertTrue(
            all(
                not set(x.meta_train_domains) & set(x.meta_test_domains)
                and "outer_holdout" not in x.meta_train_domains + x.meta_test_domains
                for x in first
            )
        )

    def test_leave_one_out_and_invalid_overlap(self):
        samples = fixture_meta_samples()
        source, final = split_source_and_final(samples, ("outer_holdout",))
        config = MetaLearningConfig(enabled=True, strategy="leave_one_domain_out")
        episodes = DomainEpisodeSampler(config).build(samples, source, final, "bank")
        self.assertEqual(len(episodes), len(source))
        with self.assertRaises(ValueError):
            DomainEpisode("x", 1, "repeated_holdout", ("a",), ("a",), ("1",), ("2",))


class FirewallIdentityTests(unittest.TestCase):
    def test_identity_ignores_random_ids(self):
        seed = load_skill_package(
            ROOT / "skills/seeds/claim_decomposition", status=SkillStatus.CANDIDATE
        )
        a = EvolutionProposal("a", EvolutionOperation.EDIT, "r", (seed.name,), (seed,))
        b = replace(a, proposal_id="b", candidate_skills=(replace(seed, skill_id="random"),))
        self.assertEqual(candidate_identity(a).fingerprint, candidate_identity(b).fingerprint)

    def test_firewall_blocks_heldout_content(self):
        samples = fixture_meta_samples()
        source, final = split_source_and_final(samples, ("outer_holdout",))
        episode = DomainEpisodeSampler(MetaLearningConfig(enabled=True), 1).build(
            samples, source, final, "bank"
        )[0]
        heldout = next(
            sample for sample in samples if sample.sample_id in episode.meta_test_sample_ids
        )
        seed = load_skill_package(
            ROOT / "skills/seeds/claim_decomposition", status=SkillStatus.CANDIDATE
        )
        proposal = EvolutionProposal(
            "leak",
            EvolutionOperation.EDIT,
            "r",
            (seed.name,),
            (replace(seed, instructions=heldout.text),),
        )
        with self.assertRaises(ValueError):
            CandidateFirewall(samples, final).validate_candidate(proposal, episode)


class TransferGateTests(unittest.TestCase):
    @staticmethod
    def result(domain, old_correct, new_correct, fingerprint="c"):
        old = evaluate(
            [SampleEvaluation(domain, "FAKE", "FAKE" if old_correct else "REAL", 0.8, domain)]
        )
        new = evaluate(
            [SampleEvaluation(domain, "FAKE", "FAKE" if new_correct else "REAL", 0.8, domain)]
        )
        delta = {
            k: new.aggregate_metrics[k] - old.aggregate_metrics[k] for k in old.aggregate_metrics
        }
        return EpisodeEvaluation(domain, fingerprint, "p", old, new, delta, {domain: delta})

    def test_aggregate_and_generalize(self):
        utility = CrossEpisodeAggregator(1).aggregate(
            [self.result(x, False, True) for x in ("a", "b", "c")]
        )["c"]
        decision = MetaValidationGate(MetaLearningConfig(enabled=True, min_coverage=0)).decide(
            utility
        )
        self.assertEqual(decision.disposition, "generalized")
        self.assertTrue(decision.accepted)
        self.assertGreater(utility.confidence_interval[0], 0)

    def test_no_cross_episode_aggregation_ablation(self):
        rows = [self.result(x, False, True) for x in ("a", "b", "c")]
        utility = CrossEpisodeAggregator(1, aggregate_across_episodes=False).aggregate(rows)["c"]
        self.assertEqual(utility.episode_count, 1)

    def test_specialize_and_reject(self):
        utility = TransferUtility(
            "x",
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
            {"a": 0.2, "b": -0.18},
        )
        decision = MetaValidationGate(
            MetaLearningConfig(enabled=True, min_coverage=0, max_negative_transfer_rate=0.25)
        ).decide(utility)
        self.assertEqual(decision.disposition, "specialized")
        self.assertEqual(decision.target_domains, ("a",))
        blocked = MetaValidationGate(MetaLearningConfig(enabled=True)).decide(
            utility, safety_level="blocked"
        )
        self.assertEqual(blocked.disposition, "rejected")

    def test_pareto_review_retire_and_ablation_switches(self):
        pareto = TransferUtility(
            "p",
            3,
            ("a", "b"),
            0.02,
            0.1,
            (-0.01, 0.05),
            0.7,
            0,
            0,
            1,
            0,
            0,
            1,
            {"a": 0.02, "b": 0.02},
        )
        gate = MetaValidationGate(MetaLearningConfig(enabled=True, min_coverage=0))
        self.assertEqual(gate.decide(pareto).disposition, "pareto")
        self.assertEqual(
            gate.decide(pareto, safety_level="review_required").disposition, "review_required"
        )
        harmful = TransferUtility(
            "r",
            3,
            ("a", "b"),
            -0.2,
            0.01,
            (-0.3, -0.1),
            0,
            1,
            -0.3,
            1,
            0,
            0,
            1,
            {"a": -0.2, "b": -0.3},
        )
        self.assertEqual(gate.decide(harmful, retirement_candidate=True).disposition, "rejected")
        helpful_removal = replace(
            harmful,
            mean_gain=0.2,
            confidence_interval=(0.1, 0.3),
            negative_transfer_rate=0,
            worst_domain_drop=0,
            domain_gains={"a": 0.2, "b": 0.3},
        )
        self.assertEqual(
            gate.decide(helpful_removal, retirement_candidate=True).disposition, "retired"
        )
        regression = TransferUtility(
            "w",
            3,
            ("a", "b"),
            0.1,
            0.01,
            (0.05, 0.15),
            1,
            0,
            -0.2,
            1,
            0,
            0,
            1,
            {"a": 0.2, "b": -0.2},
        )
        strict = MetaValidationGate(
            MetaLearningConfig(
                enabled=True, min_coverage=0, max_worst_domain_drop=0.1, allow_specialization=False
            )
        )
        relaxed = MetaValidationGate(
            MetaLearningConfig(
                enabled=True,
                min_coverage=0,
                max_worst_domain_drop=0.1,
                allow_specialization=False,
                enforce_worst_domain=False,
            )
        )
        self.assertNotEqual(strict.decide(regression).disposition, "generalized")
        self.assertEqual(relaxed.decide(regression).disposition, "generalized")


class MetaCheckpointAndRunnerTests(unittest.TestCase):
    def test_checkpoint_fingerprint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MetaCheckpointStore(Path(directory) / "c.json")
            store.save(
                {"config_fingerprint": "a", "data_fingerprint": "b", "skillbank_snapshot_id": "c"}
            )
            self.assertTrue(
                store.load(config_fingerprint="a", data_fingerprint="b", skillbank_snapshot_id="c")
            )
            with self.assertRaises(ValueError):
                store.load(config_fingerprint="x", data_fingerprint="b", skillbank_snapshot_id="c")

    def test_end_to_end_evaluation_only(self):
        config = load_config(ROOT / "configs/demse_dry_run.yaml")
        with tempfile.TemporaryDirectory() as directory:
            meta = replace(
                config.meta_learning, checkpoint_path=Path(directory) / "checkpoint.json"
            )
            runner = MetaEvolutionRunner(replace(config, meta_learning=meta), ROOT)
            outcome = asyncio.run(
                runner.run(
                    fixture_meta_samples(),
                    final_test_domains=("outer_holdout",),
                    evaluation_only=True,
                )
            )
            self.assertEqual(len(outcome.episodes), 5)
            self.assertTrue(outcome.episode_results)
            self.assertTrue(outcome.decisions)
            self.assertFalse(outcome.committed_snapshots)
            resumed = asyncio.run(
                runner.run(
                    fixture_meta_samples(),
                    final_test_domains=("outer_holdout",),
                    resume=True,
                    evaluation_only=True,
                )
            )
            self.assertEqual(outcome.episode_results, resumed.episode_results)

    def test_atomic_commit_uses_repository_snapshot(self):
        config = load_config(ROOT / "configs/demse_dry_run.yaml")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repository = SkillRepository(base / "skills")
            for path in sorted((ROOT / "skills/seeds").iterdir()):
                if (path / "SKILL.md").is_file():
                    repository.promote(load_skill_package(path))
            before = set(repository.active())
            meta = replace(config.meta_learning, checkpoint_path=base / "checkpoint.json")
            outcome = asyncio.run(
                MetaEvolutionRunner(replace(config, meta_learning=meta), ROOT, repository).run(
                    fixture_meta_samples(), final_test_domains=("outer_holdout",)
                )
            )
            self.assertTrue(outcome.committed_snapshots)
            self.assertEqual(set(repository.active()), before)
            self.assertTrue(
                any(event["action"] == "meta_promote" for event in repository.history())
            )


if __name__ == "__main__":
    unittest.main()
