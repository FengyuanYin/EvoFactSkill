import asyncio
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.attribution.rules import attribute_trace
from evofact.cli import _run, build_parser
from evofact.config import load_config
from evofact.core.models import Prediction
from evofact.experiments.adversarial_runner import AdversarialEvolutionRunner
from evofact.experiments.runner import ExperimentRunner, _gold
from evofact.generation.data import fixture_adversarial_data, load_samples, validate_fact_links
from evofact.generation.generator import ChallengeGenerator
from evofact.generation.models import GenerationConfig
from evofact.generation.prompts import GENERATOR_SYSTEM, VERIFIER_SYSTEM
from evofact.generation.split import split_construction_probe
from evofact.generation.verifier import ChallengeVerifier
from evofact.reporting.adversarial_report import write_adversarial_report
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.skills.repository import SkillRepository


class FakeJSONBackend:
    """Test double only; no offline generation mode in production."""

    def __init__(self):
        self.calls = []

    async def _call(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if system == VERIFIER_SYSTEM:
            return {
                "reviews": [
                    {
                        "sample_id": s["sample_id"],
                        "label": "REAL" if s["text"].startswith("记录确认") else "FAKE",
                        "reason": "fixture evidence check",
                    }
                    for s in payload["samples"]
                ]
            }
        assert system == GENERATOR_SYSTEM
        rows, decisions = [], []
        for index in range(payload["batch_size"]):
            example = payload["examples"][index % len(payload["examples"])]
            source = example["source_sample"]
            row = deepcopy(source)
            row["sample_id"] = payload["id_prefix"] + str(index)
            row["label"] = "REAL" if index % 2 == 0 else "FAKE"
            row["text"] = (
                ("记录确认：" if row["label"] == "REAL" else "记录否认：")
                + source["evidence"][0]["text"]
                + f"（第{index}份转述）"
            )
            rows.append(row)
            decisions.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sample_id": source["sample_id"],
                    "strategy": "success_boundary_extension"
                    if example["outcome"] == "success"
                    else "minimal_contrast",
                    "reason": "成功类型升难度"
                    if example["outcome"] == "success"
                    else "失败类型增样强化",
                    "source_trace_ids": [example["trace"]["trace_id"]],
                }
            )
        return {"samples": rows, "decisions": decisions}


def config_for(directory):
    config = load_config(ROOT / "configs/adversarial_llm.yaml")
    return replace(
        config,
        backend="mock",
        model="mock-v1",
        output_dir=directory / "reports",
        skill_store=directory / "detector",
        generation=replace(
            config.generation, batch_size=4, store_path=directory / "generator.json"
        ),
        meta_learning=replace(config.meta_learning, checkpoint_path=directory / "checkpoint.json"),
    )


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.facts = fixture_adversarial_data()
        self.config = GenerationConfig(batch_size=4)
        self.backend = FakeJSONBackend()
        runner = ExperimentRunner(load_config(ROOT / "configs/dry_run.yaml"), ROOT)
        traces, _ = asyncio.run(runner.run(self.samples[:4]))
        # Explicit success and failure cases, independent of mock detector heuristics.
        traces[0] = replace(traces[0], decision=Prediction(_gold(self.samples[0].label), 0.9))
        traces[1] = replace(traces[1], decision=Prediction("REAL", 0.9))
        reports = [attribute_trace(t, _gold(s.label)) for t, s in zip(traces, self.samples[:4])]
        self.request = ChallengeGenerator().build_request(
            self.samples[:4], traces, reports, config=self.config, episode_id="test"
        )
        self.response = asyncio.run(ChallengeGenerator().generate(self.backend, self.request))

    def test_only_llm_config_and_success_failure_context(self):
        with self.assertRaisesRegex(ValueError, "only supports llm"):
            GenerationConfig(proposer="deterministic")
        with self.assertRaises(TypeError):
            GenerationConfig(mode="weights")
        self.assertEqual({e["outcome"] for e in self.request["examples"]}, {"success", "failure"})
        self.assertIn("同失败类型", GENERATOR_SYSTEM)
        self.assertIn("更难", GENERATOR_SYSTEM)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(set(self.response), {"samples", "decisions"})

    def test_free_text_survives_and_reviewer_is_blind(self):
        checked, rejected = ChallengeVerifier().validate(
            self.response, self.request, config=self.config, existing=self.samples[:4]
        )
        self.assertFalse(rejected)
        accepted, rejected, _ = asyncio.run(ChallengeVerifier().verify(checked, self.backend))
        self.assertEqual(len(accepted), 4)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0].text, self.response["samples"][0]["text"])
        for sample in accepted:
            self.assertEqual(sample.dataset, "fixture")
            self.assertEqual(sample.public_view()["metadata"], {})
            self.assertNotIn("label", sample.public_view())
        review_payload = self.backend.calls[-1][1]
        self.assertNotIn("decisions", review_payload)
        self.assertTrue(
            all(set(s) == {"sample_id", "text", "evidence"} for s in review_payload["samples"])
        )

    def test_provenance_schema_and_leakage_reject(self):
        for field, value in (
            ("dataset", "held-out"),
            ("event_id", "unknown"),
            ("metadata", {"strategy": "secret"}),
            ("label", "MAYBE"),
            ("text", ""),
            ("evidence", []),
        ):
            response = deepcopy(self.response)
            response["samples"][0][field] = value
            accepted, rejected = ChallengeVerifier().validate(
                response, self.request, config=self.config
            )
            self.assertEqual(len(accepted), 3, field)
            self.assertTrue(rejected, field)
        for field, value in (
            ("source_sample_id", "held-out"),
            ("source_trace_ids", ["held-out"]),
            ("strategy", "unrecognized"),
        ):
            response = deepcopy(self.response)
            response["decisions"][0][field] = value
            _, rejected = ChallengeVerifier().validate(response, self.request, config=self.config)
            self.assertTrue(rejected)
        response = deepcopy(self.response)
        response["samples"][0]["text"] = self.samples[-1].text
        _, rejected = ChallengeVerifier().validate(
            response, self.request, config=self.config, forbidden=self.samples[-2:]
        )
        self.assertTrue(rejected)
        response = deepcopy(self.response)
        response["samples"][0]["text"] = self.samples[0].text
        _, rejected = ChallengeVerifier().validate(
            response, self.request, config=self.config, existing=self.samples
        )
        self.assertTrue(rejected)

    def test_bad_batch_rejected_without_retry(self):
        for response in (
            {},
            {"samples": [], "decisions": [{}]},
            {"samples": self.response["samples"] * 2, "decisions": self.response["decisions"] * 2},
        ):
            with self.assertRaises(ValueError):
                ChallengeVerifier().validate(response, self.request, config=self.config)
        self.assertEqual(len(self.backend.calls), 1)

    def test_unknown_or_wrong_label_not_used_for_training(self):
        checked, _ = ChallengeVerifier().validate(self.response, self.request, config=self.config)

        class Reviewer:
            async def _call(self, system, payload):
                return {
                    "reviews": [
                        {"sample_id": s["sample_id"], "label": "UNKNOWN", "reason": "insufficient"}
                        for s in payload["samples"]
                    ]
                }

        accepted, rejected, _ = asyncio.run(ChallengeVerifier().verify(checked, Reviewer()))
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 4)

        class WrongReviewer:
            async def _call(self, system, payload):
                return {
                    "reviews": [
                        {
                            "sample_id": s.sample_id,
                            "label": "FAKE" if s.label == "REAL" else "REAL",
                            "reason": "contradictory verdict",
                        }
                        for s in checked
                    ]
                }

        accepted, rejected, _ = asyncio.run(ChallengeVerifier().verify(checked, WrongReviewer()))
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 4)

        class MissingReviewer:
            async def _call(self, system, payload):
                return {"reviews": []}

        with self.assertRaisesRegex(ValueError, "every sample"):
            asyncio.run(ChallengeVerifier().verify(checked, MissingReviewer()))

    def test_split_preserves_groups_without_numeric_facts(self):
        train = [s for s in self.samples if s.domain != "outer_holdout"]
        construction, probe, facts = split_construction_probe(train, [], fraction=0.4, seed="x")
        self.assertFalse({s.event_id for s in construction} & {s.event_id for s in probe})
        self.assertFalse(
            {e.text for s in construction for e in s.evidence}
            & {e.text for s in probe for e in s.evidence}
        )
        self.assertFalse(facts)
        self.assertEqual(
            (construction, probe, facts),
            split_construction_probe(list(reversed(train)), [], fraction=0.4, seed="x"),
        )
        with self.assertRaisesRegex(ValueError, "two independent"):
            split_construction_probe(self.samples[:2], [], fraction=0.4, seed="x")
        with self.assertRaises(ValueError):
            validate_fact_links(self.samples, [replace(self.facts[0], source="wrong")])

    def test_source_snapshot_groups_even_when_evidence_wording_differs(self):
        samples = list(self.samples[:8])
        shared = "archive://shared-document"
        for i in (0, 1, 2, 3):
            samples[i] = replace(
                samples[i],
                evidence=(replace(samples[i].evidence[0], source=shared, text=f"不同段落 {i}"),),
            )
        construction, probe, _ = split_construction_probe(samples, [], fraction=0.4, seed="x")
        self.assertTrue(
            {samples[i].sample_id for i in range(4)} <= {s.sample_id for s in construction}
            or {samples[i].sample_id for i in range(4)} <= {s.sample_id for s in probe}
        )

    def test_real_backend_serializes_evidence_dates_without_network(self):
        backend = OpenAICompatibleBackend("https://example.invalid/v1", "test-key", "test-model")
        with patch("evofact.runtime.openai_backend.urllib.request.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": {"content": '{"samples": [], "decisions": []}'}}]}
            ).encode()
            asyncio.run(backend._call("test", {"published_at": datetime(2026, 1, 1)}))
            body = json.loads(request.call_args.args[0].data)
            self.assertEqual(
                json.loads(body["messages"][1]["content"])["published_at"], "2026-01-01 00:00:00"
            )


class AdversarialRunnerTests(unittest.TestCase):
    def test_future_evidence_and_cross_domain_sources_rejected_before_calls(self):
        samples, _ = fixture_adversarial_data()
        when = datetime(2026, 1, 1)
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            variants = [
                [
                    replace(
                        samples[0],
                        published_at=when,
                        evidence=(
                            replace(samples[0].evidence[0], published_at=when + timedelta(days=1)),
                        ),
                    ),
                    *samples[1:],
                ],
                [
                    replace(
                        samples[0],
                        evidence=(
                            replace(samples[0].evidence[0], source=samples[-1].evidence[0].source),
                        ),
                    ),
                    *samples[1:],
                ],
            ]
            for rows in variants:
                backend = FakeJSONBackend()
                with self.assertRaisesRegex(ValueError, "future evidence|crosses adversarial"):
                    asyncio.run(
                        AdversarialEvolutionRunner(config, ROOT, generation_backend=backend).run(
                            rows, final_test_domains=("outer_holdout",), evaluation_only=True
                        )
                    )
                self.assertFalse(backend.calls)

    def test_end_to_end_one_call_isolation_and_resume(self):
        samples, facts = fixture_adversarial_data()
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            backend = FakeJSONBackend()
            runner = AdversarialEvolutionRunner(config, ROOT, facts, generation_backend=backend)
            outcome = asyncio.run(
                runner.run(samples, final_test_domains=("outer_holdout",), evaluation_only=True)
            )
            self.assertEqual(len(outcome.episodes), 5)
            self.assertTrue(outcome.episode_results)
            self.assertEqual(sum(s == GENERATOR_SYSTEM for s, _ in backend.calls), 5)
            self.assertEqual(sum(s == VERIFIER_SYSTEM for s, _ in backend.calls), 5)
            self.assertFalse(config.generation.store_path.exists())
            self.assertFalse((config.skill_store / "active.json").exists())
            for episode in outcome.episodes:
                record = runner.audit[episode.episode_id]
                self.assertEqual(
                    set(record["construction_ids"]) | set(record["probe_ids"]),
                    set(episode.meta_train_sample_ids),
                )
                blob = json.dumps(record["request"], ensure_ascii=False)
                forbidden = set(record["probe_ids"]) | set(episode.meta_test_sample_ids)
                forbidden |= {s.sample_id for s in samples if s.domain == "outer_holdout"}
                self.assertTrue(all(sid not in blob for sid in forbidden))
                self.assertEqual(record["metrics"]["accepted"], 4)
                self.assertEqual(record["generator_calls"], 1)
            paths = write_adversarial_report(
                config.output_dir, outcome, runner.audit, config.generation
            )
            for target in paths["sample_files"].values():
                self.assertEqual(len(load_samples(target)), 4)
            restored = AdversarialEvolutionRunner(
                config, ROOT, facts, generation_backend=FakeJSONBackend()
            )
            with patch.object(
                restored.generator,
                "generate",
                side_effect=AssertionError("completed episode repeated"),
            ):
                resumed = asyncio.run(
                    restored.run(
                        samples,
                        final_test_domains=("outer_holdout",),
                        resume=True,
                        evaluation_only=True,
                    )
                )
            self.assertEqual(outcome.episode_results, resumed.episode_results)
            self.assertEqual(runner.audit, restored.audit)

    def test_commit_audit_idempotent_no_policy_weights(self):
        samples, facts = fixture_adversarial_data()
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            repo = SkillRepository(config.skill_store)
            runner = AdversarialEvolutionRunner(
                config, ROOT, facts, repo, generation_backend=FakeJSONBackend()
            )
            outcome = asyncio.run(runner.run(samples, final_test_domains=("outer_holdout",)))
            before = config.generation.store_path.read_bytes()
            audit = json.loads(before)
            self.assertIn(outcome.run_id, audit["runs"])
            self.assertNotIn("scoped_policies", audit)
            restored = AdversarialEvolutionRunner(
                config, ROOT, facts, repo, generation_backend=FakeJSONBackend()
            )
            asyncio.run(restored.run(samples, final_test_domains=("outer_holdout",), resume=True))
            self.assertEqual(before, config.generation.store_path.read_bytes())

    def test_changed_generation_budget_cannot_resume(self):
        samples, facts = fixture_adversarial_data()
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            asyncio.run(
                AdversarialEvolutionRunner(
                    config, ROOT, facts, generation_backend=FakeJSONBackend()
                ).run(samples, final_test_domains=("outer_holdout",), evaluation_only=True)
            )
            changed = replace(config, generation=replace(config.generation, batch_size=6))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                asyncio.run(
                    AdversarialEvolutionRunner(
                        changed, ROOT, facts, generation_backend=FakeJSONBackend()
                    ).run(
                        samples,
                        final_test_domains=("outer_holdout",),
                        resume=True,
                        evaluation_only=True,
                    )
                )

    def test_cli_samples_without_facts_and_no_mock_fallback(self):
        samples, _ = fixture_adversarial_data()
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            sample_path = Path(temp) / "samples.jsonl"
            sample_path.write_text(
                "\n".join(json.dumps(asdict(s), ensure_ascii=False) for s in samples),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["adversarial-evolve", "--samples", str(sample_path), "--evaluation-only"]
            )
            with (
                patch("evofact.cli.load_config", return_value=config),
                patch.object(ExperimentRunner, "_backend", wraps=None) as unused,
            ):
                with self.assertRaisesRegex(ValueError, "real backend"):
                    asyncio.run(_run(args))
                unused.assert_not_called()
            original = AdversarialEvolutionRunner.__init__

            def injected(instance, *args, **kwargs):
                original(instance, *args, **kwargs, generation_backend=FakeJSONBackend())

            with (
                patch("evofact.cli.load_config", return_value=config),
                patch.object(AdversarialEvolutionRunner, "__init__", injected),
            ):
                result = asyncio.run(_run(args))
            self.assertEqual(result["proposer"], "llm")
            self.assertEqual(result["generator_calls"], 5)
            self.assertEqual(result["accepted_samples"], 20)

    def test_interrupt_after_generation_reuses_response(self):
        samples, _ = fixture_adversarial_data()
        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            backend = FakeJSONBackend()
            runner = AdversarialEvolutionRunner(config, ROOT, generation_backend=backend)
            with patch.object(
                runner.verifier, "verify", side_effect=RuntimeError("injected interruption")
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    asyncio.run(
                        runner.run(
                            samples, final_test_domains=("outer_holdout",), evaluation_only=True
                        )
                    )
            self.assertEqual(sum(s == GENERATOR_SYSTEM for s, _ in backend.calls), 1)
            restored = AdversarialEvolutionRunner(config, ROOT, generation_backend=backend)
            asyncio.run(
                restored.run(
                    samples,
                    final_test_domains=("outer_holdout",),
                    resume=True,
                    evaluation_only=True,
                )
            )
            self.assertEqual(sum(s == GENERATOR_SYSTEM for s, _ in backend.calls), 5)

    def test_all_rejected_falls_back_to_original_construction(self):
        samples, _ = fixture_adversarial_data()

        class EmptyBackend(FakeJSONBackend):
            async def _call(self, system, payload):
                self.calls.append((system, deepcopy(payload)))
                return {"samples": [], "decisions": []}

        with tempfile.TemporaryDirectory() as temp:
            config = config_for(Path(temp))
            backend = EmptyBackend()
            runner = AdversarialEvolutionRunner(config, ROOT, generation_backend=backend)
            outcome = asyncio.run(
                runner.run(samples, final_test_domains=("outer_holdout",), evaluation_only=True)
            )
            self.assertEqual(len(outcome.episodes), 5)
            self.assertEqual(len(backend.calls), 5)
            self.assertTrue(all(e["metrics"]["accepted"] == 0 for e in runner.audit.values()))

    def test_ordinary_config_keeps_generation_disabled(self):
        self.assertFalse(load_config(ROOT / "configs/dry_run.yaml").generation.enabled)


if __name__ == "__main__":
    unittest.main()
