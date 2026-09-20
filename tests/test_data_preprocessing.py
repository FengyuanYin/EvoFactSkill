import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.cli import _run, build_parser
from evofact.config import DataConfig, load_config
from evofact.core.models import Evidence, Sample
from evofact.data.adapters.amtcele import AMTCeleAdapter
from evofact.data.deduplication import deduplicate_samples, normalize_text
from evofact.data.domains import domain_index, split_source_and_final
from evofact.data.manifests import build_manifest
from evofact.data.registry import DataRegistry
from evofact.governance.pricing_policy import load_pricing_table


class DeduplicationTests(unittest.TestCase):
    def test_normalization_and_cross_domain_conflicts(self):
        self.assertEqual(normalize_text("  Ａ\n  B  "), "a b")
        rows = (
            Sample("a", "fixture", "Same text", "REAL", "tech"),
            Sample("b", "fixture", " same   text ", "REAL", "tech"),
            Sample("c", "fixture", "Cross domain", "FAKE", "tech"),
            Sample("d", "fixture", "cross domain", "FAKE", "politics"),
            Sample("e", "fixture", "Label conflict", "REAL", "tech"),
            Sample("f", "fixture", "label conflict", "FAKE", "tech"),
        )
        result = deduplicate_samples(rows)
        self.assertEqual([sample.sample_id for sample in result], ["a"])

    def test_domain_partition_rejects_mixed_datasets_and_duplicate_content(self):
        with self.assertRaisesRegex(ValueError, "exactly one dataset"):
            split_source_and_final(
                (
                    Sample("a", "first", "one", domain="source"),
                    Sample("b", "second", "two", domain="held-out"),
                ),
                ("held-out",),
            )
        with self.assertRaisesRegex(ValueError, "duplicate content"):
            domain_index(
                (
                    Sample("a", "fixture", "Same text", domain="source"),
                    Sample("b", "fixture", " same  text ", domain="held-out"),
                )
            )

    def test_manifest_uses_configured_domain_partition(self):
        rows = (
            Sample("a", "fixture", "source one", domain="source-a"),
            Sample("b", "fixture", "source two", domain="source-b"),
            Sample("c", "fixture", "held out", domain="final"),
        )
        manifest = build_manifest(
            rows,
            train_domains=("source-a", "source-b"),
            final_test_domains=("final",),
        )
        self.assertEqual(manifest.test_ids, ("c",))
        self.assertNotIn("c", manifest.train_ids)
        self.assertEqual(
            manifest.split_policy["final_test_domains"],
            ["final"],
        )
        with self.assertRaisesRegex(ValueError, "unassigned domains"):
            build_manifest(
                rows,
                train_domains=("source-a",),
                final_test_domains=("final",),
            )

    def test_data_config_rejects_overlapping_domains(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            DataConfig(
                dataset="weibo21",
                root=Path("datasets/Weibo21"),
                train_domains=("health",),
                final_test_domains=("health",),
            )

    def test_public_view_is_an_explicit_allow_list(self):
        published_at = datetime(2026, 1, 2, 3, 4, 5)
        sample = Sample(
            "secret-id",
            "secret-dataset",
            "public claim",
            "FAKE",
            "secret-domain",
            "secret-event",
            published_at,
            (Evidence("public evidence", "source", published_at, "refute"),),
            {"split": "test", "gold": "FAKE"},
        )
        public = sample.public_view()
        self.assertEqual(set(public), {"text", "published_at", "evidence"})
        self.assertEqual(
            set(public["evidence"][0]),
            {"text", "source", "published_at"},
        )
        blob = json.dumps(public, ensure_ascii=False)
        for secret in (
            "secret-id",
            "secret-dataset",
            "secret-domain",
            "secret-event",
            "FAKE",
            "refute",
            "test",
        ):
            self.assertNotIn(secret, blob)


class AdapterTests(unittest.TestCase):
    def test_weibo_uses_only_aggregate_and_deduplicates_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text(
                json.dumps({"content": "must be ignored", "label": 1, "category": "tech"}) + "\n",
                encoding="utf-8",
            )
            rows = [
                {"content": "same", "label": 0, "category": "tech", "split": "train"},
                {"content": " same ", "label": 0, "category": "tech", "split": "test"},
                {"content": "bridge", "label": 1, "category": "tech", "split": "train"},
                {
                    "content": "BRIDGE",
                    "label": 1,
                    "category": "politics",
                    "split": "test",
                },
                {"content": "unique", "label": 1, "category": "health", "split": "val"},
                {"content": "unknown", "label": 0, "category": "无法确定", "split": "train"},
            ]
            (root / "weibo21_all.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            samples = DataRegistry().load(
                "weibo21",
                root,
                excluded_domains=("无法确定",),
            )
            self.assertEqual({sample.text.strip() for sample in samples}, {"same", "unique"})
            self.assertTrue(all("official_split" not in sample.metadata for sample in samples))
            self.assertEqual(
                {sample.metadata["source_file"] for sample in samples},
                {"weibo21_all.jsonl"},
            )

    def test_amtcele_preserves_pair_as_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AMTCele.jsonl").write_text(
                json.dumps({"domain": "biz01", "label": "legit", "text": "claim"}) + "\n",
                encoding="utf-8",
            )
            sample = next(iter(AMTCeleAdapter().load(root)))
            self.assertEqual(sample.domain, "biz")
            self.assertEqual(sample.event_id, "biz01")
            self.assertEqual(sample.metadata["pair_id"], "biz01")
            self.assertEqual(sample.sample_id, replace(sample, metadata={}).sample_id)

    def test_cross_domain_configs_isolate_datasets_and_state(self):
        weibo = load_config(ROOT / "configs/weibo21_cross_domain.yaml")
        amt = load_config(ROOT / "configs/amtcele_cross_domain.yaml")
        self.assertTrue(weibo.meta_learning.enabled)
        self.assertTrue(amt.meta_learning.enabled)
        self.assertEqual(weibo.data.dataset, "weibo21")
        self.assertEqual(amt.data.dataset, "amtcele")
        self.assertEqual(len(weibo.data.final_test_domains), 1)
        self.assertEqual(len(amt.data.final_test_domains), 1)
        self.assertFalse(set(weibo.data.train_domains) & set(weibo.data.final_test_domains))
        self.assertFalse(set(amt.data.train_domains) & set(amt.data.final_test_domains))
        self.assertEqual(weibo.data.train_sampling, "balanced_by_domain")
        self.assertEqual(weibo.data.final_test_samples_per_domain, 100)
        self.assertTrue(weibo.data.require_static_test)
        self.assertNotEqual(weibo.output_dir, amt.output_dir)
        self.assertNotEqual(weibo.skill_store, amt.skill_store)
        self.assertNotEqual(
            weibo.meta_learning.checkpoint_path,
            amt.meta_learning.checkpoint_path,
        )
        for config in (weibo, amt):
            table = load_pricing_table(ROOT / config.pricing.table_path)
            self.assertEqual(table.provider, config.pricing.provider)
            self.assertIn(config.model, table.models)

    def test_cli_selects_dataset_and_domains_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            data_root = temp_root / "weibo"
            data_root.mkdir()
            rows = (
                {"content": "source", "label": 0, "category": "tech", "split": "test"},
                {"content": "held out", "label": 1, "category": "health", "split": "train"},
                {"content": "excluded", "label": 1, "category": "unknown", "split": "train"},
            )
            (data_root / "weibo21_all.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            config_path = temp_root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "backend: mock",
                        "data:",
                        "  dataset: weibo21",
                        f"  root: {data_root.as_posix()}",
                        '  train_domains: ["tech"]',
                        '  final_test_domains: ["health"]',
                        '  excluded_domains: ["unknown"]',
                    )
                ),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(["--config", str(config_path), "data", "manifest"])
            manifest = asyncio.run(_run(args))
            self.assertEqual(len(manifest["train_ids"]), 1)
            self.assertEqual(len(manifest["test_ids"]), 1)
            self.assertEqual(manifest["split_policy"]["train_domains"], ["tech"])
            self.assertEqual(manifest["split_policy"]["final_test_domains"], ["health"])

            override = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "data",
                    "manifest",
                    "--final-test-domains",
                    "tech",
                ]
            )
            overridden = asyncio.run(_run(override))
            self.assertEqual(overridden["split_policy"]["train_domains"], ["health"])
            self.assertEqual(overridden["split_policy"]["final_test_domains"], ["tech"])
            self.assertNotEqual(manifest["test_ids"], overridden["test_ids"])

            conflict = parser.parse_args(
                ["--config", str(config_path), "--dataset", "amtcele", "data", "manifest"]
            )
            with self.assertRaisesRegex(ValueError, "conflicts with configured dataset"):
                asyncio.run(_run(conflict))


if __name__ == "__main__":
    unittest.main()
