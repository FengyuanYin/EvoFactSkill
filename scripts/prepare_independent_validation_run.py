"""Prepare isolated guarded experiments and concrete balanced validation manifests."""

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.config import (  # noqa: E402
    GeneratorMemoryConfig,
    IndependentValidationConfig,
    OptimizerMemoryConfig,
    load_config,
)
from evofact.data.manifests import build_manifest  # noqa: E402
from evofact.data.registry import DataRegistry  # noqa: E402
from evofact.data.sampling import balanced_domain_sample, select_final_test_samples  # noqa: E402
from evofact.evolution.paired_feedback import PAIRED_FEEDBACK_POLICY  # noqa: E402
from evofact.evolution.regression_policy import regression_policy_identity  # noqa: E402
from evofact.evolution.source_feedback import SOURCE_FEEDBACK_POLICY  # noqa: E402
from evofact.experiments.meta_sampling import matched_meta_plan  # noqa: E402
from evofact.validation.independent import (  # noqa: E402
    discovery_without_pool,
    independent_pool,
    independent_rows,
)


def without_sections(text, sections):
    """Replace configured top-level blocks instead of emitting duplicate YAML keys."""
    lines = []
    skip = False
    for line in text.splitlines(keepends=True):
        match = re.match(r"^([A-Za-z_][A-Za-z_0-9]*):", line)
        if match:
            skip = match[1] in sections
        if not skip:
            lines.append(line)
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-domain", type=int, default=32)
    parser.add_argument("--evaluation-samples-per-domain", type=int, default=4)
    parser.add_argument("--optimizer-memory", action="store_true")
    parser.add_argument("--optimizer-context-window", type=int)
    parser.add_argument("--generator-memory", action="store_true")
    parser.add_argument("--generator-context-window", type=int)
    parser.add_argument(
        "--independent-protection",
        action="store_true",
        help="Keep legacy independent protection for new adversarial runs instead of source feedback.",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=("feedback-evolve", "meta-evolve", "adversarial-evolve"),
        default=["feedback-evolve", "meta-evolve", "adversarial-evolve"],
    )
    args = parser.parse_args()
    OptimizerMemoryConfig(
        enabled=args.optimizer_memory, context_window=args.optimizer_context_window
    )
    GeneratorMemoryConfig(
        enabled=args.generator_memory, context_window=args.generator_context_window
    )
    IndependentValidationConfig(
        enabled=True,
        samples_per_domain=args.samples_per_domain,
        evaluation_samples_per_domain=args.evaluation_samples_per_domain,
    )
    tag = (
        datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_root = Path("docs/independent-validation") / tag
    registry = DataRegistry()
    prepared = []
    for route in args.routes:
        source_feedback = route == "adversarial-evolve" and not args.independent_protection
        name = {
            "feedback-evolve": "feedback_evolve",
            "meta-evolve": "meta_evolve",
            "adversarial-evolve": "adversarial_knowledge",
        }[route]
        for fold in (1, 2):
            source = ROOT / f"configs/weibo21_{name}_fold_{fold}.yaml"
            output = run_root / route / f"fold_{fold}"
            artifact = output / route
            store = (
                Path("skills/store_independent_validation") / tag / route / f"fold_{fold}" / route
            )
            text = source.read_text(encoding="utf-8")
            original = load_config(source)
            replacements = {
                "output_dir": artifact,
                "skill_store": store,
                "trace_log": artifact / "training-traces.jsonl",
                "checkpoint_path": artifact / "train-checkpoint.json",
                "store_path": artifact / "generation-audit.json",
            }
            for key, value in replacements.items():
                text, count = re.subn(
                    rf"(?m)^([ \t]*{key}:).*?$", lambda m: m[1] + " " + value.as_posix(), text
                )
                if count == 0 and key == "store_path" and route != "adversarial-evolve":
                    continue
                if count != 1:
                    raise ValueError(f"Expected one {key}: {source}")
            sections = {"independent_validation"}
            if args.optimizer_memory:
                sections.add("optimizer_memory")
            if args.generator_memory and route == "adversarial-evolve":
                sections.add("generator_memory")
            text = without_sections(text, sections)
            text += f"\nindependent_validation:\n  enabled: true\n  manifest_path: {(output / 'validation-manifest.json').as_posix()}\n  samples_per_domain: {args.samples_per_domain}\n  max_positive_recall_drop: 0.0\n  max_domain_f1_drop: 0.05\n  sample_retries: 2\n"
            text += f"  evaluation_samples_per_domain: {args.evaluation_samples_per_domain}\n"
            if source_feedback:
                text += "  source_feedback: true\n"
            if args.optimizer_memory:
                window = (
                    args.optimizer_context_window
                    or original.optimizer_memory.context_window
                    or "null"
                )
                text += (
                    "\noptimizer_memory:\n  enabled: true\n"
                    f"  context_window: {window}\n  reserved_output_tokens: 8192\n"
                    "  safety_fraction: 0.1\n  max_overflow_retries: 2\n"
                )
            if args.generator_memory and route == "adversarial-evolve":
                window = (
                    args.generator_context_window
                    or original.generator_memory.context_window
                    or "null"
                )
                text += (
                    "\ngenerator_memory:\n  enabled: true\n"
                    f"  context_window: {window}\n  reserved_output_tokens: 8192\n"
                    "  safety_fraction: 0.1\n  max_overflow_retries: 2\n"
                )
            samples = registry.load(
                original.data.dataset,
                ROOT / original.data.root,
                excluded_domains=original.data.excluded_domains,
            )
            final, _ = select_final_test_samples(
                samples,
                original.data.final_test_domains,
                dataset=original.data.dataset,
                samples_per_domain=original.data.final_test_samples_per_domain,
                seed=original.seed,
                repository_root=ROOT,
                static_pattern=original.data.static_test_pattern,
                require_static=original.data.require_static_test,
            )
            source_rows = [s for s in samples if s.domain in original.data.train_domains]
            selected = source_rows + final
            if route == "feedback-evolve":
                manifest = build_manifest(
                    selected,
                    seed=original.seed,
                    train_domains=original.data.train_domains,
                    final_test_domains=original.data.final_test_domains,
                    label_contract_digest=registry.label_contracts.digest,
                )
                train, _ = balanced_domain_sample(
                    [s for s in selected if s.sample_id in set(manifest.train_ids)],
                    original.data.train_domains,
                    200,
                    seed=original.seed,
                )
                valid = [
                    s for s in selected if s.sample_id in set(manifest.evolution_validation_ids)
                ]
                pool_candidates = [
                    s
                    for s in selected
                    if s.sample_id
                    in set(manifest.evolution_validation_ids + manifest.protected_validation_ids)
                ]
                batches = [train[i : i + 16] for i in range(0, len(train), 16)]
            else:
                train, batches, valid, _ = matched_meta_plan(
                    selected,
                    source_domains=original.data.train_domains,
                    final_domains=original.data.final_test_domains,
                    seed=original.seed,
                    batch_size=16,
                    train_limit=200,
                    label_contract_digest=registry.label_contracts.digest,
                )
                pool_candidates = valid
            pool = []
            if not source_feedback:
                pool, _ = independent_pool(
                    pool_candidates,
                    train,
                    final,
                    registry=registry.label_contracts,
                    per_domain=args.samples_per_domain,
                    seed=original.seed,
                )
            discovery = valid if source_feedback else discovery_without_pool(valid, pool)
            manifest = {
                "route": route,
                "fold": fold,
                "source_config": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "training_sample_ids": [s.sample_id for s in train],
                "independent_samples": [
                    {
                        "sample_id": s.sample_id,
                        "domain": s.domain,
                        "label": registry.label_contracts.resolve_sample(s).normalize(s.label),
                    }
                    for s in pool
                ],
                "discovery_pool_size": len(discovery),
                "batch_independent_sample_ids": []
                if source_feedback
                else [
                    [
                        s.sample_id
                        for s in independent_rows(pool, b, cross_domain=route != "feedback-evolve")
                    ]
                    for b in batches
                ],
            }
            if source_feedback:
                manifest.update(
                    validation_feedback_policy=SOURCE_FEEDBACK_POLICY,
                    source_validation_sample_ids=[s.sample_id for s in valid],
                )
            prepared.append((output, text, manifest))
    for output, text, manifest in prepared:
        (ROOT / output).mkdir(parents=True, exist_ok=False)
        (ROOT / output / "config.yaml").write_text(text, encoding="utf-8")
        (ROOT / output / "validation-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (ROOT / run_root / "run.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "routes": args.routes,
                "optimizer_regression_policy": regression_policy_identity(),
                "source_feedback_diagnostics_policy": PAIRED_FEEDBACK_POLICY,
                "samples_per_domain": args.samples_per_domain,
                "evaluation_samples_per_domain": args.evaluation_samples_per_domain,
                "optimizer_memory": args.optimizer_memory,
                "optimizer_context_window": args.optimizer_context_window,
                "generator_memory": args.generator_memory,
                "generator_context_window": args.generator_context_window,
                "model_calls": 0,
                "configs": [(p / "config.yaml").as_posix() for p, _, _ in prepared],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"run_root": run_root.as_posix(), "prepared_configs": len(prepared), "model_calls": 0}
        )
    )


if __name__ == "__main__":
    main()
