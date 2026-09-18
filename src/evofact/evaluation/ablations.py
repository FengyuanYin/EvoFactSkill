from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from statistics import mean

from evofact.config import AppConfig
from evofact.experiments.runner import ExperimentRunner
from evofact.reporting.report import summarize_runs
from evofact.skills.package_adapter import package_bank_digest
from evofact.skills.repository import SkillRepository
from evofact.validation.statistics import mcnemar, paired_bootstrap

EVOLUTION_ABLATION_ARMS = (
    "full",
    "no-evolution",
    "instructions-only",
    "no-discovery",
    "rule-proposer",
    "no-evidence",
)

ABLATION_REFERENCES = {
    "no-evolution": "full",
    "instructions-only": "no-discovery",
    "no-discovery": "full",
    "rule-proposer": "full",
    "no-evidence": "full",
}

META_ABLATION_ARMS = (
    "meta-full",
    "meta-no-cross-episode-aggregation",
    "meta-no-negative-transfer-constraint",
    "meta-no-worst-domain-constraint",
    "meta-no-specialization",
)
META_ABLATION_REFERENCES = {arm: "meta-full" for arm in META_ABLATION_ARMS[1:]}

GENERATION_ABLATION_ARMS = (
    "generation-full",
    "generation-real-only",
    "generation-rule-proposer",
)
GENERATION_ABLATION_REFERENCES = {arm: "generation-full" for arm in GENERATION_ABLATION_ARMS[1:]}

ALL_ABLATION_ARMS = (
    *EVOLUTION_ABLATION_ARMS,
    *META_ABLATION_ARMS,
    *GENERATION_ABLATION_ARMS,
)


class _ReplayJSONBackend:
    """Reuse identical generator/verifier calls across paired generation arms."""

    def __init__(self, backend):
        self.backend = backend
        self.model = getattr(backend, "model", type(backend).__name__)
        self.cache = {}

    async def _call_with_usage(self, system: str, payload: dict):
        from evofact.runtime.backend import call_json_with_usage

        key = json.dumps((system, payload), ensure_ascii=False, sort_keys=True, default=str)
        if key not in self.cache:
            self.cache[key] = await call_json_with_usage(self.backend, system, payload)
        response, usage = self.cache[key]
        return copy.deepcopy(response), usage

    async def _call(self, system: str, payload: dict):
        response, _ = await self._call_with_usage(system, payload)
        return response


def apply_evolution_ablation(config: AppConfig, arm: str) -> AppConfig:
    """Apply one independent evolution-mechanism switch to a frozen config."""
    if arm not in EVOLUTION_ABLATION_ARMS:
        raise ValueError(f"unknown evolution ablation arm: {arm}")
    evolution = replace(config.evolution, enabled=True)
    if arm == "no-evolution":
        evolution = replace(evolution, enabled=False)
    elif arm == "instructions-only":
        evolution = replace(evolution, scope="instructions", discovery=False)
    elif arm == "no-discovery":
        evolution = replace(evolution, discovery=False)
    elif arm == "rule-proposer":
        if evolution.proposer == "rule":
            raise ValueError("rule-proposer is identical to full when full already uses rule")
        evolution = replace(evolution, proposer="rule")
    return replace(config, evolution=evolution)


def _without_evidence(samples):
    return [replace(sample, evidence=()) for sample in samples]


def apply_meta_ablation(config: AppConfig, arm: str) -> AppConfig:
    if arm not in META_ABLATION_ARMS:
        raise ValueError(f"unknown meta ablation arm: {arm}")
    meta = replace(config.meta_learning, enabled=True)
    if arm == "meta-no-cross-episode-aggregation":
        meta = replace(meta, aggregate_across_episodes=False, min_valid_episodes=1)
    elif arm == "meta-no-negative-transfer-constraint":
        meta = replace(meta, enforce_negative_transfer=False)
    elif arm == "meta-no-worst-domain-constraint":
        meta = replace(meta, enforce_worst_domain=False)
    elif arm == "meta-no-specialization":
        meta = replace(meta, allow_specialization=False)
    return replace(
        config,
        meta_learning=meta,
        evolution=replace(config.evolution, enabled=True),
    )


def apply_generation_ablation(config: AppConfig, arm: str) -> AppConfig:
    if arm not in GENERATION_ABLATION_ARMS:
        raise ValueError(f"unknown generation ablation arm: {arm}")
    generation = replace(config.generation, enabled=True, use_generated_in_training=True)
    if arm == "generation-real-only":
        generation = replace(generation, use_generated_in_training=False)
    elif arm == "generation-rule-proposer":
        if generation.proposer == "rule":
            raise ValueError(
                "generation-rule-proposer is identical to generation-full when full uses rule"
            )
        generation = replace(generation, proposer="rule")
    return replace(
        config,
        generation=generation,
        meta_learning=replace(config.meta_learning, enabled=True),
        evolution=replace(config.evolution, enabled=True),
    )


async def run_ablations(
    config: AppConfig,
    root: Path,
    train_samples,
    validation_samples,
    test_samples,
    *,
    manifest_id: str,
    arms: tuple[str, ...],
    seeds: int = 3,
    bootstrap_iterations: int = 1000,
    progress=None,
) -> dict:
    """Run isolated paired evolution ablations on one immutable data split.

    Each arm starts from the same seed packages and receives an isolated temporary
    package bank. Final-test sample order and metric implementations are shared.
    No arm can mutate the configured production Skill repository.
    """
    if not train_samples or not validation_samples or not test_samples:
        raise ValueError("ablation requires non-empty train, validation, and test partitions")
    split_ids = {
        "train": {sample.sample_id for sample in train_samples},
        "validation": {sample.sample_id for sample in validation_samples},
        "test": {sample.sample_id for sample in test_samples},
    }
    overlaps = {
        f"{left}/{right}": sorted(split_ids[left] & split_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        if split_ids[left] & split_ids[right]
    }
    if overlaps:
        raise ValueError(f"ablation split leakage detected: {overlaps}")
    if seeds < 1:
        raise ValueError("ablation seeds must be positive")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    normalized_arms = tuple(dict.fromkeys(arms))
    if "full" not in normalized_arms or len(normalized_arms) < 2:
        raise ValueError("ablation requires full and at least one comparison arm")
    unknown = set(normalized_arms) - set(EVOLUTION_ABLATION_ARMS)
    if unknown:
        raise ValueError("unknown evolution ablation arms: " + ", ".join(sorted(unknown)))
    if "instructions-only" in normalized_arms and config.evolution.proposer != "llm":
        raise ValueError(
            "instructions-only requires an LLM full arm; the rule proposer already "
            "produces instruction-only edits"
        )
    if "no-evidence" in normalized_arms and not any(
        sample.evidence for sample in (*train_samples, *validation_samples, *test_samples)
    ):
        raise ValueError("no-evidence requires at least one evidence-bearing sample")
    missing_references = {
        arm: ABLATION_REFERENCES[arm]
        for arm in normalized_arms
        if arm != "full" and ABLATION_REFERENCES[arm] not in normalized_arms
    }
    if missing_references:
        requirements = ", ".join(
            f"{arm} requires {reference}" for arm, reference in sorted(missing_references.items())
        )
        raise ValueError(f"ablation is missing paired reference arms: {requirements}")

    runs: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="evofact-ablation-") as temp:
        temp_root = Path(temp)
        for seed_offset in range(seeds):
            seed = config.seed + seed_offset
            seed_rows = {}
            for arm_index, arm in enumerate(normalized_arms):
                if progress is not None:
                    from evofact.runtime.progress import ProgressEvent

                    progress.update(
                        ProgressEvent(
                            "ablation",
                            arm,
                            seed_offset * len(normalized_arms) + arm_index,
                            seeds * len(normalized_arms),
                            "arms",
                        )
                    )
                arm_config = apply_evolution_ablation(replace(config, seed=seed), arm)
                arm_root = temp_root / f"seed-{seed}" / arm
                arm_config = replace(
                    arm_config,
                    skill_store=arm_root / "skill-store",
                    output_dir=arm_root / "outputs",
                )
                runner = ExperimentRunner(arm_config, root)
                arm_train = (
                    _without_evidence(train_samples) if arm == "no-evidence" else train_samples
                )
                arm_validation = (
                    _without_evidence(validation_samples)
                    if arm == "no-evidence"
                    else validation_samples
                )
                arm_test = _without_evidence(test_samples) if arm == "no-evidence" else test_samples
                evolution = None
                if arm_config.evolution.enabled:
                    evolution = await runner.closed_loop_batched(
                        list(arm_train),
                        list(arm_validation),
                        repository=SkillRepository(arm_config.skill_store),
                        progress=progress,
                    )
                _, evaluation = await runner.run(
                    list(arm_test),
                    progress=progress,
                    task_name=f"ablation {arm} seed {seed}",
                    phase="final test",
                )
                seed_rows[arm] = list(evaluation.per_sample)
                runs.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "manifest_id": manifest_id,
                        "sample_ids": [row.sample_id for row in evaluation.per_sample],
                        "mechanism": {
                            "evolution": asdict(arm_config.evolution),
                            "input_control": {
                                "evidence": "removed" if arm == "no-evidence" else "available"
                            },
                        },
                        "metrics": evaluation.aggregate_metrics,
                        "package_bank_digest": package_bank_digest(runner.packages),
                        "evolution": (
                            None
                            if evolution is None
                            else {
                                "proposals": len(evolution["proposals"]),
                                "gate_decisions": [
                                    asdict(item) for item in evolution["gate_decisions"]
                                ],
                                "batches": evolution["batches"],
                            }
                        ),
                        "budget": runner._budget_manager().snapshot().model_dump(),
                    }
                )
            for run in runs:
                if run["seed"] != seed or run["arm"] == "full":
                    continue
                reference_arm = ABLATION_REFERENCES[run["arm"]]
                baseline = seed_rows[reference_arm]
                candidate = seed_rows[run["arm"]]
                run["paired_comparison"] = {
                    "reference_arm": reference_arm,
                    "mcnemar": asdict(mcnemar(baseline, candidate, config.gate.alpha)),
                    "arm_minus_reference_accuracy_delta_ci95": paired_bootstrap(
                        baseline,
                        candidate,
                        seed=seed,
                        iterations=bootstrap_iterations,
                    ),
                }

    summaries = {
        arm: summarize_runs([run["metrics"] for run in runs if run["arm"] == arm])
        for arm in normalized_arms
    }
    return {
        "schema_version": "evolution_ablation_v1",
        "manifest_id": manifest_id,
        "arms": list(normalized_arms),
        "seeds": [config.seed + offset for offset in range(seeds)],
        "sample_order": [sample.sample_id for sample in test_samples],
        "protected_final_test": True,
        "runs": runs,
        "summary": summaries,
        "completed": True,
    }


def _episode_scores(outcome) -> dict[str, float]:
    scores = {episode.episode_id: 0.0 for episode in outcome.episodes}
    for row in outcome.episode_results:
        gain = float(row.aggregate_deltas.get("macro_f1_all", 0.0))
        scores[row.episode_id] = max(scores.get(row.episode_id, gain), gain)
    return scores


def _transfer_metrics(outcome) -> dict[str, float]:
    utilities = list(outcome.utilities.values())
    scores = _episode_scores(outcome)
    return {
        "episodes": float(len(outcome.episodes)),
        "episode_results": float(len(outcome.episode_results)),
        "candidates": float(len(utilities)),
        "accepted_candidates": float(sum(decision.accepted for decision in outcome.decisions)),
        "mean_transfer_gain": (
            mean(utility.mean_gain for utility in utilities) if utilities else 0.0
        ),
        "mean_negative_transfer_rate": (
            mean(utility.negative_transfer_rate for utility in utilities) if utilities else 0.0
        ),
        "worst_domain_drop": (
            min(utility.worst_domain_drop for utility in utilities) if utilities else 0.0
        ),
        "mean_episode_macro_f1_gain": mean(scores.values()) if scores else 0.0,
    }


def _paired_numeric_bootstrap(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    if set(reference) != set(candidate):
        raise ValueError("paired ablation arms produced different episode IDs")
    keys = sorted(reference)
    if not keys:
        raise ValueError("paired ablation requires at least one episode")
    deltas = [candidate[key] - reference[key] for key in keys]
    rng = Random(seed)
    estimates = [
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(iterations)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * iterations)],
        estimates[min(iterations - 1, int(0.975 * iterations))],
    )


def _validate_family_arms(
    arms: tuple[str, ...],
    full_arm: str,
    references: dict[str, str],
) -> None:
    if not arms:
        return
    if full_arm not in arms:
        raise ValueError(f"{full_arm} is required for its ablation family")
    missing = {
        arm: references[arm] for arm in arms if arm != full_arm and references[arm] not in arms
    }
    if missing:
        raise ValueError(f"ablation is missing paired reference arms: {missing}")


async def _run_transfer_ablations(
    config: AppConfig,
    root: Path,
    samples,
    *,
    facts,
    final_test_domains: tuple[str, ...],
    train_domains: tuple[str, ...] | None,
    manifest_id: str,
    arms: tuple[str, ...],
    family: str,
    seeds: int,
    bootstrap_iterations: int,
    progress=None,
    generation_backend=None,
) -> dict:
    from evofact.experiments.adversarial_runner import AdversarialEvolutionRunner
    from evofact.experiments.meta_runner import MetaEvolutionRunner

    is_generation = family == "generation"
    full_arm = "generation-full" if is_generation else "meta-full"
    references = GENERATION_ABLATION_REFERENCES if is_generation else META_ABLATION_REFERENCES
    _validate_family_arms(arms, full_arm, references)
    runs = []
    with tempfile.TemporaryDirectory(prefix=f"evofact-{family}-ablation-") as temp:
        temp_root = Path(temp)
        for seed_offset in range(seeds):
            seed = config.seed + seed_offset
            score_rows = {}
            shared_generation_backend = None
            if is_generation:
                backend = (
                    generation_backend
                    or ExperimentRunner(replace(config, seed=seed), root)._backend()
                )
                shared_generation_backend = _ReplayJSONBackend(backend)
            for arm_index, arm in enumerate(arms):
                if progress is not None:
                    from evofact.runtime.progress import ProgressEvent

                    progress.update(
                        ProgressEvent(
                            "ablation",
                            arm,
                            seed_offset * len(arms) + arm_index,
                            seeds * len(arms),
                            "arms",
                        )
                    )
                arm_config = (
                    apply_generation_ablation(replace(config, seed=seed), arm)
                    if is_generation
                    else apply_meta_ablation(replace(config, seed=seed), arm)
                )
                arm_root = temp_root / f"seed-{seed}" / arm
                arm_config = replace(
                    arm_config,
                    skill_store=arm_root / "skill-store",
                    output_dir=arm_root / "outputs",
                    meta_learning=replace(
                        arm_config.meta_learning,
                        checkpoint_path=arm_root / "checkpoint.json",
                    ),
                    generation=replace(
                        arm_config.generation,
                        store_path=arm_root / "generation-audit.json",
                    ),
                )
                repository = SkillRepository(arm_config.skill_store)
                if is_generation:
                    runner = AdversarialEvolutionRunner(
                        arm_config,
                        root,
                        facts,
                        repository,
                        generation_backend=shared_generation_backend,
                    )
                else:
                    runner = MetaEvolutionRunner(arm_config, root, repository)
                run_kwargs = {
                    "final_test_domains": final_test_domains,
                    "evaluation_only": True,
                    "progress": progress,
                }
                if not is_generation:
                    run_kwargs["train_domains"] = train_domains
                outcome = await runner.run(list(samples), **run_kwargs)
                scores = _episode_scores(outcome)
                score_rows[arm] = scores
                metrics = _transfer_metrics(outcome)
                audit_metrics = {}
                if is_generation:
                    audit_metrics = {
                        "generated": float(
                            sum(row["metrics"]["generated"] for row in runner.audit.values())
                        ),
                        "generated_accepted": float(
                            sum(row["metrics"]["accepted"] for row in runner.audit.values())
                        ),
                        "generated_used_in_training": float(
                            sum(
                                row["metrics"]["generated_used_in_training"]
                                for row in runner.audit.values()
                            )
                        ),
                    }
                    metrics.update(audit_metrics)
                runs.append(
                    {
                        "family": family,
                        "arm": arm,
                        "seed": seed,
                        "manifest_id": manifest_id,
                        "mechanism": (
                            {
                                "meta_learning": asdict(arm_config.meta_learning),
                                "generation": asdict(arm_config.generation),
                                "evolution": asdict(arm_config.evolution),
                            }
                            if is_generation
                            else {
                                "meta_learning": asdict(arm_config.meta_learning),
                                "evolution": asdict(arm_config.evolution),
                            }
                        ),
                        "metrics": metrics,
                        "episode_scores": scores,
                        "decisions": [asdict(decision) for decision in outcome.decisions],
                        "package_bank_digest": package_bank_digest(runner.base.packages),
                        "budget": runner.base._budget_manager().snapshot().model_dump(),
                    }
                )
            for run in runs:
                if run["seed"] != seed or run["arm"] == full_arm:
                    continue
                reference_arm = references[run["arm"]]
                run["paired_comparison"] = {
                    "reference_arm": reference_arm,
                    "paired_unit": "episode",
                    "arm_minus_reference_macro_f1_gain_ci95": _paired_numeric_bootstrap(
                        score_rows[reference_arm],
                        score_rows[run["arm"]],
                        seed=seed,
                        iterations=bootstrap_iterations,
                    ),
                }
    return {
        "family": family,
        "arms": list(arms),
        "runs": runs,
        "summary": {
            arm: summarize_runs([row["metrics"] for row in runs if row["arm"] == arm])
            for arm in arms
        },
    }


def _summary_table(summary: dict[str, dict]) -> list[dict]:
    table = []
    for arm, metrics in summary.items():
        row = {"arm": arm}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_std"] = values["std"]
        table.append(row)
    return table


async def run_unified_ablations(
    config: AppConfig,
    root: Path,
    *,
    train_samples,
    validation_samples,
    test_samples,
    all_samples,
    facts=(),
    final_test_domains: tuple[str, ...] = (),
    train_domains: tuple[str, ...] | None = None,
    manifest_id: str,
    arms: tuple[str, ...],
    seeds: int = 3,
    bootstrap_iterations: int = 1000,
    progress=None,
    generation_backend=None,
) -> dict:
    """Run evolution, DEMSE, and generation controls under one report contract."""
    normalized_arms = tuple(dict.fromkeys(arms))
    if seeds < 1:
        raise ValueError("ablation seeds must be positive")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    unknown = set(normalized_arms) - set(ALL_ABLATION_ARMS)
    if unknown:
        raise ValueError("unknown ablation arms: " + ", ".join(sorted(unknown)))
    if not normalized_arms:
        raise ValueError("at least one ablation arm is required")
    family_results = {}
    evolution_arms = tuple(arm for arm in normalized_arms if arm in EVOLUTION_ABLATION_ARMS)
    meta_arms = tuple(arm for arm in normalized_arms if arm in META_ABLATION_ARMS)
    generation_arms = tuple(arm for arm in normalized_arms if arm in GENERATION_ABLATION_ARMS)
    if evolution_arms:
        family_results["evolution"] = await run_ablations(
            config,
            root,
            train_samples,
            validation_samples,
            test_samples,
            manifest_id=manifest_id,
            arms=evolution_arms,
            seeds=seeds,
            bootstrap_iterations=bootstrap_iterations,
            progress=progress,
        )
    if meta_arms:
        family_results["meta"] = await _run_transfer_ablations(
            config,
            root,
            all_samples,
            facts=(),
            final_test_domains=final_test_domains,
            train_domains=train_domains,
            manifest_id=manifest_id,
            arms=meta_arms,
            family="meta",
            seeds=seeds,
            bootstrap_iterations=bootstrap_iterations,
            progress=progress,
        )
    if generation_arms:
        family_results["generation"] = await _run_transfer_ablations(
            config,
            root,
            all_samples,
            facts=facts,
            final_test_domains=final_test_domains,
            train_domains=train_domains,
            manifest_id=manifest_id,
            arms=generation_arms,
            family="generation",
            seeds=seeds,
            bootstrap_iterations=bootstrap_iterations,
            progress=progress,
            generation_backend=generation_backend,
        )
    runs = [
        {**run, "family": family}
        for family, result in family_results.items()
        for run in result["runs"]
    ]
    summary = {
        arm: values
        for result in family_results.values()
        for arm, values in result["summary"].items()
    }
    return {
        "schema_version": "unified_ablation_v2",
        "manifest_id": manifest_id,
        "arms": list(normalized_arms),
        "seeds": [config.seed + offset for offset in range(seeds)],
        "families": family_results,
        "runs": runs,
        "summary": summary,
        "table": _summary_table(summary),
        "protected_invariants": [
            "typed_specialist_reports",
            "package_owned_contracts",
            "candidate_firewall",
            "blind_generation_verifier",
            "split_leakage_checks",
        ],
        "completed": True,
    }
