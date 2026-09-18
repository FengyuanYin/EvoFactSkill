from __future__ import annotations

import tempfile
from dataclasses import asdict, replace
from pathlib import Path

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
)

ABLATION_REFERENCES = {
    "no-evolution": "full",
    "instructions-only": "no-discovery",
    "no-discovery": "full",
    "rule-proposer": "full",
}


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
    missing_references = {
        arm: ABLATION_REFERENCES[arm]
        for arm in normalized_arms
        if arm != "full" and ABLATION_REFERENCES[arm] not in normalized_arms
    }
    if missing_references:
        requirements = ", ".join(
            f"{arm} requires {reference}"
            for arm, reference in sorted(missing_references.items())
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
                evolution = None
                if arm_config.evolution.enabled:
                    evolution = await runner.closed_loop_batched(
                        list(train_samples),
                        list(validation_samples),
                        repository=SkillRepository(arm_config.skill_store),
                        progress=progress,
                    )
                _, evaluation = await runner.run(
                    list(test_samples),
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
                        "mechanism": asdict(arm_config.evolution),
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
