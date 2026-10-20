"""Feedback evolution: ordinary sample splits with durable within-batch retries."""

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from evofact.core.models import DataManifest
from evofact.data.domains import data_fingerprint, effective_domain
from evofact.data.leakage import detect_leakage
from evofact.reporting.validation_curve import write_validation_curve

from .meta_batch_runner import MetaBatchRunner
from .runner import proposal_history


class FeedbackEvolveRunner(MetaBatchRunner):
    mode = "feedback-evolve"
    checkpoint_schema = "feedback_evolve_checkpoint_v1"
    run_prefix = "feedback-evolve-"

    def __init__(self, config, project_root, repository=None, *, checkpoint_path, provenance=None):
        config = replace(
            config,
            meta_learning=replace(config.meta_learning, checkpoint_path=Path(checkpoint_path)),
        )
        super().__init__(config, project_root, repository)
        self.provenance = provenance or {}
        self.validation_samples = []
        self.independent_validation_samples = []

    def _check_enabled(self):
        if not self.config.evolution.enabled:
            raise ValueError("evolution.enabled must be true")

    def _identity_extra(self):
        return {
            "provenance": self.provenance,
            "model_identity": self.base.model_identity(),
            "validation_data": data_fingerprint(self.validation_samples),
            **(
                {
                    "independent_validation_data": data_fingerprint(
                        self.independent_validation_samples
                    )
                }
                if self.config.independent_validation.enabled
                else {}
            ),
        }

    def _prepare_plan(self, samples, **kwargs):
        train, valid = list(samples), list(self.validation_samples)
        if not train or not valid:
            raise ValueError("evolution requires non-empty training and validation samples")
        if len({s.sample_id for s in train + valid}) != len(train) + len(valid):
            raise ValueError("duplicate sample IDs in training/validation pool")
        manifest = DataManifest(
            "evolve-validation",
            {},
            tuple(s.sample_id for s in train),
            tuple(s.sample_id for s in valid),
            label_contract_digest=self.base.label_contract_registry.digest,
        )
        errors = detect_leakage(train + valid, manifest)
        if any(s.metadata.get("split") == "test" for s in train + valid):
            errors.append("test-only sample in training/validation pool")
        if errors:
            raise ValueError("evolution data leakage: " + "; ".join(errors))
        random.Random(self.config.seed + 1).shuffle(valid)
        size = self.config.execution.batch_size
        batches = [train[i : i + size] for i in range(0, len(train), size)]
        source = dict.fromkeys(effective_domain(s) for s in train + valid)
        return source, (), train, valid, batches, (), None

    def _proposal_history(self, state):
        history = []
        for attempt in state["attempts"]:
            if "decision" in attempt:
                summaries = proposal_history(
                    self.base.packages,
                    attempt.get("proposals", [attempt["proposal"]]),
                    [attempt["decision"]] * len(attempt.get("proposals", [attempt["proposal"]])),
                )
                if "proposals" in attempt:
                    summaries = tuple(
                        dict(
                            item,
                            joint_outcome=True,
                            joint_rationale=attempt.get("joint_rationale", ""),
                            joint_proposal_ids=[p["proposal_id"] for p in attempt["proposals"]],
                        )
                        for item in summaries
                    )
                history.extend(summaries)
            else:
                history.append(
                    {
                        "batch": attempt["batch"],
                        "attempt": attempt["attempt"],
                        "status": attempt["status"],
                        "reason": attempt.get("reason", ""),
                    }
                )
        return tuple(history[-12:])

    def _after_save(self, checkpoint):
        write_validation_curve(checkpoint, self.root / self.config.output_dir)

    async def run(
        self, samples, validation_samples, *, independent_validation_samples=None, **kwargs
    ):
        self.validation_samples = list(validation_samples)
        self.independent_validation_samples = list(
            independent_validation_samples or validation_samples
        )
        outcome = await super().run(samples, **kwargs)
        judged = [a for a in outcome["attempts"] if "decision" in a]
        outcome.update(
            proposals=[a["proposal"] for a in judged],
            proposal_groups=[a.get("proposals", [a["proposal"]]) for a in judged],
            gate_decisions=[a["decision"] for a in judged],
            evaluation_only=kwargs.get("evaluation_only", False),
            provenance=self.provenance,
            validation_pool_size=len(self.validation_samples),
            validation_samples_per_attempt=(
                self.config.data.evolution_validation_samples
                * outcome["validation_aggregation"]["rounds"]
            ),
            validation_policy="disjoint_sample_pool_without_replacement",
        )
        return outcome


async def run_cli(args, config, root, manifest, select, progress, data_sampling):
    """Adapt the ordinary CLI's sample split and paths to the shared retry runner."""
    from evofact.skills.repository import SkillRepository

    from .runner import fixture_samples, fixture_validation_samples

    train = select(manifest.train_ids, role="train") if manifest else fixture_samples()
    validation = (
        select(manifest.evolution_validation_ids) if manifest else fixture_validation_samples()
    )
    if args.resume and args.evaluation_only:
        raise ValueError("feedback-evolve --resume cannot be combined with --evaluation-only")
    if args.max_attempts_per_batch < 1:
        raise ValueError("max-attempts-per-batch must be positive")
    checkpoint = Path(
        args.checkpoint or Path(config.output_dir) / "feedback-evolve-checkpoint.json"
    )
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if args.resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("schema_version") != FeedbackEvolveRunner.checkpoint_schema:
            raise ValueError(
                "incompatible feedback-evolve checkpoint; ordinary evolve checkpoints must use evolve"
            )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    provenance = {
        "config_file": str(config_path.resolve()),
        "config_file_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_id": manifest.manifest_id if manifest else "fixture",
        "limit": args.limit,
    }
    trace_log = Path(
        args.trace_log
        or config.execution.trace_log
        or Path(config.output_dir) / "training-traces.jsonl"
    )
    if not trace_log.is_absolute():
        trace_log = root / trace_log
    if manifest is None:
        config = replace(
            config,
            data=replace(
                config.data,
                evolution_validation_samples=min(
                    config.data.evolution_validation_samples, len(validation)
                ),
            ),
        )

    async def execute(path, effective_config):
        runner = FeedbackEvolveRunner(
            effective_config,
            root,
            None if args.evaluation_only else SkillRepository(root / config.skill_store),
            checkpoint_path=path,
            provenance=provenance,
        )
        outcome = await runner.run(
            train,
            validation,
            independent_validation_samples=(
                select(manifest.evolution_validation_ids + manifest.protected_validation_ids)
                if effective_config.independent_validation.enabled and manifest
                else None
            ),
            max_attempts_per_batch=args.max_attempts_per_batch,
            validation_rounds=args.validation_rounds,
            resume=args.resume,
            evaluation_only=args.evaluation_only,
            progress=progress,
            trace_log_path=trace_log,
        )
        outcome["skill_bank"] = (
            runner.repository.package_bank_info("active") if runner.repository else None
        )
        outcome["trace_log"] = str(trace_log)
        outcome["data_sampling"] = data_sampling
        if not args.evaluation_only:
            outcome["validation_curve"] = {
                "csv": str(root / config.output_dir / "validation-curve.csv"),
                "active_csv": str(root / config.output_dir / "validation-active.csv"),
                "svg": str(root / config.output_dir / "validation-curve.svg"),
            }
        return outcome

    if args.evaluation_only:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="evofact-feedback-evolve-") as directory:
            outcome = await execute(
                Path(directory) / "checkpoint.json", replace(config, output_dir=Path(directory))
            )
        outcome["checkpoint"] = None
        return outcome
    return await execute(checkpoint, config)
