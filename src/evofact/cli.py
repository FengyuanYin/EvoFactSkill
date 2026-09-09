from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from evofact.config import load_config
from evofact.core.models import SkillStatus
from evofact.data.leakage import detect_leakage
from evofact.data.manifests import build_manifest, manifest_json
from evofact.data.registry import DataRegistry
from evofact.evaluation.ablations import run_ablations
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.reporting.meta_report import write_meta_report
from evofact.reporting.report import skill_evolution_curve, summarize_runs, write_report
from evofact.skills.candidates import apply_candidate
from evofact.skills.repository import SkillRepository


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=lambda x: asdict(x) if hasattr(x, "__dataclass_fields__") else str(x),
    )


def build_parser():
    """创建顶层命令行解析器及其所有子命令。"""
    parser = argparse.ArgumentParser(prog="evofact")

    # 全局参数需要放在所选子命令之前。
    parser.add_argument("--config", default="configs/dry_run.yaml")
    parser.add_argument(
        "--dataset",
        choices=("weibo21", "amtcele", "livefact", "advfake"),
    )
    parser.add_argument("--data-root")
    parser.add_argument("--limit", type=int, default=0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 数据集检查，以及可复现的数据划分清单生成命令。
    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        required=True,
    )
    data_subparsers.add_parser("inspect")
    manifest_parser = data_subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output")

    # 无须额外专属参数的命令。
    for name in ("dry-run", "evolve", "validate", "test", "ablation", "report"):
        subparsers.add_parser(name)

    # 跨领域元演化参数，以及可选的消融实验配置覆盖项。
    meta_parser = subparsers.add_parser("meta-evolve")
    meta_parser.add_argument(
        "--final-test-domains",
        default="outer_holdout",
    )
    meta_parser.add_argument("--episodes", type=int)
    meta_parser.add_argument(
        "--strategy",
        choices=("repeated_holdout", "leave_one_domain_out"),
    )
    meta_parser.add_argument("--meta-test-domain-count", type=int)
    meta_parser.add_argument(
        "--ablation",
        choices=(
            "no-cross-episode-aggregation",
            "no-worst-domain-constraint",
            "no-specialization",
        ),
    )
    meta_parser.add_argument("--resume", action="store_true")
    meta_parser.add_argument("--evaluation-only", action="store_true")

    adversarial = subparsers.add_parser("adversarial-evolve")
    adversarial.add_argument(
        "--samples", help="normalized Sample JSONL including evidence snapshots"
    )
    adversarial.add_argument("--facts", help="verified structured evidence facts JSONL")
    adversarial.add_argument("--final-test-domains", default="outer_holdout")
    adversarial.add_argument("--resume", action="store_true")
    adversarial.add_argument("--evaluation-only", action="store_true")

    # 技能仓库查询及技能生命周期管理命令。
    skills_parser = subparsers.add_parser("skills")
    skills_subparsers = skills_parser.add_subparsers(
        dest="skills_command",
        required=True,
    )
    for name in ("list", "show", "diff", "freeze", "retire", "rollback"):
        skill_parser = skills_subparsers.add_parser(name)
        if name != "list":
            skill_parser.add_argument("name")
        if name in {"show", "diff", "rollback"}:
            skill_parser.add_argument("--snapshot", required=name == "rollback")

    return parser


async def _run(args):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config if not Path(args.config).is_absolute() else args.config)
    runner = ExperimentRunner(config, root)
    if (
        args.command in {"dry-run", "test", "evolve", "validate", "ablation", "report"}
        and (root / config.skill_store / "active.json").exists()
    ):  # 在runner内注册技能注册器
        runner.skills = list(SkillRepository(root / config.skill_store).active().values())
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.data_root and not args.dataset:
        raise ValueError("--data-root requires --dataset")
    all_samples = None
    manifest = None
    if args.dataset:  # 加载数据集
        data_root = (
            Path(args.data_root) if args.data_root else config.dataset_roots.get(args.dataset)
        )
        if not data_root:
            raise ValueError(f"no data root configured for {args.dataset}")
        all_samples = DataRegistry().load(args.dataset, data_root)
        manifest = build_manifest(all_samples, seed=config.seed)
        errors = detect_leakage(all_samples, manifest)
        if errors:
            raise ValueError("dataset leakage: " + "; ".join(errors))
        if not all_samples:
            raise ValueError("dataset has no samples")

    def select(ids):
        chosen = [s for s in (all_samples or []) if s.sample_id in set(ids)]
        return chosen[: args.limit] if args.limit > 0 else chosen

    if args.command == "data":
        if args.data_command == "inspect":
            return [asdict(x) for x in DataRegistry().inspect(config.dataset_roots)]
        manifest = manifest or build_manifest(fixture_samples(), seed=config.seed)
        text = manifest_json(manifest)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
        return json.loads(text)
    if args.command in {"dry-run", "test"}:
        samples = None
        if manifest:
            samples = select(manifest.test_ids if args.command == "test" else manifest.train_ids)
        if manifest and not samples:
            raise ValueError(f"no samples available for {args.command}")
        traces, result = await runner.run(samples)
        return {
            "mode": args.command,
            "manifest_id": manifest.manifest_id if manifest else "fixture",
            "n_traces": len(traces),
            "metrics": result.aggregate_metrics,
            "traces": [asdict(t) for t in traces],
        }
    if args.command == "evolve":  # 使用skill 进化
        train = select(manifest.train_ids) if manifest else None
        validation = select(manifest.evolution_validation_ids) if manifest else None
        if manifest and (not train or not validation):
            raise ValueError("evolution requires non-empty train and evolution-validation splits")
        result = await runner.closed_loop(train, validation)
        repo = SkillRepository(root / config.skill_store)
        if not repo.active():
            for seed in runner.skills:
                repo.promote(seed)
        for proposal, decision in zip(result["proposals"], result["gate_decisions"]):
            if decision.disposition == "active":
                bank = list(repo.active().values())
                from evofact.data.domains import skillbank_fingerprint

                repo.commit_bank(
                    apply_candidate(bank, proposal),
                    run_id="fixed-" + proposal.proposal_id + "-" + skillbank_fingerprint(bank),
                    baseline=bank,
                    audit={"proposal": asdict(proposal), "decision": asdict(decision)},
                )
                continue
            for skill in proposal.candidate_skills:
                repo.save(
                    replace(
                        skill,
                        status=SkillStatus.PARETO
                        if decision.disposition == "pareto"
                        else SkillStatus.CANDIDATE,
                    ),
                    decision.disposition,
                )
        return result
    if args.command == "meta-evolve":  # 使用元进化
        rows = fixture_meta_samples() if all_samples is None else all_samples
        final_domains = tuple(
            sorted(x.strip() for x in args.final_test_domains.split(",") if x.strip())
        )
        overrides = {}
        if args.episodes is not None:
            overrides["episodes"] = args.episodes
        if args.strategy is not None:
            overrides["strategy"] = args.strategy
        if args.meta_test_domain_count is not None:
            overrides["meta_test_domain_count"] = args.meta_test_domain_count
        if args.ablation == "no-cross-episode-aggregation":
            overrides.update(aggregate_across_episodes=False, min_valid_episodes=1)
        if args.ablation == "no-worst-domain-constraint":
            overrides["enforce_worst_domain"] = False
        if args.ablation == "no-specialization":
            overrides["allow_specialization"] = False
        if overrides:
            config = replace(config, meta_learning=replace(config.meta_learning, **overrides))
        repository = None if args.evaluation_only else SkillRepository(root / config.skill_store)
        outcome = await MetaEvolutionRunner(config, root, repository).run(
            rows,
            final_test_domains=final_domains,
            resume=args.resume,
            evaluation_only=args.evaluation_only,
        )
        paths = write_meta_report(root / config.output_dir, outcome)
        return {
            "mode": "meta-evolve",
            "run_id": outcome.run_id,
            "episodes": len(outcome.episodes),
            "episode_results": len(outcome.episode_results),
            "decisions": [asdict(x) for x in outcome.decisions],
            "committed_snapshots": outcome.committed_snapshots,
            "mock_results": outcome.mock_results,
            "reports": paths,
        }
    if args.command == "adversarial-evolve":  # 使用元-对抗进化
        from evofact.experiments.adversarial_runner import AdversarialEvolutionRunner
        from evofact.generation.data import fixture_adversarial_data, load_facts, load_samples
        from evofact.reporting.adversarial_report import write_adversarial_report

        if args.limit:
            raise ValueError(
                "adversarial-evolve preserves complete evidence groups; --limit is unsupported"
            )
        if args.samples:
            if all_samples is not None:
                raise ValueError("--samples cannot be combined with --dataset")
            rows, facts = load_samples(args.samples), load_facts(args.facts) if args.facts else []
        elif all_samples is not None:
            rows, facts = all_samples, load_facts(args.facts) if args.facts else []
        else:
            if args.facts:
                raise ValueError("--facts requires --samples or --dataset")
            rows, facts = fixture_adversarial_data()
        final = tuple(sorted(x.strip() for x in args.final_test_domains.split(",") if x.strip()))
        repo = None if args.evaluation_only else SkillRepository(root / config.skill_store)
        adversarial_runner = AdversarialEvolutionRunner(config, root, facts, repo)
        outcome = await adversarial_runner.run(
            rows, final_test_domains=final, resume=args.resume, evaluation_only=args.evaluation_only
        )
        paths = write_adversarial_report(
            root / config.output_dir, outcome, adversarial_runner.audit, config.generation
        )
        return {
            "mode": args.command,
            "run_id": outcome.run_id,
            "mock_results": outcome.mock_results,
            "episodes": len(outcome.episodes),
            "proposer": config.generation.proposer,
            "generator_calls": sum(e["generator_calls"] for e in adversarial_runner.audit.values()),
            "accepted_samples": sum(
                e["metrics"]["accepted"] for e in adversarial_runner.audit.values()
            ),
            "committed_snapshots": outcome.committed_snapshots,
            "reports": paths,
        }
    if args.command == "validate":
        train = select(manifest.train_ids) if manifest else None
        validation = select(manifest.evolution_validation_ids) if manifest else None
        if manifest and (not train or not validation):
            raise ValueError("validation requires non-empty train and evolution-validation splits")
        result = await runner.closed_loop(train, validation)
        return {"gate_decisions": result["gate_decisions"]}
    if args.command == "ablation":
        return await run_ablations(
            runner,
            select(manifest.test_ids or manifest.protected_validation_ids) if manifest else None,
        )
    if args.command == "report":
        report_rows = select(manifest.test_ids) if manifest else None
        if manifest and not report_rows:
            raise ValueError("report requires a non-empty final-test split")
        runs = []
        for offset in range(3):
            seeded = ExperimentRunner(replace(config, seed=config.seed + offset), root)
            seeded.skills = runner.skills
            _, result = await seeded.run(report_rows)
            runs.append(result.aggregate_metrics)
        summary = summarize_runs(runs)
        _, grouped = await runner.run(report_rows)
        repository = SkillRepository(root / config.skill_store)
        payload = {
            "metrics": {k: v["mean"] for k, v in summary.items()},
            "manifest_id": manifest.manifest_id if manifest else "fixture",
            "mock_results": config.backend == "mock",
            "multi_seed": summary,
            "domain_metrics": grouped.domain_metrics,
            "temporal_metrics": grouped.temporal_metrics,
            "paired_gate": [],
            "skill_evolution_curve": skill_evolution_curve(repository.history()),
            "completed": True,
        }
        return write_report(root / config.output_dir, "EvoFactSkill multi-seed report", payload)
    repo = SkillRepository(root / config.skill_store)
    command = args.skills_command
    if command == "list":
        return {n: asdict(s) for n, s in repo.active().items()}
    if command == "show":
        if args.snapshot:
            return asdict(repo.get_snapshot(args.snapshot))
        return asdict(repo.active()[args.name])
    if command == "diff":
        return {"name": args.name, "diff": repo.diff(args.name, args.snapshot)}
    if command == "freeze":
        return {"snapshot": repo.freeze(args.name)}
    if command == "retire":
        return {"snapshot": repo.retire(args.name, "CLI request")}
    if command == "rollback":
        repo.rollback(args.name, args.snapshot)
        return {"rolled_back": args.snapshot}


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(_json(asyncio.run(_run(args))))


if __name__ == "__main__":
    main()
