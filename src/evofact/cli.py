from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from evofact.config import load_config
from evofact.core.models import RunBudget
from evofact.data.leakage import detect_leakage
from evofact.data.manifests import build_manifest, manifest_json
from evofact.data.registry import DataRegistry
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.reporting.meta_report import write_meta_report
from evofact.reporting.report import skill_evolution_curve, summarize_runs, write_report
from evofact.runtime.progress import NullProgressSink, ProgressEvent, make_progress_sink
from evofact.skills.repository import SkillRepository


def _json_default(value):
    """函数作用：为 `json.dumps` 提供数据类与其它对象的兜底序列化。
    输入要求：`value`（未显式标注）为无法直接序列化的对象。
    输出：返回 `dict`（数据类）或 `str` 结果。"""
    return asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)


def _json(value, *, ensure_ascii: bool = False):
    """函数作用：负责当前模块中的 `_json` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定；`ensure_ascii`（bool，默认 `False`）控制是否转义非 ASCII 字符。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=2, default=_json_default)


def _is_utf8_stream(stream) -> bool:
    """函数作用：判断目标流是否按 UTF-8 编码，只有 UTF-8 才能安全写出原始非 ASCII 文本。
    输入要求：`stream` 需为具备或不具备 `encoding` 属性的文本流。
    输出：返回 `bool`；非 UTF-8（如 Windows GBK/cp1252 重定向管道）返回 `False`。"""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    normalized = encoding.lower().replace("-", "").replace("_", "")
    return normalized in {"utf8", "cp65001"}


def _emit(value, *, stream=None, output=None) -> None:
    """函数作用：写出命令结果 JSON；给出 `output` 时以 UTF-8 写文件，否则写标准输出。
    输入要求：`value`（未显式标注）为待输出的结果对象；`stream`（默认 `None`）为可写文本流；`output`（默认 `None`）为结果文件路径。
    输出：返回 `None`；按函数职责产生写文件或标准输出副作用。"""
    payload = _json(value)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 逐段写出，避免为超大结果再造一份 `payload + "\n"` 副本。
        with path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        print(f"wrote {path}", file=sys.stderr)
        return
    target = sys.stdout if stream is None else stream
    if not _is_utf8_stream(target):
        # Windows PowerShell 5.1 的 `> file` 会让 stdout 变成 GBK 之类的本地编码，并由 shell
        # 按自己的编码回读管道。此时直接写非 ASCII 要么抛 UnicodeEncodeError（整轮结果全丢），
        # 要么被 shell 解码成乱码；退化为纯 ASCII 转义 JSON 则语义不变且任何解码方都能读回。
        payload = _json(value, ensure_ascii=True)
    target.write(payload)
    target.write("\n")
    target.flush()


def build_parser():
    """函数作用：创建顶层命令行解析器及其所有子命令。
    输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    parser = argparse.ArgumentParser(prog="evofact")

    # 全局参数需要放在所选子命令之前。
    parser.add_argument("--config", default="configs/dry_run.yaml")
    parser.add_argument(
        "--dataset",
        choices=("weibo21", "amtcele", "livefact", "advfake"),
    )
    parser.add_argument("--data-root")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="global sample limit applied after splitting; place before the subcommand (0=all)",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sample-concurrency", type=int)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_const", const="on")
    progress_group.add_argument("--no-progress", dest="progress", action="store_const", const="off")

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
    manifest_parser.add_argument("--final-test-domains", default=None)

    # 无须额外专属参数的命令；统一支持 --output，把结果写文件以避免控制台重定向的编码问题。
    output_help = "把结果 JSON 以 UTF-8 写入该路径，而不是打印到标准输出"
    for name in ("dry-run", "evolve", "validate", "test", "ablation", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--output", help=output_help)
        if name != "ablation":
            command.add_argument("--final-test-domains", default=None)

    # 跨领域元演化参数，以及可选的消融实验配置覆盖项。
    meta_parser = subparsers.add_parser("meta-evolve")
    meta_parser.add_argument("--output", help=output_help)
    meta_parser.add_argument(
        "--final-test-domains",
        default=None,
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
    adversarial.add_argument("--output", help=output_help)
    adversarial.add_argument(
        "--samples", help="normalized Sample JSONL including evidence snapshots"
    )
    adversarial.add_argument("--facts", help="verified structured evidence facts JSONL")
    adversarial.add_argument("--final-test-domains", default=None)
    adversarial.add_argument("--resume", action="store_true")
    adversarial.add_argument("--evaluation-only", action="store_true")

    package_parser = subparsers.add_parser("package")
    package_subparsers = package_parser.add_subparsers(dest="package_command", required=True)
    for name in ("show", "validate", "test"):
        command = package_subparsers.add_parser(name)
        command.add_argument("target")
    package_diff = package_subparsers.add_parser("diff")
    package_diff.add_argument("before")
    package_diff.add_argument("after")

    plan_parser = subparsers.add_parser("plan")
    plan_subparsers = plan_parser.add_subparsers(dest="plan_command", required=True)
    for name in ("inspect", "validate"):
        command = plan_subparsers.add_parser(name)
        command.add_argument("path")

    generation_parser = subparsers.add_parser("generation")
    generation_subparsers = generation_parser.add_subparsers(
        dest="generation_command", required=True
    )
    for name in ("audit", "report"):
        command = generation_subparsers.add_parser(name)
        command.add_argument("path")

    generator_evolve = subparsers.add_parser("generator-evolve")
    generator_evolve.add_argument("--evaluation-only", action="store_true")
    generator_evolve.add_argument("--propose-only", action="store_true")
    generator_candidate = generator_evolve.add_mutually_exclusive_group(required=True)
    generator_candidate.add_argument("--candidate")
    generator_candidate.add_argument(
        "--audit",
        help="generation audit JSON used by the Package Optimizer to produce a candidate",
    )
    generator_evolve.add_argument("--evaluation")
    generator_evolve.add_argument("--run-id")

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


async def _run_impl(args, progress):
    """函数作用：根据命令行子命令装配数据、运行器和报告流程，并分派实际任务。
    输入要求：`args`（未显式标注）需符合函数签名约定。
    输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config if not Path(args.config).is_absolute() else args.config)
    execution_overrides = {}
    if args.batch_size is not None:
        execution_overrides["batch_size"] = args.batch_size
    if args.sample_concurrency is not None:
        execution_overrides["max_concurrent_samples"] = args.sample_concurrency
    if args.progress is not None:
        execution_overrides["progress"] = args.progress
    if execution_overrides:
        config = replace(
            config,
            execution=replace(config.execution, **execution_overrides),
        )
    runner = ExperimentRunner(config, root)
    if args.command in {
        "dry-run",
        "test",
        "evolve",
        "validate",
        "ablation",
        "report",
        "plan",
    }:
        repository = SkillRepository(root / config.skill_store)
        active_packages = repository.active_packages()
        if active_packages:
            from evofact.skills.package_adapter import package_to_skill_spec

            runner.packages = list(active_packages.values())
            runner.skills = [package_to_skill_spec(package) for package in runner.packages]
        elif (root / config.skill_store / "active.json").exists():
            runner.skills = list(repository.active().values())
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    if args.command == "package":
        from evofact.governance.package_policy import validate_package
        from evofact.skills.package_diff import diff_packages
        from evofact.skills.package_loader import load_package
        from evofact.skills.package_serializer import package_to_dict

        repository = SkillRepository(root / config.skill_store)

        def resolve_package(target):
            path = Path(target)
            if path.is_dir():
                return load_package(path)
            if len(target) == 64:
                return repository.get_package(target)
            return repository.active_packages()[target]

        if args.package_command == "diff":
            return asdict(diff_packages(resolve_package(args.before), resolve_package(args.after)))
        package = resolve_package(args.target)
        if args.package_command == "show":
            return package_to_dict(package)
        report = validate_package(package)
        return {
            "command": args.package_command,
            "package_digest": package.package_digest,
            "validation": asdict(report),
            "declared_tests": list(package.manifest.entrypoints.tests),
        }

    if args.command == "plan":
        from evofact.governance.dag_policy import validate_plan
        from evofact.routing.plan_normalizer import normalize_plan

        raw = json.loads(Path(args.path).read_text(encoding="utf-8"))
        budget = RunBudget(**raw.get("budget", {}))
        plan = normalize_plan(
            raw["nodes"],
            reasons=raw.get("reasons", {}),
            confidence=float(raw.get("confidence", 0)),
            budget=budget,
            fallback_used=bool(raw.get("fallback_used", False)),
        )
        report = validate_plan(plan, runner.skills, limits=runner._budget_manager().limits)
        return {
            "command": args.plan_command,
            "plan": asdict(plan),
            "validation": asdict(report),
        }

    if args.command == "generation":
        from evofact.generation.audit_store import GenerationAuditStore

        payload = GenerationAuditStore(args.path).load_raw()
        if args.generation_command == "audit":
            return payload
        entries = payload.get("entries", [])
        return {
            "schema_version": payload["schema_version"],
            "accepted": sum(bool(item.get("accepted")) for item in entries),
            "rejected": sum(not bool(item.get("accepted")) for item in entries),
        }

    if args.command == "generator-evolve":
        from evofact.core.generation_models import GenerationMetrics, PairedGeneratorEvaluation
        from evofact.evolution.package_candidate import PackageCandidate
        from evofact.evolution.package_optimizer import PackageOptimizerAgent
        from evofact.experiments.generator_runner import (
            GeneratorEvolutionRunner,
            validate_evaluation_provenance,
        )
        from evofact.generation.prompts import VERIFIER_SYSTEM
        from evofact.governance.package_policy import validate_package
        from evofact.security.package_scanner import scan_package
        from evofact.skills.package_loader import load_package
        from evofact.skills.package_serializer import package_from_dict, package_to_dict

        repository = SkillRepository(root / config.skill_store)
        active = repository.active_packages()
        package_bank = list(active.values()) if active else list(runner.packages)
        champion = next(item for item in package_bank if item.manifest.name == "generation_agent")
        if args.candidate:
            candidate_path = Path(args.candidate)
            package = (
                load_package(candidate_path)
                if candidate_path.is_dir()
                else package_from_dict(json.loads(candidate_path.read_text(encoding="utf-8")))
            )
            candidate = None
        else:
            audit_raw = json.loads(Path(args.audit).read_text(encoding="utf-8"))
            audits = audit_raw.get("entries") if isinstance(audit_raw, dict) else audit_raw
            if not isinstance(audits, list) or not audits:
                raise ValueError("--audit must contain a non-empty generation audit entry list")
            optimizer_packages = [
                item
                for item in package_bank
                if item.manifest.name == config.evolution.optimizer_skill
            ]
            if len(optimizer_packages) != 1:
                raise ValueError("optimizer Package must resolve to exactly one Package")
            candidate = await PackageOptimizerAgent(
                runner._backend(),
                optimizer_packages[0],
                budget_manager=runner._budget_manager(),
            ).propose(
                champion,
                package_bank,
                audits=audits,
            )
            if candidate is None:
                raise ValueError("Package Optimizer proposed no Generator change")
            package = candidate.package
        validation = validate_package(package)
        if not validation.valid:
            raise ValueError("candidate Generator Package failed fixed validation")
        safety = scan_package(package)
        if safety.level == "blocked":
            raise ValueError("candidate Generator Package failed safety scan")
        if candidate is None:
            candidate = PackageCandidate(
                champion,
                None,
                package,
                validation,
                safety.level,
                safety.findings,
            )
        if args.propose_only:
            if args.evaluation is not None:
                raise ValueError("--propose-only cannot be combined with --evaluation")
            return {
                "mode": "generator-propose",
                "candidate_digest": package.package_digest,
                "candidate_package": package_to_dict(package),
                "safety_level": safety.level,
                "safety_findings": list(safety.findings),
                "budget_snapshot": runner._budget_manager().snapshot().model_dump(),
            }
        if args.evaluation is None:
            raise ValueError("generator-evolve requires --evaluation unless --propose-only is set")
        raw = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))

        def metrics(value):
            value = dict(value)
            value["cost"] = Decimal(str(value["cost"])) if value.get("cost") is not None else None
            return GenerationMetrics(**value)

        evaluation = PairedGeneratorEvaluation(
            source_episode_id=raw["source_episode_id"],
            evaluation_episode_id=raw["evaluation_episode_id"],
            champion_digest=raw["champion_digest"],
            challenger_digest=raw["challenger_digest"],
            champion=metrics(raw["champion"]),
            challenger=metrics(raw["challenger"]),
            metadata=raw.get("metadata", {}),
            label_contract_digest=raw.get("label_contract_digest", ""),
        )
        if (
            evaluation.label_contract_digest
            and evaluation.label_contract_digest != runner.label_contract_registry.digest
        ):
            raise ValueError("paired evaluation label contract identity has drifted")
        pricing_identity = runner.pricing_identity()
        verifier_fingerprint = hashlib.sha256(
            f"{VERIFIER_SYSTEM}\0{config.backend}\0{config.model}".encode("utf-8")
        ).hexdigest()
        validate_evaluation_provenance(
            evaluation,
            verifier_fingerprint=verifier_fingerprint,
            pricing_identity=pricing_identity,
            label_contract_digest=runner.label_contract_registry.digest,
        )
        if not args.evaluation_only and not active:
            repository.commit_package_bank(
                runner.packages,
                run_id="initialize-generator-evolution-bank",
                expected_active={},
                audit={"source": "seed-packages"},
            )
        run_id = args.run_id or (
            "generator-evolve-"
            + evaluation.evaluation_episode_id
            + "-"
            + package.package_digest[:12]
        )
        outcome = GeneratorEvolutionRunner(repository).finalize(
            candidate,
            evaluation,
            run_id=run_id,
            evaluation_only=args.evaluation_only,
        )
        return {
            "mode": "generator-evolve",
            "run_id": run_id,
            "evaluation_only": args.evaluation_only,
            "candidate_digest": outcome.candidate_digest,
            "evaluation": asdict(outcome.evaluation),
            "decision": asdict(outcome.decision),
            "active_bank_updated": outcome.committed,
            "label_contract_digest": runner.label_contract_registry.digest,
        }

    configured_dataset = config.data.dataset
    if args.dataset and configured_dataset and args.dataset != configured_dataset:
        raise ValueError(
            f"--dataset {args.dataset!r} conflicts with configured dataset {configured_dataset!r}"
        )
    dataset = args.dataset or configured_dataset
    if args.data_root and not dataset:
        raise ValueError("--data-root requires a configured or command-line dataset")
    raw_cli_final = getattr(args, "final_test_domains", None)
    cli_final_domains = (
        tuple(sorted(x.strip() for x in raw_cli_final.split(",") if x.strip()))
        if raw_cli_final
        else ()
    )

    all_samples = None
    manifest = None
    train_domains: tuple[str, ...] | None = None
    final_test_domains: tuple[str, ...] | None = None
    if dataset:  # 加载单一数据集，并应用配置中的领域边界。
        uses_data_config = dataset == configured_dataset and configured_dataset is not None
        data_root = Path(args.data_root) if args.data_root else None
        if data_root is None and uses_data_config:
            data_root = config.data.root
        if data_root is None:
            data_root = config.dataset_roots.get(dataset)
        if not data_root:
            raise ValueError(f"no data root configured for {dataset}")
        if not data_root.is_absolute():
            data_root = root / data_root

        excluded_domains = config.data.excluded_domains if uses_data_config else ()
        data_registry = DataRegistry()
        all_samples = data_registry.load(
            dataset,
            data_root,
            excluded_domains=excluded_domains,
        )
        if not all_samples:
            raise ValueError("dataset has no samples")

        if uses_data_config:
            train_domains = config.data.train_domains
            final_test_domains = config.data.final_test_domains
        if cli_final_domains:
            known = {str(sample.domain or sample.dataset) for sample in all_samples}
            unknown = set(cli_final_domains) - known
            if unknown:
                raise ValueError(
                    "--final-test-domains contains unknown domains: " + ", ".join(sorted(unknown))
                )
            final_test_domains = cli_final_domains
            # An explicit evaluation-domain override defines a new legal holdout split.
            # Recompute the source side from all remaining loaded domains so the manifest,
            # leakage checks, and output identity describe what was actually evaluated.
            train_domains = tuple(sorted(known - set(final_test_domains)))
        if final_test_domains and train_domains is None:
            known = {str(sample.domain or sample.dataset) for sample in all_samples}
            train_domains = tuple(sorted(known - set(final_test_domains)))

        manifest = build_manifest(
            all_samples,
            seed=config.seed,
            train_domains=train_domains,
            final_test_domains=final_test_domains,
            label_contract_digest=data_registry.label_contracts.digest,
        )
        errors = detect_leakage(all_samples, manifest)
        if errors:
            raise ValueError("dataset leakage: " + "; ".join(errors))

    def select(ids):
        """函数作用：根据样本、技能范围、历史效用和预算选择本次调用的技能。
        输入要求：`ids`（未显式标注）需符合函数签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        chosen = [s for s in (all_samples or []) if s.sample_id in set(ids)]
        return chosen[: args.limit] if args.limit > 0 else chosen

    if args.command == "data":
        if args.data_command == "inspect":
            roots = dict(config.dataset_roots)
            if configured_dataset and config.data.root:
                configured_root = config.data.root
                if not configured_root.is_absolute():
                    configured_root = root / configured_root
                roots[configured_dataset] = configured_root
            return [asdict(x) for x in DataRegistry().inspect(roots)]
        manifest = manifest or build_manifest(
            fixture_samples(),
            seed=config.seed,
            label_contract_digest=runner.label_contract_registry.digest,
        )
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
        traces, result = await runner.run(
            samples,
            progress=progress if args.command == "test" else None,
            task_name=args.command,
        )
        return {
            "mode": args.command,
            "manifest_id": manifest.manifest_id if manifest else "fixture",
            "n_traces": len(traces),
            "metrics": result.aggregate_metrics,
            "traces": [asdict(t) for t in traces],
            "budget": runner._budget_manager().snapshot().model_dump(),
        }
    if args.command == "evolve":  # 使用skill 进化
        train = select(manifest.train_ids) if manifest else None
        validation = select(manifest.evolution_validation_ids) if manifest else None
        if manifest and (not train or not validation):
            raise ValueError("evolution requires non-empty train and evolution-validation splits")
        repo = SkillRepository(root / config.skill_store)
        return await runner.closed_loop_batched(
            train,
            validation,
            repository=repo,
            progress=progress,
        )
    if args.command == "meta-evolve":  # 使用元进化
        rows = fixture_meta_samples() if all_samples is None else all_samples
        final_domains = final_test_domains or cli_final_domains or ("outer_holdout",)
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
            train_domains=train_domains,
            resume=args.resume,
            evaluation_only=args.evaluation_only,
            progress=progress,
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
        final = final_test_domains or cli_final_domains or ("outer_holdout",)
        repo = None if args.evaluation_only else SkillRepository(root / config.skill_store)
        adversarial_runner = AdversarialEvolutionRunner(config, root, facts, repo)
        outcome = await adversarial_runner.run(
            rows,
            final_test_domains=final,
            resume=args.resume,
            evaluation_only=args.evaluation_only,
            progress=progress,
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
        raise ValueError(
            "ablation is disabled until every named arm has an independent implementation"
        )
    if args.command == "report":
        report_rows = select(manifest.test_ids) if manifest else None
        if manifest and not report_rows:
            raise ValueError("report requires a non-empty final-test split")
        runs = []
        for offset in range(3):
            progress.update(
                ProgressEvent(
                    "report",
                    "seeds",
                    offset,
                    3,
                    "seeds",
                    f"seed {offset + 1}/3",
                )
            )
            seeded = ExperimentRunner(replace(config, seed=config.seed + offset), root)
            seeded.skills = runner.skills
            _, result = await seeded.run(
                report_rows,
                progress=progress,
                task_name=f"report seed {offset + 1}/3",
            )
            runs.append(result.aggregate_metrics)
            progress.update(ProgressEvent("report", "seeds", offset + 1, 3, "seeds"))
        summary = summarize_runs(runs)
        _, grouped = await runner.run(
            report_rows,
            progress=progress,
            task_name="report grouped metrics",
        )
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


async def _run(args):
    root = Path(__file__).resolve().parents[2]
    configured = load_config(
        root / args.config if not Path(args.config).is_absolute() else args.config
    )
    mode = args.progress or configured.execution.progress
    observed = {"test", "report", "evolve", "meta-evolve", "adversarial-evolve"}
    progress = make_progress_sink(mode) if args.command in observed else NullProgressSink()
    success = False
    try:
        result = await _run_impl(args, progress)
        success = True
        return result
    finally:
        progress.close(success=success)


def main(argv=None):
    """函数作用：作为命令行入口解析参数、运行异步任务并输出 JSON 结果。
    输入要求：`argv`（未显式标注，默认 `None`）需符合函数签名约定。
    输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
    args = build_parser().parse_args(argv)
    try:
        value = asyncio.run(_run(args))
    except KeyboardInterrupt:
        _emit(
            {
                "cancelled": True,
                "reason": "keyboard interrupt",
                "resume": "Use --resume for meta-evolve or adversarial-evolve.",
            }
        )
        raise SystemExit(130) from None
    output = getattr(args, "output", None)
    # `data manifest` 已按 manifest 文本自行写盘，这里保持其原有的写文件 + 打印行为。
    if output and args.command != "data":
        _emit(value, output=output)
        return
    _emit(value)


if __name__ == "__main__":
    main()
