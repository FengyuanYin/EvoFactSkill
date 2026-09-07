from __future__ import annotations

import argparse, asyncio, json
from dataclasses import asdict, replace
from pathlib import Path

from evofact.config import load_config
from evofact.core.models import SkillStatus
from evofact.data.manifests import build_manifest, manifest_json
from evofact.data.registry import DataRegistry
from evofact.evaluation.ablations import run_ablations
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.reporting.report import skill_evolution_curve, summarize_runs, write_report
from evofact.reporting.meta_report import write_meta_report
from evofact.skills.loader import load_skill_package
from evofact.skills.repository import SkillRepository


def _json(value): return json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:asdict(x) if hasattr(x,"__dataclass_fields__") else str(x))

def build_parser():
    parser=argparse.ArgumentParser(prog="evofact"); parser.add_argument("--config",default="configs/dry_run.yaml"); parser.add_argument("--dataset",choices=("weibo21","amtcele","livefact","advfake")); parser.add_argument("--data-root"); parser.add_argument("--limit",type=int,default=0); sub=parser.add_subparsers(dest="command",required=True)
    data=sub.add_parser("data"); data_sub=data.add_subparsers(dest="data_command",required=True); data_sub.add_parser("inspect"); manifest=data_sub.add_parser("manifest"); manifest.add_argument("--output")
    for name in ("dry-run","evolve","validate","test","ablation","report"): sub.add_parser(name)
    meta=sub.add_parser("meta-evolve"); meta.add_argument("--final-test-domains",default="outer_holdout"); meta.add_argument("--episodes",type=int); meta.add_argument("--strategy",choices=("repeated_holdout","leave_one_domain_out")); meta.add_argument("--meta-test-domain-count",type=int); meta.add_argument("--ablation",choices=("no-cross-episode-aggregation","no-worst-domain-constraint","no-specialization")); meta.add_argument("--resume",action="store_true"); meta.add_argument("--evaluation-only",action="store_true")
    skills=sub.add_parser("skills"); ss=skills.add_subparsers(dest="skills_command",required=True)
    for name in ("list","show","diff","freeze","retire","rollback"):
        p=ss.add_parser(name)
        if name not in {"list"}: p.add_argument("name")
        if name in {"show","diff","rollback"}: p.add_argument("--snapshot")
    return parser

async def _run(args):
    root=Path(__file__).resolve().parents[2]; config=load_config(root/args.config if not Path(args.config).is_absolute() else args.config); runner=ExperimentRunner(config,root)
    all_samples=None; manifest=None
    if args.dataset:
        data_root=Path(args.data_root) if args.data_root else config.dataset_roots.get(args.dataset)
        if not data_root: raise ValueError(f"no data root configured for {args.dataset}")
        all_samples=DataRegistry().load(args.dataset,data_root); manifest=build_manifest(all_samples,seed=config.seed)
    def select(ids):
        chosen=[s for s in (all_samples or []) if s.sample_id in set(ids)]
        return chosen[:args.limit] if args.limit>0 else chosen
    if args.command=="data":
        if args.data_command=="inspect": return [asdict(x) for x in DataRegistry().inspect(config.dataset_roots)]
        manifest=manifest or build_manifest(fixture_samples(),seed=config.seed); text=manifest_json(manifest)
        if args.output: Path(args.output).write_text(text,encoding="utf-8")
        return json.loads(text)
    if args.command in {"dry-run","test"}:
        samples=None
        if manifest: samples=select(manifest.test_ids if args.command=="test" else manifest.train_ids)
        if manifest and not samples: raise ValueError(f"no samples available for {args.command}")
        traces,result=await runner.run(samples); return {"mode":args.command,"manifest_id":manifest.manifest_id if manifest else "fixture","n_traces":len(traces),"metrics":result.aggregate_metrics,"traces":[asdict(t) for t in traces]}
    if args.command=="evolve":
        train=select(manifest.train_ids) if manifest else None; validation=select(manifest.evolution_validation_ids) if manifest else None
        if manifest and (not train or not validation): raise ValueError("evolution requires non-empty train and evolution-validation splits")
        result=await runner.closed_loop(train,validation); repo=SkillRepository(root/config.skill_store)
        if not repo.active():
            for seed in runner.skills: repo.promote(seed)
        for proposal,decision in zip(result["proposals"],result["gate_decisions"]):
            for skill in proposal.candidate_skills:
                if decision.disposition=="active": repo.promote(skill)
                else: repo.save(replace(skill,status=SkillStatus.PARETO if decision.disposition=="pareto" else SkillStatus.CANDIDATE),decision.disposition)
        return result
    if args.command=="meta-evolve":
        rows=all_samples or fixture_meta_samples(); final_domains=tuple(sorted(x.strip() for x in args.final_test_domains.split(",") if x.strip()))
        overrides={}
        if args.episodes is not None: overrides["episodes"]=args.episodes
        if args.strategy is not None: overrides["strategy"]=args.strategy
        if args.meta_test_domain_count is not None: overrides["meta_test_domain_count"]=args.meta_test_domain_count
        if args.ablation=="no-cross-episode-aggregation": overrides.update(aggregate_across_episodes=False,min_valid_episodes=1)
        if args.ablation=="no-worst-domain-constraint": overrides["enforce_worst_domain"]=False
        if args.ablation=="no-specialization": overrides["allow_specialization"]=False
        if overrides: config=replace(config,meta_learning=replace(config.meta_learning,**overrides))
        repository=None if args.evaluation_only else SkillRepository(root/config.skill_store)
        outcome=await MetaEvolutionRunner(config,root,repository).run(rows,final_test_domains=final_domains,resume=args.resume,evaluation_only=args.evaluation_only)
        paths=write_meta_report(root/config.output_dir,outcome)
        return {"mode":"meta-evolve","run_id":outcome.run_id,"episodes":len(outcome.episodes),"episode_results":len(outcome.episode_results),"decisions":[asdict(x) for x in outcome.decisions],"committed_snapshots":outcome.committed_snapshots,"mock_results":outcome.mock_results,"reports":paths}
    if args.command=="validate":
        train=select(manifest.train_ids) if manifest else None; validation=select(manifest.evolution_validation_ids) if manifest else None
        if manifest and (not train or not validation): raise ValueError("validation requires non-empty train and evolution-validation splits")
        result=await runner.closed_loop(train,validation); return {"gate_decisions":result["gate_decisions"]}
    if args.command=="ablation": return await run_ablations(runner,select(manifest.test_ids or manifest.protected_validation_ids) if manifest else None)
    if args.command=="report":
        runs=[]
        for offset in range(3):
            seeded=ExperimentRunner(replace(config,seed=config.seed+offset),root); _,result=await seeded.run(); runs.append(result.aggregate_metrics)
        summary=summarize_runs(runs); _,grouped=await runner.run(); evolution=await runner.closed_loop(); repository=SkillRepository(root/config.skill_store)
        payload={"metrics":{k:v["mean"] for k,v in summary.items()},"multi_seed":summary,"domain_metrics":grouped.domain_metrics,"temporal_metrics":grouped.temporal_metrics,"paired_gate":[asdict(x) for x in evolution["gate_decisions"]],"skill_evolution_curve":skill_evolution_curve(repository.history()),"completed":True}
        return write_report(root/config.output_dir,"EvoFactSkill multi-seed report",payload)
    repo=SkillRepository(root/config.skill_store); command=args.skills_command
    if command=="list": return {n:asdict(s) for n,s in repo.active().items()}
    if command=="show":
        if args.snapshot:return asdict(repo.get_snapshot(args.snapshot))
        return asdict(repo.active()[args.name])
    if command=="diff":
        history=[x for x in repo.history() if x.get("name")==args.name]; return history
    if command=="freeze": return {"snapshot":repo.freeze(args.name)}
    if command=="retire": return {"snapshot":repo.retire(args.name,"CLI request")}
    if command=="rollback": repo.rollback(args.name,args.snapshot); return {"rolled_back":args.snapshot}

def main(argv=None):
    args=build_parser().parse_args(argv); print(_json(asyncio.run(_run(args))))

if __name__=="__main__": main()
