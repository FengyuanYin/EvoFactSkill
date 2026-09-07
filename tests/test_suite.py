import asyncio, json, os, subprocess, sys, tempfile, unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from evofact.config import GateConfig, load_config
from evofact.core.models import (DataManifest, Evidence, ErrorType, EvolutionOperation,
    EvolutionProposal, InferenceTrace, Prediction, RoutingDecision, Sample,
    SampleEvaluation, SkillStatus, SpecialistReport, StatisticalTestResult)
from evofact.attribution.rules import attribute_trace
from evofact.data.leakage import detect_leakage
from evofact.data.manifests import build_manifest
from evofact.data.registry import DataRegistry
from evofact.evaluation.metrics import compute_metrics
from evofact.experiments.runner import ExperimentRunner
from evofact.experiments.checkpoint import Checkpoint
from evofact.evolution.meta import MetaEvolutionLoop
from evofact.reporting.report import write_report
from evofact.security.scanner import scan_resources
from evofact.skills.lifecycle import execute
from evofact.skills.loader import load_skill_package
from evofact.skills.repository import SkillRepository
from evofact.skills.utility import UtilityTracker
from evofact.validation.evaluator import evaluate
from evofact.validation.gate import ValidationGate
from evofact.validation.pareto import dominates
from evofact.validation.statistics import paired_bootstrap


class CoreTests(unittest.TestCase):
    def test_config(self):
        c=load_config(ROOT/"configs/dry_run.yaml"); self.assertEqual(c.backend,"mock"); self.assertEqual(c.gate.repeats,2)
    def test_redaction_and_freeze(self):
        s=Sample("1","x","claim",1,metadata={"gold":"x","nested":{"Answer":2}}); self.assertNotIn("label",s.public_view()); self.assertEqual(s.public_view()["metadata"],{"nested":{}})
        with self.assertRaises(FrozenInstanceError): s.text="changed"
    def test_manifest_overlap(self):
        with self.assertRaises(ValueError): DataManifest("x",{},("1",),(),(),("1",))

class DataTests(unittest.TestCase):
    def test_registry(self):
        r=DataRegistry(); self.assertEqual(r.names(),("advfake","amtcele","livefact","weibo21")); self.assertTrue(all(not x.available for x in r.inspect({})))
    def test_manifest(self):
        rows=[Sample(str(i),"x",f"text {i}",domain="d") for i in range(10)]+[Sample("a","advfake","attack",metadata={"split_role":"test","robustness_only":True})]
        self.assertEqual(build_manifest(rows,seed=9).manifest_id,build_manifest(reversed(rows),seed=9).manifest_id); self.assertEqual(build_manifest(rows,seed=9).test_ids,("a",))
    def test_event_and_future_evidence(self):
        rows=[Sample(f"e{i}","x",f"u{i}",event_id="same") for i in range(4)]; self.assertFalse(detect_leakage(rows,build_manifest(rows)))
        now=datetime(2025,1,1); s=Sample("x","x","claim",published_at=now,evidence=(Evidence("future",published_at=now+timedelta(days=1)),)); self.assertTrue(any("future" in x for x in detect_leakage([s],build_manifest([s]))))
    def test_all_adapters_load_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); registry=DataRegistry()
            weibo=root/"weibo"; weibo.mkdir(); (weibo/"train.jsonl").write_text('{"content":"claim","label":1,"category":"tech"}\n',encoding="utf-8")
            amt=root/"amt"; amt.mkdir(); (amt/"data.csv").write_text("text,label,domain\nclaim,legit,tech1\n",encoding="utf-8")
            live=root/"live"/"2026-01"; live.mkdir(parents=True); (live/"livefact_+3_cls.jsonl").write_text('{"claim":"claim","label":"FAKE"}\n',encoding="utf-8")
            adv=root/"adv"; adv.mkdir(); (adv/"final.csv").write_text("id,adversarial_text,label\n1,attack,FAKE\n",encoding="utf-8")
            self.assertEqual(len(registry.load("weibo21",weibo)),1); self.assertEqual(registry.load("amtcele",amt)[0].domain,"tech")
            self.assertEqual(registry.load("livefact",root/"live")[0].metadata["split_role"],"test"); self.assertTrue(registry.load("advfake",adv)[0].metadata["robustness_only"])

class MetricTests(unittest.TestCase):
    def _rows(self,preds): return [SampleEvaluation(str(i),"REAL" if i%2==0 else "FAKE",p,.8,domain="protected") for i,p in enumerate(preds)]
    def test_abstention_denominator(self):
        rows=[SampleEvaluation(str(i),"REAL","REAL" if i==0 else "ABSTAIN",.9) for i in range(50)]; m=compute_metrics(rows); self.assertEqual(m["accuracy_all"],.02); self.assertEqual(m["coverage"],.02)
    def test_statistics_and_pareto(self):
        old=self._rows(["FAKE","REAL","REAL","REAL"]); new=self._rows(["REAL","FAKE","REAL","FAKE"]); self.assertEqual(paired_bootstrap(old,new,seed=1,iterations=100),paired_bootstrap(old,new,seed=1,iterations=100)); self.assertTrue(dominates({"macro_f1_all":.9,"coverage":1,"ece":.1,"mean_cost":1},{"macro_f1_all":.8,"coverage":1,"ece":.2,"mean_cost":2}))
    def test_gate_rejects_abstention(self):
        old=evaluate(self._rows(["REAL","FAKE","REAL","FAKE"])); new=evaluate(self._rows(["ABSTAIN"]*4),{"paired_accuracy_delta":(-1,-.5)}); d=ValidationGate(GateConfig(repeats=2,min_coverage=.8)).decide(old,new,StatisticalTestResult("x",1,.01,True)); self.assertFalse(d.accepted); self.assertTrue(d.regression_failures)

class SkillTests(unittest.TestCase):
    def seed(self): return load_skill_package(ROOT/"skills/seeds/claim_decomposition")
    def test_seed_bank(self): self.assertEqual(len([load_skill_package(p) for p in (ROOT/"skills/seeds").iterdir() if (p/"SKILL.md").exists()]),9)
    def test_repository(self):
        with tempfile.TemporaryDirectory() as d:
            repo=SkillRepository(Path(d)); s=self.seed(); first=repo.promote(s); repo.freeze(s.name); repo.rollback(s.name,first); self.assertEqual(repo.active()[s.name].status,SkillStatus.ACTIVE)
            before=repo.active()
            with self.assertRaises(Exception): repo.rollback(s.name,"missing")
            self.assertEqual(repo.active(),before); repo.retire(s.name,"harmful"); self.assertNotIn(s.name,repo.active())
    def test_lifecycle_operations(self):
        cases=[(EvolutionOperation.ADD,(),1),(EvolutionOperation.EDIT,("a",),1),(EvolutionOperation.SPLIT,("a",),2),(EvolutionOperation.MERGE,("a","b"),1),(EvolutionOperation.GENERALIZE,("a",),1),(EvolutionOperation.SPECIALIZE,("a",),1)]
        with tempfile.TemporaryDirectory() as d:
            repo=SkillRepository(Path(d))
            for operation,targets,count in cases:
                candidates=tuple(replace(self.seed(),skill_id=f"{operation}{i}",name=f"skill_{operation}_{i}",status=SkillStatus.CANDIDATE) for i in range(count)); self.assertEqual(len(execute(repo,EvolutionProposal(str(operation),operation,"reason",targets,candidates))),count)
            before=repo.history()
            with self.assertRaises(ValueError): execute(repo,EvolutionProposal("bad",EvolutionOperation.SPLIT,"bad",("a",),(self.seed(),)))
            self.assertEqual(repo.history(),before)
    def test_security_and_utility(self):
        self.assertEqual(scan_resources({"references/a":"ok"}).level,"safe"); self.assertEqual(scan_resources({"scripts/a.py":"import subprocess"}).level,"blocked"); self.assertEqual(scan_resources({"scripts/a.py":"def f(x:int): return x"}).level,"review_required")
        tracker=UtilityTracker(3)
        for _ in range(2): tracker.update("s",success=False,delta=-1,cost=1)
        self.assertEqual(tracker.recommendation("s"),"downweight"); tracker.update("s",success=False,delta=-1,cost=1); self.assertEqual(tracker.recommendation("s"),"retire")
    def test_review_required_cannot_promote(self):
        row=SampleEvaluation("x","REAL","REAL",.8,domain="d"); result=evaluate([row],{"paired_accuracy_delta":(.1,.2)}); gate=ValidationGate(GateConfig(repeats=2,min_macro_f1_gain=0,min_coverage=0)).decide(result,result,StatisticalTestResult("x",1,.01,True),safety="review_required"); self.assertFalse(gate.accepted); self.assertEqual(gate.disposition,"review_required")

class IntegrationTests(unittest.TestCase):
    def runner(self): return ExperimentRunner(load_config(ROOT/"configs/dry_run.yaml"),ROOT)
    def test_trace_and_evolution(self):
        traces,result=asyncio.run(self.runner().run()); self.assertEqual(len(traces),4); self.assertNotIn("label",traces[0].sample_public); self.assertEqual(result.aggregate_metrics["coverage"],1)
        outcome=asyncio.run(self.runner().closed_loop()); self.assertTrue(any(ErrorType.REASONING_ERROR in x.error_types for x in outcome["attributions"])); self.assertTrue(outcome["proposals"]); self.assertTrue(outcome["gate_decisions"])
    def test_all_error_taxonomy_paths(self):
        base=asyncio.run(self.runner().run())[0][0]; sid=base.routing.selected_skill_ids[0]
        variants=[
            attribute_trace(replace(base,routing=RoutingDecision((),fallback_used=True),specialist_reports=(),decision=Prediction("REAL",.8)),"FAKE"),
            attribute_trace(replace(base,specialist_reports=(),decision=Prediction("REAL",.8)),"FAKE"),
            attribute_trace(replace(base,specialist_reports=(SpecialistReport(sid,assessment="fake"),),decision=Prediction("REAL",.8)),"FAKE"),
            attribute_trace(replace(base,specialist_reports=(SpecialistReport(sid,assessment="real"),),decision=Prediction("REAL",.8)),"FAKE"),
            attribute_trace(replace(base,decision=Prediction("ABSTAIN",.2)),"FAKE"),
            attribute_trace(replace(base,errors=("hallucinated evidence","future evidence")),"REAL"),
            attribute_trace(base,"UNKNOWN"),
        ]
        found={e for report in variants for e in report.error_types}; self.assertEqual(found,set(ErrorType))
    def test_test_mode_keeps_seed_bank_read_only(self):
        files=sorted((ROOT/"skills/seeds").rglob("*")); before={str(p):p.read_bytes() for p in files if p.is_file()}; asyncio.run(self.runner().run()); after={str(p):p.read_bytes() for p in files if p.is_file()}; self.assertEqual(before,after)
    def test_eight_arms(self):
        env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"}; result=subprocess.run([sys.executable,"-m","evofact.cli","--config","configs/dry_run.yaml","ablation"],cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",check=True); self.assertEqual(len(json.loads(result.stdout)),8)
    def test_checkpoint_meta_and_invalid_report(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); checkpoint=Checkpoint(base/"checkpoint.jsonl")
            self.assertTrue(checkpoint.append("x",{"ok":True})); self.assertFalse(checkpoint.append("x",{"ok":True})); self.assertEqual(Checkpoint(base/"checkpoint.jsonl").load(),{"x"})
            disabled=MetaEvolutionLoop(False,base/"meta/events.jsonl"); self.assertFalse(disabled.run([])["enabled"]); self.assertFalse((base/"meta/events.jsonl").exists())
            enabled=MetaEvolutionLoop(True,base/"meta/events.jsonl"); enabled.run([{"x":1}]); self.assertTrue((base/"meta/events.jsonl").exists())
            paths=write_report(base/"report","invalid",{"metrics":{"n":0},"completed":False}); payload=json.loads(Path(paths["json"]).read_text(encoding="utf-8")); self.assertFalse(payload["run_status"]["valid"])

if __name__=="__main__": unittest.main()
