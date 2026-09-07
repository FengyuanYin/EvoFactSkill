import csv, json
from dataclasses import asdict
from pathlib import Path
from math import sqrt
from statistics import mean, stdev

def summarize_runs(metric_runs:list[dict[str,float]])->dict:
    keys=sorted(set().union(*(x.keys() for x in metric_runs))) if metric_runs else []
    summary={}
    for key in keys:
        values=[float(x[key]) for x in metric_runs if key in x]
        sd=stdev(values) if len(values)>1 else 0.0; half=1.96*sd/sqrt(len(values)) if values else 0.0; avg=mean(values) if values else 0.0
        summary[key]={"mean":avg,"std":sd,"ci95":[avg-half,avg+half],"n_runs":len(values)}
    return summary

def skill_evolution_curve(events:list[dict])->list[dict]:
    return [{"step":index+1,"action":event.get("action"),"skill":event.get("name"),"status":event.get("status"),"timestamp":event.get("timestamp")} for index,event in enumerate(events)]

def write_report(path:Path,title:str,payload:dict)->dict[str,str]:
    path=Path(path); path.mkdir(parents=True,exist_ok=True); json_path=path/"report.json"; md_path=path/"report.md"; csv_path=path/"metrics.csv"
    clean=json.loads(json.dumps(payload,default=lambda x:asdict(x) if hasattr(x,"__dataclass_fields__") else str(x)))
    metrics=clean.get("metrics") or clean.get("evaluation",{}).get("aggregate_metrics",{})
    reasons=[]
    if clean.get("completed") is False: reasons.append("run is incomplete")
    if metrics.get("n",1)<1: reasons.append("no evaluated samples")
    if clean.get("parse_failures",0)>0: reasons.append("model output parse failures occurred")
    clean["run_status"]={"valid":not reasons,"reasons":reasons}
    json_path.write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding="utf-8")
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(["metric","value"]); writer.writerows(sorted(metrics.items()))
    md_path.write_text(f"# {title}\n\n```json\n{json.dumps(metrics,ensure_ascii=False,indent=2)}\n```\n",encoding="utf-8")
    return {"json":str(json_path),"csv":str(csv_path),"markdown":str(md_path)}
