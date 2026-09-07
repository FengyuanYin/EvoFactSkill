def utility(metrics: dict[str,float], *, cost_weight: float=.01, calibration_weight: float=.05) -> float:
    return metrics.get("macro_f1_all",0)+.2*metrics.get("coverage",0)-cost_weight*metrics.get("mean_cost",0)-calibration_weight*metrics.get("ece",0)

def regression_failures(baseline_domains: dict[str,dict[str,float]], candidate_domains: dict[str,dict[str,float]], max_drop: float) -> tuple[str,...]:
    failures=[]
    for domain,old in baseline_domains.items():
        drop=old.get("macro_f1_all",0)-candidate_domains.get(domain,{}).get("macro_f1_all",0)
        if drop>max_drop: failures.append(f"protected domain {domain} dropped by {drop:.4f}")
    return tuple(failures)
