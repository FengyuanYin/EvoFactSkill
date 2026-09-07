import math, random
from evofact.core.models import SampleEvaluation, StatisticalTestResult

def mcnemar(baseline: list[SampleEvaluation], candidate: list[SampleEvaluation], alpha: float=.05) -> StatisticalTestResult:
    pairs={r.sample_id:r for r in baseline}; b=c=0
    for row in candidate:
        old=pairs[row.sample_id]; old_ok=old.gold==old.predicted; new_ok=row.gold==row.predicted
        if old_ok and not new_ok:b+=1
        elif new_ok and not old_ok:c+=1
    stat=(abs(b-c)-1)**2/(b+c) if b+c else 0
    p=math.erfc(math.sqrt(stat/2)) if b+c else 1
    return StatisticalTestResult("mcnemar",stat,p,p<alpha)

def paired_bootstrap(baseline: list[SampleEvaluation], candidate: list[SampleEvaluation], *, seed: int=42, iterations: int=1000) -> tuple[float,float]:
    old={r.sample_id:r for r in baseline}; pairs=[(old[r.sample_id],r) for r in candidate]; rng=random.Random(seed); deltas=[]
    if not pairs:return (0,0)
    for _ in range(iterations):
        chosen=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum((n.gold==n.predicted)-(o.gold==o.predicted) for o,n in chosen)/len(chosen))
    deltas.sort(); return deltas[int(.025*iterations)], deltas[min(iterations-1,int(.975*iterations))]
