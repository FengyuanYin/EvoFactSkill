OBJECTIVES={"macro_f1_all":1,"coverage":1,"ece":-1,"mean_cost":-1}

def dominates(left: dict[str,float], right: dict[str,float]) -> bool:
    weak=[]; strict=[]
    for key,direction in OBJECTIVES.items():
        a=direction*left.get(key,0); b=direction*right.get(key,0); weak.append(a>=b); strict.append(a>b)
    return all(weak) and any(strict)

def pareto_front(candidates: list[dict[str,float]]) -> list[int]:
    return [i for i,x in enumerate(candidates) if not any(j!=i and dominates(y,x) for j,y in enumerate(candidates))]
