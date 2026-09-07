import hashlib, json, random
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable
from evofact.core.models import DataManifest, Sample

def sample_fingerprint(sample: Sample) -> str:
    normalized = " ".join(sample.text.casefold().split())
    return hashlib.sha256(f"{sample.dataset}\0{normalized}".encode()).hexdigest()

def build_manifest(samples: Iterable[Sample], *, seed: int = 42, evolution_ratio: float = .15, protected_ratio: float = .15) -> DataManifest:
    rows = list(samples); grouped: dict[str, list[Sample]] = defaultdict(list); event_groups: dict[str,list[Sample]]=defaultdict(list); test=[]; fps={}
    for sample in rows:
        fps[sample.sample_id] = sample_fingerprint(sample)
        official = str(sample.metadata.get("official_split", "")).casefold()
        role=sample.metadata.get("split_role")
        if role == "test" or official == "test": test.append(sample.sample_id)
        elif role == "protected": protected_id=sample.sample_id; grouped[f"__forced_protected__:{protected_id}"]=[]; grouped[f"__forced_protected__:{protected_id}"].append(sample)
        elif sample.event_id: event_groups[sample.event_id].append(sample)
        else: grouped[sample.domain or "unknown"].append(sample)
    train=[]; evolution=[]; protected=[]
    for key in sorted(grouped):
        bucket=sorted(grouped[key], key=lambda s:s.sample_id); random.Random(f"{seed}:{key}").shuffle(bucket); n=len(bucket)
        if key.startswith("__forced_protected__:"):
            protected += [s.sample_id for s in bucket]; continue
        ne=max(1,round(n*evolution_ratio)) if n>=3 else 0; np=max(1,round(n*protected_ratio)) if n>=3 else 0
        evolution += [s.sample_id for s in bucket[:ne]]; protected += [s.sample_id for s in bucket[ne:ne+np]]; train += [s.sample_id for s in bucket[ne+np:]]
    for event_id,bucket in sorted(event_groups.items()):
        draw=random.Random(f"{seed}:event:{event_id}").random(); ids=[s.sample_id for s in bucket]
        if draw<evolution_ratio: evolution+=ids
        elif draw<evolution_ratio+protected_ratio: protected+=ids
        else: train+=ids
    payload={"seed":seed,"train":sorted(train),"evolution":sorted(evolution),"protected":sorted(protected),"test":sorted(test),"fingerprints":dict(sorted(fps.items()))}
    mid=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    return DataManifest(mid,fps,tuple(payload["train"]),tuple(payload["evolution"]),tuple(payload["protected"]),tuple(payload["test"]),{"seed":seed,"evolution_ratio":evolution_ratio,"protected_ratio":protected_ratio})

def manifest_json(manifest: DataManifest) -> str:
    return json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2)
