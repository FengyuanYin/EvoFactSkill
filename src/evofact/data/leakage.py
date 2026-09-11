from collections import defaultdict

from evofact.core.models import DataManifest, Sample

from .manifests import sample_fingerprint


def detect_leakage(samples: list[Sample], manifest: DataManifest) -> list[str]:
    """函数作用：负责当前模块中的 `detect_leakage` 处理，封装调用方需要复用的业务步骤。
    输入要求：`samples`（list[Sample]）需符合函数签名约定；`manifest`（DataManifest）需符合函数签名约定。
    输出：返回 `list[str]` 类型结果；校验或下游调用失败时异常向上传递。"""
    by_id = {s.sample_id: s for s in samples}
    split_of = {}
    errors = []
    groups = (
        ("train", manifest.train_ids),
        ("evolution", manifest.evolution_validation_ids),
        ("protected", manifest.protected_validation_ids),
        ("test", manifest.test_ids),
    )
    for name, ids in groups:
        for sid in ids:
            if sid in split_of:
                errors.append(f"sample id appears in multiple splits: {sid}")
            split_of[sid] = name
    fps: dict[str, set[str]] = defaultdict(set)
    events: dict[str, set[str]] = defaultdict(set)
    for sid, split in split_of.items():
        sample = by_id.get(sid)
        if not sample:
            errors.append(f"manifest references missing sample: {sid}")
            continue
        fps[sample_fingerprint(sample)].add(split)
        if sample.event_id:
            events[sample.event_id].add(split)
        if split != "test" and sample.metadata.get("robustness_only"):
            errors.append(f"robustness-only sample outside test: {sid}")
        if any(
            sample.published_at and e.published_at and e.published_at > sample.published_at
            for e in sample.evidence
        ):
            errors.append(f"future evidence for {sid}")
    errors += [f"duplicate content across splits: {sorted(v)}" for v in fps.values() if len(v) > 1]
    errors += [f"event {k} crosses splits: {sorted(v)}" for k, v in events.items() if len(v) > 1]
    return sorted(set(errors))
