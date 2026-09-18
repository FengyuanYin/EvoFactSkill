"""Group-preserving construction/probe split inside meta-train only."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from evofact.generation.lineage import group_real_samples
from evofact.governance.data_policy import DataIsolationPolicy


@dataclass(frozen=True)
class RealOnlySplit:
    meta_train_ids: tuple[str, ...]
    meta_test_ids: tuple[str, ...]
    final_test_ids: tuple[str, ...]
    lineage_by_sample: dict[str, str]
    split_digest: str


def split_real_only(samples, *, seed: str, policy: DataIsolationPolicy = DataIsolationPolicy()):
    groups = group_real_samples(samples, policy=policy)
    if len(groups) < 3:
        raise ValueError("real-only split requires at least three independent lineage groups")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group.lineage_id}".encode()).hexdigest(),
    )
    train_count = max(1, int(len(ordered) * policy.meta_train_fraction))
    test_count = max(1, int(len(ordered) * policy.meta_test_fraction))
    if train_count + test_count >= len(ordered):
        train_count = len(ordered) - 2
        test_count = 1
    train_groups = ordered[:train_count]
    test_groups = ordered[train_count : train_count + test_count]
    final_groups = ordered[train_count + test_count :]
    lineage = {sample_id: group.lineage_id for group in groups for sample_id in group.sample_ids}
    payload = "|".join(
        (
            "train:"
            + ",".join(sorted(item for group in train_groups for item in group.sample_ids)),
            "test:" + ",".join(sorted(item for group in test_groups for item in group.sample_ids)),
            "final:"
            + ",".join(sorted(item for group in final_groups for item in group.sample_ids)),
        )
    )
    return RealOnlySplit(
        tuple(sorted(item for group in train_groups for item in group.sample_ids)),
        tuple(sorted(item for group in test_groups for item in group.sample_ids)),
        tuple(sorted(item for group in final_groups for item in group.sample_ids)),
        lineage,
        hashlib.sha256(payload.encode()).hexdigest(),
    )


def split_construction_probe(
    samples,
    facts,
    *,
    fraction: float,
    seed: str,
    label_contract_registry=None,
):
    """函数作用：按事件、证据正文和来源分组，将 meta-train 隔离为 construction 与 probe。
    输入要求：`samples`（未显式标注）需符合函数签名约定；`facts`（未显式标注）需符合函数签名约定；`fraction`（float）需以关键字传入并符合签名约定；`seed`（str）需以关键字传入并符合签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    by_id = {s.sample_id: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate training sample IDs")
    facts = [f for f in facts if f.sample_id in by_id]
    parent = {s.sample_id: s.sample_id for s in samples}

    def find(key):
        """函数作用：负责当前模块中的 `find` 处理，封装调用方需要复用的业务步骤。
        输入要求：`key`（未显式标注）需符合函数签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for sample in samples:
        keys = [("text", " ".join(sample.text.casefold().split()))]
        if sample.event_id:
            keys.append(("event", sample.event_id))
        keys += [("evidence", " ".join(e.text.casefold().split())) for e in sample.evidence]
        keys += [("source", e.source) for e in sample.evidence if e.source]
        keys += [("source", f.source) for f in facts if f.sample_id == sample.sample_id]
        for key in keys:
            if key in owners:
                parent[find(sample.sample_id)] = find(owners[key])
            else:
                owners[key] = sample.sample_id
    groups = defaultdict(list)
    for sample in samples:
        groups[find(sample.sample_id)].append(sample)
    domains = defaultdict(list)
    for group in groups.values():
        names = {s.domain or s.dataset for s in group}
        if len(names) != 1:
            raise ValueError("evidence/event group crosses domains")
        domains[next(iter(names))].append(group)
    construction_groups, probe_groups = [], []
    for domain, grouped in sorted(domains.items()):
        if len(grouped) < 2:
            raise ValueError(
                f"domain {domain} requires at least two independent event/evidence groups"
            )
        grouped.sort(
            key=lambda g: hashlib.sha256(
                f"{seed}:{min(s.sample_id for s in g)}".encode()
            ).hexdigest()
        )
        count = min(len(grouped) - 1, max(1, round(len(grouped) * fraction)))
        probe_groups.extend(grouped[:count])
        construction_groups.extend(grouped[count:])

    if label_contract_registry is not None:

        def labels(group):
            return {
                (
                    sample.dataset,
                    sample.label_schema_id,
                    label_contract_registry.resolve_sample(sample).normalize(sample.label),
                )
                for sample in group
            }

        all_labels = set().union(*(labels(group) for group in construction_groups + probe_groups))
        probe_labels = set().union(*(labels(group) for group in probe_groups))
        for missing in sorted(all_labels - probe_labels):
            source = next(
                (group for group in construction_groups if missing in labels(group)),
                None,
            )
            if source is None:
                continue
            domain = source[0].domain or source[0].dataset
            replacement = next(
                (
                    group
                    for group in probe_groups
                    if (group[0].domain or group[0].dataset) == domain
                    and all(
                        any(label in labels(other) for other in probe_groups if other is not group)
                        for label in labels(group)
                    )
                ),
                None,
            )
            if replacement is None:
                continue
            construction_groups.remove(source)
            probe_groups.remove(replacement)
            construction_groups.append(replacement)
            probe_groups.append(source)

    construction = [sample for group in construction_groups for sample in group]
    probe = [sample for group in probe_groups for sample in group]
    allowed = {s.sample_id for s in construction}
    selected = [f for f in facts if f.sample_id in allowed and f.verified is True]
    return (
        sorted(construction, key=lambda s: s.sample_id),
        sorted(probe, key=lambda s: s.sample_id),
        selected,
    )
