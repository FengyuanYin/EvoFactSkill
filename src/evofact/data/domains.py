from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict

from evofact.core.models import Sample, SkillSpec

from .deduplication import content_fingerprint


def effective_domain(sample: Sample) -> str:
    """函数作用：负责当前模块中的 `effective_domain` 处理，封装调用方需要复用的业务步骤。
    输入要求：`sample`（Sample）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    domain = str(sample.domain or sample.dataset or "").strip()
    if not domain:
        raise ValueError(f"sample {sample.sample_id!r} has no domain or dataset")
    return domain


def domain_index(samples: Sequence[Sample]) -> dict[str, tuple[Sample, ...]]:
    """函数作用：负责当前模块中的 `domain_index` 处理，封装调用方需要复用的业务步骤。
    输入要求：`samples`（Sequence[Sample]）需符合函数签名约定。
    输出：返回 `dict[str, tuple[Sample, ...]]` 类型结果；校验或下游调用失败时异常向上传递。"""
    groups: dict[str, list[Sample]] = defaultdict(list)
    seen: set[str] = set()
    content_owners: dict[str, tuple[str, str]] = {}
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample id: {sample.sample_id}")
        seen.add(sample.sample_id)
        domain = effective_domain(sample)
        fingerprint = content_fingerprint(sample.dataset, sample.text)
        if fingerprint in content_owners:
            owner_id, owner_domain = content_owners[fingerprint]
            raise ValueError(
                "duplicate content must belong to exactly one sample and domain: "
                f"{owner_id!r}/{owner_domain!r} and {sample.sample_id!r}/{domain!r}"
            )
        content_owners[fingerprint] = (sample.sample_id, domain)
        groups[domain].append(sample)
    return {
        name: tuple(sorted(rows, key=lambda row: row.sample_id))
        for name, rows in sorted(groups.items())
    }


def data_fingerprint(samples: Sequence[Sample]) -> str:
    """函数作用：负责当前模块中的 `data_fingerprint` 处理，封装调用方需要复用的业务步骤。
    输入要求：`samples`（Sequence[Sample]）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    payload = [asdict(s) for s in sorted(samples, key=lambda s: s.sample_id)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def skillbank_fingerprint(skills: Sequence[SkillSpec]) -> str:
    """函数作用：负责当前模块中的 `skillbank_fingerprint` 处理，封装调用方需要复用的业务步骤。
    输入要求：`skills`（Sequence[SkillSpec]）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    payload = [asdict(s) for s in sorted(skills, key=lambda s: s.skill_id)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def split_source_and_final(
    samples: Sequence[Sample],
    final_test_domains: Sequence[str],
    train_domains: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """函数作用：拆分 `split_source_and_final` 所表示的数据，供当前模块后续流程使用。
    输入要求：`samples`（Sequence[Sample]）需符合函数签名约定；`final_test_domains`（Sequence[str]）需符合函数签名约定。
    输出：返回 `tuple[tuple[str, ...], tuple[str, ...]]` 类型结果；校验或下游调用失败时异常向上传递。"""
    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(
            f"cross-domain runs require exactly one dataset; received {sorted(datasets)}"
        )
    known = set(domain_index(samples))
    final = tuple(sorted(set(final_test_domains)))
    source = (
        tuple(sorted(known - set(final)))
        if train_domains is None
        else tuple(sorted(set(train_domains)))
    )
    missing = (set(source) | set(final)) - known
    if missing:
        raise ValueError(f"unknown configured domains: {sorted(missing)}")
    if set(source) & set(final):
        raise ValueError("source and final test domains overlap")
    unassigned = known - set(source) - set(final)
    if train_domains is not None and unassigned:
        raise ValueError(f"dataset contains unassigned domains: {sorted(unassigned)}")
    if not source or not final:
        raise ValueError("train and final test domains must both be non-empty")
    return source, final
