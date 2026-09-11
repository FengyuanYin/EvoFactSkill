from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence

from evofact.config import MetaLearningConfig
from evofact.core.models import DomainEpisode, Sample

from .domains import data_fingerprint, domain_index


class DomainEpisodeSampler:
    def __init__(self, config: MetaLearningConfig, seed: int = 42):
        """函数作用：创建并初始化 `DomainEpisodeSampler` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `DomainEpisodeSampler` 实例；`config`（MetaLearningConfig）需符合函数签名约定；`seed`（int，默认 `42`）需符合函数签名约定。
        输出：返回 `None`；初始化 `DomainEpisodeSampler` 的实例状态，构造参数非法时可能抛出异常。"""
        self.config = config
        self.seed = seed

    def build(
        self,
        samples: Sequence[Sample],
        source_domains: Sequence[str],
        final_test_domains: Sequence[str],
        skillbank_snapshot_id: str,
    ) -> tuple[DomainEpisode, ...]:
        """函数作用：负责`DomainEpisodeSampler` 中的 `build` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `DomainEpisodeSampler` 实例；`samples`（Sequence[Sample]）需符合函数签名约定；`source_domains`（Sequence[str]）需符合函数签名约定；`final_test_domains`（Sequence[str]）需符合函数签名约定；`skillbank_snapshot_id`（str）需符合函数签名约定。
        输出：返回 `tuple[DomainEpisode, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
        groups = domain_index(samples)
        source = tuple(sorted(set(source_domains)))
        final = tuple(sorted(set(final_test_domains)))
        if len(source) < self.config.min_source_domains:
            raise ValueError(
                f"DEMSE requires at least {self.config.min_source_domains} source domains"
            )
        if (set(source) & set(final)) or not (set(source) | set(final)) <= set(groups):
            raise ValueError("invalid source/final domain partition")
        if self.config.meta_test_domain_count >= len(source):
            raise ValueError("meta-test domain count must be smaller than source domain count")
        fingerprint = data_fingerprint(samples)
        test_sets = self._test_sets(source)
        episodes = []
        for index, test_domains in enumerate(test_sets):
            train_domains = tuple(domain for domain in source if domain not in test_domains)
            train_ids = tuple(s.sample_id for domain in train_domains for s in groups[domain])
            test_ids = tuple(s.sample_id for domain in test_domains for s in groups[domain])
            payload = {
                "seed": self.seed,
                "index": index,
                "strategy": self.config.strategy,
                "train": train_domains,
                "test": test_domains,
                "data": fingerprint,
                "bank": skillbank_snapshot_id,
            }
            episode_id = (
                "episode-"
                + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            )
            episodes.append(
                DomainEpisode(
                    episode_id,
                    self.seed,
                    self.config.strategy,
                    train_domains,
                    test_domains,
                    train_ids,
                    test_ids,
                    final,
                    fingerprint,
                    skillbank_snapshot_id,
                )
            )
        covered = set().union(*(set(e.meta_test_domains) for e in episodes))
        if covered != set(source):
            raise RuntimeError("episode plan does not cover every source domain as meta-test")
        return tuple(episodes)

    def _test_sets(self, source: tuple[str, ...]) -> list[tuple[str, ...]]:
        """函数作用：负责`DomainEpisodeSampler` 中的 `_test_sets` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `DomainEpisodeSampler` 实例；`source`（tuple[str, ...]）需符合函数签名约定。
        输出：返回 `list[tuple[str, ...]]` 类型结果；校验或下游调用失败时异常向上传递。"""
        if self.config.strategy == "leave_one_domain_out":
            return [(domain,) for domain in source]
        shuffled = list(source)
        random.Random(self.seed).shuffle(shuffled)
        width = self.config.meta_test_domain_count
        required = math.ceil(len(source) / width)
        count = max(self.config.episodes, required)
        return [
            tuple(
                sorted(
                    shuffled[(index * width + offset) % len(shuffled)] for offset in range(width)
                )
            )
            for index in range(count)
        ]
