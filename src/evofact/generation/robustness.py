from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedRobustnessSet:
    sample_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    generator_package_digests: tuple[str, ...]
    formal_metric: bool = False


def build_robustness_set(samples) -> GeneratedRobustnessSet:
    return GeneratedRobustnessSet(
        tuple(item.sample.sample_id for item in samples),
        tuple(item.lineage.lineage_id for item in samples),
        tuple(sorted({item.lineage.generator_package_digest for item in samples})),
        False,
    )
