from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataIsolationPolicy:
    merge_common_sources: bool = False
    controlled_sources: tuple[str, ...] = ()
    meta_train_fraction: float = 0.7
    meta_test_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0 < self.meta_train_fraction < 1:
            raise ValueError("meta_train_fraction must be in (0, 1)")
        if not 0 < self.meta_test_fraction < 1:
            raise ValueError("meta_test_fraction must be in (0, 1)")
        if self.meta_train_fraction + self.meta_test_fraction >= 1:
            raise ValueError("real-only split must reserve final-test data")
