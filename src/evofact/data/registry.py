from pathlib import Path

from .adapters.advfake import AdvFakeAdapter
from .adapters.amtcele import AMTCeleAdapter
from .adapters.livefact import LiveFactAdapter
from .adapters.weibo21 import Weibo21Adapter
from .base import DatasetAdapter, DatasetDiagnostic


class DataRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in (Weibo21Adapter(), AMTCeleAdapter(), LiveFactAdapter(), AdvFakeAdapter()):
            self.register(adapter)

    def register(self, adapter: DatasetAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"duplicate dataset adapter: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def inspect(self, roots: dict[str, Path]) -> list[DatasetDiagnostic]:
        return [
            self._adapters[n].discover(roots.get(n, Path("data/raw") / n)) for n in self.names()
        ]

    def load(self, name: str, root: Path):
        if name not in self._adapters:
            raise KeyError(name)
        return list(self._adapters[name].load(root))
