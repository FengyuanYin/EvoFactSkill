from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from evofact.core.models import Sample

@dataclass(frozen=True)
class DatasetDiagnostic:
    name: str
    available: bool
    files: tuple[str, ...] = ()
    message: str = ""

class DatasetAdapter(Protocol):
    name: str
    def discover(self, root: Path) -> DatasetDiagnostic: ...
    def load(self, root: Path) -> Iterable[Sample]: ...
