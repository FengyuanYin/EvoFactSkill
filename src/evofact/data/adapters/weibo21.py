import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic


class Weibo21Adapter:
    name = "weibo21"

    def discover(self, root: Path) -> DatasetDiagnostic:
        files = tuple(
            str(p)
            for p in sorted(root.glob("*"))
            if p.suffix.casefold() in {".jsonl", ".json", ".pkl"}
        )
        return DatasetDiagnostic(
            self.name, bool(files), files, "ready" if files else f"no Weibo21 files under {root}"
        )

    def _rows(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.casefold() == ".jsonl":
            return [
                json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
        if path.suffix.casefold() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else [value]
        with path.open("rb") as stream:
            value = pickle.load(stream)
        return value.to_dict(orient="records") if hasattr(value, "to_dict") else list(value)

    def load(self, root: Path) -> Iterable[Sample]:
        diag = self.discover(root)
        if not diag.available:
            raise FileNotFoundError(diag.message)
        paths = (
            sorted(root.glob("*.jsonl"))
            or sorted(root.glob("*.json"))
            or sorted(root.glob("*.pkl"))
        )
        for path in paths:
            split = path.stem.casefold()
            for i, row in enumerate(self._rows(path)):
                text = str(row.get("content") or row.get("text") or "").strip()
                if text:
                    yield Sample(
                        str(row.get("id") or f"weibo21:{split}:{i}"),
                        self.name,
                        text,
                        row.get("label"),
                        row.get("category"),
                        row.get("event_id"),
                        metadata={"official_split": split, "source_file": path.name},
                    )
