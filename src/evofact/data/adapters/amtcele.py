import csv, json
from pathlib import Path
from typing import Iterable
from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic

class AMTCeleAdapter:
    name = "amtcele"
    def discover(self, root: Path) -> DatasetDiagnostic:
        files = tuple(str(p) for p in sorted(root.glob("*")) if p.suffix.casefold() in {".csv", ".jsonl"})
        return DatasetDiagnostic(self.name, bool(files), files, "ready" if files else f"no AMTCele files under {root}")
    def load(self, root: Path) -> Iterable[Sample]:
        diag = self.discover(root)
        if not diag.available: raise FileNotFoundError(diag.message)
        path = next(iter(sorted(root.glob("*.jsonl"))), None)
        if path: rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            path = next(iter(sorted(root.glob("*.csv"))))
            with path.open("r", encoding="utf-8-sig", newline="") as stream: rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            text = str(row.get("text") or row.get("content") or "").strip()
            if text: yield Sample(str(row.get("id") or f"amtcele:{i}"), self.name, text, row.get("label"), str(row.get("domain") or "").rstrip("0123456789") or None, metadata={"source_file": path.name})
