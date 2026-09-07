import json, re
from pathlib import Path
from typing import Iterable
from evofact.core.models import Evidence, Sample
from evofact.data.base import DatasetDiagnostic
PATTERN = re.compile(r"livefact_(?P<window>[+-]?\d+)_(?P<task>cls|inf)\.jsonl$")

class LiveFactAdapter:
    name = "livefact"
    def discover(self, root: Path) -> DatasetDiagnostic:
        files = tuple(str(p) for p in sorted(root.glob("**/livefact_*_*.jsonl")))
        return DatasetDiagnostic(self.name, bool(files), files, "ready" if files else f"no LiveFact files under {root}")
    def load(self, root: Path) -> Iterable[Sample]:
        diag = self.discover(root)
        if not diag.available: raise FileNotFoundError(diag.message)
        for path_text in diag.files:
            path = Path(path_text); match = PATTERN.match(path.name)
            if not match: continue
            month, window, task = path.parent.name, match.group("window"), match.group("task")
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip(): continue
                row = json.loads(line); text = str(row.get("claim") or row.get("text") or row.get("content") or "").strip()
                evidence = row.get("evidence") or []; evidence = [evidence] if isinstance(evidence, str) else evidence
                ev = tuple(Evidence(str(x.get("text", "")), x.get("source")) if isinstance(x, dict) else Evidence(str(x)) for x in evidence)
                role="test" if window.startswith("+3") else ("protected" if window in {"0","+0"} else "train")
                if text: yield Sample(str(row.get("id") or f"livefact:{month}:{window}:{task}:{i}"), self.name, text, row.get("label"), row.get("domain"), row.get("event_id"), evidence=ev, metadata={"month": month, "temporal_window": window, "task_type": task, "source_file": path.name, "split_role":role})
