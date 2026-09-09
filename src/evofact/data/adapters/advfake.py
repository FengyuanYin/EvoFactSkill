import csv
from pathlib import Path
from typing import Iterable

from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic


class AdvFakeAdapter:
    name = "advfake"

    def discover(self, root: Path) -> DatasetDiagnostic:
        files = tuple(str(p) for p in sorted(root.glob("*.csv")))
        return DatasetDiagnostic(
            self.name, bool(files), files, "ready" if files else f"no AdvFake CSV under {root}"
        )

    def load(self, root: Path) -> Iterable[Sample]:
        diag = self.discover(root)
        if not diag.available:
            raise FileNotFoundError(diag.message)
        path = Path(diag.files[0])
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            text = str(
                row.get("adversarial") or row.get("adversarial_text") or row.get("text") or ""
            ).strip()
            pair = str(row.get("id") or f"advfake:{i}")
            if text:
                yield Sample(
                    f"{pair}:adversarial",
                    self.name,
                    text,
                    row.get("label"),
                    event_id=pair,
                    metadata={
                        "original_text": row.get("original") or row.get("original_text"),
                        "attack_type": row.get("attack_type"),
                        "split_role": "test",
                        "robustness_only": True,
                    },
                )
