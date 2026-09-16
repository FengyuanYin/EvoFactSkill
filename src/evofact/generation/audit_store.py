from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from evofact.core.generation_models import GenerationAuditEntry
from evofact.governance import GENERATION_AUDIT_SCHEMA_VERSION


class GenerationAuditStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, entries: tuple[GenerationAuditEntry, ...] | list[GenerationAuditEntry]) -> None:
        payload = {
            "schema_version": GENERATION_AUDIT_SCHEMA_VERSION,
            "entries": [asdict(item) for item in entries],
        }
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".audit-", suffix=".json"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload, stream, ensure_ascii=False, sort_keys=True, indent=2, default=str
                )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_raw(self) -> dict:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != GENERATION_AUDIT_SCHEMA_VERSION:
            raise ValueError("generation audit schema requires explicit migration")
        return payload
