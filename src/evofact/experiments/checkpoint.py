import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path


class Checkpoint:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.completed = set()

    def load(self):
        if self.path.exists():
            self.completed = {
                json.loads(x)["sample_id"]
                for x in self.path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            }
        return self.completed

    def append(self, sample_id: str, payload: dict):
        if sample_id in self.completed:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"sample_id": sample_id, **payload}, ensure_ascii=False, default=str)
                + "\n"
            )
        self.completed.add(sample_id)
        return True


class MetaCheckpointStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(
        self, *, config_fingerprint: str, data_fingerprint: str, skillbank_snapshot_id: str
    ) -> dict:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        expected = {
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "skillbank_snapshot_id": skillbank_snapshot_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint {key} mismatch")
        return payload

    def save(self, payload) -> None:
        value = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".meta-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda item: item.value if hasattr(item, "value") else str(item),
                )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
