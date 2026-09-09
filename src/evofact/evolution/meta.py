import json
from pathlib import Path


class MetaEvolutionLoop:
    def __init__(self, enabled: bool = False, event_store: Path | None = None):
        self.enabled = enabled
        self.events = []
        self.event_store = event_store

    def run(self, evolution_events: list[dict]) -> dict:
        if not self.enabled:
            return {"enabled": False, "proposals": []}
        record = {
            "schema_version": "meta_event_v1",
            "enabled": True,
            "source_events": len(evolution_events),
            "proposals": [],
        }
        self.events.append(record)
        if self.event_store:
            self.event_store.parent.mkdir(parents=True, exist_ok=True)
            with self.event_store.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        return record

    async def run_demse(self, runner, samples, **kwargs):
        if not self.enabled:
            raise ValueError("DEMSE meta evolution is disabled")
        outcome = await runner.run(samples, **kwargs)
        self.events.append(
            {
                "schema_version": "meta_event_v2",
                "enabled": True,
                "run_id": outcome.run_id,
                "episodes": len(outcome.episodes),
                "decisions": len(outcome.decisions),
            }
        )
        return outcome
