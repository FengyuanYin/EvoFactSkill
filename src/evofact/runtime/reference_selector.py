from __future__ import annotations

from dataclasses import dataclass

from evofact.core.models import SkillSpec


@dataclass(frozen=True)
class SelectedReference:
    path: str
    content: str


class ReferenceSelector:
    def __init__(self, *, max_files: int = 4, max_chars: int = 12000):
        self.max_files = max_files
        self.max_chars = max_chars

    def select(self, skill: SkillSpec, sample: dict) -> tuple[SelectedReference, ...]:
        text = str(sample.get("text", "")).casefold()
        scored = []
        for path, content in skill.resources.items():
            if not path.startswith("references/"):
                continue
            tokens = {
                token
                for token in path.replace("/", " ").replace("_", " ").casefold().split()
                if len(token) > 2
            }
            score = sum(token in text for token in tokens)
            scored.append((-score, path, content))
        chosen = []
        used = 0
        for _, path, content in sorted(scored):
            if len(chosen) >= self.max_files:
                break
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            value = content[:remaining]
            chosen.append(SelectedReference(path, value))
            used += len(value)
        return tuple(chosen)
