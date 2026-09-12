import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.core.models import RunBudget, SkillKind
from evofact.routing.router import LLMSkillRouter
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.skills.loader import load_skill_package


def _load_seed_skills():
    seed_root = ROOT / "skills" / "seeds"
    return [
        load_skill_package(path)
        for path in sorted(seed_root.iterdir())
        if (path / "SKILL.md").is_file()
    ]


class CapturingBackend(OpenAICompatibleBackend):
    def __init__(self):
        super().__init__("https://example.invalid/v1", "unused", "test-model")
        self.system = ""
        self.payload = {}

    async def _call(self, system: str, payload: dict) -> dict:
        self.system = system
        self.payload = payload
        return {
            "selected_skill_ids": [payload["candidates"][0]["skill_id"]],
            "reasons": {
                payload["candidates"][0]["skill_id"]: "test selection",
            },
            "confidence": 0.9,
        }


async def test_router_prompt_and_candidates_reach_backend():
    skills = _load_seed_skills()
    router_skill = next(skill for skill in skills if skill.kind == SkillKind.ROUTER)
    backend = CapturingBackend()
    router = LLMSkillRouter(backend)

    result = await router.route(
        {
            "sample_id": "prompt-forwarding",
            "dataset": "test",
            "domain": "finance",
            "text": "Revenue rose from 100 to 120.",
            "metadata": {},
        },
        skills,
        {},
        RunBudget(max_skills=1, max_calls=1, max_tokens=1000),
    )

    assert router_skill.instructions in backend.system
    assert backend.payload["sample"]["sample_id"] == "prompt-forwarding"
    assert {item["name"] for item in backend.payload["candidates"]} == {
        skill.name for skill in skills if skill.kind == SkillKind.SPECIALIST
    }
    assert result.value.fallback_used is False
    assert len(result.value.selected_skill_ids) == 1
