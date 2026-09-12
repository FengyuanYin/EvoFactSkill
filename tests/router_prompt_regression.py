"""Live regression test for the LLM router prompt.

This is intentionally a standalone script instead of a normal pytest test: it
uses a paid/networked LLM endpoint and must only run when ``--live`` is given.
Edit ROUTER_CASES when you want to add project-specific routing expectations.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.config import load_config
from evofact.core.models import RunBudget, SkillKind
from evofact.routing.router import LLMSkillRouter, SkillRouter
from evofact.runtime.openai_backend import OpenAICompatibleBackend
from evofact.skills.loader import load_skill_package


@dataclass(frozen=True)
class RouterCase:
    case_id: str
    domain: str
    text: str
    expected_names: tuple[str, ...]


# These cases test selection from the existing specialist-skill bank. They do
# not ask the router to create a new skill. Each budget equals the number of
# expected skills, so an exact, minimal selection is required.
ROUTER_CASES = (
    RouterCase(
        "numerical_mismatch",
        "finance",
        (
            "A report says revenue rose from 100 million yuan to 130 million "
            "yuan, while the headline describes the increase as 50 percent."
        ),
        ("numerical_consistency",),
    ),
    RouterCase(
        "future_evidence",
        "politics",
        (
            "A claim published on January 1 cites a correction that was not "
            "released until January 5 as if it were already available."
        ),
        ("temporal_reasoning",),
    ),
    RouterCase(
        "conflicting_sources",
        "science",
        (
            "Two independent laboratories report opposite outcomes for the "
            "same experiment, population, measurement and time period."
        ),
        ("cross_source_contradiction",),
    ),
    RouterCase(
        "anonymous_source",
        "health",
        (
            "An anonymous social-media account claims that a drink cures a "
            "disease, but gives no author, institution or primary source."
        ),
        ("source_credibility",),
    ),
    RouterCase(
        "sensational_framing",
        "society",
        (
            "SHOCKING truth THEY do not want you to know!!! The post removes "
            "the speaker's qualifiers and replaces them with an absolute claim."
        ),
        ("linguistic_manipulation",),
    ),
    RouterCase(
        "number_and_timeline",
        "finance",
        (
            "On March 1 a post claimed a 40 percent annual increase by citing "
            "figures published on March 10; the figures rise only from 100 to 120."
        ),
        ("numerical_consistency", "temporal_reasoning"),
    ),
)


def _load_seed_skills():
    seed_root = ROOT / "skills" / "seeds"
    return [
        load_skill_package(path)
        for path in sorted(seed_root.iterdir())
        if (path / "SKILL.md").is_file()
    ]


async def run_live(config_path: Path, case_ids: set[str]) -> int:
    config = load_config(config_path)
    skills = _load_seed_skills()
    by_id = {skill.skill_id: skill.name for skill in skills}
    known_names = set(by_id.values())

    selected_cases = [
        case for case in ROUTER_CASES if not case_ids or case.case_id in case_ids
    ]
    unknown_cases = case_ids - {case.case_id for case in ROUTER_CASES}
    if unknown_cases:
        print(f"Unknown case IDs: {', '.join(sorted(unknown_cases))}")
        return 2
    if not selected_cases:
        print("No router cases selected.")
        return 2

    expected_names = {
        name for case in selected_cases for name in case.expected_names
    }
    missing_skills = expected_names - known_names
    if missing_skills:
        print(f"Expected skills are not installed: {', '.join(sorted(missing_skills))}")
        return 2

    router_skills = [skill for skill in skills if skill.kind == SkillKind.ROUTER]
    if len(router_skills) != 1:
        print(f"Expected exactly one router skill, found {len(router_skills)}.")
        return 2

    prompt_path = ROOT / "skills" / "seeds" / "coordinator_routing" / "SKILL.md"
    prompt_bytes = prompt_path.read_bytes()
    print(f"Prompt: {prompt_path}")
    print(f"Prompt SHA256: {hashlib.sha256(prompt_bytes).hexdigest()[:16]}")
    print(f"Model: {config.model}")

    backend = OpenAICompatibleBackend(
        config.base_url,
        config.resolved_api_key(),
        config.model,
    )
    router = LLMSkillRouter(
        backend,
        fallback=SkillRouter("utility-aware", seed=config.seed),
    )

    passed = 0
    for case in selected_cases:
        budget = RunBudget(
            max_skills=len(case.expected_names),
            max_calls=1,
            max_tokens=2000,
        )
        result = await router.route(
            {
                "sample_id": case.case_id,
                "dataset": "router_prompt_regression",
                "domain": case.domain,
                "text": case.text,
                "metadata": {},
            },
            skills,
            {},
            budget,
        )
        decision = result.value
        actual_names = tuple(
            by_id.get(skill_id, f"<unknown:{skill_id}>")
            for skill_id in decision.selected_skill_ids
        )
        expected_set = set(case.expected_names)
        actual_set = set(actual_names)
        case_passed = (
            not decision.fallback_used
            and actual_set == expected_set
            and len(actual_names) == len(case.expected_names)
        )
        passed += int(case_passed)

        print(f"\n[{'PASS' if case_passed else 'FAIL'}] {case.case_id}")
        print(f"  expected: {', '.join(case.expected_names)}")
        print(f"  actual:   {', '.join(actual_names) if actual_names else '<none>'}")
        print(f"  fallback: {decision.fallback_used}")
        print(f"  confidence: {decision.confidence:.3f}")
        for skill_id in decision.selected_skill_ids:
            print(f"  reason[{by_id.get(skill_id, skill_id)}]: {decision.reasons.get(skill_id, '')}")

    total = len(selected_cases)
    print(f"\nResult: {passed}/{total} passed ({passed / total:.0%})")
    if passed != total:
        print("Edit coordinator_routing/SKILL.md, then run this command again.")
        return 1
    return 0


def _print_cases() -> None:
    print("Available router prompt regression cases:")
    for case in ROUTER_CASES:
        print(f"  {case.case_id}: {', '.join(case.expected_names)}")
    print("\nNo API request was made. Add --live to execute the regression test.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the live LLM router prompt.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow calls to the configured LLM API",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "adversarial_llm.yaml",
        help="OpenAI-compatible backend configuration",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run one case ID; may be specified more than once",
    )
    args = parser.parse_args()

    if not args.live:
        _print_cases()
        return 0
    return asyncio.run(run_live(args.config.resolve(), set(args.case)))


if __name__ == "__main__":
    raise SystemExit(main())
