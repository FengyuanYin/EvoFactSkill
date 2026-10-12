from evofact.runtime.backend import BackendResult
from evofact.runtime.mock_backend import MockBackend


class EditingOptimizerBackend(MockBackend):
    """Deterministic optimizer response for exercising candidate evaluation offline."""

    async def optimize_package(self, context, optimizer_skill):
        target = next(
            package
            for package in context["skills"]
            if package["manifest"]["name"] == "source_credibility"
        )
        instructions = next(item for item in target["files"] if item["path"] == "SKILL.md")
        return BackendResult(
            {
                "action": "edit",
                "target_skill_id": target["skill_id"],
                "rationale": "Exercise trace-driven optimization.",
                "file_operations": [
                    {
                        "operation": "update",
                        "path": "SKILL.md",
                        "content": "Use supplied evidence and inspect its provenance carefully.\n",
                        "expected_digest": instructions["digest"],
                        "media_type": "text/markdown",
                    }
                ],
            }
        )
