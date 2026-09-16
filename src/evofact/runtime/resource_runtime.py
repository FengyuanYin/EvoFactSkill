from __future__ import annotations

import hashlib
from dataclasses import dataclass

from evofact.core.dag_models import ResourceSelection
from evofact.core.models import SkillSpec

from .reference_selector import ReferenceSelector
from .sandbox import SandboxLimits, UnavailableSandbox
from .template_renderer import TemplateRenderer


@dataclass(frozen=True)
class RuntimeResources:
    payload: dict
    selection: ResourceSelection


class ResourceRuntime:
    def __init__(self, *, selector=None, renderer=None, sandbox=None):
        self.selector = selector or ReferenceSelector()
        self.renderer = renderer or TemplateRenderer()
        self.sandbox = sandbox or UnavailableSandbox()

    async def prepare(self, skill: SkillSpec, sample: dict, upstream: tuple) -> RuntimeResources:
        references = self.selector.select(skill, sample)
        entrypoints = getattr(skill, "entrypoints", None)
        template_path = getattr(entrypoints, "template", None)
        script_path = getattr(entrypoints, "script", None)
        template = skill.resources.get(template_path) if template_path else None
        reference_rows = [{"path": item.path, "content": item.content} for item in references]
        payload = {
            **sample,
            "_upstream": [item.model_dump() for item in upstream],
            "_references": reference_rows,
        }
        if template is not None:
            payload = {
                "rendered": self.renderer.render(
                    template,
                    sample=sample,
                    upstream=payload["_upstream"],
                    references=reference_rows,
                    scripts={},
                    summary={},
                )
            }
        digests = {
            item.path: hashlib.sha256(item.content.encode()).hexdigest() for item in references
        }
        if template_path and template is not None:
            digests[template_path] = hashlib.sha256(template.encode()).hexdigest()
        if script_path:
            script = skill.resources.get(script_path)
            if script is None:
                raise ValueError("declared script is not available in runtime resources")
            result = await self.sandbox.execute(script.encode(), payload, SandboxLimits())
            if not result.succeeded:
                raise RuntimeError(result.error or "sandbox execution failed")
            payload["script_output"] = result.output
            digests[script_path] = hashlib.sha256(script.encode()).hexdigest()
        return RuntimeResources(
            payload,
            ResourceSelection(
                tuple(item.path for item in references), template_path, script_path, digests
            ),
        )
