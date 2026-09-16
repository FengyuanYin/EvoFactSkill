from __future__ import annotations

import asyncio

import pytest

from evofact.runtime.sandbox import SandboxLimits, UnavailableSandbox
from evofact.runtime.template_renderer import TemplateRenderer


def test_template_renderer_rejects_attribute_access() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        TemplateRenderer().render("{sample.__class__}", sample={})


def test_unavailable_sandbox_never_executes_host_code(tmp_path) -> None:
    side_effect = tmp_path / "side-effect"
    script = f"open({str(side_effect)!r}, 'w').write('bad')".encode()
    result = asyncio.run(UnavailableSandbox().execute(script, {}, SandboxLimits()))
    assert not result.succeeded
    assert not result.available
    assert not side_effect.exists()
