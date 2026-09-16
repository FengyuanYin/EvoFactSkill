from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SandboxLimits:
    timeout_ms: int = 5000
    memory_mb: int = 128
    cpu_seconds: int = 2
    max_output_bytes: int = 65536
    network: bool = False


@dataclass(frozen=True)
class SandboxResult:
    available: bool
    succeeded: bool
    output: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class SandboxAdapter(Protocol):
    reliable: bool

    async def execute(
        self, script: bytes, payload: dict[str, Any], limits: SandboxLimits
    ) -> SandboxResult: ...


class UnavailableSandbox:
    reliable = False

    async def execute(
        self, script: bytes, payload: dict[str, Any], limits: SandboxLimits
    ) -> SandboxResult:
        del script, payload, limits
        return SandboxResult(False, False, error="reliable sandbox is not configured")
