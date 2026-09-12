from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .inference import InferenceRuntime


__all__ = ["InferenceRuntime"]


def __getattr__(name: str) -> Any:
    """按需导入运行时对象，避免 runtime 与 routing 之间循环导入。"""
    if name == "InferenceRuntime":
        from .inference import InferenceRuntime

        return InferenceRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
