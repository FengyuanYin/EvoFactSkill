from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, TextIO

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean(value: str) -> str:
    return _CONTROL.sub(" ", str(value)).strip()


@dataclass(frozen=True)
class ProgressEvent:
    task: str
    phase: str
    completed: int
    total: int
    unit: str = "items"
    detail: str = ""

    def __post_init__(self) -> None:
        if self.completed < 0 or self.total < 0 or self.completed > self.total:
            raise ValueError("invalid progress counts")


class ProgressSink(Protocol):
    def update(self, event: ProgressEvent) -> None: ...

    def close(self, *, success: bool) -> None: ...


class NullProgressSink:
    def update(self, event: ProgressEvent) -> None:
        del event

    def close(self, *, success: bool) -> None:
        del success


class TerminalProgressSink:
    def __init__(self, stream: TextIO, *, dynamic: bool | None = None, width: int = 24):
        self.stream = stream
        self.dynamic = bool(stream.isatty()) if dynamic is None else dynamic
        self.width = max(10, width)
        self.started = monotonic()
        self.last_event: ProgressEvent | None = None
        self.closed = False

    def update(self, event: ProgressEvent) -> None:
        if self.closed:
            return
        self.last_event = event
        elapsed = max(0.0, monotonic() - self.started)
        ratio = event.completed / event.total if event.total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = event.completed / elapsed if elapsed > 0 else 0.0
        detail = f" {_clean(event.detail)}" if event.detail else ""
        line = (
            f"[{bar}] {event.completed}/{event.total} {_clean(event.unit)} "
            f"{_clean(event.task)}:{_clean(event.phase)}{detail} "
            f"{elapsed:.1f}s {rate:.2f}/s"
        )
        if self.dynamic:
            self.stream.write("\r" + line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self, *, success: bool) -> None:
        if self.closed:
            return
        if self.dynamic and self.last_event is not None:
            self.stream.write("\n")
        if not success:
            task = _clean(self.last_event.task) if self.last_event else "task"
            self.stream.write(f"{task}: failed or cancelled\n")
        self.stream.flush()
        self.closed = True


def make_progress_sink(mode: str = "auto", *, stream: TextIO | None = None) -> ProgressSink:
    if mode not in {"auto", "on", "off"}:
        raise ValueError("progress mode must be auto, on, or off")
    target = sys.stderr if stream is None else stream
    if mode == "off" or (mode == "auto" and not target.isatty()):
        return NullProgressSink()
    return TerminalProgressSink(target)
