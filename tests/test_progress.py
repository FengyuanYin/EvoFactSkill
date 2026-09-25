from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper

from evofact.cli import _emit, main
from evofact.runtime.progress import (
    NullProgressSink,
    ProgressEvent,
    TerminalProgressSink,
    make_progress_sink,
)


class Stream(StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_auto_progress_is_disabled_for_non_tty() -> None:
    stream = Stream(False)
    sink = make_progress_sink("auto", stream=stream)
    assert isinstance(sink, NullProgressSink)
    sink.update(ProgressEvent("test", "inference", 1, 2, "samples"))
    sink.close(success=True)
    assert stream.getvalue() == ""


def test_explicit_progress_uses_plain_lines_for_non_tty() -> None:
    stream = Stream(False)
    sink = make_progress_sink("on", stream=stream)
    sink.update(ProgressEvent("test", "inference", 1, 2, "samples"))
    sink.close(success=True)
    output = stream.getvalue()
    assert "1/2 samples" in output
    assert "\x1b" not in output
    assert "\r" not in output


def test_tty_progress_is_dynamic_and_failure_is_visible() -> None:
    stream = Stream(True)
    sink = TerminalProgressSink(stream)
    sink.update(ProgressEvent("evolve", "batch", 0, 2, "batches", "unsafe\ntext"))
    sink.close(success=False)
    output = stream.getvalue()
    assert output.startswith("\r")
    assert "unsafe text" in output
    assert "failed or cancelled" in output


def test_progress_off_is_always_silent() -> None:
    stream = Stream(True)
    sink = make_progress_sink("off", stream=stream)
    sink.update(ProgressEvent("test", "inference", 1, 1))
    sink.close(success=False)
    assert stream.getvalue() == ""


def test_cli_progress_stays_on_stderr_and_stdout_is_json() -> None:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        main(
            [
                "--config",
                "configs/dry_run.yaml",
                "--batch-size",
                "2",
                "--sample-concurrency",
                "2",
                "--progress",
                "test",
            ]
        )
    payload = json.loads(stdout.getvalue())
    assert payload["mode"] == "test"
    assert "samples test:inference" in stderr.getvalue()


def test_cli_stdout_survives_a_gbk_redirect(monkeypatch) -> None:
    """Windows `> file` 会把 stdout 变成 GBK：CLI 必须转义而不是抛 UnicodeEncodeError 丢掉整轮结果。"""

    async def fake_run(args):
        return {"mode": "test", "traces": [{"text": "旗帜 🇯🇵 与 🇰🇷"}]}

    monkeypatch.setattr("evofact.cli._run", fake_run)
    raw = BytesIO()
    stream = TextIOWrapper(raw, encoding="gbk")
    with redirect_stdout(stream):
        main(["--config", "configs/dry_run.yaml", "test"])
    stream.flush()

    payload = json.loads(raw.getvalue().decode("gbk"))
    assert payload["traces"][0]["text"] == "旗帜 🇯🇵 与 🇰🇷"


def test_emit_escapes_characters_the_stream_encoding_cannot_represent() -> None:
    """流编码无法表示内容时（如 GBK 控制台写入国旗 emoji），退化为 ASCII 转义 JSON 而不是丢结果。"""
    raw = BytesIO()
    stream = TextIOWrapper(raw, encoding="gbk")
    value = {"sample_id": "gbk-1", "text": "旗帜 🇯🇵 与 🇰🇷 的对比"}

    _emit(value, stream=stream)

    stream.flush()
    written = raw.getvalue().decode("gbk")
    assert written.isascii()
    assert json.loads(written) == value


def test_cli_output_flag_writes_utf8_and_keeps_stdout_clean(tmp_path) -> None:
    """`--output` 让 CLI 自己以 UTF-8 写文件，绕开 shell 重定向的编码问题。"""
    stdout = StringIO()
    stderr = StringIO()
    target = tmp_path / "nested" / "result.json"
    with redirect_stdout(stdout), redirect_stderr(stderr):
        main(
            [
                "--config",
                "configs/dry_run.yaml",
                "--batch-size",
                "2",
                "test",
                "--output",
                str(target),
            ]
        )

    assert stdout.getvalue() == ""
    assert f"wrote {target}" in stderr.getvalue()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["mode"] == "test"
    assert set(payload["domain_metrics"]) == {"health", "science", "social"}
    assert sum(item["n"] for item in payload["domain_metrics"].values()) == payload["metrics"]["n"]
    assert payload["domain_metrics"]["social"]["n"] == 2
