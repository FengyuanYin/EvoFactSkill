from __future__ import annotations

from typing import Any

SENSITIVE_KEYS = {
    "label",
    "gold",
    "gold_label",
    "gold_native_label",
    "answer",
    "target",
    "expected",
    "expected_result",
    "ground_truth",
}


def redact_sensitive(value: Any) -> Any:
    """函数作用：递归移除 `redact_sensitive` 所表示的数据，供当前模块后续流程使用。
    输入要求：`value`（Any）需符合函数签名约定。
    输出：返回 `Any` 类型结果；校验或下游调用失败时异常向上传递。"""
    if isinstance(value, dict):
        return {
            key: redact_sensitive(item)
            for key, item in value.items()
            if str(key).casefold() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value
