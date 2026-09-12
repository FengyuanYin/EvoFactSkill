# Day 5：演示入口、连接测试与问题定位

## 学习目标

- 理解 `router.py` 的直接执行入口。
- 区分 Mock、live 和非法输出演示。
- 掌握 WindowsPath 错误的根因。
- 根据 fallback 定位真实连接问题。

## 一、入口生命周期

直接执行 `src/evofact/routing/router.py` 时，文件底部的 `if __name__ == "__main__"` 调用 `_main`。它解析 `--live`、`--config`、`--skip-invalid-test`，再通过 `asyncio.run` 启动异步演示。

作为模块 import 时不会执行这些样本。

## 二、WindowsPath 修复

错误写法 `Path(__file__).resolve().parent[3]` 把单个 Path 当成序列，因此抛出 `WindowsPath object is not subscriptable`。

正确写法是 `.parents[3]`：

```text
parents[0] = routing
parents[1] = evofact
parents[2] = src
parents[3] = 项目根目录
```

## 三、三种演示的含义

- 默认 Mock：检查接口、预算和输出结构，不评价提示词。
- `--live`：读取配置和 API Key，真正调用模型。
- `_invalid_output_demo`：故意返回 `invented-skill-id`，验证白名单拒绝和规则回退。

真实调用时必须看到 `fallback_used=False` 才能证明 LLM 路由成功。selected 看起来合理但 fallback 为 True，通常表示 API、JSON 或校验失败，结果来自规则 Router。

## 四、当前旧演示的成本陷阱

`_router_demo` 的 live 路径先为三个样本请求并打印，又重新请求一遍以统计 fallback，因此可能产生 6 次 Router API 调用。Day 6 的独立回归脚本每个案例只请求一次，更适合反复修改提示词。

## 五、排错表

| 现象 | 检查点 |
|---|---|
| `ModuleNotFoundError: evofact` | 当前解释器是否安装项目或加入 `src` |
| missing API key | 当前 PowerShell 是否设置 `DEEPSEEK_API_KEY` |
| fallback 为 True | HTTP、JSON、ID 白名单、预算和 confidence |
| 修改提示词后 Mock 不变 | 正常；Mock 不读取提示词语义 |
| 时间新闻选不到时间 Skill | 名称应为 `temporal_reasoning` |

## 六、自测题与答案

1. `.parent[3]` 为什么错误？**`.parent` 是单个 Path，应使用祖先序列 `.parents`。**
2. Mock 能评价提示词吗？**不能。**
3. 非法输出测试保护什么？**禁止模型选择候选白名单外的 ID。**
4. 当前旧 live 演示请求几次？**三个样本两轮，共 6 次。**

## 七、建议阅读顺序

`_main` → `_router_demo` → backend 分支 → 输出构造 → live fallback 统计 → `_invalid_output_demo`。
