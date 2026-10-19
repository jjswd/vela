# Vela

[![CI](https://github.com/jjswd/vela/actions/workflows/ci.yml/badge.svg)](https://github.com/jjswd/vela/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-179%20passing-brightgreen)](./tests)
[![Code size](https://img.shields.io/badge/source-3.6k%20lines-lightgrey)](./src)

一个小型 Agent 运行时：**编排、执行、能力三层分离**，并对不可信的模型输出提供有界恢复路径。

写给需要自己搭 Agent 运行时的开发者：把「协议适配」「异常兜底」「上下文压缩」
「成本控制」这四件容易做错的事，做成可复用、可验证、边界清晰的组件。

> **一分钟验证**
>
> ```bash
> pip install -e ".[dev]"
> python examples/verify.py     # 46 项检查，无需 API key
> pytest -q                     # 179 个测试
> ```
>
> `verify.py` 逐项跑通工具循环、三类恢复路径、增量流式、跨协议转换、会话隔离、
> 压缩、缓存成本模型与工具防护，并打印每项结果。失败时返回非零退出码。

```
AgentLoop      面向渠道：会话、压缩、生命周期、并发隔离
AgentRunner    面向模型：ReAct 循环与全部恢复路径
Provider/Tool  面向后端：模型协议与工具能力
```

依赖只能向下。`core` 不认识 transport，`providers` 不认识会话。
这条约束让「换模型」不影响路由，「换接入方式」不影响模型循环。

## 快速开始

```bash
pip install -e ".[dev]"

# 无需 API key：脚本化 provider 跑通工具循环与全部恢复路径
python examples/demo.py

# 接真实模型（后端通过配置选择，代码零改动）
export VELA_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export VELA_MODEL=claude-sonnet-4-5
python examples/demo.py

# 也可以切到 OpenAI 兼容系后端
export VELA_BACKEND=deepseek DEEPSEEK_API_KEY=sk-... VELA_MODEL=deepseek-chat

python examples/verify.py   # 46 checks, no API key needed
pytest -q                     # 179 passed
```

最小用法：

```python
import asyncio
from vela import build_default_loop, StreamingEmitter

async def main():
    loop = build_default_loop(
        backend="anthropic",          # 或 openai / deepseek / moonshot / openrouter / ollama
        api_key="sk-...",
        model="claude-sonnet-4-5",
        workspace="./workspace",
    )
    try:
        # 同一 session 的多次调用共享上下文
        print((await loop.run("chat", "记住我的生日是 3 月 5 日")).content)
        print((await loop.run("chat", "我的生日是什么时候？")).content)

        # 流式
        await loop.run(
            "chat",
            "写一首关于代码的短诗",
            emitter=StreamingEmitter(on_text=lambda d: print(d, end="", flush=True)),
        )
    finally:
        await loop.aclose()

asyncio.run(main())
```

## 自定义工具

工具的 schema 是**模型契约**：模型靠它决定调用什么。改名或改描述都会改变模型行为。

```python
from vela import Tool, ToolRegistry

class WeatherTool(Tool):
    name = "get_weather"
    description = "查询指定城市的当前天气。"
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "minLength": 1}},
        "required": ["city"],
    }

    async def run(self, arguments: dict) -> str:
        return f"{arguments['city']}: 22°C 晴"

loop = build_default_loop(..., extra_tools=[WeatherTool()])
```

## Prompt caching

```python
from vela import build_default_loop, CachePolicy, CacheTTL

loop = build_default_loop(
    backend="anthropic",
    api_key="...",
    model="claude-sonnet-4-5",
    cache_policy=CachePolicy(enabled=True, ttl=CacheTTL.ONE_HOUR),
    expected_turns=20,          # 会话预计轮数，用于判断是否值得开
)
print(loop.provider.cache_stats.summary())
# {'turns': 10, 'cache_read_tokens': 81000, 'hit_rate': 0.81, 'cost_ratio': 0.352}
```

实测（模拟稳定前缀 9000 token + 变动尾部 1000 token，共 10 轮）：
**输入 token 成本降低 64.8%，缓存命中率 81%**。

### 缓存不是免费的

后端对**写入**缓存收费高于正常输入（5 分钟 1.25×、1 小时 2.0×），只有**读取**才打折（0.1×）。
所以开启条件是「前缀会被复用足够多次」：

| TTL | 写入价 | 盈亏平衡轮次 |
| --- | --- | --- |
| `5m` | 1.25× | 2 轮 |
| `1h` | 2.0× | 3 轮 |

低于这个次数，开缓存会让**每次请求都更贵**。因此默认关闭，且由
`CachePolicy.should_cache(expected_turns)` 依据预计会话长度决定是否打标记。

### 断点只打在稳定前缀上

在「下一轮就会变化」的内容上打缓存标记，会让它前面的所有前缀一起失效——
等于每次都白写一次缓存。所以标记位置刻意避开最近两轮：

| 位置 | 是否打标记 | 原因 |
| --- | --- | --- |
| 工具定义 | ✅ | 注册表固定，除非重新配置 |
| system 提示 | ✅ | 部署期间不变 |
| 历史消息（距末尾 ≥3 条） | ✅ | 已稳定 |
| 最近两轮 | ❌ | 每次请求都在变 |

另受限于 API 最多 4 个断点，因此标记集中规划而非随手添加。

### 成本可核对，不只靠信任

`CacheStats` 记录真实的命中与写入并算出相对成本，`cost_ratio > 1.0`
就说明写入溢价尚未回本——策略是否有效用事实检验，而不是相信配置。

## 多后端

后端差异全部吸收在 `providers/` 内，`core/` 不含任何按后端分支的逻辑。

```python
from vela import available_providers, create_provider

available_providers()
# ['anthropic', 'deepseek', 'moonshot', 'ollama', 'openai', 'openrouter']

provider = create_provider("anthropic", model="claude-sonnet-4-5", api_key="...")
```

已注册后端中除 Anthropic 外均使用 OpenAI 兼容协议（同一实现覆盖 DeepSeek、
Moonshot、OpenRouter 及 Ollama 等本地服务）。

### 两种协议的三处结构性差异

|维度 | OpenAI 兼容 | Anthropic |
| --- | --- | --- |
| system 提示 | `messages` 里的 `role="system"` | 顶层 `system` 字段 |
| 工具结果 | 每条结果一条 `role="tool"` 消息 | **合并**为紧随其后的一条 user 消息内的 `tool_result` block |
| 工具 schema | `parameters` | `input_schema` |
| 终止原因 | `stop` / `length` / `tool_calls` | `end_turn` / `max_tokens` / `tool_use` |

第三点是实现里最容易出错的地方：内部 transcript 每条工具结果独立成消息，
而 Anthropic 要求**同一个 assistant turn 的全部结果合并进一条 user 消息**，
且必须紧跟其请求方。顺序错或拆开都会被 API 以 400 拒绝，
`tests/test_anthropic.py` 对这些形态逐个做了断言。

不想写类也可以用装饰器，schema 从类型注解推导：

```python
from vela import tool

@tool(description="把华氏温度转成摄氏温度")
def fahrenheit_to_celsius(f: float) -> str:
    return f"{(f - 32) * 5 / 9:.1f}C"
```

## 设计要点

### 1. 恢复路径全部有界

模型输出不可信。以下每种异常都有明确的、有上限的恢复策略：

| 异常 | 处理 | 上限 |
| --- | --- | --- |
| 空回复 | 重试，超限后强制收尾 | 2 次 |
| 输出截断 | 收集分段并要求续写，最后拼接 | 3 段 |
| 工具名畸形 | 解析期直接丢弃 | 0 |
| 参数不合法 | 一次性回报全部错误 | — |
| 工具崩溃/超时 | 转为 error 结果，不中断本轮 | — |
| 模型错误 | 回报错误，历史写入占位符 | — |
| 迭代超限 | 无工具的收尾请求 + 本地兜底 | — |
| 运行中注入 | 迭代边界取走，单轮上限 | 3 条 / 5 轮 |

每个上限都存在，因为**针对不可信输入的无界恢复只是一次烧 token 的方式**。

### 2. 迭代超限不是硬停

预算耗尽时先发一次**不带工具定义**的请求，让模型用已获得的信息给出总结。用移除工具定义而不是提示词来约束模型，模型在结构上就无法发起新工作。

### 3. 错误不是 assistant 说的话

模型出错时，历史里写入中性占位符而非错误文本。否则后续轮次会把错误当成助手先前的回答。

### 4. `None` 不等于 `0`

`cache_read_tokens=None` 表示「provider 没报这个指标」，`0` 表示「报的是零」。累加时区分二者，统计代码里**沉默的缺失比错误的数字更危险**。

### 5. 会话串行是正确性要求

每个 session 一个 worker。并发执行会让工具结果以与请求不一致的顺序进入历史。

### 6. 注入只在边界发生

运行中到达的消息在迭代边界取走并入队，不在请求进行中修改 message 数组。

### 7. **流式是逐片回调，不是等结果**
   `Provider.complete()` 接受 `on_delta` 回调，在每个片段解析到的当下推送，
   而不是等完整响应返回再一次性交付——后者在任何用户可见意义上都不是流式。
   对忽略该回调的后端保留了补发路径，避免旧后端接入后变成静默无输出。

8. 压缩失败不能丢历史

压缩是长对话的常规路径，summary 服务故障只能损失细节，不能损失上下文。

## 项目结构

```
src/vela/
├── core/
│   ├── contracts.py     跨层数据契约（无依赖）
│   ├── runner.py        执行层：ReAct 循环 + 恢复路径
│   └── loop.py          编排层：会话、持久化、压缩、并发
├── providers/
│   ├── base.py          Provider 抽象 + 错误分类 + JSON 修复
│   ├── factory.py       后端注册表与构造（按配置选择）
│   ├── openai_compat.py OpenAI 兼容实现（含 SSE 流式重组）
│   ├── anthropic.py     Anthropic 原生协议 + 消息格式转换
│   ├── caching.py       缓存成本模型（盈亏平衡、命中核算）
│   └── anthropic_cache.py  cache_control 断点规划
└── tools/
    ├── base.py          Tool 基类 + schema 校验 + contextvar
    ├── registry.py      注册表 + 并发分发 + 名称纠错
    └── builtin.py       内置工具

tests/                   179 个测试
examples/demo.py端到端演示，无需 API key
```

### 测试分布

```
tests/test_runner.py             恢复路径（16）
tests/test_loop.py会话、压缩、隔离（10）
tests/test_providers_tools.py    解析、错误分类、schema（29）
tests/test_anthropic.py          Anthropic 格式转换（30）
tests/test_backends.py           跨后端一致性 + HTTP（18）
tests/test_streaming.py          SSE 协议解析 + 分片传输（34）
tests/test_streaming_runtime.py  runner 侧流式消费（13）
tests/test_caching.py            缓存成本模型与断点放置（34）
```

`test_backends.py` 是解耦是否成立的证据：它断言同一份 transcript 经两种协议
转换后工具调用往返信息等价，并验证 `core` 层的工具循环与恢复路径对后端无感知。

### 关于流式测试的一个要点

`test_streaming.py` 里的 `DelayedByteStream` 会把响应体切成小块逐个 yield，
并且**故意让块边界与 SSE 帧边界错开**。

原因是：用 `MockTransport` 直接返回完整 body 时，`aiter_lines` 的缓冲逻辑
不会被触发，任何「帧被截断后丢弃内容」的 bug 都测不出来。同理，只断言
`response.content` 正确也不够——把整段文本一次性回调也满足该断言，
但用户看到的是瞬间刷出的全文，而非逐字效果。**因此这些测试同时断言
回调次数与片段内容。**

## 测试覆盖

```
tests/test_runner.py            恢复路径（16）
tests/test_loop.py              会话、压缩、隔离（10）
tests/test_providers_tools.py   解析、错误分类、schema（29）
```

测试重点是**恢复路径的契约**：为什么每个上限存在、被删除时会发生什么。这比测试正常流程更能防止回归。

## 文档

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — 分层、数据流、每条恢复路径的推理
- [`INTERVIEW.md`](./INTERVIEW.md) — 面试问答准备与可复述的数字

## 已知边界

- 单进程内存态，会话落盘为 JSON；未做跨进程锁
- 压缩阈值为字符/token 估算，未接精确 tokenizer
- OpenAI 兼容后端的缓存由服务端自动管理，未做请求路由优化
- 缓存断点为静态规划，未实现基于实际命中率的动态调整
- 同步接口，未做背压与限流

## License

MIT
