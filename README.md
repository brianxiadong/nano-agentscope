# Nano-AgentScope

![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)

A simplified, educational version of [AgentScope](https://github.com/agentscope-ai/agentscope): **small enough to read in one sitting, but complete enough to run a real ReAct + Tool Calling loop**.

> 中文：`nano-agentscope` 是一个“手写 AI Agent 框架”的教学实现——强调**边界清晰、可扩展、可排障**，用最小的代码把 Agent 的关键闭环跑通。

---

## 特性（面向真实工程痛点）

- **可跑通的 ReAct 闭环**：`Reasoning -> Tool Use -> Tool Result -> Reasoning`，直到输出最终答案。
- **工具系统（Tool Calling）**：
  - 支持将普通 Python 函数注册为工具
  - 自动从 **函数签名 + docstring** 生成 OpenAI function-calling 风格 JSON Schema
- **MCP 远程工具支持**：通过 `HttpStatelessClient` 连接 MCP Server，将远程工具像本地函数一样注册进 `Toolkit`。
- **可观测与可排障**：支持打印 LLM 请求、工具入参、工具结果、Token 统计；工具结果支持“**不截断**”。
- **扩展能力一览（教学用实现）**：
  - `pipeline`：`sequential_pipeline` / `loop_pipeline` + `MsgHub`（多智能体编排与广播）
  - `steering`：`SteerableAgent` + 人工干预 / 确认工具（可中断、人机协作）
  - `rag`：`SimpleKnowledge` + `create_retrieve_tool`（简易 RAG：关键词检索 + 工具化）

---

## 适合谁？不适合谁？

适合：
- 想从 0 搭一套自己的 Agent 基建（至少要能 Tool Calling、能调试）
- 想读懂 AgentScope/LangChain 这类框架背后的“最小必要抽象”
- 想做 Agent 相关教学/分享/源码导读

不适合（当前阶段）：
- 直接上生产、追求完备的安全隔离/沙箱执行/复杂记忆/评测体系
- 需要完整兼容所有厂商/所有工具协议细节

---

## 快速开始（30 秒跑起来）

### 1) 安装

```bash
cd nano-agentscope
pip install -e .
```

开发依赖（跑测试用）：

```bash
pip install -e ".[dev]"
```

### 2) 配置模型 API Key

支持的模型：

| Provider | 类名 | 环境变量 | 示例模型 |
|---|---|---|---|
| DashScope（通义千问） | `DashScopeChatModel` | `DASHSCOPE_API_KEY` | `qwen-max`, `qwen-plus`, `qwen-turbo` |
| OpenAI | `OpenAIChatModel` | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o` |

推荐：国内环境优先使用 DashScope。

### 3) 运行示例

```bash
export DASHSCOPE_API_KEY="your_key"
python examples/01_hello_world.py
```

或：

```bash
export OPENAI_API_KEY="your_key"
python examples/01_hello_world.py --openai
```

---

## 示例导航（建议按顺序跑）

- `examples/01_hello_world.py`：最小对话闭环（无工具）
- `examples/02_tool_calling.py`：本地工具调用（自动 schema）
- `examples/03_multi_turn_conversation.py`：多轮对话与记忆
- `examples/04_simple_rag.py`：简易 RAG（`SimpleKnowledge` + 检索工具）
- `examples/05_multi_agent_conversation.py`：多智能体编排（pipeline + MsgHub）
- `examples/06_realtime_steering.py`：实时干预与中断（Steering）
- `examples/07_mcp.py`：连接 MCP Server，注册远程工具并给 Agent 使用

---

## 最小心智模型：一条可 debug 的 Agent 链路

你可以把 `nano-agentscope` 理解成 5 个可替换模块 + 1 个编排器：

- **`Msg`（内部协议）**：文本、工具调用、工具结果、图片等统一成 `ContentBlock`
- **`Memory`（上下文）**：存历史消息
- **`Formatter`（协议转换）**：内部 `Msg` -> SDK `messages`（默认 OpenAI 风格）
- **`Model`（厂商适配）**：SDK response -> 统一 `ChatResponse`
- **`Toolkit`（工具系统）**：注册工具、生成 schema、执行工具
- **`ReActAgent`（编排器）**：控制 ReAct 循环

最小闭环：

```
User Msg
  -> Memory.add
  -> Formatter.format
  -> Model(... tools=Toolkit.get_json_schemas())
  -> Msg(content=[text/tool_use/...])
  -> if tool_use: Toolkit.call_tool_function -> tool_result -> Memory.add
  -> else: return
```

---

## 工具调用（Tool Calling）

### 1) 定义并注册一个工具

工具函数就是普通 Python 函数：写清楚类型注解，并在 docstring 里写 `Args:` 描述，框架会自动生成 JSON Schema。

```python
from nano_agentscope import Toolkit, ToolResponse, TextBlock

def get_weather(city: str) -> ToolResponse:
    """获取城市天气

    Args:
        city: 城市名称，如 "北京"
    """
    return ToolResponse(content=[TextBlock(type="text", text=f"{city} 晴 25°C")])

toolkit = Toolkit()
toolkit.register_tool_function(get_weather)
```

### 2) 交给 `ReActAgent` 自动决定何时调用

`ReActAgent` 会把 tools schema 交给模型，并在响应出现 `tool_use` 时执行工具，将 `tool_result` 回写记忆后继续推理。

---

## MCP：把远程工具当成本地工具用

`examples/07_mcp.py` 展示了完整流程：

1. `HttpStatelessClient(..., transport="streamable_http" | "sse")` 连接 MCP Server
2. `await toolkit.register_mcp_client(client)` 拉取并注册远程工具
3. 让 `ReActAgent` 使用这些工具

提示：示例里使用了一个演示 URL，请替换成你自己的 MCP Server。

---

## 日志与调试（强烈建议打开）

Agent 的问题往往出在“模型看到的 messages / tools”和“工具回写的格式”上，因此可观测性非常关键。

环境变量：

- `NANO_AGENTSCOPE_VERBOSE=1`
  - 打印更详细的 LLM 请求、工具调用参数、Token 使用统计
- `NANO_AGENTSCOPE_LOG_MAX_LENGTH=0`
  - 工具结果日志不截断（默认会截断，0 表示完全不截断）

推荐调试配置：

```bash
export NANO_AGENTSCOPE_VERBOSE=1
export NANO_AGENTSCOPE_LOG_MAX_LENGTH=0
```

---

## 代码结构（当前实现）

```
src/nano_agentscope/
├── __init__.py          # 对外导出
├── message.py           # Msg / ContentBlock（内部协议）
├── memory.py            # InMemoryMemory 等
├── formatter.py         # OpenAIFormatter 等
├── model.py             # DashScopeChatModel / OpenAIChatModel
├── tool.py              # Toolkit / ToolResponse / schema 生成
├── agent.py             # ReActAgent（推理-行动循环）
├── mcp.py               # MCP 客户端与工具包装
├── pipeline.py          # sequential_pipeline / loop_pipeline / MsgHub
├── rag.py               # SimpleKnowledge / create_retrieve_tool
└── steering.py          # SteerableAgent / 人工干预与确认工具
```

---

## 文档/文章

- `docs/从0开始手写AI Agent框架：nano-agentscope（一）.md`

---

## 贡献

欢迎 PR / Issue（尤其是：更多示例、更多模型适配、工具系统增强、MCP 实战案例）。

本地测试：

```bash
pytest
```

---

## License

Apache License 2.0
