# Nano-AgentScope

A simplified educational version of [AgentScope](https://github.com/agentscope-ai/agentscope) - a Multi-Agent Framework.

## 项目简介

Nano-AgentScope 是 AgentScope 的简化教学版本，旨在帮助开发者理解多智能体框架的核心概念和实现原理。

## 核心模块

```
nano_agentscope/
├── __init__.py          # 入口模块
├── message.py           # 消息模块：Msg 和 ContentBlock
├── model.py             # 模型模块：ChatModelBase, OpenAIChatModel
├── memory.py            # 记忆模块：MemoryBase, InMemoryMemory
├── tool.py              # 工具模块：Toolkit, ToolResponse
├── formatter.py         # 格式化模块：FormatterBase, OpenAIFormatter
└── agent.py             # 智能体模块：AgentBase, ReActAgent
```

## 架构概述

### 1. 消息流 (Message Flow)

```
User Input → Msg → Formatter → API Request → Model → ChatResponse → Msg → Output
```

### 2. ReAct 循环 (Reasoning-Acting Loop)

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │     Memory      │
                    │  (存储对话历史)  │
                    └────────┬────────┘
                             ▼
              ┌──────────────────────────────┐
              │       Reasoning (推理)        │
              │  Formatter → Model → Response │
              └──────────────┬───────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Tool Calls?    │
                    └────────┬────────┘
                      Yes    │    No
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌─────────────┐    ┌─────────────┐
           │   Acting    │    │   Output    │
           │ (执行工具)  │    │  (返回结果)  │
           └──────┬──────┘    └─────────────┘
                  │
                  └──────────────────┐
                                     ▼
                    ┌─────────────────┐
                    │ Update Memory   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Continue Loop   │
                    └─────────────────┘
```

### 3. 核心类图

```
                         ┌──────────────┐
                         │  AgentBase   │
                         │  (异步基类)   │
                         └──────┬───────┘
                                │
                         ┌──────┴───────┐
                         │  ReActAgent  │
                         └──────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  Model  │ │Formatter│ │ Memory  │ │ Toolkit │ │   Msg   │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## 安装

```bash
cd nano-agentscope
pip install -e .
```

## 支持的模型

| 模型提供商 | 类名 | 环境变量 | 模型示例 |
|-----------|------|----------|----------|
| 阿里云 DashScope | `DashScopeChatModel` | `DASHSCOPE_API_KEY` | qwen-max, qwen-plus, qwen-turbo |
| OpenAI | `OpenAIChatModel` | `OPENAI_API_KEY` | gpt-4o-mini, gpt-4o |

**推荐使用 DashScope（通义千问）**：国内访问稳定，价格实惠。

## 快速开始

```python
import asyncio
from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,  # 阿里云通义千问
    # OpenAIChatModel,   # 或使用 OpenAI
    OpenAIFormatter,
    Toolkit,
    InMemoryMemory,
    ToolResponse,
    TextBlock,
)

# 定义一个简单的工具函数
def get_weather(city: str) -> ToolResponse:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称
    """
    return ToolResponse(
        content=[TextBlock(type="text", text=f"{city}今天天气晴朗，温度25°C")]
    )

async def main():
    # 创建工具集
    toolkit = Toolkit()
    toolkit.register_tool_function(get_weather)
    
    # 创建智能体（使用 DashScope 通义千问）
    agent = ReActAgent(
        name="小助手",
        sys_prompt="你是一个有帮助的AI助手。",
        model=DashScopeChatModel(model_name="qwen-max"),  # 通义千问
        # model=OpenAIChatModel(model_name="gpt-4o-mini"),  # 或 OpenAI
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    
    # 与智能体对话
    from nano_agentscope import Msg
    response = await agent(Msg(name="user", content="北京今天天气怎么样？", role="user"))
    print(response.get_text_content())

asyncio.run(main())
```

## 学习路径

1. **理解消息结构** - 阅读 `message.py`，了解 Msg 和各种 ContentBlock
2. **理解模型调用** - 阅读 `model.py`，了解如何封装 LLM API
3. **理解格式化器** - 阅读 `formatter.py`，了解如何将 Msg 转换为 API 格式
4. **理解工具调用** - 阅读 `tool.py`，了解函数调用的实现
5. **理解智能体** - 阅读 `agent.py`，了解 ReAct 循环的实现
6. **运行示例** - 运行 `examples/` 目录下的示例

## 许可证

Apache License 2.0

