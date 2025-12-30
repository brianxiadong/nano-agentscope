# -*- coding: utf-8 -*-
"""
Nano-AgentScope - AgentScope 的简化教学版本

这是一个用于学习多智能体框架核心概念的简化实现。
通过阅读源码，你可以了解：

1. 消息系统 (message.py)
   - Msg: 智能体之间通信的基本单位
   - ContentBlock: 支持文本、工具调用、图片等多种类型

2. 模型封装 (model.py)
   - ChatModelBase: 定义统一的模型调用接口
   - OpenAIChatModel: OpenAI API 的具体实现
   - ChatResponse: 模型响应的统一格式

3. 记忆管理 (memory.py)
   - MemoryBase: 记忆的抽象接口
   - InMemoryMemory: 基于内存列表的简单实现

4. 工具系统 (tool.py)
   - Toolkit: 工具函数的注册和管理
   - ToolResponse: 工具执行结果
   - 自动从 docstring 解析 JSON Schema

5. 格式化器 (formatter.py)
   - FormatterBase: 定义格式化接口
   - OpenAIFormatter: 转换为 OpenAI API 格式

6. 智能体 (agent.py)
   - AgentBase: 智能体的基类
   - ReActAgent: ReAct 模式的实现
   - UserAgent: 获取用户输入

快速开始:
    >>> import asyncio
    >>> from nano_agentscope import (
    ...     ReActAgent, DashScopeChatModel, OpenAIFormatter,
    ...     Toolkit, InMemoryMemory, Msg, ToolResponse, TextBlock,
    ... )
    >>> 
    >>> # 定义工具
    >>> def get_time() -> ToolResponse:
    ...     '''获取当前时间'''
    ...     from datetime import datetime
    ...     return ToolResponse(content=[
    ...         TextBlock(type="text", text=datetime.now().strftime("%H:%M:%S"))
    ...     ])
    >>> 
    >>> # 创建工具集和智能体
    >>> toolkit = Toolkit()
    >>> toolkit.register_tool_function(get_time)
    >>> 
    >>> agent = ReActAgent(
    ...     name="助手",
    ...     sys_prompt="你是一个有帮助的助手。",
    ...     model=DashScopeChatModel(model_name="qwen-max"),  # 通义千问
    ...     # model=OpenAIChatModel(model_name="gpt-4o-mini"),  # 或 OpenAI
    ...     formatter=OpenAIFormatter(),
    ...     toolkit=toolkit,
    ... )
    >>> 
    >>> # 对话
    >>> async def main():
    ...     response = await agent(Msg(name="user", content="现在几点了？", role="user"))
    ...     print(response.get_text_content())
    >>> 
    >>> asyncio.run(main())
"""

__version__ = "0.1.0"

# 消息模块
from .message import (
    Msg,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    ContentBlock,
)

# 模型模块
from .model import (
    ChatModelBase,
    DashScopeChatModel,
    OpenAIChatModel,
    ChatResponse,
    ChatUsage,
)

# 记忆模块
from .memory import (
    MemoryBase,
    InMemoryMemory,
)

# 工具模块
from .tool import (
    Toolkit,
    ToolResponse,
    calculator,
    get_current_time,
)

# MCP 模块
from .mcp import (
    HttpStatelessClient,
    MCPToolFunction,
)

# 格式化模块
from .formatter import (
    FormatterBase,
    OpenAIFormatter,
    SimpleFormatter,
)

# 智能体模块
from .agent import (
    AgentBase,
    ReActAgent,
    UserAgent,
)

# RAG 模块
from .rag import (
    Document,
    SimpleKnowledge,
    create_retrieve_tool,
)

# Pipeline 模块
from .pipeline import (
    sequential_pipeline,
    loop_pipeline,
    MsgHub,
)

# Steering 模块 (实时干预)
from .steering import (
    SteerableAgent,
    create_human_intervention_tool,
    create_confirmation_tool,
)


__all__ = [
    # 版本
    "__version__",
    # 消息
    "Msg",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "ContentBlock",
    # 模型
    "ChatModelBase",
    "DashScopeChatModel",
    "OpenAIChatModel",
    "ChatResponse",
    "ChatUsage",
    # 记忆
    "MemoryBase",
    "InMemoryMemory",
    # 工具
    "Toolkit",
    "ToolResponse",
    "calculator",
    "get_current_time",
    # MCP
    "HttpStatelessClient",
    "MCPToolFunction",
    # 格式化
    "FormatterBase",
    "OpenAIFormatter",
    "SimpleFormatter",
    # 智能体
    "AgentBase",
    "ReActAgent",
    "UserAgent",
    # RAG
    "Document",
    "SimpleKnowledge",
    "create_retrieve_tool",
    # Pipeline
    "sequential_pipeline",
    "loop_pipeline",
    "MsgHub",
    # Steering (实时干预)
    "SteerableAgent",
    "create_human_intervention_tool",
    "create_confirmation_tool",
]



