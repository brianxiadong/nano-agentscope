# -*- coding: utf-8 -*-
"""
智能体模块 - 定义 Agent 的抽象和实现

本模块定义了智能体的核心组件：
1. AgentBase - 智能体基类，定义基本接口
2. ReActAgent - ReAct 模式的智能体实现

学习要点：
- Agent 是 LLM 应用的核心，协调各个组件
- ReAct (Reasoning + Acting) 是一种常用的 Agent 模式
- 通过 Memory 维持上下文，通过 Tool 与外界交互

ReAct 循环流程：
1. 接收用户输入 -> 存入 Memory
2. 调用 LLM 推理 (Reasoning)
3. 如果需要调用工具 -> 执行工具 (Acting) -> 返回步骤 2
4. 如果不需要工具 -> 输出结果
"""

import asyncio
from abc import abstractmethod
from typing import Any, AsyncGenerator

from .message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from .model import ChatModelBase, ChatResponse
from .memory import MemoryBase, InMemoryMemory
from .formatter import FormatterBase, OpenAIFormatter
from .tool import Toolkit, ToolResponse


class AgentBase:
    """智能体基类 - 定义智能体的基本接口
    
    所有智能体都应继承此类并实现 reply 和 observe 方法。
    
    核心方法：
    - reply: 生成回复（主要逻辑）
    - observe: 观察消息（不产生回复）
    - __call__: 调用入口，包装 reply
    
    设计原则：
    - 异步优先：所有方法都是异步的
    - 可扩展：通过继承实现不同类型的 Agent
    """
    
    name: str  # 智能体名称
    
    @abstractmethod
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """生成回复消息
        
        Args:
            msg: 输入消息
            
        Returns:
            回复消息
        """
        pass
    
    @abstractmethod
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息，不产生回复
        
        Args:
            msg: 要观察的消息
        """
        pass
    
    async def __call__(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """调用智能体
        
        这是调用智能体的主要入口，内部调用 reply 方法。
        """
        return await self.reply(msg)


class ReActAgent(AgentBase):
    """ReAct 智能体 - 实现 Reasoning + Acting 循环
    
    ReAct 是一种强大的 Agent 模式，通过交替进行推理（调用 LLM）
    和行动（执行工具）来完成复杂任务。
    
    组件：
    - model: LLM 模型，用于推理
    - formatter: 格式化器，转换消息格式
    - memory: 记忆，存储对话历史
    - toolkit: 工具集，提供可调用的函数
    
    工作流程：
    ```
    用户输入 -> Memory 存储 -> LLM 推理 -> 
    -> 有工具调用? -> 是 -> 执行工具 -> 存储结果 -> 继续推理
                  -> 否 -> 返回结果
    ```
    
    Example:
        >>> agent = ReActAgent(
        ...     name="助手",
        ...     sys_prompt="你是一个有帮助的助手",
        ...     model=OpenAIChatModel(model_name="gpt-4o-mini"),
        ...     formatter=OpenAIFormatter(),
        ...     toolkit=toolkit,
        ...     memory=InMemoryMemory(),
        ... )
        >>> response = await agent(Msg(name="user", content="你好", role="user"))
    """
    
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        toolkit: Toolkit | None = None,
        memory: MemoryBase | None = None,
        max_iters: int = 10,
    ) -> None:
        """初始化 ReAct 智能体
        
        Args:
            name: 智能体名称
            sys_prompt: 系统提示词，定义智能体的角色和行为
            model: LLM 模型
            formatter: 消息格式化器
            toolkit: 工具集（可选）
            memory: 记忆模块（可选，默认使用 InMemoryMemory）
            max_iters: 最大推理-行动循环次数，防止无限循环
        """
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.formatter = formatter
        self.toolkit = toolkit or Toolkit()
        self.memory = memory or InMemoryMemory()
        self.max_iters = max_iters
    
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """生成回复 - ReAct 循环的主逻辑
        
        流程：
        1. 将输入消息存入记忆
        2. 进入推理-行动循环
        3. 推理：调用 LLM 获取响应
        4. 检查响应中是否有工具调用
        5. 如果有工具调用：执行工具，存储结果，继续循环
        6. 如果没有工具调用：返回文本响应
        7. 超过最大迭代次数：强制总结并返回
        """
        # Step 1: 存储输入消息
        await self.memory.add(msg)
        
        # Step 2-6: ReAct 循环
        for _ in range(self.max_iters):
            # 推理步骤
            response_msg = await self._reasoning()
            
            # 检查是否有工具调用
            tool_use_blocks = response_msg.get_content_blocks("tool_use")
            
            if not tool_use_blocks:
                # 没有工具调用，直接返回
                return response_msg
            
            # 执行工具调用
            for tool_call in tool_use_blocks:
                await self._acting(tool_call)
        
        # Step 7: 超过最大迭代，强制总结
        return await self._summarize()
    
    async def _reasoning(self) -> Msg:
        """推理步骤 - 调用 LLM 生成响应
        
        流程：
        1. 构建消息列表（系统提示 + 记忆）
        2. 格式化为 API 格式
        3. 调用模型
        4. 解析响应为 Msg
        5. 存储到记忆并打印
        """
        # 构建消息列表
        msgs = [
            Msg(name="system", content=self.sys_prompt, role="system"),
            *await self.memory.get_memory(),
        ]
        
        # 格式化消息
        formatted_msgs = await self.formatter.format(msgs)
        
        # 获取工具 schema
        tools = self.toolkit.get_json_schemas() or None
        
        # 调用模型
        response = await self.model(
            messages=formatted_msgs,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        
        # 处理响应（流式或非流式）
        if isinstance(response, AsyncGenerator):
            # 流式响应：累积所有 chunk
            final_response = None
            async for chunk in response:
                final_response = chunk
                # 可以在这里添加流式输出逻辑
                self._print_streaming(chunk)
            response = final_response
            print()  # 换行
        
        # 转换为 Msg
        response_msg = Msg(
            name=self.name,
            content=list(response.content) if response else [],
            role="assistant",
            metadata=response.metadata if response else None,
        )
        
        # 存储到记忆
        await self.memory.add(response_msg)
        
        # 打印非流式响应
        if not self.model.stream:
            self._print_response(response_msg)
        
        return response_msg
    
    async def _acting(self, tool_call: ToolUseBlock) -> None:
        """行动步骤 - 执行工具调用
        
        Args:
            tool_call: 工具调用块
        """
        # 执行工具
        tool_result = await self.toolkit.call_tool_function(tool_call)
        
        # 构建工具结果消息
        result_msg = Msg(
            name="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=tool_result.content,
                )
            ],
            role="system",
        )
        
        # 存储到记忆
        await self.memory.add(result_msg)
        
        # 打印工具结果
        self._print_tool_result(tool_call, tool_result)
    
    async def _summarize(self) -> Msg:
        """超过最大迭代次数时的总结
        
        强制 LLM 生成一个总结性回复。
        """
        # 添加提示消息
        hint_msg = Msg(
            name="system",
            content="你已经达到最大迭代次数，请直接给出总结性回答。",
            role="user",
        )
        
        # 构建消息
        msgs = [
            Msg(name="system", content=self.sys_prompt, role="system"),
            *await self.memory.get_memory(),
            hint_msg,
        ]
        
        # 格式化并调用模型（不使用工具）
        formatted_msgs = await self.formatter.format(msgs)
        response = await self.model(messages=formatted_msgs)
        
        # 处理流式响应
        if isinstance(response, AsyncGenerator):
            final_response = None
            async for chunk in response:
                final_response = chunk
                self._print_streaming(chunk)
            response = final_response
            print()
        
        # 转换为 Msg
        response_msg = Msg(
            name=self.name,
            content=list(response.content) if response else [],
            role="assistant",
        )
        
        await self.memory.add(response_msg)
        
        if not self.model.stream:
            self._print_response(response_msg)
        
        return response_msg
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息，存入记忆但不产生回复"""
        await self.memory.add(msg)
    
    def _print_streaming(self, chunk: ChatResponse) -> None:
        """打印流式响应"""
        for block in chunk.content:
            if block.get("type") == "text":
                # 简单的流式打印（仅打印最新内容）
                # 实际实现可能需要更复杂的逻辑
                pass
        
        # 打印文本内容
        text_blocks = [b for b in chunk.content if b.get("type") == "text"]
        if text_blocks:
            text = text_blocks[-1].get("text", "")
            print(f"\r{self.name}: {text}", end="", flush=True)
    
    def _print_response(self, msg: Msg) -> None:
        """打印响应消息"""
        text = msg.get_text_content()
        if text:
            print(f"{msg.name}: {text}")
        
        # 打印工具调用
        for block in msg.get_content_blocks("tool_use"):
            print(f"  [调用工具] {block['name']}({block.get('input', {})})")
    
    def _print_tool_result(
        self,
        tool_call: ToolUseBlock,
        result: ToolResponse,
    ) -> None:
        """打印工具执行结果"""
        text = ""
        for block in result.content:
            if block.get("type") == "text":
                text += block.get("text", "")
        print(f"  [工具结果] {tool_call['name']}: {text[:100]}...")


class UserAgent(AgentBase):
    """用户智能体 - 获取用户输入
    
    这是一个特殊的智能体，用于获取用户的输入。
    
    Example:
        >>> user = UserAgent("User")
        >>> msg = await user()  # 等待用户输入
        >>> print(msg.content)
    """
    
    def __init__(self, name: str = "User") -> None:
        """初始化用户智能体
        
        Args:
            name: 用户名称
        """
        self.name = name
    
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """获取用户输入"""
        # 显示上一条消息（如果有）
        if isinstance(msg, Msg):
            text = msg.get_text_content()
            if text:
                print()  # 换行
        
        # 获取用户输入
        user_input = input(f"{self.name}: ")
        
        return Msg(
            name=self.name,
            content=user_input,
            role="user",
        )
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """用户智能体不需要观察功能"""
        pass


