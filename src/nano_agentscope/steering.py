# -*- coding: utf-8 -*-
"""
实时干预模块 - 支持中断和人工干预

本模块提供 Agent 执行过程中的实时控制能力：
1. SteerableAgent - 可中断的 Agent 封装器
2. create_human_intervention_tool - 创建人工干预工具

学习要点：
- 异步取消机制 (asyncio.CancelledError)
- Agent 执行状态管理
- 人机协作模式

核心概念：
- Interruption: 用户主动打断 Agent 执行
- Intervention: Agent 请求人类帮助
- Steering: 实时调整 Agent 行为
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .agent import AgentBase

from .message import Msg, TextBlock
from .tool import ToolResponse


class SteerableAgent:
    """可中断的 Agent 封装器
    
    将普通 Agent 封装为支持实时中断的版本。
    
    工作原理：
    1. 跟踪当前执行的 asyncio Task
    2. 提供 interrupt() 方法取消执行
    3. 捕获 CancelledError 并调用 handle_interrupt()
    
    使用场景：
    - 长时间运行的 Agent 任务
    - 需要人工干预的场景
    - 对话系统中的实时控制
    
    Example:
        >>> from nano_agentscope import ReActAgent
        >>> from nano_agentscope.steering import SteerableAgent
        >>> 
        >>> agent = ReActAgent(name="助手", ...)
        >>> steerable = SteerableAgent(agent)
        >>> 
        >>> # 在另一个协程中可以调用 steerable.interrupt() 来中断
        >>> result = await steerable(msg)
    
    注意：
        - 中断是异步操作，可能不会立即生效
        - 中断后会调用 agent.handle_interrupt() 方法
        - 需要在异步环境中使用
    """
    
    def __init__(self, agent: "AgentBase") -> None:
        """初始化可中断 Agent
        
        Args:
            agent: 要封装的 Agent 实例
        """
        self.agent = agent
        self._current_task: asyncio.Task | None = None
        self._is_running: bool = False
    
    async def __call__(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """执行 Agent 并支持中断
        
        Args:
            msg: 输入消息
            
        Returns:
            Agent 的回复消息（正常完成或中断后的响应）
        """
        self._current_task = asyncio.current_task()
        self._is_running = True
        
        try:
            result = await self.agent(msg)
            return result
        except asyncio.CancelledError:
            # 调用 Agent 的中断处理方法
            return await self.agent.handle_interrupt(msg)
        finally:
            self._is_running = False
            self._current_task = None
    
    def interrupt(self) -> bool:
        """中断正在执行的 Agent
        
        返回:
            bool: 是否成功发送中断信号
            
        Note:
            - 中断是异步的，调用后任务不会立即停止
            - 如果 Agent 没有在执行，返回 False
        """
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            return True
        return False
    
    @property
    def is_running(self) -> bool:
        """Agent 是否正在执行"""
        return self._is_running
    
    @property
    def name(self) -> str:
        """获取被封装 Agent 的名称"""
        return self.agent.name


def create_human_intervention_tool(
    prompt: str = "请输入您的指令：",
    tool_name: str = "ask_human",
    tool_description: str | None = None,
) -> Callable:
    """创建人工干预工具
    
    生成一个工具函数，让 Agent 可以在执行过程中请求人类帮助。
    
    使用场景：
    - Agent 遇到不确定的决策
    - 需要用户确认敏感操作
    - 请求额外信息或澄清
    
    Example:
        >>> tool = create_human_intervention_tool()
        >>> toolkit.register_tool_function(tool)
        >>> 
        >>> # Agent 可以调用 ask_human 来请求帮助
        >>> # Agent: "我不确定这个操作，让我问问用户..."
        >>> # [调用 ask_human(question="是否继续删除文件?")]
        >>> # 用户输入: "是"
        >>> # Agent: "好的，用户确认了，继续执行..."
    
    Args:
        prompt: 提示用户输入时显示的文本
        tool_name: 工具函数的名称
        tool_description: 工具的描述（可选）
        
    Returns:
        一个可以注册到 Toolkit 的工具函数
    """
    
    async def ask_human(question: str) -> ToolResponse:
        """向人类请求帮助或确认
        
        当你遇到以下情况时可以使用此工具：
        - 需要用户确认某个操作
        - 需要额外的信息来完成任务
        - 遇到不确定或有风险的决策
        
        Args:
            question: 需要人类回答的问题
            
        Returns:
            ToolResponse: 包含人类回复的工具响应
        """
        print(f"\n{'='*50}")
        print(f"🙋 Agent 请求帮助:")
        print(f"   {question}")
        print(f"{'='*50}")
        
        # 获取用户输入
        try:
            answer = input(prompt)
        except EOFError:
            answer = "(用户未提供输入)"
        except KeyboardInterrupt:
            return ToolResponse(
                content=[TextBlock(type="text", text="(用户取消了输入)")],
                is_interrupted=True,
            )
        
        return ToolResponse(
            content=[TextBlock(
                type="text", 
                text=f"人类回复: {answer}"
            )]
        )
    
    # 设置函数名称
    ask_human.__name__ = tool_name
    
    # 设置描述（如果提供）
    if tool_description:
        ask_human.__doc__ = f"""{tool_description}
        
        Args:
            question: 需要人类回答的问题
        """
    
    return ask_human


def create_confirmation_tool(
    yes_prompt: str = "确认执行？(y/n): ",
    tool_name: str = "confirm_action",
) -> Callable:
    """创建确认工具
    
    生成一个简单的是/否确认工具。
    
    Args:
        yes_prompt: 确认提示文本
        tool_name: 工具名称
        
    Returns:
        确认工具函数
    """
    
    async def confirm_action(action_description: str) -> ToolResponse:
        """请求用户确认是否执行某个操作
        
        Args:
            action_description: 需要确认的操作描述
            
        Returns:
            确认结果
        """
        print(f"\n⚠️  需要确认:")
        print(f"   {action_description}")
        
        try:
            response = input(yes_prompt).strip().lower()
            confirmed = response in ("y", "yes", "是", "确认")
        except (EOFError, KeyboardInterrupt):
            confirmed = False
        
        if confirmed:
            return ToolResponse(
                content=[TextBlock(type="text", text="用户已确认，可以继续执行")]
            )
        else:
            return ToolResponse(
                content=[TextBlock(type="text", text="用户拒绝执行该操作")]
            )
    
    confirm_action.__name__ = tool_name
    return confirm_action
