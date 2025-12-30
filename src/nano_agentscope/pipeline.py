# -*- coding: utf-8 -*-
"""
管道模块 - 多智能体协同编排

本模块提供多智能体协同工作的工具：
1. sequential_pipeline - 顺序执行多个 Agent
2. loop_pipeline - 循环执行多个 Agent
3. MsgHub - 消息广播上下文管理器

学习要点：
- 多智能体系统需要协调各个 Agent 的执行顺序
- Pipeline 是一种常见的编排模式
- MsgHub 实现了"群聊"式的消息共享

核心概念：
- Sequential: 链式传递，A -> B -> C
- Loop: 循环讨论，A -> B -> C -> A -> B -> C -> ...
- Broadcast: 广播通知，A 说话 -> B,C,D 都能听到
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentBase
from .message import Msg


async def sequential_pipeline(
    agents: list["AgentBase"],
    msg: Msg | list[Msg] | None = None,
) -> Msg | None:
    """顺序执行管道 - 依次执行多个 Agent
    
    执行流程：
    1. 将初始消息传给第一个 Agent
    2. 第一个 Agent 回复后，将回复传给第二个 Agent
    3. 依此类推，直到最后一个 Agent
    4. 返回最后一个 Agent 的回复
    
    适用场景：
    - 任务分解：分析 -> 规划 -> 执行
    - 多轮审核：草稿 -> 审核 -> 修改
    - 翻译链：中文 -> 英文 -> 日文
    
    Example:
        >>> analyzer = ReActAgent(name="分析师", ...)
        >>> planner = ReActAgent(name="规划师", ...)
        >>> executor = ReActAgent(name="执行者", ...)
        >>> 
        >>> result = await sequential_pipeline(
        ...     agents=[analyzer, planner, executor],
        ...     msg=Msg(name="user", content="请帮我完成任务", role="user")
        ... )
        >>> print(result.get_text_content())
    
    Args:
        agents: Agent 列表，按执行顺序排列
        msg: 初始输入消息
        
    Returns:
        最后一个 Agent 的回复消息
    """
    current_msg = msg
    for agent in agents:
        current_msg = await agent(current_msg)
    return current_msg


async def loop_pipeline(
    agents: list["AgentBase"],
    msg: Msg | list[Msg] | None = None,
    max_rounds: int = 3,
) -> Msg | None:
    """循环执行管道 - 多轮循环执行 Agent 组
    
    执行流程：
    1. 按顺序执行所有 Agent（一轮）
    2. 重复执行指定轮数
    3. 返回最后一个 Agent 的最后一轮回复
    
    适用场景：
    - 辩论：正方 -> 反方 -> 正方 -> 反方
    - 迭代优化：生成 -> 评估 -> 生成 -> 评估
    - 多人讨论：A -> B -> C -> A -> B -> C
    
    Example:
        >>> agent_a = ReActAgent(name="正方", ...)
        >>> agent_b = ReActAgent(name="反方", ...)
        >>> 
        >>> result = await loop_pipeline(
        ...     agents=[agent_a, agent_b],
        ...     msg=Msg(name="主持人", content="请辩论AI是否有益", role="user"),
        ...     max_rounds=3
        ... )
    
    Args:
        agents: Agent 列表
        msg: 初始输入消息
        max_rounds: 最大循环轮数
        
    Returns:
        最后一个 Agent 的最后轮回复
    """
    current_msg = msg
    
    for round_num in range(max_rounds):
        print(f"\n{'='*40}")
        print(f"📢 第 {round_num + 1}/{max_rounds} 轮")
        print(f"{'='*40}")
        
        for agent in agents:
            current_msg = await agent(current_msg)
    
    return current_msg


class MsgHub:
    """消息广播中心 - 管理多智能体消息共享
    
    MsgHub 实现了"群聊"模式：
    - 所有参与者共享同一个消息空间
    - 任何人的发言都会被其他人"看到"（observe）
    - 支持动态添加/移除参与者
    
    工作原理：
    1. 进入上下文时，向所有参与者广播公告
    2. 提供 broadcast() 方法手动广播消息
    3. 退出上下文时自动清理
    
    适用场景：
    - 多人讨论：让所有人都能看到对话
    - 信息同步：确保所有 Agent 获得相同信息
    - 群体协作：模拟会议/讨论场景
    
    Example:
        >>> moderator = ReActAgent(name="主持人", ...)
        >>> expert_a = ReActAgent(name="专家A", ...)
        >>> expert_b = ReActAgent(name="专家B", ...)
        >>> 
        >>> async with MsgHub(
        ...     participants=[moderator, expert_a, expert_b],
        ...     announcement=Msg(name="系统", content="讨论开始", role="system")
        ... ) as hub:
        ...     # 在这里，所有人都看到了公告
        ...     response = await moderator(Msg(...))
        ...     # 手动广播给其他人
        ...     await hub.broadcast(response)
    
    注意：
        - 本实现是简化版，不会自动广播 Agent 的回复
        - 需要手动调用 broadcast() 来共享消息
        - AgentScope 完整版支持自动广播
    """
    
    def __init__(
        self,
        participants: list["AgentBase"],
        announcement: Msg | list[Msg] | None = None,
    ) -> None:
        """初始化消息广播中心
        
        Args:
            participants: 参与者列表
            announcement: 进入时的公告消息
        """
        self.participants: list["AgentBase"] = list(participants)
        self.announcement = announcement
    
    async def __aenter__(self) -> "MsgHub":
        """进入上下文 - 广播公告"""
        if self.announcement:
            await self.broadcast(self.announcement)
        return self
    
    async def __aexit__(self, *args) -> None:
        """退出上下文"""
        # 简化版不需要清理
        pass
    
    async def broadcast(self, msg: Msg | list[Msg]) -> None:
        """广播消息给所有参与者
        
        调用每个参与者的 observe() 方法，让他们"看到"消息。
        
        Args:
            msg: 要广播的消息
        """
        for participant in self.participants:
            await participant.observe(msg)
    
    def add(self, agent: "AgentBase" | list["AgentBase"]) -> None:
        """添加参与者
        
        Args:
            agent: 要添加的 Agent（单个或列表）
        """
        if isinstance(agent, list):
            for a in agent:
                if a not in self.participants:
                    self.participants.append(a)
        else:
            if agent not in self.participants:
                self.participants.append(agent)
    
    def remove(self, agent: "AgentBase" | list["AgentBase"]) -> None:
        """移除参与者
        
        Args:
            agent: 要移除的 Agent（单个或列表）
        """
        if isinstance(agent, list):
            for a in agent:
                if a in self.participants:
                    self.participants.remove(a)
        else:
            if agent in self.participants:
                self.participants.remove(agent)
    
    @property
    def size(self) -> int:
        """参与者数量"""
        return len(self.participants)
