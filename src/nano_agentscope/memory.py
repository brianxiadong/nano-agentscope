# -*- coding: utf-8 -*-
"""
记忆模块 - 管理对话历史

本模块定义了记忆系统的抽象和实现：
1. MemoryBase - 记忆基类，定义统一接口
2. InMemoryMemory - 基于内存的简单实现

学习要点：
- 记忆模块负责存储和管理对话历史
- 智能体通过记忆来维持上下文连贯性
- 可以扩展实现更复杂的记忆管理（如压缩、检索等）
"""

from abc import abstractmethod
from typing import Any

from .message import Msg


class MemoryBase:
    """记忆基类 - 定义记忆管理的统一接口
    
    记忆模块是智能体的核心组件之一，用于：
    1. 存储对话历史
    2. 提供上下文给 LLM
    3. 支持记忆的持久化和检索
    
    所有记忆实现都应继承此类并实现抽象方法。
    """
    
    @abstractmethod
    async def add(self, msg: Msg | list[Msg] | None) -> None:
        """添加消息到记忆
        
        Args:
            msg: 要添加的消息，可以是单条消息、消息列表或 None
        """
        pass
    
    @abstractmethod
    async def get_memory(self) -> list[Msg]:
        """获取记忆中的所有消息
        
        Returns:
            消息列表
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """清空记忆"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """获取记忆中消息的数量"""
        pass
    
    def state_dict(self) -> dict:
        """获取记忆的状态字典，用于序列化"""
        raise NotImplementedError
    
    def load_state_dict(self, state_dict: dict) -> None:
        """从状态字典恢复记忆"""
        raise NotImplementedError


class InMemoryMemory(MemoryBase):
    """基于内存的简单记忆实现
    
    将消息存储在 Python 列表中，适用于：
    1. 短期对话场景
    2. 原型开发和测试
    3. 不需要持久化的场景
    
    Example:
        >>> memory = InMemoryMemory()
        >>> await memory.add(Msg(name="user", content="你好", role="user"))
        >>> msgs = await memory.get_memory()
        >>> print(len(msgs))  # 1
    """
    
    def __init__(self) -> None:
        """初始化记忆对象"""
        self.content: list[Msg] = []
    
    async def add(
        self,
        msg: Msg | list[Msg] | None,
        allow_duplicates: bool = False,
    ) -> None:
        """添加消息到记忆
        
        Args:
            msg: 要添加的消息
            allow_duplicates: 是否允许重复消息（基于消息 ID 判断）
        """
        if msg is None:
            return
        
        # 统一转换为列表
        if isinstance(msg, Msg):
            messages = [msg]
        else:
            messages = list(msg)
        
        # 检查重复
        if not allow_duplicates:
            existing_ids = {m.id for m in self.content}
            messages = [m for m in messages if m.id not in existing_ids]
        
        self.content.extend(messages)
    
    async def get_memory(self) -> list[Msg]:
        """获取所有记忆消息"""
        return self.content
    
    async def clear(self) -> None:
        """清空记忆"""
        self.content = []
    
    async def size(self) -> int:
        """获取消息数量"""
        return len(self.content)
    
    async def delete(self, index: int | list[int]) -> None:
        """删除指定索引的消息
        
        Args:
            index: 要删除的消息索引或索引列表
        """
        if isinstance(index, int):
            index = [index]
        
        # 从后往前删除，避免索引错位
        for idx in sorted(index, reverse=True):
            if 0 <= idx < len(self.content):
                self.content.pop(idx)
    
    def state_dict(self) -> dict:
        """获取状态字典用于序列化"""
        return {
            "content": [msg.to_dict() for msg in self.content]
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """从状态字典恢复"""
        self.content = [
            Msg.from_dict(data) for data in state_dict.get("content", [])
        ]


