# -*- coding: utf-8 -*-
"""
消息模块 - Nano-AgentScope 的核心消息结构

本模块定义了消息的基本结构，包括：
1. Msg - 消息类，是 Agent 之间通信的基本单位
2. ContentBlock - 内容块，支持文本、工具调用、图片等多种类型

学习要点：
- Msg 是智能体之间传递信息的载体
- role 字段标识消息来源（user/assistant/system）
- content 可以是字符串或 ContentBlock 列表
"""

from datetime import datetime
from typing import Literal, Sequence
from typing_extensions import TypedDict, Required
import uuid


# ============== Content Block 定义 ==============
# ContentBlock 使用 TypedDict 定义，是一种轻量级的类型定义方式
# 它们用于表示消息内容的不同部分

class TextBlock(TypedDict, total=False):
    """文本内容块
    
    Example:
        >>> block = TextBlock(type="text", text="你好，世界！")
    """
    type: Required[Literal["text"]]  # 必须是 "text"
    text: str  # 文本内容


class ToolUseBlock(TypedDict, total=False):
    """工具调用块 - 表示 LLM 想要调用某个工具
    
    Example:
        >>> block = ToolUseBlock(
        ...     type="tool_use",
        ...     id="call_123",
        ...     name="get_weather",
        ...     input={"city": "北京"}
        ... )
    """
    type: Required[Literal["tool_use"]]
    id: Required[str]  # 调用的唯一标识
    name: Required[str]  # 工具函数名
    input: Required[dict[str, object]]  # 调用参数


class ToolResultBlock(TypedDict, total=False):
    """工具结果块 - 表示工具执行的返回结果
    
    Example:
        >>> block = ToolResultBlock(
        ...     type="tool_result",
        ...     id="call_123",
        ...     name="get_weather",
        ...     output="北京今天晴天，25度"
        ... )
    """
    type: Required[Literal["tool_result"]]
    id: Required[str]  # 对应 ToolUseBlock 的 id
    name: Required[str]  # 工具函数名
    output: Required[str | list]  # 执行结果


class ImageBlock(TypedDict, total=False):
    """图片内容块（简化版，仅支持 URL）
    
    Example:
        >>> block = ImageBlock(
        ...     type="image",
        ...     url="https://example.com/image.png"
        ... )
    """
    type: Required[Literal["image"]]
    url: str  # 图片 URL


# 所有支持的内容块类型
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock


# ============== Msg 消息类 ==============

class Msg:
    """消息类 - Agent 之间通信的基本单位
    
    Msg 是 AgentScope 中最核心的数据结构之一，用于在智能体之间传递信息。
    
    核心属性：
        - name: 发送者名称
        - content: 消息内容，可以是字符串或 ContentBlock 列表
        - role: 角色类型（user/assistant/system）
        - metadata: 附加元数据，如结构化输出
    
    Example:
        >>> # 简单文本消息
        >>> msg = Msg(name="user", content="你好", role="user")
        
        >>> # 包含工具调用的消息
        >>> msg = Msg(
        ...     name="assistant",
        ...     content=[
        ...         TextBlock(type="text", text="让我查一下天气"),
        ...         ToolUseBlock(type="tool_use", id="1", name="get_weather", input={"city": "北京"})
        ...     ],
        ...     role="assistant"
        ... )
    """
    
    def __init__(
        self,
        name: str,
        content: str | Sequence[ContentBlock],
        role: Literal["user", "assistant", "system"],
        metadata: dict | None = None,
        timestamp: str | None = None,
    ) -> None:
        """初始化消息对象
        
        Args:
            name: 发送者名称，如 "user", "assistant", "小助手" 等
            content: 消息内容
            role: 角色类型
                - "user": 用户消息
                - "assistant": 助手（LLM）消息
                - "system": 系统消息（如工具结果）
            metadata: 可选的元数据，常用于存储结构化输出
            timestamp: 时间戳，不提供则自动生成
        """
        self.name = name
        self.content = content
        self.role = role
        self.metadata = metadata
        
        # 自动生成 ID 和时间戳
        self.id = str(uuid.uuid4())[:8]
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_text_content(self, separator: str = "\n") -> str | None:
        """获取消息中的纯文本内容
        
        如果 content 是字符串，直接返回。
        如果 content 是 ContentBlock 列表，则提取所有 TextBlock 并拼接。
        
        Args:
            separator: 多个文本块之间的分隔符
            
        Returns:
            拼接后的文本内容，如果没有文本则返回 None
        """
        if isinstance(self.content, str):
            return self.content
        
        texts = []
        for block in self.content:
            if block.get("type") == "text":
                texts.append(block["text"])
        
        return separator.join(texts) if texts else None
    
    def get_content_blocks(
        self,
        block_type: Literal["text", "tool_use", "tool_result", "image"] | None = None,
    ) -> Sequence[ContentBlock]:
        """获取指定类型的内容块
        
        Args:
            block_type: 要筛选的块类型，None 表示获取所有块
            
        Returns:
            ContentBlock 列表
        """
        # 如果 content 是字符串，转换为 TextBlock
        if isinstance(self.content, str):
            blocks = [TextBlock(type="text", text=self.content)]
        else:
            blocks = list(self.content) if self.content else []
        
        # 按类型筛选
        if block_type:
            blocks = [b for b in blocks if b.get("type") == block_type]
        
        return blocks
    
    def has_content_blocks(
        self,
        block_type: Literal["text", "tool_use", "tool_result", "image"] | None = None,
    ) -> bool:
        """检查消息是否包含指定类型的内容块"""
        return len(self.get_content_blocks(block_type)) > 0
    
    def to_dict(self) -> dict:
        """将消息转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "role": self.role,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Msg":
        """从字典创建消息对象"""
        msg = cls(
            name=data["name"],
            content=data["content"],
            role=data["role"],
            metadata=data.get("metadata"),
            timestamp=data.get("timestamp"),
        )
        msg.id = data.get("id", msg.id)
        return msg
    
    def __repr__(self) -> str:
        content_preview = (
            self.content[:50] + "..." 
            if isinstance(self.content, str) and len(self.content) > 50 
            else str(self.content)[:50] + "..."
        )
        return f"Msg(name='{self.name}', role='{self.role}', content={content_preview})"


