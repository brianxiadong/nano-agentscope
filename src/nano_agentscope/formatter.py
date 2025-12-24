# -*- coding: utf-8 -*-
"""
格式化器模块 - 将 Msg 转换为 API 格式

本模块定义了格式化器的抽象和实现：
1. FormatterBase - 格式化器基类
2. OpenAIFormatter - OpenAI API 格式的实现

学习要点：
- 不同的 LLM API 有不同的消息格式要求
- 格式化器负责将统一的 Msg 格式转换为 API 特定格式
- 包括处理多模态内容、工具调用等

消息格式对比:

Nano-AgentScope Msg:
    Msg(name="user", content="你好", role="user")

OpenAI API 格式:
    {"role": "user", "name": "user", "content": "你好"}
"""

import json
from abc import abstractmethod
from typing import Any

from .message import Msg, TextBlock, ToolUseBlock, ToolResultBlock


class FormatterBase:
    """格式化器基类 - 定义格式化接口
    
    格式化器的职责：
    1. 将 Msg 对象列表转换为 API 要求的格式
    2. 处理不同类型的 ContentBlock
    3. 正确映射 role 和其他字段
    
    所有格式化器实现都应继承此类并实现 format 方法。
    """
    
    @abstractmethod
    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        """将消息列表格式化为 API 要求的格式
        
        Args:
            msgs: Msg 对象列表
            
        Returns:
            格式化后的消息字典列表
        """
        pass
    
    @staticmethod
    def _assert_msgs(msgs: list[Msg]) -> None:
        """验证输入是否为 Msg 列表"""
        if not isinstance(msgs, list):
            raise TypeError(f"msgs 必须是列表，但收到 {type(msgs)}")
        for msg in msgs:
            if not isinstance(msg, Msg):
                raise TypeError(f"列表元素必须是 Msg，但收到 {type(msg)}")


class OpenAIFormatter(FormatterBase):
    """OpenAI API 格式化器
    
    将 Msg 对象转换为 OpenAI Chat Completions API 要求的格式。
    
    支持的内容类型：
    - TextBlock -> {"type": "text", "text": "..."}
    - ToolUseBlock -> message.tool_calls
    - ToolResultBlock -> {"role": "tool", ...}
    - ImageBlock -> {"type": "image_url", ...}（简化版暂不实现）
    
    Example:
        >>> formatter = OpenAIFormatter()
        >>> msgs = [
        ...     Msg(name="system", content="你是助手", role="system"),
        ...     Msg(name="user", content="你好", role="user"),
        ... ]
        >>> formatted = await formatter.format(msgs)
        >>> print(formatted[0])
        # {"role": "system", "content": [{"type": "text", "text": "你是助手"}]}
    """
    
    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        """格式化消息列表为 OpenAI API 格式
        
        处理逻辑：
        1. 遍历每条消息的 ContentBlock
        2. 根据块类型转换为 API 格式
        3. 特殊处理工具调用和工具结果
        """
        self._assert_msgs(msgs)
        
        formatted_msgs = []
        
        for msg in msgs:
            content_blocks = []
            tool_calls = []
            
            # 处理每个 ContentBlock
            for block in msg.get_content_blocks():
                block_type = block.get("type")
                
                if block_type == "text":
                    # 文本块
                    content_blocks.append({
                        "type": "text",
                        "text": block["text"],
                    })
                
                elif block_type == "tool_use":
                    # 工具调用块 -> 转换为 OpenAI tool_calls 格式
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input", {}),
                                ensure_ascii=False,
                            ),
                        },
                    })
                
                elif block_type == "tool_result":
                    # 工具结果块 -> 单独的 tool 消息
                    output = block.get("output", "")
                    if isinstance(output, list):
                        # 如果输出是列表，提取文本
                        texts = [
                            b["text"] for b in output 
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        output = "\n".join(texts)
                    
                    formatted_msgs.append({
                        "role": "tool",
                        "tool_call_id": block["id"],
                        "name": block.get("name", ""),
                        "content": str(output),
                    })
                
                elif block_type == "image":
                    # 图片块（简化版）
                    if "url" in block:
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": block["url"]},
                        })
            
            # 构建 OpenAI 消息
            if content_blocks or tool_calls:
                openai_msg = {
                    "role": msg.role,
                    "name": msg.name,
                }
                
                if content_blocks:
                    openai_msg["content"] = content_blocks
                else:
                    openai_msg["content"] = None
                
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                
                formatted_msgs.append(openai_msg)
        
        return formatted_msgs


class SimpleFormatter(FormatterBase):
    """简单格式化器 - 仅处理文本内容
    
    这是一个教学用的简化实现，只处理纯文本消息，
    适合理解格式化器的基本概念。
    
    Example:
        >>> formatter = SimpleFormatter()
        >>> msgs = [Msg(name="user", content="你好", role="user")]
        >>> formatted = await formatter.format(msgs)
        >>> print(formatted[0])
        # {"role": "user", "content": "你好"}
    """
    
    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        """简单格式化 - 只提取文本内容"""
        self._assert_msgs(msgs)
        
        formatted_msgs = []
        
        for msg in msgs:
            text_content = msg.get_text_content() or ""
            
            formatted_msgs.append({
                "role": msg.role,
                "content": text_content,
            })
        
        return formatted_msgs


