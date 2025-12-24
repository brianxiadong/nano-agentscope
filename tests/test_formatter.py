# -*- coding: utf-8 -*-
"""
测试格式化器模块
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.formatter import OpenAIFormatter, SimpleFormatter
from nano_agentscope.message import (
    Msg,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


class TestOpenAIFormatter:
    """测试 OpenAI 格式化器"""
    
    @pytest.fixture
    def formatter(self):
        return OpenAIFormatter()
    
    @pytest.mark.asyncio
    async def test_format_text_msg(self, formatter):
        """测试格式化文本消息"""
        msgs = [
            Msg(name="user", content="你好", role="user")
        ]
        
        result = await formatter.format(msgs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["name"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "你好"
    
    @pytest.mark.asyncio
    async def test_format_system_msg(self, formatter):
        """测试格式化系统消息"""
        msgs = [
            Msg(name="system", content="你是助手", role="system")
        ]
        
        result = await formatter.format(msgs)
        
        assert result[0]["role"] == "system"
    
    @pytest.mark.asyncio
    async def test_format_multiple_msgs(self, formatter):
        """测试格式化多条消息"""
        msgs = [
            Msg(name="system", content="系统提示", role="system"),
            Msg(name="user", content="用户消息", role="user"),
            Msg(name="assistant", content="助手回复", role="assistant"),
        ]
        
        result = await formatter.format(msgs)
        
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_format_tool_use_block(self, formatter):
        """测试格式化工具调用块"""
        msgs = [
            Msg(
                name="assistant",
                content=[
                    TextBlock(type="text", text="让我查一下"),
                    ToolUseBlock(
                        type="tool_use",
                        id="call_123",
                        name="get_weather",
                        input={"city": "北京"},
                    ),
                ],
                role="assistant",
            )
        ]
        
        result = await formatter.format(msgs)
        
        assert len(result) == 1
        msg = result[0]
        
        # 检查文本内容
        assert msg["content"][0]["text"] == "让我查一下"
        
        # 检查工具调用
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tool_call = msg["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"city": "北京"}
    
    @pytest.mark.asyncio
    async def test_format_tool_result_block(self, formatter):
        """测试格式化工具结果块"""
        msgs = [
            Msg(
                name="system",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id="call_123",
                        name="get_weather",
                        output="北京今天晴天",
                    ),
                ],
                role="system",
            )
        ]
        
        result = await formatter.format(msgs)
        
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "北京今天晴天"
    
    @pytest.mark.asyncio
    async def test_format_invalid_input(self, formatter):
        """测试无效输入"""
        with pytest.raises(TypeError):
            await formatter.format("not a list")
        
        with pytest.raises(TypeError):
            await formatter.format([{"not": "a msg"}])


class TestSimpleFormatter:
    """测试简单格式化器"""
    
    @pytest.fixture
    def formatter(self):
        return SimpleFormatter()
    
    @pytest.mark.asyncio
    async def test_format_text(self, formatter):
        """测试简单文本格式化"""
        msgs = [
            Msg(name="user", content="测试", role="user")
        ]
        
        result = await formatter.format(msgs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "测试"
    
    @pytest.mark.asyncio
    async def test_format_blocks_to_text(self, formatter):
        """测试将 ContentBlock 转换为纯文本"""
        msgs = [
            Msg(
                name="user",
                content=[
                    TextBlock(type="text", text="第一段"),
                    TextBlock(type="text", text="第二段"),
                ],
                role="user",
            )
        ]
        
        result = await formatter.format(msgs)
        
        # SimpleFormatter 只提取文本
        assert "第一段" in result[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


