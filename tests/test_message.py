# -*- coding: utf-8 -*-
"""
测试消息模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.message import (
    Msg,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


class TestMsg:
    """测试 Msg 类"""
    
    def test_create_text_msg(self):
        """测试创建文本消息"""
        msg = Msg(name="user", content="你好", role="user")
        
        assert msg.name == "user"
        assert msg.content == "你好"
        assert msg.role == "user"
        assert msg.id is not None
        assert msg.timestamp is not None
    
    def test_create_msg_with_blocks(self):
        """测试创建包含 ContentBlock 的消息"""
        blocks = [
            TextBlock(type="text", text="让我查一下"),
            ToolUseBlock(
                type="tool_use",
                id="call_1",
                name="get_weather",
                input={"city": "北京"},
            ),
        ]
        msg = Msg(name="assistant", content=blocks, role="assistant")
        
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "tool_use"
    
    def test_get_text_content_string(self):
        """测试从字符串内容获取文本"""
        msg = Msg(name="user", content="测试内容", role="user")
        assert msg.get_text_content() == "测试内容"
    
    def test_get_text_content_blocks(self):
        """测试从 TextBlock 获取文本"""
        blocks = [
            TextBlock(type="text", text="第一段"),
            TextBlock(type="text", text="第二段"),
        ]
        msg = Msg(name="user", content=blocks, role="user")
        
        assert msg.get_text_content() == "第一段\n第二段"
        assert msg.get_text_content(separator=" ") == "第一段 第二段"
    
    def test_get_content_blocks(self):
        """测试获取内容块"""
        blocks = [
            TextBlock(type="text", text="文本"),
            ToolUseBlock(type="tool_use", id="1", name="func", input={}),
        ]
        msg = Msg(name="assistant", content=blocks, role="assistant")
        
        # 获取所有块
        all_blocks = msg.get_content_blocks()
        assert len(all_blocks) == 2
        
        # 按类型筛选
        text_blocks = msg.get_content_blocks("text")
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "文本"
        
        tool_blocks = msg.get_content_blocks("tool_use")
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "func"
    
    def test_has_content_blocks(self):
        """测试检查是否包含特定类型的块"""
        blocks = [TextBlock(type="text", text="test")]
        msg = Msg(name="user", content=blocks, role="user")
        
        assert msg.has_content_blocks("text") is True
        assert msg.has_content_blocks("tool_use") is False
    
    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        original = Msg(
            name="user",
            content="测试",
            role="user",
            metadata={"key": "value"},
        )
        
        # 转换为字典
        data = original.to_dict()
        assert data["name"] == "user"
        assert data["content"] == "测试"
        assert data["metadata"] == {"key": "value"}
        
        # 从字典恢复
        restored = Msg.from_dict(data)
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.role == original.role
        assert restored.metadata == original.metadata


class TestContentBlocks:
    """测试 ContentBlock 类型"""
    
    def test_text_block(self):
        """测试 TextBlock"""
        block = TextBlock(type="text", text="Hello")
        assert block["type"] == "text"
        assert block["text"] == "Hello"
    
    def test_tool_use_block(self):
        """测试 ToolUseBlock"""
        block = ToolUseBlock(
            type="tool_use",
            id="call_123",
            name="my_function",
            input={"arg1": "value1"},
        )
        assert block["type"] == "tool_use"
        assert block["id"] == "call_123"
        assert block["name"] == "my_function"
        assert block["input"]["arg1"] == "value1"
    
    def test_tool_result_block(self):
        """测试 ToolResultBlock"""
        block = ToolResultBlock(
            type="tool_result",
            id="call_123",
            name="my_function",
            output="执行成功",
        )
        assert block["type"] == "tool_result"
        assert block["output"] == "执行成功"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


