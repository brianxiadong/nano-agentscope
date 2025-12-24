# -*- coding: utf-8 -*-
"""
测试工具模块
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.tool import (
    Toolkit,
    ToolResponse,
    _parse_function_to_schema,
)
from nano_agentscope.message import TextBlock, ToolUseBlock


# 测试用的工具函数
def simple_func() -> ToolResponse:
    """简单函数"""
    return ToolResponse(content=[TextBlock(type="text", text="OK")])


def func_with_args(name: str, count: int = 1) -> ToolResponse:
    """带参数的函数
    
    Args:
        name: 名称
        count: 数量
    """
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Hello {name} x {count}")]
    )


async def async_func(value: str) -> ToolResponse:
    """异步函数
    
    Args:
        value: 输入值
    """
    await asyncio.sleep(0.01)
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Async: {value}")]
    )


class TestParseFunction:
    """测试函数解析"""
    
    def test_parse_simple_func(self):
        """测试解析简单函数"""
        schema = _parse_function_to_schema(simple_func)
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "simple_func"
        assert "description" in schema["function"]
    
    def test_parse_func_with_args(self):
        """测试解析带参数的函数"""
        schema = _parse_function_to_schema(func_with_args)
        
        func_schema = schema["function"]
        assert func_schema["name"] == "func_with_args"
        
        params = func_schema["parameters"]
        assert "properties" in params
        assert "name" in params["properties"]
        assert "count" in params["properties"]
        
        # 必需参数
        assert "name" in params.get("required", [])
        # 可选参数不应该在 required 中
        # (count 有默认值)


class TestToolkit:
    """测试 Toolkit 类"""
    
    @pytest.fixture
    def toolkit(self):
        """创建测试用的 toolkit"""
        return Toolkit()
    
    def test_register_function(self, toolkit):
        """测试注册函数"""
        toolkit.register_tool_function(simple_func)
        
        assert "simple_func" in toolkit.tools
        assert len(toolkit.get_json_schemas()) == 1
    
    def test_register_with_description(self, toolkit):
        """测试自定义描述"""
        toolkit.register_tool_function(simple_func, description="自定义描述")
        
        schema = toolkit.get_json_schemas()[0]
        assert schema["function"]["description"] == "自定义描述"
    
    def test_remove_function(self, toolkit):
        """测试移除函数"""
        toolkit.register_tool_function(simple_func)
        assert "simple_func" in toolkit.tools
        
        toolkit.remove_tool_function("simple_func")
        assert "simple_func" not in toolkit.tools
    
    def test_clear(self, toolkit):
        """测试清空工具集"""
        toolkit.register_tool_function(simple_func)
        toolkit.register_tool_function(func_with_args)
        assert len(toolkit.tools) == 2
        
        toolkit.clear()
        assert len(toolkit.tools) == 0
    
    @pytest.mark.asyncio
    async def test_call_sync_function(self, toolkit):
        """测试调用同步函数"""
        toolkit.register_tool_function(func_with_args)
        
        tool_call = ToolUseBlock(
            type="tool_use",
            id="test_1",
            name="func_with_args",
            input={"name": "World", "count": 3},
        )
        
        result = await toolkit.call_tool_function(tool_call)
        
        assert isinstance(result, ToolResponse)
        assert len(result.content) > 0
        assert "Hello World x 3" in result.content[0]["text"]
    
    @pytest.mark.asyncio
    async def test_call_async_function(self, toolkit):
        """测试调用异步函数"""
        toolkit.register_tool_function(async_func)
        
        tool_call = ToolUseBlock(
            type="tool_use",
            id="test_2",
            name="async_func",
            input={"value": "test"},
        )
        
        result = await toolkit.call_tool_function(tool_call)
        
        assert "Async: test" in result.content[0]["text"]
    
    @pytest.mark.asyncio
    async def test_call_nonexistent_function(self, toolkit):
        """测试调用不存在的函数"""
        tool_call = ToolUseBlock(
            type="tool_use",
            id="test_3",
            name="nonexistent",
            input={},
        )
        
        result = await toolkit.call_tool_function(tool_call)
        
        assert "Error" in result.content[0]["text"]
    
    @pytest.mark.asyncio
    async def test_call_function_with_error(self, toolkit):
        """测试函数执行出错"""
        def error_func() -> ToolResponse:
            raise ValueError("测试错误")
        
        toolkit.register_tool_function(error_func)
        
        tool_call = ToolUseBlock(
            type="tool_use",
            id="test_4",
            name="error_func",
            input={},
        )
        
        result = await toolkit.call_tool_function(tool_call)
        
        assert "Error" in result.content[0]["text"]
        assert "测试错误" in result.content[0]["text"]


class TestToolResponse:
    """测试 ToolResponse"""
    
    def test_create_response(self):
        """测试创建响应"""
        response = ToolResponse(
            content=[TextBlock(type="text", text="结果")],
            metadata={"key": "value"},
        )
        
        assert len(response.content) == 1
        assert response.metadata["key"] == "value"
        assert response.is_last is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


