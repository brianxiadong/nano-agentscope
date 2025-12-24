# -*- coding: utf-8 -*-
"""
测试记忆模块
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.memory import InMemoryMemory
from nano_agentscope.message import Msg


class TestInMemoryMemory:
    """测试 InMemoryMemory 类"""
    
    @pytest.fixture
    def memory(self):
        """创建测试用的 memory 实例"""
        return InMemoryMemory()
    
    @pytest.mark.asyncio
    async def test_add_single_msg(self, memory):
        """测试添加单条消息"""
        msg = Msg(name="user", content="测试", role="user")
        await memory.add(msg)
        
        assert await memory.size() == 1
        msgs = await memory.get_memory()
        assert msgs[0].content == "测试"
    
    @pytest.mark.asyncio
    async def test_add_multiple_msgs(self, memory):
        """测试添加多条消息"""
        msgs = [
            Msg(name="user", content="消息1", role="user"),
            Msg(name="assistant", content="消息2", role="assistant"),
        ]
        await memory.add(msgs)
        
        assert await memory.size() == 2
    
    @pytest.mark.asyncio
    async def test_add_none(self, memory):
        """测试添加 None"""
        await memory.add(None)
        assert await memory.size() == 0
    
    @pytest.mark.asyncio
    async def test_no_duplicates(self, memory):
        """测试不允许重复消息"""
        msg = Msg(name="user", content="测试", role="user")
        
        await memory.add(msg)
        await memory.add(msg)  # 同一消息再次添加
        
        assert await memory.size() == 1  # 应该还是只有一条
    
    @pytest.mark.asyncio
    async def test_allow_duplicates(self, memory):
        """测试允许重复消息"""
        msg = Msg(name="user", content="测试", role="user")
        
        await memory.add(msg, allow_duplicates=True)
        await memory.add(msg, allow_duplicates=True)
        
        assert await memory.size() == 2
    
    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """测试清空记忆"""
        await memory.add(Msg(name="user", content="测试", role="user"))
        assert await memory.size() == 1
        
        await memory.clear()
        assert await memory.size() == 0
    
    @pytest.mark.asyncio
    async def test_delete_single(self, memory):
        """测试删除单条消息"""
        await memory.add(Msg(name="user", content="消息0", role="user"))
        await memory.add(Msg(name="user", content="消息1", role="user"))
        await memory.add(Msg(name="user", content="消息2", role="user"))
        
        await memory.delete(1)  # 删除索引 1
        
        assert await memory.size() == 2
        msgs = await memory.get_memory()
        assert msgs[0].content == "消息0"
        assert msgs[1].content == "消息2"
    
    @pytest.mark.asyncio
    async def test_delete_multiple(self, memory):
        """测试删除多条消息"""
        for i in range(5):
            await memory.add(Msg(name="user", content=f"消息{i}", role="user"))
        
        await memory.delete([1, 3])  # 删除索引 1 和 3
        
        assert await memory.size() == 3
        msgs = await memory.get_memory()
        contents = [m.content for m in msgs]
        assert contents == ["消息0", "消息2", "消息4"]
    
    @pytest.mark.asyncio
    async def test_state_dict(self, memory):
        """测试状态序列化"""
        await memory.add(Msg(name="user", content="测试", role="user"))
        
        state = memory.state_dict()
        
        assert "content" in state
        assert len(state["content"]) == 1
        assert state["content"][0]["content"] == "测试"
    
    @pytest.mark.asyncio
    async def test_load_state_dict(self, memory):
        """测试状态恢复"""
        state = {
            "content": [
                {"name": "user", "content": "恢复的消息", "role": "user"}
            ]
        }
        
        memory.load_state_dict(state)
        
        assert await memory.size() == 1
        msgs = await memory.get_memory()
        assert msgs[0].content == "恢复的消息"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


