# -*- coding: utf-8 -*-
"""
测试 Steering 模块
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.steering import (
    SteerableAgent,
    create_human_intervention_tool,
    create_confirmation_tool,
)
from nano_agentscope.message import Msg
from nano_agentscope.agent import AgentBase
from nano_agentscope.tool import ToolResponse


class MockAgent(AgentBase):
    """用于测试的模拟 Agent"""
    
    def __init__(self, name: str, delay: float = 0.1):
        self.name = name
        self.delay = delay
        self.reply_called = False
        self.interrupt_called = False
    
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """模拟回复，可以设置延迟"""
        self.reply_called = True
        # 模拟长时间运行
        await asyncio.sleep(self.delay)
        return Msg(
            name=self.name,
            content="正常回复",
            role="assistant"
        )
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        pass
    
    async def handle_interrupt(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """处理中断"""
        self.interrupt_called = True
        return Msg(
            name=self.name,
            content="已中断",
            role="assistant",
            metadata={"_is_interrupted": True}
        )


class TestSteerableAgent:
    """测试 SteerableAgent 类"""
    
    @pytest.mark.asyncio
    async def test_normal_execution(self):
        """测试正常执行（无中断）"""
        agent = MockAgent("TestAgent")
        steerable = SteerableAgent(agent)
        
        result = await steerable(Msg(name="user", content="测试", role="user"))
        
        assert agent.reply_called
        assert not agent.interrupt_called
        assert result.content == "正常回复"
    
    @pytest.mark.asyncio
    async def test_interrupt_execution(self):
        """测试中断执行"""
        agent = MockAgent("TestAgent", delay=1.0)  # 长延迟
        steerable = SteerableAgent(agent)
        
        async def run_agent():
            return await steerable(Msg(name="user", content="测试", role="user"))
        
        # 启动任务
        task = asyncio.create_task(run_agent())
        
        # 等待一小段时间让任务开始
        await asyncio.sleep(0.05)
        
        # 发送中断
        assert steerable.is_running
        interrupted = steerable.interrupt()
        assert interrupted
        
        # 等待任务完成
        result = await task
        
        assert agent.interrupt_called
        assert result.metadata.get("_is_interrupted") == True
    
    @pytest.mark.asyncio
    async def test_interrupt_when_not_running(self):
        """测试未运行时中断"""
        agent = MockAgent("TestAgent")
        steerable = SteerableAgent(agent)
        
        # 未运行时中断应返回 False
        assert not steerable.is_running
        assert not steerable.interrupt()
    
    @pytest.mark.asyncio
    async def test_name_property(self):
        """测试 name 属性"""
        agent = MockAgent("TestAgent")
        steerable = SteerableAgent(agent)
        
        assert steerable.name == "TestAgent"
    
    @pytest.mark.asyncio
    async def test_is_running_flag(self):
        """测试 is_running 标志"""
        agent = MockAgent("TestAgent", delay=0.5)
        steerable = SteerableAgent(agent)
        
        assert not steerable.is_running
        
        async def run_and_check():
            task = asyncio.create_task(
                steerable(Msg(name="user", content="测试", role="user"))
            )
            await asyncio.sleep(0.1)
            running = steerable.is_running
            await task
            return running
        
        was_running = await run_and_check()
        assert was_running
        assert not steerable.is_running


class TestHumanInterventionTool:
    """测试人工干预工具"""
    
    def test_create_tool(self):
        """测试创建工具"""
        tool = create_human_intervention_tool()
        
        assert tool.__name__ == "ask_human"
        assert callable(tool)
    
    def test_custom_name(self):
        """测试自定义工具名称"""
        tool = create_human_intervention_tool(
            tool_name="request_help",
            tool_description="自定义描述"
        )
        
        assert tool.__name__ == "request_help"
    
    @pytest.mark.asyncio
    async def test_tool_has_docstring(self):
        """测试工具有正确的 docstring"""
        tool = create_human_intervention_tool()
        
        assert tool.__doc__ is not None
        assert "question" in tool.__doc__


class TestConfirmationTool:
    """测试确认工具"""
    
    def test_create_tool(self):
        """测试创建工具"""
        tool = create_confirmation_tool()
        
        assert tool.__name__ == "confirm_action"
        assert callable(tool)
    
    def test_custom_name(self):
        """测试自定义工具名称"""
        tool = create_confirmation_tool(tool_name="verify_action")
        
        assert tool.__name__ == "verify_action"


class TestToolResponseInterrupted:
    """测试 ToolResponse 的 is_interrupted 字段"""
    
    def test_default_not_interrupted(self):
        """测试默认不是中断状态"""
        from nano_agentscope.message import TextBlock
        
        response = ToolResponse(
            content=[TextBlock(type="text", text="测试")]
        )
        
        assert response.is_interrupted == False
    
    def test_set_interrupted(self):
        """测试设置中断状态"""
        from nano_agentscope.message import TextBlock
        
        response = ToolResponse(
            content=[TextBlock(type="text", text="测试")],
            is_interrupted=True
        )
        
        assert response.is_interrupted == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
