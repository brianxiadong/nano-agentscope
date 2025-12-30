# -*- coding: utf-8 -*-
"""
测试 Pipeline 模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.pipeline import sequential_pipeline, loop_pipeline, MsgHub
from nano_agentscope.message import Msg
from nano_agentscope.agent import AgentBase


class MockAgent(AgentBase):
    """用于测试的模拟 Agent"""
    
    def __init__(self, name: str, response_prefix: str = "回复"):
        self.name = name
        self.response_prefix = response_prefix
        self.observed_messages: list[Msg] = []
        self.call_count = 0
    
    async def reply(self, msg: Msg | list[Msg] | None = None) -> Msg:
        """生成模拟回复"""
        self.call_count += 1
        
        # 提取输入内容
        if isinstance(msg, Msg):
            input_content = msg.get_text_content()
        elif isinstance(msg, list) and msg:
            input_content = msg[-1].get_text_content()
        else:
            input_content = "无输入"
        
        return Msg(
            name=self.name,
            content=f"{self.response_prefix}_{self.call_count}: 收到 '{input_content}'",
            role="assistant"
        )
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """观察消息"""
        if msg is None:
            return
        if isinstance(msg, list):
            self.observed_messages.extend(msg)
        else:
            self.observed_messages.append(msg)


class TestSequentialPipeline:
    """测试 sequential_pipeline 函数"""
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """测试顺序执行"""
        agent1 = MockAgent("Agent1", "A1")
        agent2 = MockAgent("Agent2", "A2")
        agent3 = MockAgent("Agent3", "A3")
        
        msg = Msg(name="user", content="你好", role="user")
        
        result = await sequential_pipeline(
            agents=[agent1, agent2, agent3],
            msg=msg
        )
        
        # 检查每个 agent 都被调用了一次
        assert agent1.call_count == 1
        assert agent2.call_count == 1
        assert agent3.call_count == 1
        
        # 检查最终结果来自最后一个 agent
        assert result.name == "Agent3"
    
    @pytest.mark.asyncio
    async def test_sequential_message_passing(self):
        """测试消息传递"""
        agent1 = MockAgent("Agent1", "A1")
        agent2 = MockAgent("Agent2", "A2")
        
        msg = Msg(name="user", content="初始消息", role="user")
        
        result = await sequential_pipeline(
            agents=[agent1, agent2],
            msg=msg
        )
        
        # Agent2 应该收到 Agent1 的回复
        assert "A1_1" in result.content
    
    @pytest.mark.asyncio
    async def test_empty_agents(self):
        """测试空 agent 列表"""
        msg = Msg(name="user", content="测试", role="user")
        result = await sequential_pipeline(agents=[], msg=msg)
        
        # 没有 agent 时返回原始消息
        assert result == msg
    
    @pytest.mark.asyncio
    async def test_single_agent(self):
        """测试单个 agent"""
        agent = MockAgent("Agent", "A")
        msg = Msg(name="user", content="测试", role="user")
        
        result = await sequential_pipeline(agents=[agent], msg=msg)
        
        assert agent.call_count == 1
        assert result.name == "Agent"


class TestLoopPipeline:
    """测试 loop_pipeline 函数"""
    
    @pytest.mark.asyncio
    async def test_loop_execution(self):
        """测试循环执行"""
        agent1 = MockAgent("Agent1", "A1")
        agent2 = MockAgent("Agent2", "A2")
        
        msg = Msg(name="user", content="开始", role="user")
        
        await loop_pipeline(
            agents=[agent1, agent2],
            msg=msg,
            max_rounds=3
        )
        
        # 每个 agent 应该被调用 3 次
        assert agent1.call_count == 3
        assert agent2.call_count == 3
    
    @pytest.mark.asyncio
    async def test_loop_single_round(self):
        """测试单轮循环"""
        agent = MockAgent("Agent", "A")
        msg = Msg(name="user", content="测试", role="user")
        
        await loop_pipeline(
            agents=[agent],
            msg=msg,
            max_rounds=1
        )
        
        assert agent.call_count == 1
    
    @pytest.mark.asyncio
    async def test_loop_returns_last_message(self):
        """测试返回最后一条消息"""
        agent1 = MockAgent("Agent1", "A1")
        agent2 = MockAgent("Agent2", "A2")
        
        msg = Msg(name="user", content="开始", role="user")
        
        result = await loop_pipeline(
            agents=[agent1, agent2],
            msg=msg,
            max_rounds=2
        )
        
        # 最后一条消息应该来自 Agent2
        assert result.name == "Agent2"


class TestMsgHub:
    """测试 MsgHub 类"""
    
    @pytest.mark.asyncio
    async def test_broadcast_announcement(self):
        """测试进入时广播公告"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        
        announcement = Msg(name="系统", content="欢迎", role="system")
        
        async with MsgHub(
            participants=[agent1, agent2],
            announcement=announcement
        ):
            pass  # 仅测试进入时的广播
        
        # 两个 agent 都应该收到公告
        assert len(agent1.observed_messages) == 1
        assert len(agent2.observed_messages) == 1
        assert agent1.observed_messages[0].content == "欢迎"
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """测试手动广播消息"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        
        async with MsgHub(participants=[agent1, agent2]) as hub:
            msg = Msg(name="用户", content="大家好", role="user")
            await hub.broadcast(msg)
        
        # 两个 agent 都应该收到消息
        assert len(agent1.observed_messages) == 1
        assert len(agent2.observed_messages) == 1
    
    @pytest.mark.asyncio
    async def test_add_participant(self):
        """测试动态添加参与者"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        
        async with MsgHub(participants=[agent1]) as hub:
            assert hub.size == 1
            
            hub.add(agent2)
            assert hub.size == 2
            
            # 广播应该发送给所有人
            await hub.broadcast(Msg(name="系统", content="测试", role="system"))
        
        assert len(agent1.observed_messages) == 1
        assert len(agent2.observed_messages) == 1
    
    @pytest.mark.asyncio
    async def test_remove_participant(self):
        """测试动态移除参与者"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        
        async with MsgHub(participants=[agent1, agent2]) as hub:
            assert hub.size == 2
            
            hub.remove(agent2)
            assert hub.size == 1
            
            # 广播只发送给剩余的人
            await hub.broadcast(Msg(name="系统", content="测试", role="system"))
        
        assert len(agent1.observed_messages) == 1
        assert len(agent2.observed_messages) == 0
    
    @pytest.mark.asyncio
    async def test_add_multiple_participants(self):
        """测试批量添加参与者"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        agent3 = MockAgent("Agent3")
        
        async with MsgHub(participants=[agent1]) as hub:
            hub.add([agent2, agent3])
            assert hub.size == 3
    
    @pytest.mark.asyncio
    async def test_remove_multiple_participants(self):
        """测试批量移除参与者"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        agent3 = MockAgent("Agent3")
        
        async with MsgHub(participants=[agent1, agent2, agent3]) as hub:
            hub.remove([agent1, agent2])
            assert hub.size == 1
            assert agent3 in hub.participants
    
    @pytest.mark.asyncio
    async def test_no_announcement(self):
        """测试无公告情况"""
        agent = MockAgent("Agent")
        
        async with MsgHub(participants=[agent]):
            pass
        
        # 没有公告时不应该收到消息
        assert len(agent.observed_messages) == 0
    
    @pytest.mark.asyncio
    async def test_add_duplicate_participant(self):
        """测试添加重复参与者"""
        agent = MockAgent("Agent")
        
        async with MsgHub(participants=[agent]) as hub:
            hub.add(agent)  # 尝试再次添加
            assert hub.size == 1  # 不应该重复添加


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
