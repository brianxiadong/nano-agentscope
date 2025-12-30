# -*- coding: utf-8 -*-
"""
示例 05：多智能体协同对话

本示例演示如何使用 MsgHub 和 Pipeline 实现多智能体协同：
1. MsgHub - 消息广播中心，实现"群聊"模式
2. sequential_pipeline - 顺序执行多个 Agent
3. loop_pipeline - 循环执行多轮讨论

教学目标：
1. 理解多智能体系统的消息传递机制
2. 学习如何编排多个 Agent 的执行顺序
3. 观察 Agent 之间如何协作完成任务

运行方式：
    export DASHSCOPE_API_KEY=your_key  # 或 OPENAI_API_KEY
    python examples/05_multi_agent_conversation.py
"""

import asyncio
import os

from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,
    OpenAIChatModel,
    OpenAIFormatter,
    Toolkit,
    InMemoryMemory,
    Msg,
    MsgHub,
    sequential_pipeline,
    loop_pipeline,
)


def create_model():
    """根据环境变量选择模型"""
    if os.environ.get("DASHSCOPE_API_KEY"):
        return DashScopeChatModel(model_name="qwen-max")
    elif os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatModel(model_name="gpt-4o-mini")
    else:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")


async def demo_sequential_pipeline():
    """演示顺序执行管道"""
    print("\n" + "=" * 60)
    print("📋 Demo 1: Sequential Pipeline (顺序执行)")
    print("=" * 60)
    print("场景：任务分解 - 分析师 -> 规划师 -> 执行者")
    
    # 创建三个不同角色的 Agent
    analyst = ReActAgent(
        name="分析师",
        sys_prompt="""你是一个任务分析师。
收到任务后，分析任务的关键点和难点，列出需要注意的事项。
回复要简洁，不超过100字。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    planner = ReActAgent(
        name="规划师",
        sys_prompt="""你是一个任务规划师。
根据分析师的分析结果，制定具体的执行步骤。
回复要简洁，列出3-5个步骤即可。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    executor = ReActAgent(
        name="执行者",
        sys_prompt="""你是一个任务执行者。
根据规划师的计划，总结最终的执行方案。
回复要简洁，给出最终建议。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 使用顺序管道执行
    task = Msg(
        name="用户",
        content="请帮我制定一个学习 Python 的计划",
        role="user"
    )
    
    print(f"\n📝 用户任务: {task.content}")
    print("-" * 40)
    
    result = await sequential_pipeline(
        agents=[analyst, planner, executor],
        msg=task
    )
    
    print(f"\n✅ 最终结果: {result.get_text_content()}")


async def demo_loop_pipeline():
    """演示循环执行管道"""
    print("\n" + "=" * 60)
    print("🔄 Demo 2: Loop Pipeline (循环讨论)")
    print("=" * 60)
    print("场景：辩论赛 - 正方 vs 反方，进行2轮辩论")
    
    # 创建辩论双方
    pro_side = ReActAgent(
        name="正方",
        sys_prompt="""你是一场辩论赛的正方辩手。
辩题是：AI 技术的发展对人类社会利大于弊。
你支持这个观点，每次发言要简洁有力，不超过80字。
注意回应对方的论点。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    con_side = ReActAgent(
        name="反方",
        sys_prompt="""你是一场辩论赛的反方辩手。
辩题是：AI 技术的发展对人类社会利大于弊。
你反对这个观点，每次发言要简洁有力，不超过80字。
注意回应对方的论点。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 开场词
    opening = Msg(
        name="主持人",
        content="辩论开始！正方先发言，请论述你们的观点。",
        role="user"
    )
    
    print(f"\n🎤 主持人: {opening.content}")
    print("-" * 40)
    
    # 进行2轮辩论
    await loop_pipeline(
        agents=[pro_side, con_side],
        msg=opening,
        max_rounds=2
    )
    
    print("\n✅ 辩论结束！")


async def demo_msghub():
    """演示消息广播中心"""
    print("\n" + "=" * 60)
    print("📢 Demo 3: MsgHub (消息广播)")
    print("=" * 60)
    print("场景：技术讨论会 - 主持人发布话题，三位专家分别发表看法")
    
    # 创建主持人和专家
    moderator = ReActAgent(
        name="主持人",
        sys_prompt="""你是一场技术讨论会的主持人。
负责引导讨论，总结各方观点。
回复简洁，不超过50字。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    expert_a = ReActAgent(
        name="专家A",
        sys_prompt="""你是一位AI技术专家，专注于技术实现层面。
讨论时从技术角度发表看法，回复简洁，不超过60字。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    expert_b = ReActAgent(
        name="专家B",
        sys_prompt="""你是一位产品经理，专注于用户体验和商业价值。
讨论时从产品角度发表看法，回复简洁，不超过60字。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    expert_c = ReActAgent(
        name="专家C",
        sys_prompt="""你是一位伦理学者，关注技术对社会的影响。
讨论时从伦理和社会影响角度发表看法，回复简洁，不超过60字。""",
        model=create_model(),
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 讨论话题
    topic = Msg(
        name="主持人",
        content="今天我们讨论的话题是：大语言模型是否应该开源？请各位专家发表看法。",
        role="assistant"
    )
    
    print(f"\n🎤 {topic.name}: {topic.content}")
    print("-" * 40)
    
    # 使用 MsgHub 广播消息给所有参与者
    async with MsgHub(
        participants=[expert_a, expert_b, expert_c],
        announcement=topic  # 进入时广播话题给所有人
    ) as hub:
        # 所有专家现在都"看到"了话题
        # 依次让每位专家发言
        experts = [expert_a, expert_b, expert_c]
        
        for expert in experts:
            # 让专家发言（他们的 memory 中已经有话题了）
            response = await expert(None)  # 不需要传入消息，因为已经通过 observe 看到了
            print(f"\n💬 {response.name}: {response.get_text_content()}")
            
            # 将发言广播给其他专家
            await hub.broadcast(response)
    
    print("\n✅ 讨论结束！")


async def main():
    print("=" * 60)
    print("🤝 Nano-AgentScope 示例：多智能体协同对话")
    print("=" * 60)
    
    try:
        # 检查 API Key
        create_model()
    except ValueError as e:
        print(f"\n⚠️ {e}")
        return
    
    # Demo 1: 顺序执行
    await demo_sequential_pipeline()
    
    # Demo 2: 循环讨论
    await demo_loop_pipeline()
    
    # Demo 3: 消息广播
    await demo_msghub()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例完成！")
    print("=" * 60)
    print("\n💡 教学要点：")
    print("  1. sequential_pipeline: 链式传递，适合任务分解场景")
    print("  2. loop_pipeline: 循环执行，适合多轮讨论/迭代场景")
    print("  3. MsgHub: 消息广播，让所有参与者共享信息")


if __name__ == "__main__":
    asyncio.run(main())
