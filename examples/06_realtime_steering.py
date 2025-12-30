# -*- coding: utf-8 -*-
"""
示例 06：实时干预 (中断与人工协助)

本示例演示 Nano-AgentScope 的实时干预功能：
1. SteerableAgent - 支持中断的 Agent 封装
2. create_human_intervention_tool - 人工干预工具
3. create_confirmation_tool - 确认工具

教学目标：
1. 理解 Agent 执行的可控性
2. 学习如何实现人机协作
3. 观察中断机制的工作原理

运行方式：
    export DASHSCOPE_API_KEY=your_key  # 或 OPENAI_API_KEY
    python examples/06_realtime_steering.py
"""

import asyncio
import os
import signal

from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,
    OpenAIChatModel,
    OpenAIFormatter,
    Toolkit,
    InMemoryMemory,
    Msg,
    SteerableAgent,
    create_human_intervention_tool,
    create_confirmation_tool,
)


async def demo_human_intervention():
    """演示人工干预工具的使用"""
    print("\n" + "=" * 60)
    print("🙋 Demo 1: 人工干预 (Human Intervention)")
    print("=" * 60)
    print("场景：Agent 在执行任务时请求人类帮助")
    
    # 选择模型
    if os.environ.get("DASHSCOPE_API_KEY"):
        model = DashScopeChatModel(model_name="qwen-max")
    elif os.environ.get("OPENAI_API_KEY"):
        model = OpenAIChatModel(model_name="gpt-4o-mini")
    else:
        print("⚠️ 未设置 API Key，跳过此演示")
        return
    
    # 创建工具集
    toolkit = Toolkit()
    
    # 注册人工干预工具
    ask_human = create_human_intervention_tool(
        prompt="您的回复: ",
    )
    toolkit.register_tool_function(ask_human)
    
    # 创建 Agent
    agent = ReActAgent(
        name="助手",
        sys_prompt="""你是一个谨慎的助手。
当遇到不确定的问题或需要用户确认时，使用 ask_human 工具询问用户。
例如：用户的偏好、敏感操作确认等。""",
        model=model,
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    
    # 测试对话
    question = "我想订一张明天去上海的机票，请帮我安排"
    print(f"\n📝 用户: {question}")
    print("-" * 40)
    
    response = await agent(Msg(name="user", content=question, role="user"))
    print(f"\n✅ 最终回复: {response.get_text_content()}")


async def demo_confirmation_tool():
    """演示确认工具的使用"""
    print("\n" + "=" * 60)
    print("⚠️  Demo 2: 操作确认 (Confirmation)")
    print("=" * 60)
    print("场景：Agent 执行敏感操作前请求确认")
    
    # 选择模型
    if os.environ.get("DASHSCOPE_API_KEY"):
        model = DashScopeChatModel(model_name="qwen-max")
    elif os.environ.get("OPENAI_API_KEY"):
        model = OpenAIChatModel(model_name="gpt-4o-mini")
    else:
        print("⚠️ 未设置 API Key，跳过此演示")
        return
    
    # 创建工具集
    toolkit = Toolkit()
    
    # 注册确认工具
    confirm = create_confirmation_tool()
    toolkit.register_tool_function(confirm)
    
    # 创建 Agent
    agent = ReActAgent(
        name="文件助手",
        sys_prompt="""你是一个文件管理助手。
当用户要求执行危险操作（如删除、覆盖）时，务必使用 confirm_action 工具请求确认。
只有在用户确认后才能继续执行。""",
        model=model,
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    
    # 测试对话
    request = "请删除 /tmp/test.txt 文件"
    print(f"\n📝 用户: {request}")
    print("-" * 40)
    
    response = await agent(Msg(name="user", content=request, role="user"))
    print(f"\n✅ 最终回复: {response.get_text_content()}")


async def demo_steerable_agent():
    """演示可中断的 Agent"""
    print("\n" + "=" * 60)
    print("🛑 Demo 3: 可中断执行 (Steerable Agent)")
    print("=" * 60)
    print("场景：用户可以随时中断 Agent 的执行")
    print("提示：程序将启动一个长任务，3秒后自动中断")
    
    # 选择模型
    if os.environ.get("DASHSCOPE_API_KEY"):
        model = DashScopeChatModel(model_name="qwen-max")
    elif os.environ.get("OPENAI_API_KEY"):
        model = OpenAIChatModel(model_name="gpt-4o-mini")
    else:
        print("⚠️ 未设置 API Key，跳过此演示")
        return
    
    # 创建 Agent 并封装为可中断版本
    agent = ReActAgent(
        name="研究助手",
        sys_prompt="你是一个研究助手，会详细分析问题并给出深入的回答。",
        model=model,
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    steerable = SteerableAgent(agent)
    
    # 创建一个会被中断的任务
    async def long_task():
        request = "请详细分析人工智能的发展历史、现状和未来趋势"
        print(f"\n📝 用户: {request}")
        print("-" * 40)
        return await steerable(Msg(name="user", content=request, role="user"))
    
    # 启动任务
    task = asyncio.create_task(long_task())
    
    # 3秒后中断
    await asyncio.sleep(3)
    if steerable.is_running:
        print("\n\n⏹️  发送中断信号...")
        steerable.interrupt()
    
    # 等待任务完成
    try:
        response = await task
        print(f"\n✅ 结果: {response.get_text_content()}")
    except asyncio.CancelledError:
        print("\n任务已取消")


async def main():
    print("=" * 60)
    print("🎮 Nano-AgentScope 示例：实时干预")
    print("=" * 60)
    
    # 检查 API Key
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("\n⚠️ 请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        return
    
    # Demo 1: 人工干预
    await demo_human_intervention()
    
    # Demo 2: 确认工具
    await demo_confirmation_tool()
    
    # Demo 3: 可中断执行
    await demo_steerable_agent()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("=" * 60)
    print("\n💡 教学要点：")
    print("  1. create_human_intervention_tool: 让 Agent 请求人类帮助")
    print("  2. create_confirmation_tool: 危险操作前的确认机制")
    print("  3. SteerableAgent: 支持随时中断的 Agent 封装")


if __name__ == "__main__":
    asyncio.run(main())
