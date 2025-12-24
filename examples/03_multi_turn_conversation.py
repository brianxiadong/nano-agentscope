#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例 3: 多轮对话 - 理解记忆系统

这个示例展示了如何：
1. 进行多轮对话
2. 理解 Memory 如何维护上下文
3. 查看和管理对话历史

记忆的作用：
- 存储对话历史
- 为 LLM 提供上下文
- 让智能体"记住"之前说过的话

运行方式:
    # 使用 DashScope（推荐）
    export DASHSCOPE_API_KEY="sk-xxx"
    python 03_multi_turn_conversation.py
    
    # 或使用 OpenAI
    export OPENAI_API_KEY="sk-xxx"
    python 03_multi_turn_conversation.py --openai
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope import (
    ReActAgent,
    UserAgent,
    DashScopeChatModel,
    OpenAIChatModel,
    OpenAIFormatter,
    InMemoryMemory,
    Msg,
)


def create_model(use_openai: bool = False):
    """创建 LLM 模型"""
    if use_openai:
        return OpenAIChatModel(model_name="gpt-4o-mini", stream=True)
    else:
        return DashScopeChatModel(model_name="qwen-max", stream=True)


async def main(use_openai: bool = False):
    """交互式多轮对话"""
    
    model = create_model(use_openai)
    print(f"使用模型: {model.model_name}")
    
    # 创建智能体
    agent = ReActAgent(
        name="记忆助手",
        sys_prompt="""你是一个具有良好记忆力的 AI 助手。
请记住用户告诉你的信息，并在后续对话中使用这些信息。
回答要简洁友好。""",
        model=model,
        formatter=OpenAIFormatter(),
        memory=InMemoryMemory(),
    )
    
    # 创建用户智能体
    user = UserAgent("用户")
    
    print("=" * 50)
    print("多轮对话示例 - 输入 'exit' 退出")
    print("输入 'memory' 查看当前记忆")
    print("输入 'clear' 清空记忆")
    print("=" * 50)
    
    msg = None
    while True:
        # 获取用户输入
        msg = await user(msg)
        
        user_text = msg.get_text_content()
        
        # 特殊命令处理
        if user_text.lower() == "exit":
            print("再见！")
            break
        
        if user_text.lower() == "memory":
            # 显示当前记忆
            print("\n" + "=" * 30 + " 当前记忆 " + "=" * 30)
            memories = await agent.memory.get_memory()
            for i, mem in enumerate(memories):
                text = mem.get_text_content() or str(mem.content)[:100]
                print(f"[{i}] {mem.role}/{mem.name}: {text[:50]}...")
            print("=" * 70 + "\n")
            msg = None
            continue
        
        if user_text.lower() == "clear":
            await agent.memory.clear()
            print("记忆已清空！\n")
            msg = None
            continue
        
        # 正常对话
        msg = await agent(msg)


async def demo_memory_context(use_openai: bool = False):
    """演示记忆如何提供上下文"""
    
    print("\n" + "=" * 50)
    print("演示：记忆如何帮助智能体理解上下文")
    print("=" * 50)
    
    model = create_model(use_openai)
    model.stream = False  # 演示用非流式
    print(f"使用模型: {model.model_name}")
    
    agent = ReActAgent(
        name="助手",
        sys_prompt="你是一个友好的助手，请记住用户的信息。",
        model=model,
        formatter=OpenAIFormatter(),
    )
    
    # 第一轮：介绍信息
    print("\n--- 第一轮对话 ---")
    await agent(Msg(name="user", content="我叫张三，今年25岁，是一名程序员。", role="user"))
    
    # 第二轮：基于上下文的问题
    print("\n--- 第二轮对话 ---")
    await agent(Msg(name="user", content="我叫什么名字？做什么工作？", role="user"))
    
    # 显示记忆状态
    print("\n--- 记忆状态 ---")
    memories = await agent.memory.get_memory()
    print(f"共有 {len(memories)} 条记忆")
    for mem in memories:
        role = "👤" if mem.role == "user" else "🤖"
        text = mem.get_text_content() or "[非文本内容]"
        print(f"  {role} {mem.name}: {text[:60]}...")


if __name__ == "__main__":
    use_openai = "--openai" in sys.argv
    
    if use_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("请设置 OPENAI_API_KEY 环境变量")
            sys.exit(1)
    else:
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("请设置 DASHSCOPE_API_KEY 环境变量")
            print("export DASHSCOPE_API_KEY='sk-xxx'")
            print("\n或使用: python 03_multi_turn_conversation.py --openai")
            sys.exit(1)
    
    # 先运行演示
    asyncio.run(demo_memory_context(use_openai))
    
    print("\n" + "=" * 50)
    print("现在进入交互模式...")
    print("=" * 50)
    
    # 进入交互模式
    asyncio.run(main(use_openai))
