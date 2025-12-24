#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例 1: Hello World - 最简单的智能体对话

这个示例展示了如何：
1. 创建一个基本的 ReAct 智能体
2. 与智能体进行简单对话
3. 理解 Msg 消息的基本结构

运行方式:
    # 使用 DashScope（阿里云通义千问）- 推荐
    export DASHSCOPE_API_KEY="sk-xxx"
    python 01_hello_world.py
    
    # 或使用 OpenAI
    export OPENAI_API_KEY="sk-xxx"
    python 01_hello_world.py --openai
"""

import asyncio
import os
import sys

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,
    OpenAIChatModel,
    OpenAIFormatter,
    InMemoryMemory,
    Msg,
)


def create_model(use_openai: bool = False):
    """创建 LLM 模型
    
    Args:
        use_openai: 是否使用 OpenAI 模型，否则使用 DashScope
    """
    if use_openai:
        return OpenAIChatModel(
            model_name="gpt-4o-mini",
            stream=True,
        )
    else:
        return DashScopeChatModel(
            model_name="qwen-max",  # 通义千问 qwen-max / qwen-plus / qwen-turbo
            stream=True,
        )


async def main(use_openai: bool = False):
    """主函数：创建智能体并进行对话"""
    
    # 1. 创建 LLM 模型
    model = create_model(use_openai)
    print(f"使用模型: {model.model_name}")
    
    # 2. 创建格式化器
    # DashScope 和 OpenAI 使用相同的消息格式
    formatter = OpenAIFormatter()
    
    # 3. 创建记忆模块
    memory = InMemoryMemory()
    
    # 4. 创建智能体
    agent = ReActAgent(
        name="小助手",
        sys_prompt="你是一个友好的 AI 助手，用简洁的中文回答问题。",
        model=model,
        formatter=formatter,
        memory=memory,
    )
    
    # 5. 创建用户消息
    user_msg = Msg(
        name="用户",
        content="你好！请用一句话介绍一下你自己。",
        role="user",
    )
    
    print("=" * 50)
    print("用户:", user_msg.content)
    print("=" * 50)
    
    # 6. 调用智能体获取回复
    response = await agent(user_msg)
    
    # 7. 打印回复
    print("=" * 50)
    print(f"回复完成！")
    print(f"消息 ID: {response.id}")
    print(f"时间戳: {response.timestamp}")


if __name__ == "__main__":
    # 检查命令行参数
    use_openai = "--openai" in sys.argv
    
    # 检查 API 密钥
    if use_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("请设置 OPENAI_API_KEY 环境变量")
            print("export OPENAI_API_KEY='sk-xxx'")
            sys.exit(1)
    else:
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("请设置 DASHSCOPE_API_KEY 环境变量")
            print("export DASHSCOPE_API_KEY='sk-xxx'")
            print("\n或者使用 OpenAI: python 01_hello_world.py --openai")
            sys.exit(1)
    
    asyncio.run(main(use_openai))
