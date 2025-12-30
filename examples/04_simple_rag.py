# -*- coding: utf-8 -*-
"""
示例 04：简易 RAG (知识库检索)

本示例演示如何使用 SimpleKnowledge 创建一个知识库，
并将其包装为工具供 ReActAgent 使用。

教学目标：
1. 理解 RAG (Retrieval Augmented Generation) 的核心思想
2. 学习如何将知识库检索封装为 Agent 工具
3. 观察 Agent 如何结合外部知识回答问题

运行方式：
    export DASHSCOPE_API_KEY=your_key  # 或 OPENAI_API_KEY
    python examples/04_simple_rag.py
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
    SimpleKnowledge,
    create_retrieve_tool,
)


async def main():
    print("=" * 60)
    print("📚 Nano-AgentScope 示例：简易 RAG (知识库检索)")
    print("=" * 60)
    
    # ============ Step 1: 创建知识库 ============
    print("\n📖 Step 1: 创建知识库并添加文档...")
    
    knowledge = SimpleKnowledge()
    
    # 添加一些示例文档
    await knowledge.add_document(
        name="Python简介",
        content="""Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。
Python 由 Guido van Rossum 于 1989 年底发明，第一个公开发行版发行于 1991 年。
Python 的设计理念强调代码的可读性和简洁的语法，使用空格缩进划分代码块。"""
    )
    
    await knowledge.add_document(
        name="Agent框架",
        content="""Agent 框架是用于构建 AI 智能体的软件框架。
常见的 Agent 框架包括：LangChain、AutoGPT、AgentScope 等。
Agent 可以使用工具、记忆和规划能力来完成复杂任务。
ReAct 是一种常用的 Agent 模式，结合推理 (Reasoning) 和行动 (Acting)。"""
    )
    
    await knowledge.add_document(
        name="RAG技术",
        content="""RAG (Retrieval Augmented Generation) 是一种增强大语言模型能力的技术。
RAG 的核心思想是：在生成回答前，先从知识库中检索相关信息。
这样可以让模型回答更准确、更新，并减少幻觉问题。
RAG 的典型流程：查询 -> 检索相关文档 -> 将文档作为上下文 -> 生成回答。"""
    )
    
    await knowledge.add_document(
        name="MCP协议",
        content="""MCP (Model Context Protocol) 是一种用于连接 AI 模型和外部工具的协议。
MCP 定义了标准的工具调用接口，支持多种传输方式。
通过 MCP，Agent 可以调用远程服务器上的工具，实现更强大的功能。"""
    )
    
    print(f"  ✅ 已添加 {knowledge.size} 个文档到知识库")
    
    # ============ Step 2: 创建检索工具 ============
    print("\n🔧 Step 2: 将知识库检索封装为工具...")
    
    search_tool = create_retrieve_tool(
        knowledge=knowledge,
        tool_name="search_knowledge",
        tool_description="搜索内部知识库，获取 Python、Agent 框架、RAG 等技术相关信息",
    )
    
    toolkit = Toolkit()
    toolkit.register_tool_function(search_tool)
    
    print("  ✅ 已注册检索工具: search_knowledge")
    
    # ============ Step 3: 创建 Agent ============
    print("\n🤖 Step 3: 创建带知识库的 Agent...")
    
    # 选择模型
    if os.environ.get("DASHSCOPE_API_KEY"):
        model = DashScopeChatModel(model_name="qwen-max")
        print("  使用模型: DashScope (通义千问)")
    elif os.environ.get("OPENAI_API_KEY"):
        model = OpenAIChatModel(model_name="gpt-4o-mini")
        print("  使用模型: OpenAI")
    else:
        print("  ⚠️ 未设置 API Key，请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        return
    
    agent = ReActAgent(
        name="知识助手",
        sys_prompt="""你是一个技术知识助手。
当用户询问技术问题时，请先使用 search_knowledge 工具搜索知识库。
根据搜索结果回答用户问题，如果知识库中没有相关信息，请诚实地说明。
回答要简洁准确，并标明信息来源。""",
        model=model,
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    
    print("  ✅ Agent 创建完成")
    
    # ============ Step 4: 测试对话 ============
    print("\n💬 Step 4: 开始对话测试...")
    print("-" * 40)
    
    # 测试问题列表
    test_questions = [
        "什么是 RAG 技术？它有什么用？",
        "Python 是谁发明的？",
        "常见的 Agent 框架有哪些？",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[问题 {i}] {question}")
        print("-" * 40)
        
        response = await agent(Msg(name="user", content=question, role="user"))
        
        print(f"\n[回答] {response.get_text_content()}")
        print("=" * 60)
    
    print("\n✅ 示例完成！")
    print("\n💡 教学要点：")
    print("  1. SimpleKnowledge 使用关键词匹配进行检索（生产环境使用向量检索）")
    print("  2. create_retrieve_tool 将检索功能包装为 Agent 可调用的工具")
    print("  3. Agent 会先搜索知识库，再基于搜索结果生成回答")


if __name__ == "__main__":
    asyncio.run(main())
