#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例 2: 工具调用 - 让智能体使用工具

这个示例展示了如何：
1. 定义工具函数
2. 注册工具到 Toolkit
3. 让智能体自动决定何时调用工具

工具调用流程：
    用户: "现在几点了？"
    ↓
    LLM 推理: 需要调用 get_time 工具
    ↓
    执行工具: get_time() -> "14:30:00"
    ↓
    LLM 生成回复: "现在是 14:30:00"

运行方式:
    # 使用 DashScope（推荐）
    export DASHSCOPE_API_KEY="sk-xxx"
    python 02_tool_calling.py
    
    # 或使用 OpenAI
    export OPENAI_API_KEY="sk-xxx"
    python 02_tool_calling.py --openai
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,
    OpenAIChatModel,
    OpenAIFormatter,
    Toolkit,
    ToolResponse,
    TextBlock,
    Msg,
)


# ============== 定义工具函数 ==============

def get_current_time() -> ToolResponse:
    """获取当前时间
    
    这个工具函数会返回当前的日期和时间。
    注意：docstring 会被自动解析为工具描述。
    
    Returns:
        当前时间
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ToolResponse(
        content=[TextBlock(type="text", text=f"当前时间是: {now}")]
    )


def calculate(expression: str) -> ToolResponse:
    """计算数学表达式
    
    支持基本的数学运算，如加减乘除。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    try:
        # 注意：生产环境中不应使用 eval
        result = eval(expression, {"__builtins__": {}}, {})
        return ToolResponse(
            content=[TextBlock(type="text", text=f"{expression} = {result}")]
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"计算错误: {str(e)}")]
        )


def get_weather(city: str) -> ToolResponse:
    """获取城市天气信息
    
    查询指定城市的当前天气状况。
    
    Args:
        city: 城市名称，如 "北京"、"上海"
        
    Returns:
        天气信息
    """
    # 模拟天气数据（实际应该调用天气 API）
    weather_data = {
        "北京": "晴天，温度 25°C，湿度 40%",
        "上海": "多云，温度 28°C，湿度 60%",
        "广州": "小雨，温度 30°C，湿度 80%",
        "深圳": "晴天，温度 29°C，湿度 55%",
        "杭州": "阴天，温度 26°C，湿度 65%",
    }
    
    weather = weather_data.get(city, f"暂无 {city} 的天气数据")
    return ToolResponse(
        content=[TextBlock(type="text", text=f"{city}天气: {weather}")]
    )


def create_model(use_openai: bool = False):
    """创建 LLM 模型"""
    if use_openai:
        return OpenAIChatModel(
            model_name="gpt-4o-mini",
            stream=False,  # 工具调用演示使用非流式
        )
    else:
        return DashScopeChatModel(
            model_name="qwen-max",
            stream=False,
        )


async def main(use_openai: bool = False):
    """主函数"""
    
    # 1. 创建并配置工具集
    toolkit = Toolkit()
    
    # 注册工具函数
    toolkit.register_tool_function(get_current_time)
    toolkit.register_tool_function(calculate)
    toolkit.register_tool_function(get_weather)
    
    # 查看注册的工具 schema
    print("=" * 50)
    print("已注册的工具:")
    for schema in toolkit.get_json_schemas():
        func_info = schema["function"]
        print(f"  - {func_info['name']}: {func_info.get('description', '无描述')[:50]}...")
    print("=" * 50)
    
    # 2. 创建智能体
    model = create_model(use_openai)
    print(f"使用模型: {model.model_name}")
    
    agent = ReActAgent(
        name="工具助手",
        sys_prompt="""你是一个能够使用工具的 AI 助手。
你可以使用以下工具来帮助用户：
- get_current_time: 获取当前时间
- calculate: 计算数学表达式
- get_weather: 查询城市天气

请根据用户的问题选择合适的工具。""",
        model=model,
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
    )
    
    # 3. 测试不同场景
    test_cases = [
        "现在几点了？",
        "帮我计算 (15 + 27) * 3",
        "北京今天天气怎么样？",
        "你好，请自我介绍一下",  # 这个不需要工具
    ]
    
    for question in test_cases:
        print("\n" + "=" * 50)
        print(f"用户: {question}")
        print("-" * 50)
        
        response = await agent(Msg(name="user", content=question, role="user"))
        
        # 清空记忆，每次测试独立
        await agent.memory.clear()


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
            print("\n或使用: python 02_tool_calling.py --openai")
            sys.exit(1)
    
    asyncio.run(main(use_openai))
