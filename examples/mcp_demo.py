# -*- coding: utf-8 -*-
"""
MCP 使用示例 - 连接到远程工具服务器

本示例展示如何使用 nano-agentscope 的 MCP 功能连接到远程工具服务器。

运行前请确保：
1. 设置环境变量 DASHSCOPE_API_KEY（如果使用通义千问）
2. 有可用的 MCP 服务器 URL

Example:
    python examples/mcp_demo.py
"""

import asyncio
import os

from nano_agentscope import (
    ReActAgent,
    DashScopeChatModel,
    OpenAIFormatter,
    Toolkit,
    HttpStatelessClient,
    Msg,
)


async def demo_list_tools():
    """示例 1: 列出 MCP 服务器的可用工具"""
    print("=" * 50)
    print("示例 1: 列出 MCP 服务器的可用工具")
    print("=" * 50)
    
    # 创建 MCP 客户端
    # 这里使用一个示例 URL，你需要替换为实际的 MCP 服务器地址
    client = HttpStatelessClient(
        name="example",
        transport="streamable_http",
        url="https://mcp.api-inference.modelscope.net/f0361d8ec74544/mcp",  # 示例 URL
    )
    
    try:
        # 列出可用工具
        tools = await client.list_tools()
        
        print(f"\n找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"连接失败: {e}")
        print("请确保 MCP 服务器 URL 正确且可访问")
        # 尝试重试一次
        try:
            print("正在重试...")
            tools = await client.list_tools()
            print(f"\n重试成功！找到 {len(tools)} 个工具:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
        except Exception as retry_error:
            print(f"重试也失败了: {retry_error}")


async def demo_register_to_toolkit():
    """示例 2: 将 MCP 工具注册到 Toolkit"""
    print("\n" + "=" * 50)
    print("示例 2: 将 MCP 工具注册到 Toolkit")
    print("=" * 50)
    
    # 创建 MCP 客户端
    client = HttpStatelessClient(
        name="demo",
        transport="streamable_http",
        url="https://mcp.api-inference.modelscope.net/f0361d8ec74544/mcp",  # 替换为实际 URL
    )
    
    # 创建 Toolkit
    toolkit = Toolkit()
    
    try:
        await toolkit.register_mcp_client(client)
        
        # 查看注册的工具
        schemas = toolkit.get_json_schemas()
        print(f"\n已注册 {len(schemas)} 个工具到 Toolkit")
        for schema in schemas:
            print(f"  - {schema['function']['name']}")
    except Exception as e:
        print(f"注册失败: {e}")
        # 尝试重试一次
        try:
            print("正在重试注册...")
            await toolkit.register_mcp_client(client)
            schemas = toolkit.get_json_schemas()
            print(f"\n重试成功！已注册 {len(schemas)} 个工具到 Toolkit")
            for schema in schemas:
                print(f"  - {schema['function']['name']}")
        except Exception as retry_error:
            print(f"重试注册也失败了: {retry_error}")


async def demo_with_agent():
    """示例 3: 结合 ReActAgent 使用 MCP 工具"""
    print("\n" + "=" * 50)
    print("示例 3: 结合 ReActAgent 使用 MCP 工具")
    print("=" * 50)
    
    # 检查 API Key
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # ========== 日志配置 ==========
    # 1. 设置完整的工具结果日志（不截断）
    os.environ.setdefault("NANO_AGENTSCOPE_LOG_MAX_LENGTH", "0")
    
    # 2. 启用详细模式（显示 LLM 请求、Token 统计等）
    os.environ.setdefault("NANO_AGENTSCOPE_VERBOSE", "1")
    
    print("\n📋 日志配置:")
    print(f"  - 工具结果最大长度: {os.environ['NANO_AGENTSCOPE_LOG_MAX_LENGTH']} (0=不截断)")
    print(f"  - 详细模式: {os.environ['NANO_AGENTSCOPE_VERBOSE']} (1=开启)")
    print()
    
    # 创建 MCP 客户端
    client = HttpStatelessClient(
        name="12306-mcp",
        transport="streamable_http",
        url="https://mcp.api-inference.modelscope.net/f0361d8ec74544/mcp",
    )
    
    # 创建 Toolkit 并注册 MCP 工具
    toolkit = Toolkit()
    
    try:
        await toolkit.register_mcp_client(client)
    except Exception as e:
        print(f"MCP 连接失败: {e}")
        print("跳过此示例")
        return
    
    # 创建 Agent
    agent = ReActAgent(
        name="列车助手",
        sys_prompt="""你是一个列车助手，可以帮助用户查询列车信息。

重要提示：
1. 当用户说"明天"时，你需要：
   - 首先调用 get-current-date 获取今天的日期
   - 然后计算明天的日期（今天日期 + 1天）
   - 使用明天的日期调用 get-tickets
   
2. 当用户询问"时间最短"或"最快"的车次时：
   - 比较所有车次的"历时"字段
   - 找出历时最短的车次
   
3. 查询列车信息的步骤：
   - 步骤1: 使用 get-current-date 获取当前日期
   - 步骤2: 使用 get-station-code-of-citys 获取城市的站点代码
   - 步骤3: 使用 get-tickets 查询车次信息（注意日期格式：YYYY-MM-DD）
   
请严格按照以上步骤执行。""",
        model=DashScopeChatModel(model_name="qwen-max"),
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
    )
    
    # 对话
    response = await agent(
        Msg(name="user", content="明天从北京到上海的车次中时间最短的是哪一个", role="user")
    )
    
    print(f"\n助手回复: {response.get_text_content()}")


async def main():
    """运行所有示例"""
    print("nano-agentscope MCP 功能演示")
    print("=" * 50)
    
    # 运行示例
    await demo_list_tools()
    await demo_register_to_toolkit()
    await demo_with_agent()
    
    print("\n提示: 取消注释上面的函数调用来运行示例")
    print("请确保:")
    print("  1. 替换示例中的 MCP URL 为实际可用的服务器地址")
    print("  2. 设置相应的环境变量 (DASHSCOPE_API_KEY 等)")


if __name__ == "__main__":
    asyncio.run(main())
