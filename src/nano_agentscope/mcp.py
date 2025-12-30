# -*- coding: utf-8 -*-
"""
MCP 模块 - Model Context Protocol 支持

本模块提供 MCP (Model Context Protocol) 的简化实现，允许 Agent 连接到远程工具服务器。

学习要点：
- MCP 是一个标准协议，用于 Agent 与工具服务器的通信
- HttpStatelessClient 是最简单的客户端类型，每次调用都是独立会话
- MCPToolFunction 将远程工具包装成可调用的函数对象

主要组件：
1. HttpStatelessClient - 无状态 HTTP 客户端
2. MCPToolFunction - MCP 工具函数包装类

Example:
    >>> import asyncio
    >>> from nano_agentscope import HttpStatelessClient, Toolkit
    >>> 
    >>> async def main():
    ...     # 创建 MCP 客户端
    ...     client = HttpStatelessClient(
    ...         name="example",
    ...         transport="streamable_http",
    ...         url="https://example.com/mcp",
    ...     )
    ...     
    ...     # 列出可用工具
    ...     tools = await client.list_tools()
    ...     print(f"可用工具: {[t.name for t in tools]}")
    ...     
    ...     # 注册到 Toolkit
    ...     toolkit = Toolkit()
    ...     await toolkit.register_mcp_client(client)
    >>> 
    >>> asyncio.run(main())
"""

from typing import Any, Callable, Literal, List
from contextlib import _AsyncGeneratorContextManager
import asyncio

import mcp.types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from .message import TextBlock, ImageBlock
from .tool import ToolResponse


# ============== 辅助函数 ==============

def _extract_json_schema_from_mcp_tool(tool: mcp.types.Tool) -> dict[str, Any]:
    """从 MCP Tool 对象提取 JSON Schema
    
    MCP 服务器返回的工具描述需要转换为 OpenAI function calling 格式。
    
    Args:
        tool: MCP Tool 对象，包含 name, description, inputSchema
        
    Returns:
        符合 OpenAI function calling 格式的 JSON Schema
        
    Example:
        >>> # MCP Tool 格式
        >>> tool.name = "get_weather"
        >>> tool.description = "获取天气"
        >>> tool.inputSchema = {"type": "object", "properties": {"city": {"type": "string"}}}
        >>> 
        >>> # 转换后
        >>> schema = _extract_json_schema_from_mcp_tool(tool)
        >>> schema["function"]["name"]  # "get_weather"
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", []),
            },
        },
    }


def _convert_mcp_content_to_blocks(
    mcp_content_blocks: list,
) -> List[TextBlock | ImageBlock]:
    """将 MCP 响应内容转换为 nano-agentscope 的内容块
    
    MCP 服务器返回的响应使用 mcp.types 中的类型，
    需要转换为 nano-agentscope 使用的 ContentBlock 类型。
    
    Args:
        mcp_content_blocks: MCP 响应中的内容块列表
        
    Returns:
        转换后的 ContentBlock 列表
    """
    result: list = []
    
    for content in mcp_content_blocks:
        if isinstance(content, mcp.types.TextContent):
            # 文本内容
            result.append(
                TextBlock(
                    type="text",
                    text=content.text,
                )
            )
        elif isinstance(content, mcp.types.ImageContent):
            # 图片内容（使用 data URL）
            result.append(
                ImageBlock(
                    type="image",
                    url=f"data:{content.mimeType};base64,{content.data}",
                )
            )
        elif isinstance(content, mcp.types.EmbeddedResource):
            # 嵌入资源（转为文本）
            if isinstance(content.resource, mcp.types.TextResourceContents):
                result.append(
                    TextBlock(
                        type="text",
                        text=content.resource.text,
                    )
                )
        # 其他类型暂时跳过
    
    return result


# ============== MCPToolFunction ==============

class MCPToolFunction:
    """MCP 工具函数包装类
    
    将 MCP 服务器提供的工具包装成可调用的 Python 函数。
    每次调用时会建立新的连接（无状态模式）。
    
    核心属性：
        - name: 工具名称
        - description: 工具描述
        - json_schema: 符合 OpenAI 格式的 JSON Schema
        - mcp_name: 所属 MCP 服务器的名称
    
    Example:
        >>> # 通常不直接创建，而是通过 HttpStatelessClient 获取
        >>> func = await client.get_callable_function("get_weather")
        >>> result = await func(city="北京")
    """
    
    name: str
    """工具函数名称"""
    
    description: str
    """工具函数描述"""
    
    json_schema: dict[str, Any]
    """符合 OpenAI function calling 格式的 JSON Schema"""
    
    mcp_name: str
    """所属 MCP 服务器的名称"""
    
    def __init__(
        self,
        mcp_name: str,
        tool: mcp.types.Tool,
        client_gen: Callable[..., _AsyncGeneratorContextManager[Any]],
        wrap_tool_result: bool = True,
    ) -> None:
        """初始化 MCP 工具函数
        
        Args:
            mcp_name: MCP 服务器名称
            tool: MCP Tool 对象
            client_gen: 用于创建客户端连接的生成器函数
            wrap_tool_result: 是否将结果包装为 ToolResponse
        """
        self.mcp_name = mcp_name
        self.name = tool.name
        self.description = tool.description or ""
        self.json_schema = _extract_json_schema_from_mcp_tool(tool)
        self.wrap_tool_result = wrap_tool_result
        self._client_gen = client_gen
    
    async def __call__(self, **kwargs: Any) -> mcp.types.CallToolResult | ToolResponse:
        """调用 MCP 工具函数
        
        每次调用都会建立新的连接，执行完成后关闭。
        包含自动重试机制来处理网络错误。
        
        Args:
            **kwargs: 传递给工具函数的参数
            
        Returns:
            如果 wrap_tool_result=True，返回 ToolResponse
            否则返回原始的 mcp.types.CallToolResult
        """
        import aiohttp
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # 建立连接并调用
                async with self._client_gen() as cli:
                    read_stream, write_stream = cli[0], cli[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        res = await session.call_tool(self.name, arguments=kwargs)
                
                # 是否包装结果
                if self.wrap_tool_result:
                    content_blocks = _convert_mcp_content_to_blocks(res.content)
                    return ToolResponse(
                        content=content_blocks,
                        metadata=res.meta if hasattr(res, 'meta') else None,
                    )
                
                return res
                
            except (aiohttp.ClientPayloadError, aiohttp.ClientError, 
                    aiohttp.ClientConnectorError) as e:
                if attempt < max_retries:
                    print(f"工具调用失败（尝试 {attempt + 1}/{max_retries + 1}）: {type(e).__name__}: {e}")
                    print(f"正在重试工具 {self.name}...")
                    await asyncio.sleep(1)  # 等待 1 秒后重试
                    continue
                else:
                    raise RuntimeError(f"工具调用失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
            
            except Exception as e:
                # 对于其他类型的错误，不重试
                raise


# ============== HttpStatelessClient ==============

class HttpStatelessClient:
    """无状态 HTTP MCP 客户端
    
    这是最简单的 MCP 客户端类型，每次工具调用都是独立的会话。
    支持两种传输协议：
    - streamable_http: 适用于现代 MCP 服务器（URL 通常以 /mcp 结尾）
    - sse: Server-Sent Events，较老的协议（URL 通常以 /sse 结尾）
    
    设计原则：
    - 无状态：每次调用独立，不保持会话
    - 简单：适合教学和大多数使用场景
    - 灵活：支持自定义 headers 和超时
    
    Example:
        >>> # 创建客户端
        >>> client = HttpStatelessClient(
        ...     name="gaode",
        ...     transport="streamable_http",
        ...     url="https://mcp.amap.com/mcp",
        ...     headers={"Authorization": "Bearer xxx"},
        ... )
        >>> 
        >>> # 列出工具
        >>> tools = await client.list_tools()
        >>> 
        >>> # 获取可调用函数
        >>> search_func = await client.get_callable_function("search_places")
        >>> result = await search_func(keyword="餐厅", city="北京")
    """
    
    stateful: bool = False
    """是否为有状态客户端（本类始终为 False）"""
    
    def __init__(
        self,
        name: str,
        transport: Literal["streamable_http", "sse"],
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60,
        sse_read_timeout: float = 300,
        **client_kwargs: Any,
    ) -> None:
        """初始化 HTTP MCP 客户端
        
        Args:
            name: MCP 服务器的唯一标识名称
            transport: 传输协议类型
                - "streamable_http": 现代 MCP 协议
                - "sse": Server-Sent Events
            url: MCP 服务器的 URL
            headers: 附加的 HTTP 请求头（如认证信息）
            timeout: HTTP 请求超时时间（秒）
            sse_read_timeout: SSE 读取超时时间（秒）
            **client_kwargs: 传递给底层客户端的额外参数
        """
        self.name = name
        
        if transport not in ["streamable_http", "sse"]:
            raise ValueError(
                f"不支持的传输类型: {transport}。"
                "支持的类型: 'streamable_http', 'sse'"
            )
        
        self.transport = transport
        self.client_config = {
            "url": url,
            "headers": headers or {},
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout,
            **client_kwargs,
        }
        
        # 缓存工具列表
        self._tools: list[mcp.types.Tool] | None = None
    
    def get_client(self) -> _AsyncGeneratorContextManager[Any]:
        """获取一次性的 MCP 客户端连接
        
        返回一个上下文管理器，用于建立和管理连接。
        
        Returns:
            异步上下文管理器
        """
        if self.transport == "sse":
            return sse_client(**self.client_config)
        
        if self.transport == "streamable_http":
            return streamablehttp_client(**self.client_config)
        
        raise ValueError(f"不支持的传输类型: {self.transport}")
    
    @property
    def url(self) -> str:
        """获取 MCP 服务器 URL"""
        return self.client_config.get("url", "")
    
    async def list_tools(self) -> List[mcp.types.Tool]:
        """列出 MCP 服务器上的所有可用工具
        
        首次调用会连接服务器获取工具列表并缓存。
        后续调用直接返回缓存结果。
        
        Returns:
            MCP Tool 对象列表
            
        Example:
            >>> tools = await client.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        import aiohttp
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with self.get_client() as cli:
                    read_stream, write_stream = cli[0], cli[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        res = await session.list_tools()
                        self._tools = res.tools
                        return res.tools
                        
            except (aiohttp.ClientPayloadError, aiohttp.ClientError,
                    aiohttp.ClientConnectorError) as e:
                if attempt < max_retries:
                    print(f"获取工具列表失败（尝试 {attempt + 1}/{max_retries + 1}）: {type(e).__name__}: {e}")
                    print(f"正在重试连接到 {self.url}...")
                    await asyncio.sleep(1)  # 等待 1 秒后重试
                    continue
                else:
                    raise RuntimeError(f"连接 MCP 服务器失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
                    
            except Exception as e:
                # 对于其他类型的错误，不重试
                raise
    
    async def get_callable_function(
        self,
        func_name: str,
        wrap_tool_result: bool = True,
    ) -> MCPToolFunction:
        """获取指定名称的可调用工具函数
        
        Args:
            func_name: 工具函数名称
            wrap_tool_result: 是否将结果包装为 ToolResponse
            
        Returns:
            MCPToolFunction 对象，可直接 await 调用
            
        Raises:
            ValueError: 如果找不到指定名称的工具
            
        Example:
            >>> func = await client.get_callable_function("get_weather")
            >>> result = await func(city="北京")
            >>> print(result.content[0]["text"])
        """
        # 确保已获取工具列表
        if self._tools is None:
            await self.list_tools()
        
        # 查找目标工具
        target_tool = None
        for tool in self._tools:
            if tool.name == func_name:
                target_tool = tool
                break
        
        if target_tool is None:
            available = [t.name for t in self._tools]
            raise ValueError(
                f"找不到工具 '{func_name}'。"
                f"可用工具: {available}"
            )
        
        return MCPToolFunction(
            mcp_name=self.name,
            tool=target_tool,
            client_gen=self.get_client,
            wrap_tool_result=wrap_tool_result,
        )
