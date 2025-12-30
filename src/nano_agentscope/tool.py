# -*- coding: utf-8 -*-
"""
工具模块 - 管理和执行工具函数

本模块定义了工具系统的核心组件：
1. ToolResponse - 工具执行结果的数据结构
2. Toolkit - 工具管理器，负责注册、解析和执行工具

学习要点：
- 工具函数是 Agent 与外部世界交互的桥梁
- JSON Schema 用于向 LLM 描述工具的参数
- 从 docstring 自动提取函数描述和参数信息
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .mcp import MCPToolFunction, HttpStatelessClient

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from .message import TextBlock, ToolUseBlock


@dataclass
class ToolResponse:
    """工具执行结果
    
    工具函数执行后返回此结构，包含：
    - content: 执行结果内容（文本或其他块）
    - metadata: 元数据（可选）
    - is_last: 是否为最后一个响应块（用于流式）
    - is_interrupted: 是否被用户中断（用于实时干预）
    
    Example:
        >>> def get_time() -> ToolResponse:
        ...     import datetime
        ...     now = datetime.datetime.now().strftime("%H:%M:%S")
        ...     return ToolResponse(
        ...         content=[TextBlock(type="text", text=f"当前时间: {now}")]
        ...     )
    """
    content: list[TextBlock] = field(default_factory=list)
    metadata: dict | None = None
    is_last: bool = True
    is_interrupted: bool = False  # 用于实时中断标记


def _parse_function_to_schema(func: Callable) -> dict:
    """从函数签名和 docstring 解析 JSON Schema
    
    这是工具系统的核心函数，它能够：
    1. 从函数签名提取参数类型
    2. 从 docstring 提取函数描述和参数描述
    3. 生成符合 OpenAI function calling 格式的 JSON Schema
    
    Args:
        func: 要解析的函数
        
    Returns:
        JSON Schema 字典
        
    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''两数相加
        ...     
        ...     Args:
        ...         a: 第一个数
        ...         b: 第二个数
        ...     '''
        ...     return a + b
        >>> schema = _parse_function_to_schema(add)
        >>> print(schema["function"]["name"])  # "add"
    """
    # 解析 docstring
    docstring = parse(func.__doc__ or "")
    params_doc = {p.arg_name: p.description for p in docstring.params}
    
    # 函数描述
    descriptions = []
    if docstring.short_description:
        descriptions.append(docstring.short_description)
    if docstring.long_description:
        descriptions.append(docstring.long_description)
    func_description = "\n".join(descriptions)
    
    # 构建 Pydantic 模型来生成 JSON Schema
    fields = {}
    for name, param in inspect.signature(func).parameters.items():
        # 跳过 self, cls
        if name in ["self", "cls"]:
            continue
        
        # 获取类型注解
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            annotation = Any
        
        # 获取默认值
        if param.default == inspect.Parameter.empty:
            default = ...  # 必需参数
        else:
            default = param.default
        
        # 获取参数描述
        description = params_doc.get(name, None)
        
        fields[name] = (annotation, Field(default=default, description=description))
    
    # 动态创建 Pydantic 模型
    if fields:
        DynamicModel = create_model("DynamicModel", **fields)
        params_schema = DynamicModel.model_json_schema()
        # 移除多余的 title 字段
        params_schema.pop("title", None)
        for prop in params_schema.get("properties", {}).values():
            prop.pop("title", None)
    else:
        params_schema = {"type": "object", "properties": {}}
    
    # 构建最终的 JSON Schema
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "parameters": params_schema,
        }
    }
    
    if func_description:
        schema["function"]["description"] = func_description
    
    return schema


class Toolkit:
    """工具管理器 - 注册、管理和执行工具函数
    
    Toolkit 是工具系统的核心，提供：
    1. 工具函数注册和解析
    2. 获取工具的 JSON Schema 列表
    3. 根据 ToolUseBlock 执行工具函数
    
    设计原则：
    - 简单：只需注册函数，自动解析参数
    - 灵活：支持同步和异步函数
    - 统一：所有工具返回 ToolResponse
    
    Example:
        >>> toolkit = Toolkit()
        >>> 
        >>> def get_weather(city: str) -> ToolResponse:
        ...     '''获取天气信息
        ...     
        ...     Args:
        ...         city: 城市名称
        ...     '''
        ...     return ToolResponse(
        ...         content=[TextBlock(type="text", text=f"{city}天气晴")]
        ...     )
        >>> 
        >>> toolkit.register_tool_function(get_weather)
        >>> schemas = toolkit.get_json_schemas()
        >>> print(len(schemas))  # 1
    """
    
    def __init__(self) -> None:
        """初始化工具管理器"""
        # 存储注册的工具: name -> (function, schema)
        self._tools: dict[str, tuple[Callable, dict]] = {}
    
    def register_tool_function(
        self,
        func: Callable,
        description: str | None = None,
    ) -> None:
        """注册工具函数
        
        Args:
            func: 工具函数，应返回 ToolResponse
            description: 函数描述，不提供则从 docstring 提取
        """
        # 解析 JSON Schema
        schema = _parse_function_to_schema(func)
        
        # 覆盖描述（如果提供）
        if description:
            schema["function"]["description"] = description
        
        # 存储
        self._tools[func.__name__] = (func, schema)
    
    def remove_tool_function(self, name: str) -> None:
        """移除工具函数
        
        Args:
            name: 函数名
        """
        self._tools.pop(name, None)
    
    def get_json_schemas(self) -> list[dict]:
        """获取所有工具的 JSON Schema 列表
        
        返回格式符合 OpenAI function calling API 要求。
        
        Returns:
            JSON Schema 列表
        """
        return [schema for _, schema in self._tools.values()]
    
    @property
    def tools(self) -> dict[str, tuple[Callable, dict]]:
        """获取所有注册的工具"""
        return self._tools
    
    async def call_tool_function(
        self,
        tool_call: ToolUseBlock,
    ) -> ToolResponse:
        """执行工具函数
        
        Args:
            tool_call: 工具调用块，包含函数名和参数
            
        Returns:
            工具执行结果
        """
        func_name = tool_call["name"]
        
        # 检查函数是否存在
        if func_name not in self._tools:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=f"Error: 找不到工具函数 '{func_name}'"
                )]
            )
        
        func, _ = self._tools[func_name]
        kwargs = tool_call.get("input", {}) or {}
        
        try:
            # 执行函数（支持同步、异步函数和可调用对象如 MCPToolFunction）
            if inspect.iscoroutinefunction(func):
                # 异步函数
                result = await func(**kwargs)
            elif hasattr(func, '__call__') and inspect.iscoroutinefunction(func.__call__):
                # 具有异步 __call__ 的对象（如 MCPToolFunction）
                result = await func(**kwargs)
            else:
                # 同步函数
                result = func(**kwargs)
            
            # 确保返回 ToolResponse
            if isinstance(result, ToolResponse):
                return result
            else:
                # 自动包装其他返回值
                return ToolResponse(
                    content=[TextBlock(type="text", text=str(result))]
                )
                
        except Exception as e:
            # 捕获异常并返回错误信息
            return ToolResponse(
                content=[TextBlock(type="text", text=f"Error: {str(e)}")]
            )
    
    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
    
    # ============== MCP 支持 ==============
    
    def register_mcp_tool_function(
        self,
        mcp_func: "MCPToolFunction",
    ) -> None:
        """注册 MCP 工具函数
        
        将 MCPToolFunction 对象注册到工具管理器中。
        
        Args:
            mcp_func: MCPToolFunction 对象
            
        Example:
            >>> func = await client.get_callable_function("get_weather")
            >>> toolkit.register_mcp_tool_function(func)
        """
        # 存储 MCP 函数，使用其内置的 json_schema
        self._tools[mcp_func.name] = (mcp_func, mcp_func.json_schema)
    
    async def register_mcp_client(
        self,
        mcp_client: "HttpStatelessClient",
        enable_funcs: list[str] | None = None,
        disable_funcs: list[str] | None = None,
    ) -> None:
        """注册 MCP 客户端的所有工具函数
        
        连接到 MCP 服务器，获取所有可用工具并注册。
        
        Args:
            mcp_client: HttpStatelessClient 实例
            enable_funcs: 只注册这些函数（可选）
            disable_funcs: 排除这些函数（可选）
            
        Example:
            >>> client = HttpStatelessClient(
            ...     name="gaode",
            ...     transport="streamable_http", 
            ...     url="https://mcp.amap.com/mcp",
            ... )
            >>> await toolkit.register_mcp_client(client)
        """
        # 检查参数冲突
        if enable_funcs is not None and disable_funcs is not None:
            intersection = set(enable_funcs) & set(disable_funcs)
            if intersection:
                raise ValueError(
                    f"enable_funcs 和 disable_funcs 不能有重叠: {intersection}"
                )
        
        # 获取工具列表
        tools = await mcp_client.list_tools()
        
        registered_count = 0
        for tool in tools:
            # 过滤
            if enable_funcs is not None and tool.name not in enable_funcs:
                continue
            if disable_funcs is not None and tool.name in disable_funcs:
                continue
            
            # 获取可调用函数并注册
            func = await mcp_client.get_callable_function(
                func_name=tool.name,
                wrap_tool_result=True,
            )
            self.register_mcp_tool_function(func)
            registered_count += 1
        
        print(f"已从 MCP '{mcp_client.name}' 注册 {registered_count} 个工具函数")


# ============== 示例工具函数 ==============

def calculator(expression: str) -> ToolResponse:
    """简单计算器 - 计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    try:
        # 注意：eval 在生产环境中不安全，这里仅作演示
        result = eval(expression, {"__builtins__": {}}, {})
        return ToolResponse(
            content=[TextBlock(type="text", text=f"计算结果: {expression} = {result}")]
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"计算错误: {str(e)}")]
        )


def get_current_time() -> ToolResponse:
    """获取当前时间
    
    Returns:
        当前时间字符串
    """
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ToolResponse(
        content=[TextBlock(type="text", text=f"当前时间: {now}")]
    )


