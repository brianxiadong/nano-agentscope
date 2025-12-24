# -*- coding: utf-8 -*-
"""
模型模块 - 封装 LLM API 调用

本模块定义了模型调用的抽象和实现：
1. ChatResponse - 模型响应的数据结构
2. ChatModelBase - 模型基类，定义统一接口
3. DashScopeChatModel - 阿里云 DashScope（通义千问）API 实现
4. OpenAIChatModel - OpenAI API 的具体实现

学习要点：
- 使用抽象基类定义统一接口，方便扩展不同的模型
- 支持流式（streaming）和非流式两种输出模式
- 将 API 响应统一转换为内部格式（ChatResponse）
"""

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Literal

from .message import TextBlock, ToolUseBlock


@dataclass
class ChatUsage:
    """Token 使用统计
    
    记录一次 API 调用的 token 消耗，用于成本估算和监控。
    """
    input_tokens: int = 0   # 输入 token 数
    output_tokens: int = 0  # 输出 token 数
    time: float = 0.0       # 耗时（秒）


@dataclass
class ChatResponse:
    """模型响应数据结构
    
    将不同 LLM API 的响应统一转换为此格式，包含：
    - content: 响应内容（TextBlock 或 ToolUseBlock 列表）
    - usage: Token 使用统计
    - metadata: 额外信息（如结构化输出）
    
    Example:
        >>> response = ChatResponse(
        ...     content=[TextBlock(type="text", text="你好！")]
        ... )
    """
    content: list[TextBlock | ToolUseBlock] = field(default_factory=list)
    usage: ChatUsage | None = None
    metadata: dict | None = None


class ChatModelBase:
    """模型基类 - 定义统一的模型调用接口
    
    所有模型实现都应继承此类，并实现 __call__ 方法。
    
    设计原则：
    1. 统一接口：不同模型使用相同的调用方式
    2. 流式支持：通过 stream 属性控制是否流式输出
    3. 工具调用：支持 function calling / tool use
    
    Attributes:
        model_name: 模型名称
        stream: 是否使用流式输出
    """
    
    model_name: str
    stream: bool
    
    def __init__(self, model_name: str, stream: bool = True) -> None:
        self.model_name = model_name
        self.stream = stream
    
    @abstractmethod
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """调用模型生成响应
        
        Args:
            messages: 消息列表，格式为 API 要求的格式
            tools: 可用工具的 JSON schema 列表
            tool_choice: 工具选择模式
                - "auto": 自动决定是否调用工具
                - "none": 禁止调用工具
                - "required": 必须调用工具
            **kwargs: 其他参数，如 temperature, max_tokens 等
            
        Returns:
            如果 stream=False，返回 ChatResponse
            如果 stream=True，返回 AsyncGenerator[ChatResponse, None]
        """
        pass


class DashScopeChatModel(ChatModelBase):
    """DashScope（阿里云通义千问）Chat API 模型实现
    
    支持通义千问系列模型，如 qwen-max, qwen-plus, qwen-turbo 等。
    
    DashScope API 特点：
    - 使用 dashscope SDK
    - 支持工具调用（Function Calling）
    - 支持流式输出
    
    Example:
        >>> model = DashScopeChatModel(
        ...     model_name="qwen-max",
        ...     api_key="sk-xxx",  # 可选，默认从环境变量 DASHSCOPE_API_KEY 读取
        ... )
        >>> response = await model(messages=[{"role": "user", "content": "你好"}])
    """
    
    def __init__(
        self,
        model_name: str = "qwen-max",
        api_key: str | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 DashScope 模型
        
        Args:
            model_name: 模型名称，如 "qwen-max", "qwen-plus", "qwen-turbo"
            api_key: API 密钥，不提供则从 DASHSCOPE_API_KEY 环境变量读取
            stream: 是否使用流式输出
            **kwargs: 传递给生成 API 的其他参数，如 temperature
        """
        super().__init__(model_name, stream)
        
        import os
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "需要提供 api_key 参数或设置 DASHSCOPE_API_KEY 环境变量"
            )
        
        self.generate_kwargs = kwargs
    
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """调用 DashScope Generation API
        
        流程：
        1. 构建 API 请求参数
        2. 调用 dashscope SDK
        3. 解析响应为 ChatResponse 格式
        """
        # 延迟导入
        import dashscope
        from dashscope.aigc.generation import AioGeneration
        
        start_time = datetime.now()
        
        # 构建请求参数
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "api_key": self.api_key,
            "stream": self.stream,
            "result_format": "message",  # 使用 message 格式
            "incremental_output": self.stream,  # 流式时使用增量输出
            **self.generate_kwargs,
            **kwargs,
        }
        
        # 添加工具
        if tools:
            request_kwargs["tools"] = tools
        
        # 工具选择（DashScope 不支持 required，转为 auto）
        if tool_choice:
            if tool_choice == "required":
                tool_choice = "auto"
            request_kwargs["tool_choice"] = tool_choice
        
        # 调用 API
        response = await AioGeneration.call(**request_kwargs)
        
        if self.stream:
            return self._parse_stream_response(response, start_time)
        else:
            return self._parse_response(response, start_time)
    
    def _parse_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> ChatResponse:
        """解析非流式 API 响应"""
        from http import HTTPStatus
        
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope API 错误: {response}")
        
        content_blocks = []
        message = response.output.choices[0].message
        
        # 解析文本内容
        content = message.get("content")
        if content:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        content_blocks.append(
                            TextBlock(type="text", text=item["text"])
                        )
            else:
                content_blocks.append(
                    TextBlock(type="text", text=str(content))
                )
        
        # 解析工具调用
        for tool_call in message.get("tool_calls", []) or []:
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                input_dict = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                input_dict = {}
            
            content_blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tool_call.get("id", ""),
                    name=tool_call.get("function", {}).get("name", ""),
                    input=input_dict,
                )
            )
        
        # 构建 usage
        usage = None
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                time=(datetime.now() - start_time).total_seconds(),
            )
        
        return ChatResponse(content=content_blocks, usage=usage)
    
    async def _parse_stream_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> AsyncGenerator[ChatResponse, None]:
        """解析流式 API 响应"""
        from http import HTTPStatus
        
        text = ""
        tool_calls: dict[int, dict] = {}
        usage = None
        
        async for chunk in response:
            if chunk.status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope API 错误: {chunk}")
            
            message = chunk.output.choices[0].message
            
            # 累积文本（增量模式）
            content = message.get("content")
            if content:
                if isinstance(content, str):
                    text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text += item["text"]
            
            # 累积工具调用
            for tc in message.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    }
                else:
                    # 追加增量数据
                    if tc.get("id"):
                        tool_calls[idx]["id"] += tc["id"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        tool_calls[idx]["name"] += func["name"]
                    if func.get("arguments"):
                        tool_calls[idx]["arguments"] += func["arguments"]
            
            # 解析 usage
            if chunk.usage:
                usage = ChatUsage(
                    input_tokens=chunk.usage.input_tokens,
                    output_tokens=chunk.usage.output_tokens,
                    time=(datetime.now() - start_time).total_seconds(),
                )
            
            # 构建响应
            content_blocks = []
            if text:
                content_blocks.append(TextBlock(type="text", text=text))
            
            for tc in tool_calls.values():
                try:
                    input_dict = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    input_dict = {}
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tc["id"],
                        name=tc["name"],
                        input=input_dict,
                    )
                )
            
            yield ChatResponse(content=content_blocks, usage=usage)


class OpenAIChatModel(ChatModelBase):
    """OpenAI Chat API 模型实现
    
    支持 OpenAI 及兼容 API（如 Azure OpenAI, DeepSeek, 通义千问等）。
    
    Example:
        >>> model = OpenAIChatModel(
        ...     model_name="gpt-4o-mini",
        ...     api_key="sk-xxx",  # 可选，默认从环境变量读取
        ... )
        >>> response = await model(messages=[{"role": "user", "content": "你好"}])
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI 模型
        
        Args:
            model_name: 模型名称，如 "gpt-4o-mini", "gpt-4o"
            api_key: API 密钥，不提供则从 OPENAI_API_KEY 环境变量读取
            base_url: API 基础 URL，用于兼容其他 API
            stream: 是否使用流式输出
            **kwargs: 传递给 OpenAI 客户端的其他参数
        """
        super().__init__(model_name, stream)
        
        # 延迟导入，避免未安装 openai 时报错
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """调用 OpenAI Chat Completions API
        
        流程：
        1. 构建 API 请求参数
        2. 调用 API
        3. 解析响应为 ChatResponse 格式
        """
        start_time = datetime.now()
        
        # 构建请求参数
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            **kwargs,
        }
        
        # 添加工具相关参数
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice:
            request_kwargs["tool_choice"] = tool_choice
        
        # 流式输出需要包含 usage 信息
        if self.stream:
            request_kwargs["stream_options"] = {"include_usage": True}
        
        # 调用 API
        response = await self.client.chat.completions.create(**request_kwargs)
        
        if self.stream:
            # 流式模式：返回异步生成器
            return self._parse_stream_response(response, start_time)
        else:
            # 非流式模式：直接返回解析后的响应
            return self._parse_response(response, start_time)
    
    def _parse_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> ChatResponse:
        """解析非流式 API 响应"""
        content_blocks = []
        
        if response.choices:
            choice = response.choices[0]
            
            # 解析文本内容
            if choice.message.content:
                content_blocks.append(
                    TextBlock(type="text", text=choice.message.content)
                )
            
            # 解析工具调用
            for tool_call in choice.message.tool_calls or []:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
        
        # 构建 usage 统计
        usage = None
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                time=(datetime.now() - start_time).total_seconds(),
            )
        
        return ChatResponse(content=content_blocks, usage=usage)
    
    async def _parse_stream_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> AsyncGenerator[ChatResponse, None]:
        """解析流式 API 响应
        
        流式响应的特点：
        1. 逐块返回内容
        2. 需要累积文本和工具调用
        3. 最后一个 chunk 包含 usage 信息
        """
        text = ""
        tool_calls: dict[int, dict] = {}  # index -> tool_call 信息
        usage = None
        
        async for chunk in response:
            # 处理 usage 信息（通常在最后一个 chunk）
            if chunk.usage:
                usage = ChatUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    time=(datetime.now() - start_time).total_seconds(),
                )
            
            if not chunk.choices:
                # 最后一个 chunk 可能只有 usage
                if usage:
                    yield self._build_stream_response(text, tool_calls, usage)
                continue
            
            choice = chunk.choices[0]
            
            # 累积文本内容
            text += getattr(choice.delta, "content", None) or ""
            
            # 累积工具调用
            for tc in choice.delta.tool_calls or []:
                if tc.index not in tool_calls:
                    tool_calls[tc.index] = {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name if tc.function else "",
                        "input": tc.function.arguments if tc.function else "",
                    }
                else:
                    # 追加参数字符串
                    if tc.function and tc.function.arguments:
                        tool_calls[tc.index]["input"] += tc.function.arguments
            
            # 每个 chunk 都 yield 当前累积状态
            yield self._build_stream_response(text, tool_calls, usage)
    
    def _build_stream_response(
        self,
        text: str,
        tool_calls: dict[int, dict],
        usage: ChatUsage | None,
    ) -> ChatResponse:
        """构建流式响应的 ChatResponse"""
        content_blocks = []
        
        if text:
            content_blocks.append(TextBlock(type="text", text=text))
        
        for tc in tool_calls.values():
            # 尝试解析 JSON 参数
            try:
                input_dict = json.loads(tc["input"] or "{}")
            except json.JSONDecodeError:
                input_dict = {}
            
            content_blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["name"],
                    input=input_dict,
                )
            )
        
        return ChatResponse(content=content_blocks, usage=usage)

