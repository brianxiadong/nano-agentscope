# -*- coding: utf-8 -*-
"""
知识库模块 - 简易 RAG (Retrieval Augmented Generation) 实现

本模块提供了一个简化的知识库实现，用于教学目的：
1. SimpleKnowledge - 基于关键词匹配的简易知识库
2. retrieve_knowledge_tool - 将知识库检索包装为 Agent 可用的工具

学习要点：
- RAG 是增强 LLM 知识能力的核心技术
- 通过外部数据源弥补模型的知识边界
- 检索结果作为上下文注入 Prompt

实际生产中的 RAG 通常使用：
- 向量嵌入 (Embedding) 进行语义匹配
- 向量数据库 (如 FAISS, Chroma, Milvus)
- 文档分块和预处理

本模块为简化教学，使用关键词匹配代替向量检索。
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .message import TextBlock
from .tool import ToolResponse


@dataclass
class Document:
    """文档数据结构
    
    Attributes:
        name: 文档名称/标题
        content: 文档内容
        metadata: 元数据（可选）
    
    Example:
        >>> doc = Document(
        ...     name="Python基础",
        ...     content="Python 是一种解释型编程语言...",
        ...     metadata={"source": "tutorial.md"}
        ... )
    """
    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SimpleKnowledge:
    """简易知识库 - 基于关键词匹配的文档检索
    
    这是一个教学用的简化 RAG 实现，使用关键词匹配代替向量检索。
    
    核心功能：
    - 存储文档（名称 + 内容）
    - 根据查询关键词检索相关文档
    - 返回匹配度最高的结果
    
    匹配算法：
    - 计算查询词在文档（名称 + 内容）中出现的次数
    - 按匹配次数降序排序
    
    Example:
        >>> kb = SimpleKnowledge()
        >>> await kb.add_document("Python", "Python 是一种编程语言...")
        >>> await kb.add_document("Java", "Java 是一种面向对象语言...")
        >>> results = await kb.retrieve("Python 编程")
        >>> print(results[0].name)  # "Python"
    
    教学对比：
        实际生产中的 RAG 使用向量嵌入：
        1. 将文档转换为向量 (Embedding)
        2. 将查询转换为向量
        3. 计算向量相似度（余弦相似度）
        4. 返回最相似的文档
        
        本实现使用简单的关键词匹配，便于理解核心流程。
    """
    
    def __init__(self, documents: list[Document] | None = None) -> None:
        """初始化知识库
        
        Args:
            documents: 初始文档列表（可选）
        """
        self._documents: list[Document] = []
        if documents:
            for doc in documents:
                self._documents.append(doc)
    
    async def add_document(
        self,
        name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加文档到知识库
        
        Args:
            name: 文档名称
            content: 文档内容
            metadata: 元数据（可选）
        """
        doc = Document(
            name=name,
            content=content,
            metadata=metadata or {},
        )
        self._documents.append(doc)
    
    async def add_documents(self, documents: list[Document]) -> None:
        """批量添加文档
        
        Args:
            documents: 文档列表
        """
        self._documents.extend(documents)
    
    async def retrieve(
        self,
        query: str,
        limit: int = 3,
    ) -> list[Document]:
        """检索相关文档
        
        使用简单的关键词匹配算法：
        1. 将查询拆分为关键词
        2. 计算每个文档的匹配分数
        3. 返回分数最高的文档
        
        Args:
            query: 查询字符串
            limit: 返回结果数量限制
            
        Returns:
            匹配的文档列表，按相关度降序排序
        """
        if not self._documents:
            return []
        
        # 简单分词：按空格和标点分割
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return self._documents[:limit]
        
        # 计算每个文档的匹配分数
        scored_docs: list[tuple[Document, int]] = []
        for doc in self._documents:
            score = self._calculate_score(doc, query_terms)
            if score > 0:
                scored_docs.append((doc, score))
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 limit 个
        return [doc for doc, _ in scored_docs[:limit]]
    
    def _tokenize(self, text: str) -> list[str]:
        """简单分词
        
        将文本拆分为词汇列表（小写化）。
        
        Args:
            text: 输入文本
            
        Returns:
            词汇列表
        """
        # 简单处理：按空格分割，移除标点，转小写
        import re
        # 保留中文、英文、数字
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        return words
    
    def _calculate_score(self, doc: Document, query_terms: list[str]) -> int:
        """计算文档与查询的匹配分数
        
        匹配策略：
        - 名称匹配权重较高 (x3)
        - 内容匹配权重较低 (x1)
        
        Args:
            doc: 文档
            query_terms: 查询词列表
            
        Returns:
            匹配分数
        """
        score = 0
        
        # 处理名称
        name_lower = doc.name.lower()
        name_tokens = self._tokenize(doc.name)
        
        # 处理内容
        content_lower = doc.content.lower()
        content_tokens = self._tokenize(doc.content)
        
        for term in query_terms:
            # 名称匹配（权重 3）
            if term in name_lower or term in name_tokens:
                score += 3
            
            # 内容匹配（权重 1）
            score += content_tokens.count(term)
        
        return score
    
    async def list_documents(self) -> list[Document]:
        """列出所有文档
        
        Returns:
            所有文档列表
        """
        return self._documents.copy()
    
    async def clear(self) -> None:
        """清空知识库"""
        self._documents.clear()
    
    @property
    def size(self) -> int:
        """知识库中的文档数量"""
        return len(self._documents)


def create_retrieve_tool(
    knowledge: SimpleKnowledge,
    tool_name: str = "search_knowledge",
    tool_description: str | None = None,
) -> Callable:
    """创建知识库检索工具函数
    
    将 SimpleKnowledge 的检索功能包装为一个可注册到 Toolkit 的工具函数。
    
    Args:
        knowledge: SimpleKnowledge 实例
        tool_name: 工具函数名称
        tool_description: 工具描述
        
    Returns:
        可注册的工具函数
        
    Example:
        >>> kb = SimpleKnowledge()
        >>> await kb.add_document("FAQ", "常见问题...")
        >>> 
        >>> search_tool = create_retrieve_tool(kb)
        >>> toolkit.register_tool_function(search_tool)
    """
    description = tool_description or "搜索知识库获取相关信息"
    
    async def retrieve_func(query: str, limit: int = 3) -> ToolResponse:
        """搜索知识库获取相关信息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
        """
        results = await knowledge.retrieve(query, limit)
        
        if not results:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=f"未找到与 '{query}' 相关的内容。"
                )]
            )
        
        # 格式化结果
        output_parts = [f"找到 {len(results)} 条相关结果：\n"]
        for i, doc in enumerate(results, 1):
            output_parts.append(f"\n【{i}. {doc.name}】\n{doc.content}")
        
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text="".join(output_parts)
            )]
        )
    
    # 设置函数名称和文档
    retrieve_func.__name__ = tool_name
    retrieve_func.__doc__ = f"""{description}
    
    Args:
        query: 搜索查询关键词
        limit: 返回结果数量限制
    """
    
    return retrieve_func
