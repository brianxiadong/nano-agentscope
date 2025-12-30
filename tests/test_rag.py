# -*- coding: utf-8 -*-
"""
测试 RAG 模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nano_agentscope.rag import SimpleKnowledge, Document, create_retrieve_tool


class TestDocument:
    """测试 Document 数据类"""
    
    def test_create_document(self):
        """测试创建文档"""
        doc = Document(name="测试", content="测试内容")
        assert doc.name == "测试"
        assert doc.content == "测试内容"
        assert doc.metadata == {}
    
    def test_create_document_with_metadata(self):
        """测试创建带元数据的文档"""
        doc = Document(
            name="测试",
            content="测试内容",
            metadata={"source": "test.md"}
        )
        assert doc.metadata["source"] == "test.md"


class TestSimpleKnowledge:
    """测试 SimpleKnowledge 类"""
    
    @pytest.fixture
    def knowledge(self):
        """创建测试用的知识库实例"""
        return SimpleKnowledge()
    
    @pytest.mark.asyncio
    async def test_add_document(self, knowledge):
        """测试添加文档"""
        await knowledge.add_document("Python", "Python 是一种编程语言")
        assert knowledge.size == 1
    
    @pytest.mark.asyncio
    async def test_add_documents(self, knowledge):
        """测试批量添加文档"""
        docs = [
            Document(name="Doc1", content="内容1"),
            Document(name="Doc2", content="内容2"),
        ]
        await knowledge.add_documents(docs)
        assert knowledge.size == 2
    
    @pytest.mark.asyncio
    async def test_retrieve_exact_match(self, knowledge):
        """测试精确匹配检索"""
        await knowledge.add_document("Python", "Python 是一种编程语言")
        await knowledge.add_document("Java", "Java 是一种面向对象语言")
        
        results = await knowledge.retrieve("Python")
        assert len(results) >= 1
        assert results[0].name == "Python"
    
    @pytest.mark.asyncio
    async def test_retrieve_partial_match(self, knowledge):
        """测试部分匹配检索"""
        await knowledge.add_document("Python教程", "Python 是编程语言，Python 很流行")
        await knowledge.add_document("Java教程", "Java 是面向对象语言")
        
        # 使用能够匹配的关键词 (tokenizer 会按词拆分)
        results = await knowledge.retrieve("Python")
        assert len(results) >= 1
        # Python教程 包含 "编程"
        assert any("Python" in doc.name for doc in results)
    
    @pytest.mark.asyncio
    async def test_retrieve_limit(self, knowledge):
        """测试结果数量限制"""
        for i in range(10):
            await knowledge.add_document(f"文档{i}", f"内容{i}")
        
        results = await knowledge.retrieve("文档", limit=3)
        assert len(results) <= 3
    
    @pytest.mark.asyncio
    async def test_retrieve_no_match(self, knowledge):
        """测试无匹配情况"""
        await knowledge.add_document("Python", "编程语言")
        
        results = await knowledge.retrieve("火星探索")
        # 没有匹配时返回空列表
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_knowledge(self, knowledge):
        """测试空知识库检索"""
        results = await knowledge.retrieve("任何内容")
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_list_documents(self, knowledge):
        """测试列出所有文档"""
        await knowledge.add_document("Doc1", "内容1")
        await knowledge.add_document("Doc2", "内容2")
        
        docs = await knowledge.list_documents()
        assert len(docs) == 2
    
    @pytest.mark.asyncio
    async def test_clear(self, knowledge):
        """测试清空知识库"""
        await knowledge.add_document("Doc1", "内容1")
        assert knowledge.size == 1
        
        await knowledge.clear()
        assert knowledge.size == 0
    
    @pytest.mark.asyncio
    async def test_init_with_documents(self):
        """测试使用初始文档创建知识库"""
        docs = [
            Document(name="Doc1", content="内容1"),
            Document(name="Doc2", content="内容2"),
        ]
        knowledge = SimpleKnowledge(documents=docs)
        assert knowledge.size == 2
    
    @pytest.mark.asyncio
    async def test_name_weight_higher(self, knowledge):
        """测试名称匹配权重高于内容"""
        # 名称中包含 Python 的文档
        await knowledge.add_document("Python入门", "这是一个教程")
        # 内容中包含 Python 的文档
        await knowledge.add_document("编程教程", "Python 是一种语言")
        
        results = await knowledge.retrieve("Python")
        # 名称匹配的应该排在前面
        assert results[0].name == "Python入门"


class TestCreateRetrieveTool:
    """测试 create_retrieve_tool 函数"""
    
    @pytest.mark.asyncio
    async def test_create_tool(self):
        """测试创建检索工具"""
        knowledge = SimpleKnowledge()
        await knowledge.add_document("测试", "测试内容")
        
        tool = create_retrieve_tool(knowledge)
        
        assert tool.__name__ == "search_knowledge"
        assert callable(tool)
    
    @pytest.mark.asyncio
    async def test_custom_tool_name(self):
        """测试自定义工具名称"""
        knowledge = SimpleKnowledge()
        tool = create_retrieve_tool(
            knowledge,
            tool_name="my_search",
            tool_description="自定义搜索"
        )
        
        assert tool.__name__ == "my_search"
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """测试工具执行"""
        knowledge = SimpleKnowledge()
        await knowledge.add_document("Python", "Python 是一种编程语言")
        
        tool = create_retrieve_tool(knowledge)
        result = await tool(query="Python")
        
        assert result.content is not None
        assert len(result.content) > 0
        # 检查结果包含 Python 相关内容
        text = result.content[0].get("text", "")
        assert "Python" in text
    
    @pytest.mark.asyncio
    async def test_tool_no_results(self):
        """测试无结果时的工具返回"""
        knowledge = SimpleKnowledge()
        await knowledge.add_document("Python", "编程语言")
        
        tool = create_retrieve_tool(knowledge)
        result = await tool(query="火星探索")
        
        assert result.content is not None
        text = result.content[0].get("text", "")
        assert "未找到" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
