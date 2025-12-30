# Nano-AgentScope 日志配置指南

## 概述

Nano-AgentScope 提供了灵活的日志配置选项，帮助你在开发和调试过程中更好地了解 Agent 的运行状态。

## 环境变量配置

### 1. NANO_AGENTSCOPE_LOG_MAX_LENGTH

控制工具调用结果的显示长度。

**取值:**
- `0`: 不截断，显示完整内容（推荐用于开发调试）
- `>0`: 截断到指定字符数（如 `2000`、`5000`）
- 默认值: `2000`

**设置方法:**

```bash
# 方法 1: 命令行设置
export NANO_AGENTSCOPE_LOG_MAX_LENGTH=0

# 方法 2: 在代码中设置
import os
os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "0"
```

**示例输出:**

```
# 截断模式 (默认)
[工具结果] get-tickets: 车次|出发站 -> 到达站|...
    ... (已截断，总长度: 5000 字符)

# 完整模式 (设置为 0)
[工具结果] get-tickets: 车次|出发站 -> 到达站|出发时间 -> 到达时间|历时
G21 北京南(telecode:VNP) -> 上海虹桥(telecode:AOH) 17:00 -> 21:18 历时：04:18
- 商务座: 剩余11张票 2318元
- 一等座: 剩余2张票 1060元
...（完整显示所有内容）
```

### 2. NANO_AGENTSCOPE_VERBOSE

启用详细日志模式，显示更多调试信息。

**取值:**
- `0`: 简洁模式（默认）
- `1`: 详细模式

**显示内容:**
- ✅ LLM 请求详情（消息列表、可用工具）
- ✅ 工具调用参数（格式化显示）
- ✅ Token 使用统计（输入/输出/总计/耗时）

**设置方法:**

```bash
# 命令行设置
export NANO_AGENTSCOPE_VERBOSE=1

# 代码中设置
import os
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"
```

**示例输出:**

```
================================================================================
🤖 [LLM 请求]
================================================================================

📝 消息数量: 3
  1. [system] 你是一个列车助手，可以帮助用户查询列车信息...
  2. [user] 明天从北京到上海的车次中时间最短的是哪一个
  3. [assistant] 让我先获取当前日期...

🔧 可用工具: 8
  - get-current-date: 获取当前日期
  - get-station-code-of-citys: 根据城市名称获取站点代码
  - get-tickets: 查询列车车次信息
  ...

================================================================================

🔧 [调用工具] get-current-date
  参数: {}

[工具结果] get-current-date: 2025-12-30

📊 [Token 使用]
  输入: 1234 tokens
  输出: 156 tokens
  总计: 1390 tokens
  耗时: 1.25s
```

## 完整配置示例

### 示例 1: 开发调试模式（推荐）

```python
import os
import asyncio
from nano_agentscope import ReActAgent, DashScopeChatModel, Msg

async def main():
    # ========== 配置详细日志 ==========
    os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "0"  # 完整显示工具结果
    os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"         # 显示详细信息
    
    print("📋 日志配置:")
    print("  ✅ 工具结果: 完整显示")
    print("  ✅ 详细模式: 开启")
    print("  ✅ 显示: LLM请求 + 工具调用 + Token统计")
    print()
    
    # 创建 Agent
    agent = ReActAgent(
        name="助手",
        sys_prompt="你是一个智能助手",
        model=DashScopeChatModel(model_name="qwen-max"),
    )
    
    # 对话
    response = await agent(Msg(name="user", content="你好", role="user"))
    print(response.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 2: 生产环境模式

```python
import os
import asyncio
from nano_agentscope import ReActAgent, DashScopeChatModel, Msg

async def main():
    # ========== 简洁日志配置 ==========
    os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "500"  # 限制长度
    os.environ["NANO_AGENTSCOPE_VERBOSE"] = "0"           # 关闭详细模式
    
    # 创建 Agent
    agent = ReActAgent(
        name="助手",
        sys_prompt="你是一个智能助手",
        model=DashScopeChatModel(model_name="qwen-max"),
    )
    
    # 对话
    response = await agent(Msg(name="user", content="你好", role="user"))
    print(response.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例 3: 命令行配置

```bash
# 设置环境变量后运行
export NANO_AGENTSCOPE_LOG_MAX_LENGTH=0
export NANO_AGENTSCOPE_VERBOSE=1
python your_script.py

# 或者一行命令
NANO_AGENTSCOPE_LOG_MAX_LENGTH=0 NANO_AGENTSCOPE_VERBOSE=1 python your_script.py
```

## 日志模式对比

| 功能 | 简洁模式 (默认) | 详细模式 (VERBOSE=1) |
|------|-----------------|---------------------|
| Agent 响应 | ✅ | ✅ |
| 工具调用 | ✅ 简要显示 | ✅ 完整参数 |
| 工具结果 | ✅ 截断显示 | ✅ 根据配置 |
| LLM 请求详情 | ❌ | ✅ 完整显示 |
| 消息历史 | ❌ | ✅ 显示所有消息 |
| 可用工具列表 | ❌ | ✅ 显示工具和描述 |
| Token 统计 | ❌ | ✅ 详细统计 |
| 耗时信息 | ❌ | ✅ 显示 |

## 使用建议

### 开发阶段

```python
# 推荐配置
os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "0"
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"
```

**优点:**
- 完整查看所有数据
- 便于调试工具调用问题
- 理解 Agent 的推理过程
- 监控 Token 使用情况

### 测试阶段

```python
# 推荐配置
os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "2000"
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"
```

**优点:**
- 保留关键信息
- 避免日志过长
- 仍可查看详细统计

### 生产环境

```python
# 推荐配置
os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "500"
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "0"
```

**优点:**
- 日志简洁
- 性能更好
- 适合长时间运行

## 常见问题

### Q1: 为什么我看不到 LLM 的请求内容？

**A:** 需要设置 `NANO_AGENTSCOPE_VERBOSE=1` 才能看到详细的 LLM 请求信息。

### Q2: 工具结果被截断了怎么办？

**A:** 设置 `NANO_AGENTSCOPE_LOG_MAX_LENGTH=0` 即可显示完整内容。

### Q3: 如何在代码中动态控制日志级别？

**A:** 可以在运行时修改环境变量：

```python
import os

# 临时启用详细模式
old_verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0")
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"

# 执行需要详细日志的操作
await agent(msg)

# 恢复原设置
os.environ["NANO_AGENTSCOPE_VERBOSE"] = old_verbose
```

### Q4: 日志太多影响性能怎么办？

**A:** 
1. 使用默认配置（简洁模式）
2. 减小 `LOG_MAX_LENGTH` 的值
3. 将日志重定向到文件：`python script.py > output.log 2>&1`

### Q5: 如何保存日志到文件？

**A:**

```bash
# 方法 1: 重定向输出
python your_script.py > logs/output.log 2>&1

# 方法 2: 使用 tee 命令（同时显示和保存）
python your_script.py 2>&1 | tee logs/output.log

# 方法 3: 在代码中配置 logging
import logging
logging.basicConfig(filename='agent.log', level=logging.INFO)
```

## 实际案例

### 案例 1: 调试 MCP 工具调用

```python
import os
os.environ["NANO_AGENTSCOPE_LOG_MAX_LENGTH"] = "0"
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"

# 运行后你会看到：
# 1. LLM 收到了哪些消息
# 2. 有哪些工具可用
# 3. LLM 决定调用哪个工具
# 4. 工具的完整输入参数
# 5. 工具的完整返回结果
# 6. Token 使用情况
```

### 案例 2: 分析 Agent 性能

```python
import os
os.environ["NANO_AGENTSCOPE_VERBOSE"] = "1"

# 运行后查看 Token 统计：
# 📊 [Token 使用]
#   输入: 1234 tokens
#   输出: 156 tokens
#   总计: 1390 tokens
#   耗时: 1.25s

# 可以据此优化：
# - 减少输入 tokens（精简 prompt）
# - 减少工具数量（只注册必要的工具）
# - 使用更快的模型
```

## 参考资料

- [MCP 日志配置](./MCP_LOGGING.md)
- [Agent 开发指南](../README.md)
- [示例代码](../examples/mcp_demo.py)
