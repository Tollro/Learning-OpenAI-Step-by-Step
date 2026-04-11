# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:55:25 2025

@author: liguo
"""
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.chat_models import ChatTongyi
import os
# 👇 核心计算函数，不被 @tool 装饰，避免递归调用工具对象
def _factorial(n: int) -> int:
    """内部阶乘计算函数，使用循环避免递归栈和工具调用问题"""
    if n < 0:
        raise ValueError("阶乘不支持负数")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# 👇 工具层函数，负责接收输入、类型转换、调用核心函数
@tool
def calculate_factorial(n: int) -> int:
    """计算给定数字的阶乘"""
    # 确保输入是整数（兼容字符串输入）
    if isinstance(n, str):
        n = int(n.strip())
    return _factorial(n)

# 工具列表
tools = [calculate_factorial]

# 初始化 Qwen 模型（请替换为你自己的 DashScope API Key）
llm = ChatTongyi(
    model="qwen-max",        # 也可用 qwen-turbo, qwen-plus
    temperature=0,
    #dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    # 推荐：在终端或系统环境变量中设置 DASHSCOPE_API_KEY
)

# 拉取 ReAct 提示词模板
prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 创建执行器
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行任务
if __name__ == "__main__":
    response = executor.invoke({"input": "计算5的阶乘"})
    print("\n" + "="*50)
    print("最终结果:", response["output"])

