# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:02:49 2025

@author: liguo
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import Tool
from langchain_community.chat_models.tongyi import ChatTongyi  # ✅ 正确导入方式
import os

# 设置你的 DashScope API Key（从阿里云百炼平台获取）
#os.environ["DASHSCOPE_API_KEY"] = "your-dashscope-api-key-here" #推荐：在终端或系统环境变量中设置 DASHSCOPE_API_KEY

# 定义简单工具
def echo(input: str) -> str:
    return f"你说了: {input}"

tools = [Tool(name="Echo", func=echo, description="回显输入的工具")]

# 创建带回调的Agent
llm = ChatTongyi(model="qwen-max", temperature=0)  # ✅ 使用 ChatTongyi

# 拉取 ReAct 提示模板
prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 自定义回调函数（使用 CallbackHandler 类方式）
from langchain_core.callbacks import BaseCallbackHandler

class CustomAgentCallback(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        print("\n=== 决策过程 ===")
        print(f"思考: {action.log}")
        print(f"选择工具: {action.tool}")
        print(f"工具输入: {action.tool_input}")

# 创建执行器
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[CustomAgentCallback()],
    handle_parsing_errors=True,  # 避免解析失败中断
)

# 执行
result = executor.invoke({"input": "你好"})
print("\n=== 最终输出 ===")
print(result["output"])

