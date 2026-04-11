# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:18:54 2025

@author: liguo
"""
from langgraph.graph import StateGraph, END

# 定义状态结构（可选，但推荐）
from typing import TypedDict

class State(TypedDict):
    start: bool
    processed: bool

# 创建图
workflow = StateGraph(State)

# 定义节点函数
def start_node(state):
    print("开始执行")
    return {"start": True}

def process_node(state):
    print("处理中...")
    return {"processed": True}

# 添加节点
workflow.add_node("begin", start_node)
workflow.add_node("process", process_node)

# 设置入口
workflow.set_entry_point("begin")

# 添加边
workflow.add_edge("begin", "process")
workflow.add_edge("process", END)  # 使用 END 表示结束

# 编译
app = workflow.compile()

# 执行
result = app.invoke({"start": True})
print("执行结果:", result)
