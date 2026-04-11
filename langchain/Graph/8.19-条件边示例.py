# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:46:52 2025

@author: liguo
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态结构
class State(TypedDict):
    value: int

def start(state: State) -> State:
    print("启动节点")
    return state

def process_a(state: State) -> State:
    print("执行路径A")
    return state

def process_b(state: State) -> State:
    print("执行路径B")
    return state

def decide_next(state: State) -> str:
    if state["value"] > 3:
        return "process_a"
    return "process_b"

# 创建 StateGraph 实例
workflow = StateGraph(State)

# 添加节点
workflow.add_node("start", start)
workflow.add_node("process_a", process_a)
workflow.add_node("process_b", process_b)

# 设置入口
workflow.set_entry_point("start")

# 添加条件边（注意：现在直接返回目标节点名）
workflow.add_conditional_edges(
    "start",
    decide_next,
    {  # 可选：显式映射，但如果你的 decide_next 直接返回节点名，可以省略这个字典
        "process_a": "process_a",
        "process_b": "process_b"
    }
)

# 添加普通边
workflow.add_edge("process_a", "process_b")
workflow.add_edge("process_b", END)  # 设置结束点

# 编译并运行
app = workflow.compile()
app.invoke({"value": 4})

