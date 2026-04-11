# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:37:58 2025

@author: liguo
"""
from langgraph.graph import StateGraph

# 定义状态类型，这在新版本中是必需的
from typing import TypedDict

class State(TypedDict):
    value: int

workflow = StateGraph(State)

def start(state: State) -> State:
    print("开始工作流")
    return {"value": 0}

def increment(state: State) -> State:
    new_value = state["value"] + 1
    print(f"增加值到 {new_value}")
    return {"value": new_value}

def end(state: State) -> State:
    print(f"最终值: {state['value']}")
    return state

workflow.add_node("start", start)
workflow.add_node("increment", increment)
workflow.add_node("end", end)

workflow.set_entry_point("start")
workflow.add_edge("start", "increment")
workflow.add_edge("increment", "end")
workflow.set_finish_point("end")

app = workflow.compile()
result = app.invoke({"value": 0})  # 初始状态需要符合State类型定义
print("工作流执行结果:", result)

