from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from model import create_gpt_call

def get_weather():
    """获取天气"""
    return "晴天"

# 共享状态
class AgentState(TypedDict):
    # add_messages 是一个内置的 reducer，新消息被追加，而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]


# 1. LLM 节点：负责调用大模型
def call_model(state: AgentState) -> dict:
    """调用大模型，并将响应包装为 AIMessage 返回。"""
    messages = state["messages"]
    # 大模型会理解上下文，并决定是生成文本还是发起工具调用
    response = llm_with_tools.invoke(messages)
    # 返回字典，状态中的 "messages" 字段将使用 add_messages 自动追加
    return {"messages": [response]}

llm = create_gpt_call()

tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)

# 2. Tool 节点：负责执行工具调用
# 可以使用 LangGraph 的预置组件，它能自动解析 AIMessage 中的 tool_calls 并执行
tool_node = ToolNode(tools)

# 条件边函数：决定下一步是去工具节点还是直接结束
def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """检查最后一条消息，决定下一步去向。"""
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息包含了工具调用，则路由到 "tool_node"
    if last_message.tool_calls:
        return "tool_node"
    # 否则，流程结束
    return END

# 实例化图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("llm", call_model)
workflow.add_node("tool_node", tool_node)

# 添加边
workflow.add_edge(START, "llm") # 从起点开始，直接到 LLM 节点
workflow.add_edge("tool_node", "llm") # 工具节点执行完后，必须回到 LLM 节点进行思考

# 添加条件边
workflow.add_conditional_edges(
    "llm", # 从 LLM 节点出发
    should_continue, # 根据这个函数的返回值决定
    {
        "tool_node": "tool_node", # 如果返回 "tool_node"，则去工具节点
        END: END                 # 如果返回 END，则结束
    }
)

# 编译图
app = workflow.compile()
