import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# 定义一个全局store来保存不同会话的历史记录
store = {}

def create_llm(temperature=0.7) -> AzureChatOpenAI:
    """创建一个 AzureChatOpenAI 模型实例"""
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
        api_key=os.getenv("AZURE_GPT4O_API_KEY"),
        api_version="2025-01-01-preview",
        model="gpt-4o",
        temperature=temperature,
        max_tokens=800
    )
    return llm

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取一个消息历史记录对象，支持会话 ID 来区分不同用户的历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

if __name__ == "__main__":
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个乐于助人的智能助手，擅长{ability}。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ]
    )

    chain = prompt | create_llm(0.7) | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # 设置会话ID
    session_id = "default_session"

    print("开始对话！输入 'exit' 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        result = chain_with_history.invoke(
            {
                "input": user_input,
                "ability": "回答各种问题"
            },
            config={"configurable": {"session_id": session_id}}
        )
        print("助手:", result)