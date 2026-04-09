from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import os

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

def is_question(input: str) -> bool:
    """简单判断输入是否是一个问题"""
    return input.strip().endswith("?")

def question_prompt(input: str) -> str:
    """针对问题的提示语"""
    return f"：这是个问题，请回答{input}"

def elaborate_prompt(input: str) -> str:
    """针对非问题的提示语"""
    return f"：这是个陈述，请详细说明{input}"

if __name__ == "__main__":
    
    # 创建一个 RunnableBranch 实例
    branch = RunnableBranch(
        (is_question, RunnableLambda(question_prompt)),
        RunnableLambda(elaborate_prompt)  # 默认分支
        
    )

    result = branch.invoke("今天天气不错。")
    print("分支结果", result)

    # chain = branch | create_llm() | StrOutputParser()

    # # 测试输入
    # chain.invoke("今天天气不错。")
    # print("原始文本：")