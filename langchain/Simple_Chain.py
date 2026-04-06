import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化Azure OpenAI模型    
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
    api_key=os.getenv("AZURE_GPT4O_API_KEY"),
    api_version="2025-01-01-preview",  # API 版本
    model="gpt-4o",
    temperature=0.7,
    max_tokens=800
)
print("✅ Azure OpenAI 模型初始化成功！")

prompt = PromptTemplate(
    input_variables=["question"],
    template="你是一个友好的AI助手。用户提问: {question}"
)

# 提取文本内容的输出解析器
parser = StrOutputParser()

# 管道运算符（|）将提示词和模型组合成一个 Chain
chain = prompt | llm | parser

# 测试链
result = chain.invoke("介绍下苹果（一种水果）")
print(result)