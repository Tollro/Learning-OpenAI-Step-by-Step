import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.messages import HumanMessage

def add_prefix(text: str) -> str:
    return "介绍: " + text

def create_gpt_call(temperature: float) -> AzureChatOpenAI:
    def call_llm(prompt: str) -> str:
        if isinstance(prompt, HumanMessage):
            content = prompt.content
        else:
            content = str(prompt)
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
            api_key=os.getenv("AZURE_GPT4O_API_KEY"),
            api_version="2025-01-01-preview",
            model="gpt-4o",
            temperature=temperature,
            max_tokens=800
        )
        if not llm:
            raise ValueError("Azure OpenAI 模型初始化失败，请检查环境变量设置。")
        else:
            print("✅ Azure OpenAI 模型初始化成功！")
            parser = StrOutputParser()
            chain_chat = llm | parser
            return chain_chat.invoke(content)
    return RunnableLambda(call_llm)

# # 初始化Azure OpenAI模型    
# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
#     api_key=os.getenv("AZURE_GPT4O_API_KEY"),
#     api_version="2025-01-01-preview",  # API 版本
#     model="gpt-4o",
#     temperature=0.7,
#     max_tokens=800
# )
# print("✅ Azure OpenAI 模型初始化成功！")

prompt = PromptTemplate(
    input_variables=["question"],
    template="请简要介绍说明{question}。"
)

parallel_processor = RunnableParallel(
    {
        "创意回答：": create_gpt_call(0.7),
        "严谨回答：": create_gpt_call(0.2)
    }
)
chain = prompt | parallel_processor
result = chain.invoke(HumanMessage(content="如何写一首思乡的诗？"))
print("创意回答：", result["创意回答："],'\n',"-"*20,'\n')
print("严谨回答：", result["严谨回答："])


# # 提取文本内容的输出解析器
# parser = StrOutputParser()

# # 创建一个可运行对象，用于添加前缀
# add_prefix_runnable = RunnableLambda(add_prefix)

# # 管道运算符（|）将提示词和模型组合成一个 Chain
# chain = prompt | llm | parser | add_prefix_runnable

# # 测试链
# result = chain.invoke("介绍下苹果公司")
# print(result)