from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template = "请简要解释{topic}的基本概念和应用。"
)

formatted_prompt = prompt.format(topic="人工智能")
print(formatted_prompt)