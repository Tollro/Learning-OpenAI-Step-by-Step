#Multi-Query 多路召回
from operator import itemgetter

from langchain_classic.load import dumps, loads
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_ollama import OllamaEmbeddings

from model import create_gpt_call

#获得访问大模型和嵌入模型客户端
llm = create_gpt_call()

# 加载文档
loader = TextLoader("./RAG/advanced/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings_model = OllamaEmbeddings(
    model="bge-m3:latest"
)
# 创建向量数据库
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embeddings_model)
# 创建检索器
retriever = vectorstore.as_retriever()

relevant_docs= retriever.invoke('deepseek的应用场景')
print(relevant_docs)
print(len(relevant_docs))


#创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

chain = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("--------------优化前-------------------")
response = chain.invoke({"question": "deepseek的应用场景"})
print(response)

print("--------------开始优化-------------------")
#方法一：使用langchain的MultiQueryRetriever
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
retrieval_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
unique_docs = retrieval_from_llm.invoke({"question":'deepseek的应用场景'})
print(unique_docs)
print(len(unique_docs))

#方法二：自定义prompt
# prompt模版
template = """你是一个AI语言模型助手。你的任务是生成5个给定用户问题的不同版本，以从向量中检索相关文档
数据库。通过对用户问题产生多种观点，你的目标是提供帮助用户克服了基于距离的相似性搜索的一些限制。
提供了这些用换行符隔开的可选问题。原始问题: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

response = generate_queries.invoke({"question":'deepseek的应用场景'})
print(response)

def get_unique_union(documents: list[list]):
    """ 获取检索文档的唯一并集 """
    # 将列表中的列表展开，并将每个 Document 转换为字符串
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 文档去重
    unique_docs = list(set(flattened_docs))
    # 返回去重后的文档列表
    return [loads(doc) for doc in unique_docs]

# 进行检索
'''假设 generate_queries 生成了以下查询列表：["deepseek的应用场景", "deepseek的使用方法", "deepseek的优势"]
'''
question = "deepseek的应用场景"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
print(len(docs))

print("--------------优化后-------------------")
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, 
    "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)


question = "deepseek的应用场景"
response = final_rag_chain.invoke({"question":question})
print(response)