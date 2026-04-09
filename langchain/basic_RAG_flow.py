from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_ollama import OllamaEmbeddings
import os

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_GPT4O_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_GPT4O_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = "2025-01-01-preview"

def create_llm(temprature=0.7):
    llm = AzureChatOpenAI(
        model="gpt-4o",
        temperature=temprature,
        max_tokens=800
    )
    return llm

def basic_rag_flow():
    # # 1. 加载文档
    # loader = TextLoader("./langchain/人事管理流程docu.txt", encoding='utf-8')
    # documents = loader.load()
    
    # # 2. 分割文档
    # text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    # texts = text_splitter.split_documents(documents)
    
    # 3. 创建嵌入和向量存储（使用通义千问的 Embedding 模型）
    embeddings = OllamaEmbeddings(
        model="bge-m3:latest"
    )
    # # 清空之前的向量数据库（如果存在）
    # vectorstore = Chroma.from_documents(
    #     texts, 
    #     embeddings,
    #     persist_directory="./langchain/text_chroma_db"  # 可选：指定持久化目录
    #     )

    vectorstore = Chroma(
        persist_directory="./langchain/text_chroma_db",
        embedding_function=embeddings,
        collection_name="langchain",
        collection_metadata={"hnsw:search_ef": 100}
        )
    # 4. 构建检索器, 默认top_k=4
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # 返回相似度最高的 6 个文档块
    )
    
    # 5. 创建提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于根据文本回答问题的助手。请根据以下上下文内容回答用户的问题。如果无法从上下文中找到答案，请如实告知。\n\n上下文：\n{context}"),
        ("human", "{question}")
    ])

    # 6. 创建 LLM
    llm = create_llm(temprature=0)
    
    # 7. 辅助函数：将检索到的文档列表格式化为字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 8. 使用 LCEL 管道构建 RAG 链
    qa_chain = (
        {
            "context": retriever | format_docs,   # 检索 + 格式化
            "question": RunnablePassthrough()     # 直接传递用户输入
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # 9. 执行查询并输出结果
    query = "入职有什么注意事项？顺便将获得到的上下文也展示一下。"
    result = qa_chain.invoke(query)
    print("问题：", query)
    print("回答：", result)
    
    return qa_chain

if __name__ == "__main__":
    basic_rag_flow()