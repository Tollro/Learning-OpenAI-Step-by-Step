import chromadb
from chromadb.config import Settings
import json
import os
import ollama
import requests
from tqdm import tqdm
from openai import AzureOpenAI

# def embed_with_ollama_bge_m3(texts: list[str]) -> list[list[float]]:
#     """
#     使用 Ollama BGE-M3 模型获取文本的向量表示
#     """
#     embeddings = []
#     for text in texts:
#         try:
#             response = ollama.embeddings(model="bge-m3", prompt=text)
#             embeddings.append(response["embedding"])
#         except Exception as e:
#             print(f"Embedding failed for text '{text[:50]}...': {e}")
#             # 根据需求决定：可以返回空向量，或重新抛出异常
#             # 这里返回零向量保持长度一致，或 raise
#             raise RuntimeError(f"Ollama embedding error: {e}")
#     return embeddings

# 批量处理GPU加速
def embed_batch_with_ollama_api(texts: list[str], batch_size: int = 1024) -> list[list[float]]:
    all_embeddings = []
    url = "http://localhost:11434/api/embed"
    total = len(texts)
    
    with tqdm(total=total, desc="语义向量化进度", unit="条") as pbar:
        for start_idx in range(0, total, batch_size):
            batch = texts[start_idx:start_idx + batch_size]
            payload = {"model": "bge-m3", "input": batch}
            
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend(result["embeddings"])
                pbar.update(len(batch))
            except Exception as e:
                print(f"批量请求失败 (start_idx={start_idx}): {e}")
                raise
    
    return all_embeddings

class My_chromadb:
    # 初始化，创建客户端，指定数据库路径
    def __init__(self, path: str, embedding_fn=embed_batch_with_ollama_api):
        self.db_path = path
        self.embedding_fn = embedding_fn
        self.chroma_client = chromadb.PersistentClient(
            path = path,
            settings = Settings(anonymized_telemetry=False)
        )
        print(f"chroma数据库初始化完成，路径: {self.db_path}")

    # 创建集合，指定集合名和向量化函数
    def add_collection(self, collection_name: str):
        # 防止重复创建同名集合，先检查是否存在
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        return collection
    
    # 添加数据元(分批处理，单次chroma最大存储量为5461条，超过会失败)
    def add_data(self, collection_name, ids, instructions: list[str], documents: list[str], chunk_size=4096):
        total = len(instructions)
        if not (len(ids) == total == len(documents)):
            raise ValueError("ids, instructions, documents 长度不一致")
        
        print(f"开始分块添加，总数据量: {total}，每块大小: {chunk_size}")
        
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            # 取出当前块的数据
            chunk_ids = ids[start:end]
            chunk_instructions = instructions[start:end]
            chunk_documents = documents[start:end]
            
            print(f"处理块 {start // chunk_size + 1}: 索引 {start}-{end-1}，共 {len(chunk_ids)} 条")
            
            # 向量化当前块（这里会调用 embedding_fn，它内部会进一步分小批请求 Ollama）
            embeddings = self.embedding_fn(chunk_instructions)
            
            # 立即添加到 ChromaDB
            self.chroma_client.get_or_create_collection(name=collection_name).add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_documents
            )
            print(f"块 {start // chunk_size + 1} 已完成，总进度: {end}/{total}")

        print("所有数据添加完成！")
    
    # embedding + 检索（一步到位）
    def search(self, collection_name: str, query_text: str, top_k=3):
        query_embedding = self.embedding_fn([query_text])
        results = self.chroma_client.get_or_create_collection(name=collection_name).query(
            query_embeddings=query_embedding,
            # 最相关的个结果
            n_results=top_k
        )
        return results
    
    # 查看集合数量
    def list_collections(self):
        return [c.name for c in self.chroma_client.list_collections()]

    # 查看集合数据条数
    def count_collection(self, collection_name: str):
        return self.chroma_client.get_or_create_collection(name=collection_name).count()
    
    # 删除集合中指定数据（按 id）
    def delete_by_ids(self, collection_name: str, ids: list[str]):
        """
        ids: list[str] 或 单个 str
        """
        if isinstance(ids, str):
            ids = [ids]

        self.chroma_client.get_or_create_collection(name=collection_name).delete(ids = ids)
        print(f"已删除 {len(ids)} 条数据")

    # 删除集合（调试用）
    def delete_collection(self, collection_name: str):
        self.chroma_client.delete_collection(collection_name)

def jsonl_load(file_path):
    rows: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows

def embed_text_from_item(item: dict) -> str:
    """
    用于算向量的文本：用户问题侧，便于检索时与 query 对齐。
    input 非空时拼在 instruction 后面。
    """
    instruction = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()
    if inp:
        return f"{instruction}\n{inp}"
    return instruction

def document_from_item(item: dict) -> str:
    """写入 Chroma 的 documents，便于检索后直接看到问答全文。"""
    instruction = item.get("instruction") or ""
    output = item.get("output") or ""
    return f"问题：{instruction}\n回答：{output}"

# Azure OpenAI 资源信息
AZURE_ENDPOINT = os.getenv("AZURE_GPT4O_ENDPOINT")  # 替换为你的终结点
AZURE_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")  # API Key
DEPLOYMENT_NAME = "gpt-4o"  # 模型部署名

def initialize_client():
    """初始化 Azure OpenAI 客户端"""
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version="2025-01-01-preview"  # API 版本
        )
        print("✅ Azure OpenAI 客户端初始化成功！")
        return client
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return None

def gpt4o_chat(messages: list[dict], model="gpt-4o", temperature=0.2, max_tokens=2048):
    """
    调用 Azure OpenAI 的 GPT-4o 模型进行聊天式文本生成。
    messages: [{"role": "system/user/assistant", "content": "text"}, ...]
    """
    try:
        client = initialize_client()
        if not client:
            raise RuntimeError("Azure OpenAI 客户端初始化失败，无法调用 GPT-4o")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT-4o 请求失败: {e}")
        raise RuntimeError(f"GPT-4o API error: {e}")

if __name__ == "__main__":
    # 加载数据（JSONL：一行一条）
    # data_path = "D:/Git/temps/train_zh.json"  # 数据路径
    # data_list = jsonl_load(data_path)

    # # 分批次参数, 提高运行速度，减少内存占用
    # BATCH_SIZE = 100   # 每批处理100条，根据内存情况调整
    # total = len(data_list)

    # instructions = [embed_text_from_item(item) for item in data_list]
    # documents = [document_from_item(item) for item in data_list]
    # ids = [str(idx) for idx in range(len(data_list))]

    # # 初始化Chroma数据库,保存到当前文件夹下的 chroma_db 子目录（硬盘）
    # treatment_db = My_chromadb("D:/Git/temps/treatment_chroma_db")
    # print("chromadb 已创建！")

    # treatment_db.add_collection("treatments")
    # print("添加新collection")

    # treatment_db.add_data("treatments", ids, instructions, documents)
    # print("已对treatments collection添加数据{}条".format(treatment_db.count_collection("treatments")))

    # print(treatment_db.list_collections())
    # print(treatment_db.chroma_client.get_collection(name="treatments").configuration)

    # # 删除指定 id 的数据
    # treatment_db.delete_by_ids("treatments", ["0"])
    # print(treatment_db.chroma_client.get_collection(name="treatments").count())
    # print(treatment_db.count_collection("treatments"))

    # （示例）连接本地数据库，查询
    mydb = My_chromadb(
        path="D:/Git/temps/treatment_chroma_db",
    )

    print(mydb.list_collections())
    print(mydb.count_collection("treatments"))

    userinput = ""
    history = []
    messages = [
        {"role": "system", "content":
        """
        你是一个医疗助手。
        你的任务是根据已知的信息回答用户的问题。
        确保你的回复完全依据下述已知信息。不要编造答案。
        如果信息不足以回答问题，请直接回复“信息不足，无法回答”。
        """
        }
    ]
    # 增加了历史问题输入
    while True:
        userinput = input("请输入查询内容：")
        history.append(userinput)
        
        if userinput.lower() in {"exit", "quit"}:
            print("退出查询。")
            break
        elif not userinput.strip():
            print("查询内容不能为空，请重新输入。")
            continue
        else:
            results = mydb.search("treatments", userinput, top_k=3)
            enable_llm = input("是否启用大模型生成回答？(y/n): ").strip().lower()
            if enable_llm == 'y':
                # 构建消息，使用检索结果作为上下文
                context = "\n".join(results["documents"][0])
                # messages.append(
                #     {"role": "user", "content": 
                #      f"""
                #      已知信息：【
                #      __{context}；
                #      历史搜索记录：
                #      {history}__】
                #      用户问题：【
                #      __{userinput}__】
                #      用中文回答用户问题，尽可能帮助用户解决问题。
                #      """}
                # )
                prompt = [
                    {"role": "system", "content":
                    """
                    你是一个医疗助手。
                    你的任务是根据已知的信息回答用户的问题。
                    确保你的回复依据下述已知信息。不要编造答案。
                    如果信息不足以回答问题，请直接回复“信息不足，无法回答”。
                    """
                    },
                    {"role": "user", "content": 
                     f"""
                     已知信息：【
                     __{context}；
                     历史搜索记录：
                     {history}__】
                     用户问题：【
                     __{userinput}__】
                     用中文回答用户问题，尽可能帮助用户解决问题。
                     """}]
                try:
                    assistant_message = gpt4o_chat(prompt)
                    
                    # messages.append({"role": "assistant", "content": assistant_message})

                    print(f"\n🤖 助手:：{assistant_message}")
                except Exception as e:
                    print(f"大模型调用失败：{e}")
                    print("检索结果：")
                    for i, doc in enumerate(results["documents"][0]):
                        print("-"*10,f"{results['ids'][0][i]}","-"*10)
                        print(doc)
            else:
                print("检索结果：")
                for i, doc in enumerate(results["documents"][0]):
                    print("-"*10,f"{results['ids'][0][i]}","-"*10)
                    print(doc)
            