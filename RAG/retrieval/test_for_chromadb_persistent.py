from xmlrpc import client

import chromadb
from chromadb.config import Settings
import json
import os
import ollama

def embed_with_ollama_bge_m3(texts: list[str]) -> list[list[float]]:
    """
    使用 Ollama BGE-M3 模型获取文本的向量表示
    """
    embeddings = []
    for text in texts:
        try:
            response = ollama.embeddings(model="bge-m3", prompt=text)
            embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Embedding failed for text '{text[:50]}...': {e}")
            # 根据需求决定：可以返回空向量，或重新抛出异常
            # 这里返回零向量保持长度一致，或 raise
            raise RuntimeError(f"Ollama embedding error: {e}")
    return embeddings

class My_chromadb:
    # 初始化，创建客户端，指定数据库路径
    def __init__(self, path: str, embedding_fn=embed_with_ollama_bge_m3):
        self.db_path = path
        self.embeding_fn = embedding_fn
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
    
    # 添加数据元
    def add_data(self, collection_name, ids, instructions: list[str], documents: list[str]):
        print(len(instructions))
        embeddings = self.embeding_fn(instructions)
        print(len(embeddings))
        self.chroma_client.get_or_create_collection(name=collection_name).add(
            ids = ids,
            embeddings = embeddings,
            documents = documents
        )
        print(self.chroma_client.get_or_create_collection(name=collection_name).count())
    
    # embedding + 检索（一步到位）
    def search(self, collection_name: str, query_text: str, top_k=3):
        query_embedding = self.embeding_fn([query_text])
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

if __name__ == "__main__":
    # 加载数据（JSONL：一行一条）
    data_path = "D:/Git/temps/train.json"  # 数据路径
    data_list = jsonl_load(data_path)

    # 调试时只处理前 N 条，避免全量跑 embedding 过久
    _max = 10
    if _max:
        data_list = data_list[: int(_max)]

    instructions = [embed_text_from_item(item) for item in data_list]
    documents = [document_from_item(item) for item in data_list]
    ids = [str(idx) for idx in range(len(data_list))]

    # 初始化Chroma数据库,保存到当前文件夹下的 chroma_db 子目录（硬盘）
    treatment_db = My_chromadb("./RAG/retrieval/test_db")
    print("chromadb 已创建！")

    treatment_db.add_collection("treatments")
    print("添加新collection")

    treatment_db.add_data("treatments", ids, instructions, documents)
    print("已对treatments collection添加数据{}条".format(treatment_db.count_collection("treatments")))

    # print(treatment_db.list_collections())
    # print(treatment_db.chroma_client.get_collection(name="treatments").configuration)

    # # 删除指定 id 的数据
    # treatment_db.delete_by_ids("treatments", ["0"])
    # print(treatment_db.chroma_client.get_collection(name="treatments").count())
    # print(treatment_db.count_collection("treatments"))

    # # （示例）连接本地数据库，查询
    # mydb = My_chromadb(
    #     path="./test_db",  # 与创建时的路径一致
    # )

    # print(mydb.list_collections())
    # print(mydb.count_collection("treatments"))

    # results = mydb.search("treatments", "如何治疗感冒？", top_k=3)
    # print("检索结果：")
    # # print(results)
    # for i, doc in enumerate(results["documents"][0]):
    #     print("-"*10,f"{results['ids'][0][i]}","-"*10)
    #     print(doc)