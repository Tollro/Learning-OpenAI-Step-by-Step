# import chromadb
# from chromadb.config import Settings
# from your_script import OllamaBGE3EmbeddingFunction

# # 连接到已有数据库
# client = chromadb.PersistentClient(
#     path="./test_db",  # 与创建时的路径一致
#     settings=Settings(anonymized_telemetry=False)
# )

# # 获取集合（如果确定存在）
# collection = client.get_collection("treatments")

# # 查询（需要提供 embedding 函数，因为检索时会自动对查询文本做向量化）
# # 方式 A：使用您之前定义的 OllamaBGE3EmbeddingFunction 类
# embed_fn = OllamaBGE3EmbeddingFunction()
# collection = client.get_collection("treatments", embedding_function=embed_fn)

# # 打印结果
# print(results["documents"])