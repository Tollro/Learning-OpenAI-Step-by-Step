import os
import ollama
import openai

def embed_with_ollama_bge_m3(text: str, model: str = "bge-m3") -> list[float]:
    """
    用 Ollama 的 bge-m3 本地模型生成文本向量，直接使用 ollama 库
    """
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def embed_with_azure_embeddings(
    text: str,
    endpoint: str,
    api_key: str,
    api_version: str = "2025-01-01-preview",
    deployment: str = "text-embedding-3-small",
) -> list[float]:
    """
    用 Azure OpenAI 的 embedding 部署生成文本向量。
    推荐直接用 openai 官方库更简洁、安全，支持错误处理和未来兼容性。
    需要 openai >= 1.0.0
    """
    if not endpoint:
        raise ValueError("Azure endpoint is empty. Set AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Azure api_key is empty. Set AZURE_OPENAI_API_KEY.")
    if not deployment:
        raise ValueError("Azure embeddings deployment is empty. Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT.")

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    response = client.embeddings.create(
        model=deployment,
        input=text,
    )
    # 返回第一个向量
    return response.data[0].embedding

# 示例: 选择一个接口进行嵌入
if __name__ == "__main__":
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-small")

    text = "三体是一部著名的科幻小说。"
    print("Ollama bge-m3 embedding:")
    try:
        vector = embed_with_ollama_bge_m3(text)
        print(vector[:8], "...(total:", len(vector), ")")
    except Exception as e:
        print("Ollama embedding failed:", e)

    # # Azure API 示例（需要实际的 API 参数）
    # print("Azure embeddings:")
    # try:
    #     vector = embed_with_azure_embeddings(
    #         text,
    #         AZURE_ENDPOINT,
    #         AZURE_API_KEY,
    #         deployment=AZURE_EMBEDDINGS_DEPLOYMENT,
    #     )
    #     print(vector[:8], "...(total:", len(vector), ")")
    # except Exception as e:
    #     print("Azure embedding failed:", e)