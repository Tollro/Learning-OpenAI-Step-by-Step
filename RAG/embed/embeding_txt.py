import os
import time
import urllib.error
import urllib.request

import ollama
import openai


def _ollama_base_url() -> str:
    """与 ollama Python 库一致，默认本机 11434。"""
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def wait_for_ollama(
    max_wait_seconds: float | None = None,
    poll_interval: float = 1.0,
) -> bool:
    """
    轮询 Ollama HTTP 服务，直到能访问 /api/tags 或超时。
    解决「刚开机 / 刚点完 Ollama 程序尚未监听」时的连接失败。
    """
    if max_wait_seconds is None:
        max_wait_seconds = float(os.getenv("OLLAMA_WAIT_MAX_SECONDS", "120"))
    if max_wait_seconds <= 0:
        return True

    base = _ollama_base_url()
    url = f"{base}/api/tags"
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)
    return False


def embed_with_ollama_bge_m3(
    text: str,
    model: str = "bge-m3",
    *,
    wait_for_server: bool = True,
    connect_retries: int = 3,
) -> list[float]:
    """
    用 Ollama 的 bge-m3 本地模型生成文本向量，直接使用 ollama 库。

    - wait_for_server: 为 True 时先等待 Ollama 进程开始监听（见 wait_for_ollama）。
    - 连接类错误会重试 connect_retries 次（间隔递增）。
    """
    if wait_for_server:
        max_wait = float(os.getenv("OLLAMA_WAIT_MAX_SECONDS", "120"))
        if not wait_for_ollama(max_wait_seconds=max_wait):
            raise ConnectionError(
                f"在 {max_wait} 秒内无法连接 Ollama（{ _ollama_base_url() }）。"
                "请先启动 Ollama 桌面端或在本机运行 `ollama serve`，再重试。"
            )

    for attempt in range(connect_retries):
        try:
            response = ollama.embeddings(model=model, prompt=text)
            return response["embedding"]
        except Exception as e:
            msg = str(e).lower()
            transient = "connect" in msg or "connection" in msg or "refused" in msg or "timed out" in msg
            if transient and attempt < connect_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


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
