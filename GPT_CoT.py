import os
from pyexpat.errors import messages
from openai import AzureOpenAI

# 请在这里填写你的 Azure OpenAI 资源信息
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
    
def build_cot_prompt(user_input):
    """构建包含 Chain of Thought 的提示词"""
    cot_prompt = f"""
    你是一个乐于助人的智能助手。当用户提出问题时，请先分析问题，列出解决问题的步骤，然后再给出最终答案。
    问题: {user_input}
    要求：
    1. 明确问题中的关键信息和需要求的结果；
    2. 每一步推理都需要说明为什么这么做（如“第一步：计算总消耗，因为需要先知道用掉多少”）；
    3. 不允许跳步，即使是简单计算也需要说明理由；
    4. 推理结束后给出一个清晰的答案，并且在开头加上“最终答案：”总结结果。
    """
    return cot_prompt

def chat_with_azure_openai():
    # 进行多轮对话的主函数
    client = initialize_client()
    if not client:
        return
    
    messages = [
        {"role": "system", "content": "你是一个乐于助人的智能助手。"}
    ]
    
    print("\n🤖 gpt-4o CoT推理已启动 (输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    while True:
        # 获取用户输入
        user_input = input("\n👤 你: ")
        
        # 退出条件
        if user_input.lower() in ['quit', 'exit']:
            print("👋 对话结束，再见！")
            break
        
        if not user_input.strip():
            print("⚠️ 输入不能为空，请重新输入。")
            continue

        # 将用户消息添加到对话历史中
        messages.append({"role": "user", "content": build_cot_prompt(user_input)})
        try:
            # 调用 Azure OpenAI 模型生成回复 [citation:4]
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.2,  # cot需要更确定的回复，降低随机性
                max_tokens=1000,   # 控制回复的最大长度
            )
            
            # 提取助手的回复
            assistant_message = response.choices[0].message.content
            
            # 将助手的回复也添加到对话历史中，实现多轮对话
            messages.append({"role": "assistant", "content": assistant_message})
            
            # 打印助手的回复
            print(f"\n🤖 助手: {assistant_message}")
            
            # 可选：打印 Token 使用情况，用于监控成本 [citation:6]
            print(f"[Token 使用: {response.usage.total_tokens} tokens]")

            # print(f"\n对话历史: {messages}")  # 查看完整对话历史
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            # 如果发生错误，移除刚刚添加的用户消息，避免破坏对话历史
            if messages[-1]["role"] == "user":
                messages.pop()
            print("请稍后重试。")

if __name__ == "__main__":
    # 检查必要的配置是否已填写
    if (not AZURE_ENDPOINT or not AZURE_API_KEY or not DEPLOYMENT_NAME):
        print("⚠️ 警告：请在代码顶部的 '配置区域' 填写你的 Azure OpenAI 终结点、API Key 和部署名。")
    else:
        chat_with_azure_openai()