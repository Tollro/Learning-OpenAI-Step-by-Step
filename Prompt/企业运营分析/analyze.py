import os
import json
from openai import AzureOpenAI

# 请在这里填写你的 Azure OpenAI 资源信息
AZURE_ENDPOINT = os.getenv("AZURE_GPT4O_ENDPOINT")  # 替换为你的终结点
AZURE_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")  # API Key
DEPLOYMENT_NAME = "gpt-4o"  # 模型部署名

def load_financial_data(file_path):
    """读取 JSON 格式的财务报表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_analysis_prompt(financial_data):
    """构造发送给 ChatGPT 的提示词，包含财务数据和分析要求"""
    # 将 JSON 数据转换为易于理解的文本描述
    # 可根据实际数据结构进行定制，这里仅作示例
    prompt = f"""
你是一位专业的财务分析师。请根据以下公司的财务报表数据，分析该企业的运行状况。重点关注：

1. 盈利能力（毛利率、净利率、ROE等）
2. 偿债能力（流动比率、速动比率、资产负债率）
3. 营运能力（存货周转率、应收账款周转率）
4. 现金流状况（经营现金流净额、自由现金流）
5. 整体评价与风险提示

财务报表数据（JSON 格式）：
{json.dumps(financial_data, indent=2, ensure_ascii=False)}

请提供清晰、专业的分析报告。
"""
    return prompt

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

def chat_with_azure_openai():
    """进行多轮对话的主函数"""
    client = initialize_client()
    if not client:
        return

    # 初始化对话历史，包含系统角色来定义助手的行为
    messages = [
        {"role": "system", "content": "你是一位专业的财务分析师。"}
    ]

    # 加载财务数据并构建提示词
    financial_data = load_financial_data("Prompt/企业运营分析/688027_performance_report.json")  # 替换为你的财务报表文件路径
    messages.append({"role": "user", "content": build_analysis_prompt(financial_data)}) # 将分析请求添加到对话历史中
    try:
            # 调用 Azure OpenAI 模型生成回复 [citation:4]
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.7,  # 控制随机性，范围0-2
                max_tokens=2000,   # 控制回复的最大长度
            )
            
            # 提取助手的回复
            assistant_message = response.choices[0].message.content
            
            # 将助手的回复也添加到对话历史中，实现多轮对话
            messages.append({"role": "assistant", "content": assistant_message})
            
            # 打印助手的回复
            print(f"\n🤖 助手: {assistant_message}")
            
            # 可选：打印 Token 使用情况，用于监控成本 [citation:6]
            print(f"[Token 使用: {response.usage.total_tokens} tokens]")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        # 如果发生错误，移除刚刚添加的用户消息，避免破坏对话历史
        if messages[-1]["role"] == "user":
            messages.pop()
        print("请稍后重试。")
    
    print("-" * 50)
    print("\n🤖 还有其他疑问吗？欢迎提问！ (输入 'quit' 或 'exit' 退出)")

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
        messages.append({"role": "user", "content": user_input})
        
        try:

            # 调用 Azure OpenAI 模型生成回复 [citation:4]
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.7,  # 控制随机性，范围0-2
                max_tokens=800,   # 控制回复的最大长度
            )
            
            # 提取助手的回复
            assistant_message = response.choices[0].message.content
            
            # 将助手的回复也添加到对话历史中，实现多轮对话
            messages.append({"role": "assistant", "content": assistant_message})
            
            # 打印助手的回复
            print(f"\n🤖 助手: {assistant_message}")
            
            # 可选：打印 Token 使用情况，用于监控成本 [citation:6]
            print(f"[Token 使用: {response.usage.total_tokens} tokens]")
            
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