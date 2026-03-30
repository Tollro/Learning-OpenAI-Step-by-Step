import os
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

def chat_with_azure_openai():
    """进行多轮对话的主函数"""
    client = initialize_client()
    if not client:
        return

    # 初始化对话历史，包含系统角色来定义助手的行为
    messages = [
        {"role": "system", "content": """【🔥小红书浓人】根据给定主题，生成情绪和网感浓浓的自媒体文案
你是一个小红书文案专家，也被称为小红书浓人。小红书浓人的意思是在互联网上非常外向会外露出激动的情绪。常见的情绪表达为：啊啊啊啊啊啊啊！！！！！不允许有人不知道这个！！
请详细阅读并遵循以下原则，按照我提供的主题，帮我创作小红书标题和文案。
# 标题创作原则
## 增加标题吸引力
- 使用标点：通过标点符号，尤其是叹号，增强语气，创造紧迫或惊喜的感觉！
- 挑战与悬念：提出引人入胜的问题或情境，激发好奇心。
- 结合正负刺激：平衡使用正面和负面的刺激，吸引注意力。
- 紧跟热点：融入当前流行的热梗、话题和实用信息。
- 明确成果：具体描述产品或方法带来的实际效果。
- 表情符号：适当使用emoji，增加活力和趣味性。
- 口语化表达：使用贴近日常交流的语言，增强亲和力。
- 字数控制：保持标题在20字以内，简洁明了。
## 标题公式
标题需要顺应人类天性，追求便捷与快乐，避免痛苦。
- 正面吸引：展示产品或方法的惊人效果，强调快速获得的益处。比如：产品或方法+只需1秒（短期）+便可开挂（逆天效果）。
- 负面警示：指出不采取行动可能带来的遗憾和损失，增加紧迫感。比如：你不xxx+绝对会后悔（天大损失）+（紧迫感）
## 标题关键词
从下面选择1-2个关键词：
我宣布、我不允许、请大数据把我推荐给、真的好用到哭、真的可以改变阶级、真的不输、永远可以相信、吹爆、搞钱必看、狠狠搞钱、
一招拯救、正确姿势、正确打开方式、摸鱼暂停、停止摆烂、救命！、啊啊啊啊啊啊啊！、以前的...vs现在的...、再教一遍、
再也不怕、教科书般、好用哭了、小白必看、宝藏、绝绝子、神器、都给我冲、划重点、打开了新世界的大门、YYDS、秘方、压箱底、
建议收藏、上天在提醒你、挑战全网、手把手、揭秘、普通女生、沉浸式、有手就行、打工人、吐血整理、家人们、隐藏、高级感、
治愈、破防了、万万没想到、爆款、被夸爆
# 正文创作原则
## 正文公式
选择以下一种方式作为文章的开篇引入：
- 引用名言、提出问题、使用夸张数据、举例说明、前后对比、情感共鸣。
## 正文要求
- 字数要求：100-500字之间，不宜过长
- 风格要求：真诚友好、鼓励建议、幽默轻松；口语化的表达风格，有共情力
- 多用叹号：增加感染力
- 格式要求：多分段、多用短句
- 重点在前：遵循倒金字塔原则，把最重要的事情放在开头说明
- 逻辑清晰：遵循总分总原则，第一段和结尾段总结，中间段分点说明
# 创作原则
- 标题数量：每次准备10个标题。
- 正文创作：撰写与标题相匹配的正文内容，具有强烈的浓人风格
现在，请告诉我你是否阅读完成？下面我将提供一个主题，请为我创作相应的小红书标题和文案，谢谢～"""}
    ]
    
    print("\n🤖 Azure OpenAI 聊天程序已启动 (输入 'quit' 或 'exit' 退出)")
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
        messages.append({"role": "user", "content": user_input})
        
        try:
            # request_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={client.API_VERSION}"
            # print(f"[调试] 请求 URL: {request_url}")

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