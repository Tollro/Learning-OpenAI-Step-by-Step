import os
from flask import Flask, request, jsonify, send_from_directory
from openai import AzureOpenAI

# 从环境变量读取 Azure OpenAI 配置
AZURE_ENDPOINT = os.getenv("AZURE_GPT4O_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")
DEPLOYMENT_NAME = "gpt-4o"

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    raise ValueError("请设置环境变量 AZURE_GPT4O_ENDPOINT 和 AZURE_GPT4O_API_KEY")

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

app = Flask(__name__)

@app.route('/')
def index():
    """返回独立的 index.html 文件"""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': '缺少 messages 字段'}), 400

        messages = data['messages']
        if not isinstance(messages, list):
            return jsonify({'error': 'messages 必须是列表'}), 400

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        assistant_reply = response.choices[0].message.content
        usage = response.usage

        return jsonify({
            'reply': assistant_reply,
            'usage': {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            } if usage else None
        })

    except Exception as e:
        print(f"错误: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)