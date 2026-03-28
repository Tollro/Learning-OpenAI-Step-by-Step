from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in chat('deepseek-r1:1.5b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)