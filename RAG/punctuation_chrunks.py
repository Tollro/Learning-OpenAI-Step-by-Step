import os
import re

def clean_text(text):
    # 去除所有空格和换行
    return text.replace(' ', '').replace('\n', '').replace('\r', '')

def punctuation_chunks(text):
    """
    按标点断句。适用于中文文本，常见句号、问号、感叹号等标点。
    """
    # 以中文/英文句号、问号、感叹号、分号等进行切分，标点留在句子尾部
    pattern = r'([^。！？；!?;]*[。！？；!?;])'
    sentences = re.findall(pattern, text)
    # 有些最后一句可能没有以标点结束，单独加上
    tail = re.sub(pattern, '', text)
    if tail.strip() != '':
        sentences.append(tail)
    return sentences

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '三体简介.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = clean_text(raw_text)
    chunks = punctuation_chunks(text)
    for idx, chunk in enumerate(chunks):
        print(f'Chunk {idx + 1}:')
        print(chunk)
        print('-' * 40)