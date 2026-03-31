import os

def clean_text(text):
    # 去除所有空格和换行
    return text.replace(' ', '').replace('\n', '').replace('\r', '')

def sliding_window_chunks(text, chunk_size, step):
    """
    采用滑动窗口产生chunk片段
    chunk_size: 每个chunk的字符数
    step: 滑动步长
    """
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, step):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    # 若最后剩下的部分长度大于0小于chunk_size，则补充最后一个chunk
    if len(text) % step != 0 and len(text) > chunk_size:
        last_chunk = text[-chunk_size:]
        if last_chunk not in chunks:
            chunks.append(last_chunk)
    return chunks

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '三体简介.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = clean_text(raw_text)
    chunk_size = 100  # 每个chunk包含的字符数
    step = 50         # 窗口滑动步长

    chunks = sliding_window_chunks(text, chunk_size, step)
    for idx, chunk in enumerate(chunks):
        print(f'Chunk {idx + 1}:')
        print(chunk)
        print('-' * 40)