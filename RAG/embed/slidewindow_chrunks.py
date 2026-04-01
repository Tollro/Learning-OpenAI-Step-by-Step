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
    n = len(text)
    if n == 0:
        return chunks

    # 只生成“完整长度”的窗口，避免末尾产生短 chunk / 空 chunk
    last_full_start = max(0, n - chunk_size)
    for i in range(0, last_full_start + 1, step):
        chunks.append(text[i:i + chunk_size])

    # 若最后一个完整窗口无法覆盖到文本末尾，则补齐末尾 chunk（保持顺序）
    if n > chunk_size and (not chunks or chunks[-1] != text[-chunk_size:]):
        chunks.append(text[-chunk_size:])
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