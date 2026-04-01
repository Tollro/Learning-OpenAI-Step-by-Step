import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def merge_too_small_chunks(
    chunks: list[str],
    *,
    min_chars: int = 5,
    punctuation_only: set[str] | None = None,
) -> list[str]:
    """
    Post-process splitter output to avoid tiny fragments (e.g. a single '。').
    """
    if punctuation_only is None:
        punctuation_only = {"。", "！", "？", "；", "，", ".", "!", "?", ";", ",", "、"}

    merged: list[str] = []
    pending_prefix = ""

    for raw in chunks:
        chunk = raw.strip()
        if not chunk:
            continue

        is_punct_only = chunk in punctuation_only
        is_too_small = len(chunk) < min_chars

        # If we're at the beginning (no previous chunk), cache tiny fragments
        # and prepend them to the next real chunk.
        if not merged and (is_punct_only or is_too_small):
            pending_prefix += chunk
            continue

        if pending_prefix:
            chunk = pending_prefix + chunk
            pending_prefix = ""

        if merged and (is_punct_only or is_too_small):
            merged[-1] = merged[-1].rstrip() + chunk
        else:
            merged.append(chunk)

    if pending_prefix:
        # Edge case: input is only punctuation/tiny fragments.
        merged.append(pending_prefix)

    return merged


def main() -> None:
    file_path = os.path.join(os.path.dirname(__file__), "三体简介.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=15,
        # Do NOT include "" here, otherwise it can fall back to character-level splits
        # and produce single punctuation chunks.
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ",],
        keep_separator=True,
    )

    chunks = splitter.split_text(text)
    chunks = merge_too_small_chunks(chunks, min_chars=5)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Chunk {idx}:")
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    main()
