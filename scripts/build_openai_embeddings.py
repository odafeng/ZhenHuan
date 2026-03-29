"""Build embeddings.npy using OpenAI API (offline, run once).

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/build_openai_embeddings.py

This will:
1. Load segments.pkl from data/cache/index/
2. Filter garbage singles (same logic as LLMReranker)
3. Embed each segment's text_current via OpenAI text-embedding-3-small
4. Save embeddings.npy alongside segments.pkl

Upload embeddings.npy to your HF dataset (odafeng/zhenhuan-data) afterward.
"""

import pickle
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from openai import OpenAI

# --- Same garbage filter as reranker.py ---
_GARBAGE_PATTERN = re.compile(
    r"www|com\.|GTN|普級|菸酒|編曲|作曲|主唱|姚貝娜|敬請斟酌"
)


def _is_garbage(seg) -> bool:
    text = seg.text_current.strip()
    if len(text) <= 2:
        return True
    if seg.current_start < 55:
        return True
    if seg.current_start > 2400:
        return True
    if _GARBAGE_PATTERN.search(text):
        return True
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars < len(text) * 0.5 and len(text) > 3:
        return True
    return False


def main():
    index_dir = Path("data/cache/index")
    seg_path = index_dir / "segments.pkl"
    out_path = index_dir / "embeddings.npy"

    if not seg_path.exists():
        print(f"ERROR: {seg_path} not found. Run indexing first.")
        sys.exit(1)

    with open(seg_path, "rb") as f:
        all_segs = pickle.load(f)

    singles = [s for s in all_segs if s.window_tier == "single" and not _is_garbage(s)]
    print(f"Segments to embed: {len(singles)}")

    texts = [s.text_current for s in singles]

    # Embed in batches of 500 (OpenAI limit is 2048)
    client = OpenAI()
    model = "text-embedding-3-small"
    batch_size = 500
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)...")
        response = client.embeddings.create(input=batch, model=model)
        vecs = [item.embedding for item in response.data]
        all_embeddings.extend(vecs)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(out_path, embeddings)
    print(f"\nSaved {embeddings.shape} to {out_path}")
    print(f"Upload to HF: huggingface-cli upload odafeng/zhenhuan-data {out_path} embeddings.npy --repo-type dataset")


if __name__ == "__main__":
    main()
