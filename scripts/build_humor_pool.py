"""Build humor_pool.pkl — pre-filter segments with meme potential (offline, run once).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/build_humor_pool.py

Cost: ~$0.55 one-time (Haiku scans all 53K segments).
Output: data/cache/index/humor_pool.pkl (~3-5K segments with meme potential).
Upload to HF afterward:
    huggingface-cli upload odafeng/zhenhuan-data data/cache/index/humor_pool.pkl humor_pool.pkl --repo-type dataset
"""

import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

POOL_PROMPT = """你是「甄嬛傳迷因產生器」的預篩系統。

你的任務：從甄嬛傳台詞中，篩選出「有潛力當梗圖回應」的台詞。

有潛力的台詞特徵：
- 有雙關或歧義（能在不同語境下產生新意思）
- 語氣強烈（命令、感嘆、質問、諷刺）
- 有畫面感（能想像搭配嚴肅古裝表情的反差）
- 能當萬用回覆（放在各種現代口語後面都有接話感）
- 經典語錄級（「臣妾做不到」「賤人就是矯情」等）

不要選的：
- 太短或太普通（「是」「嗯」「皇上」）
- 純敘事無情緒（「皇上駕到」「擺駕回宮」）
- OCR 雜訊或字幕資訊

回傳有潛力的台詞編號，用逗號分隔。
如果這批都沒有，回傳 0。
只回傳數字，不要解釋。"""

# Same garbage filter as reranker.py
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


def _filter_chunk(client, chunk_lines, chunk_segments, chunk_id, total_chunks):
    """Send one chunk to Haiku for humor pre-filtering."""
    text = "\n".join(chunk_lines)
    user_msg = f"以下是甄嬛傳台詞，請篩選有梗圖回應潛力的：\n{text}\n\n回傳有潛力的編號："

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=POOL_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.3,
        )
        raw = response.content[0].text.strip()
        if raw == "0":
            print(f"  [{chunk_id}/{total_chunks}] 0 picks")
            return []

        indices = []
        for x in raw.split(","):
            try:
                idx = int(x.strip()) - 1
                if 0 <= idx < len(chunk_segments):
                    indices.append(idx)
            except ValueError:
                continue

        picks = [chunk_segments[i] for i in indices]
        print(f"  [{chunk_id}/{total_chunks}] {len(picks)} picks")
        return picks
    except Exception as e:
        print(f"  [{chunk_id}/{total_chunks}] ERROR: {e}")
        return []


def main():
    index_dir = Path("data/cache/index")
    seg_path = index_dir / "segments.pkl"
    out_path = index_dir / "humor_pool.pkl"

    if not seg_path.exists():
        print(f"ERROR: {seg_path} not found.")
        sys.exit(1)

    with open(seg_path, "rb") as f:
        all_segs = pickle.load(f)

    singles = [s for s in all_segs if s.window_tier == "single" and not _is_garbage(s)]
    print(f"Total segments to scan: {len(singles)}")

    # Build chunks
    chunk_size = 3000
    lines = [f"{i+1}. {s.text_current}" for i, s in enumerate(singles)]

    chunks = []
    for start in range(0, len(lines), chunk_size):
        chunk_lines = lines[start:start + chunk_size]
        # Renumber within chunk
        renumbered = [f"{i+1}. {l.split('. ', 1)[1]}" for i, l in enumerate(chunk_lines)]
        chunk_segments = singles[start:start + chunk_size]
        chunks.append((renumbered, chunk_segments))

    total_chunks = len(chunks)
    print(f"Split into {total_chunks} chunks of ~{chunk_size} lines each")

    client = anthropic.Anthropic()
    all_picks = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_filter_chunk, client, cl, cs, i + 1, total_chunks): i
            for i, (cl, cs) in enumerate(chunks)
        }
        for future in futures:
            picks = future.result()
            all_picks.extend(picks)

    # Deduplicate by segment_id
    seen = set()
    unique = []
    for s in all_picks:
        if s.segment_id not in seen:
            seen.add(s.segment_id)
            unique.append(s)

    print(f"\nTotal humor pool: {len(unique)} segments (from {len(singles)} singles)")

    with open(out_path, "wb") as f:
        pickle.dump(unique, f)

    print(f"Saved to {out_path}")
    print(f"Upload: huggingface-cli upload odafeng/zhenhuan-data {out_path} humor_pool.pkl --repo-type dataset")


if __name__ == "__main__":
    main()
