"""LLM-based selector: directly pick funny responses from all segments, no embedding needed."""

import base64
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from schemas import Segment


PICK_PROMPT = """你是「甄嬛傳迷因產生器」。

使用者輸入一句現代口語，你要從甄嬛傳的台詞中，選出最適合拿來當「幽默回應」的台詞。

核心：不是找「相似」的台詞，而是找放在使用者那句話後面、會讓人笑出來的回覆。
好笑來自「語境錯位」——古裝劇正經台詞被當成現代口語的回覆，產生歧義、雙關或荒謬感。

經典範例：
- 「發現他很大」→「快拿出來看看」「幾次差點活不下來」「既然是好東西你就多用」
- 「今天加班到很晚」→「臣妾做不到」「這日子沒法過了」
- 「你是在酸我嗎」→「酸」「臣妾沒試過 不知道是酸的」

評分標準：
1. 回應感：放在使用者那句話後面，像在接話/回應
2. 語境反差：原意正經，當回覆後產生新的（通常是黃色或諷刺的）意思
3. 畫面感：想像搭配古裝劇的嚴肅畫面，反差越大越好

回傳你選的台詞編號（最多 3 個），用逗號分隔，最好笑的排前面。
如果這批台詞裡沒有好笑的回應，回傳 0。
只回傳數字，不要解釋。"""


FINAL_VISION_PROMPT = """你是「甄嬛傳迷因產生器」的最終評審。

你會看到多組候選：每組包含一張甄嬛傳截圖和對應的台詞。
使用者輸入了一句現代口語，你要選出「截圖＋台詞」搭配起來最好笑的回應。

評分標準：
1. 台詞接話感：放在使用者那句話後面，像在回嘴/吐槽
2. 語境反差：原意正經，當回覆後產生新的（黃色、諷刺、荒謬）意思
3. 畫面加分：角色表情嚴肅/憤怒/無奈/陰險 搭配台詞產生的反差越大越好
4. 多樣性：不同類型的笑點比重複同一個梗好

只回傳編號，用逗號分隔，最好笑的排前面。不要解釋。"""


# Patterns to filter out garbage segments
_GARBAGE_PATTERN = re.compile(
    r"www|com\.|GTN|普級|菸酒|編曲|作曲|主唱|姚貝娜|敬請斟酌"
)


def _parse_indices(raw: str, n: int) -> list[int]:
    """Extract 1-based indices from an LLM reply, return valid 0-based indices.

    Tolerant of full-width commas, trailing punctuation, and stray text
    (e.g. "我選 12、45", "3,7,", "編號: 2 5 9"). "0" means no pick.
    """
    nums = re.findall(r"\d+", raw)
    out = []
    for tok in nums:
        idx = int(tok) - 1
        if 0 <= idx < n and idx not in out:
            out.append(idx)
    return out


def _read_frame_bytes(frame_path: str) -> bytes | None:
    """Read a frame's raw bytes for vision rerank.

    When PUBLIC_BASE_URL is set (Cloudflare container), frames live in R2
    and are fetched over HTTP via the Worker. Otherwise read from local
    disk (CLI, local dev, tests). Returns None on any failure so the
    caller can degrade that candidate to text-only.
    """
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if base:
        name = Path(frame_path).name
        try:
            with urllib.request.urlopen(f"{base}/frames/{name}", timeout=10) as resp:
                return resp.read()
        except Exception:
            return None
    p = Path(frame_path)
    if p.exists():
        return p.read_bytes()
    return None


def _is_garbage(seg: Segment) -> bool:
    """Check if a segment is garbage (credits, OCR noise, too short)."""
    text = seg.text_current.strip()
    # Too short to be a meaningful reply
    if len(text) <= 2:
        return True
    # Opening credits / theme song (first 55 seconds of each episode)
    if seg.current_start < 55:
        return True
    # Ending credits (after 40 minutes)
    if seg.current_start > 2400:
        return True
    # OCR noise: too many digits, special chars, or known garbage keywords
    if _GARBAGE_PATTERN.search(text):
        return True
    # Mostly non-Chinese characters (OCR artifacts)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars < len(text) * 0.5 and len(text) > 3:
        return True
    return False


class LLMReranker:
    """Directly select funny responses from all segments using LLM."""

    def __init__(
        self,
        fast_model: str = "claude-haiku-4-5-20251001",
        smart_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.client = anthropic.Anthropic()
        self.fast_model = fast_model    # Text scan + vision rerank
        self.smart_model = smart_model  # Reserved for future use
        self._segments: list[Segment] | None = None
        self._singles: list[Segment] | None = None
        self._scan_pool: list[Segment] | None = None  # Pre-filtered humor candidates

    def load_segments(self, cache_dir: Path) -> None:
        """Load segments from cache. Prefers humor_pool.pkl (small, pre-filtered) over full scan."""
        import pickle
        seg_path = cache_dir / "index" / "segments.pkl"
        if not seg_path.exists():
            raise FileNotFoundError(f"Segments not found: {seg_path}")
        with open(seg_path, "rb") as f:
            all_segs = pickle.load(f)

        before = len([s for s in all_segs if s.window_tier == "single"])
        self._segments = [s for s in all_segs if not (s.window_tier == "single" and _is_garbage(s))]
        self._singles = [s for s in self._segments if s.window_tier == "single"]
        after = len(self._singles)
        print(f"  Loaded segments: {after} usable singles (filtered {before - after} garbage)")

        # Try to load pre-filtered humor pool (much smaller → cheaper runtime)
        pool_path = cache_dir / "index" / "humor_pool.pkl"
        if pool_path.exists():
            with open(pool_path, "rb") as f:
                self._scan_pool = pickle.load(f)
            print(f"  Loaded humor pool: {len(self._scan_pool)} candidates (fast mode)")
        else:
            self._scan_pool = None
            print(f"  No humor_pool.pkl found, will full-scan {after} segments (slow & expensive)")

    def rerank(
        self, query: str, _candidates: list, top_k: int = 5
    ) -> list:
        """Not used in direct mode. See select() instead."""
        return self.select(query, top_k)

    def select(self, query: str, top_k: int = 5) -> list[dict]:
        """Select funny responses using Haiku batch scan."""
        if not self._segments:
            raise RuntimeError("Call load_segments() first")

        # Use pre-filtered humor pool if available, else full scan
        singles = self._scan_pool or self._singles or [s for s in self._segments if s.window_tier == "single"]
        pool_mode = "humor pool" if self._scan_pool else "full scan"

        # Small batches so Haiku can reliably locate indices.
        # batch_size here = episodes per batch group; the real per-prompt
        # cap is enforced inside _pick_from_batch (chunk_size).
        batches = self._group_by_episode_batch(singles, batch_size=10)

        print(f"  Haiku {pool_mode}: {len(singles)} segments across {len(batches)} batches...")

        batch_winners: list[Segment] = []
        done = 0
        total_batches = sum(1 for segs in batches.values() if segs)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._pick_from_batch, query, segs): name
                for name, segs in batches.items()
                if segs
            }
            for future in futures:
                done += 1
                name = futures[future]
                results = future.result()
                batch_winners.extend(results)
                print(f"  [{done}/{total_batches}] {name}: {len(results)} picks", flush=True)

        if not batch_winners:
            # Safety net: Stage 1 found nothing (query too off, or all batches
            # returned 0). Don't return an empty gallery — sample some usable
            # singles so the user always sees candidate memes.
            print("  Stage 1 empty, falling back to sampled singles", flush=True)
            import random
            pool = self._scan_pool or self._singles or []
            with_frames = [s for s in pool if s.frame_path]
            sample_src = with_frames or pool
            if not sample_src:
                return []
            k = min(top_k, len(sample_src))
            chosen = random.sample(sample_src, k)
            from schemas import SearchResult
            return [SearchResult(segment=s, score=0.0) for s in chosen]

        print(f"  Stage 1 done: {len(batch_winners)} picks", flush=True)

        # Stage 2: Haiku vision — pick best with screenshot context
        if len(batch_winners) <= top_k:
            final = batch_winners
        else:
            print(f"  Stage 2: Haiku vision rerank...", flush=True)
            final = self._final_select_with_vision(query, batch_winners, top_k)

        from schemas import SearchResult
        return [SearchResult(segment=s, score=1.0) for s in final]

    def _group_by_episode_batch(
        self, segments: list[Segment], batch_size: int = 10
    ) -> dict[str, list[Segment]]:
        batches: dict[str, list[Segment]] = {}
        for s in segments:
            ep_num = int(s.episode_id.replace("ep", ""))
            batch_idx = (ep_num - 1) // batch_size
            batch_start = batch_idx * batch_size + 1
            batch_end = min(batch_start + batch_size - 1, 76)
            name = f"ep{batch_start:02d}-{batch_end:02d}"
            batches.setdefault(name, []).append(s)
        return batches

    def _pick_from_batch(self, query: str, segments: list[Segment]) -> list[Segment]:
        """Stage 1: LLM picks up to 3 best responses from a batch (text-only).

        Splits the batch into small chunks (<= chunk_size lines) so Haiku can
        locate indices reliably. Each chunk is renumbered 1..len(chunk) locally,
        and the reply is parsed against the chunk, not the whole batch.
        """
        chunk_size = 80
        all_picks: list[Segment] = []

        for start in range(0, len(segments), chunk_size):
            chunk_segments = segments[start:start + chunk_size]
            # Renumber 1..N within this chunk (indices map directly to chunk_segments)
            text = "\n".join(
                f"{i}. {s.text_current}" for i, s in enumerate(chunk_segments, 1)
            )
            user_msg = (
                f"使用者說：「{query}」\n\n"
                f"以下是甄嬛傳台詞：\n{text}\n\n"
                f"選最適合當幽默回應的（最多3個），只回編號："
            )

            try:
                response = self.client.messages.create(
                    model=self.fast_model,
                    max_tokens=50,
                    system=PICK_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.5,
                )
                raw = response.content[0].text.strip()
                for idx in _parse_indices(raw, len(chunk_segments)):
                    all_picks.append(chunk_segments[idx])
            except anthropic.BadRequestError as e:
                if "credit balance" in str(e):
                    raise RuntimeError(
                        "Anthropic API 餘額不足，請至 "
                        "https://console.anthropic.com/settings/billing 加值"
                    ) from e
                print(f"    Batch error: {e}", flush=True)
            except Exception as e:
                print(f"    Batch error: {e}", flush=True)

        return all_picks

    def _final_select_with_vision(
        self, query: str, winners: list[Segment], top_k: int
    ) -> list[Segment]:
        """Stage 2: Final selection using vision — Claude sees screenshots + text."""
        # Build multimodal content: for each candidate, include image + text
        content_blocks = [
            {"type": "text", "text": f"使用者說：「{query}」\n\n以下是候選回應，每個包含截圖和台詞：\n"},
        ]

        valid_winners = []
        for i, s in enumerate(winners, 1):
            img_bytes = _read_frame_bytes(s.frame_path) if s.frame_path else None
            if img_bytes:
                img_data = base64.b64encode(img_bytes).decode()
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- 候選 {len(valid_winners) + 1} ---",
                })
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data,
                    },
                })
                content_blocks.append({
                    "type": "text",
                    "text": f"台詞：「{s.text_current}」（{s.episode_id}）",
                })
                valid_winners.append(s)
            else:
                # No image, include as text-only candidate
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- 候選 {len(valid_winners) + 1} ---\n（無截圖）台詞：「{s.text_current}」（{s.episode_id}）",
                })
                valid_winners.append(s)

            # Limit to ~20 candidates to avoid token overflow
            if len(valid_winners) >= 20:
                break

        content_blocks.append({
            "type": "text",
            "text": f"\n\n從以上候選中選最好笑的 {top_k} 個（考慮截圖畫面＋台詞＋使用者輸入的搭配效果）：",
        })

        try:
            response = self.client.messages.create(
                model=self.fast_model,
                max_tokens=100,
                system=FINAL_VISION_PROMPT,
                messages=[{"role": "user", "content": content_blocks}],
                temperature=0.3,
            )

            raw = response.content[0].text.strip()
            indices = _parse_indices(raw, len(valid_winners))
            if not indices:
                return valid_winners[:top_k]
            return [valid_winners[i] for i in indices[:top_k]]
        except Exception as e:
            print(f"  Vision final select error: {e}")
            # Fallback: return first top_k without vision
            return winners[:top_k]

    def _final_select_text(
        self, query: str, winners: list[Segment], top_k: int
    ) -> list[Segment]:
        """Fallback: text-only final selection."""
        lines = []
        for i, s in enumerate(winners, 1):
            lines.append(f"{i}. [{s.episode_id}] {s.text_current}")

        user_msg = (
            f"使用者說：「{query}」\n\n候選回應：\n" +
            "\n".join(lines) +
            f"\n\n選最好笑的 {top_k} 個："
        )

        response = self.client.messages.create(
            model=self.smart_model,
            max_tokens=100,
            system=FINAL_VISION_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.3,
        )

        raw = response.content[0].text.strip()
        indices = _parse_indices(raw, len(winners))
        if not indices:
            return winners[:top_k]
        return [winners[i] for i in indices[:top_k]]
