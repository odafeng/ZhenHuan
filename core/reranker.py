"""LLM-based selector: directly pick funny responses from all segments, no embedding needed."""

import re
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


FINAL_PROMPT = """你是「甄嬛傳迷因產生器」的最終評審。

從以下候選回應中，選出搭配使用者輸入最好笑的，排序輸出。
要求多樣性——不同類型的笑點比重複同一個梗好。

只回傳編號，用逗號分隔。不要解釋。"""


class LLMReranker:
    """Directly select funny responses from all segments using LLM."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self._segments: list[Segment] | None = None

    def load_segments(self, cache_dir: Path) -> None:
        """Load all segments from the index cache."""
        import pickle
        seg_path = cache_dir / "index" / "segments.pkl"
        if not seg_path.exists():
            raise FileNotFoundError(f"Segments not found: {seg_path}")
        with open(seg_path, "rb") as f:
            self._segments = pickle.load(f)

    def rerank(
        self, query: str, _candidates: list, top_k: int = 5
    ) -> list:
        """Not used in direct mode. See select() instead."""
        return self.select(query, top_k)

    def select(self, query: str, top_k: int = 5) -> list[dict]:
        """Directly select funny responses from all segments."""
        if not self._segments:
            raise RuntimeError("Call load_segments() first")

        # Only use single-tier segments (one line each) to reduce volume
        singles = [s for s in self._segments if s.window_tier == "single"]

        # Group into batches of 10 episodes
        batches = self._group_by_episode_batch(singles, batch_size=10)

        print(f"  Searching {len(singles)} segments across {len(batches)} batches...")

        # Pick best from each batch (parallel)
        batch_winners: list[Segment] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self._pick_from_batch, query, segs): name
                for name, segs in batches.items()
                if segs
            }
            for future in futures:
                results = future.result()
                batch_winners.extend(results)

        if not batch_winners:
            return []

        print(f"  Batch winners: {len(batch_winners)}, selecting top {top_k}...")

        # Final selection
        if len(batch_winners) <= top_k:
            final = batch_winners
        else:
            final = self._final_select(query, batch_winners, top_k)

        # Convert to SearchResult-like dicts
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
        """LLM picks up to 3 best responses from a batch of segments."""
        # Build numbered list
        lines = []
        for i, s in enumerate(segments, 1):
            lines.append(f"{i}. {s.text_current}")

        # Split into chunks if too large (~4000 lines max per call)
        chunk_size = 3000
        all_picks: list[Segment] = []

        for start in range(0, len(lines), chunk_size):
            chunk_lines = lines[start:start + chunk_size]
            chunk_segments = segments[start:start + chunk_size]

            text = "\n".join(chunk_lines)
            user_msg = f"使用者說：「{query}」\n\n以下是甄嬛傳台詞：\n{text}\n\n選最適合當幽默回應的（最多3個）："

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=50,
                    system=PICK_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.5,
                )
                raw = response.content[0].text.strip()
                if raw == "0":
                    continue
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                for idx in indices:
                    if 0 <= idx < len(chunk_segments):
                        all_picks.append(chunk_segments[idx])
            except Exception as e:
                print(f"    Batch error: {e}")

        return all_picks

    def _final_select(
        self, query: str, winners: list[Segment], top_k: int
    ) -> list[Segment]:
        lines = []
        for i, s in enumerate(winners, 1):
            lines.append(f"{i}. [{s.episode_id}] {s.text_current}")

        user_msg = (
            f"使用者說：「{query}」\n\n候選回應：\n" +
            "\n".join(lines) +
            f"\n\n選最好笑的 {top_k} 個："
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            system=FINAL_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.3,
        )

        raw = response.content[0].text.strip()
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            indices = [i for i in indices if 0 <= i < len(winners)]
            return [winners[i] for i in indices[:top_k]]
        except (ValueError, IndexError):
            return winners[:top_k]
