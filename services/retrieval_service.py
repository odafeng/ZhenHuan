"""Query pipeline: embed query → search → return pre-extracted frames.
   Or: LLM direct select → return pre-extracted frames.

Frames are pre-extracted at build time at current_start (OCR-verified timestamp).
MediaExtractor is only needed as fallback for segments without pre-extracted frames.
"""

from __future__ import annotations

from config import AppConfig
from core.embedder import Embedder
from core.media_extractor import MediaExtractor
from core.reranker import LLMReranker
from core.searcher import Searcher
from schemas import SearchRequest, SearchResponse


class RetrievalService:
    """Orchestrate the full retrieval pipeline."""

    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        searcher: Searcher,
        extractor: MediaExtractor | None = None,
        reranker: LLMReranker | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.searcher = searcher
        self.extractor = extractor
        self.reranker = reranker

    def search(self, request: SearchRequest) -> SearchResponse:
        """Full retrieval: embed → search → return results with pre-extracted frames."""
        if self.reranker:
            return self._search_llm(request)
        return self._search_embedding(request)

    def _search_embedding(self, request: SearchRequest) -> SearchResponse:
        """Pure embedding search."""
        query_vec = self.embedder.embed_query(request.query)
        self.config.final_top_k = request.top_k
        results = self.searcher.search(query_vec)
        self._attach_frames(results)
        return SearchResponse(query=request.query, results=results)

    def _search_llm(self, request: SearchRequest) -> SearchResponse:
        """LLM directly selects from all segments."""
        extra = request.top_k + 5
        candidates = self.reranker.select(request.query, top_k=extra)

        results = []
        for r in candidates:
            if len(results) >= request.top_k:
                break
            # Use pre-extracted frame
            if r.segment.frame_path:
                r.frame_path = r.segment.frame_path
                results.append(r)
            elif self.extractor:
                try:
                    frame = self._fallback_extract(r)
                    if frame:
                        r.frame_path = frame
                        results.append(r)
                except Exception:
                    continue
            else:
                results.append(r)

        return SearchResponse(query=request.query, results=results)

    def _attach_frames(self, results: list) -> None:
        """Attach pre-extracted frames, with optional fallback."""
        for r in results:
            if r.segment.frame_path:
                r.frame_path = r.segment.frame_path
            elif self.extractor:
                try:
                    r.frame_path = self._fallback_extract(r)
                except Exception:
                    r.frame_path = None

    def _fallback_extract(self, r) -> str | None:
        """Fallback: extract frame at current_start (not mid_time)."""
        t = r.segment.current_start if r.segment.current_start > 0 else r.segment.mid_time
        frame_path = self.extractor.extract_frame(r.segment.video_path, t)
        return str(frame_path) if frame_path else None
