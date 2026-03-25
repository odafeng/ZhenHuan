"""Query pipeline: embed query → search → extract frames → response.
   Or: LLM direct select → extract frames → response.
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
        extractor: MediaExtractor,
        reranker: LLMReranker | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.searcher = searcher
        self.extractor = extractor
        self.reranker = reranker

    def search(self, request: SearchRequest) -> SearchResponse:
        """Full retrieval: embed → search → extract frames."""
        if self.reranker:
            return self._search_llm(request)
        return self._search_embedding(request)

    def _search_embedding(self, request: SearchRequest) -> SearchResponse:
        """Pure embedding search."""
        query_vec = self.embedder.embed_query(request.query)
        self.config.final_top_k = request.top_k
        results = self.searcher.search(query_vec)
        self._extract_frames(results)
        return SearchResponse(query=request.query, results=results)

    def _search_llm(self, request: SearchRequest) -> SearchResponse:
        """LLM directly selects from all segments, with subtitle gating."""
        # Request extra candidates so we can skip ones without subtitles
        extra = request.top_k + 5
        candidates = self.reranker.select(request.query, top_k=extra)

        results = []
        for r in candidates:
            if len(results) >= request.top_k:
                break
            try:
                cs = r.segment.current_start
                ce = r.segment.current_end
                if cs > 0 and ce > cs:
                    frame_path = self.extractor.extract_frame_with_subtitle(
                        r.segment.video_path, cs, ce
                    )
                else:
                    frame_path = self.extractor.extract_frame(
                        r.segment.video_path, r.segment.mid_time
                    )

                if frame_path and self.extractor._has_subtitle(frame_path):
                    r.frame_path = str(frame_path)
                    results.append(r)
                else:
                    # Skip: no subtitle visible
                    continue
            except Exception:
                continue

        return SearchResponse(query=request.query, results=results)

    def _extract_frames(self, results: list) -> None:
        """Extract frames with subtitle gating."""
        for r in results:
            try:
                cs = r.segment.current_start
                ce = r.segment.current_end
                if cs > 0 and ce > cs:
                    # Try multiple frames, pick one with visible subtitle
                    frame_path = self.extractor.extract_frame_with_subtitle(
                        r.segment.video_path, cs, ce
                    )
                else:
                    frame_path = self.extractor.extract_frame(
                        r.segment.video_path, r.segment.mid_time
                    )
                r.frame_path = str(frame_path) if frame_path else None
            except Exception:
                r.frame_path = None
