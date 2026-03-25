"""Query pipeline: embed query → search → extract frames → response."""

from config import AppConfig
from core.embedder import Embedder
from core.media_extractor import MediaExtractor
from core.searcher import Searcher
from schemas import SearchRequest, SearchResponse


class RetrievalService:
    """Orchestrate the full retrieval pipeline.

    This is the single entry point that interfaces (CLI/Gradio/FastAPI) call.
    """

    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        searcher: Searcher,
        extractor: MediaExtractor,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.searcher = searcher
        self.extractor = extractor

    def search(self, request: SearchRequest) -> SearchResponse:
        """Full retrieval: embed → search → extract frames → return."""
        # 1. Embed query
        query_vec = self.embedder.embed_query(request.query)

        # 2. Search
        self.config.final_top_k = request.top_k
        results = self.searcher.search(query_vec)

        # 3. Extract frames (skip if video doesn't exist)
        for r in results:
            try:
                frame_path = self.extractor.extract_frame(
                    r.segment.video_path, r.segment.mid_time
                )
                r.frame_path = str(frame_path)
            except Exception:
                r.frame_path = None

        return SearchResponse(query=request.query, results=results)
