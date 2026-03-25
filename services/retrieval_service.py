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
        raise NotImplementedError
