"""Index building pipeline: parse → segment → embed → index → save."""

from pathlib import Path

from config import AppConfig
from core.embedder import Embedder
from core.searcher import Searcher
from core.segment_builder import SegmentBuilder
from core.subtitle_parser import SubtitleParser


class IndexService:
    """Orchestrate the full index building pipeline."""

    def __init__(
        self,
        config: AppConfig,
        parser: SubtitleParser,
        builder: SegmentBuilder,
        embedder: Embedder,
        searcher: Searcher,
    ) -> None:
        self.config = config
        self.parser = parser
        self.builder = builder
        self.embedder = embedder
        self.searcher = searcher

    def build_all(self, srt_dir: Path) -> None:
        """Run full pipeline: parse SRTs → build segments → embed → build index → save."""
        raise NotImplementedError

    def load(self) -> None:
        """Load pre-built index from cache."""
        raise NotImplementedError
