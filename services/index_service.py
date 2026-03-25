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
        # 1. Parse
        entries = self.parser.parse_dir(srt_dir)
        if not entries:
            raise ValueError(f"No subtitle entries found in {srt_dir}")

        # 2. Build segments
        segments = self.builder.build(entries)
        if not segments:
            raise ValueError("No segments produced after building")

        # 3. Embed
        texts = [s.embedding_text for s in segments]
        embeddings = self.embedder.embed_texts(texts)

        # 4. Build index
        self.searcher.build_index(segments, embeddings)

        # 5. Save
        save_path = self.config.cache_dir / "index"
        self.searcher.save(save_path)

    def load(self) -> None:
        """Load pre-built index from cache."""
        load_path = self.config.cache_dir / "index"
        self.searcher.load(load_path)
