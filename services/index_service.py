"""Index building pipeline: parse → segment → embed → index → extract frames → save."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import AppConfig
from core.embedder import Embedder
from core.media_extractor import MediaExtractor
from core.searcher import Searcher
from core.segment_builder import SegmentBuilder
from core.subtitle_parser import SubtitleParser
from schemas import Segment


class IndexService:
    """Orchestrate the full index building pipeline."""

    def __init__(
        self,
        config: AppConfig,
        parser: SubtitleParser,
        builder: SegmentBuilder,
        embedder: Embedder,
        searcher: Searcher,
        extractor: MediaExtractor | None = None,
    ) -> None:
        self.config = config
        self.parser = parser
        self.builder = builder
        self.embedder = embedder
        self.searcher = searcher
        self.extractor = extractor

    def build_all(
        self, srt_dir: Path, patterns: list[str] | None = None
    ) -> None:
        """Run full pipeline: parse → segment → embed → index → extract frames → save."""
        # 1. Parse
        entries = self.parser.parse_dir(srt_dir, patterns=patterns)
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

        # 5. Pre-extract frames (parallel)
        if self.extractor:
            self._extract_all_frames(segments)

        # 6. Save (segments now carry frame_path)
        save_path = self.config.cache_dir / "index"
        self.searcher.save(save_path)

    def _extract_all_frames(self, segments: list[Segment], workers: int = 8) -> None:
        """Pre-extract one frame per segment at current_start (OCR-verified timestamp)."""
        # Deduplicate: segments sharing the same video + current_start get the same frame
        groups: dict[str, list[Segment]] = {}
        for seg in segments:
            t = seg.current_start if seg.current_start > 0 else seg.mid_time
            key = f"{seg.video_path}:{t:.1f}"
            groups.setdefault(key, []).append(seg)

        tasks = [(key, segs[0], segs) for key, segs in groups.items()]
        total = len(tasks)
        print(f"Extracting frames for {total} unique timestamps ({len(segments)} segments)...")

        def extract_one(task):
            _key, rep, segs = task
            t = rep.current_start if rep.current_start > 0 else rep.mid_time
            try:
                frame = self.extractor.extract_frame(rep.video_path, t)
                path_str = str(frame) if frame else None
            except Exception:
                path_str = None
            for s in segs:
                s.frame_path = path_str
            return path_str is not None

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, t): t for t in tasks}
            for future in as_completed(futures):
                done += 1
                if done % 500 == 0 or done == total:
                    print(f"  [{done}/{total}] frames extracted")

        extracted = sum(1 for s in segments if s.frame_path)
        print(f"Frame extraction complete: {extracted}/{len(segments)} segments have frames.")

    def load(self) -> None:
        """Load pre-built index from cache."""
        load_path = self.config.cache_dir / "index"
        self.searcher.load(load_path)
