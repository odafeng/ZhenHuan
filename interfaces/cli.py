"""CLI interface for ZhenHuan retrieval system.

Usage:
    python -m interfaces.cli --query "你是在酸我嗎"
    python -m interfaces.cli --query "臣妾做不到" --top-k 3
    python -m interfaces.cli --build --srt-dir data/subtitles
"""

import argparse
import sys
from pathlib import Path

from config import AppConfig
from core.embedder import LocalEmbedder
from core.media_extractor import MediaExtractor
from core.searcher import Searcher
from core.segment_builder import SegmentBuilder
from core.subtitle_parser import SubtitleParser
from schemas import SearchRequest
from services.index_service import IndexService
from services.retrieval_service import RetrievalService


def build_index(config: AppConfig, srt_dir: Path) -> None:
    """Build the FAISS index from SRT files."""
    print(f"Building index from: {srt_dir}")
    parser = SubtitleParser(to_traditional=config.to_traditional)
    builder = SegmentBuilder(config)
    embedder = LocalEmbedder(config.local_model_name)
    searcher = Searcher(config)

    index_svc = IndexService(config, parser, builder, embedder, searcher)
    index_svc.build_all(srt_dir)
    print("Index built and saved.")


def search_query(config: AppConfig, query: str, top_k: int) -> None:
    """Search the index and print results."""
    embedder = LocalEmbedder(config.local_model_name)
    searcher = Searcher(config)
    searcher.load(config.cache_dir / "index")
    extractor = MediaExtractor(output_dir=config.output_dir)

    retrieval_svc = RetrievalService(config, embedder, searcher, extractor)
    request = SearchRequest(query=query, top_k=top_k)
    response = retrieval_svc.search(request)

    print(f"\nQuery: {response.query}")
    print(f"Results: {len(response.results)}")
    print("-" * 60)
    for i, r in enumerate(response.results, 1):
        seg = r.segment
        print(f"\n#{i} [score={r.score:.4f}] [{seg.window_tier}]")
        print(f"  Episode: {seg.episode_id}")
        print(f"  Time: {seg.start_time:.1f}s - {seg.end_time:.1f}s (mid={seg.mid_time:.1f}s)")
        print(f"  Current: {seg.text_current}")
        print(f"  Window:  {seg.text_window}")
        if r.frame_path:
            print(f"  Frame:   {r.frame_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZhenHuan Reaction Retrieval System")
    parser.add_argument("--query", "-q", type=str, help="Search query text")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--build", action="store_true", help="Build index from SRT files")
    parser.add_argument("--srt-dir", type=Path, default=Path("data/subtitles"), help="SRT directory")

    args = parser.parse_args()
    config = AppConfig()
    config.ensure_dirs()

    if args.build:
        build_index(config, args.srt_dir)
    elif args.query:
        search_query(config, args.query, args.top_k)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
