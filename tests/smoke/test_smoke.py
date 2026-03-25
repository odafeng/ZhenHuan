"""Smoke tests: end-to-end pipeline verification.

These tests use mock embeddings and don't require GPU or real models.
They verify the full pipeline runs without crashing and returns sane output.
"""

from pathlib import Path

from config import AppConfig
from core.subtitle_parser import SubtitleParser
from core.segment_builder import SegmentBuilder
from core.searcher import Searcher
from schemas import SearchRequest
from services.index_service import IndexService
from services.retrieval_service import RetrievalService
from core.media_extractor import MediaExtractor


def test_full_pipeline(sample_srt: Path, app_config: AppConfig, mock_embedder) -> None:
    """End-to-end: SRT → parse → build segments → embed → index → query → results.

    Verifies:
    - Pipeline doesn't crash
    - Returns correct number of results
    - Each result has a segment and score
    """
    parser = SubtitleParser()
    builder = SegmentBuilder(app_config)
    searcher = Searcher(app_config)

    # Build index
    index_svc = IndexService(app_config, parser, builder, mock_embedder, searcher)
    index_svc.build_all(sample_srt.parent)

    # Query
    extractor = MediaExtractor(output_dir=app_config.output_dir)
    retrieval_svc = RetrievalService(app_config, mock_embedder, searcher, extractor)

    request = SearchRequest(query="做不到", top_k=3)
    response = retrieval_svc.search(request)

    assert response.query == "做不到"
    assert len(response.results) <= 3
    assert all(r.segment is not None for r in response.results)
    assert all(isinstance(r.score, float) for r in response.results)


def test_index_persistence(sample_srt: Path, app_config: AppConfig, mock_embedder, tmp_path: Path) -> None:
    """Build index → save → load → query should give consistent results."""
    parser = SubtitleParser()
    builder = SegmentBuilder(app_config)

    # Build and save
    searcher1 = Searcher(app_config)
    index_svc = IndexService(app_config, parser, builder, mock_embedder, searcher1)
    index_svc.build_all(sample_srt.parent)

    save_path = tmp_path / "index_save"
    searcher1.save(save_path)

    # Load into new searcher
    searcher2 = Searcher(app_config)
    searcher2.load(save_path)

    # Same query should give same results
    query_vec = mock_embedder.embed_query("做不到")
    r1 = searcher1.search(query_vec)
    r2 = searcher2.search(query_vec)

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.segment.segment_id == b.segment.segment_id
        assert abs(a.score - b.score) < 1e-6
