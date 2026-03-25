"""Unit tests for Searcher."""

import numpy as np

from config import AppConfig
from core.searcher import Searcher
from schemas import Segment


def _make_segments(n: int, tier: str = "single", episode: str = "ep01") -> list[Segment]:
    """Helper to create dummy segments."""
    return [
        Segment(
            segment_id=f"{episode}_{tier}_{i}",
            episode_id=episode,
            video_path=f"data/videos/{episode}.mp4",
            start_time=float(i * 3),
            end_time=float(i * 3 + 2.5),
            duration=2.5,
            text_current=f"台詞{i}",
            text_window=f"台詞{i}",
            window_tier=tier,
            mid_time=float(i * 3 + 1.25),
            embedding_text=f"台詞{i}",
        )
        for i in range(n)
    ]


class TestSearcher:
    def test_build_and_search(self, app_config: AppConfig) -> None:
        """Build index and retrieve results."""
        searcher = Searcher(app_config)
        segments = _make_segments(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 64).astype(np.float32)
        searcher.build_index(segments, embeddings)

        query_vec = rng.randn(1, 64).astype(np.float32)
        results = searcher.search(query_vec)
        assert len(results) <= app_config.final_top_k
        assert all(r.score is not None for r in results)

    def test_top_k_limit(self, app_config: AppConfig) -> None:
        """Should not return more than final_top_k."""
        app_config.final_top_k = 3
        searcher = Searcher(app_config)
        segments = _make_segments(20)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 64).astype(np.float32)
        searcher.build_index(segments, embeddings)

        query_vec = rng.randn(1, 64).astype(np.float32)
        results = searcher.search(query_vec)
        assert len(results) <= 3

    def test_diversity_constraint(self, app_config: AppConfig) -> None:
        """Same episode should not dominate results."""
        app_config.max_same_episode_scene = 2
        app_config.final_top_k = 5
        searcher = Searcher(app_config)
        # All from same episode
        segments = _make_segments(20, episode="ep01")
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 64).astype(np.float32)
        searcher.build_index(segments, embeddings)

        query_vec = rng.randn(1, 64).astype(np.float32)
        results = searcher.search(query_vec)
        ep01_count = sum(1 for r in results if r.segment.episode_id == "ep01")
        assert ep01_count <= app_config.max_same_episode_scene

    def test_save_load_roundtrip(self, app_config: AppConfig, tmp_path) -> None:
        """Index should survive save/load cycle."""
        searcher = Searcher(app_config)
        segments = _make_segments(5)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(5, 64).astype(np.float32)
        searcher.build_index(segments, embeddings)

        save_path = tmp_path / "index"
        searcher.save(save_path)

        searcher2 = Searcher(app_config)
        searcher2.load(save_path)

        query_vec = rng.randn(1, 64).astype(np.float32)
        r1 = searcher.search(query_vec)
        r2 = searcher2.search(query_vec)
        assert len(r1) == len(r2)

    def test_empty_index(self, app_config: AppConfig) -> None:
        """Search on empty index should return empty list."""
        searcher = Searcher(app_config)
        segments = []
        embeddings = np.empty((0, 64), dtype=np.float32)
        searcher.build_index(segments, embeddings)

        query_vec = np.random.randn(1, 64).astype(np.float32)
        results = searcher.search(query_vec)
        assert results == []
