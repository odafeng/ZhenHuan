"""FAISS-based search with merge, dedup, and diversity enforcement."""

import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from config import AppConfig
from schemas import SearchResult, Segment


class Searcher:
    """Search FAISS index with multi-tier merge and diversity control."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._index: faiss.IndexFlatIP | None = None
        self._segments: list[Segment] = []

    def build_index(self, segments: list[Segment], embeddings: np.ndarray) -> None:
        """Build FAISS index from segments and their embeddings.

        Uses inner product (cosine similarity on normalized vectors).
        """
        if len(segments) == 0:
            self._index = None
            self._segments = []
            return

        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._segments = list(segments)

    def search(self, query_embedding: np.ndarray) -> list[SearchResult]:
        """Search index, merge across tiers, dedup, enforce diversity.

        Returns up to config.final_top_k results, with at most
        config.max_same_episode_scene from the same episode.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(query)

        fetch_k = min(
            self.config.top_k_per_tier * len(self.config.window_tiers),
            self._index.ntotal,
        )
        scores, indices = self._index.search(query, fetch_k)

        candidates: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidates.append(
                SearchResult(segment=self._segments[idx], score=float(score))
            )

        deduped = self._dedup_overlapping(candidates)
        diverse = self._enforce_diversity(deduped)
        return diverse[: self.config.final_top_k]

    def _dedup_overlapping(self, candidates: list[SearchResult]) -> list[SearchResult]:
        """Remove segments that overlap in time within the same episode + tier."""
        kept: list[SearchResult] = []
        for c in candidates:
            is_dup = False
            for k in kept:
                if (
                    c.segment.episode_id == k.segment.episode_id
                    and c.segment.window_tier == k.segment.window_tier
                    and c.segment.start_time < k.segment.end_time
                    and c.segment.end_time > k.segment.start_time
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
        return kept

    def _enforce_diversity(self, candidates: list[SearchResult]) -> list[SearchResult]:
        """Limit results from the same episode to max_same_episode_scene."""
        episode_counts: dict[str, int] = defaultdict(int)
        diverse: list[SearchResult] = []
        for c in candidates:
            ep = c.segment.episode_id
            if episode_counts[ep] < self.config.max_same_episode_scene:
                diverse.append(c)
                episode_counts[ep] += 1
        return diverse

    def save(self, path: Path) -> None:
        """Save FAISS index and segment metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "segments.pkl", "wb") as f:
            pickle.dump(self._segments, f)

    def load(self, path: Path) -> None:
        """Load FAISS index and segment metadata from disk."""
        index_path = path / "faiss.index"
        segments_path = path / "segments.pkl"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        else:
            self._index = None
        if segments_path.exists():
            with open(segments_path, "rb") as f:
                self._segments = pickle.load(f)
        else:
            self._segments = []
