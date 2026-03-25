"""FAISS-based search with merge, dedup, and diversity enforcement."""

from pathlib import Path

import numpy as np

from config import AppConfig
from schemas import SearchResult, Segment


class Searcher:
    """Search FAISS index with multi-tier merge and diversity control."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._index = None
        self._segments: list[Segment] = []

    def build_index(self, segments: list[Segment], embeddings: np.ndarray) -> None:
        """Build FAISS index from segments and their embeddings."""
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray) -> list[SearchResult]:
        """Search index, merge across tiers, dedup, enforce diversity."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save index and segment metadata to disk."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load index and segment metadata from disk."""
        raise NotImplementedError
