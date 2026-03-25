"""Embedding interface with local and OpenAI backends."""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedding."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) array."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (1, dim) array."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class LocalEmbedder(Embedder):
    """Local Chinese embedding using sentence-transformers."""

    def __init__(self, model_name: str = "shibing624/text2vec-bge-large-chinese") -> None:
        self.model_name = model_name
        self._model = None  # lazy load

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    """OpenAI embedding API backend."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self.model_name = model_name

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError
