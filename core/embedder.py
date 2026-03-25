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
    """Local Chinese embedding using sentence-transformers.

    Lazy-loads model on first use to avoid slow startup.
    """

    def __init__(self, model_name: str = "shibing624/text2vec-bge-large-chinese") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


class OpenAIEmbedder(Embedder):
    """OpenAI embedding API backend.

    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    # Known dimensions for OpenAI embedding models
    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def dim(self) -> int:
        return self._MODEL_DIMS.get(self.model_name, 1536)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model_name)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])
