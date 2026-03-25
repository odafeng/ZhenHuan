"""Unit tests for Embedder interface."""

import numpy as np

from tests.conftest import MockEmbedder


class TestEmbedder:
    def test_embed_texts_shape(self, mock_embedder: MockEmbedder) -> None:
        texts = ["你好", "世界", "甄嬛傳"]
        result = mock_embedder.embed_texts(texts)
        assert result.shape == (3, mock_embedder.dim)

    def test_embed_query_shape(self, mock_embedder: MockEmbedder) -> None:
        result = mock_embedder.embed_query("臣妾做不到啊")
        assert result.shape == (1, mock_embedder.dim)

    def test_embed_texts_dtype(self, mock_embedder: MockEmbedder) -> None:
        result = mock_embedder.embed_texts(["test"])
        assert result.dtype == np.float32

    def test_embed_empty_list(self, mock_embedder: MockEmbedder) -> None:
        result = mock_embedder.embed_texts([])
        assert result.shape[0] == 0
