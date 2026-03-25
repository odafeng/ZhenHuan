"""Shared test fixtures for ZhenHuan tests."""

import pytest
import numpy as np
from pathlib import Path

from config import AppConfig
from schemas import SubtitleEntry


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """AppConfig pointing to temp directories."""
    return AppConfig(
        project_dir=tmp_path,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "data" / "cache",
        videos_dir=tmp_path / "data" / "videos",
        subtitles_dir=tmp_path / "data" / "subtitles",
        output_dir=tmp_path / "data" / "processed",
        annotations_dir=tmp_path / "data" / "annotations",
    )


@pytest.fixture
def sample_srt(tmp_path: Path) -> Path:
    """Create a minimal .srt file for testing."""
    srt_content = """1
00:00:01,000 --> 00:00:03,500
臣妾做不到啊

2
00:00:04,000 --> 00:00:06,200
你有什麼做不到的

3
00:00:06,500 --> 00:00:08,000
皇上恕罪

4
00:00:08,200 --> 00:00:09,500
臣妾知錯了

5
00:00:15,000 --> 00:00:28,000
這是一句時間跨度超過10秒的字幕用來測試duration constraint
"""
    srt_path = tmp_path / "data" / "subtitles" / "ep01.srt"
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(srt_content, encoding="utf-8")
    return srt_path


@pytest.fixture
def sample_entries() -> list[SubtitleEntry]:
    """Pre-parsed subtitle entries for testing without SRT parsing."""
    return [
        SubtitleEntry(0, "ep01", "data/videos/ep01.mp4", 1.0, 3.5, "臣妾做不到啊"),
        SubtitleEntry(1, "ep01", "data/videos/ep01.mp4", 4.0, 6.2, "你有什麼做不到的"),
        SubtitleEntry(2, "ep01", "data/videos/ep01.mp4", 6.5, 8.0, "皇上恕罪"),
        SubtitleEntry(3, "ep01", "data/videos/ep01.mp4", 8.2, 9.5, "臣妾知錯了"),
        SubtitleEntry(4, "ep01", "data/videos/ep01.mp4", 15.0, 28.0, "這是一句時間跨度超過10秒的字幕"),
    ]


class MockEmbedder:
    """Deterministic mock embedder for testing (no model needed)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), self._dim).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        rng = np.random.RandomState(hash(query) % 2**31)
        return rng.randn(1, self._dim).astype(np.float32)


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()
