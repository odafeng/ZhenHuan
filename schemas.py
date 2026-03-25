"""Data models for ZhenHuan retrieval system.

All models use dataclass for MVP.
Future: swap to pydantic.BaseModel for FastAPI integration (fields stay the same).
"""

from dataclasses import dataclass, field


@dataclass
class SubtitleEntry:
    subtitle_id: int
    episode_id: str
    video_path: str
    start_time: float
    end_time: float
    text: str


@dataclass
class Segment:
    segment_id: str
    episode_id: str
    video_path: str
    start_time: float
    end_time: float
    duration: float
    text_current: str
    text_window: str
    window_tier: str  # "single" | "local_window" | "expanded_window"
    mid_time: float
    embedding_text: str


@dataclass
class SearchResult:
    segment: Segment
    score: float
    frame_path: str | None = None


@dataclass
class SearchRequest:
    query: str
    top_k: int = 5


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
