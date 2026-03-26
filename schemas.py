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
    mid_time: float  # midpoint of the current (center) entry
    embedding_text: str
    current_start: float = 0.0  # start time of the center entry
    current_end: float = 0.0  # end time of the center entry
    frame_path: str | None = None  # pre-extracted frame (set at build time)


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
