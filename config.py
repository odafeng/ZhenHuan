"""Centralized configuration for ZhenHuan retrieval system.

Uses dataclass for MVP.
Future: swap to pydantic_settings.BaseSettings for FastAPI (fields stay the same).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    # Embedding
    embedding_backend: str = "local"  # "local" | "openai"
    local_model_name: str = "shibing624/text2vec-bge-large-chinese"
    openai_model_name: str = "text-embedding-3-small"

    # Parser
    to_traditional: bool = True  # convert simplified → traditional Chinese

    # Window
    window_tiers: list[str] = field(
        default_factory=lambda: ["single", "local_window", "expanded_window"]
    )
    window_max_duration_sec: float = 10.0  # first-class constraint
    local_window_size: int = 3  # sentences for local_window
    expanded_window_size: int = 5  # sentences for expanded_window

    # Search
    top_k_per_tier: int = 10
    final_top_k: int = 5
    max_same_episode_scene: int = 2  # diversity constraint

    # Paths
    project_dir: Path = Path(".")
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    videos_dir: Path = Path("data/videos")
    subtitles_dir: Path = Path("data/subtitles")
    output_dir: Path = Path("data/processed")
    annotations_dir: Path = Path("data/annotations")

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [
            self.data_dir,
            self.cache_dir,
            self.videos_dir,
            self.subtitles_dir,
            self.output_dir,
            self.annotations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
