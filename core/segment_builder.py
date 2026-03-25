"""Build multi-window segments from parsed subtitle entries."""

from config import AppConfig
from schemas import Segment, SubtitleEntry


class SegmentBuilder:
    """Build segments with 3 window tiers from subtitle entries."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build(self, entries: list[SubtitleEntry]) -> list[Segment]:
        """Build segments for all configured window tiers.

        Duration constraint (config.window_max_duration_sec) is first-class.
        Sentence count is soft guide.
        """
        raise NotImplementedError
