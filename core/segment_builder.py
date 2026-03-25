"""Build multi-window segments from parsed subtitle entries."""

from config import AppConfig
from schemas import Segment, SubtitleEntry


class SegmentBuilder:
    """Build segments with 3 window tiers from subtitle entries.

    Duration constraint is first-class: any segment exceeding
    window_max_duration_sec is excluded or trimmed.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build(self, entries: list[SubtitleEntry]) -> list[Segment]:
        """Build segments for all configured window tiers."""
        segments: list[Segment] = []
        # Group entries by episode to avoid cross-episode windows
        by_episode: dict[str, list[SubtitleEntry]] = {}
        for e in entries:
            by_episode.setdefault(e.episode_id, []).append(e)

        for episode_id, ep_entries in by_episode.items():
            ep_entries.sort(key=lambda e: e.start_time)
            for tier in self.config.window_tiers:
                segments.extend(self._build_tier(ep_entries, tier))

        return segments

    def _build_tier(self, entries: list[SubtitleEntry], tier: str) -> list[Segment]:
        """Build segments for a single tier across one episode's entries."""
        if tier == "single":
            radius = 0
        elif tier == "local_window":
            radius = self.config.local_window_size // 2  # 3 → 1
        elif tier == "expanded_window":
            radius = self.config.expanded_window_size // 2  # 5 → 2
        else:
            return []

        segments: list[Segment] = []
        for i, center in enumerate(entries):
            # Determine window range
            lo = max(0, i - radius)
            hi = min(len(entries) - 1, i + radius)

            # Enforce duration constraint: trim from farthest end
            lo, hi = self._trim_to_duration(entries, i, lo, hi)

            # After trimming, check if the window is still valid
            window_start = entries[lo].start_time
            window_end = entries[hi].end_time
            duration = window_end - window_start

            if duration > self.config.window_max_duration_sec:
                # Even the center entry alone exceeds limit — skip
                continue

            window_texts = [entries[j].text for j in range(lo, hi + 1)]
            text_window = " ".join(window_texts)

            mid_time = (center.start_time + center.end_time) / 2

            seg = Segment(
                segment_id=f"{center.episode_id}_{tier}_{center.subtitle_id}",
                episode_id=center.episode_id,
                video_path=center.video_path,
                start_time=window_start,
                end_time=window_end,
                duration=duration,
                text_current=center.text,
                text_window=text_window,
                window_tier=tier,
                mid_time=mid_time,
                current_start=center.start_time,
                current_end=center.end_time,
                embedding_text=text_window,
            )
            segments.append(seg)

        return segments

    def _trim_to_duration(
        self, entries: list[SubtitleEntry], center: int, lo: int, hi: int
    ) -> tuple[int, int]:
        """Trim window bounds so total duration stays within limit.

        Removes the farthest entry from center first.
        """
        max_dur = self.config.window_max_duration_sec

        while lo < hi:
            window_start = entries[lo].start_time
            window_end = entries[hi].end_time
            if (window_end - window_start) <= max_dur:
                break
            # Remove the entry farthest from center
            dist_lo = center - lo
            dist_hi = hi - center
            if dist_lo >= dist_hi:
                lo += 1
            else:
                hi -= 1

        return lo, hi
