"""Unit tests for SegmentBuilder."""

from config import AppConfig
from core.segment_builder import SegmentBuilder
from schemas import SubtitleEntry


class TestSegmentBuilder:
    def test_single_tier(self, app_config: AppConfig, sample_entries: list[SubtitleEntry]) -> None:
        """Single tier: one segment per entry."""
        app_config.window_tiers = ["single"]
        builder = SegmentBuilder(app_config)
        segments = builder.build(sample_entries)
        singles = [s for s in segments if s.window_tier == "single"]
        assert len(singles) == len(sample_entries)
        assert all(s.text_current == s.text_window for s in singles)

    def test_local_window_tier(self, app_config: AppConfig, sample_entries: list[SubtitleEntry]) -> None:
        """Local window: 3-sentence context."""
        app_config.window_tiers = ["local_window"]
        builder = SegmentBuilder(app_config)
        segments = builder.build(sample_entries)
        # Middle entry should have prev + current + next in text_window
        mid = [s for s in segments if s.text_current == "皇上恕罪"]
        assert len(mid) == 1
        assert "你有什麼做不到的" in mid[0].text_window
        assert "臣妾知錯了" in mid[0].text_window

    def test_duration_constraint(self, app_config: AppConfig, sample_entries: list[SubtitleEntry]) -> None:
        """Window must not exceed max duration."""
        app_config.window_max_duration_sec = 10.0
        builder = SegmentBuilder(app_config)
        segments = builder.build(sample_entries)
        for seg in segments:
            assert seg.duration <= app_config.window_max_duration_sec + 0.1  # tolerance

    def test_first_last_boundary(self, app_config: AppConfig, sample_entries: list[SubtitleEntry]) -> None:
        """First and last entries should not crash on window building."""
        builder = SegmentBuilder(app_config)
        segments = builder.build(sample_entries)
        assert len(segments) > 0
