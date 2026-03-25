"""Unit tests for SubtitleParser."""

from pathlib import Path

from core.subtitle_parser import SubtitleParser


class TestSubtitleParser:
    def test_parse_normal_srt(self, sample_srt: Path) -> None:
        """Parse a well-formed SRT file."""
        parser = SubtitleParser()
        entries = parser.parse(sample_srt)
        assert len(entries) == 5
        assert entries[0].text == "臣妾做不到啊"
        assert entries[0].start_time == 1.0
        assert entries[0].end_time == 3.5

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Handle empty SRT file gracefully."""
        empty_srt = tmp_path / "empty.srt"
        empty_srt.write_text("", encoding="utf-8")
        parser = SubtitleParser()
        entries = parser.parse(empty_srt)
        assert entries == []

    def test_parse_malformed_timestamp(self, tmp_path: Path) -> None:
        """Handle SRT with bad timestamp format."""
        bad_srt = tmp_path / "bad.srt"
        bad_srt.write_text("1\nNOT_A_TIMESTAMP\n有問題的字幕\n", encoding="utf-8")
        parser = SubtitleParser()
        # Should not crash; may skip or handle gracefully
        entries = parser.parse(bad_srt)
        assert isinstance(entries, list)
