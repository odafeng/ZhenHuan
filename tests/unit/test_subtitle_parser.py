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
        entries = parser.parse(bad_srt)
        assert isinstance(entries, list)

    def test_parse_ass_format(self, tmp_path: Path) -> None:
        """Parse a well-formed ASS file."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:03:10.76,0:03:12.91,chs,,0000,0000,0000,,{\\\\blur3}西元一七二二年\n"
            "Dialogue: 0,0:03:13.12,0:03:15.39,chs,,0000,0000,0000,,{\\\\blur3}清康熙皇帝驾崩\n"
        )
        ass_path = tmp_path / "甄嬛传_E01.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
        parser = SubtitleParser()
        entries = parser.parse(ass_path)
        assert len(entries) == 2
        assert entries[0].text == "西元一七二二年"
        assert entries[0].episode_id == "ep01"
        assert abs(entries[0].start_time - 190.76) < 0.01

    def test_parse_ass_strips_tags(self, tmp_path: Path) -> None:
        """ASS override tags like {\\blur3} should be stripped."""
        ass_content = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:03.00,chs,,0000,0000,0000,,{\\\\blur3}{\\\\pos(192,230)}你好世界\n"
        )
        ass_path = tmp_path / "test_E02.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
        parser = SubtitleParser()
        entries = parser.parse(ass_path)
        assert len(entries) == 1
        assert entries[0].text == "你好世界"
        assert entries[0].episode_id == "ep02"

    def test_episode_id_extraction(self, tmp_path: Path) -> None:
        """Episode ID should be extracted from filename."""
        parser = SubtitleParser()
        assert parser._extract_episode_id(Path("甄嬛传_E01_720p.ass")) == "ep01"
        assert parser._extract_episode_id(Path("E12.srt")) == "ep12"
        assert parser._extract_episode_id(Path("no_episode.srt")) == "no_episode"

    def test_parse_dir_mixed_formats(self, tmp_path: Path) -> None:
        """parse_dir should handle both SRT and ASS files."""
        srt = tmp_path / "ep01.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nSRT台詞\n", encoding="utf-8")
        ass = tmp_path / "甄嬛传_E02.ass"
        ass.write_text(
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,chs,,0000,0000,0000,,ASS台詞\n",
            encoding="utf-8",
        )
        parser = SubtitleParser()
        entries = parser.parse_dir(tmp_path)
        assert len(entries) == 2
        texts = {e.text for e in entries}
        assert "SRT台詞" in texts
        assert "ASS台詞" in texts
