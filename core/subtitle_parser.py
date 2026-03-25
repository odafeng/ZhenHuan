"""Parse .srt subtitle files into structured SubtitleEntry list."""

import re
from pathlib import Path

from schemas import SubtitleEntry

_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


class SubtitleParser:
    """Parse SRT files into structured subtitle entries."""

    def parse(self, srt_path: Path) -> list[SubtitleEntry]:
        """Parse a single .srt file into a list of SubtitleEntry."""
        text = self._read_file(srt_path)
        if not text.strip():
            return []

        episode_id = srt_path.stem  # e.g. "ep01"
        video_path = f"data/videos/{episode_id}.mp4"

        entries: list[SubtitleEntry] = []
        blocks = re.split(r"\n\s*\n", text.strip())

        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue

            # Find the timestamp line (may not be the second line in all cases)
            ts_match = None
            ts_line_idx = -1
            for i, line in enumerate(lines):
                ts_match = _TIMESTAMP_RE.match(line.strip())
                if ts_match:
                    ts_line_idx = i
                    break

            if ts_match is None:
                continue  # skip block with no valid timestamp

            g = ts_match.groups()
            start = _ts_to_sec(g[0], g[1], g[2], g[3])
            end = _ts_to_sec(g[4], g[5], g[6], g[7])

            # Text is everything after the timestamp line
            text_lines = lines[ts_line_idx + 1 :]
            subtitle_text = " ".join(line.strip() for line in text_lines if line.strip())

            if not subtitle_text:
                continue

            # Strip HTML tags (some SRTs use <i>, <b>, etc.)
            subtitle_text = re.sub(r"<[^>]+>", "", subtitle_text)

            entries.append(
                SubtitleEntry(
                    subtitle_id=len(entries),
                    episode_id=episode_id,
                    video_path=video_path,
                    start_time=start,
                    end_time=end,
                    text=subtitle_text,
                )
            )

        return entries

    def parse_dir(self, srt_dir: Path) -> list[SubtitleEntry]:
        """Parse all .srt files in a directory."""
        entries: list[SubtitleEntry] = []
        for srt_path in sorted(srt_dir.glob("*.srt")):
            entries.extend(self.parse(srt_path))
        return entries

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read file with encoding fallback: UTF-8-BOM → UTF-8 → BIG5 → latin1."""
        for enc in ("utf-8-sig", "utf-8", "big5", "latin1"):
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return ""
