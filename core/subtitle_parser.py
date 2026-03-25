"""Parse .srt and .ass subtitle files into structured SubtitleEntry list."""

import re
from pathlib import Path

from schemas import SubtitleEntry

_SRT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)

_ASS_DIALOGUE_RE = re.compile(
    r"^Dialogue:\s*\d+,"
    r"(\d+):(\d{2}):(\d{2})\.(\d{2}),"  # start h:mm:ss.cc
    r"(\d+):(\d{2}):(\d{2})\.(\d{2}),"  # end h:mm:ss.cc
    r"([^,]*),"   # style
    r"([^,]*),"   # name
    r"\d+,\d+,\d+,"  # marginL, marginR, marginV
    r"([^,]*),"   # effect
    r"(.*)"       # text
)

# ASS override tags like {\blur3}, {\pos(x,y)}, {\an8}, etc.
_ASS_TAGS_RE = re.compile(r"\{[^}]*\}")


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _ass_ts_to_sec(h: str, m: str, s: str, cs: str) -> float:
    """ASS uses centiseconds (2 digits), not milliseconds."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100


class SubtitleParser:
    """Parse SRT and ASS subtitle files into structured subtitle entries."""

    def parse(self, path: Path) -> list[SubtitleEntry]:
        """Parse a subtitle file (auto-detect format by extension)."""
        suffix = path.suffix.lower()
        if suffix == ".srt":
            return self._parse_srt(path)
        elif suffix in (".ass", ".ssa"):
            return self._parse_ass(path)
        else:
            raise ValueError(f"Unsupported subtitle format: {suffix}")

    def _parse_srt(self, srt_path: Path) -> list[SubtitleEntry]:
        """Parse a .srt file into a list of SubtitleEntry."""
        text = self._read_file(srt_path)
        if not text.strip():
            return []

        episode_id = self._extract_episode_id(srt_path)
        video_path = f"data/videos/{episode_id}.mp4"

        entries: list[SubtitleEntry] = []
        blocks = re.split(r"\n\s*\n", text.strip())

        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue

            ts_match = None
            ts_line_idx = -1
            for i, line in enumerate(lines):
                ts_match = _SRT_TIMESTAMP_RE.match(line.strip())
                if ts_match:
                    ts_line_idx = i
                    break

            if ts_match is None:
                continue

            g = ts_match.groups()
            start = _ts_to_sec(g[0], g[1], g[2], g[3])
            end = _ts_to_sec(g[4], g[5], g[6], g[7])

            text_lines = lines[ts_line_idx + 1 :]
            subtitle_text = " ".join(line.strip() for line in text_lines if line.strip())

            if not subtitle_text:
                continue

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

    def _parse_ass(self, ass_path: Path) -> list[SubtitleEntry]:
        """Parse a .ass/.ssa file into a list of SubtitleEntry."""
        text = self._read_file(ass_path)
        if not text.strip():
            return []

        episode_id = self._extract_episode_id(ass_path)
        video_path = f"data/videos/{episode_id}.mp4"

        entries: list[SubtitleEntry] = []

        for line in text.splitlines():
            line = line.strip()
            m = _ASS_DIALOGUE_RE.match(line)
            if not m:
                continue

            g = m.groups()
            start = _ass_ts_to_sec(g[0], g[1], g[2], g[3])
            end = _ass_ts_to_sec(g[4], g[5], g[6], g[7])

            raw_text = g[11]  # the text field

            # Strip ASS override tags like {\blur3}
            clean = _ASS_TAGS_RE.sub("", raw_text)
            # ASS uses \N for line breaks
            clean = clean.replace("\\N", " ").replace("\\n", " ")
            clean = clean.strip()

            if not clean:
                continue

            entries.append(
                SubtitleEntry(
                    subtitle_id=len(entries),
                    episode_id=episode_id,
                    video_path=video_path,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    text=clean,
                )
            )

        return entries

    def parse_dir(self, srt_dir: Path) -> list[SubtitleEntry]:
        """Parse all subtitle files (.srt, .ass, .ssa) in a directory."""
        entries: list[SubtitleEntry] = []
        for ext in ("*.srt", "*.ass", "*.ssa"):
            for path in sorted(srt_dir.glob(ext)):
                entries.extend(self.parse(path))
        return entries

    @staticmethod
    def _extract_episode_id(path: Path) -> str:
        """Extract episode ID from filename.

        Tries to find 'E01' pattern; falls back to full stem.
        """
        stem = path.stem
        m = re.search(r"[Ee](\d{1,3})", stem)
        if m:
            return f"ep{int(m.group(1)):02d}"
        return stem

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read file with encoding fallback: UTF-8-BOM → UTF-8 → BIG5 → latin1."""
        for enc in ("utf-8-sig", "utf-8", "big5", "latin1"):
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return ""
