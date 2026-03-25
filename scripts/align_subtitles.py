"""Align Whisper-generated subtitles with reference SRT text.

Strategy:
  1. Whisper SRT → accurate timestamps, possibly wrong text
  2. Reference SRT → accurate text, wrong timestamps
  3. For each Whisper entry, fuzzy-match against reference text
  4. Replace Whisper text with reference text when match is good

Usage:
    python scripts/align_subtitles.py \
        --whisper data/subtitles/whisper/ep01.srt \
        --reference data/subtitles/ZhenHuan.E01...chs.srt \
        --output data/subtitles/aligned/ep01.srt \
        --threshold 0.4
"""

import argparse
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class SrtEntry:
    index: int
    start: str  # "HH:MM:SS,mmm"
    end: str
    text: str


def parse_srt(path: Path) -> list[SrtEntry]:
    """Parse an SRT file into entries."""
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find timestamp line
        ts_line = None
        ts_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line.strip()
                ts_idx = i
                break
        if ts_line is None:
            continue
        parts = ts_line.split("-->")
        if len(parts) != 2:
            continue
        text_lines = lines[ts_idx + 1:]
        text = " ".join(l.strip() for l in text_lines if l.strip())
        if not text:
            continue
        entries.append(SrtEntry(
            index=len(entries) + 1,
            start=parts[0].strip(),
            end=parts[1].strip(),
            text=text,
        ))
    return entries


def normalize(text: str) -> str:
    """Normalize text for comparison: remove punctuation and whitespace."""
    # Remove common Chinese/English punctuation
    text = re.sub(r"[，。！？、；：""''（）《》【】\s,.!?;:\"'()\[\]…—\-]", "", text)
    return text


def fuzzy_match(whisper_text: str, ref_entries: list[SrtEntry], threshold: float) -> SrtEntry | None:
    """Find the best matching reference entry for a Whisper entry."""
    w_norm = normalize(whisper_text)
    if not w_norm:
        return None

    best_score = 0.0
    best_entry = None

    for ref in ref_entries:
        r_norm = normalize(ref.text)
        if not r_norm:
            continue

        # Use SequenceMatcher for fuzzy matching
        score = SequenceMatcher(None, w_norm, r_norm).ratio()

        if score > best_score:
            best_score = score
            best_entry = ref

    if best_score >= threshold:
        return best_entry
    return None


def align(
    whisper_entries: list[SrtEntry],
    ref_entries: list[SrtEntry],
    threshold: float,
) -> list[SrtEntry]:
    """Align Whisper timestamps with reference text."""
    aligned = []
    used_refs: set[int] = set()
    stats = {"matched": 0, "kept_whisper": 0, "total": len(whisper_entries)}

    for w in whisper_entries:
        w_norm = normalize(w.text)
        if not w_norm:
            continue

        best_score = 0.0
        best_ref = None
        best_ref_idx = -1

        for i, ref in enumerate(ref_entries):
            if i in used_refs:
                continue
            r_norm = normalize(ref.text)
            if not r_norm:
                continue
            score = SequenceMatcher(None, w_norm, r_norm).ratio()
            if score > best_score:
                best_score = score
                best_ref = ref
                best_ref_idx = i

        if best_score >= threshold and best_ref is not None:
            # Use Whisper's timestamp + reference's text
            aligned.append(SrtEntry(
                index=len(aligned) + 1,
                start=w.start,
                end=w.end,
                text=best_ref.text,
            ))
            used_refs.add(best_ref_idx)
            stats["matched"] += 1
        else:
            # Keep Whisper's own text (no good match found)
            aligned.append(SrtEntry(
                index=len(aligned) + 1,
                start=w.start,
                end=w.end,
                text=w.text,
            ))
            stats["kept_whisper"] += 1

    print(f"  Matched: {stats['matched']}/{stats['total']} "
          f"({stats['matched']/stats['total']*100:.1f}%)")
    print(f"  Kept Whisper text: {stats['kept_whisper']}/{stats['total']}")
    return aligned


def write_srt(entries: list[SrtEntry], path: Path) -> None:
    """Write entries to an SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for e in entries:
        lines.append(str(e.index))
        lines.append(f"{e.start} --> {e.end}")
        lines.append(e.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Align Whisper SRT with reference SRT")
    parser.add_argument("--whisper", type=Path, required=True, help="Whisper-generated SRT")
    parser.add_argument("--reference", type=Path, required=True, help="Reference SRT (accurate text)")
    parser.add_argument("--output", type=Path, required=True, help="Output aligned SRT")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum similarity for match")
    args = parser.parse_args()

    print(f"Whisper:    {args.whisper}")
    print(f"Reference:  {args.reference}")

    w_entries = parse_srt(args.whisper)
    r_entries = parse_srt(args.reference)
    print(f"Whisper entries: {len(w_entries)}, Reference entries: {len(r_entries)}")

    aligned = align(w_entries, r_entries, args.threshold)
    write_srt(aligned, args.output)
    print(f"Output: {args.output} ({len(aligned)} entries)")


if __name__ == "__main__":
    main()
