"""Clean corrected SRT files: strip credits, garbage, duplicates.

Reads from corrected/ → writes to cleaned/. Designed to run after correct_ocr_srt.py.

Usage:
    python scripts/clean_srt.py --episodes 1-76
"""

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path


# --- Garbage patterns ---
_GARBAGE_RE = re.compile(
    r"www|com\.|GTN|普級|菸酒|編曲|作曲|主唱|姚貝娜|敬請斟酌"
    r"|本節目|觀賞|第.{0,2}集|片尾曲|片頭曲|導演|監製|出品"
    r"|演員|編劇|製片|攝影|美術|剪輯|錄音|配音"
    r"|全劇終|金劇終|劇終|完結"
    r"|[\^@#\[\]{}]"
)

# Characters that are clearly OCR noise
_NOISE_CHARS_RE = re.compile(r"[0-9a-zA-Z_=+<>|/\\`~]")


def ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig")
    entries = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        for i, line in enumerate(lines):
            match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                line,
            )
            if match:
                text = " ".join(l.strip() for l in lines[i + 1:] if l.strip())
                if text:
                    entries.append({
                        "start_sec": ts_to_sec(match.group(1)),
                        "end_sec": ts_to_sec(match.group(2)),
                        "text": text,
                    })
                break
    return entries


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def chinese_ratio(text: str) -> float:
    total = len(text.strip())
    if total == 0:
        return 0
    return chinese_char_count(text) / total


def normalize(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def is_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def is_garbage(entry: dict, credits_start: float = 55, credits_end: float = 2400) -> tuple[bool, str]:
    """Check if an entry should be removed. Returns (is_garbage, reason)."""
    text = entry["text"].strip()
    sec = entry["start_sec"]

    # 0. Empty text
    if not text:
        return True, "empty"

    # 1. Credits region
    if sec < credits_start:
        return True, "opening_credits"
    if sec > credits_end:
        return True, "ending_credits"

    # 2. Known garbage patterns
    if _GARBAGE_RE.search(text):
        return True, "garbage_pattern"

    # 3. Too short (≤ 2 Chinese characters)
    cc = chinese_char_count(text)
    if cc <= 2 and len(text.strip()) <= 3:
        return True, "too_short"

    # 4. Too few Chinese characters (OCR noise)
    if len(text.strip()) > 3 and chinese_ratio(text) < 0.5:
        return True, "low_chinese_ratio"

    # 5. Too many noise characters
    noise_count = len(_NOISE_CHARS_RE.findall(text))
    if noise_count > len(text) * 0.3 and len(text) > 3:
        return True, "noise_chars"

    return False, ""


def dedup_consecutive(entries: list[dict]) -> list[dict]:
    """Remove consecutive near-duplicate entries, keeping the one with more Chinese chars."""
    if not entries:
        return []

    result = [entries[0]]
    for entry in entries[1:]:
        prev = result[-1]
        time_gap = entry["start_sec"] - prev["start_sec"]

        # If within 3 seconds and similar text → duplicate
        if time_gap <= 3 and is_similar(entry["text"], prev["text"]):
            # Keep the one with more Chinese characters (usually better OCR)
            if chinese_char_count(entry["text"]) > chinese_char_count(prev["text"]):
                result[-1] = entry
            # Extend end time to cover both
            result[-1]["end_sec"] = max(prev["end_sec"], entry["end_sec"])
        else:
            result.append(entry)

    return result


def clean_episode(input_path: Path, output_path: Path) -> dict:
    """Clean one episode's SRT file. Returns stats."""
    entries = parse_srt(input_path)
    stats = {
        "total": len(entries),
        "opening_credits": 0,
        "ending_credits": 0,
        "garbage_pattern": 0,
        "too_short": 0,
        "low_chinese_ratio": 0,
        "noise_chars": 0,
        "duplicates": 0,
        "kept": 0,
    }

    # Step 1: Remove garbage
    clean = []
    for entry in entries:
        bad, reason = is_garbage(entry)
        if bad:
            stats[reason] = stats.get(reason, 0) + 1
        else:
            clean.append(entry)

    # Step 2: Dedup consecutive similar entries
    before_dedup = len(clean)
    clean = dedup_consecutive(clean)
    stats["duplicates"] = before_dedup - len(clean)

    # Step 3: Write
    stats["kept"] = len(clean)
    write_srt(clean, output_path)
    return stats


def write_srt(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, entry in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(entry['start_sec'])} --> {sec_to_ts(entry['end_sec'])}")
        lines.append(entry["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Clean corrected SRT files")
    parser.add_argument("--episodes", type=str, required=True, help="e.g. 1-76 or 1")
    parser.add_argument("--input-dir", type=Path, default=Path("data/subtitles/corrected"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subtitles/cleaned"))
    args = parser.parse_args()

    if "-" in args.episodes:
        start, end = args.episodes.split("-")
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(args.episodes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_stats = {
        "total": 0, "kept": 0, "opening_credits": 0, "ending_credits": 0,
        "garbage_pattern": 0, "too_short": 0, "low_chinese_ratio": 0,
        "noise_chars": 0, "duplicates": 0, "empty": 0,
    }

    print(f"Cleaning {len(episodes)} episodes")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}\n")
    print(f"{'EP':>4} {'Total':>6} {'Kept':>5} {'Credits':>8} {'Garbage':>8} {'Short':>6} {'Dupes':>6} {'Clean%':>7}")
    print("-" * 65)

    for ep in episodes:
        input_path = args.input_dir / f"ep{ep:02d}.srt"
        output_path = args.output_dir / f"ep{ep:02d}.srt"
        if not input_path.exists():
            print(f"ep{ep:02d}  -- not found, skipping")
            continue

        stats = clean_episode(input_path, output_path)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        credits = stats["opening_credits"] + stats["ending_credits"]
        pct = stats["kept"] / stats["total"] * 100 if stats["total"] else 0
        print(
            f"ep{ep:02d} {stats['total']:6d} {stats['kept']:5d} "
            f"{credits:8d} {stats['garbage_pattern']:8d} "
            f"{stats['too_short']:6d} {stats['duplicates']:6d} {pct:6.1f}%"
        )

    print("-" * 65)
    total_credits = total_stats["opening_credits"] + total_stats["ending_credits"]
    total_removed = total_stats["total"] - total_stats["kept"]
    print(
        f"{'SUM':>4} {total_stats['total']:6d} {total_stats['kept']:5d} "
        f"{total_credits:8d} {total_stats['garbage_pattern']:8d} "
        f"{total_stats['too_short']:6d} {total_stats['duplicates']:6d} "
        f"{total_stats['kept']/total_stats['total']*100:.1f}%"
    )
    print(f"\nRemoved {total_removed} entries total. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
