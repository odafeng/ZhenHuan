"""Align OCR subtitles with reference SRT using sequential ordering constraint.

Key insight: even with deleted scenes, subtitle ORDER is preserved.
If OCR #50 matches reference #45, then OCR #51 must be near reference #46+.
This dramatically reduces false matches vs scanning all reference entries.

Pipeline:
  1. Forward pass: for each OCR entry, search reference starting from last match
  2. Use fuzzy text match with a limited search window (not full scan)
  3. Allow window to expand if no match found (handles deletions)
  4. Second pass: fill gaps using interpolated position estimates

Usage:
    python scripts/align_sequential.py --episodes 1-76
    python scripts/align_sequential.py --episodes 1 --verbose
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import opencc

converter = opencc.OpenCC("s2t")


def parse_srt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig")
    entries = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                line,
            )
            if m:
                text = " ".join(l.strip() for l in lines[i + 1 :] if l.strip())
                if text:
                    entries.append({
                        "start_sec": ts_to_sec(m.group(1)),
                        "end_sec": ts_to_sec(m.group(2)),
                        "text": text,
                    })
                break
    return entries


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


def normalize(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def sequential_align(
    ocr_entries: list[dict],
    ref_entries: list[dict],
    initial_window: int = 15,
    max_window: int = 50,
    match_threshold: float = 0.45,
    high_confidence: float = 0.70,
) -> list[dict]:
    """Align OCR to reference using sequential constraint.

    For each OCR entry, search a window of reference entries starting
    from the last matched position. This enforces ordering.

    Args:
        initial_window: how many reference entries to search forward
        max_window: expand to this if initial window finds nothing
        match_threshold: minimum similarity to accept a match
        high_confidence: above this, advance the cursor aggressively
    """
    results = []
    ref_cursor = 0  # current position in reference
    stats = {"matched": 0, "unmatched": 0, "total": len(ocr_entries)}

    for ocr_idx, ocr in enumerate(ocr_entries):
        ocr_norm = normalize(ocr["text"])

        # Skip entries with too little text to match
        if len(ocr_norm) < 2:
            results.append({
                "start_sec": ocr["start_sec"],
                "end_sec": ocr["end_sec"],
                "text": ocr["text"],
                "source": "ocr_short",
            })
            stats["unmatched"] += 1
            continue

        # Search forward from cursor with initial window
        best_score = 0.0
        best_ref_idx = -1
        best_ref = None

        # First try: narrow window (fast, avoids false matches)
        search_start = max(0, ref_cursor - 3)  # allow small backtrack
        search_end = min(len(ref_entries), ref_cursor + initial_window)

        for j in range(search_start, search_end):
            score = similarity(ocr["text"], ref_entries[j]["text"])
            if score > best_score:
                best_score = score
                best_ref_idx = j
                best_ref = ref_entries[j]

        # Second try: expand window if nothing found
        if best_score < match_threshold:
            search_end = min(len(ref_entries), ref_cursor + max_window)
            for j in range(ref_cursor + initial_window, search_end):
                score = similarity(ocr["text"], ref_entries[j]["text"])
                if score > best_score:
                    best_score = score
                    best_ref_idx = j
                    best_ref = ref_entries[j]

        if best_score >= match_threshold and best_ref is not None:
            # Use OCR timestamp + reference text (converted to traditional)
            ref_text_cht = converter.convert(best_ref["text"])
            results.append({
                "start_sec": ocr["start_sec"],
                "end_sec": ocr["end_sec"],
                "text": ref_text_cht,
                "source": "ref",
                "score": best_score,
            })
            stats["matched"] += 1

            # Advance cursor past this match
            if best_score >= high_confidence:
                ref_cursor = best_ref_idx + 1
            else:
                # Lower confidence: don't jump too far
                ref_cursor = max(ref_cursor, best_ref_idx)
        else:
            # No match: convert OCR text to traditional and keep it
            results.append({
                "start_sec": ocr["start_sec"],
                "end_sec": ocr["end_sec"],
                "text": converter.convert(ocr["text"]),
                "source": "ocr",
            })
            stats["unmatched"] += 1

    return results, stats


def process_episode(
    ep_num: int,
    ocr_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> str:
    ep_str = f"ep{ep_num:02d}"
    ocr_path = ocr_dir / f"{ep_str}.srt"

    if not ocr_path.exists():
        return f"[{ep_str}] No OCR file, skipping"

    # Find reference SRT
    pattern = f"ZhenHuan.E{ep_num:02d}*.chs.srt"
    ref_matches = [p for p in ref_dir.glob(pattern) if "synced" not in p.name]
    if not ref_matches:
        # Try other naming patterns
        for alt in [f"*E{ep_num:02d}*.srt", f"*e{ep_num:02d}*.srt"]:
            ref_matches = [
                p for p in ref_dir.glob(alt)
                if "ocr" not in p.name and "corrected" not in p.name
                and "cleaned" not in p.name and "synced" not in p.name
            ]
            if ref_matches:
                break

    if not ref_matches:
        return f"[{ep_str}] No reference file found"

    ocr_entries = parse_srt(ocr_path)
    ref_entries = parse_srt(ref_matches[0])

    if not ocr_entries:
        return f"[{ep_str}] Empty OCR file"
    if not ref_entries:
        return f"[{ep_str}] Empty reference file"

    # Align
    aligned, stats = sequential_align(ocr_entries, ref_entries)

    if verbose:
        print(f"\n[{ep_str}] Alignment details:")
        for i, r in enumerate(aligned):
            src = r["source"]
            marker = "  " if src == "ref" else ">>"
            score_str = f" ({r['score']:.2f})" if "score" in r else ""
            print(f"  {marker} [{r['start_sec']:.0f}s] {r['text'][:40]}{score_str} [{src}]")

    # Write output
    output_path = output_dir / f"{ep_str}.srt"
    write_srt(aligned, output_path)

    rate = stats["matched"] / stats["total"] * 100 if stats["total"] else 0
    return f"[{ep_str}] {stats['matched']}/{stats['total']} matched ({rate:.0f}%) ref={ref_matches[0].name}"


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
    parser = argparse.ArgumentParser(description="Sequential subtitle alignment")
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ocr-dir", type=Path, default=Path("data/subtitles/ocr"))
    parser.add_argument("--ref-dir", type=Path, default=Path("data/subtitles"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subtitles/corrected"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if "-" in args.episodes:
        start, end = args.episodes.split("-")
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(args.episodes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sequential alignment: {len(episodes)} episodes")
    print(f"  OCR dir: {args.ocr_dir}")
    print(f"  Ref dir: {args.ref_dir}")
    print(f"  Output:  {args.output_dir}\n")

    if args.verbose or len(episodes) == 1:
        # Single-threaded for verbose output
        for ep in episodes:
            result = process_episode(ep, args.ocr_dir, args.ref_dir, args.output_dir, args.verbose)
            print(result)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_episode, ep, args.ocr_dir, args.ref_dir, args.output_dir
                ): ep
                for ep in episodes
            }
            for future in as_completed(futures):
                done += 1
                try:
                    print(f"({done}/{len(episodes)}) {future.result()}")
                except Exception as e:
                    ep = futures[future]
                    print(f"({done}/{len(episodes)}) [ep{ep:02d}] FAILED: {e}")

    print(f"\nDone. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
