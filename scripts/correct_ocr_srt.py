"""Post-process OCR subtitles: align with reference SRT and correct text.

OCR gives accurate timestamps (from burned-in subtitles).
Reference SRT gives accurate text (but timestamps may be offset).
This script combines: OCR timestamps + corrected reference text (in traditional Chinese).

Usage:
    python scripts/correct_ocr_srt.py --episodes 1-76 --workers 8
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import opencc

converter = opencc.OpenCC("s2t")  # simplified → traditional


def parse_srt(path: Path) -> list[dict]:
    """Parse SRT file into list of {index, start, end, text}."""
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
                text_lines = lines[i + 1 :]
                text = " ".join(l.strip() for l in text_lines if l.strip())
                if text:
                    entries.append({
                        "start": match.group(1),
                        "end": match.group(2),
                        "start_sec": ts_to_sec(match.group(1)),
                        "end_sec": ts_to_sec(match.group(2)),
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
    return re.sub(r"[，。！？、；：""''（）《》【】\s,.!?;:\"'()\[\]…—\-=]", "", text)


def correct_entry(ocr_text: str, ref_entries_cht: list[dict], threshold: float = 0.45) -> str | None:
    """Find best matching reference text for an OCR entry."""
    ocr_norm = normalize(ocr_text)
    if not ocr_norm or len(ocr_norm) < 2:
        return None

    best_score = 0.0
    best_ref = None
    for ref in ref_entries_cht:
        ref_norm = normalize(ref["text_cht"])
        if not ref_norm:
            continue
        score = SequenceMatcher(None, ocr_norm, ref_norm).ratio()
        if score > best_score:
            best_score = score
            best_ref = ref["text_cht"]

    return best_ref if best_score >= threshold else None


def process_episode(
    ep_num: int, ocr_dir: Path, ref_dir: Path, output_dir: Path
) -> str:
    ep_str = f"ep{ep_num:02d}"
    ocr_path = ocr_dir / f"{ep_str}.srt"
    output_path = output_dir / f"{ep_str}.srt"

    if not ocr_path.exists():
        return f"[{ep_str}] No OCR file, skipping"

    # Load OCR entries
    ocr_entries = parse_srt(ocr_path)
    if not ocr_entries:
        return f"[{ep_str}] Empty OCR file"

    # Load reference SRT (simplified Chinese)
    pattern = f"ZhenHuan.E{ep_num:02d}*.chs.srt"
    ref_matches = [p for p in ref_dir.glob(pattern) if "synced" not in p.name]

    ref_entries_cht = []
    if ref_matches:
        ref_entries = parse_srt(ref_matches[0])
        # Convert reference to traditional Chinese for matching
        for r in ref_entries:
            r["text_cht"] = converter.convert(r["text"])
        ref_entries_cht = ref_entries

    # Correct each OCR entry
    corrected_count = 0
    result_entries = []
    for ocr in ocr_entries:
        if ref_entries_cht:
            corrected = correct_entry(ocr["text"], ref_entries_cht)
            if corrected:
                result_entries.append((ocr["start_sec"], ocr["end_sec"], corrected))
                corrected_count += 1
            else:
                # Keep OCR text as-is
                result_entries.append((ocr["start_sec"], ocr["end_sec"], ocr["text"]))
        else:
            result_entries.append((ocr["start_sec"], ocr["end_sec"], ocr["text"]))

    # Write corrected SRT
    write_srt(result_entries, output_path)

    total = len(ocr_entries)
    return f"[{ep_str}] {corrected_count}/{total} corrected ({corrected_count/total*100:.0f}%)"


def write_srt(entries: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(start)} --> {sec_to_ts(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ocr-dir", type=Path, default=Path("data/subtitles/ocr"))
    parser.add_argument("--ref-dir", type=Path, default=Path("data/subtitles"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subtitles/corrected"))
    args = parser.parse_args()

    if "-" in args.episodes:
        start, end = args.episodes.split("-")
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(args.episodes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Correcting {len(episodes)} episodes, workers={args.workers}")
    print(f"  OCR dir:    {args.ocr_dir}")
    print(f"  Ref dir:    {args.ref_dir}")
    print(f"  Output dir: {args.output_dir}\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_episode, ep, args.ocr_dir, args.ref_dir, args.output_dir): ep
            for ep in episodes
        }
        for future in as_completed(futures):
            done += 1
            ep = futures[future]
            try:
                print(f"({done}/{len(episodes)}) {future.result()}")
            except Exception as e:
                print(f"({done}/{len(episodes)}) [ep{ep:02d}] FAILED: {e}")

    print(f"\nDone. Corrected SRTs in {args.output_dir}")


if __name__ == "__main__":
    main()
