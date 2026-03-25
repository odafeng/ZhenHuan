"""Batch align all Whisper SRTs with reference SRTs.

Usage:
    python scripts/batch_align.py
"""

from pathlib import Path
from align_subtitles import parse_srt, align, write_srt

WHISPER_DIR = Path("data/subtitles/whisper")
REFERENCE_DIR = Path("data/subtitles")
OUTPUT_DIR = Path("data/subtitles/aligned")
THRESHOLD = 0.4


def find_reference(ep_num: int) -> Path | None:
    """Find the reference SRT for a given episode number."""
    pattern = f"ZhenHuan.E{ep_num:02d}*.chs.srt"
    # Exclude synced files
    matches = [
        p for p in REFERENCE_DIR.glob(pattern)
        if "synced" not in p.name
    ]
    return matches[0] if matches else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    whisper_files = sorted(WHISPER_DIR.glob("ep*.srt"))
    print(f"Found {len(whisper_files)} Whisper SRT files\n")

    total_matched = 0
    total_entries = 0

    for wf in whisper_files:
        # Extract episode number from filename like "ep01.srt"
        ep_num = int(wf.stem.replace("ep", ""))
        ref_path = find_reference(ep_num)

        if ref_path is None:
            print(f"[ep{ep_num:02d}] No reference SRT found, skipping")
            continue

        output_path = OUTPUT_DIR / wf.name
        print(f"[ep{ep_num:02d}] Aligning...")

        w_entries = parse_srt(wf)
        r_entries = parse_srt(ref_path)

        aligned = align(w_entries, r_entries, THRESHOLD)
        write_srt(aligned, output_path)

        matched = sum(1 for a, w in zip(aligned, w_entries) if a.text != w.text)
        total_matched += matched
        total_entries += len(w_entries)

    print(f"\n{'='*60}")
    print(f"Total: {total_matched}/{total_entries} entries corrected "
          f"({total_matched/total_entries*100:.1f}%)" if total_entries > 0 else "No entries")


if __name__ == "__main__":
    main()
