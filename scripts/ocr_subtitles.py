"""OCR burned-in subtitles from video files.

Usage:
    python scripts/ocr_subtitles.py --episodes 1-76 --workers 2
"""

import argparse
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

# Thread-local storage for EasyOCR reader (one per worker thread, not per episode)
_thread_local = threading.local()


def get_reader() -> easyocr.Reader:
    """Get or create a thread-local EasyOCR reader."""
    if not hasattr(_thread_local, "reader"):
        _thread_local.reader = easyocr.Reader(["ch_tra", "en"], gpu=False, verbose=False)
    return _thread_local.reader


def extract_frame(video_path: str, t: float) -> str | None:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", tmp.name],
        capture_output=True,
    )
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        return None
    return tmp.name


def has_subtitle(frame_path: str) -> bool:
    img = Image.open(frame_path)
    w, h = img.size
    bottom = img.crop((0, int(h * 0.80), w, int(h * 0.95)))
    arr = np.array(bottom.convert("L"))
    return (arr > 200).mean() >= 0.01


def ocr_frame(reader: easyocr.Reader, frame_path: str) -> str:
    img = Image.open(frame_path)
    w, h = img.size
    bottom = img.crop((int(w * 0.1), int(h * 0.78), int(w * 0.9), int(h * 0.95)))
    arr = np.array(bottom)
    results = reader.readtext(arr, detail=0)
    return " ".join(results).strip()


def load_reference_texts(ref_dir: Path, ep_num: int) -> list[str]:
    """Load reference SRT texts for fuzzy correction."""
    pattern = f"ZhenHuan.E{ep_num:02d}*.chs.srt"
    matches = [p for p in ref_dir.glob(pattern) if "synced" not in p.name]
    if not matches:
        return []
    import re
    content = matches[0].read_text(encoding="utf-8-sig")
    texts = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            if "-->" in line:
                text_lines = lines[i + 1:]
                text = " ".join(l.strip() for l in text_lines if l.strip())
                if text:
                    texts.append(text)
                break
    return texts


def normalize(text: str) -> str:
    import re
    return re.sub(r"[，。！？、；：""''（）《》【】\s,.!?;:\"'()\[\]…—\-=]", "", text)


def correct_ocr(ocr_text: str, ref_texts: list[str], threshold: float = 0.5) -> str:
    """Correct OCR text using reference SRT via fuzzy matching."""
    if not ocr_text or not ref_texts:
        return ocr_text
    ocr_norm = normalize(ocr_text)
    if not ocr_norm:
        return ocr_text

    best_score = 0.0
    best_ref = ocr_text
    for ref in ref_texts:
        ref_norm = normalize(ref)
        if not ref_norm:
            continue
        score = SequenceMatcher(None, ocr_norm, ref_norm).ratio()
        if score > best_score:
            best_score = score
            best_ref = ref

    return best_ref if best_score >= threshold else ocr_text


def is_same_subtitle(text_a: str, text_b: str, threshold: float = 0.7) -> bool:
    """Fuzzy check if two OCR texts are the same subtitle (despite OCR noise)."""
    if text_a == text_b:
        return True
    if not text_a or not text_b:
        return False
    norm_a = normalize(text_a)
    norm_b = normalize(text_b)
    if not norm_a or not norm_b:
        return False
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold


def process_episode(
    ep_num: int, videos_dir: Path, output_dir: Path, ref_dir: Path
) -> str:
    """OCR one episode's burned-in subtitles."""
    reader = get_reader()  # thread-local, reused across episodes
    ep_str = f"ep{ep_num:02d}"
    video_path = videos_dir / f"{ep_str}.mp4"
    output_path = output_dir / f"{ep_str}.srt"

    if output_path.exists():
        return f"[{ep_str}] Already exists, skipping"
    if not video_path.exists():
        return f"[{ep_str}] Video not found"

    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())

    # Load reference texts for correction
    ref_texts = load_reference_texts(ref_dir, ep_num)

    # OCR loop: every 1 second
    entries = []  # (start_time, end_time, text)
    prev_text = ""
    prev_raw_texts = []  # collect raw OCR texts for this subtitle span
    current_start = 0.0
    start_time = time.time()

    for t_ms in range(0, int(duration * 1000), 1000):
        t = t_ms / 1000
        frame = extract_frame(str(video_path), t)
        if not frame:
            continue

        if not has_subtitle(frame):
            Path(frame).unlink(missing_ok=True)
            if prev_text:
                # Pick the best raw text (most common or longest) then correct
                best = _pick_best_raw(prev_raw_texts)
                corrected = correct_ocr(best, ref_texts)
                entries.append((current_start, t, corrected))
                prev_text = ""
                prev_raw_texts = []
            continue

        text = ocr_frame(reader, frame)
        Path(frame).unlink(missing_ok=True)

        if not text:
            if prev_text:
                best = _pick_best_raw(prev_raw_texts)
                corrected = correct_ocr(best, ref_texts)
                entries.append((current_start, t, corrected))
                prev_text = ""
                prev_raw_texts = []
            continue

        if is_same_subtitle(text, prev_text):
            # Same subtitle, just accumulate the raw text
            prev_raw_texts.append(text)
        else:
            # New subtitle
            if prev_text:
                best = _pick_best_raw(prev_raw_texts)
                corrected = correct_ocr(best, ref_texts)
                entries.append((current_start, t, corrected))
            current_start = t
            prev_text = text
            prev_raw_texts = [text]

    # Final entry
    if prev_text:
        best = _pick_best_raw(prev_raw_texts)
        corrected = correct_ocr(best, ref_texts)
        entries.append((current_start, duration, corrected))

    elapsed = time.time() - start_time

    # Write SRT
    write_srt(entries, output_path)

    return f"[{ep_str}] Done: {len(entries)} entries, {elapsed:.0f}s"


def _pick_best_raw(raw_texts: list[str]) -> str:
    """Pick the best OCR text from multiple reads of the same subtitle.

    Strategy: pick the longest text (usually most complete OCR result).
    """
    if not raw_texts:
        return ""
    return max(raw_texts, key=len)


def write_srt(entries: list[tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--videos-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subtitles/ocr"))
    parser.add_argument("--ref-dir", type=Path, default=Path("data/subtitles"))
    args = parser.parse_args()

    if "-" in args.episodes:
        start, end = args.episodes.split("-")
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(args.episodes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    todo = [ep for ep in episodes if not (args.output_dir / f"ep{ep:02d}.srt").exists()]
    print(f"Total: {len(episodes)}, Todo: {len(todo)}, Workers: {args.workers}\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_episode, ep, args.videos_dir, args.output_dir, args.ref_dir): ep
            for ep in todo
        }
        for future in as_completed(futures):
            done += 1
            ep = futures[future]
            try:
                print(f"({done}/{len(todo)}) {future.result()}")
            except Exception as e:
                print(f"({done}/{len(todo)}) [ep{ep:02d}] FAILED: {e}")

    print(f"\nDone. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
