"""Transcribe video audio using OpenAI Whisper API.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/whisper_transcribe.py --episodes 1
    python scripts/whisper_transcribe.py --episodes 1-76
    python scripts/whisper_transcribe.py --episodes 1-76 --workers 5
"""

import argparse
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video as MP3 (64kbps to stay under 25MB)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame", "-b:a", "64k",
            str(audio_path),
        ],
        capture_output=True,
        check=True,
    )


def transcribe(client: OpenAI, audio_path: Path, max_retries: int = 3) -> str:
    """Call OpenAI Whisper API and return SRT content."""
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="srt",
                    language="zh",
                )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                raise


def process_episode(ep: int, videos_dir: Path, output_dir: Path) -> str:
    """Process a single episode: extract audio → transcribe → save."""
    client = OpenAI()
    ep_str = f"ep{ep:02d}"
    video_path = videos_dir / f"{ep_str}.mp4"
    output_path = output_dir / f"{ep_str}.srt"

    if output_path.exists():
        return f"[{ep_str}] Already exists, skipping"

    if not video_path.exists():
        return f"[{ep_str}] Video not found"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        extract_audio(video_path, audio_path)
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        srt_content = transcribe(client, audio_path)
        output_path.write_text(srt_content, encoding="utf-8")
        return f"[{ep_str}] Done ({size_mb:.1f}MB)"
    except Exception as e:
        return f"[{ep_str}] FAILED: {e}"
    finally:
        audio_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True, help="Episode range: '1' or '1-76'")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/subtitles/whisper"))
    args = parser.parse_args()

    if "-" in args.episodes:
        start, end = args.episodes.split("-")
        episodes = list(range(int(start), int(end) + 1))
    else:
        episodes = [int(args.episodes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only episodes that need processing
    todo = []
    for ep in episodes:
        out = args.output_dir / f"ep{ep:02d}.srt"
        if not out.exists():
            todo.append(ep)

    print(f"Total: {len(episodes)}, Already done: {len(episodes)-len(todo)}, Todo: {len(todo)}")
    print(f"Workers: {args.workers}\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_episode, ep, args.videos_dir, args.output_dir): ep
            for ep in todo
        }
        for future in as_completed(futures):
            done += 1
            result = future.result()
            print(f"({done}/{len(todo)}) {result}")

    print(f"\nAll done. Files in {args.output_dir}")


if __name__ == "__main__":
    main()
