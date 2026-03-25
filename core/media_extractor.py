"""FFmpeg wrapper for frame and clip extraction."""

import subprocess
from pathlib import Path


class MediaExtractor:
    """Extract frames and clips from video files using ffmpeg."""

    def __init__(self, output_dir: Path = Path("data/processed")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame(self, video_path: str, time_sec: float) -> Path:
        """Extract a single frame at the given timestamp.

        Returns path to the saved .jpg file.
        Raises RuntimeError if ffmpeg fails.
        """
        vp = Path(video_path)
        stem = vp.stem
        out_name = f"{stem}_{time_sec:.1f}s.jpg"
        out_path = self.output_dir / out_name

        if out_path.exists():
            return out_path

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{time_sec:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path),
            ],
            capture_output=True,
        )

        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"ffmpeg frame extraction failed at {time_sec}s: "
                f"{result.stderr.decode(errors='replace')}"
            )

        return out_path

    def extract_clip(self, video_path: str, start: float, end: float) -> Path:
        """Extract a video clip between start and end timestamps.

        Returns path to the saved .mp4 file.
        Raises RuntimeError if ffmpeg fails.
        """
        vp = Path(video_path)
        stem = vp.stem
        out_name = f"{stem}_{start:.1f}s_{end:.1f}s.mp4"
        out_path = self.output_dir / out_name

        if out_path.exists():
            return out_path

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", str(video_path),
                "-c", "copy",
                str(out_path),
            ],
            capture_output=True,
        )

        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"ffmpeg clip extraction failed [{start}s-{end}s]: "
                f"{result.stderr.decode(errors='replace')}"
            )

        return out_path
