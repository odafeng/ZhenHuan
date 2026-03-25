"""FFmpeg wrapper for frame and clip extraction with subtitle gating."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# Minimum percentage of bright pixels (>200) in bottom strip to confirm subtitle presence
_SUBTITLE_BRIGHT_THRESHOLD = 0.01  # 1%


class MediaExtractor:
    """Extract frames and clips from video files using ffmpeg."""

    def __init__(self, output_dir: Path = Path("data/processed")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame_with_subtitle(
        self, video_path: str, start: float, end: float, num_samples: int = 5
    ) -> Path | None:
        """Extract the best frame with visible burned-in subtitle.

        Samples multiple frames within [start, end], checks each for
        subtitle presence, returns the best one (closest to midpoint).
        Returns None if no frame has a visible subtitle.
        """
        if end <= start:
            end = start + 1.0

        duration = end - start
        mid = (start + end) / 2

        # Generate sample times, centered around midpoint
        if duration < 1.0:
            times = [mid]
        else:
            step = duration / (num_samples + 1)
            times = [start + step * (i + 1) for i in range(num_samples)]

        # Sort by distance to midpoint (prefer center frames)
        times.sort(key=lambda t: abs(t - mid))

        for t in times:
            tmp_path = self._extract_raw_frame(video_path, t)
            if tmp_path and self._has_subtitle(tmp_path):
                # Move to output dir with proper name
                vp = Path(video_path)
                out_name = f"{vp.stem}_{t:.1f}s.jpg"
                out_path = self.output_dir / out_name
                if out_path.exists():
                    tmp_path.unlink(missing_ok=True)
                    return out_path
                tmp_path.rename(out_path)
                return out_path
            if tmp_path:
                tmp_path.unlink(missing_ok=True)

        # Fallback: return midpoint frame even without subtitle
        return self.extract_frame(video_path, mid)

    def extract_frame(self, video_path: str, time_sec: float) -> Path:
        """Extract a single frame at the given timestamp."""
        vp = Path(video_path)
        out_name = f"{vp.stem}_{time_sec:.1f}s.jpg"
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
        """Extract a video clip between start and end timestamps."""
        vp = Path(video_path)
        out_name = f"{vp.stem}_{start:.1f}s_{end:.1f}s.mp4"
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

    def _extract_raw_frame(self, video_path: str, time_sec: float) -> Path | None:
        """Extract frame to a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{time_sec:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(tmp_path),
            ],
            capture_output=True,
        )

        if result.returncode != 0 or not tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
            return None
        return tmp_path

    @staticmethod
    def _has_subtitle(frame_path: Path) -> bool:
        """Check if a frame has visible burned-in subtitle in the bottom region."""
        try:
            img = Image.open(frame_path)
            w, h = img.size
            # Crop bottom 15% (where subtitles appear)
            bottom = img.crop((0, int(h * 0.80), w, int(h * 0.95)))
            arr = np.array(bottom.convert("L"))
            bright_ratio = (arr > 200).mean()
            return bright_ratio >= _SUBTITLE_BRIGHT_THRESHOLD
        except Exception:
            return False
