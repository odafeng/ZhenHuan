"""FFmpeg wrapper for frame and clip extraction."""

from pathlib import Path


class MediaExtractor:
    """Extract frames and clips from video files using ffmpeg."""

    def __init__(self, output_dir: Path = Path("data/processed")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame(self, video_path: str, time_sec: float) -> Path:
        """Extract a single frame at the given timestamp.

        Returns path to the saved .jpg file.
        """
        raise NotImplementedError

    def extract_clip(self, video_path: str, start: float, end: float) -> Path:
        """Extract a video clip between start and end timestamps.

        Returns path to the saved .mp4 file.
        """
        raise NotImplementedError
