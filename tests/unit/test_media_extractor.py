"""Unit tests for MediaExtractor."""

import subprocess
from pathlib import Path

import pytest

from core.media_extractor import MediaExtractor


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Generate a 10-second black video with ffmpeg for testing."""
    video_path = tmp_path / "test_video.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=black:s=320x240:d=10",
            "-c:v", "libx264", "-t", "10",
            str(video_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("ffmpeg not available")
    return video_path


class TestMediaExtractor:
    def test_extract_frame_creates_file(self, dummy_video: Path, tmp_path: Path) -> None:
        """Frame extraction should produce a readable .jpg."""
        extractor = MediaExtractor(output_dir=tmp_path / "frames")
        frame_path = extractor.extract_frame(str(dummy_video), time_sec=2.0)
        assert frame_path.exists()
        assert frame_path.suffix == ".jpg"
        assert frame_path.stat().st_size > 0

    def test_extract_clip_creates_file(self, dummy_video: Path, tmp_path: Path) -> None:
        """Clip extraction should produce a readable .mp4."""
        extractor = MediaExtractor(output_dir=tmp_path / "clips")
        clip_path = extractor.extract_clip(str(dummy_video), start=1.0, end=4.0)
        assert clip_path.exists()
        assert clip_path.suffix == ".mp4"
        assert clip_path.stat().st_size > 0

    def test_extract_frame_out_of_range(self, dummy_video: Path, tmp_path: Path) -> None:
        """Frame at time beyond video length should handle gracefully."""
        extractor = MediaExtractor(output_dir=tmp_path / "frames")
        # 10-second video, request frame at 999s
        # Should either return last frame or raise a clear error
        with pytest.raises(Exception):
            extractor.extract_frame(str(dummy_video), time_sec=999.0)
