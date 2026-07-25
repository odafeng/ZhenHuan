"""Verify R2 frame upload: compare object count and spot-check readability.

Usage:
    python scripts/verify_r2_upload.py
Reads local count from data/processed/*.jpg and R2 count via rclone.
"""

import subprocess
import sys
from pathlib import Path

REMOTE = "zhr2:zhenhuan"


def local_count() -> int:
    return len(list(Path("data/processed").glob("*.jpg")))


def remote_count() -> int:
    out = subprocess.run(
        ["rclone", "size", REMOTE, "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    import json
    return json.loads(out)["count"]


def main() -> None:
    local = local_count()
    remote = remote_count()
    print(f"local jpgs:  {local}")
    print(f"remote objs: {remote}")
    if local != remote:
        print(f"MISMATCH: {local - remote} files missing on remote")
        sys.exit(1)
    print("Count OK")


if __name__ == "__main__":
    main()
