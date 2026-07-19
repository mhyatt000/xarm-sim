"""Env-grid video tiling and a streamed mp4 writer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

BORDER = {0: (150, 150, 150), 1: (0, 200, 0), 2: (220, 30, 30)}  # live/success/fail


def tile_grid(frames: np.ndarray, max_width: int,
              status: np.ndarray | None = None,
              upscale: bool = False) -> np.ndarray:
    """(B, H, W, 3) -> near-square grid canvas, tiles resized to fit max_width.

    ``status`` draws a per-tile BORDER color (live/success/fail). ``upscale``
    lets tiles grow past their source size (nearest-neighbor; for small policy
    frames) — off, tiles only shrink (area; for full-res renders).
    """
    import cv2

    b, h, w, _ = frames.shape
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)
    tw = max(2, max_width // cols if upscale else min(w, max_width // cols)) // 2 * 2
    th = max(2, round(h * tw / w)) // 2 * 2
    interp = cv2.INTER_AREA if tw <= w else cv2.INTER_NEAREST
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    t = max(2, tw // 64)
    for i in range(b):
        r, c = divmod(i, cols)
        tile = cv2.resize(frames[i], (tw, th), interpolation=interp)
        if status is not None:
            tile[:t], tile[-t:], tile[:, :t], tile[:, -t:] = (BORDER[int(status[i])],) * 4
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
    return canvas


class VideoSink:
    """Streamed h264 mp4 writer: RGB uint8 frames piped to an ffmpeg
    subprocess, opened lazily on the first frame so callers never hold a full
    rollout of grids in RAM. h264 (not cv2's mp4v) so the wandb web UI can
    play the video inline."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps = path, fps
        self._p = None

    def add(self, frame: np.ndarray) -> None:
        if self._p is None:
            import shutil
            import subprocess

            # probe the PATH ffmpeg before trusting it: a broken install (e.g.
            # missing shared libs) dies at startup and surfaces here as an
            # opaque BrokenPipeError on the first write
            exe = shutil.which("ffmpeg")
            if exe is not None:
                try:
                    subprocess.run([exe, "-version"], capture_output=True, check=True)
                except (OSError, subprocess.CalledProcessError):
                    exe = None
            if exe is None:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._p = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.fps}",
                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 str(self.path)],
                stdin=subprocess.PIPE)
        self._p.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._p is not None:
            self._p.stdin.close()
            self._p.wait()
