"""Media helpers: type detection, audio decode (ffmpeg), frame sampling (OpenCV)."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUD = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def kind(path) -> str | None:
    ext = Path(path).suffix.lower()
    return "image" if ext in IMG else "video" if ext in VID else "audio" if ext in AUD else None


def load_audio(path, sr: int = 16000):
    """Mono float32 array at `sr`, or None if ffmpeg is missing / no audio track."""
    if not shutil.which("ffmpeg"):
        return None
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, timeout=180)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32)


def sample_frames(path, n: int = 8):
    """Evenly spaced RGB frames."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for i in (np.linspace(0, max(total - 1, 0), n).astype(int) if total else range(n)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
