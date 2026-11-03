"""Media helpers: type detection, audio decode (ffmpeg), frame sampling (OpenCV)."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUD = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def kind(path) -> str | None:
    ext = Path(path).suffix.lower()

    if ext in IMG:
        return "image"

    if ext in VID:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            streams = result.stdout.strip().splitlines()

            if "video" in streams:
                return "video"

            if "audio" in streams:
                return "audio"

        except Exception:
            return "video"

    if ext in AUD:
        return "audio"

    return None
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


def open_image(path) -> Image.Image:
    """RGB image with phone EXIF rotation applied."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


_HAAR = {}


def face_crop(img: Image.Image, margin: float = 0.3, force: bool = False):
    """Crop the largest face (+margin). The classifier was trained on face crops, so full-scene
    frames are out-of-domain. Images that are already face-sized are left alone unless force=True.
    Returns (PIL image, found_face)."""
    if "c" not in _HAAR:
        _HAAR["c"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    faces = _HAAR["c"].detectMultiScale(g, 1.1, 5, minSize=(48, 48))
    if not len(faces):
        return img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    if not force and w > 0.6 * img.width:
        return img, True
    m = int(margin * max(w, h))
    box = (max(0, x - m), max(0, y - m), min(img.width, x + w + m), min(img.height, y + h + m))
    return img.crop(box), True
