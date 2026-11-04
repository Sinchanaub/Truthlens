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


_DET = {}


def _pick_detector():
    """Choose a face detector once. Order: OpenCV Haar -> MediaPipe -> none.

    OpenCV 5.x dropped cv2.CascadeClassifier, so requirements pin opencv<5; this is the safety net
    so a bad OpenCV build can never again turn every image into "Analysis failed"."""
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not c.empty():
                return "haar", c
    except Exception:  # noqa: BLE001
        pass
    try:
        import mediapipe as mp
        return "mediapipe", mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except Exception:  # noqa: BLE001
        return "none", None


def _detector():
    if "d" not in _DET:
        _DET["d"] = _pick_detector()
    return _DET["d"]


def face_detector_name() -> str:
    return _detector()[0]


def haar():
    """The OpenCV Haar classifier, or None when this OpenCV build has no cascade support."""
    name, det = _detector()
    return det if name == "haar" else None


def detect_faces(img: Image.Image) -> list:
    """[(x, y, w, h)] in pixels. Empty list when no face or no detector is available. Never raises."""
    name, det = _detector()
    try:
        if name == "haar":
            g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            return [tuple(int(v) for v in f) for f in det.detectMultiScale(g, 1.1, 5, minSize=(48, 48))]
        if name == "mediapipe":
            res = det.process(np.asarray(img))
            out = []
            for d in res.detections or []:
                b = d.location_data.relative_bounding_box
                out.append((int(b.xmin * img.width), int(b.ymin * img.height),
                            int(b.width * img.width), int(b.height * img.height)))
            return out
    except Exception:  # noqa: BLE001
        pass
    return []


def face_crop(img: Image.Image, margin: float = 0.3, force: bool = False):
    """Crop the largest face (+margin). The classifier was trained on face crops, so full-scene
    frames are out-of-domain. Images that are already face-sized are left alone unless force=True.
    Returns (PIL image, found_face)."""
    faces = [f for f in detect_faces(img) if f[2] > 0 and f[3] > 0]
    if not faces:
        return img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    if not force and w > 0.6 * img.width:
        return img, True
    m = int(margin * max(w, h))
    box = (max(0, x - m), max(0, y - m), min(img.width, x + w + m), min(img.height, y + h + m))
    return img.crop(box), True
