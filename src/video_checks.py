"""Physical-consistency check: does mouth motion correlate with the audio energy envelope?
Needs no training. Uses MediaPipe FaceMesh if installed, else OpenCV Haar + mouth-region motion."""
import cv2
import numpy as np

_MESH = {}


def _mesh():
    if "m" not in _MESH:
        try:
            import mediapipe as mp
            _MESH["m"] = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        except Exception:
            _MESH["m"] = None
    return _MESH["m"]


def mouth_series(path, max_frames: int = 600):
    """Returns (mouth signal per frame with NaN for no-face, fps, face_ratio, method)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    mesh = _mesh()
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    vals, prev, n = [], None, 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if fr.shape[1] > 320:
            fr = cv2.resize(fr, (320, int(fr.shape[0] * 320 / fr.shape[1])))
        v = np.nan
        if mesh is not None:
            res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                v = abs(lm[13].y - lm[14].y) / (abs(lm[10].y - lm[152].y) + 1e-6)
        else:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(g, 1.2, 5, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi = cv2.resize(g[y + int(.62 * h): y + h, x + int(.2 * w): x + int(.8 * w)], (32, 16)).astype(np.float32)
                v = float(np.abs(roi - prev).mean()) if prev is not None else np.nan
                prev = roi
            else:
                prev = None
        vals.append(v)
    cap.release()
    arr = np.array(vals, dtype=np.float32)
    ratio = float(np.mean(~np.isnan(arr))) if len(arr) else 0.0
    return arr, fps, ratio, "mediapipe-facemesh" if mesh is not None else "opencv-haar-motion"


def envelope(wav: np.ndarray, sr: int, fps: float, n: int) -> np.ndarray:
    hop = max(1, int(sr / fps))
    return np.array([np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2)) if len(wav[i * hop:(i + 1) * hop]) else 0.0
                     for i in range(n)], dtype=np.float32)


def _z(x):
    x = np.convolve(x, np.ones(3) / 3, mode="same")
    return (x - x.mean()) / (x.std() + 1e-9)


def best_corr(mouth: np.ndarray, env: np.ndarray, max_lag: int = 4) -> float:
    """Max Pearson correlation over small lags (handles A/V offset)."""
    m = mouth.copy()
    idx = np.arange(len(m))
    ok = ~np.isnan(m)
    if ok.sum() < 10:
        return 0.0
    m = np.interp(idx, idx[ok], m[ok])
    n = min(len(m), len(env))
    m, e = _z(m[:n]), _z(env[:n])
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        a, b = m[max(0, lag): n + min(0, lag)], e[max(0, -lag): n + min(0, -lag)]
        if len(a) > 8:
            best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


def sync_to_p_fake(r: float) -> float:
    """Weak correlation -> likely dubbed/synthesised audio. Calibrated downstream."""
    return float(1 / (1 + np.exp(6 * (r - 0.25))))


def analyse(path, wav, sr: int = 16000) -> dict:
    mouth, fps, ratio, method = mouth_series(path)
    if wav is None or ratio < 0.5 or len(mouth) < 20:
        return {"applicable": False, "reason": "Not a talking-head clip with audio (no stable face or no audio track)."}
    env = envelope(wav, sr, fps, len(mouth))
    r = best_corr(mouth, env)
    ref = np.interp(np.arange(len(mouth)), np.arange(len(mouth))[~np.isnan(mouth)], mouth[~np.isnan(mouth)])
    return {"applicable": True, "sync": r, "p_fake": sync_to_p_fake(r), "method": method,
            "series": {"t": [round(i / fps, 3) for i in range(len(mouth))][::2],
                       "mouth": _z(ref).round(3).tolist()[::2], "audio": _z(env).round(3).tolist()[::2]}}


def swap_test(mouth: np.ndarray, own_env: np.ndarray, other_env: np.ndarray) -> tuple[float, float]:
    """Self-generated negative: correlation with own audio vs another clip's audio."""
    return best_corr(mouth, own_env), best_corr(mouth, other_env)
