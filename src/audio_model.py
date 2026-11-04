"""Audio deepfake detector: frozen wav2vec2 embeddings + logistic-regression head.
Without a trained head (models/audio_head.joblib) it abstains (p=0.5)."""
import logging
import os
import time
from pathlib import Path

import numpy as np

HEAD = Path(os.getenv("MODEL_DIR", "models")) / "audio_head.joblib"
_STATE = {}
_RETRY_AFTER = 30  # seconds before retrying a failed load (e.g. HuggingFace download hiccup)
log = logging.getLogger("truthlens.audio")


def _load():
    """Returns the trained head, or None. Missing head file is permanent; a load *failure* is recorded
    in _STATE["error"] (shown in the report and /api/health) and retried instead of being cached forever."""
    if _STATE.get("head") is not None:
        return _STATE["head"]
    if not HEAD.exists():
        _STATE["error"] = None
        return None
    if _STATE.get("error") and time.time() - _STATE.get("failed_at", 0) < _RETRY_AFTER:
        return None
    try:
        import joblib
        from transformers import AutoFeatureExtractor, AutoModel
        name = os.getenv("AUDIO_ENCODER", "facebook/wav2vec2-base")
        head = joblib.load(HEAD)
        _STATE["fx"] = AutoFeatureExtractor.from_pretrained(name)
        _STATE["enc"] = AutoModel.from_pretrained(name).eval()
        _STATE.update(head=head, error=None)
    except Exception as e:  # noqa: BLE001
        _STATE.update(head=None, error=f"{type(e).__name__}: {e}"[:300], failed_at=time.time())
        log.exception("audio model failed to load")
    return _STATE.get("head")


def status() -> dict:
    return {"head_file": HEAD.exists(), "loaded": _STATE.get("head") is not None, "error": _STATE.get("error")}


def embed(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    import torch
    fx, enc = _STATE["fx"], _STATE["enc"]
    wav = wav[: sr * 20]
    with torch.no_grad():
        h = enc(**fx(wav, sampling_rate=sr, return_tensors="pt")).last_hidden_state
    return torch.cat([h.mean(1), h.std(1)], dim=1)[0].numpy()


def predict(wav: np.ndarray) -> dict:
    if wav is None or len(wav) < 16000:
        return {"p_fake": None, "method": "audio too short or missing"}
    head = _load()
    if head is None:
        if _STATE.get("error"):
            return {"p_fake": 0.5, "trained": False, "error": _STATE["error"],
                    "method": "audio model failed to load (abstains)"}
        return {"p_fake": 0.5, "trained": False, "method": "wav2vec2 head not trained (abstains)"}
    p = float(head.predict_proba(embed(wav)[None])[0, 1])
    return {"p_fake": p, "trained": True, "method": "wav2vec2 embeddings + logistic regression"}


def train(data_dir="data/audio"):
    """data_dir/{real,fake}/*.wav|mp3 -> models/audio_head.joblib. Hindi/Telugu test clips: data/audio_in/."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from .media import load_audio
    _STATE.clear(); HEAD.unlink(missing_ok=True)
    from transformers import AutoFeatureExtractor, AutoModel
    name = os.getenv("AUDIO_ENCODER", "facebook/wav2vec2-base")
    _STATE.update(fx=AutoFeatureExtractor.from_pretrained(name), enc=AutoModel.from_pretrained(name).eval())
    X, y = [], []
    for label, cls in enumerate(("real", "fake")):
        for f in sorted(Path(data_dir, cls).glob("*")):
            w = load_audio(f)
            if w is not None and len(w) > 16000:
                X.append(embed(w)); y.append(label)
    Xtr, Xte, ytr, yte = train_test_split(np.array(X), np.array(y), test_size=0.2, stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    z = clf.decision_function(Xteaaa
    acc = float(clf.score(Xte, yte))
    print("val acc", acc)
    # Record the numbers a reviewer will ask for, and calibrate the audio score (it had no temperature before).
    import json

    from . import calibrate
    T = calibrate.fit_temperature(z, yte)
    cov, cov_acc = calibrate.coverage_accuracy(1 / (1 + np.exp(-z / T)), yte)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/audio_eval.json").write_text(json.dumps(
        {"val_acc": acc, "n_train": len(ytr), "n_val": len(yte), "temperature": T,
         "coverage": cov, "accuracy_on_covered": cov_acc}))
    tp = HEAD.parent / "temperatures.json"
    temps = json.loads(tp.read_text()) if tp.exists() else {}
    temps["audio"] = T
    HEAD.parent.mkdir(exist_ok=True); tp.write_text(json.dumps(temps))
    joblib.dump(clf, HEAD)


if __name__ == "__main__":
    train()
