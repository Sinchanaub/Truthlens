"""Temperature scaling + abstention (Real / Fake / Uncertain)."""
import json
import os
from pathlib import Path

import numpy as np

LO, HI = 0.35, 0.65  # abstention band
_TPATH = Path(os.getenv("MODEL_DIR", "models")) / "temperatures.json"


def temperature_scale(p: float, T: float = 1.0) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(1 / (1 + np.exp(-np.log(p / (1 - p)) / T)))


def fit_temperature(logits, labels) -> float:
    """Grid-search T minimising NLL on a validation set."""
    z, y = np.asarray(logits, float), np.asarray(labels, float)
    best, bt = 1e9, 1.0
    for T in np.linspace(0.5, 5, 46):
        p = np.clip(1 / (1 + np.exp(-z / T)), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, bt = nll, float(T)
    return bt


def get_T(name: str) -> float:
    try:
        return float(json.loads(_TPATH.read_text()).get(name, 1.0))
    except Exception:
        return 1.0


def decide(p: float, lo: float = LO, hi: float = HI) -> str:
    return "Fake" if p >= hi else "Real" if p <= lo else "Uncertain"


def coverage_accuracy(probs, labels, lo=LO, hi=HI):
    """Coverage (share not abstained) and accuracy on the covered subset."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    keep = (probs >= hi) | (probs <= lo)
    if not keep.any():
        return 0.0, None
    return float(keep.mean()), float(((probs[keep] >= hi) == labels[keep].astype(bool)).mean())
