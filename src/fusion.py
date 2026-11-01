"""Fuse per-modality calibrated probabilities and surface cross-modal conflicts."""
import numpy as np

from .calibrate import decide

WEIGHTS = {"image": 1.0, "frames": 0.8, "lipsync": 1.2, "audio": 1.0}


def fuse(probs: dict[str, float]) -> dict:
    if not probs:
        return {"p_fake": None, "verdict": "Uncertain", "modalities": {}, "conflicts": [],
                "reason": "No analysable modality."}
    verdicts = {k: decide(p) for k, p in probs.items()}
    w = np.array([WEIGHTS.get(k, 1.0) for k in probs])
    z = np.array([np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) for p in probs.values()])
    p = float(1 / (1 + np.exp(-(w * z).sum() / w.sum())))
    fakes = [k for k, v in verdicts.items() if v == "Fake"]
    reals = [k for k, v in verdicts.items() if v == "Real"]
    conflicts = [f"{', '.join(reals)} look real but {', '.join(fakes)} look synthetic"] if fakes and reals else []
    verdict = "Uncertain" if conflicts else decide(p)
    return {"p_fake": p, "verdict": verdict, "conflicts": conflicts,
            "modalities": {k: {"p_fake": probs[k], "verdict": verdicts[k]} for k in probs}}
