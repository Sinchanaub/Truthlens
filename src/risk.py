"""Disinformation risk = synthetic confidence + modality conflict + provenance + caption.
A triage signal for human reviewers, NOT a truth verdict."""
import re
from pathlib import Path

from PIL import Image

from .media import kind

NEWSY = re.compile(r"\b(breaking|urgent|leaked|exposed|shocking|share (this|now)|forward|confirmed|viral|they don'?t want)\b", re.I)


def provenance(path) -> dict:
    """Returns {'score': 0..1 (higher = less trustworthy), 'signals': [...]}."""
    signals, score = [], 0.0
    head = Path(path).read_bytes()[:2_000_000]
    if b"c2pa" in head or b"jumb" in head:
        return {"score": 0.0, "signals": ["C2PA/content credentials marker present (not cryptographically verified)"]}
    if kind(path) == "image":
        try:
            im = Image.open(path)
            if not im.getexif():
                signals.append("No EXIF metadata"); score += 0.4
            q = getattr(im, "quantization", None)
            if q:
                avg = sum(q[0]) / len(q[0])
                if avg > 20:
                    signals.append("Heavy JPEG re-encoding"); score += 0.4
        except Exception:
            pass
    else:
        signals.append("No provenance credentials found"); score += 0.3
    return {"score": min(score, 1.0), "signals": signals}


def caption_signal(caption: str) -> float:
    return min(1.0, 0.5 * len(NEWSY.findall(caption or "")))


def risk_score(p_fake, conflict: bool, prov: dict, caption: str) -> dict:
    synth = 0.5 if p_fake is None else p_fake
    s = 0.5 * synth + 0.2 * float(conflict) + 0.2 * prov["score"] + 0.1 * caption_signal(caption)
    return {"score": round(min(s, 1.0), 3), "level": "High" if s >= 0.6 else "Medium" if s >= 0.35 else "Low",
            "signals": prov["signals"] + (["Cross-modal conflict"] if conflict else [])
            + (["News-style caption language"] if caption_signal(caption) else [])}
