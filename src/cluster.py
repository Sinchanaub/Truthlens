"""Perceptual-hash (dHash) clustering of near-duplicate uploads -> campaign families."""
import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 8) -> str:
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def find_family(h: str, existing: list[tuple[str, str]], max_dist: int = 10) -> str | None:
    """existing: [(job_id, hash)] oldest first. Returns the family root job id or None."""
    for job_id, h2 in existing:
        if hamming(h, h2) <= max_dist:
            return job_id
    return None
