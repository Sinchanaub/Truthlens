"""Lip-sync detection rate: original clips vs audio-swapped clips (self-generated negatives).
Usage: python scripts/lipsync_eval.py data/clips/*.mp4"""
import sys

from src import media, video_checks as vc

clips = [(p, *vc.mouth_series(p)[:3]) for p in sys.argv[1:]]
wavs = {p: media.load_audio(p) for p, *_ in clips}
pos = neg = tot = 0
for i, (p, mouth, fps, ratio) in enumerate(clips):
    if ratio < .5 or wavs[p] is None:
        continue
    other = wavs[clips[(i + 1) % len(clips)][0]]
    own, swp = vc.swap_test(mouth, vc.envelope(wavs[p], 16000, fps, len(mouth)), vc.envelope(other, 16000, fps, len(mouth)))
    tot += 1; pos += own > 0.25; neg += swp <= 0.25
print(f"clips={tot}  original judged in-sync={pos / max(tot, 1):.0%}  swapped judged out-of-sync={neg / max(tot, 1):.0%}")
