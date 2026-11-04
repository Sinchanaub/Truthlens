"""Orchestrates analysers -> calibration -> fusion -> risk -> evidence report."""
import time

from PIL import Image

from . import audio_model, cluster, db, image_model, media, risk, video_checks
from .calibrate import get_T, temperature_scale
from .fusion import fuse


def _cal(name, p):
    return None if p is None else temperature_scale(p, get_T(name))


def _audio(wav, raw, evidence, notes, untrained):
    a = audio_model.predict(wav)
    if a["p_fake"] is None:
        notes.append(a["method"])
        return
    evidence["audio_method"] = a["method"]
    if a.get("trained"):
        raw["audio"] = a["p_fake"]
    elif a.get("error"):
        notes.append(f"Audio model could not run, so audio is not part of this verdict ({a['error']}).")
    else:
        untrained.append("audio")


def analyse(path, caption: str = "") -> dict:
    t0, k = time.time(), media.kind(path)
    raw, evidence, notes, probs, phash, untrained = {}, {}, [], {}, None, []

    if k == "image":
        img = media.open_image(path)
        phash = cluster.dhash(img)
        face, found = media.face_crop(img)
        if not found:
            notes.append("No face detector available on this server; the whole image was analysed. The model was trained on faces, so treat this result with caution."
                         if media.face_detector_name() == "none" else
                         "No face detected. The model was trained on faces, so treat this result with extra caution.")
        r = image_model.predict(face)
        if r.get("error"):
            notes.append(f"Image model could not run ({r['error']}).")
        evidence["heatmap"] = r["heatmap"]; evidence["image_method"] = r["method"]
        if r["trained"]:
            raw["image"] = r["p_fake"]
        else:
            untrained.append("image")

    elif k == "video":
        frames = media.sample_frames(path, 8)
        wav = media.load_audio(path)
        v = video_checks.analyse(path, wav)
        if v["applicable"]:
            raw["lipsync"] = v["p_fake"]
            evidence["sync"] = {"r": round(v["sync"], 3), "method": v["method"], **v["series"]}
        else:
            notes.append(v["reason"] + " Falling back to frame model.")
        if frames:
            crops = [media.face_crop(Image.fromarray(f), force=True)[0] for f in frames]
            rs = [image_model.predict(c, heatmap=False) for c in crops]
            if rs[0]["trained"]:
                raw["frames"] = float(sum(x["p_fake"] for x in rs) / len(rs))
            else:
                untrained.append("frames")
            worst = max(range(len(rs)), key=lambda i: rs[i]["p_fake"])
            evidence["heatmap"] = image_model.predict(crops[worst])["heatmap"]
            evidence["image_method"] = rs[0]["method"]
            phash = cluster.dhash(Image.fromarray(frames[len(frames) // 2]))
        if wav is not None:
            _audio(wav, raw, evidence, notes, untrained)

    elif k == "audio":
        wav = media.load_audio(path)
        if wav is None:
            notes.append("Could not decode the audio (ffmpeg missing or unreadable file).")
        _audio(wav, raw, evidence, notes, untrained)
    else:
        raise ValueError("Unsupported file type")

    probs = {n: _cal(n, p) for n, p in raw.items()}
    fusion = fuse(probs)
    if untrained:
        notes.append("Excluded from the verdict (no trained weights): " + ", ".join(untrained) + ".")
    prov = risk.provenance(path)
    rk = risk.risk_score(fusion["p_fake"], bool(fusion["conflicts"]), prov, caption)
    return {"kind": k, "fusion": fusion, "risk": rk, "evidence": evidence, "notes": notes,
            "phash": phash, "seconds": round(time.time() - t0, 2)}


def process(job_id: str, path: str, caption: str = ""):
    """Worker entry point (called by RQ or the in-process pool)."""
    try:
        rep = analyse(path, caption)
        fam = None
        if rep["phash"]:
            fam = cluster.find_family(rep["phash"], [h for h in db.hashes() if h[0] != job_id]) or job_id
            rep["family"] = {"id": fam, "size": db.family_size(fam) + 1}
        db.finish_job(job_id, "done", rep, rep["phash"], fam)
    except Exception as e:  # noqa: BLE001
        db.finish_job(job_id, "error", {"error": str(e)})
