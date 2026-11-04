"""FastAPI app: upload (single/batch), job polling, review queue, stats."""
import csv
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, media, monitor, worker

UPLOADS = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))


def _warm():
    """Load models in the background at boot so the first upload is fast and load errors show in /api/health."""
    from . import audio_model, image_model
    media.face_detector_name()
    image_model._load()
    audio_model._load()


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("WARMUP", "1") == "1":
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="TruthLens", version="1.0", lifespan=lifespan)


@app.get("/api/health")
def health(deep: bool = False):
    """deep=true forces model loading now and reports why anything failed (open this on the deployed site)."""
    import cv2

    from . import audio_model, image_model
    if deep:
        image_model._load()
        audio_model._load()
    return {"ok": True, "queue": "redis" if os.getenv("REDIS_URL") else "in-process",
            "image_weights": image_model.WEIGHTS.exists(), "audio_head": audio_model.HEAD.exists(),
            "opencv": cv2.__version__, "face_detector": media.face_detector_name(),
            "image_model": image_model.status(), "audio_model": audio_model.status()}


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), caption: str = Form("")):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    ids = []
    for f in files:
        name = Path(f.filename or "upload").name
        if not media.kind(name):
            raise HTTPException(415, f"Unsupported file type: {name}")
        data = await f.read()
        if len(data) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"{name} exceeds {MAX_MB} MB")
        jid = uuid.uuid4().hex[:10]
        path = UPLOADS / f"{jid}_{name}"
        path.write_bytes(data)
        db.create_job(jid, name, caption)
        worker.submit(jid, str(path.resolve()), caption)
        ids.append(jid)
    return {"jobs": ids}


@app.get("/api/jobs")
def jobs(limit: int = 50):
    return [{k: v for k, v in j.items() if k != "report"} | {
        "verdict": (j["report"] or {}).get("fusion", {}).get("verdict"),
        "risk": (j["report"] or {}).get("risk", {}).get("score")} for j in db.list_jobs(limit)]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@app.get("/api/review")
def review_queue():
    return db.review_queue()


class Review(BaseModel):
    label: str
    note: str = ""


@app.post("/api/review/{job_id}")
def review(job_id: str, body: Review):
    if body.label not in ("real", "fake"):
        raise HTTPException(422, "label must be 'real' or 'fake'")
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    db.add_review(job_id, body.label, body.note)
    out = Path("data/review_labels.csv")  # DVC-tracked; feeds retraining
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["job_id", "filename", "label", "note"])
        w.writerow([job_id, j["filename"], body.label, body.note])
    return {"ok": True}


@app.get("/api/stats")
def stats():
    js = db.list_jobs(500)
    done = [j for j in js if j["status"] == "done"]
    counts = {"Real": 0, "Fake": 0, "Uncertain": 0}
    for j in done:
        counts[j["report"]["fusion"]["verdict"]] += 1
    s = db.scores()
    value = monitor.psi(s[: max(5, len(s) // 2)], s[len(s) // 2:]) if len(s) >= 10 else 0.0
    rob = Path("reports/robustness.json")
    import json
    return {"total": len(js), "done": len(done), "verdicts": counts,
            "families": len({j["family"] for j in done if j["family"] and db.family_size(j["family"]) > 1}),
            "avg_seconds": round(sum(j["report"]["seconds"] for j in done) / len(done), 2) if done else None,
            "drift": {"psi": round(value, 3), "status": monitor.drift_status(value)},
            "robustness": json.loads(rob.read_text()) if rob.exists() else None}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "static", html=True), name="ui")
