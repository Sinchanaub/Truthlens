# TruthLens

Forensic, multimodal deepfake and synthetic-media triage for **Neural Storm 2026, FT-02**.
Instead of one opaque score, TruthLens returns an **evidence report**: per-modality verdicts on a calibrated
Real / **Uncertain** / Fake scale, a lip-sync physical check, cross-modal conflicts, a Grad-CAM heatmap,
a disinformation risk score, campaign clustering, and a human review queue.

## Run it

```bash
# Option A: Docker (API + UI + Redis + 2 workers)
docker compose up --build            # open http://localhost:8000
docker compose up --scale worker=4   # more workers

# Option B: local, no Redis (in-process worker pool). Needs ffmpeg on PATH for video/audio.
pip install -r requirements.txt
uvicorn src.serve:app --reload
```

Add `WITH_ML=1` (Docker) or `pip install -r requirements-ml.txt` (local) for torch, wav2vec2, MediaPipe, MLflow, Evidently.

## What works without any training

| Feature | Status |
|---|---|
| Lip-sync check (mouth motion vs audio envelope) | Works out of the box, no training |
| Calibration, abstention, fusion, conflict flags, risk score | Works |
| Campaign clustering (dHash), review queue, drift (PSI), queue + batch | Works |
| Image model (EfficientNet-B0 + Grad-CAM) | **Needs training** (`python -m src.image_model`). Until then it shows an ELA heatmap and **abstains** |
| Audio model (wav2vec2 + logistic regression) | **Needs training** (`python -m src.audio_model`). Until then it abstains |

Untrained models never guess: they report 50% and the verdict becomes "Uncertain". The UI says so.

## Train

```
data/images/{real,fake}/*.jpg     # e.g. subset of 140k Real and Fake Faces (check licence)
data/audio/{real,fake}/*.wav      # e.g. Fake-or-Real / ASVspoof subset
python -m src.image_model         # trains w/ compression+blur aug, fits temperature, writes reports/robustness.json
python -m src.audio_model
```
Runs are logged to MLflow if installed (`docker compose --profile tracking up mlflow`). `dvc repro` rebuilds from `dvc.yaml`.

## Evaluate

```
python scripts/lipsync_eval.py data/clips/*.mp4   # original vs audio-swapped clips
python scripts/benchmark.py http://localhost:8000 40   # files/min; rerun with different worker counts
pytest -q
```

## API

`POST /api/analyze` (multipart `files[]`, `caption`) → `{jobs:[id]}` · `GET /api/jobs/{id}` · `GET /api/review` ·
`POST /api/review/{id}` `{label: real|fake}` · `GET /api/stats` · `GET /api/health`

## Layout

`src/` image_model, audio_model, video_checks, calibrate, fusion, risk, cluster, explain, monitor, pipeline, worker, serve, db ·
`static/index.html` UI · `tests/` · `scripts/` · `dvc.yaml` · `docker-compose.yml` · `.github/workflows/ci.yml` · `MODEL_CARD.md`

## Not implemented yet

Face-identity embedding drift and landmark-jitter checks, C2PA signature verification (only a marker check),
embedding-based clustering (perceptual hash only), automatic retraining trigger (drift status is reported,
retraining is manual), Telegram bot, Hindi/Telugu test-set loader.

Video analysis has been unit-tested on synthetic signals, not yet on real footage. Run `lipsync_eval.py` on your clips and report the numbers honestly.
