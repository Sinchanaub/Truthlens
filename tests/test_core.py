import io
import os
import tempfile
import time

import numpy as np
from PIL import Image

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from src import augment, calibrate, cluster, fusion, monitor, risk, video_checks as vc  # noqa: E402
from src.serve import app  # noqa: E402


def test_abstention_band():
    assert calibrate.decide(0.9) == "Fake" and calibrate.decide(0.1) == "Real" and calibrate.decide(0.5) == "Uncertain"


def test_temperature_softens():
    assert calibrate.temperature_scale(0.9, 3.0) < 0.9


def test_fit_temperature_overconfident():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    z = np.where(rng.random(500) < 0.7, 1, -1) * (2 * y - 1) * 8.0  # 70% right but huge logits
    assert calibrate.fit_temperature(z, y) > 1.5


def test_fusion_conflict_forces_uncertain():
    r = fusion.fuse({"frames": 0.05, "audio": 0.95})
    assert r["conflicts"] and r["verdict"] == "Uncertain"


def test_cluster_near_duplicates():
    a = Image.effect_noise((128, 128), 60).convert("RGB")
    h1, h2 = cluster.dhash(a), cluster.dhash(augment.jpeg(a, 40))
    assert cluster.hamming(h1, h2) <= 12


def test_lipsync_correlation_and_swap():
    rng = np.random.default_rng(1)
    env = np.convolve(rng.random(200), np.ones(5) / 5, "same")
    mouth = env + 0.1 * rng.random(200)
    other = np.convolve(rng.random(200), np.ones(5) / 5, "same")
    own, swapped = vc.swap_test(mouth, env, other)
    assert own > 0.8 and swapped < own - 0.3
    assert vc.sync_to_p_fake(own) < 0.3 < vc.sync_to_p_fake(swapped) or swapped > 0.25


def test_risk_caption_and_provenance():
    assert risk.caption_signal("BREAKING leaked video, share now") > 0
    r = risk.risk_score(0.9, True, {"score": 0.4, "signals": []}, "")
    assert r["level"] in ("Medium", "High")


def test_psi_detects_shift():
    assert monitor.psi(np.random.beta(2, 8, 300), np.random.beta(8, 2, 300)) > 0.25


def test_api_image_roundtrip():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    buf = io.BytesIO(); Image.effect_noise((200, 200), 50).convert("RGB").save(buf, "JPEG")
    jid = c.post("/api/analyze", files=[("files", ("x.jpg", buf.getvalue(), "image/jpeg"))]).json()["jobs"][0]
    for _ in range(60):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "queued":
            break
        time.sleep(0.5)
    assert j["status"] == "done" and j["report"]["fusion"]["verdict"] in ("Real", "Fake", "Uncertain")
    assert c.post("/api/analyze", files=[("files", ("x.txt", b"hi", "text/plain"))]).status_code == 415


# ---- regression tests for the OpenCV-5 / silent-model-failure bugs ----
def test_face_crop_survives_missing_cascade(monkeypatch):
    """OpenCV 5 has no cv2.CascadeClassifier: analysis must degrade, not crash."""
    import cv2

    from src import media
    monkeypatch.delattr(cv2, "CascadeClassifier", raising=False)
    monkeypatch.setitem(media._DET, "d", media._pick_detector())
    img = Image.effect_noise((200, 200), 50).convert("RGB")
    out, found = media.face_crop(img)
    assert out.size == (200, 200) and found is False


def test_pipeline_image_without_face_detector(monkeypatch, tmp_path):
    from src import media, pipeline
    monkeypatch.setitem(media._DET, "d", ("none", None))
    p = tmp_path / "a.jpg"
    Image.effect_noise((200, 200), 50).convert("RGB").save(p)
    rep = pipeline.analyse(str(p))
    assert rep["fusion"]["verdict"] in ("Real", "Fake", "Uncertain")
    assert any("No face detector" in n for n in rep["notes"])


def test_audio_load_failure_is_reported_not_silent(monkeypatch, tmp_path):
    from src import audio_model, pipeline
    head = tmp_path / "audio_head.joblib"
    head.write_bytes(b"not a real model")
    monkeypatch.setattr(audio_model, "HEAD", head)
    audio_model._STATE.clear()
    r = audio_model.predict(np.zeros(32000, dtype=np.float32))
    assert r["trained"] is False and r["error"]
    raw, ev, notes, untrained = {}, {}, [], []
    pipeline._audio(np.zeros(32000, dtype=np.float32), raw, ev, notes, untrained)
    assert not raw and not untrained and any("could not run" in n for n in notes)
    audio_model._STATE.clear()


def test_health_reports_model_status():
    h = TestClient(app).get("/api/health").json()
    assert {"opencv", "face_detector", "image_model", "audio_model"} <= set(h)


def test_audio_train_records_accuracy_and_temperature(monkeypatch, tmp_path):
    """Smoke-test train() with a stub encoder (no HuggingFace download)."""
    import json
    import sys
    import types

    from src import audio_model, media
    fake = types.SimpleNamespace(
        AutoFeatureExtractor=types.SimpleNamespace(from_pretrained=lambda n: None),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda n: types.SimpleNamespace(eval=lambda: None)))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    rng = np.random.default_rng(0)
    for cls in ("real", "fake"):
        (tmp_path / "data" / cls).mkdir(parents=True)
        for i in range(20):
            (tmp_path / "data" / cls / f"{i}.wav").write_bytes(bytes([i]))
    monkeypatch.setattr(media, "load_audio", lambda f: np.zeros(20000, dtype=np.float32))
    monkeypatch.setattr(audio_model, "embed", lambda w: rng.normal(size=32) + (1.5 if rng.random() < .5 else -1.5))
    monkeypatch.setattr(audio_model, "HEAD", tmp_path / "models" / "audio_head.joblib")
    monkeypatch.chdir(tmp_path)
    audio_model.train(str(tmp_path / "data"))
    assert (tmp_path / "models" / "audio_head.joblib").exists()
    assert "audio" in json.loads((tmp_path / "models" / "temperatures.json").read_text())
    assert "val_acc" in json.loads((tmp_path / "reports" / "audio_eval.json").read_text())
    audio_model._STATE.clear()
