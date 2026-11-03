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
