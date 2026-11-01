"""Throughput benchmark: files/minute against a running API (change WORKERS / replicas and re-run)."""
import io
import random
import sys
import time

import requests
from PIL import Image

base, n = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"), int(sys.argv[2] if len(sys.argv) > 2 else 40)
files = []
for i in range(n):
    im = Image.effect_noise((256, 256), random.randint(20, 90)).convert("RGB")
    b = io.BytesIO(); im.save(b, "JPEG"); files.append(("files", (f"bench{i}.jpg", b.getvalue(), "image/jpeg")))
t0 = time.time()
ids = requests.post(f"{base}/api/analyze", files=files).json()["jobs"]
pending = set(ids)
while pending:
    pending = {j for j in pending if requests.get(f"{base}/api/jobs/{j}").json()["status"] in ("queued",)}
    time.sleep(0.5)
dt = time.time() - t0
print(f"{n} files in {dt:.1f}s -> {n / dt * 60:.0f} files/min")
