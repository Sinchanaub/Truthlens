"""EfficientNet-B0 image detector. Falls back to an abstaining ELA-only analyser
when trained weights (models/image_effnet.pt) are not present."""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import explain

WEIGHTS = Path(os.getenv("MODEL_DIR", "models")) / "image_effnet.pt"
_MODEL = None
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def build():
    import torch
    import torchvision
    m = torchvision.models.efficientnet_b0(weights="DEFAULT" if not WEIGHTS.exists() else None)
    m.classifier[1] = torch.nn.Linear(1280, 1)  # logit for P(fake)
    return m


def _load():
    global _MODEL
    if _MODEL is None:
        try:
            import torch
            if WEIGHTS.exists():
                m = build()
                m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
                _MODEL = m.eval()
            else:
                _MODEL = False
        except Exception:
            _MODEL = False
    return _MODEL or None


def to_tensor(img: Image.Image):
    import torch
    a = (np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255 - MEAN) / STD
    return torch.tensor(a, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def predict(img: Image.Image, heatmap: bool = True) -> dict:
    m = _load()
    if m is None:
        return {"p_fake": 0.5, "logit": 0.0, "trained": False, "method": "ELA only (no trained weights, abstains)",
                "heatmap": explain.overlay(img, explain.ela_heat(img)) if heatmap else None}
    import torch
    x = to_tensor(img)
    with torch.no_grad():
        logit = float(m(x).item())
    return {"p_fake": float(1 / (1 + np.exp(-logit))), "logit": logit, "trained": True,
            "method": "EfficientNet-B0 + Grad-CAM", "heatmap": explain.gradcam(m, to_tensor(img), img) if heatmap else None}


def train(data_dir="data/images", epochs=3, bs=32, lr=1e-4):
    """data_dir/{real,fake}/*.jpg. Trains with compression/blur augmentation, logs to MLflow,
    fits temperature, writes a JPEG-quality robustness curve to reports/robustness.json."""
    import json
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import ImageFolder
    from . import augment, calibrate
    tf_train = T.Compose([T.Lambda(augment.random_degrade), T.Resize((224, 224)), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = ImageFolder(data_dir, transform=tf_train)
    fake_idx = ds.class_to_idx["fake"]
    n_val = max(1, len(ds) // 5)
    tr, va = random_split(ds, [len(ds) - n_val, n_val])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, loss_fn = build().to(dev), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    try:
        import mlflow
        mlflow.start_run(); mlflow.log_params({"epochs": epochs, "lr": lr, "bs": bs, "augment": True})
    except Exception:
        mlflow = None
    for ep in range(epochs):
        m.train()
        for x, y in DataLoader(tr, bs, shuffle=True, num_workers=2):
            x, y = x.to(dev), (y == fake_idx).float().unsqueeze(1).to(dev)
            opt.zero_grad(); loss = loss_fn(m(x), y); loss.backward(); opt.step()
        if mlflow:
            mlflow.log_metric("train_loss", float(loss), step=ep)

    def evaluate(quality=None):
        m.eval(); Z, Y = [], []
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
        with torch.no_grad():
            for i in va.indices:
                path, y = ds.samples[i]
                img = Image.open(path).convert("RGB")
                img = augment.jpeg(img, quality) if quality else img
                Z.append(float(m(tf(img).unsqueeze(0).to(dev)))); Y.append(float(y == fake_idx))
        return np.array(Z), np.array(Y)

    Z, Y = evaluate()
    acc = float(((Z > 0) == Y.astype(bool)).mean())
    curve = {q: float(((evaluate(q)[0] > 0) == Y.astype(bool)).mean()) for q in (95, 75, 50, 30)}
    Path("reports").mkdir(exist_ok=True); WEIGHTS.parent.mkdir(exist_ok=True)
    Path("reports/robustness.json").write_text(json.dumps({"clean": acc, "jpeg": curve}))
    tp = WEIGHTS.parent / "temperatures.json"
    temps = json.loads(tp.read_text()) if tp.exists() else {}
    temps["image"] = temps["frames"] = calibrate.fit_temperature(Z, Y)
    tp.write_text(json.dumps(temps))
    torch.save(m.cpu().state_dict(), WEIGHTS)
    if mlflow:
        mlflow.log_metric("val_acc", acc); mlflow.log_metrics({f"val_acc_jpeg{q}": a for q, a in curve.items()}); mlflow.end_run()
    print("val acc", acc, "jpeg robustness", curve)


if __name__ == "__main__":
    train()
