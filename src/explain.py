"""Explanations: Grad-CAM (trained model) or Error Level Analysis (fallback)."""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageChops


def to_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> str:
    base = np.asarray(img.convert("RGB").resize((320, int(320 * img.height / img.width))))
    h = cv2.resize(heat.astype(np.float32), (base.shape[1], base.shape[0]))
    h = (255 * (h - h.min()) / (np.ptp(h) + 1e-9)).astype(np.uint8)
    col = cv2.cvtColor(cv2.applyColorMap(h, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return to_b64((base * (1 - alpha) + col * alpha).astype(np.uint8))


def ela_heat(img: Image.Image, q: int = 90) -> np.ndarray:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    diff = ImageChops.difference(img.convert("RGB"), Image.open(buf))
    return np.asarray(diff).astype(np.float32).mean(axis=2)


def gradcam(model, x, img: Image.Image) -> str:
    """Grad-CAM on the last conv block of EfficientNet-B0. x: 1x3xHxW tensor."""
    import torch
    acts, grads = {}, {}
    layer = model.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    model(x.requires_grad_(True)).sum().backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    return overlay(img, cam)
