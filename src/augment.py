"""Simulate WhatsApp/Instagram-style degradation: resize, blur, JPEG re-encode."""
import io
import random

from PIL import Image, ImageFilter


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_cycle(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(8, int(w * scale)), max(8, int(h * scale)))).resize((w, h))


def random_degrade(img: Image.Image, p: float = 0.7) -> Image.Image:
    if random.random() > p:
        return img
    if random.random() < 0.5:
        img = resize_cycle(img, random.uniform(0.4, 0.9))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    return jpeg(img, random.randint(25, 90))
