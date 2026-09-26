"""Шумодав (SCUNet) и увеличение разрешения (Real-ESRGAN) локально на видеокарте.

torch и spandrel необязательны (install_ai.bat), импортируются лениво. spandrel сам собирает
архитектуру по файлу весов. Веса скачиваются один раз с официальных страниц релизов в models/.
Обработка тайлами с перекрытием, чтобы полный кадр 24 Мп помещался в 8 ГБ видеопамяти.
Модуль без Qt: вызывается и из окна, и из процессов экспорта (у каждого процесса свой кэш моделей).
"""
from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path

import numpy as np

__all__ = ["GPU_LOCK", "MODELS", "available", "denoise", "iso_of", "iso_strength", "upscale"]

MODELS = {
    "denoise": ("scunet_color_real_psnr.pth",
                "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"),
    "x2": ("RealESRGAN_x2plus.pth",
           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    "x4": ("RealESRGAN_x4plus.pth",
           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_models: dict = {}
# Один замок на всё, что трогает torch (здесь, в segment.py и scene.py): импорт torch/transformers/open_clip
# из двух потоков сразу взаимно блокирует импорт модулей, и программа висит навсегда.
GPU_LOCK = threading.RLock()


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "spandrel"))


def iso_of(path) -> int | None:
    """ISO снимка из EXIF (ARW/DNG/JPEG); None, если прочитать не удалось."""
    try:
        import exifread
    except ImportError:  # старое окружение без exifread: просто без ISO
        return None
    try:
        with open(path, "rb") as fh:  # open() в Python понимает кириллицу в пути
            tags = exifread.process_file(io.BytesIO(fh.read(256 * 1024)), details=False)
        t = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF PhotographicSensitivity")
        return int(str(t).split(",")[0].strip("[] ")) if t else None
    except (OSError, ValueError, KeyError):
        return None


def iso_strength(iso: int | None) -> int:
    """Сила шумодава по ISO: до 1600 шум обычно не мешает."""
    if not iso or iso < 1600:
        return 0
    return int(np.interp(np.log2(iso), np.log2([1600, 3200, 6400, 12800]), [30, 50, 70, 85]))


def _model(kind: str):
    if kind not in _models:
        import torch
        from spandrel import ModelLoader

        name, url = MODELS[kind]
        path = MODELS_DIR / name
        if not path.exists():
            MODELS_DIR.mkdir(exist_ok=True)
            tmp = path.with_suffix(".part")
            torch.hub.download_url_to_file(url, str(tmp), progress=False)
            tmp.replace(path)
        desc = ModelLoader().load_from_file(str(path))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        desc = desc.to(dev).eval()
        half = dev == "cuda" and desc.supports_half
        if half:
            desc = desc.half()
        _models[kind] = (desc, dev, half)
    return _models[kind]


def _run_tiled(img: np.ndarray, kind: str, tile: int = 512, pad: int = 32) -> np.ndarray:
    """Сеть по тайлам: каждый тайл берётся с полями pad, в результат идёт только его середина —
    швов нет. img float32 RGB 0..1 → результат в масштабе модели."""
    import torch

    desc, dev, half = _model(kind)
    scale = desc.scale
    mult = max(1, desc.size_requirements.multiple_of)
    h, w = img.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), np.float32)
    src = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            th, tw = min(tile, h - y), min(tile, w - x)
            patch = src[y:y + th + 2 * pad, x:x + tw + 2 * pad]
            ph, pw = patch.shape[:2]
            eh, ew = -ph % mult, -pw % mult  # добиваем до кратности, которую требует сеть
            if eh or ew:
                patch = np.pad(patch, ((0, eh), (0, ew), (0, 0)), mode="reflect")
            t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None].to(dev)
            t = t.half() if half else t
            with torch.no_grad():
                res = desc(t)[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            s = scale
            out[y * s:(y + th) * s, x * s:(x + tw) * s] = res[pad * s:(pad + th) * s, pad * s:(pad + tw) * s]
    return out


def denoise(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Шумодав; strength 0..1 смешивает с исходником (0 — без изменений)."""
    if strength <= 0:
        return img
    src = np.clip(img, 0, 1).astype(np.float32)
    with GPU_LOCK:
        clean = _run_tiled(src, "denoise")
    return src + (clean - src) * min(strength, 1.0)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Увеличение ×2 или ×4 (Real-ESRGAN)."""
    if factor not in (2, 4):
        return img
    with GPU_LOCK:
        return _run_tiled(np.clip(img, 0, 1).astype(np.float32), f"x{factor}")
