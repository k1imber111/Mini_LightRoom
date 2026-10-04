"""ИИ-маски: семантическая сегментация SegFormer (ADE20K, 150 классов) локально на видеокарте.

torch и transformers необязательны (install_ai.bat), поэтому импортируются только внутри
Segmenter. Веса (~110 МБ) скачиваются один раз в папку models/.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

from .enhance import GPU_LOCK

__all__ = ["CATEGORIES", "Segmenter", "available", "get_segmenter", "loaded", "refine"]

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Без этого transformers после загрузки весов .bin запускает свой поток, который ходит в интернет за
# конвертацией в safetensors и трогает torch мимо GPU_LOCK: с шумодавом подряд программа висла или падала.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# Категория маски → названия классов ADE20K (первое слово метки модели).
CATEGORIES = {
    "sky": ("Небо", ("sky",)),
    "people": ("Люди", ("person",)),
    "greenery": ("Зелень", ("tree", "grass", "plant", "field", "palm", "flower")),
    "water": ("Вода", ("water", "sea", "river", "lake", "swimming")),
    "buildings": ("Здания", ("building", "house", "skyscraper", "tower", "bridge", "wall")),
    # Предметы: птицы и звери, корабли, техника — то, что чаще всего главное в кадре (авто-кадр, маска «Предметы»)
    "objects": ("Предметы", ("animal", "boat", "ship", "airplane", "car", "bus", "truck", "bicycle", "minibike",
                             "sculpture", "fountain", "tower", "flower")),
}


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


_segmenter = None


def loaded() -> bool:
    return _segmenter is not None


def get_segmenter(models_dir: Path) -> Segmenter:
    """Одна модель на программу: загрузка под общим замком, повторные вызовы — готовая."""
    global _segmenter
    with GPU_LOCK:
        if _segmenter is None:
            _segmenter = Segmenter(models_dir)
    return _segmenter


def refine(prob: np.ndarray, guide_rgb: np.ndarray, radius: int | None = None, eps: float = 1e-3) -> np.ndarray:
    """Guided filter (He et al.): края маски прилипают к краям кадра — без ореолов вокруг деревьев и людей.
    prob и guide одного размера, результат float32 0..1."""
    g = cv2.cvtColor(np.clip(guide_rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    p = prob.astype(np.float32)
    r = radius or max(2, round(max(g.shape) / 250))
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mg, mp = box(g), box(p)
    a = (box(g * p) - mg * mp) / (box(g * g) - mg * mg + eps)
    b = mp - a * mg
    return np.clip(box(a) * g + box(b), 0, 1)


class Segmenter:
    def __init__(self, models_dir: Path):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:  # веса уже в models/ — без обращений к сети (прокси может отвечать минутами)
            self.proc, self.model = self._load(models_dir, True)
        except OSError:  # первый запуск: скачать
            self.proc, self.model = self._load(models_dir, False)
        self.model = self.model.to(self.device).eval()
        labels = {int(i): n.split(";")[0].split(",")[0].strip().lower()
                  for i, n in self.model.config.id2label.items()}
        self.ids = {cat: [i for i, n in labels.items() if n in names] for cat, (_, names) in CATEGORIES.items()}

    @staticmethod
    def _load(models_dir: Path, offline: bool):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        kw = {"cache_dir": str(models_dir), "local_files_only": offline}
        return (SegformerImageProcessor.from_pretrained(MODEL, **kw),
                SegformerForSemanticSegmentation.from_pretrained(MODEL, **kw))

    def masks(self, rgb: np.ndarray, cats: list[str]) -> dict[str, np.ndarray]:
        """rgb float 0..1 (превью кадра) → {категория: мягкая маска 0..1 того же размера, края уточнены}."""
        torch = self.torch
        h, w = rgb.shape[:2]
        u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        with GPU_LOCK, torch.no_grad():
            inputs = self.proc(images=u8, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            prob = logits.softmax(dim=1)[0]
            out = {}
            for cat in cats:
                m = prob[self.ids[cat]].sum(0).float().cpu().numpy() if self.ids[cat] else np.zeros((h, w), np.float32)
                out[cat] = refine(m, rgb)
        return out
