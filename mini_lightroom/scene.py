"""Распознавание сцены на кадре локально: CLIP ViT-B/32 (open_clip) на видеокарте.

torch и open_clip тяжёлые и необязательные (requirements-ai.txt), поэтому импортируются
только внутри SceneClassifier. Веса (~600 МБ) скачиваются один раз в папку models/.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from .enhance import GPU_LOCK

__all__ = ["SceneClassifier", "available", "get_classifier", "loaded"]

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows без прав на симлинки — не шуметь

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # где лежат веса PRETRAINED


def available() -> bool:
    """Установлены ли библиотеки ИИ."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "open_clip"))


_classifier = None


def loaded() -> bool:
    return _classifier is not None


def get_classifier(scenes: list[dict], models_dir: Path) -> SceneClassifier:
    """Одна модель на программу: загрузка под общим замком, повторные вызовы — готовая."""
    global _classifier
    with GPU_LOCK:
        if _classifier is None:
            _classifier = SceneClassifier(scenes, models_dir)
    return _classifier


class SceneClassifier:
    """Zero-shot: сравнивает кадр с текстовыми описаниями каждой сцены."""

    def __init__(self, scenes: list[dict], models_dir: Path):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from huggingface_hub import try_to_load_from_cache

        # Веса уже в models/ — файлом, без обращений к сети (прокси может отвечать минутами).
        local = try_to_load_from_cache(HF_REPO, "open_clip_model.safetensors", cache_dir=str(models_dir))
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=local if isinstance(local, str) else PRETRAINED, cache_dir=str(models_dir),
            device=self.device)
        self.model = model.eval()
        if self.device == "cuda":
            self.model = self.model.half()
        tokenizer = open_clip.get_tokenizer(MODEL)
        self.ids = [s["id"] for s in scenes]
        with torch.no_grad():
            embs = []
            for s in scenes:  # несколько описаний на сцену — среднее направление устойчивее
                t = self.model.encode_text(tokenizer(s["prompts"]).to(self.device)).float()
                t = t / t.norm(dim=-1, keepdim=True)
                embs.append(t.mean(0))
            e = torch.stack(embs)
            self.text = e / e.norm(dim=-1, keepdim=True)

    def classify(self, images: list[np.ndarray], batch: int = 32) -> list[tuple[str, float]]:
        """uint8 RGB-миниатюры → [(id сцены, уверенность 0..1)]."""
        from PIL import Image

        torch = self.torch
        out: list[tuple[str, float]] = []
        with GPU_LOCK, torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([self.preprocess(Image.fromarray(im)) for im in images[i:i + batch]]).to(self.device)
                if self.device == "cuda":
                    x = x.half()
                f = self.model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                prob = (100 * f @ self.text.T).softmax(dim=-1).cpu().numpy()
                out += [(self.ids[int(p.argmax())], float(p.max())) for p in prob]
        return out
