"""Распознавание сцены на кадре локально: CLIP ViT-B/32 (open_clip) на видеокарте.

torch и open_clip тяжёлые и необязательные (requirements-ai.txt), поэтому импортируются
только внутри SceneClassifier. Веса (~600 МБ) скачиваются один раз в папку models/.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

__all__ = ["SceneClassifier", "available"]

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows без прав на симлинки — не шуметь

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def available() -> bool:
    """Установлены ли библиотеки ИИ."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "open_clip"))


class SceneClassifier:
    """Zero-shot: сравнивает кадр с текстовыми описаниями каждой сцены."""

    def __init__(self, scenes: list[dict], models_dir: Path):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED, cache_dir=str(models_dir), device=self.device)
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
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([self.preprocess(Image.fromarray(im)) for im in images[i:i + batch]]).to(self.device)
                if self.device == "cuda":
                    x = x.half()
                f = self.model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                prob = (100 * f @ self.text.T).softmax(dim=-1).cpu().numpy()
                out += [(self.ids[int(p.argmax())], float(p.max())) for p in prob]
        return out
