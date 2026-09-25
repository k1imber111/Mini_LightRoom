"""Локальные маски: линейный и радиальный градиент, кисть, ИИ-маска. Без Qt.

Геометрия хранится в долях полного кадра (0..1), поэтому одна и та же маска рисуется
в превью, в увеличенной вырезке и при экспорте полного разрешения. frame=(fw, fh, x0, y0) —
как в engine.process: img — вырезка из кадра размером fw×fh, её левый верх в (x0, y0).

Слой маски (dict, лежит в params["masks"], сохраняется в sidecar):
    type: "linear" | "radial" | "brush" | "ai"
    name: подпись в списке; on: включена; invert: инвертировать
    adj: {ключ ползунка: значение} — локальные правки (как глобальные ползунки)
    linear: a=[x, y], b=[x, y] — от полной силы в a до нуля в b
    radial: c=[x, y], r=[rx, ry] (доли ширины/высоты кадра), feather 0..100
    brush:  strokes=[{"pts": [[x, y], ...], "r": радиус в долях ширины, "erase": bool}], feather 0..100
    ai:     cat — категория сегментации; сама маска (uint8, весь кадр) передаётся в ключе "arr"
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["LOCAL_KEYS", "layer_mask", "new_layer"]

# Какие ползунки доступны в маске (виньетка локально не имеет смысла).
LOCAL_KEYS = ["exposure", "contrast", "highlights", "shadows", "clarity",
              "temperature", "tint", "vibrance", "saturation", "sharpness"]


def new_layer(kind: str, **geom) -> dict:
    names = {"linear": "Линейный градиент", "radial": "Радиальный градиент", "brush": "Кисть", "ai": "Маска ИИ"}
    layer = {"type": kind, "name": names[kind], "on": True, "invert": False, "adj": {}}
    if kind == "radial":
        layer["feather"] = 50
    if kind == "brush":
        layer.update(strokes=[], feather=50)
    layer.update(geom)
    return layer


def _grid(h: int, w: int, frame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Координаты пикселей вырезки в пикселях полного кадра (центры пикселей)."""
    fw, fh, x0, y0 = frame
    xs = (x0 + np.arange(w, dtype=np.float32) + 0.5)[None, :]
    ys = (y0 + np.arange(h, dtype=np.float32) + 0.5)[:, None]
    return xs, ys, float(fw), float(fh)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _linear(layer, h, w, frame):
    xs, ys, fw, fh = _grid(h, w, frame)
    ax, ay = layer["a"][0] * fw, layer["a"][1] * fh
    bx, by = layer["b"][0] * fw, layer["b"][1] * fh
    dx, dy = bx - ax, by - ay
    t = ((xs - ax) * dx + (ys - ay) * dy) / max(dx * dx + dy * dy, 1e-6)
    return 1 - _smoothstep(t)


def _radial(layer, h, w, frame):
    xs, ys, fw, fh = _grid(h, w, frame)
    cx, cy = layer["c"][0] * fw, layer["c"][1] * fh
    rx, ry = max(layer["r"][0] * fw, 1), max(layer["r"][1] * fh, 1)
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    soft = max(layer.get("feather", 50) / 100, 0.01)
    return 1 - _smoothstep((d - (1 - soft)) / soft)


def _brush(layer, h, w, frame):
    fw, fh, x0, y0 = frame
    strokes = layer.get("strokes", [])
    if not strokes:
        return np.zeros((h, w), np.float32)
    rmax = max(s["r"] for s in strokes) * fw
    pad = int(rmax * 2) + 2  # штрихи рядом с вырезкой тоже влияют через размытие
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    for s in strokes:
        pts = np.array([[p[0] * fw - x0 + pad, p[1] * fh - y0 + pad] for p in s["pts"]], np.float32)
        r = max(1, round(s["r"] * fw))
        val = 0 if s.get("erase") else 255
        pts_i = np.round(pts).astype(np.int32)
        if len(pts_i) == 1:
            cv2.circle(canvas, tuple(pts_i[0]), r, val, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [pts_i], False, val, 2 * r, cv2.LINE_AA)
    m = canvas.astype(np.float32) / 255
    soft = layer.get("feather", 50) / 100
    if soft > 0:
        sigma = soft * rmax * 0.5
        m = cv2.GaussianBlur(m, (0, 0), max(sigma, 0.5))
    return m[pad:pad + h, pad:pad + w]


def _ai(layer, h, w, frame):
    arr = layer.get("arr")
    if arr is None:
        return None
    fw, fh, x0, y0 = frame
    mh, mw = arr.shape[:2]
    sx, sy = mw / fw, mh / fh
    # Пиксель вырезки (i, j) → точка маски: масштаб и сдвиг, одной операцией для любой вырезки.
    M = np.array([[sx, 0, (x0 + 0.5) * sx - 0.5], [0, sy, (y0 + 0.5) * sy - 0.5]], np.float32)
    m = cv2.warpAffine(arr.astype(np.float32) / 255, M, (w, h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return m


_KINDS = {"linear": _linear, "radial": _radial, "brush": _brush, "ai": _ai}


def layer_mask(layer: dict, h: int, w: int, frame=None) -> np.ndarray | None:
    """Маска слоя 0..1 размером h×w (None — у ИИ-слоя ещё нет данных)."""
    frame = frame or (w, h, 0, 0)
    m = _KINDS[layer["type"]](layer, h, w, frame)
    if m is None:
        return None
    m = np.clip(m, 0, 1).astype(np.float32)
    return 1 - m if layer.get("invert") else m
