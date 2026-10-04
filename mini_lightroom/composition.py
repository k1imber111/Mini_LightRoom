"""Авто-кадр по правилам композиции: варианты обрезки с оценкой и причиной. Без Qt.

Идея «сетки якорей» (GAIC, CVPR 2019): не перебирать миллионы окон, а ставить точку интереса на силовые точки
правил (трети, золотая сетка, спираль, центр) при нескольких масштабах и пропорциях — меньше сотни окон на
правило; дальше оценка по жанру (веса в composition.json). Все координаты — доли ПОВЁРНУТОГО холста, как у
`params["crop"]` (точки полного кадра переводит окно через `engine.crop_matrix`).

Субъект — словарь: {"kind": "face"|"person"|"object", "point": (x, y) — куда ставить правило (глаза лица,
верх силуэта, центр пятна), "box": (x0, y0, x1, y1) — что должно остаться в кадре целиком, "facing": -1|0|1 —
куда смотрит (-1 влево, 1 вправо, 0 неизвестно)}.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import crop_overlays as CO
from . import engine as E

__all__ = ["CFG", "RULE_LABEL", "genre_of", "subject_from", "suggest"]

APP_DIR = Path(__file__).resolve().parent.parent
CFG = E.read_json(APP_DIR / "composition.json", {})
RULE_LABEL = {"thirds": "Правило третей", "phi": "Золотое сечение", "spiral": "Золотая спираль", "center": "Симметрия"}
RULE_OVERLAY = {"thirds": "thirds", "phi": "phi", "spiral": "spiral", "center": "center"}
_G = 1 / (1 + CO.PHI)  # 0.382


def genre_of(scene_id: str | None, has_face: bool = False) -> str:
    """Жанр для весов: лицо в кадре — портрет; иначе по сцене CLIP (composition.json), неизвестная — «other»."""
    if has_face:
        return "portrait"
    return CFG.get("scene_genre", {}).get(scene_id or "", "other")


def _spiral_eye(flip: int) -> tuple[float, float]:
    """«Глаз» золотой спирали в единичном квадрате для положения flip (как в crop_overlays)."""
    x, y = CO.spiral()[-1]
    return (1 - x if flip & 1 else x), (1 - y if flip & 2 else y)


def _anchors(rule: str, upper_only: bool) -> list[tuple[float, float, int]]:
    """Силовые точки правила в долях рамки: (tx, ty, flip спирали). upper_only — глаза лица на верхних линиях."""
    if rule == "thirds":
        pts = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    elif rule == "phi":
        pts = [(_G, _G), (1 - _G, _G), (_G, 1 - _G), (1 - _G, 1 - _G)]
    elif rule == "center":
        pts = [(0.5, 0.5)]
    else:  # spiral
        return [(*_spiral_eye(f), f) for f in range(4) if not upper_only or _spiral_eye(f)[1] < 0.5]
    return [(x, y, 0) for x, y in pts if not upper_only or y < 0.5]


def _overlap(box, r) -> float:
    """Доля площади box, лежащая внутри r."""
    w = max(0.0, min(box[2], r[2]) - max(box[0], r[0]))
    h = max(0.0, min(box[3], r[3]) - max(box[1], r[1]))
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return w * h / area


def _gauss(d: float, s: float) -> float:
    return math.exp(-(d / s) ** 2)


def suggest(W: float, H: float, angle: float, subject: dict | None, horizon: float | None, genre: str = "other",
            aspect: float | None = None, weight: np.ndarray | None = None) -> list[dict]:
    """Варианты кадрирования, лучшие первыми: [{rule, label, rect, aspect, score, why, overlay, flip}].
    aspect — зафиксированная пропорция (ширина/высота в пикселях) или None — набор из composition.json
    (0 в нём — исходная). horizon — доля высоты холста (где кончается небо) или None.
    weight — карта важности (H×W, доли холста), чтобы не отрезать значимое. Пустой список — нечего улучшать."""
    w = {"contain": 1, "anchor": 1, "headroom": 0, "lead": 0, "horizon": 1, "kept": 1, "size": 1, "symmetry": 0}
    w.update(CFG.get("genres", {}).get(genre, {}))
    aspects = [aspect] if aspect else [(a or W / H) for a in CFG.get("aspects", [0])]
    kind = (subject or {}).get("kind")
    poi = subject["point"] if subject else (0.5, horizon if horizon is not None else 0.5)
    box = subject.get("box") if subject else None
    facing = (subject or {}).get("facing", 0)
    wsum = float(weight.sum()) if weight is not None else 0.0
    best: dict[tuple, dict] = {}
    for rule in ("thirds", "phi", "spiral", "center"):
        anchors = _anchors(rule, upper_only=kind == "face")
        if not subject:  # нет объекта: ставим по горизонту, по ширине — в центр
            ys = {"thirds": (1 / 3, 2 / 3), "phi": (_G, 1 - _G), "center": (0.5,), "spiral": ()}[rule]
            anchors = [(0.5, y, 0) for y in ys] if horizon is not None else []
        for asp in dict.fromkeys(round(a, 4) for a in aspects):
            bx0, by0, bx1, by1 = E.aspect_crop(W, H, asp, angle)
            for s in CFG.get("scales", [1.0]):
                cw, ch = (bx1 - bx0) * s, (by1 - by0) * s
                if cw * ch < CFG.get("min_area", 0.25):
                    continue
                for tx, ty, flip in anchors:
                    x0 = min(max(poi[0] - tx * cw, 0.0), 1 - cw)
                    y0 = min(max(poi[1] - ty * ch, 0.0), 1 - ch)
                    r = [x0, y0, x0 + cw, y0 + ch]
                    if not E.crop_valid(W, H, r, angle):
                        continue
                    cand = _score(r, rule, flip, (tx, ty), poi, box, kind, facing, horizon, weight, wsum, w, bool(subject))
                    if cand is None:
                        continue
                    cand.update(aspect=asp, rect=[float(v) for v in r])
                    key = (rule, asp)
                    if key not in best or cand["score"] > best[key]["score"]:
                        best[key] = cand
    # по одному лучшему на правило, остальные — разные пропорции/масштабы без почти-дублей
    ranked = sorted(best.values(), key=lambda c: -c["score"])
    out: list[dict] = []
    for c in ranked:
        if any(o["rule"] == c["rule"] for o in out) or any(_iou(o["rect"], c["rect"]) > 0.85 for o in out):
            continue
        out.append(c)
        if len(out) >= CFG.get("variants", 5):
            break
    return out


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _score(r, rule, flip, anchor, poi, box, kind, facing, horizon, weight, wsum, w, has_subject):
    x0, y0, x1, y1 = r
    cw, ch = x1 - x0, y1 - y0
    comp: dict[str, float] = {}
    why: list[str] = []
    if box is not None:
        inside = _overlap(box, r)
        if kind == "face" and inside < 0.97:
            return None  # лицо режет только тот, кто не смотрит на результат
        margin = min(box[0] - x0, x1 - box[2], box[1] - y0, y1 - box[3])
        comp["contain"] = inside ** 3 * (1.0 if margin >= CFG.get("margin", 0.03) * min(cw, ch) else 0.8)
        top = (box[1] - y0) / ch
        if kind in ("face", "person"):
            comp["headroom"] = 1.0 if 0.04 <= top <= 0.16 else max(0.0, 1 - min(abs(top - 0.04), abs(top - 0.16)) / 0.25)
            why.append(f"запас над головой {max(0, top) * 100:.0f}%")
    if has_subject:
        rx, ry = (poi[0] - x0) / cw, (poi[1] - y0) / ch
        d = math.hypot(rx - anchor[0], ry - anchor[1])
        comp["anchor"] = _gauss(d, 0.07)
        if comp["anchor"] > 0.6 and rule != "center":
            why.append({"face": "глаза", "person": "человек", "object": "объект"}.get(kind, "объект")
                       + " на " + {"thirds": "пересечении третей", "phi": "золотой точке", "spiral": "глазу спирали"}[rule])
        if facing:
            ahead = (1 - rx) if facing > 0 else rx
            comp["lead"] = min(1.0, max(0.0, (ahead - 0.3) / 0.3))
            why.append(f"перед взглядом {ahead * 100:.0f}%")
        if rule == "center":
            comp["symmetry"] = _gauss(abs(rx - 0.5), 0.05)
    if horizon is not None:
        hr = (horizon - y0) / ch
        if 0 < hr < 1:
            lines = (_G, 1 - _G) if rule == "phi" else (1 / 3, 2 / 3)
            hs = max(_gauss(hr - ln, 0.04) for ln in lines)
            comp["horizon"] = hs * (0.4 if abs(hr - 0.5) < 0.07 else 1.0)
            if hs > 0.6:
                why.append("горизонт на трети" if rule != "phi" else "горизонт на золотой линии")
        else:
            comp["horizon"] = 0.5
    if weight is not None and wsum > 0:
        h, wd = weight.shape
        part = weight[int(y0 * h): max(int(y0 * h) + 1, int(y1 * h)), int(x0 * wd): max(int(x0 * wd) + 1, int(x1 * wd))]
        kept = float(part.sum()) / wsum
        comp["kept"] = kept ** 1.5
        why.append(f"сохранено {kept * 100:.0f}% главного")
    comp["size"] = math.sqrt(cw * ch)
    num = sum(w.get(k, 0) * v for k, v in comp.items())
    den = sum(w.get(k, 0) for k in comp)
    if den <= 0:
        return None
    return {"rule": rule, "label": RULE_LABEL[rule], "score": num / den, "why": " · ".join(why) or "без потери разрешения",
            "overlay": RULE_OVERLAY[rule], "flip": flip, "components": {k: round(v, 3) for k, v in comp.items()}}


def subject_from(faces: list[dict] | None, people: np.ndarray | None, salient: np.ndarray | None,
                 min_share: float = 0.015) -> dict | None:
    """Главный объект кадра по приоритету: лицо (глаза, куда смотрит) → человек (верх силуэта) → заметное пятно.
    Маски — float/uint8 H×W в долях полного кадра. None — объекта нет (пейзаж, город)."""
    if faces:
        f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        el, er = f["eyes"]["left"], f["eyes"]["right"]
        x0, y0, x1, y1 = f["box"]
        fh = y1 - y0
        head = (x0, max(0.0, y0 - 0.3 * fh), x1, y1)  # лицо без причёски: добавляем «лоб и волосы» сверху
        yaw = f.get("yaw", 0.0)
        mx, my = (el[0] + el[2] + er[0] + er[2]) / 4, (el[1] + el[3] + er[1] + er[3]) / 4
        # «должно остаться в кадре» — голова целиком; тело обрезать можно (ниже пояса не требуем)
        return {"kind": "face", "point": (mx, my), "box": head, "facing": 0 if abs(yaw) < 0.12 else (1 if yaw > 0 else -1)}
    if people is not None and people.mean() >= min_share:
        pt = E.subject_point(people)
        m = people > 0.5
        ys, xs = np.nonzero(m)
        if pt is not None and xs.size:
            hh, ww = m.shape
            return {"kind": "person", "point": pt, "box": (xs.min() / ww, ys.min() / hh, xs.max() / ww, ys.max() / hh),
                    "facing": 0}
    if salient is not None and salient.mean() >= 0.02:
        ys, xs = np.nonzero(salient > 0.5)
        if xs.size:
            hh, ww = salient.shape
            return {"kind": "object", "point": (float(xs.mean() / ww), float(ys.mean() / hh)),
                    "box": (xs.min() / ww, ys.min() / hh, xs.max() / ww, ys.max() / hh), "facing": 0}
    return None
