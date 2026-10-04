"""Авто-кадр по правилам композиции: поиск главного объекта и кадрирование под выбранное правило. Без Qt.

Идея «сетки якорей» (GAIC, CVPR 2019): не перебирать миллионы окон, а ставить точку интереса на силовые точки
правила (пересечения линий его сетки) при нескольких масштабах — меньше сотни окон; дальше оценка по жанру
(веса в composition.json). Все координаты рамки — доли ПОВЁРНУТОГО холста, как у `params["crop"]`.

Анализ кадра (`subject_from`, `analysis_to_canvas`) считается один раз в долях ПОЛНОГО кадра; после этого
перебор правил мгновенный — это и даёт «выбрал правило — кадр сразу перекадрирован».

Главный объект ищется каскадом: лицо → человек → предмет (сегментация: птица, корабль, машина…) → изолированное
яркое или тёмное пятно (луна, солнце). Иначе объекта нет — окно просит указать точку.
Субъект — словарь: {"kind": "face"|"person"|"object", "point": (x, y) — куда ставить правило, "box": (x0, y0, x1, y1) —
что должно остаться в кадре целиком, "facing": -1|0|1 — куда смотрит, "how": чем найден}.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import crop_overlays as CO
from . import engine as E

__all__ = ["CFG", "RULES", "RULE_LABEL", "analysis_to_canvas", "genre_of", "isolated_blob", "plan", "subject_from",
           "suggest"]

APP_DIR = Path(__file__).resolve().parent.parent
CFG = E.read_json(APP_DIR / "composition.json", {})
RULES = ("thirds", "phi", "spiral", "triangle", "diagonal", "center", "grid")  # совпадают с видами сеток crop_overlays
RULE_LABEL = {"thirds": "Правило третей", "phi": "Золотое сечение", "spiral": "Золотая спираль",
              "triangle": "Золотой треугольник", "diagonal": "Диагонали", "center": "Симметрия", "grid": "Сетка"}
ANCHOR_WHY = {"thirds": "на пересечении третей", "phi": "на золотой точке", "spiral": "на глазу спирали",
              "triangle": "на линии треугольника", "diagonal": "на пересечении диагоналей", "grid": "на узле сетки"}
_G = 1 / (1 + CO.PHI)  # 0.382
# Линии, на которые встаёт горизонт, когда объекта нет: у золотых правил — золотые, у сетки — четверти, у остальных — трети
HORIZON_LINES = {"phi": (_G, 1 - _G), "grid": (0.25, 0.75), "center": (0.5,)}
OBJECT_ID = "object"


def genre_of(scene_id: str | None, has_face: bool = False) -> str:
    """Жанр для весов: лицо в кадре — портрет; иначе по сцене CLIP (composition.json), неизвестная — «other»."""
    if has_face:
        return "portrait"
    return CFG.get("scene_genre", {}).get(scene_id or "", "other")


def _spiral_eye(flip: int) -> tuple[float, float]:
    """«Глаз» золотой спирали в единичном квадрате для положения flip (как в crop_overlays)."""
    x, y = CO.spiral()[-1]
    return (1 - x if flip & 1 else x), (1 - y if flip & 2 else y)


def _intersections(lines: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Внутренние точки пересечения отрезков сетки (для диагоналей и квадратной сетки): силовые точки правила."""
    pts: list[tuple[float, float]] = []
    segs = [(p[0], p[-1]) for p in lines]
    for i, (a, b) in enumerate(segs):
        for c, d in segs[i + 1:]:
            r, s = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
            u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
            if 0.02 < t < 0.98 and 0.02 < u < 0.98:
                pts.append((a[0] + t * r[0], a[1] + t * r[1]))
    return list(dict.fromkeys((round(x, 4), round(y, 4)) for x, y in pts))


def _anchors(rule: str, upper_only: bool, asp: float, flip: int | None = None) -> list[tuple[float, float, int]]:
    """Силовые точки правила в долях рамки: (tx, ty, положение сетки). upper_only — глаза лица на верхних линиях;
    flip — зафиксированное положение спирали/треугольника (None — любое)."""
    flips = range(4) if flip is None else (flip,)
    if rule == "thirds":
        pts = [(x, y, 0) for x in (1 / 3, 2 / 3) for y in (1 / 3, 2 / 3)]
    elif rule == "phi":
        pts = [(x, y, 0) for x in (_G, 1 - _G) for y in (_G, 1 - _G)]
    elif rule == "center":
        pts = [(0.5, 0.5, 0)]
    elif rule == "spiral":
        pts = [(*_spiral_eye(f), f) for f in flips]
    elif rule == "triangle":  # основания перпендикуляров на диагональ — узлы золотого треугольника
        pts = [(*seg[1], f) for f in flips for seg in CO.lines("triangle", f, asp)[1:]]
    else:  # diagonal, grid: пересечения линий сетки
        pts = [(x, y, 0) for x, y in _intersections(CO.lines(rule, 0, asp))]
    return [p for p in pts if not upper_only or p[1] < 0.5]


def _overlap(box, r) -> float:
    """Доля площади box, лежащая внутри r."""
    w = max(0.0, min(box[2], r[2]) - max(box[0], r[0]))
    h = max(0.0, min(box[3], r[3]) - max(box[1], r[1]))
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    return w * h / area


def _gauss(d: float, s: float) -> float:
    return math.exp(-(d / s) ** 2)


def suggest(W: float, H: float, angle: float, subject: dict | None, horizon: float | None, genre: str = "other",
            aspect: float | None = None, weight: np.ndarray | None = None, rules: tuple | None = None,
            flip: int | None = None) -> list[dict]:
    """Варианты кадрирования, лучшие первыми: [{rule, label, rect, aspect, score, why, overlay, flip}].
    aspect — зафиксированная пропорция (ширина/высота в пикселях) или None — набор из composition.json
    (0 в нём — исходная). horizon — доля высоты холста (где кончается небо) или None. weight — карта важности
    (H×W, доли холста), чтобы не отрезать значимое. rules — только эти правила (по умолчанию все семь; с одним
    правилом возвращается один лучший вариант). flip — зафиксированное положение спирали/треугольника.
    Пустой список — нечего улучшать (нет ни объекта, ни горизонта)."""
    w = {"contain": 1, "anchor": 1, "headroom": 0, "lead": 0, "horizon": 1, "kept": 1, "size": 1, "symmetry": 0}
    w.update(CFG.get("genres", {}).get(genre, {}))
    aspects = [aspect] if aspect else [(a or W / H) for a in CFG.get("aspects", [0])]
    kind = (subject or {}).get("kind")
    poi = subject["point"] if subject else (0.5, horizon if horizon is not None else 0.5)
    box = subject.get("box") if subject else None
    facing = (subject or {}).get("facing", 0)
    wsum = float(weight.sum()) if weight is not None else 0.0
    prior = CFG.get("rule_prior", {}).get(genre, {})
    best: dict[tuple, dict] = {}
    for rule in rules or RULES:
        for asp in dict.fromkeys(round(a, 4) for a in aspects):
            anchors = _anchors(rule, kind == "face", asp, flip)
            if not subject:  # нет объекта: ставим по горизонту, по ширине — в центр
                ys = HORIZON_LINES.get(rule, (1 / 3, 2 / 3))
                anchors = [(0.5, y, 0) for y in ys] if horizon is not None else []
            bx0, by0, bx1, by1 = E.aspect_crop(W, H, asp, angle)
            for s in CFG.get("scales", [1.0]):
                cw, ch = (bx1 - bx0) * s, (by1 - by0) * s
                if cw * ch < CFG.get("min_area", 0.25):
                    continue
                for tx, ty, fl in anchors:
                    x0 = min(max(poi[0] - tx * cw, 0.0), 1 - cw)
                    y0 = min(max(poi[1] - ty * ch, 0.0), 1 - ch)
                    r = [x0, y0, x0 + cw, y0 + ch]
                    if not E.crop_valid(W, H, r, angle):
                        continue
                    cand = _score(r, rule, fl, (tx, ty), poi, box, kind, facing, horizon, weight, wsum, w, bool(subject))
                    if cand is None:
                        continue
                    cand.update(aspect=asp, rect=[float(v) for v in r])
                    cand["score"] *= prior.get(rule, 1.0)  # какое правило жанру обычно подходит больше
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
                       + " " + ANCHOR_WHY[rule])
        if facing:
            ahead = (1 - rx) if facing > 0 else rx
            comp["lead"] = min(1.0, max(0.0, (ahead - 0.3) / 0.3))
            why.append(f"перед взглядом {ahead * 100:.0f}%")
        if rule == "center":
            comp["symmetry"] = _gauss(abs(rx - 0.5), 0.05)
    if horizon is not None:
        hr = (horizon - y0) / ch
        if 0 < hr < 1:
            lines = HORIZON_LINES.get(rule, (1 / 3, 2 / 3))
            hs = max(_gauss(hr - ln, 0.04) for ln in lines)
            comp["horizon"] = hs * (0.4 if abs(hr - 0.5) < 0.07 and rule != "center" else 1.0)
            if hs > 0.6:
                why.append("горизонт на " + ("золотой линии" if rule == "phi" else "четверти" if rule == "grid" else
                                             "середине" if rule == "center" else "трети"))
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
            "overlay": rule, "flip": flip, "components": {k: round(v, 3) for k, v in comp.items()}}


# ---------------------------------------------------------------- поиск главного объекта

_BLOB_SIGMAS = (2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 18.0, 26.0)  # на копии 256 px: от 0.5% до 25% ширины кадра


def isolated_blob(img: np.ndarray) -> dict | None:
    """Изолированное яркое или тёмное компактное пятно (луна, солнце, одинокая птица): многомасштабная разность
    гауссианов по яркости с гаммой. Пик берётся, только если он в разы сильнее любого другого места кадра:
    у гор, города и закатных облаков много похожих откликов — тогда объекта нет (None), и окно просит указать точку.
    img — RGB float 0..1 с авто-тоном. Результат: {"point", "box"} в долях кадра."""
    g = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    h0, w0 = g.shape
    g = np.sqrt(cv2.resize(g, (256, max(16, round(256 * h0 / w0))), interpolation=cv2.INTER_AREA))
    resp = np.stack([np.abs(cv2.GaussianBlur(g, (0, 0), s) - cv2.GaussianBlur(g, (0, 0), s * 1.6)) for s in _BLOB_SIGMAS])
    best, idx = resp.max(0), resp.argmax(0)
    h, w = best.shape
    y, x = np.unravel_index(best.argmax(), best.shape)
    peak, s = float(best[y, x]), _BLOB_SIGMAS[idx[y, x]]
    yy, xx = np.mgrid[0:h, 0:w]
    far = (yy - y) ** 2 + (xx - x) ** 2 > (3.5 * s) ** 2
    second = float(best[far].max()) if far.any() else 0.0
    if peak < 0.03 or peak / max(second, 1e-6) < 2.4 or peak / max(float(np.percentile(best, 90)), 1e-6) < 3.5:
        return None
    r = 1.8 * s  # радиус пятна в пикселях копии
    rx, ry = r / w, r / h
    box = (max(0.0, x / w - rx), max(0.0, y / h - ry), min(1.0, x / w + rx), min(1.0, y / h + ry))
    return {"point": (x / w, y / h), "box": box}


def clipped_disc(img: np.ndarray) -> dict | None:
    """Пересвеченный круглый диск на заметно более тёмном фоне (закатное солнце, половина диска у горизонта): белое пятно
    компактной формы. Рваные пересветы (снег, окна, блики) отсеиваются заполнением рамки и формой."""
    g = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    h0, w0 = g.shape
    g = cv2.resize(g, (400, max(16, round(400 * h0 / w0))), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats((g >= 0.97).astype(np.uint8))
    best = None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] / (h * w)
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if not 0.0008 <= a <= 0.05 or st[i, cv2.CC_STAT_AREA] / (bw * bh) < 0.65 or not 0.4 <= bw / bh <= 2.6:
            continue
        comp = lab == i
        k = np.ones((3, 3), np.uint8)
        ring = cv2.dilate(comp.astype(np.uint8), k, iterations=max(4, int(max(bw, bh) * 0.7))) > 0
        ring &= ~(cv2.dilate(comp.astype(np.uint8), k, iterations=2) > 0)
        contrast = 1.0 - float(g[ring].mean())
        if contrast > 0.4 and (best is None or a * contrast > best[0]):
            x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            best = (a * contrast, {"point": (float(cen[i][0] / w), float(cen[i][1] / h)),
                                   "box": (max(0.0, (x - 0.3 * bw) / w), max(0.0, (y - 0.3 * bh) / h),
                                           min(1.0, (x + 1.3 * bw) / w), min(1.0, (y + 1.3 * bh) / h))})
    return best[1] if best else None


def _components(mask: np.ndarray | None, kind: str, min_share: float, thr: float = 0.5) -> list[dict]:
    """Связные области маски (вероятность > thr): площадь, рамка, оценка «главности»: крупнее, ближе к центру и не
    обрезано краем кадра. У предметов thr ниже: сглаживание края «съедает» мелкие объекты (цапля в 0.2% кадра)."""
    if mask is None:
        return []
    m = (mask > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    out = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] / (h * w)
        if a < min_share:
            continue
        cx, cy = cen[i][0] / w, cen[i][1] / h
        x, y, bw, bh = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        centrality = math.exp(-((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / 0.25)
        cut = 0.6 if (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1) else 1.0  # обрезано краем — вряд ли главное
        score = math.sqrt(a) * (0.3 + 1.2 * centrality) * cut * (1.3 if kind == "person" else 1.0)
        out.append({"kind": kind, "area": a, "score": score, "box": (x / w, y / h, (x + bw) / w, (y + bh) / h), "label": lab,
                    "i": i, "shape": (h, w)})
    return out


def subject_from(faces: list[dict] | None, people: np.ndarray | None, objects: np.ndarray | None = None,
                 img: np.ndarray | None = None) -> dict | None:
    """Главный объект кадра по каскаду: лицо (глаза, куда смотрит) → человек (верх силуэта) → предмет из сегментации
    (птица, корабль, машина) → изолированное пятно (луна, солнце). Маски — float H×W в долях полного кадра, img —
    RGB с авто-тоном для пятна. None — объекта нет (пейзаж, город, облака): окно просит указать точку."""
    if faces:
        f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        el, er = f["eyes"]["left"], f["eyes"]["right"]
        x0, y0, x1, y1 = f["box"]
        head = (x0, max(0.0, y0 - 0.3 * (y1 - y0)), x1, y1)  # лицо без причёски: добавляем «лоб и волосы» сверху
        yaw = f.get("yaw", 0.0)
        mx, my = (el[0] + el[2] + er[0] + er[2]) / 4, (el[1] + el[3] + er[1] + er[3]) / 4
        # «должно остаться в кадре» — голова целиком; тело обрезать можно (ниже пояса не требуем)
        return {"kind": "face", "point": (mx, my), "box": head, "how": "лицо",
                "facing": 0 if abs(yaw) < 0.12 else (1 if yaw > 0 else -1)}
    cands = _components(people, "person", 0.004) + _components(objects, OBJECT_ID, 0.0003, 0.3)
    if cands:
        best = max(cands, key=lambda c: c["score"])
        bc = ((best["box"][0] + best["box"][2]) / 2, (best["box"][1] + best["box"][3]) / 2)
        diag = math.hypot(best["box"][2] - best["box"][0], best["box"][3] - best["box"][1])

        def near(c) -> bool:  # пара птиц, семья: похожие по размеру и рядом, а не любые предметы по всему кадру
            cc = ((c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2)
            return 0.4 * best["area"] <= c["area"] <= 2.5 * best["area"] and math.hypot(cc[0] - bc[0], cc[1] - bc[1]) <= 1.5 * diag

        group = [c for c in cands if c is best or (c["kind"] == best["kind"] and near(c))]
        box = (min(c["box"][0] for c in group), min(c["box"][1] for c in group),
               max(c["box"][2] for c in group), max(c["box"][3] for c in group))
        if best["kind"] == "person":  # верх силуэта — там глаза, их и ставят на силовую точку
            h, w = best["shape"]
            top = int(best["box"][1] * h)
            ys, xs = np.nonzero((best["label"] == best["i"])[top: top + max(1, int((best["box"][3] - best["box"][1]) * h / 4))])
            pt = (float(xs.mean() / w), float((ys.mean() + top) / h)) if xs.size else                 ((box[0] + box[2]) / 2, box[1] + 0.1 * (box[3] - box[1]))
            return {"kind": "person", "point": pt, "box": box, "facing": 0, "how": "человек"}
        return {"kind": "object", "point": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "box": box, "facing": 0,
                "how": "предмет"}
    if img is not None:
        b = isolated_blob(img)
        if b:
            return {"kind": "object", "point": b["point"], "box": b["box"], "facing": 0, "how": "яркое пятно"}
        b = clipped_disc(img)
        if b:
            return {"kind": "object", "point": b["point"], "box": b["box"], "facing": 0, "how": "яркий диск"}
    return None


def manual_subject(x: float, y: float, w: float = 0.08) -> dict:
    """Объект по точке, которую указал пользователь (доли полного кадра): рамка «должно остаться» — ±w по ширине."""
    return {"kind": "object", "point": (x, y), "box": (max(0.0, x - w), max(0.0, y - w * 1.4), min(1.0, x + w),
                                                       min(1.0, y + w * 1.4)), "facing": 0, "how": "ваша точка"}


# ---------------------------------------------------------------- от анализа к рамке

def analysis_to_canvas(an: dict, W: float, H: float, angle: float, manual: tuple | None = None) -> tuple:
    """Анализ кадра (доли полного кадра) → (субъект, горизонт) в долях ПОВЁРНУТОГО холста. manual — точка, указанная
    пользователем: заменяет найденный объект."""
    A, _ = E.crop_matrix(W, H, None, angle)

    def tp(x, y):
        rx, ry = A @ (x * W, y * H, 1.0)
        return float(rx / W), float(ry / H)

    subject = manual_subject(*manual) if manual else (dict(an["subject"]) if an.get("subject") else None)
    if subject:
        subject["point"] = tp(*subject["point"])
        xs, ys = zip(*(tp(x, y) for x in (subject["box"][0], subject["box"][2]) for y in (subject["box"][1], subject["box"][3])),
                     strict=True)
        subject["box"] = (min(xs), min(ys), max(xs), max(ys))
    horizon = tp(0.5, an["horizon"])[1] if an.get("horizon") is not None else None
    return subject, horizon


def plan(an: dict, W: float, H: float, angle: float, aspect: float | None, rules: tuple | None = None,
         manual: tuple | None = None, flip: int | None = None) -> list[dict]:
    """Кадрирование по анализу кадра: rules=(правило,) — один лучший вариант под выбранное правило."""
    subject, horizon = analysis_to_canvas(an, W, H, angle, manual)
    return suggest(W, H, angle, subject, horizon, an.get("genre", "other"), aspect, an.get("weight"), rules, flip)
