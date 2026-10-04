"""Оценка брака кадра: расфокус, смаз, промах фокуса. Без Qt, только numpy/OpenCV/rawpy.

Метрика — ширина размытия края σ (в пикселях) по отношению энергий градиента исходника и повторно размытой
копии (Zhuo, Sim): для края, размытого гауссианом σ, отношение равно √(σ²+σ₀²)/σ и не зависит от контраста.
Поэтому закат, небо и луна не дают ложных срабатываний, как дисперсия лапласиана. Считается на ПОЛНОМ
разрешении (на копии в 1/4 разрешения слепа: PROJECT_STATE, спайк S4) и по ЭНЕРГИИ градиента, а не по отобранным
краям: при смазе сильно размытые края выпадают из выборки, и по оставшимся резким кадр казался нормальным.
Расфокус изотропен, смаз — нет: σ считается отдельно по 4 направлениям градиента. Резкий объект на размытом
фоне — не брак: оценивается область, где резкость обязана быть (лицо, точка автофокуса), а не весь кадр.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import rawpy

from . import engine as E

__all__ = ["CFG", "af_point", "analyze_file", "analyze_gray", "energy_grid", "load_gray", "region_sigma"]

APP_DIR = Path(__file__).resolve().parent.parent
QUALITY_FILE = APP_DIR / "quality.json"
SIG0 = 1.0                 # σ повторного размытия
THETAS = (0, 45, 90, 135)  # направления градиента, градусы (0 — вдоль x, y вниз)
LEVELS = ("ok", "doubt", "bad")
_DEFAULTS = {"version": 2, "soft_doubt": 1.15, "soft_bad": 1.7, "motion_aniso": 1.6, "motion_doubt": 1.4,
             "motion_bad": 2.0, "miss_ratio": 1.7, "miss_best": 0.9, "min_struct": 800, "coarse_min": 6.0,
             "cell": 64, "block": 4, "norm_long_side": 6000, "af_box": 0.07, "strip_cells": 8}
CFG = {**_DEFAULTS, **(E.read_json(QUALITY_FILE, {}) if QUALITY_FILE.exists() else {})}
_FLIP = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}


# ---------------------------------------------------------------- загрузка

def load_gray(path) -> tuple[np.ndarray, int]:
    """Серый uint8 кадр в полном разрешении, повёрнутый как в окне, и flip из RAW (для пересчёта точки АФ).
    RAW: встроенный JPEG (у Sony полноразмерный, без демозаики); маленький JPEG — проявка в половину размера."""
    path = Path(path)
    data = E._read_bytes(path)
    if path.suffix.lower() not in E.RAW_EXT:
        g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise ValueError(f"Не удалось прочитать файл {path.name}")
        return g, 0
    with rawpy.imread(io.BytesIO(data)) as raw:
        flip = getattr(raw.sizes, "flip", 0)
        g = None
        try:
            th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                g = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        except Exception:  # noqa: BLE001 — нет встроенного JPEG: проявим сами
            g = None
        if g is None or max(g.shape) < 0.5 * max(raw.sizes.width, raw.sizes.height):
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True, output_bps=8)
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (cv2.rotate(g, _FLIP[flip]) if flip in _FLIP else g), flip


def af_point(path, flip: int = 0) -> tuple[float, float] | None:
    """Точка автофокуса из MakerNote Sony (тег 0x2027 = [ширина, высота, x, y]) в долях кадра, как в окне."""
    try:
        import exifread
        tags = exifread.process_file(io.BytesIO(E._read_bytes(Path(path))), details=True)
        tag = tags.get("MakerNote Tag 0x2027")
        v = [float(x) for x in tag.values] if tag is not None else []
    except Exception:  # noqa: BLE001 — нет тега или нестандартный файл: точки АФ просто нет
        return None
    if len(v) < 4 or v[0] <= 0 or v[1] <= 0:
        return None
    x, y = v[-2] / v[0], v[-1] / v[1]
    return _orient(x, y, flip) if 0 < x < 1 and 0 < y < 1 else None


def _orient(x: float, y: float, flip: int) -> tuple[float, float]:
    """Точка в долях кадра из ориентации сенсора в ориентацию окна (как cv2.rotate в load_gray)."""
    if flip == 3:
        return 1 - x, 1 - y
    if flip == 5:  # 90° против часовой: левый верх уходит в левый низ
        return y, 1 - x
    if flip == 6:  # 90° по часовой: левый верх уходит в правый верх
        return 1 - y, x
    return x, y


# ---------------------------------------------------------------- метрика

class EnergyGrid:
    """Энергии градиента по клеткам `cell` px: e1[θ] — исходник, e2[θ] — размытая копия, n — число пикселей
    с заметной структурой. Только пиксели, где крупный масштаб видит край (он переживает и расфокус, и смаз)."""

    def __init__(self, e1: np.ndarray, e2: np.ndarray, n: np.ndarray, size: tuple, cell: int):
        self.e1, self.e2, self.n, self.size, self.cell = e1, e2, n, size, cell  # e1/e2: (4, rows, cols)


def _cell_sum(a: np.ndarray, c: int) -> np.ndarray:
    h, w = a.shape
    a = np.pad(a, ((0, -h % c), (0, -w % c)))
    return a.reshape(a.shape[0] // c, c, a.shape[1] // c, c).sum((1, 3))


def energy_grid(g8: np.ndarray) -> EnergyGrid:
    """Сетка энергий градиента. Полосами по `strip_cells` клеток: на 21 Мп целиком пик памяти был бы ≈ 1 ГБ."""
    H, W = g8.shape
    c, pad = CFG["cell"], 8
    rows, cols = -(-H // c), -(-W // c)
    e1, e2 = np.zeros((4, rows, cols), np.float64), np.zeros((4, rows, cols), np.float64)
    n = np.zeros((rows, cols), np.float64)
    cs, sn = [np.cos(np.deg2rad(t)) for t in THETAS], [np.sin(np.deg2rad(t)) for t in THETAS]
    step = CFG["strip_cells"] * c
    for r0 in range(0, H, step):
        a, b = max(0, r0 - pad), min(H, r0 + step + pad)
        g = g8[a:b].astype(np.float32)
        gb = cv2.GaussianBlur(g, (0, 0), SIG0)
        coarse = cv2.GaussianBlur(gb, (0, 0), 2.83)  # итого σ = 3
        m = np.hypot(cv2.Sobel(coarse, cv2.CV_32F, 1, 0), cv2.Sobel(coarse, cv2.CV_32F, 0, 1)) >= CFG["coarse_min"]
        top, bot = r0 - a, min(r0 + step, H) - a      # рабочие строки полосы (поля — только для фильтров)
        m = m[top:bot]
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0)[top:bot], cv2.Sobel(g, cv2.CV_32F, 0, 1)[top:bot]
        bx, by = cv2.Sobel(gb, cv2.CV_32F, 1, 0)[top:bot], cv2.Sobel(gb, cv2.CV_32F, 0, 1)[top:bot]
        ri = r0 // c
        nr = -(-m.shape[0] // c)
        n[ri: ri + nr] = _cell_sum(m.astype(np.float32), c)
        for i in range(4):
            d, db = cs[i] * gx + sn[i] * gy, cs[i] * bx + sn[i] * by
            e1[i, ri: ri + nr] = _cell_sum(np.where(m, d * d, 0), c)
            e2[i, ri: ri + nr] = _cell_sum(np.where(m, db * db, 0), c)
    return EnergyGrid(e1, e2, n, (W, H), c)


def _sigma(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """σ размытия из отношения энергий (R = √(σ²+σ₀²)/σ → σ = σ₀/√(R²−1))."""
    r = np.maximum(e1 / np.maximum(e2, 1e-9), 1.001)
    return SIG0 / np.sqrt(np.maximum(r * r - 1.0, 1e-4))


def _slice(f: EnergyGrid, box: tuple | None) -> tuple[slice, slice]:
    if box is None:
        return slice(None), slice(None)
    c = f.cell
    x0, y0, x1, y1 = box
    r0, c0 = int(y0 // c), int(x0 // c)
    return slice(r0, max(r0 + 1, int(-(-y1 // c)))), slice(c0, max(c0 + 1, int(-(-x1 // c))))


def region_sigma(f: EnergyGrid, box: tuple | None = None, mask: np.ndarray | None = None) -> dict:
    """Размытие в рамке (x0, y0, x1, y1 в пикселях) или по маске клеток: n (пикселей структуры), sigma (медиана
    по 4 направлениям), smax (самое размытое направление), aniso (smax / самое резкое), dir (направление смаза,
    градусы, 0 — вправо). Пусто (только n), если структуры мало — небо, туман, тёмный угол."""
    rs, cs = _slice(f, box)
    sel = np.zeros(f.n.shape, bool)
    sel[rs, cs] = True
    if mask is not None:
        sel &= mask
    out = {"n": int(f.n[sel].sum())}
    if out["n"] < CFG["min_struct"]:
        return out
    per = _sigma(f.e1[:, sel].sum(1), f.e2[:, sel].sum(1))
    kmax, kmin = int(per.argmax()), int(per.argmin())
    out.update(sigma=float(np.median(per)), smax=float(per[kmax]), aniso=float(per[kmax] / per[kmin]),
               dir=float(THETAS[kmax]))
    return out


def _blocks(f: EnergyGrid) -> np.ndarray:
    """σ по блокам block×block клеток (NaN — структуры мало)."""
    b = CFG["block"]
    rows, cols = f.n.shape
    pr, pc = -rows % b, -cols % b

    def blk(a: np.ndarray) -> np.ndarray:
        a = np.pad(a, ((0, pr), (0, pc)))
        return a.reshape(a.shape[0] // b, b, a.shape[1] // b, b).sum((1, 3))

    sig = _sigma(blk(f.e1.sum(0)), blk(f.e2.sum(0)))
    sig[blk(f.n) < CFG["min_struct"]] = np.nan
    return sig


def _level(sigma: float, motion: bool) -> str:
    bad, doubt = (CFG["motion_bad"], CFG["motion_doubt"]) if motion else (CFG["soft_bad"], CFG["soft_doubt"])
    return "bad" if sigma >= bad else "doubt" if sigma >= doubt else "ok"


def analyze_gray(g8: np.ndarray, rois: list[dict] | None = None, af: tuple | None = None) -> dict:
    """Брак кадра по серому изображению. rois — [{"name", "box": [x0,y0,x1,y1] в долях кадра}] (лица, главный
    объект) по убыванию важности; af — точка автофокуса камеры в долях. Результат (JSON):
    verdict ok|doubt|bad|unknown, defects [{type soft|motion|focus_miss, level, sigma, name, roi, ...}],
    ref — самая резкая область кадра для сравнения, size, regions (что и как измерено)."""
    H, W = g8.shape
    k = CFG["norm_long_side"] / max(H, W)  # σ в «пикселях кадра 6000 px»: порог не зависит от разрешения камеры
    f = energy_grid(g8)
    res = {"v": CFG["version"], "size": [W, H], "verdict": "unknown", "defects": [], "regions": []}
    bsig = _blocks(f)
    valid = bsig[~np.isnan(bsig)]
    best = float(np.percentile(valid, 10)) * k if valid.size >= 3 else None
    frame_mask = None  # клетки «резких» блоков: не хуже 1.5× лучшего — фон боке сюда не попадает, смаз всей картинки — да
    if best is not None:
        b = CFG["block"]
        iy, ix = np.unravel_index(np.nanargmin(np.where(np.isnan(bsig), np.inf, bsig)), bsig.shape)
        c = f.cell * b
        res["ref"] = [ix * c / W, iy * c / H, min(W, (ix + 1) * c) / W, min(H, (iy + 1) * c) / H]
        res["best_sigma"] = best
        ok_block = np.nan_to_num(bsig, nan=np.inf) <= 1.5 * best / k
        frame_mask = np.kron(ok_block, np.ones((b, b), bool))[: f.n.shape[0], : f.n.shape[1]]

    regions = list(rois or [])
    if af is not None:
        r = CFG["af_box"]
        regions.append({"name": "Точка фокуса", "box": [af[0] - r, af[1] - r * W / H, af[0] + r, af[1] + r * W / H]})
    if not regions:  # нет подсказки, где должно быть резко: лучшие места кадра (резкость хотя бы где-то обязана быть)
        regions = [{"name": "Кадр", "box": None}]
    levels: list[str] = []

    def judge(roi: dict, i: int) -> None:
        box = roi["box"]
        px = None if box is None else (max(0, box[0] * W), max(0, box[1] * H), min(W, box[2] * W), min(H, box[3] * H))
        st = region_sigma(f, px, frame_mask if box is None else None)
        entry = {"name": roi["name"], "box": box, "n": st["n"]}
        res["regions"].append(entry)
        if "sigma" not in st:
            return  # мало структуры (небо, туман): оценить нечего — это не брак
        moving = st["aniso"] >= CFG["motion_aniso"] and st["smax"] * k >= CFG["motion_doubt"]
        sig = (st["smax"] if moving else st["sigma"]) * k
        entry["sigma"] = sig
        level = _level(sig, moving)
        if i > 0 and level == "bad":
            level = "doubt"  # второстепенная область (второе лицо, точка АФ) сама по себе кадр не браковёт
        levels.append(level)
        if level == "ok":
            return
        kind = "motion" if moving else "soft"
        if not moving and best is not None and best <= CFG["miss_best"] and sig / best >= CFG["miss_ratio"] \
                and box is not None:
            kind = "focus_miss"
        d = {"type": kind, "level": level, "sigma": sig, "name": roi["name"], "roi": box}
        if box is None and "ref" in res:
            d["roi"] = res["ref"]
        if moving:
            d["angle"] = st["dir"]
            d["aniso"] = st["aniso"]
        res["defects"].append(d)

    for i, roi in enumerate(regions):
        judge(roi, i)
    if not levels and regions[0]["box"] is not None:  # в областях нет структуры (небо): смотрим лучшие места кадра
        judge({"name": "Кадр", "box": None}, 0)
    if levels:
        res["verdict"] = max(levels, key=LEVELS.index)
    return res


def analyze_file(path, rois: list[dict] | None = None) -> dict:
    """Загрузка + точка автофокуса + анализ. rois — области, которые обязаны быть резкими (лица и т.п.)."""
    g, flip = load_gray(path)
    out = analyze_gray(g, rois, af_point(path, flip))
    out["file"] = [Path(path).stat().st_size, int(Path(path).stat().st_mtime)]  # для сброса кеша при замене файла
    return out


# ---------------------------------------------------------------- лица и глаза (из faces.py)

def face_rois(faces: list[dict], max_faces: int = 3) -> list[dict]:
    """Области резкости из лиц (по убыванию площади): верхние 70% рамки лица — глаза, нос, брови.
    Мелкие лица (уже 3% кадра) и сильно меньше главного не учитываются: далёкая толпа не бракует кадр."""
    out = []
    main = (faces[0]["box"][2] - faces[0]["box"][0]) * (faces[0]["box"][3] - faces[0]["box"][1]) if faces else 0
    for f in faces:
        x0, y0, x1, y1 = f["box"]
        if x1 - x0 < 0.03 or (x1 - x0) * (y1 - y0) < 0.25 * main or len(out) >= max_faces:
            continue
        out.append({"name": "Лицо" if not out else f"Лицо {len(out) + 1}", "box": [x0, y0, x1, y0 + 0.7 * (y1 - y0)],
                    "face": f})
    return out


def add_eyes(res: dict, rois: list[dict]) -> dict:
    """Дописывает в результат закрытые глаза: у главного лица — «брак», у остальных — «сомнительно»; рамка — глаз."""
    names = {"left": "Левый глаз", "right": "Правый глаз"}
    for i, r in enumerate(rois):
        f = r["face"]
        closed = f.get("closed")
        if closed is None:
            r["eyes_checked"] = False  # глаза слишком мелкие: честно «не проверено»
            continue
        if not closed:
            continue
        both = len(closed) == 2
        box = f["eyes"][closed[0]] if not both else [
            min(f["eyes"]["left"][0], f["eyes"]["right"][0]), min(f["eyes"]["left"][1], f["eyes"]["right"][1]),
            max(f["eyes"]["left"][2], f["eyes"]["right"][2]), max(f["eyes"]["left"][3], f["eyes"]["right"][3])]
        level = "bad" if i == 0 else "doubt"
        res["defects"].append({"type": "eyes_closed", "level": level, "name": "Глаза закрыты" if both else
                               f"{names[closed[0]]} закрыт", "roi": box, "sigma": None, "face": r["name"]})
        if res["verdict"] == "unknown" or LEVELS.index(level) > LEVELS.index(res["verdict"]):
            res["verdict"] = level
    res["faces"] = [{"name": r["name"], "box": r["face"]["box"], "closed": r["face"].get("closed")} for r in rois]
    return res
