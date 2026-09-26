"""Движок Mini LightRoom: чтение RAW/JPEG, конвейер правок, авто-тон,
перенос стиля с чужого фото, экспорт LUT (.cube), оценка резкости.

Модуль не зависит от Qt, поэтому его можно звать из консоли и из
процессов экспорта.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy

from . import masks as M

RAW_EXT = {".arw", ".srf", ".sr2", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PHOTO_EXT = RAW_EXT | IMG_EXT

# (ключ, подпись, минимум, максимум, группа)
SLIDERS = [
# Значение 100 — прежний максимум; шкала расширена вдвое, старые правки выглядят так же.
# Сочность и насыщенность ниже -100 не опускаются: там цвета инвертируются.
    ("exposure", "Экспозиция", -500, 500, "Свет"),  # сотые доли EV
    ("contrast", "Контраст", -200, 200, "Свет"),
    ("highlights", "Света", -200, 200, "Свет"),
    ("shadows", "Тени", -200, 200, "Свет"),
    ("clarity", "Чёткость", -200, 200, "Свет"),
    ("temperature", "Температура", -200, 200, "Цвет"),
    ("tint", "Оттенок", -200, 200, "Цвет"),
    ("vibrance", "Сочность", -100, 200, "Цвет"),
    ("saturation", "Насыщенность", -100, 200, "Цвет"),
    ("sharpness", "Резкость", 0, 200, "Детали"),
    ("vignette", "Виньетка", -200, 200, "Детали"),
    # Шумодав (ИИ, enhance.py) применяется к исходнику до process(), поэтому process его не трогает.
    ("denoise", "Шумодав", 0, 100, "Детали"),
    # ИИ-ретушь: пространственные, по маскам ИИ из p["ai_arr"] (готовит окно); без масок — ничего не делают.
    ("retouch", "Ретушь кожи", 0, 100, "ИИ-ретушь"),
    ("bokeh", "Размытие фона", 0, 100, "ИИ-ретушь"),
]
AI_SLIDER_CATS = {"retouch": "people", "bokeh": "subject"}  # какая маска ИИ нужна ползунку


def default_params() -> dict:
    p = {key: 0 for key, *_ in SLIDERS}
    # style_strength — сила цвета стиля, style_tone — сила света (тона), style_skin — защита кожи,
    # style_mode: 0 — мягкий перенос (среднее и разброс), 1 — точный (распределения L/a/b),
    # style2/style_mix — второй стиль и его доля в смеси.
    p.update(style="", style_strength=70, style_tone=70, style_skin=60, style_mode=0, style2="", style_mix=50,
             look="", look_strength=100, masks=[],
             crop=None, angle=0.0,  # обрезка [x0, y0, x1, y1] в долях повёрнутого кадра, поворот в градусах
             hsl={}, curve={},
             ai_grade=None)  # ИИ-цветокоррекция (grading.py): {"id","name","rule","recipe","masks","strength"}  # HSL: {цвет: [оттенок, насыщ., яркость]}; кривая: {"rgb"|"r"|"g"|"b": точки 0..255}
    return p


def normalize_params(saved: dict | None) -> dict:
    """Сохранённые правки → полный набор параметров. Старые правки (до раздельных сил стиля)
    получают силу тона, равную прежней общей силе, чтобы выглядеть как раньше."""
    saved = dict(saved or {})
    if "style_strength" in saved and "style_tone" not in saved:
        saved["style_tone"] = saved["style_strength"]
    return {**default_params(), **saved}


LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------- чтение

def _read_bytes(path) -> bytes:
    # Читаем через Python, а не C-библиотекой: так работают пути с кириллицей.
    return Path(path).read_bytes()


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    if k >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def load_image(path, half: bool = True, max_side: int | None = None) -> np.ndarray:
    """Возвращает RGB float32 в диапазоне 0..1."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        with rawpy.imread(io.BytesIO(_read_bytes(path))) as raw:
            rgb = raw.postprocess(half_size=half, use_camera_wb=True,
                                  no_auto_bright=True, output_bps=16)
        img = rgb.astype(np.float32) / 65535.0
    else:
        data = np.frombuffer(_read_bytes(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Не удалось прочитать файл {path.name}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return resize_max(img, max_side) if max_side else img


def full_size(path) -> tuple[int, int]:
    """(ширина, высота) кадра в полном разрешении, без проявки RAW."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        with rawpy.imread(io.BytesIO(_read_bytes(path))) as raw:
            s = raw.sizes
            return (s.height, s.width) if s.flip in (5, 6) else (s.width, s.height)
    h, w = load_image(path).shape[:2]
    return w, h


_FLIP = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}


def camera_aspect(path) -> float | None:
    """Соотношение сторон, в котором снимала камера (по встроенному JPEG; у Sony в режиме 16:9 RAW
    всё равно полный 3:2). None — не удалось узнать."""
    try:
        th = load_thumb(path, 256)
    except Exception:  # noqa: BLE001 — нет миниатюры: просто без подсказки формата
        return None
    return th.shape[1] / th.shape[0]


def load_thumb(path, size: int = 512) -> np.ndarray:
    """Быстрая миниатюра uint8 RGB: для RAW берём встроенный в файл JPEG."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        with rawpy.imread(io.BytesIO(_read_bytes(path))) as raw:
            flip = getattr(raw.sizes, "flip", 0)
            try:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    # EXIF встроенного JPEG не учитываем: поворот задаёт flip из RAW, иначе кадр повернётся дважды.
                    bgr = cv2.imdecode(np.frombuffer(th.data, np.uint8),
                                       cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    rgb = np.asarray(th.data)
                if flip in _FLIP:
                    rgb = cv2.rotate(rgb, _FLIP[flip])
            except Exception:  # noqa: BLE001 — нет миниатюры, проявляем сами
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
        return np.ascontiguousarray(resize_max(rgb, size))
    return (resize_max(load_image(path), size) * 255).astype(np.uint8)


def sharpness(thumb_u8: np.ndarray) -> float:
    """Дисперсия лапласиана: чем меньше, тем кадр размытее."""
    gray = cv2.cvtColor(thumb_u8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def save_jpeg(path, img: np.ndarray, quality: int = 92) -> None:
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Кодирование JPEG не удалось")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.tobytes())


# ---------------------------------------------------------------- стиль

QUANTILES = np.linspace(0, 100, 65)


def lab_stats(img: np.ndarray) -> dict:
    """Статистики кадра в LAB: среднее, разброс и квантили каждого канала (для точного переноса)."""
    lab = cv2.cvtColor(np.clip(resize_max(img, 512), 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    flat = lab.reshape(-1, 3)
    return {"mean": flat.mean(0).tolist(), "std": flat.std(0).tolist(),
            "q": np.percentile(flat, QUANTILES, axis=0).T.round(3).tolist()}


def style_from_image(path, name: str) -> dict:
    img = load_image(path, half=True, max_side=1024)
    return {"name": name, **lab_stats(img)}


def mix_styles(a: dict, b: dict, t: float) -> dict:
    """Смесь стилей: t=0 — только a, t=1 — только b. Квантили смешиваются тоже,
    это даёт промежуточное распределение, а не двойную картинку."""
    out = {"name": f"{a['name']} + {b['name']}"}
    for key in ("mean", "std", "q"):
        if key in a and key in b:
            out[key] = (np.array(a[key]) * (1 - t) + np.array(b[key]) * t).tolist()
    return out


def _skin_mask(lab: np.ndarray) -> np.ndarray:
    """Мягкая маска тонов кожи в LAB: тёплый оттенок 25–70°, умеренная насыщенность."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a))
    chroma = np.hypot(a, b)
    ramp = lambda x, lo, hi, soft: np.clip((x - lo) / soft, 0, 1) * np.clip((hi - x) / soft, 0, 1)
    return ramp(hue, 20, 75, 10) * ramp(chroma, 6, 50, 6) * ramp(L, 20, 95, 10)


def _transfer_curve(src_q, ref_q, span: tuple, lo: float = 0.5, hi: float = 2.0) -> tuple:
    """Кривая «квантили кадра → квантили референса» с ограниченной крутизной.
    Без ограничения почти однородное небо растягивается до всего диапазона референса
    и вокруг ярких объектов появляются ореолы."""
    grid = np.linspace(span[0], span[1], 257, dtype=np.float32)
    raw = np.interp(grid, src_q, ref_q)
    slope = np.clip(np.diff(raw) / np.diff(grid), lo, hi)
    curve = np.concatenate([[0], np.cumsum(slope * np.diff(grid))])
    mid = len(src_q) // 2  # медиана кадра попадает точно в медиану референса
    curve += ref_q[mid] - np.interp(src_q[mid], grid, curve)
    return grid, curve.astype(np.float32)


def _apply_style(img: np.ndarray, style: dict, p: dict, src_stats: dict | None) -> np.ndarray:
    src = src_stats or lab_stats(img)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    if p.get("style_mode", 0) == 1 and "q" in style and "q" in src:
        # Точный перенос: каждый канал получает распределение референса (сопоставление по квантилям).
        out = np.empty_like(lab)
        for c in range(3):
            grid, curve = _transfer_curve(src["q"][c], style["q"][c], (0, 100) if c == 0 else (-128, 128))
            out[..., c] = np.interp(lab[..., c], grid, curve)
    else:
        # Мягкий перенос по Рейнхарду: среднее и разброс, отношение разбросов ограничено.
        s_mean, s_std = np.array(src["mean"], np.float32), np.array(src["std"], np.float32)
        r_mean, r_std = np.array(style["mean"], np.float32), np.array(style["std"], np.float32)
        out = (lab - s_mean) * np.clip(r_std / (s_std + 1e-6), 0.5, 2.0) + r_mean
    # Мягкий потолок сдвига: при доминирующем цвете (небо, зелень) картинка не «уплывает».
    shift = out - lab
    limit = np.array([40, 30, 30], np.float32)
    shift = limit * np.tanh(shift / limit)
    color = p.get("style_strength", 70) / 100
    tone = p.get("style_tone", p.get("style_strength", 70)) / 100
    if p.get("style_skin"):  # маска по исходным цветам кадра, до сдвига
        color = (color * (1 - p["style_skin"] / 100 * _skin_mask(lab)))[..., None]
    lab[..., 1:] += shift[..., 1:] * color
    lab[..., 0] = np.clip(lab[..., 0] + shift[..., 0] * tone, 0, 100)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


# ---------------------------------------------------------------- пресеты-образы (looks)
# Рецепт — dict из looks/*.json. Все операции поточечные, поэтому образ попадает и в LUT.
# Шкалы как в Lightroom: кривые 0..255, HSL ±100, тонирование — оттенок в градусах и сила 0..100.

HSL_CENTERS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180,
               "blue": 225, "purple": 270, "magenta": 315}


def hsl_weights(hue) -> np.ndarray:
    """Вклад 8 цветов HSL (порядок HSL_CENTERS) для оттенка в градусах HSV — треугольники ±45°,
    как в _apply_hsl; сумма не больше 1. Для целевой правки: какие ползунки двигать и насколько."""
    h = np.asarray(hue, np.float32)[..., None]
    c = np.array(list(HSL_CENTERS.values()), np.float32)
    w = np.clip(1 - np.abs((h - c + 180) % 360 - 180) / 45, 0, 1)
    return w / np.maximum(w.sum(-1, keepdims=True), 1)


def curve_lut(points, n: int = 1024) -> np.ndarray:
    """Монотонная кубическая кривая (Фритч — Карлсон) по точкам 0..255: без выбросов и ступенек."""
    pts = np.array(sorted(points), np.float64) / 255.0
    x, y = pts[:, 0], pts[:, 1]
    xs = np.linspace(0, 1, n)
    if len(x) < 3:
        return np.interp(xs, x, y).astype(np.float32)
    d = np.diff(y) / np.diff(x)
    m = np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2, [d[-1]]])
    for i, di in enumerate(d):
        if di == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / di, m[i + 1] / di
            r = a * a + b * b
            if r > 9:
                t = 3 / np.sqrt(r)
                m[i], m[i + 1] = t * a * di, t * b * di
    k = np.clip(np.searchsorted(x, xs) - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]
    t = (xs - x[k]) / h
    out = ((2 * t ** 3 - 3 * t ** 2 + 1) * y[k] + (t ** 3 - 2 * t ** 2 + t) * h * m[k]
           + (-2 * t ** 3 + 3 * t ** 2) * y[k + 1] + (t ** 3 - t ** 2) * h * m[k + 1])
    return np.clip(out, 0, 1).astype(np.float32)


def _apply_curve(ch: np.ndarray, points) -> np.ndarray:
    lut = curve_lut(points)
    return lut[(np.clip(ch, 0, 1) * (len(lut) - 1) + 0.5).astype(np.int32)]


def _hue_rgb(hue: float) -> np.ndarray:
    """Направление сдвига цвета для тонирования: чистый оттенок без изменения яркости."""
    rgb = cv2.cvtColor(np.array([[[hue, 1.0, 1.0]]], np.float32), cv2.COLOR_HSV2RGB)[0, 0]
    return rgb - rgb @ LUM


def _apply_hsl(img: np.ndarray, hsl: dict) -> np.ndarray:
    # Сдвиги зависят только от оттенка: считаем таблицу на 360° и берём из неё по пикселям.
    deg = np.arange(360, dtype=np.float32)
    tab = np.zeros((3, 360), np.float32)
    total = np.zeros(360, np.float32)
    for name, (sh, ss, sl) in hsl.items():
        w = np.clip(1 - np.abs((deg - HSL_CENTERS[name] + 180) % 360 - 180) / 45, 0, 1)
        total += w
        tab += w * np.array([[sh * 0.3], [ss / 100], [sl / 100]], np.float32)  # оттенок ±100 → ±30°
    tab /= np.maximum(total, 1)  # в зазорах между цветами влияние плавно слабеет
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    i = h.astype(np.int32) % 360
    hsv[..., 0] = (h + tab[0][i]) % 360
    hsv[..., 1] = np.clip(s * (1 + tab[1][i]), 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out * (1 + 0.5 * tab[2][i] * s)[..., None]  # серые пиксели яркость не меняют


def apply_look(img: np.ndarray, look: dict, strength: float = 1.0) -> np.ndarray:
    """Применяет пресет-образ; strength 0..1 плавно смешивает с исходником."""
    src = np.clip(img, 0, 1).astype(np.float32)
    out = src.copy()
    t, m = (v / 100 for v in look.get("wb", (0, 0)))
    ev = look.get("exposure", 0)
    if t or m or ev:
        gain = np.array([1 + 0.25 * t, 1 - 0.2 * m, 1 - 0.25 * t], np.float32) * np.float32(2 ** ev)
        out = np.clip(out * gain ** (1 / 2.2), 0, 1)  # = (x^2.2 · gain)^(1/2.2), без возведения пикселей
    if look.get("hsl"):
        out = np.clip(_apply_hsl(out, look["hsl"]), 0, 1)
    vib, sat = look.get("vib", 0) / 100, look.get("sat", 0) / 100
    if vib or sat:
        gray = (out @ LUM)[..., None]
        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        chroma = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b))[..., None]  # быстрее max(-1)
        out = np.clip(gray + (out - gray) * (1 + sat) * (1 + vib * (1 - np.clip(chroma * 1.5, 0, 1))), 0, 1)
    grade = look.get("grade")
    if grade:
        # Сдвиг цвета зависит только от яркости: таблица на 1024 уровня, потом выборка по пикселям.
        lv = np.linspace(0, 1, 1024, dtype=np.float32)
        bal = grade.get("balance", 0) / 200  # сдвигает границу теней и светов
        zones = {"shadows": np.clip(1 - lv / (0.5 + bal), 0, 1) ** 1.5,
                 "highlights": np.clip((lv - 0.5 - bal) / (0.5 - bal), 0, 1) ** 1.5,
                 "mid": 1 - np.abs(2 * lv - 1)}
        tab = np.zeros((1024, 3), np.float32)
        for zone, w in zones.items():
            if zone in grade and grade[zone][1]:
                hue, amount = grade[zone]
                tab += w[:, None] * (0.35 * amount / 100) * _hue_rgb(hue)
        out = np.clip(out + tab[(np.clip(out @ LUM, 0, 1) * 1023 + 0.5).astype(np.int32)], 0, 1)
    if look.get("curve"):
        out = _apply_curve(out, look["curve"])
    for i, ch in enumerate("rgb"):
        if look.get(ch):
            out[..., i] = _apply_curve(out[..., i], look[ch])
    return src + (out - src) * strength


def load_looks(folder) -> dict:
    """Все пресеты-образы из папки: {название: рецепт}, в порядке файлов."""
    looks = {}
    for f in sorted(Path(folder).glob("*.json")):
        data = read_json(f, [])
        for look in data if isinstance(data, list) else [data]:
            if isinstance(look, dict) and look.get("name"):
                looks[look["name"]] = look
    return looks


# ---------------------------------------------------------------- обработка

def _smooth(ch: np.ndarray, sigma: float) -> np.ndarray:
    """Широкое размытие через уменьшенную копию, чтобы было быстро."""
    h, w = ch.shape
    f = min(1.0, 6.0 / sigma)
    if f < 1:
        small = cv2.resize(ch, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma * f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(ch, (0, 0), sigma)


def process(img: np.ndarray, p: dict, style: dict | None = None,
            src_stats: dict | None = None, local: bool = True,
            frame: tuple | None = None, look: dict | None = None) -> np.ndarray:
    """Применяет все правки. local=False — только поточечные операции (для LUT).
    frame=(ширина, высота, x0, y0) — img вырезан из кадра такого размера: размытия и виньетка
    считаются как для целого кадра (просмотр в масштабе). look — пресет-образ, его сила в p["look_strength"]."""
    g = lambda k: p.get(k, 0) / 100.0
    img = np.maximum(img.astype(np.float32, copy=True), 0)
    h, w = img.shape[:2]
    fw, fh, x0, y0 = frame or (w, h, 0, 0)

    # Баланс белого и экспозиция в линейном свете: (x^2.2 · k)^(1/2.2) = x · k^(1/2.2), без возведения пикселей.
    t, m = g("temperature"), g("tint")
    if t or m or g("exposure"):
        gain = np.array([1 + 0.25 * t, 1 - 0.2 * m, 1 - 0.25 * t], np.float32) * np.float32(2 ** g("exposure"))
        img *= np.maximum(gain, 0) ** (1 / 2.2)

    # Света и тени: маска по сглаженной яркости, чтобы не терять локальный контраст.
    sh, hl = g("shadows"), g("highlights")
    if sh or hl:
        lum = np.clip(img @ LUM, 0, 1)
        base = _smooth(lum, fw / 40) if local else lum
        base = np.clip(base, 0, 1)
        gain = 2 ** (1.3 * sh * (1 - base) ** 2 + 0.9 * hl * base ** 2)
        img *= gain[..., None].astype(np.float32)
    img = np.clip(img, 0, 1)

    # Контраст: S-кривая через smoothstep.
    c = g("contrast")
    if c > 1:  # сверх прежнего максимума — вторая S-кривая, без переломов в тенях
        img = img * img * (3 - 2 * img)
        c -= 1
    if c:
        img = np.clip(img + c * (img * img * (3 - 2 * img) - img), 0, 1)

    # Чёткость: локальный контраст средних частот.
    cl = g("clarity")
    if cl and local:
        lum = img @ LUM
        detail = lum - _smooth(lum, fw / 60)
        mid = 1 - (2 * lum - 1) ** 2  # сильнее в полутонах
        img += (0.8 * cl * detail * mid)[..., None]

    # Сочность и насыщенность.
    vib, sat = g("vibrance"), g("saturation")
    if vib or sat:
        gray = (img @ LUM)[..., None]
        chroma = img.max(-1, keepdims=True) - img.min(-1, keepdims=True)
        k = (1 + sat) * (1 + vib * (1 - np.clip(chroma * 1.5, 0, 1)))
        img = gray + (img - gray) * k

    # Тональная кривая, затем HSL — как в Lightroom; обе поточечные, поэтому входят и в LUT.
    curves = {ch: pts for ch, pts in (p.get("curve") or {}).items() if pts}
    if curves:
        img = np.clip(img, 0, 1)
        if "rgb" in curves:
            img = _apply_curve(img, curves["rgb"])
        for i, ch in enumerate("rgb"):
            if ch in curves:
                img[..., i] = _apply_curve(img[..., i], curves[ch])
    hsl = {c: v for c, v in (p.get("hsl") or {}).items() if any(v)}
    if hsl:
        img = _apply_hsl(np.clip(img, 0, 1).astype(np.float32), hsl)

    grade = p.get("ai_grade") or {}
    g_amount = grade.get("strength", 80) / 100 if grade else 0.0
    if grade.get("recipe") and g_amount > 0:  # ИИ-цветокоррекция: рецепт поверх ваших правок, до стиля
        img = apply_look(img, grade["recipe"], g_amount)

    if style:
        img = _apply_style(np.clip(img, 0, 1), style, p, src_stats)

    if look:  # образ — финальный цвет, поверх стиля; виньетка и резкость уже после него
        img = apply_look(img, look, p.get("look_strength", 100) / 100.0)

    layers = list(p.get("masks") or [])
    if g_amount > 0:  # объектные правки ИИ-цветокоррекции — те же маски, с силой варианта
        layers += [{**lay, "amount": g_amount} for lay in grade.get("masks", [])]
    if local and layers:  # маски пространственные: в LUT не попадают
        img = apply_masks(np.clip(img, 0, 1), layers, (fw, fh, x0, y0))

    if local:  # ИИ-ретушь — до виньетки и резкости: резкость не должна подчёркивать разглаженное
        ai = p.get("ai_arr") or {}
        if g("retouch") > 0 and ai.get("people") is not None:
            img = skin_smooth(np.clip(img, 0, 1), g("retouch"), ai["people"], (fw, fh, x0, y0))
        if g("bokeh") > 0 and ai.get("subject") is not None:
            img = lens_blur(np.clip(img, 0, 1), g("bokeh"), ai["subject"], (fw, fh, x0, y0))

    if local:
        v = g("vignette")
        if v:
            xx = ((x0 + np.arange(w, dtype=np.float32) + 0.5) / fw * 2 - 1)[None, :]
            yy = ((y0 + np.arange(h, dtype=np.float32) + 0.5) / fh * 2 - 1)[:, None]
            d2 = (xx ** 2 + yy ** 2) / 2
            img *= (1 + 0.9 * v * d2 ** 1.5)[..., None]
        s = g("sharpness")
        if s:
            img = np.clip(img, 0, 1)
            blur = cv2.GaussianBlur(img, (0, 0), max(0.7, fw / 3000))
            img += 1.2 * s * (img - blur)

    return np.clip(img, 0, 1).astype(np.float32)


def apply_masks(img: np.ndarray, layers: list, frame: tuple) -> np.ndarray:
    """Локальные правки: для каждого слоя — тот же process с ползунками слоя, смешанный по маске."""
    h, w = img.shape[:2]
    for layer in layers:
        adj = {k: v for k, v in layer.get("adj", {}).items() if v}
        if not layer.get("on", True) or not adj:
            continue
        m = M.layer_mask(layer, h, w, frame)
        if m is None or m.max() < 1e-3:
            continue
        m = m * layer.get("amount", 1.0)
        if layer.get("protect_white"):  # выбитое в белый (солнце) не затемняем в грязно-серое пятно
            m = m * (1 - np.clip((img.max(-1) - 0.9) / 0.08, 0, 1))
        local = process(img, {**default_params(), **adj}, frame=frame)
        img = img + (local - img) * m[..., None]
    return img


def _ai_mask(arr: np.ndarray, h: int, w: int, frame: tuple) -> np.ndarray:
    """Маска ИИ всего кадра (uint8) → пиксели этой вырезки (превью, масштаб, полный кадр — одинаково)."""
    return M.layer_mask({"type": "ai", "arr": arr}, h, w, frame)


def _guided(img: np.ndarray, guide: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.): сглаживает внутри однородных областей, края (глаза, губы) держит.
    На боксовых фильтрах — время не зависит от радиуса, годится и для 24 Мп."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mg = box(guide)
    var = box(guide * guide) - mg * mg
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        p = img[..., c]
        mp = box(p)
        a = (box(guide * p) - mg * mp) / (var + eps)
        out[..., c] = box(a) * guide + box(mp - a * mg)
    return out


def skin_smooth(img: np.ndarray, amount: float, people: np.ndarray, frame: tuple) -> np.ndarray:
    """Ретушь кожи частотным разложением: пятна и неровности тона (средние частоты) сглаживаются,
    мелкая текстура пор возвращается — без «пластика». Только кожа людей: маска людей × цвет кожи."""
    h, w = img.shape[:2]
    fw = frame[0]
    m = _ai_mask(people, h, w, frame) * _skin_mask(cv2.cvtColor(img, cv2.COLOR_RGB2Lab))
    if m.max() < 1e-3:
        return img
    fine = img - cv2.GaussianBlur(img, (0, 0), max(0.6, fw / 3000))  # поры и волоски
    lum = img @ LUM
    smooth = _guided(img, lum, max(2, round(fw / 120)), 5e-3) + 0.6 * fine  # края контрастнее ~0.07 держатся
    return img + (smooth - img) * (m * min(amount, 1.0))[..., None]


def lens_blur(img: np.ndarray, amount: float, subject: np.ndarray, frame: tuple) -> np.ndarray:
    """Размытие фона, как у светосильного объектива. Фон размывается только своими пикселями
    (размытие, нормированное маской фона), поэтому цвет объекта не растекается ореолом за край."""
    h, w = img.shape[:2]
    fw = frame[0]
    sigma = min(amount, 1.0) * 0.012 * fw  # до 1.2% ширины кадра; поля process_region — 5%
    if sigma < 0.5 or subject.max() < 128:  # объекта нет (пустая маска всего кадра) — размывать нечего
        return img
    m = _ai_mask(subject, h, w, frame)
    bg = 1 - m
    k = max(1.0, sigma / 6)  # большое размытие — на уменьшенной копии: быстро и на 24 Мп
    size = (max(1, round(w / k)), max(1, round(h / k)))
    blur = lambda x: cv2.resize(cv2.GaussianBlur(cv2.resize(x, size, interpolation=cv2.INTER_AREA), (0, 0), sigma / k),
                                (w, h), interpolation=cv2.INTER_LINEAR)
    wsum = blur(bg)
    back = blur(img * bg[..., None]) / np.maximum(wsum, 1e-3)[..., None]
    back = np.where(wsum[..., None] > 1e-3, back, img)  # глубоко внутри объекта фона нет — там и не нужен
    return img * m[..., None] + back * bg[..., None]


def style_source_stats(img: np.ndarray, p: dict) -> dict:
    """Статистики кадра перед стилем: нужны, чтобы вырезка в масштабе красилась как весь кадр."""
    return lab_stats(process(img, {**p, "vignette": 0, "sharpness": 0}))


# ---------------------------------------------------------------- обрезка и горизонт
# Последний шаг после всех правок: кадр поворачивается вокруг центра (холст того же размера),
# затем вырезается crop — прямоугольник в долях повёрнутого холста. Маски и прочие правки живут
# в координатах полного кадра и об обрезке не знают.

def crop_matrix(W: float, H: float, crop, angle: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Аффинная матрица 2×3: точка полного кадра (пиксели W×H) → точка результата; и размер результата."""
    x0, y0, x1, y1 = crop or (0, 0, 1, 1)
    a = np.radians(angle or 0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([W / 2, H / 2])
    t = c - R @ c - np.array([x0 * W, y0 * H])
    return np.hstack([R, t[:, None]]), ((x1 - x0) * W, (y1 - y0) * H)


def crop_valid(W: float, H: float, crop, angle: float, eps: float = 1e-3) -> bool:
    """Все углы обрезки лежат на снимке (после поворота по углам холста — пустота)."""
    x0, y0, x1, y1 = crop
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return False
    a = np.radians(angle or 0)
    Rinv = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    c = np.array([W / 2, H / 2])
    pts = np.array([[x0 * W, y0 * H], [x1 * W, y0 * H], [x1 * W, y1 * H], [x0 * W, y1 * H]]) - c
    src = pts @ Rinv.T + c
    return bool((src[:, 0] >= -eps * W).all() and (src[:, 0] <= W * (1 + eps)).all()
                and (src[:, 1] >= -eps * H).all() and (src[:, 1] <= H * (1 + eps)).all())


def fit_crop(W: float, H: float, crop, angle: float) -> list[float]:
    """Сжимает обрезку к её центру, пока она не уляжется на повёрнутый снимок."""
    x0, y0, x1, y1 = crop
    if crop_valid(W, H, crop, angle):
        return list(crop)
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    if not crop_valid(W, H, (cx - 0.005, cy - 0.005, cx + 0.005, cy + 0.005), angle):
        cx, cy = 0.5, 0.5  # центр вне снимка — начинаем из центра кадра
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = crop_valid(W, H, (cx - hw * mid, cy - hh * mid, cx + hw * mid, cy + hh * mid), angle)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return [cx - hw * lo, cy - hh * lo, cx + hw * lo, cy + hh * lo]


def aspect_crop(W: float, H: float, aspect: float, angle: float = 0.0, around=None) -> list[float]:
    """Наибольшая обрезка с соотношением aspect (ширина/высота в пикселях), вписанная в снимок."""
    cx, cy = around or (0.5, 0.5)
    if aspect >= W / H:
        hw, hh = 0.5, 0.5 * (W / aspect) / H
    else:
        hw, hh = 0.5 * (H * aspect) / W, 0.5
    return fit_crop(W, H, [cx - hw, cy - hh, cx + hw, cy + hh], angle)


def has_geometry(p: dict) -> bool:
    return bool(p.get("crop")) or bool(p.get("angle"))


def apply_crop(img: np.ndarray, crop, angle: float, full_size=None) -> np.ndarray:
    """Поворот и обрезка. img может быть уменьшенной копией кадра full_size (превью)."""
    if not crop and not angle:
        return img
    h, w = img.shape[:2]
    W, H = full_size or (w, h)
    k = w / W
    A, (ow, oh) = crop_matrix(W, H, crop, angle)
    A = A.copy()
    A[:, 2] *= k  # те же координаты, но в пикселях уменьшенной копии
    return cv2.warpAffine(img, A.astype(np.float32), (max(1, round(ow * k)), max(1, round(oh * k))),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- ИИ-композиция: горизонт и трети

def auto_horizon(img: np.ndarray, sky: np.ndarray | None = None) -> float | None:
    """Угол поворота (для params["angle"]), выравнивающий горизонт и вертикали. None — нет уверенности.
    Линии Хафа: почти горизонтальные (±12°) и почти вертикальные (здания), вес — длина; граница неба
    из ИИ-маски — сильный дополнительный голос. Решение — взвешенная медиана при малом разбросе."""
    small = resize_max(np.clip(img, 0, 1), 1000)
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur((cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8), (0, 0), 1.5)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=max(w, h) // 8, maxLineGap=8)
    angles, weights = [], []
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):  # OpenCV 4: (N,1,4), 5: (N,4)
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 90) % 180 - 90                      # направление линии в (−90, 90]
        length = np.hypot(x2 - x1, y2 - y1)
        if abs(a) <= 12:
            angles.append(a)
            weights.append(length)
        elif abs(abs(a) - 90) <= 8:                  # вертикаль: наклон относительно отвеса, вес поменьше
            angles.append(a - 90 if a > 0 else a + 90)
            weights.append(length * 0.5)
    if sky is not None and sky.mean() > 0.03:
        m = cv2.resize(sky.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
        cols = np.arange(0, w, 4)
        ys = np.array([np.argmin(m[:, c]) if m[0, c] and not m[:, c].all() else -1 for c in cols])
        ok = ys > 0
        if ok.sum() > len(cols) * 0.5:
            k, _ = np.polyfit(cols[ok], ys[ok], 1)  # граница неба: наклон прямой
            angles.append(np.degrees(np.arctan(k)))
            weights.append(w * 1.5)
    if not angles or sum(weights) < max(w, h) * 0.4:
        return None
    a, wt = np.array(angles), np.array(weights)
    order = np.argsort(a)
    cum = np.cumsum(wt[order]) / wt.sum()
    med = float(a[order][np.searchsorted(cum, 0.5)])
    close = np.abs(a - med) < 1.5
    if wt[close].sum() < wt.sum() * 0.5:  # голоса разошлись — лучше ничего не делать
        return None
    tilt = float(np.average(a[close], weights=wt[close]))
    return 0.0 if abs(tilt) < 0.1 else round(-tilt, 1)  # линия вниз-вправо на a° → повернуть на −a°


def _saliency_map(img: np.ndarray) -> np.ndarray:
    """Карта заметности на копии 256 px: локальный контраст + насыщенность, лёгкий приоритет центра."""
    small = resize_max(np.clip(img, 0, 1), 256).astype(np.float32)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    blur = cv2.GaussianBlur(lab, (0, 0), 8)
    contrast = np.linalg.norm(lab - blur, axis=-1)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1] * 30
    yy, xx = np.mgrid[0:h, 0:w]
    prior = np.exp(-(((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.18))
    return cv2.GaussianBlur((contrast + sat) * (0.6 + 0.4 * prior), (0, 0), 6)


def saliency_point(img: np.ndarray) -> tuple[float, float]:
    """Самое заметное место кадра (доли)."""
    sal = _saliency_map(img)
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w]
    top = sal >= np.percentile(sal, 97)
    return float(xx[top].mean() / w), float(yy[top].mean() / h)


def main_people(people: np.ndarray, min_share: float = 0.015) -> np.ndarray | None:
    """Главные люди кадра: самый крупный силуэт и сопоставимые с ним (пара, семья) — без толпы за спиной.
    None — крупного человека нет (прохожие вдали — не объект съёмки)."""
    m = (people > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < min_share * m.size:
        return None
    keep = np.isin(lab, 1 + np.nonzero(areas >= areas.max() * 0.4)[0])
    # мягкий край модели сохраняем: берём исходную вероятность в чуть расширенной области главных
    grow = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return people * grow


def saliency_mask(img: np.ndarray) -> np.ndarray:
    """Главный объект, когда людей нет: самое заметное пятно кадра (связная область верхних 12%
    заметности с наибольшей суммой), дыры закрыты. Мягкая маска 0..1 размером с img."""
    sal = _saliency_map(img)
    w = sal.shape[1]
    top = (sal >= np.percentile(sal, 88)).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(top)
    if n < 2:
        return np.zeros(img.shape[:2], np.float32)
    best = 1 + int(np.argmax([sal[lab == i].sum() for i in range(1, n)]))
    m = (lab == best).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, w // 12) | 1,) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, w / 100))
    return cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def subject_point(people: np.ndarray | None) -> tuple[float, float] | None:
    """Точка главного человека (доли): верх его силуэта — там глаза, их и ставят на треть."""
    if people is None or people.mean() < 0.015:  # прохожий на 1% кадра — ещё не главный объект
        return None
    m = (people > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y, bh = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
    ys, xs = np.nonzero((lab == i)[y:y + max(1, bh // 4)])  # верхняя четверть силуэта, x — по всему кадру
    h, w = m.shape
    return float(xs.mean() / w), float((ys.mean() + y) / h)


def horizon_level(sky: np.ndarray | None) -> float | None:
    """Где кончается небо (доля высоты): медиана по столбцам, где верх кадра — небо."""
    if sky is None or sky.mean() < 0.05:
        return None
    m = sky > 0.5
    h, w = m.shape
    ys = [np.argmin(m[:, c]) for c in range(0, w, max(1, w // 200)) if m[0, c] and not m[:, c].all()]
    level = float(np.median(ys) / h) if len(ys) > 10 else None
    # ставить на треть имеет смысл, только если горизонт и так в средней части кадра
    return level if level is not None and 0.2 <= level <= 0.8 else None


def thirds_crop(W: float, H: float, aspect: float, angle: float, point: tuple[float, float],
                horizon: float | None = None, sure: bool = True) -> list[float]:
    """Обрезка пропорции aspect, ставящая point на ближайшее пересечение третей; без объекта — горизонт
    (доля высоты) на ближайшую треть. Размер рамки 85/75/65% от наибольшей: меньше — только если так
    объект встаёт на треть заметно точнее (обрезать лишнее не хочется). sure=False (объект угадан по
    заметности) — не меньше 75%: неуверенная догадка не должна отрезать полкадра."""
    base = aspect_crop(W, H, aspect, angle)
    px, py = point
    yref = horizon if horizon is not None else py
    best = None
    for scale in (0.85, 0.75, 0.65) if sure else (0.85, 0.75):
        cw, ch = (base[2] - base[0]) * scale, (base[3] - base[1]) * scale
        for tx in (1 / 3, 2 / 3):
            for ty in (1 / 3, 2 / 3):
                x0 = float(np.clip(px - tx * cw, 0, 1 - cw))
                y0 = float(np.clip(yref - ty * ch, 0, 1 - ch))
                err = abs(px - (x0 + tx * cw)) / cw + abs(yref - (y0 + ty * ch)) / ch + 0.3 * (0.85 - scale)
                if best is None or err < best[0] - 1e-9:
                    best = (err, [x0, y0, x0 + cw, y0 + ch])
    return fit_crop(W, H, best[1], angle)


def process_view_region(full: np.ndarray, view_rect: tuple, scale: float, p: dict,
                        style: dict | None = None, src_stats: dict | None = None,
                        look: dict | None = None, prep=None) -> tuple:
    """Видимая область обрезанного и повёрнутого кадра (просмотр в масштабе).
    view_rect=(x0, y0, x1, y1) в пикселях результата (полное разрешение). Обрабатывается охватывающий
    кусок полного кадра, затем поворачивается и вырезается той же матрицей, что и превью."""
    H, W = full.shape[:2]
    A, _ = crop_matrix(W, H, p.get("crop"), p.get("angle", 0))
    Ainv = cv2.invertAffineTransform(A)
    vx0, vy0, vx1, vy1 = view_rect
    corners = np.array([[vx0, vy0, 1], [vx1, vy0, 1], [vx1, vy1, 1], [vx0, vy1, 1]], np.float64) @ Ainv.T
    bx0, by0 = np.floor(corners.min(0)) - 2
    bx1, by1 = np.ceil(corners.max(0)) + 2
    box = (int(max(0, bx0)), int(max(0, by0)), int(min(W, bx1)), int(min(H, by1)))
    before, after = process_region(full, box, scale, p, style, src_stats, look, prep)
    if not has_geometry(p):
        return before, after
    k = after.shape[1] / (box[2] - box[0])
    b0 = np.array(box[:2], np.float64)
    M = np.hstack([A[:, :2], (k * (A[:, :2] @ b0 + A[:, 2] - np.array([vx0, vy0])))[:, None]]).astype(np.float32)
    size = (max(1, round((vx1 - vx0) * k)), max(1, round((vy1 - vy0) * k)))
    warp = lambda x: cv2.warpAffine(x, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warp(before), warp(after)


def process_region(full: np.ndarray, rect: tuple, scale: float, p: dict,
                   style: dict | None = None, src_stats: dict | None = None,
                   look: dict | None = None, prep=None) -> tuple:
    """Обрабатывает только видимую область кадра (просмотр в масштабе).
    rect=(x0, y0, x1, y1) в пикселях full; scale ≤ 1 — уменьшение перед обработкой.
    Возвращает (исходник, результат) области. Вокруг берутся поля под широкие размытия.
    prep(crop, (ix0, iy0, ix1, iy1)) — подготовка вырезки до правок (шумодав): видимая часть внутри полей."""
    H, W = full.shape[:2]
    x0, y0, x1, y1 = rect
    m = W // 20  # 2σ размытия теней/светов
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    crop = full[Y0:Y1, X0:X1]
    if scale < 1:
        crop = cv2.resize(crop, (max(1, round((X1 - X0) * scale)), max(1, round((Y1 - Y0) * scale))),
                          interpolation=cv2.INTER_AREA)
    k = crop.shape[1] / (X1 - X0)
    ix0, iy0 = round((x0 - X0) * k), round((y0 - Y0) * k)
    ix1, iy1 = ix0 + max(1, round((x1 - x0) * k)), iy0 + max(1, round((y1 - y0) * k))
    src = prep(crop, (ix0, iy0, ix1, iy1)) if prep else crop
    out = process(src, p, style, src_stats, frame=(W * k, H * k, X0 * k, Y0 * k), look=look)
    return crop[iy0:iy1, ix0:ix1], out[iy0:iy1, ix0:ix1]


# ---------------------------------------------------------------- единый цвет серии

# Что подгоняется под эталон: шаг для числовой производной и предел сдвига от стартового значения.
MATCH_VARS = {"exposure": (20, 150), "temperature": (10, 60), "tint": (10, 60), "contrast": (10, 25),
              "saturation": (10, 25)}
PER_FRAME = ("crop", "angle", "masks", "denoise", "ai_arr")  # своё у каждого кадра — эталон не копирует


def match_stats(img: np.ndarray) -> dict:
    """Что должно совпасть у кадров серии: яркость (медиана, в EV), разброс L*, средняя хрома и
    средний цвет полутонов (a*, b*). Хрома при усреднении цвета обрезана на 40: яркий объект
    (красная куртка) не тянет баланс. Нейтрали не подходят — на закатах их почти нет, и их набор
    скачет от каждого сдвига баланса, подгонка разваливалась."""
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    mid = (L > 25) & (L < 85)
    if mid.mean() < 0.05:  # почти весь кадр в тенях или в светах — цвет по всему кадру
        mid = np.ones_like(mid)
    k = np.minimum(1, 40 / np.maximum(C[mid], 1e-6))
    Y = ((np.median(L) + 16) / 116) ** 3
    return {"ev": float(np.log2(max(Y, 1e-4))), "Lstd": float(L.std()), "C": float(C.mean()),
            "a": float((a[mid] * k).mean()), "b": float((b[mid] * k).mean())}


def match_params(img: np.ndarray, ref_params: dict, ref_stats: dict, own: dict | None = None,
                 style: dict | None = None, look: dict | None = None, ref_auto: dict | None = None) -> dict:
    """Правки кадра, при которых он выглядит как эталон серии: творческие настройки эталона
    (стиль, пресет, HSL, кривая, ИИ-цвет без масок) + подгонка экспозиции, баланса, контраста и
    насыщенности по match_stats. Кадр другой сцены не уезжает: сдвиги ограничены MATCH_VARS.
    ref_auto — auto_params эталона: старт от разницы авто-тона кадра и эталона (ближе к ответу)."""
    own = own or {}
    p = {k: v for k, v in ref_params.items() if k not in PER_FRAME}
    p.update({k: own[k] for k in PER_FRAME if k in own})
    if p.get("ai_grade"):
        p["ai_grade"] = {**p["ai_grade"], "masks": []}  # маски объектов — чужого кадра
    small = resize_max(np.clip(img, 0, 1), 400)
    limits = {key: (lo, hi) for key, _, lo, hi, _ in SLIDERS}
    if ref_auto is not None:
        mine = auto_params(img)
        for k in ("exposure", "temperature", "tint"):
            p[k] = p.get(k, 0) + round(mine[k] - ref_auto[k])
    # Каждый рычаг правит «свою» меру: так подгонка устойчива (совместное решение уводило оттенок
    # и насыщенность в пределы, компенсируя одно другим).
    pairs = {"exposure": "ev", "temperature": "b", "tint": "a", "contrast": "Lstd", "saturation": "C"}
    base = {k: p.get(k, 0) for k in pairs}
    ev100 = lambda st, k: st[k] * (100 if k == "ev" else 1)
    for _ in range(5):
        cur = match_stats(process(small, p, style, local=False, look=look))
        for k, stat in pairs.items():
            d = MATCH_VARS[k][0]
            m = match_stats(process(small, {**p, k: p.get(k, 0) + d}, style, local=False, look=look))
            slope = (ev100(m, stat) - ev100(cur, stat)) / d
            if abs(slope) * MATCH_VARS[k][1] < 2:  # даже весь допустимый сдвиг почти ничего не даёт —
                continue                             # это другая сцена, рычаг не дёргаем в предел зря
            damp = 0.4 if k in ("contrast", "saturation") else 0.8  # разброс и хрома больше зависят от сюжета
            step = damp * (ev100(ref_stats, stat) - ev100(cur, stat)) / slope
            lim = MATCH_VARS[k][1]
            v = np.clip(p.get(k, 0) + step, base[k] - lim, base[k] + lim)
            p[k] = round(float(np.clip(v, *limits[k])))
    return p


# ---------------------------------------------------------------- авто

def auto_params(img: np.ndarray) -> dict:
    """Подбирает тон и баланс белого под конкретный кадр."""
    small = np.clip(resize_max(img, 512), 0, 1)
    lum = small @ LUM
    lin = np.clip(lum, 1e-4, 1) ** 2.2
    ev = float(np.clip(np.log2(0.18 / np.median(lin)), -2.0, 2.5)) * 0.85
    after = np.clip((lin * 2 ** ev) ** (1 / 2.2), 0, 1)

    clipped = float((after > 0.97).mean())
    highlights = -min(70.0, clipped * 900 + (20 if np.percentile(after, 99) > 0.9 else 0))
    shadows = min(55.0, float((after < 0.12).mean()) * 160)
    spread = float(np.percentile(after, 95) - np.percentile(after, 5))
    contrast = float(np.clip((0.75 - spread) * 80, -10, 30))

    # Баланс белого камеры уже учтён, поэтому правим мягко: закат должен остаться тёплым.
    mids = small[(lum > 0.2) & (lum < 0.85)]
    temperature = tint = 0.0
    if len(mids) > 100:
        r, gch, b = mids.mean(0)
        temperature = float(np.clip((b - r) / (gch + 1e-6) * 40, -12, 12))
        tint = float(np.clip((gch - (r + b) / 2) / (gch + 1e-6) * 50, -8, 8))

    return {"exposure": round(ev * 100), "contrast": round(contrast),
            "highlights": round(highlights), "shadows": round(shadows),
            "temperature": round(temperature), "tint": round(tint)}


def scene_preset(img: np.ndarray, scene: dict) -> dict:
    """Параметры под сцену: авто-тон кадра (подбор) + поправки сцены (вкус) + её пресет-образ."""
    p = auto_params(img)
    limits = {key: (lo, hi) for key, _, lo, hi, _ in SLIDERS}
    for key, delta in scene.get("params", {}).items():
        lo, hi = limits[key]
        p[key] = int(np.clip(p.get(key, 0) + delta, lo, hi))
    if scene.get("look"):
        p.update(look=scene["look"], look_strength=scene.get("look_strength", 80))
    return p


# ---------------------------------------------------------------- LUT

def export_cube(path, p: dict, style: dict | None, src_stats: dict | None, size: int = 33,
                look: dict | None = None) -> None:
    """3D LUT с поточечными правками и стилем: для Lightroom, DaVinci, телефона."""
    axis = np.linspace(0, 1, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")  # в .cube R меняется быстрее всех
    grid = np.stack([r, g, b], -1).reshape(-1, 1, 3)
    out = process(grid, p, style, src_stats=src_stats, local=False, look=look).reshape(-1, 3)
    lines = [f'TITLE "{Path(path).stem}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in out]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- экспорт (в отдельном процессе)

def export_one(job: dict) -> str:
    img = load_image(job["src"], half=False)
    params = dict(job["params"])
    if params.get("denoise") or job.get("upscale"):
        from . import enhance  # torch грузится только в процессе, которому он нужен
        if not enhance.available():
            raise RuntimeError("шумодав и увеличение требуют библиотек ИИ (install_ai.bat)")
    if params.get("denoise"):
        img = enhance.denoise(img, params["denoise"] / 100)
    if job.get("scene"):
        params.update(scene_preset(img, job["scene"]))  # авто-тон + пресет сцены кадра
    elif job.get("auto"):
        params.update(auto_params(img))  # тон — под кадр, стиль и цвет — ваши
    out = process(img, params, job.get("style"), look=job.get("look"))
    out = apply_crop(out, params.get("crop"), params.get("angle", 0))
    if job.get("long_edge"):
        out = resize_max(out, int(job["long_edge"]))
    if job.get("upscale"):
        out = enhance.upscale(out, int(job["upscale"]))
    save_jpeg(job["dst"], out, job.get("quality", 92))
    return job["dst"]


def read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def write_json(path, data) -> None:
    """Запись через временный файл и замену одним шагом: сбой посреди записи не портит правки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
