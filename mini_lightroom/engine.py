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
]


def default_params() -> dict:
    p = {key: 0 for key, *_ in SLIDERS}
    p.update(style="", style_strength=70)
    return p


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


def load_thumb(path, size: int = 512) -> np.ndarray:
    """Быстрая миниатюра uint8 RGB: для RAW берём встроенный в файл JPEG."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        with rawpy.imread(io.BytesIO(_read_bytes(path))) as raw:
            flip = getattr(raw.sizes, "flip", 0)
            try:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    bgr = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
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

def lab_stats(img: np.ndarray) -> dict:
    lab = cv2.cvtColor(np.clip(resize_max(img, 512), 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    flat = lab.reshape(-1, 3)
    return {"mean": flat.mean(0).tolist(), "std": flat.std(0).tolist()}


def style_from_image(path, name: str) -> dict:
    img = load_image(path, half=True, max_side=1024)
    return {"name": name, **lab_stats(img)}


def _apply_style(img, style, strength, src_stats):
    src = src_stats or lab_stats(img)
    s_mean, s_std = np.array(src["mean"], np.float32), np.array(src["std"], np.float32)
    r_mean, r_std = np.array(style["mean"], np.float32), np.array(style["std"], np.float32)
    ratio = np.clip(r_std / (s_std + 1e-6), 0.5, 2.0)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    lab = (lab - s_mean) * ratio + r_mean
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return img + (out - img) * strength


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
            frame: tuple | None = None) -> np.ndarray:
    """Применяет все правки. local=False — только поточечные операции (для LUT).
    frame=(ширина, высота, x0, y0) — img вырезан из кадра такого размера: размытия и виньетка
    считаются как для целого кадра (просмотр в масштабе)."""
    g = lambda k: p.get(k, 0) / 100.0
    img = np.maximum(img.astype(np.float32, copy=True), 0)
    h, w = img.shape[:2]
    fw, fh, x0, y0 = frame or (w, h, 0, 0)

    # Баланс белого и экспозиция в линейном свете.
    lin = img ** 2.2
    t, m = g("temperature"), g("tint")
    lin *= np.array([1 + 0.25 * t, 1 - 0.2 * m, 1 - 0.25 * t], np.float32) * np.float32(2 ** g("exposure"))
    img = lin ** (1 / 2.2)

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

    if style:
        img = _apply_style(np.clip(img, 0, 1), style, p.get("style_strength", 70) / 100.0, src_stats)

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


def style_source_stats(img: np.ndarray, p: dict) -> dict:
    """Статистики кадра перед стилем: нужны, чтобы вырезка в масштабе красилась как весь кадр."""
    return lab_stats(process(img, {**p, "vignette": 0, "sharpness": 0}))


def process_region(full: np.ndarray, rect: tuple, scale: float, p: dict,
                   style: dict | None = None, src_stats: dict | None = None) -> tuple:
    """Обрабатывает только видимую область кадра (просмотр в масштабе).
    rect=(x0, y0, x1, y1) в пикселях full; scale ≤ 1 — уменьшение перед обработкой.
    Возвращает (исходник, результат) области. Вокруг берутся поля под широкие размытия."""
    H, W = full.shape[:2]
    x0, y0, x1, y1 = rect
    m = W // 20  # 2σ размытия теней/светов
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    crop = full[Y0:Y1, X0:X1]
    if scale < 1:
        crop = cv2.resize(crop, (max(1, round((X1 - X0) * scale)), max(1, round((Y1 - Y0) * scale))),
                          interpolation=cv2.INTER_AREA)
    k = crop.shape[1] / (X1 - X0)
    out = process(crop, p, style, src_stats, frame=(W * k, H * k, X0 * k, Y0 * k))
    ix0, iy0 = round((x0 - X0) * k), round((y0 - Y0) * k)
    ix1, iy1 = ix0 + max(1, round((x1 - x0) * k)), iy0 + max(1, round((y1 - y0) * k))
    return crop[iy0:iy1, ix0:ix1], out[iy0:iy1, ix0:ix1]


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


# ---------------------------------------------------------------- LUT

def export_cube(path, p: dict, style: dict | None, src_stats: dict | None, size: int = 33) -> None:
    """3D LUT с поточечными правками и стилем: для Lightroom, DaVinci, телефона."""
    axis = np.linspace(0, 1, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")  # в .cube R меняется быстрее всех
    grid = np.stack([r, g, b], -1).reshape(-1, 1, 3)
    out = process(grid, p, style, src_stats=src_stats, local=False).reshape(-1, 3)
    lines = [f'TITLE "{Path(path).stem}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in out]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- экспорт (в отдельном процессе)

def export_one(job: dict) -> str:
    img = load_image(job["src"], half=False)
    params = dict(job["params"])
    if job.get("auto"):
        params.update(auto_params(img))  # тон — под кадр, стиль и цвет — ваши
    out = process(img, params, job.get("style"))
    if job.get("long_edge"):
        out = resize_max(out, int(job["long_edge"]))
    save_jpeg(job["dst"], out, job.get("quality", 92))
    return job["dst"]


def read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def write_json(path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
