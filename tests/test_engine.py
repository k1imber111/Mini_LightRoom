"""Быстрая проверка движка без реальных фото: python tests\\test_engine.py"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_lightroom import engine as E  # noqa: E402

rng = np.random.default_rng(0)
img = np.clip(rng.random((1000, 1500, 3), dtype=np.float32) * 0.4 + np.linspace(0, 0.5, 1500, dtype=np.float32)[None, :, None], 0, 1)

# Нейтральные параметры не должны менять картинку.
assert np.abs(E.process(img, E.default_params()) - img).max() < 1e-3, "нейтраль меняет кадр"

p = E.default_params()
p.update(exposure=50, contrast=30, highlights=-40, shadows=40, clarity=30,
         temperature=20, tint=-10, vibrance=30, saturation=10, sharpness=40, vignette=-30)
style = {"name": "t", **E.lab_stats(img[..., ::-1])}
t = time.perf_counter()
out = E.process(img, p, style)
print(f"обработка 1.5 Мп: {(time.perf_counter() - t) * 1000:.0f} мс")
assert out.shape == img.shape and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1

# Вырезка для просмотра в масштабе должна совпадать с тем же местом целого кадра.
p2 = {**p, "exposure": 150, "contrast": 180, "shadows": 150, "vignette": -150}
full = E.process(img, p2, style, E.style_source_stats(img, p2))
_, part = E.process_region(img, (600, 300, 1000, 700), 1.0, p2, style, E.style_source_stats(img, p2))
diff = np.abs(part - full[300:700, 600:1000]).mean()
print(f"вырезка против целого кадра: {diff:.4f}")
assert part.shape == (400, 400, 3) and diff < 0.01, "вырезка отличается от кадра"
_, half = E.process_region(img, (600, 300, 1000, 700), 0.5, p2)
assert half.shape == (200, 200, 3)

# Расширенный контраст остаётся монотонным (без переломов в тенях).
ramp = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(3, 2)
assert (np.diff(E.process(ramp, {**E.default_params(), "contrast": 200})[0, :, 0]) >= -1e-6).all()

# Пресеты-образы: все встроенные рабочие, сила 0 — без изменений, 50% — ровно середина.
looks = E.load_looks(Path(__file__).resolve().parent.parent / "looks")
assert 20 <= len(looks) <= 30, len(looks)
small = img[:200, :300]
for name, look in looks.items():
    full_look = E.apply_look(small, look, 1.0)
    assert full_look.shape == small.shape and np.isfinite(full_look).all(), name
    assert 0 <= full_look.min() and full_look.max() <= 1 + 1e-5, name
    assert np.abs(E.apply_look(small, look, 0.0) - small).max() < 1e-6, name
    assert np.abs(E.apply_look(small, look, 0.5) - (small + full_look) / 2).max() < 1e-5, name
for look in looks.values():
    for key in ("curve", "r", "g", "b"):
        if key in look:
            assert (np.diff(E.curve_lut(look[key])) >= 0).all(), (look["name"], key)
t = time.perf_counter()
E.process(img, {**p, "look": "x"}, look=next(iter(looks.values())))
print(f"обработка с пресетом: {(time.perf_counter() - t) * 1000:.0f} мс")

print("авто:", E.auto_params(img * 0.3))
assert E.auto_params(img * 0.3)["exposure"] > 0, "тёмный кадр должен осветляться"

with tempfile.TemporaryDirectory(prefix="мини_") as d:
    cube = Path(d) / "стиль.cube"
    E.export_cube(cube, p, style, E.lab_stats(img))
    assert len(cube.read_text(encoding="utf-8").splitlines()) == 4 + 33 ** 3
    jpg = Path(d) / "кадр.jpg"
    E.save_jpeg(jpg, out)
    back = E.load_image(jpg)
    assert back.shape == img.shape
    print("резкость:", round(E.sharpness(E.load_thumb(jpg)), 1))
print("OK")
