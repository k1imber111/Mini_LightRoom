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

# Стили (этап 2): раздельные силы, точный режим, смесь, защита кожи, потолок сдвига.
import cv2  # noqa: E402

to_lab = lambda x: cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
warm = np.clip(img * np.array([1.3, 1.0, 0.6], np.float32) + 0.1, 0, 1)   # «референс»: тёплый и светлый
ref = {"name": "ref", **E.lab_stats(warm)}
base_lab = to_lab(img)
full_p = {**E.default_params(), "style_strength": 100, "style_tone": 100, "style_skin": 0}
for mode in (0, 1):
    got = to_lab(E.process(img, {**full_p, "style_mode": mode}, ref))
    before = np.abs(base_lab.mean((0, 1)) - np.array(ref["mean"])).sum()
    after = np.abs(got.mean((0, 1)) - np.array(ref["mean"])).sum()
    print(f"стиль, режим {mode}: расстояние до референса {before:.1f} → {after:.1f}")
    assert after < before * 0.5, "перенос не приблизил кадр к референсу"
only_color = to_lab(E.process(img, {**full_p, "style_tone": 0}, ref))
dl = np.abs(only_color[..., 0] - base_lab[..., 0])  # у краёв охвата RGB обрезка чуть трогает яркость
assert dl.mean() < 0.3 and np.percentile(dl, 99) < 1.5, "«только цвет» изменил яркость"
only_tone = to_lab(E.process(img, {**full_p, "style_strength": 0}, ref))
assert np.abs(only_tone[..., 1:] - base_lab[..., 1:]).mean() < 1.0, "«только тон» изменил цвет"
far = {"name": "far", "mean": [50, 90, -90], "std": [20, 5, 5]}   # нереально кислотный референс
shift = to_lab(E.process(img, full_p, far))[..., 1:] - base_lab[..., 1:]
assert np.abs(shift).mean() < 31, "сдвиг цвета не ограничен"  # просили ~90, потолок 30
assert E.mix_styles(ref, far, 0.0)["mean"] == ref["mean"] and E.mix_styles(ref, far, 1.0)["mean"] == far["mean"]
skin = np.full((50, 50, 3), [0.85, 0.62, 0.5], np.float32)          # тон кожи
cool = {"name": "cool", **E.lab_stats(np.full((50, 50, 3), [0.3, 0.5, 0.9], np.float32))}
d_skin = lambda k: np.abs(to_lab(E.process(skin, {**full_p, "style_skin": k}, cool, E.lab_stats(img)))
                          - to_lab(skin))[..., 1:].mean()
print(f"кожа: сдвиг без защиты {d_skin(0):.1f}, с защитой {d_skin(100):.1f}")
assert d_skin(100) < d_skin(0) * 0.3, "защита кожи не работает"
assert E.normalize_params({"style_strength": 56})["style_tone"] == 56, "старые правки меняют вид"
# Почти однородное небо в точном режиме не растягивается до всего диапазона референса (иначе — ореолы).
sky = np.clip(0.45 + 0.02 * np.linspace(-1, 1, 300, dtype=np.float32)[None, :, None] * np.ones((200, 1, 3)), 0, 1)
sky_out = to_lab(E.process(sky.astype(np.float32), {**full_p, "style_mode": 1}, ref))
k = sky_out[..., 0].std() / to_lab(sky)[..., 0].std()
print(f"небо в точном режиме: контраст ×{k:.2f}")
assert k <= 2.05, "точный перенос раздувает контраст однородного неба"
with tempfile.TemporaryDirectory() as d:
    E.export_cube(Path(d) / "s.cube", {**full_p, "style_mode": 1}, ref, E.lab_stats(img))

# Маски (этап 3): геометрия в долях кадра, одинаковая в превью, вырезке и полном размере.
from mini_lightroom import masks as MK  # noqa: E402

lin = MK.layer_mask(MK.new_layer("linear", a=[0.5, 0.0], b=[0.5, 0.5]), 100, 150)
assert lin[2, 75] > 0.99 and lin[60, 75] < 0.01, "линейный градиент: сила от a к b"
rad = MK.new_layer("radial", c=[0.5, 0.5], r=[0.2, 0.2], feather=30)
m = MK.layer_mask(rad, 100, 150)
assert m[50, 75] > 0.99 and m[5, 5] < 0.01 and MK.layer_mask({**rad, "invert": True}, 100, 150)[5, 5] > 0.99
br = MK.new_layer("brush", feather=0, strokes=[{"pts": [[0.1, 0.5], [0.9, 0.5]], "r": 0.05, "erase": False},
                                               {"pts": [[0.5, 0.5]], "r": 0.08, "erase": True}])
m = MK.layer_mask(br, 100, 150)
assert m[50, 30] > 0.9 and m[50, 75] < 0.1 and m[10, 30] < 0.01, "кисть: мазок и ластик"
ai = MK.new_layer("ai", cat="sky", arr=(np.arange(300)[None, :] > 150).repeat(200, 0).astype(np.uint8) * 255)
assert MK.layer_mask(ai, 100, 150)[50, 10] < 0.01 and MK.layer_mask(ai, 100, 150)[50, 140] > 0.99
layers = [{**MK.new_layer("linear", a=[0.5, 0], b=[0.5, 0.6]), "adj": {"exposure": -150, "saturation": 40}},
          {**rad, "adj": {"exposure": 120, "clarity": 50}},
          {**br, "feather": 60, "adj": {"temperature": 60, "sharpness": 50}},
          {**ai, "adj": {"tint": 40}}]
pm = {**E.default_params(), "masks": layers}
out_m = E.process(img, pm)
assert out_m[:100].mean() < img[:100].mean() - 0.05, "маска неба не затемнила верх"
full_m = E.process(img, pm)
_, part_m = E.process_region(img, (600, 300, 1000, 700), 1.0, pm)
dm = np.abs(part_m - full_m[300:700, 600:1000]).mean()
print(f"маски: вырезка против целого кадра {dm:.4f}")
assert dm < 0.01, "маски в увеличенной вырезке расходятся с превью"
assert np.abs(E.process(img, {**pm, "masks": [{**layers[0], "on": False}]}) - E.process(img, E.default_params())).max() < 1e-5
t = time.perf_counter()
E.process(img, pm)
print(f"обработка с 4 масками: {(time.perf_counter() - t) * 1000:.0f} мс")

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

# Сцены: у каждой есть описания для CLIP и существующий пресет, поправки не выходят за шкалы.
scenes = E.read_json(Path(__file__).resolve().parent.parent / "scenes.json", [])
limits = {key: (lo, hi) for key, _, lo, hi, _ in E.SLIDERS}
for sc in scenes:
    assert sc["prompts"] and sc["look"] in looks, sc["id"]
    sp = E.scene_preset(img * 0.3, sc)
    assert sp["look"] == sc["look"] and all(limits[k][0] <= sp[k] <= limits[k][1] for k in sc["params"]), sc["id"]

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
