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

# Тональная кривая и HSL: поточечные, после базовых ползунков, входят в LUT.
mid = np.full((20, 20, 3), 0.5, np.float32)
assert E.process(mid, {**E.default_params(), "curve": {"rgb": [[0, 0], [128, 170], [255, 255]]}}).mean() > 0.6
only_b = E.process(mid, {**E.default_params(), "curve": {"b": [[0, 0], [128, 60], [255, 255]]}})
assert abs(only_b[..., 0].mean() - 0.5) < 0.01 and only_b[..., 2].mean() < 0.3, "кривая B задела другие каналы"
patch = np.zeros((10, 20, 3), np.float32)
patch[:, :10] = [0.2, 0.7, 0.2]    # зелёный
patch[:, 10:] = [0.8, 0.15, 0.15]  # красный
hs = E.process(patch, {**E.default_params(), "hsl": {"green": [0, -100, 0]}})
assert np.ptp(hs[5, 3]) < 0.02 and np.abs(hs[:, 10:] - patch[:, 10:]).max() < 0.01, "HSL задел не тот цвет"
assert np.abs(E.process(img, {**E.default_params(), "hsl": {"red": [0, 0, 0]}, "curve": {}}) - img).max() < 1e-3
with tempfile.TemporaryDirectory() as d:
    E.export_cube(Path(d) / "a.cube", E.default_params(), None, None)
    E.export_cube(Path(d) / "b.cube", {**E.default_params(), "curve": {"rgb": [[0, 30], [255, 255]]}}, None, None)
    assert (Path(d) / "a.cube").read_text() != (Path(d) / "b.cube").read_text(), "кривая не попала в LUT"

# Целевая правка: веса цветов HSL по оттенку.
wts = E.hsl_weights(120)
assert wts[list(E.HSL_CENTERS).index("green")] == 1 and wts.sum() <= 1
w45 = E.hsl_weights(45)
assert abs(w45[1] - w45[2]) < 1e-6 and w45.sum() <= 1 + 1e-6, "оранжево-жёлтый делится поровну"

# ИИ-цветокоррекция: «Естественные цвета» ведут кожу, зелень и небо к эталонам цветов памяти.
from mini_lightroom import grading as GR  # noqa: E402


def lab_fill(L, C, h, shape):
    lab = np.zeros(shape + (3,), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


gi = np.full((300, 450, 3), 0.5, np.float32)
gi[:100] = lab_fill(70, 30, 285, (100, 450))
gi[100:220, :200] = lab_fill(50, 35, 135, (120, 200))
gi[100:220, 250:400] = lab_fill(65, 24, 32, (120, 150))
gseg = {k: np.zeros((300, 450), np.float32) for k in GR.MASK_CATS}
gseg["sky"][:100] = 1
gseg["greenery"][100:220, :200] = 1
gseg["people"][100:220, 250:400] = 1
an0 = GR.analyze(gi, gseg, "landscape")
vs = GR.variants(an0)
assert {"natural", "complementary", "analogous", "subject", "sky", "bw"} <= {v["id"] for v in vs}
assert all(v["name"] and v["rule"] and isinstance(v["recipe"], dict) for v in vs)
nat = next(v for v in vs if v["id"] == "natural")
glayers = [{**m, "arr": (gseg[m["cat"]] * 255).astype(np.uint8)} for m in nat["masks"]]
gout = E.process(gi, {**E.default_params(), "ai_grade": {**nat, "masks": glayers, "strength": 100}})
an1 = GR.analyze(gout, gseg, "landscape")
for reg in ("skin", "greenery", "sky"):
    d0 = abs(GR._ang(GR.MEMORY[reg]["h"], an0["regions"][reg]["h"]))
    d1 = abs(GR._ang(GR.MEMORY[reg]["h"], an1["regions"][reg]["h"]))
    print(f"ИИ-цвет, {reg}: до эталона {d0:.1f}° → {d1:.1f}°")
    assert d1 < d0 - 5 or d1 < 3, f"{reg} не приблизился к эталону"
assert abs(an1["regions"]["skin"]["h"] - 49) < 4, "кожа не на эталоне 49°"
assert np.abs(E.process(gi, {**E.default_params(), "ai_grade": {**nat, "masks": glayers, "strength": 0}})
              - E.process(gi, E.default_params())).max() < 1e-5, "сила 0 меняет кадр"
sunset = GR.variants(GR.analyze(gi, gseg, "sunset"))
assert "golden" not in {v["id"] for v in sunset}, "на закате не нужен «золотой свет»"
assert not any(m["cat"] == "sky" for m in next(v for v in sunset if v["id"] == "natural")["masks"]), \
    "закатное небо не тянем к голубому"

# ИИ-композиция: выравнивание горизонта и правило третей.
hy, hx = np.mgrid[0:600, 0:900]
for tilt in (3.0, -2.0):
    sea = np.full((600, 900, 3), 0.55, np.float32) + np.random.default_rng(5).normal(0, 0.02, (600, 900, 3)).astype(
        np.float32)
    sea[hy > 270 + (hx - 450) * np.tan(np.radians(tilt))] *= 0.4
    fix = E.auto_horizon(np.clip(sea, 0, 1))
    print(f"горизонт: наклон {tilt:+.1f}° → поворот {fix}")
    assert fix is not None and abs(fix + tilt) < 0.3, "горизонт выровнен не туда"
assert E.auto_horizon(np.random.default_rng(6).random((600, 900, 3)).astype(np.float32)) is None, "шум ≠ горизонт"
person = np.zeros((600, 900), np.float32)
person[250:560, 430:520] = 1
head = E.subject_point(person)
tc = E.thirds_crop(900, 600, 1.5, 0, head)
fx, fy = (head[0] - tc[0]) / (tc[2] - tc[0]), (head[1] - tc[1]) / (tc[3] - tc[1])
assert min(abs(fx - 1 / 3), abs(fx - 2 / 3)) < 0.03 and min(abs(fy - 1 / 3), abs(fy - 2 / 3)) < 0.03, (fx, fy)
tiny = np.zeros((600, 900), np.float32)
tiny[100:110, 100:105] = 1
assert E.subject_point(tiny) is None, "прохожий на 0.01% кадра — не главный объект"
low_sky = np.zeros((600, 900), np.float32)
low_sky[:80] = 1
assert E.horizon_level(low_sky) is None, "горизонт у края кадра не тянем на треть"

# Обрезка и горизонт: последний шаг, одна матрица для превью, масштаба и экспорта.
W0, H0 = 1500, 1000
c169 = E.aspect_crop(W0, H0, 16 / 9)
assert E.apply_crop(img, c169, 0).shape[:2] == (844, 1500), "16:9 из 3:2"
assert E.apply_crop(img, None, 0) is img, "без обрезки — без копии"
rot = E.fit_crop(W0, H0, [0, 0, 1, 1], 5)
assert E.crop_valid(W0, H0, rot, 5) and not E.crop_valid(W0, H0, [0, 0, 1, 1], 5), "поворот без пустых углов"
pg = {**E.default_params(), "exposure": 80, "shadows": 60, "clarity": 40, "vignette": -60, "crop": rot, "angle": 5,
      "masks": [{**MK.new_layer("radial", c=[0.5, 0.5], r=[0.2, 0.3]), "adj": {"exposure": 100}}]}
shown = E.apply_crop(E.process(img, pg), rot, 5)
_, vpart = E.process_view_region(img, (300, 200, 700, 500), 1.0, pg)
dv = np.abs(vpart - shown[200:500, 300:700]).mean()
print(f"масштаб под обрезкой с поворотом: расхождение {dv:.4f}")
assert vpart.shape == (300, 400, 3) and dv < 0.01, "увеличенный вид под обрезкой расходится с превью"
with tempfile.TemporaryDirectory(prefix="обрезка_") as d:
    E.save_jpeg(Path(d) / "a.jpg", img)
    E.export_one({"src": str(Path(d) / "a.jpg"), "dst": str(Path(d) / "o.jpg"), "style": None, "auto": False,
                  "params": {**E.default_params(), "crop": c169}, "long_edge": 0, "quality": 90})
    assert E.load_image(Path(d) / "o.jpg").shape[:2] == (844, 1500), "экспорт без обрезки"

# Шумодав и увеличение (этап 4).
from mini_lightroom import enhance as N  # noqa: E402

assert N.iso_strength(800) == 0 and N.iso_strength(1600) == 30 and N.iso_strength(6400) == 70
assert N.iso_strength(25600) == 85 and N.iso_strength(None) == 0
marks = []
E.process_region(img, (600, 300, 1000, 700), 1.0, E.default_params(),
                 prep=lambda c, box: marks.append((c.shape, box)) or c)
(ch, cw, _), (bx0, by0, bx1, by1) = marks[0]
assert (bx1 - bx0, by1 - by0) == (400, 400) and bx0 > 0 and by0 > 0, "prep получил не ту видимую часть"
if N.available() and (N.MODELS_DIR / N.MODELS["denoise"][0]).exists():
    clean = np.full((600, 700, 3), 0.4, np.float32) + np.linspace(0, 0.3, 700, dtype=np.float32)[None, :, None]
    noisy = np.clip(clean + np.random.default_rng(3).normal(0, 0.05, clean.shape).astype(np.float32), 0, 1)
    t = time.perf_counter()
    dn = N.denoise(noisy, 1.0)
    err0, err1 = np.abs(noisy - clean).mean(), np.abs(dn - clean).mean()
    print(f"шумодав: ошибка {err0:.4f} → {err1:.4f} за {time.perf_counter() - t:.1f} с")
    assert dn.shape == noisy.shape and err1 < err0 * 0.5, "шумодав не убрал шум"
    seam = np.abs(np.diff(dn[508:516], axis=0)).mean()
    assert seam < np.abs(np.diff(dn, axis=0)).mean() * 2 + 1e-3, "шов на границе тайлов"
    assert np.abs(N.denoise(noisy, 0.0) - noisy).max() == 0
    if (N.MODELS_DIR / N.MODELS["x2"][0]).exists():
        assert N.upscale(clean[:100, :150], 2).shape == (200, 300, 3)
else:
    print("шумодав: библиотеки ИИ или веса не установлены — пропускаю проверку сети")

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
