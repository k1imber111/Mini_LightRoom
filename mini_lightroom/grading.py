"""ИИ-цветокоррекция по правилам фотографии: анализ кадра и варианты для карусели. Без Qt.

Кадр измеряется по областям (сегментация SegFormer: люди/кожа, небо, зелень, вода) в CIELAB,
по нейтральным участкам (баланс белого) и по доминирующим оттенкам. Из этого строятся варианты,
каждый — по своему правилу:
  • цвета памяти: кожа, небо, зелень, вода — к эталонам предпочтительной репродукции (кожа ≈ 49° h_ab
    в CIELAB одинаково для всех этносов; зритель предпочитает их на ~1 L* светлее и ~2 C* насыщеннее);
  • гармонии цветового круга: дополнительная (teal & orange), аналоговая, сплит-комплементарная;
  • 60-30-10, тёплое/холодное (тёплое выступает, холодное отступает), фигура–фон, плотное небо, ч/б по тонам.
Вариант = {"id", "name", "rule", "recipe" (для engine.apply_look), "masks" (слои ИИ-масок с adj)}.
Все оттенки в рецептах — градусы HSV, как в engine (HSL, тонирование).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import engine as E

__all__ = ["MASK_CATS", "MEMORY", "analyze", "variants"]

MASK_CATS = ["people", "sky", "greenery", "water"]
# Эталоны цветов памяти в CIELAB: оттенок h_ab, коридор хромы C*, прибавка светлоты L*.
MEMORY = {
    "skin": {"h": 49.0, "C": (16.0, 32.0), "dL": 1.5},
    "greenery": {"h": 118.0, "C": (18.0, 45.0), "dL": 1.0},
    "sky": {"h": 250.0, "C": (15.0, 45.0), "dL": 0.0},
    "water": {"h": 225.0, "C": (10.0, 35.0), "dL": 0.0},
}
LABELS = {"skin": "кожа", "greenery": "зелень", "sky": "небо", "water": "вода"}
WARM_SCENES = {"sunset", "blue_hour", "night_city"}   # тёплый/цветной свет задуман — баланс не «чиним»
BANDS = list(E.HSL_CENTERS)
SOFT_S = [[0, 0], [64, 58], [192, 198], [255, 255]]
MATTE = [[0, 26], [64, 78], [128, 136], [192, 196], [255, 244]]
MATTE_LIGHT = [[0, 14], [64, 70], [192, 196], [255, 250]]
HUE_NAMES = [(0, "красный"), (25, "оранжевый"), (50, "жёлтый"), (90, "салатовый"), (125, "зелёный"),
             (165, "бирюзовый"), (195, "голубой"), (225, "синий"), (265, "фиолетовый"), (310, "пурпурный")]


def _ang(a: float, b: float) -> float:
    """Разность углов a − b в (−180, 180]."""
    return (a - b + 180) % 360 - 180


def hue_name(h: float) -> str:
    return min(HUE_NAMES, key=lambda hn: abs(_ang(h, hn[0])))[1]


def _circ_mean(h_deg: np.ndarray, w: np.ndarray) -> tuple[float | None, float]:
    sw = float(w.sum())
    if sw <= 1e-6:
        return None, 0.0
    x = float((w * np.cos(np.radians(h_deg))).sum())
    y = float((w * np.sin(np.radians(h_deg))).sum())
    return float(np.degrees(np.arctan2(y, x)) % 360), float(np.hypot(x, y) / sw)


def _lab_hue_to_hsv(L: float, C: float, h: float) -> float:
    """Оттенок h_ab (CIELAB) → оттенок HSV того же цвета: сдвиги считаются в LAB, а применяются в HSL."""
    lab = np.array([[[L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))]]], np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])


# ---------------------------------------------------------------- анализ

def analyze(img: np.ndarray, seg: dict | None = None, scene: str | None = None) -> dict:
    """img — кадр после базовых правок (float RGB 0..1, лучше ~500 px); seg — {категория: маска 0..1}
    того же размера (без библиотек ИИ — None: области угадываются по оттенкам)."""
    img = np.clip(img, 0, 1).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    hab = np.degrees(np.arctan2(b, a)) % 360
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = L.size
    skin_like = E._skin_mask(lab)
    has_masks = bool(seg)
    if not has_masks:  # без сегментации: грубые области по цвету
        top = (np.arange(L.shape[0])[:, None] < L.shape[0] * 0.55).astype(np.float32)
        seg = {"greenery": ((H >= 70) & (H <= 160) & (S > 0.15)).astype(np.float32),
               "sky": ((H >= 185) & (H <= 250) & (S > 0.1)).astype(np.float32) * top,
               "people": (skin_like > 0.5).astype(np.float32)}
    seg = {k: np.clip(v, 0, 1).astype(np.float32) for k, v in seg.items()}

    def stats(w):
        sw = max(float(w.sum()), 1e-6)
        hl, conc = _circ_mean(hab, w * C)
        hh, _ = _circ_mean(H, w * S)
        return {"share": sw / n, "L": float((w * L).sum() / sw), "C": float((w * C).sum() / sw),
                "h": hl, "hsv_h": hh, "s": float((w * S).sum() / sw), "conc": conc}

    regions = {}
    if "people" in seg and (seg["people"] * skin_like).sum() / n >= 0.003:
        regions["skin"] = stats(seg["people"] * skin_like)
    for cat in ("sky", "greenery", "water", "people"):
        if cat in seg and seg[cat].mean() >= 0.02:
            regions[cat] = stats(seg[cat])
    excl = np.clip(seg.get("sky", 0) + skin_like * seg.get("people", 0), 0, 1)
    neutral = (C < 12) & (L > 20) & (L < 90) & (excl < 0.3)
    nrgb = img[neutral].mean(0) if neutral.sum() > 50 else None
    colorful = S > 0.12
    hist = np.bincount((H[colorful] / 10).astype(int) % 36, weights=(S * V)[colorful], minlength=36)
    hist = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))  # сглаживание по кругу
    dominant = float(np.argmax(hist) * 10 + 5) if hist.sum() > 0 else None
    key = regions["skin"]["hsv_h"] if "skin" in regions else dominant
    return {"scene": scene, "regions": regions, "has_masks": has_masks,
            "neutral_rgb": None if nrgb is None else nrgb.tolist(), "neutral_share": float(neutral.mean()),
            "cast": [float(a[neutral].mean()), float(b[neutral].mean())] if nrgb is not None else [0.0, 0.0],
            "dominant": dominant, "key": key, "L_mean": float(L.mean()), "sat_mean": float(S.mean())}


# ---------------------------------------------------------------- правила

def _wb_fix(nrgb) -> list[int]:
    """Баланс белого, делающий нейтраль серой — в единицах ползунков, той же формулой, что apply_look."""
    r, g, b = np.maximum(np.asarray(nrgb, np.float64), 1e-3)
    q = (b / r) ** 2.2
    t = (q - 1) / (q + 1) / 0.25
    gr, gb = 1 + 0.25 * t, 1 - 0.25 * t
    rb = (r * gr ** (1 / 2.2) + b * gb ** (1 / 2.2)) / 2
    m = (1 - (rb / g) ** 2.2) / 0.2
    return [int(np.clip(round(t * 100), -30, 30)), int(np.clip(round(m * 100), -30, 30))]


def _bands(hsv_h: float) -> list[str]:
    return [BANDS[i] for i, w in enumerate(E.hsl_weights(hsv_h)) if w > 0.05]


def _toward(reg: dict, target: dict) -> dict:
    """HSL-сдвиг области к эталону цвета памяти: оттенок не больше ±20°, насыщенность ±50%.
    Если область далеко от эталона (закатное небо), её не трогаем — значит, так задумано светом."""
    if reg.get("h") is None or reg.get("hsv_h") is None or reg["C"] < 4 or reg["L"] < 15:
        return {}  # почти бесцветное или очень тёмное: цвет глазу не различим — не трогаем
    dh_lab = _ang(target["h"], reg["h"])
    if abs(dh_lab) > 60:
        return {}
    L0, C0, h0 = reg["L"], max(reg["C"], 8.0), reg["h"]
    lo, hi = target["C"]
    goal_h = h0 + float(np.clip(dh_lab, -20, 20))       # не больше 20° за раз — без «кислоты»
    # Предпочтительная репродукция: +2 C*, но не выше коридора; поднимать до нижней границы — только в
    # средних и светлых тонах (листва в тени и должна быть глухой, иначе «кислота»).
    goal_C = min(max(reg["C"] + 2, lo) if L0 > 35 else reg["C"] + 2, hi)
    goal_L = L0 + target["dL"]                           # и чуть светлее
    bands = _bands(reg["hsv_h"])
    # Первое приближение — через перевод LAB → HSV, затем подгонка на самом среднем цвете области той же
    # функцией, что обрабатывает кадр (связь оттенков HSV и CIELAB нелинейна, «в лоб» кожа перелетает).
    vals = [_ang(_lab_hue_to_hsv(L0, C0, goal_h), _lab_hue_to_hsv(L0, C0, h0)) / 0.3,
            (goal_C / C0 - 1) * 100, target["dL"] * 4]
    lab0 = np.array([[[L0, C0 * np.cos(np.radians(h0)), C0 * np.sin(np.radians(h0))]]], np.float32)
    px = np.clip(cv2.cvtColor(lab0, cv2.COLOR_Lab2RGB), 0, 1)
    for _ in range(4):
        vals = [float(np.clip(v, -100, 100)) for v in vals]
        out = cv2.cvtColor(np.clip(E._apply_hsl(px, {b: vals for b in bands}), 0, 1), cv2.COLOR_RGB2Lab)[0, 0]
        Lo, Co, ho = out[0], float(np.hypot(out[1], out[2])), float(np.degrees(np.arctan2(out[2], out[1])) % 360)
        got, want = _ang(ho, h0), _ang(goal_h, h0)
        if abs(got) > 0.5 and abs(want) > 0.5:
            vals[0] *= float(np.clip(want / got, 0.3, 3.0))
        vals[1] += (goal_C - Co) / C0 * 100 * 0.8
        vals[2] += (goal_L - Lo) * 4 * 0.8
    vals = [int(np.clip(round(v), -100, 100)) for v in vals]
    vals[1] = int(np.clip(vals[1], -50, 50))
    return {band: vals for band in bands} if any(vals) else {}


def _layer(cat: str, adj: dict, name: str, invert: bool = False) -> dict:
    layer = {"type": "ai", "cat": cat, "name": name, "on": True, "invert": invert, "adj": adj}
    if cat == "sky" and not invert:
        layer["protect_white"] = True  # солнце в маске неба не сереет
    return layer


def _harmony_hsl(anchors: list[tuple[float, int]], protect_skin: bool, reach: float = 45,
                 mute: int = -35, pull: float = 0.6, limit: float = 18) -> dict:
    """Стянуть полосы HSL к опорным оттенкам гармонии (anchors: [(оттенок, +насыщ.)]), чужие приглушить."""
    hsl = {}
    for band, c in E.HSL_CENTERS.items():
        best = min(anchors, key=lambda an: abs(_ang(an[0], c)))
        d = _ang(best[0], c)
        if abs(d) <= reach:
            hsl[band] = [round(float(np.clip(d * pull, -limit, limit)) / 0.3), best[1], 0]
        elif not (protect_skin and band in ("red", "orange")):  # кожу не выцветаем
            hsl[band] = [0, mute, 0]
    return {k: v for k, v in hsl.items() if any(v)}


def variants(an: dict) -> list[dict]:
    """Варианты цветокоррекции, применимые к этому кадру (по сцене и найденным объектам)."""
    regs, scene = an["regions"], an.get("scene")
    warm = scene in WARM_SCENES
    masks_ok = an["has_masks"]
    skin = "skin" in regs
    out = []

    def add(vid, name, rule, recipe, masks=()):
        out.append({"id": vid, "name": name, "rule": rule, "recipe": recipe, "masks": list(masks)})

    # 1. Естественные цвета: нейтральный баланс + цвета памяти к эталонам.
    recipe, masks, glob, done = {"curve": SOFT_S, "vib": 5}, [], {}, []
    if (not warm and an["neutral_rgb"] and an["neutral_share"] >= 0.02
            and max(abs(c) for c in an["cast"]) >= 1.5):
        recipe["wb"] = _wb_fix(an["neutral_rgb"])
        done.append("баланс белого по серым участкам")
    for reg_name, cat in (("skin", "people"), ("greenery", "greenery"), ("sky", "sky"), ("water", "water")):
        if reg_name not in regs or (reg_name == "sky" and warm):
            continue
        shift = _toward(regs[reg_name], MEMORY[reg_name])
        if not shift:
            continue
        done.append(LABELS[reg_name])
        if masks_ok:
            masks.append(_layer(cat, {"hsl": shift}, f"ИИ: {LABELS[reg_name]}"))
        else:
            glob.update({b: [round(v / 2) for v in vals] for b, vals in shift.items()})
    if glob:
        recipe["hsl"] = glob
    add("natural", "Естественные цвета",
        "Цвета памяти к эталонам предпочтительной репродукции (кожа ≈ 49° CIELAB), чуть светлее и насыщеннее, "
        "как предпочитает глаз. Исправлено: " + (", ".join(done) if done else "почти ничего — кадр уже близок"),
        recipe, masks)

    key, dom = an.get("key"), an.get("dominant")
    if key is not None:
        # 2. Дополнительная гармония (teal & orange от ключевого цвета).
        comp = (key + 180) % 360
        add("complementary", "Дополнительная гармония",
            f"Ключевой цвет ({hue_name(key)}) и его противоположность ({hue_name(comp)}): тёплые света, "
            "холодные тени — самый сильный цветовой контраст, как teal & orange в кино",
            {"hsl": _harmony_hsl([(key, 15), (comp, 12)], skin),
             "grade": {"highlights": [key, 30], "shadows": [comp, 35], "balance": 0},
             "curve": [[0, 4], [64, 54], [192, 204], [255, 252]]})
        # 4. Сплит-комплементарная: ключ + два соседа его дополнения.
        s1, s2 = (key + 150) % 360, (key + 210) % 360
        cool = min((s1, s2), key=lambda h: abs(_ang(h, 210)))
        add("split", "Сплит-комплементарная",
            f"{hue_name(key).capitalize()} + {hue_name(s1)} и {hue_name(s2)}: контраст почти как у дополнительной "
            "пары, но мягче и богаче",
            {"hsl": _harmony_hsl([(key, 12), (s1, 10), (s2, 10)], skin, reach=40),
             "grade": {"highlights": [key, 22], "shadows": [cool, 30], "balance": 0},
             "curve": [[0, 6], [64, 58], [192, 200], [255, 252]]})
    if dom is not None:
        # 3. Аналоговая гармония: оттенки стягиваются к доминанте, чужие приглушаются.
        add("analogous", "Аналоговая гармония",
            f"Соседние оттенки вокруг доминанты ({hue_name(dom)}): спокойная цельная палитра, чужие цвета тише",
            {"hsl": _harmony_hsl([(dom, 8)], skin, reach=60, mute=-55, pull=0.6, limit=20),
             "grade": {"mid": [dom, 22], "highlights": [dom, 18]}, "curve": MATTE_LIGHT})

    # 5–7. Объектные правила — только когда есть маски объектов.
    if masks_ok and "sky" in regs and not warm:
        add("warmcool", "Тёплое и холодное",
            "Тёплые цвета выступают, холодные отступают: передний план теплее, небо холоднее — кадр глубже",
            {}, [_layer("sky", {"temperature": -70, "tint": 8}, "ИИ: небо холоднее"),
                 _layer("sky", {"temperature": 60, "exposure": 10}, "ИИ: земля теплее", invert=True)])
    elif masks_ok and "people" in regs:
        add("warmcool", "Тёплое и холодное",
            "Тёплое выступает, холодное отступает: люди теплее, фон прохладнее",
            {}, [_layer("people", {"temperature": 100, "exposure": 25}, "ИИ: люди теплее"),
                 _layer("people", {"temperature": -90, "saturation": -30}, "ИИ: фон прохладнее", invert=True)])
    if masks_ok and "people" in regs and regs["people"]["share"] >= 0.02:
        add("subject", "Акцент на объекте",
            "Фигура и фон (60-30-10): люди светлее и живее, фон тише — взгляд сразу идёт к главному",
            {}, [_layer("people", {"exposure": 60, "vibrance": 25, "clarity": 20, "shadows": 35}, "ИИ: люди"),
                 _layer("people", {"exposure": -50, "saturation": -50}, "ИИ: фон", invert=True)])
    if masks_ok and "sky" in regs and regs["sky"]["share"] >= 0.08:
        add("sky", "Плотное небо",
            "Небо не спорит с передним планом: плотнее и насыщеннее" + (", теплее" if warm else "")
            + "; тени земли приоткрыты",
            {}, [_layer("sky", {"exposure": -80, "highlights": -70, "saturation": 30,
                                "temperature": 40 if warm else -20}, "ИИ: небо"),
                 _layer("sky", {"shadows": 45, "exposure": 20}, "ИИ: земля", invert=True)])

    # 8. Пастельная гармония 60-30-10: всё тише, ключевой цвет-акцент сохранён.
    pastel = {"sat": -28, "exposure": 0.1, "curve": MATTE,
              "grade": {"highlights": [40, 10], "shadows": [210, 8], "balance": 0}}
    if key is not None:
        pastel["hsl"] = {b: [0, 25, 0] for b in _bands(key)}
    add("pastel", "Пастельная гармония",
        "60-30-10: приглушённая основа, мягкий матовый низ, кремовые света; ключевой цвет остаётся акцентом", pastel)

    # 9. Золотой свет — для дневных кадров.
    if not warm:
        add("golden", "Золотой свет",
            "Тёплый низкий свет: золотые света, прохладные тени, зелень к жёлтому — как за час до заката",
            {"wb": [60, 10], "grade": {"highlights": [38, 30], "shadows": [220, 16], "balance": 0},
             "hsl": {"green": [25, -15, 0], "yellow": [-8, 10, 8]}, "curve": SOFT_S})

    # 10. Ч/б по тонам: яркость каналов по содержимому.
    add("bw", "Ч/б по тонам",
        "Чёрно-белое, где тона разведены по смыслу: небо темнее, кожа и зелень светлее; тёплый оттенок бумаги",
        {"hsl": {"blue": [0, 0, -35], "aqua": [0, 0, -20], "orange": [0, 0, 20], "green": [0, 0, 15],
                 "yellow": [0, 0, 10], "red": [0, 0, 5]},
         "sat": -100, "curve": [[0, 0], [64, 52], [192, 206], [255, 255]],
         "grade": {"highlights": [40, 5]}})
    return out
