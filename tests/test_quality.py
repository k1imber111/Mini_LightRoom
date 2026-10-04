"""Оценка брака: расфокус, смаз, промах фокуса, боке — на синтетике 6000×4000 (как кадр камеры).
Запуск: python tests\\test_quality.py"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mini_lightroom import quality as Q  # noqa: E402

W, H = 6000, 4000


def scene(seed: int = 0) -> np.ndarray:
    """Резкая «натура»: градиент, множество контрастных фигур с антиалиасингом, лёгкая оптическая мягкость."""
    rng = np.random.default_rng(seed)
    g = np.tile(np.linspace(60, 160, W, dtype=np.float32), (H, 1)).astype(np.uint8)
    for _ in range(900):
        c = int(rng.integers(0, 255))
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(g, (x, y), int(rng.integers(20, 160)), c, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(g, (x, y), (x + int(rng.integers(30, 300)), y + int(rng.integers(30, 300))), c, -1, cv2.LINE_AA)
        else:
            cv2.line(g, (x, y), (x + int(rng.integers(-400, 400)), y + int(rng.integers(-400, 400))), c,
                     int(rng.integers(2, 9)), cv2.LINE_AA)
    g = cv2.GaussianBlur(g, (0, 0), 0.55)
    return np.clip(g.astype(np.float32) + rng.normal(0, 1.2, g.shape), 0, 255).astype(np.uint8)


def motion(g: np.ndarray, length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    cv2.line(k, (int(c - dx * c), int(c - dy * c)), (int(c + dx * c), int(c + dy * c)), 1.0, 1)
    return cv2.filter2D(g, -1, k / k.sum())


def blur(g: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(g, (0, 0), s)


def types(res: dict) -> list[str]:
    return [d["type"] for d in res["defects"]]


if __name__ == "__main__":
    t0 = time.time()
    sharp = scene()
    res = Q.analyze_gray(sharp)
    s0 = res["regions"][0]["sigma"]
    print(f"резкий кадр: вердикт {res['verdict']}, σ {s0:.2f}, {time.time() - t0:.1f} с")
    assert res["verdict"] == "ok" and not res["defects"] and s0 < Q.CFG["soft_doubt"], res

    # расфокус: σ растёт монотонно, вердикт доходит до брака, тип — «мягко»
    prev = s0
    for s, want in ((1.0, {"doubt", "bad"}), (2.0, {"bad"}), (3.5, {"bad"})):
        r = Q.analyze_gray(blur(sharp, s))
        sig = r["regions"][0]["sigma"]
        print(f"  Гаусс {s}: σ {sig:.2f}, вердикт {r['verdict']}, типы {types(r)}")
        assert r["verdict"] in want and sig > prev, (s, r["verdict"], sig, prev)
        assert types(r) == ["soft"], types(r)
        prev = sig

    # смаз от движения: тип «смаз», направление ≈ направлению движения (в пределах корзины 22.5°)
    for ang in (0, 30, 90, 150):
        r = Q.analyze_gray(motion(sharp, 31, ang))
        d = r["defects"][0]
        diff = abs((d["angle"] - ang + 90) % 180 - 90)
        print(f"  смаз 31 px под {ang}°: {d['type']}, найдено {d['angle']:.0f}°, анизотропия {d.get('aniso', 0):.2f}")
        assert d["type"] == "motion" and diff <= 25 and r["verdict"] == "bad", (ang, d)

    # боке: резкий «объект» слева на размытом фоне справа — не брак ни по области объекта, ни по лучшему месту кадра
    bokeh = sharp.copy()
    bokeh[:, 3000:] = blur(sharp, 6)[:, 3000:]
    for rois in (None, [{"name": "Объект", "box": [0.05, 0.1, 0.45, 0.9]}]):
        r = Q.analyze_gray(bokeh, rois)
        assert r["verdict"] == "ok", (rois, r)
    # промах фокуса: мягкий объект, а фон рядом резкий → «фокус ушёл»
    miss = sharp.copy()
    miss[:, :3000] = blur(sharp, 3)[:, :3000]
    r = Q.analyze_gray(miss, [{"name": "Объект", "box": [0.05, 0.1, 0.45, 0.9]}])
    print(f"  промах фокуса: {r['verdict']}, {types(r)}, ref {r.get('ref')}")
    assert r["verdict"] == "bad" and types(r) == ["focus_miss"], r
    assert r["ref"][0] >= 0.45, "эталонное резкое место должно быть в резкой половине"
    # точка автофокуса — то же, что область: падает в мягкую половину → брак, в резкую → ок
    assert Q.analyze_gray(miss, af=(0.25, 0.5))["verdict"] == "bad"
    assert Q.analyze_gray(miss, af=(0.75, 0.5))["verdict"] == "ok"
    # второстепенная область (второе лицо) сама кадр не браковёт, но «сомнительно»
    r = Q.analyze_gray(miss, [{"name": "Главное", "box": [0.6, 0.1, 0.95, 0.9]},
                              {"name": "Второе", "box": [0.05, 0.1, 0.4, 0.9]}])
    assert r["verdict"] == "doubt" and r["defects"][0]["name"] == "Второе", r

    # небо без краёв — не брак, а «нечем судить»
    sky = np.tile(np.linspace(40, 200, H, dtype=np.uint8)[:, None], (1, W))
    r = Q.analyze_gray(sky)
    assert r["verdict"] == "unknown" and not r["defects"], r
    # область без краёв, но в кадре они есть: судим по лучшему месту кадра
    r = Q.analyze_gray(np.vstack([sharp[:2000], np.full((2000, W), 90, np.uint8)]), [{"name": "Небо", "box": [0.1, 0.6, 0.9, 0.9]}])
    assert r["verdict"] == "ok", r

    # σ не зависит от разрешения камеры: тот же кадр, уменьшенный вдвое с поправкой, оценивается так же по смыслу
    small = cv2.resize(blur(sharp, 2.0), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    r = Q.analyze_gray(small)
    assert r["verdict"] in ("doubt", "bad"), r

    # малые кадры и вырожденные входы не падают
    assert Q.analyze_gray(np.zeros((50, 50), np.uint8))["verdict"] == "unknown"
    assert Q.analyze_gray(np.full((300, 400), 255, np.uint8), [{"name": "x", "box": [0, 0, 1, 1]}])["verdict"] == "unknown"

    # пересчёт точки АФ в ориентацию окна
    assert Q._orient(0.2, 0.3, 0) == (0.2, 0.3)
    assert Q._orient(0.2, 0.3, 3) == (0.8, 0.7)
    assert Q._orient(0.0, 0.0, 6) == (1.0, 0.0) and Q._orient(0.0, 0.0, 5) == (0.0, 1.0)

    # файл с кириллицей в пути: загрузка, кеш-метка, результат сериализуется в JSON
    import json
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        p = Path(d) / "кадр.jpg"
        ok, buf = cv2.imencode(".jpg", sharp[:1000, :1500], [cv2.IMWRITE_JPEG_QUALITY, 95])
        p.write_bytes(buf.tobytes())
        r = Q.analyze_file(p)
        assert r["size"] == [1500, 1000] and len(r["file"]) == 2
        json.dumps(r)
    print(f"качество OK, всего {time.time() - t0:.0f} с")
