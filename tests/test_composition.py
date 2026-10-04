"""Авто-кадр: варианты по правилам композиции на синтетических объектах и горизонте.
Запуск: python tests\\test_composition.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mini_lightroom import composition as C  # noqa: E402
from mini_lightroom import crop_overlays as CO  # noqa: E402
from mini_lightroom import engine as E  # noqa: E402

W, H = 6000, 4000


def rel(point, rect):
    return (point[0] - rect[0]) / (rect[2] - rect[0]), (point[1] - rect[1]) / (rect[3] - rect[1])


def inside(box, rect) -> float:
    return C._overlap(box, rect)


if __name__ == "__main__":
    # пейзаж: горизонт посередине → лучший вариант переносит его на треть/золотую линию, «центр» хуже
    vs = C.suggest(W, H, 0, None, 0.5, "landscape")
    assert len(vs) >= 2 and vs[0]["rule"] in ("thirds", "phi"), [(v["rule"], round(v["score"], 3)) for v in vs]
    for v in vs:
        h = rel((0.5, 0.5), v["rect"])[1]
        assert E.crop_valid(W, H, v["rect"], 0)
        if v["rule"] in ("thirds", "phi"):
            lines = (1 / 3, 2 / 3) if v["rule"] == "thirds" else (0.382, 0.618)
            assert min(abs(h - ln) for ln in lines) < 0.06, (v["rule"], h)
    print("пейзаж:", [(v["label"], round(v["score"], 2), v["why"]) for v in vs][:3])

    # портрет: лицо смотрит вправо; голова целиком, глаза на верхней силовой точке, перед взглядом место, запас над головой
    eyes = (0.50, 0.40)
    face = {"kind": "face", "point": eyes, "box": (0.43, 0.22, 0.58, 0.62), "facing": 1}
    vs = C.suggest(W, H, 0, face, None, "portrait")
    assert vs, "портрет без вариантов"
    for v in vs:
        assert inside(face["box"], v["rect"]) >= 0.97, "голову нельзя резать"
        rx, ry = rel(eyes, v["rect"])
        assert ry < 0.5, "глаза выше середины рамки"
        assert E.crop_valid(W, H, v["rect"], 0)
    top = vs[0]
    rx, ry = rel(eyes, top["rect"])
    top_gap = rel((0, face["box"][1]), top["rect"])[1]
    print(f"портрет: {top['label']}, глаза в рамке ({rx:.2f}, {ry:.2f}), запас над головой {top_gap:.2f}, {top['why']}")
    assert rx <= 0.55, "взгляд вправо: перед ним должно быть место (глаза левее середины рамки)"
    assert 0.0 <= top_gap <= 0.3
    anchors = {"thirds": [(1 / 3, 1 / 3), (2 / 3, 1 / 3)], "phi": [(0.382, 0.382), (0.618, 0.382)],
               "spiral": [(*C._spiral_eye(f),) for f in range(4)], "center": [(0.5, 0.5)]}[top["rule"]]
    assert min(np.hypot(rx - ax, ry - ay) for ax, ay in anchors) < 0.12, (top["rule"], rx, ry)

    # пропорция зафиксирована: у всех вариантов ровно она; поворот кадра не выводит рамку за снимок
    for asp, ang in ((1.0, 0), (16 / 9, 0), (1.5, 3.0), (0.8, -4.0)):
        vs = C.suggest(W, H, ang, face, 0.55, "portrait", aspect=asp)
        for v in vs:
            pix = (v["rect"][2] - v["rect"][0]) * W / ((v["rect"][3] - v["rect"][1]) * H)
            assert abs(pix - asp) < 2e-3 * asp and E.crop_valid(W, H, v["rect"], ang), (asp, ang, pix)
    # объект у края: рамка его не режет
    edge = {"kind": "object", "point": (0.95, 0.5), "box": (0.9, 0.4, 0.99, 0.6), "facing": 0}
    vs = C.suggest(W, H, 0, edge, None, "other")
    assert vs and inside(edge["box"], vs[0]["rect"]) > 0.9, vs[:1]

    # нечего улучшать — пусто, без падений
    assert C.suggest(W, H, 0, None, None, "landscape") == []

    # карта важности: рамка не отрезает самое важное
    weight = np.zeros((40, 60), np.float32)
    weight[5:25, 38:56] = 1.0  # важное в правой части
    obj = {"kind": "object", "point": (0.78, 0.37), "box": (0.63, 0.12, 0.93, 0.62), "facing": 0}
    vs = C.suggest(W, H, 0, obj, None, "other", weight=weight)
    assert vs and all(inside(obj["box"], v["rect"]) > 0.9 for v in vs[:2])

    # объект из масок и лиц
    face_in = {"box": (0.43, 0.22, 0.58, 0.55), "eyes": {"left": (0.45, 0.30, 0.50, 0.34), "right": (0.52, 0.30, 0.57, 0.34)},
               "yaw": 0.3}
    s = C.subject_from([face_in], None, None)
    assert s["kind"] == "face" and s["facing"] == 1 and abs(s["point"][1] - 0.32) < 1e-6 and s["box"][1] < 0.22
    assert C.subject_from([{**face_in, "yaw": -0.3}], None, None)["facing"] == -1
    assert C.subject_from([{**face_in, "yaw": 0.05}], None, None)["facing"] == 0
    people = np.zeros((100, 150), np.float32)
    people[30:90, 60:80] = 1.0
    s = C.subject_from(None, people, None)
    assert s["kind"] == "person" and 0.38 < s["point"][0] < 0.55 and s["point"][1] < 0.45
    blob = np.zeros((100, 150), np.float32)
    blob[40:70, 90:120] = 1.0
    s = C.subject_from(None, None, blob)
    assert s["kind"] == "object" and 0.65 < s["point"][0] < 0.8
    assert C.subject_from(None, None, np.zeros((100, 150), np.float32)) is None
    assert C.subject_from(None, np.zeros((100, 150), np.float32), None) is None

    # жанры
    assert C.genre_of("sunset") == "landscape" and C.genre_of("flowers") == "macro" and C.genre_of(None) == "other"
    assert C.genre_of("sunset", has_face=True) == "portrait" and C.genre_of("architecture") == "architecture"
    # «глаз» спирали лежит внутри квадрата и совпадает с концом линии сетки
    for f in range(4):
        ex, ey = C._spiral_eye(f)
        assert 0 < ex < 1 and 0 < ey < 1 and (ex, ey) == (CO.spiral()[-1] if f == 0 else (ex, ey))
    print("композиция OK")
