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
    # все семь правил дают ровно один лучший вариант, объект встаёт на силовую точку именно этого правила
    obj = {"kind": "object", "point": (0.62, 0.45), "box": (0.55, 0.36, 0.69, 0.54), "facing": 0}
    for rule in C.RULES:
        vs = C.suggest(W, H, 0, obj, None, "other", aspect=1.5, rules=(rule,))
        assert len(vs) == 1 and vs[0]["rule"] == rule and vs[0]["overlay"] == rule, (rule, vs)
        v = vs[0]
        assert inside(obj["box"], v["rect"]) > 0.95 and E.crop_valid(W, H, v["rect"], 0), rule
        pix = (v["rect"][2] - v["rect"][0]) * W / ((v["rect"][3] - v["rect"][1]) * H)
        assert abs(pix - 1.5) < 2e-3, (rule, pix)  # пропорция не меняется
        rx, ry = rel(obj["point"], v["rect"])
        anchors = C._anchors(rule, False, 1.5, v["flip"] if rule in ("spiral", "triangle") else None)
        assert min(np.hypot(rx - ax, ry - ay) for ax, ay, _ in anchors) < 0.08, (rule, rx, ry)
    # только горизонт (объекта нет): каждое правило ставит его на свою линию
    for rule in C.RULES:
        vs = C.suggest(W, H, 0, None, 0.5, "landscape", aspect=1.5, rules=(rule,))
        assert len(vs) == 1, rule
        h = rel((0.5, 0.5), vs[0]["rect"])[1]
        lines = C.HORIZON_LINES.get(rule, (1 / 3, 2 / 3))
        assert min(abs(h - ln) for ln in lines) < 0.07, (rule, h)
    assert C.suggest(W, H, 0, None, None, "landscape", rules=("thirds",)) == []
    # зафиксированное положение спирали/треугольника
    for fl in range(4):
        assert C.suggest(W, H, 0, obj, None, "other", aspect=1.5, rules=("spiral",), flip=fl)[0]["flip"] == fl
        assert C.suggest(W, H, 0, obj, None, "other", aspect=1.5, rules=("triangle",), flip=fl)[0]["flip"] == fl
    # план по анализу кадра: поворот холста и точка, указанная вручную, заменяет найденный объект
    an = {"subject": None, "horizon": None, "genre": "other", "weight": None}
    assert C.plan(an, W, H, 0, 1.5, ("thirds",)) == []
    vs = C.plan(an, W, H, 3.0, 1.5, ("phi",), manual=(0.7, 0.4))
    assert len(vs) == 1 and E.crop_valid(W, H, vs[0]["rect"], 3.0)
    sub, _ = C.analysis_to_canvas(an, W, H, 3.0, (0.7, 0.4))
    rx, ry = rel(sub["point"], vs[0]["rect"])
    assert min(np.hypot(rx - ax, ry - ay) for ax, ay, _ in C._anchors("phi", False, 1.5)) < 0.08
    sub0, hor0 = C.analysis_to_canvas({"subject": {"kind": "object", "point": (0.3, 0.6), "box": (0.2, 0.5, 0.4, 0.7)},
                                       "horizon": 0.55}, W, H, 0)
    assert sub0["point"] == (0.3, 0.6) and abs(hor0 - 0.55) < 1e-9, "без поворота координаты не меняются"

    # главный объект: яркий диск на тёмном фоне находится, текстура и облака без явного пятна — нет
    import cv2
    moon = np.full((400, 600, 3), 0.02, np.float32)
    cv2.circle(moon, (330, 170), 28, (0.9, 0.9, 0.85), -1, cv2.LINE_AA)
    b = C.isolated_blob(moon)
    assert b and abs(b["point"][0] - 330 / 600) < 0.03 and abs(b["point"][1] - 170 / 400) < 0.04, b
    assert 0.03 < b["box"][2] - b["box"][0] < 0.2, "рамка пятна — порядка размера диска, а не втрое больше"
    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.random((400, 600, 3), dtype=np.float32), (0, 0), 6) * 2.5
    assert C.isolated_blob(np.clip(texture, 0, 1)) is None, "ровная текстура — объекта нет"
    sky = np.tile(np.linspace(0.2, 0.8, 400, dtype=np.float32)[:, None, None], (1, 600, 3)).copy()
    for y in (120, 190, 260):
        cv2.line(sky, (0, y), (600, y + 10), (1.0, 0.5, 0.2), 14)  # несколько похожих полос облаков
    assert C.isolated_blob(sky) is None, "несколько похожих деталей — не изолированный объект"
    # предмет из сегментации: одна цапля среди прочего; две близких по размеру — группа; мелочь ниже порога — нет
    objs = np.zeros((100, 150), np.float32)
    objs[40:60, 70:80] = 1.0
    s = C.subject_from(None, None, objs)
    assert s["kind"] == "object" and s["how"] == "предмет" and 0.45 < s["point"][0] < 0.55
    objs2 = objs.copy()
    objs2[40:58, 100:110] = 1.0
    s2 = C.subject_from(None, None, objs2)
    assert s2["box"][2] > 0.65, "пара похожих по размеру предметов — одна группа"
    tiny = np.zeros((1000, 1500), np.float32)
    tiny[500:502, 700:702] = 1.0
    assert C.subject_from(None, None, tiny) is None, "точка в пару пикселей — не объект"
    # лицо важнее человека, человек — предмета, предмет — пятна
    face_first = C.subject_from([face_in], people, objs, moon)
    assert face_first["kind"] == "face" and C.subject_from(None, people, objs, moon)["kind"] == "person"
    assert C.subject_from(None, None, objs, moon)["how"] == "предмет" and C.subject_from(None, None, None, moon)["how"] == "яркое пятно"
    print("композиция OK")
