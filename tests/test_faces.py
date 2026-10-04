"""Лица и закрытые глаза: логика eyes_state на синтетических точках сетки + модель на пустом кадре.
Реальные лица в репозитории не хранятся: на живых портретах проверяет Иван (см. PROJECT_STATE, этап 3).
Запуск: python tests\\test_faces.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mini_lightroom import faces as F  # noqa: E402

W, H = 2400, 1600


def face(ear_a: float, ear_b: float, width_px: float = 140.0) -> np.ndarray:
    """478 точек сетки; контуры глаз заданной ширины и EAR (вертикальный разброс = ear · ширина)."""
    pts = np.full((478, 2), 0.5, np.float32)
    for idx, ear_pts, cx, ear in ((F.EYE_A, F.EAR_A, 0.42, ear_a), (F.EYE_B, F.EAR_B, 0.58, ear_b)):
        w, d = width_px / W, ear * width_px / H
        for i in idx:
            pts[i] = (cx, 0.4)
        p1, p2, p3, p4, p5, p6 = ear_pts
        pts[p1], pts[p4] = (cx - w / 2, 0.4), (cx + w / 2, 0.4)
        pts[p2], pts[p3] = (cx - w / 6, 0.4 - d / 2), (cx + w / 6, 0.4 - d / 2)
        pts[p6], pts[p5] = (cx - w / 6, 0.4 + d / 2), (cx + w / 6, 0.4 + d / 2)
    return pts


if __name__ == "__main__":
    # открытые глаза
    st = F.eyes_state(face(0.30, 0.30), (0.1, 0.1), (W, H))
    assert st["closed"] == [] and abs(st["ear"][0] - 0.30) < 0.01, st
    # оба закрыты
    st = F.eyes_state(face(0.05, 0.06), (0.9, 0.85), (W, H))
    assert st["closed"] == ["left", "right"], st
    # подмигивание: закрыт один глаз (левый на снимке)
    st = F.eyes_state(face(0.05, 0.30), (0.9, 0.1), (W, H))
    assert st["closed"] == ["left"], st
    st = F.eyes_state(face(0.30, 0.06), (0.1, 0.9), (W, H))
    assert st["closed"] == ["right"], st
    # взгляд вниз: веко опущено (blink высокий), но глаз открыт по контуру — не закрытый
    assert F.eyes_state(face(0.26, 0.26), (0.8, 0.8), (W, H))["closed"] == []
    # прищур без blink — не закрытый
    assert F.eyes_state(face(0.12, 0.12), (0.3, 0.3), (W, H))["closed"] == []
    # слишком мелкие глаза: судить нельзя (None), а не «открыты»
    assert F.eyes_state(face(0.05, 0.05, width_px=8), (0.9, 0.9), (W, H))["closed"] is None
    # рамки глаз внутри кадра, левая левее правой
    st = F.eyes_state(face(0.3, 0.3), (0, 0), (W, H))
    assert st["eyes"]["left"][2] < st["eyes"]["right"][2]
    assert all(0 <= v <= 1 for b in st["eyes"].values() for v in b)

    # лица → области резкости и дефекты глаз (quality.face_rois / add_eyes)
    from mini_lightroom import quality as Q
    main = F.eyes_state(face(0.05, 0.05), (0.9, 0.9), (W, H))
    main["box"] = [0.40, 0.20, 0.60, 0.50]
    second = F.eyes_state(face(0.05, 0.30), (0.9, 0.1), (W, H))
    second["box"] = [0.70, 0.30, 0.82, 0.55]
    far = F.eyes_state(face(0.05, 0.05), (0.9, 0.9), (W, H))
    far["box"] = [0.10, 0.10, 0.12, 0.13]  # уже 3% кадра — не учитывается
    rois = Q.face_rois([main, second, far])
    assert [r["name"] for r in rois] == ["Лицо", "Лицо 2"], rois
    assert abs(rois[0]["box"][3] - (0.20 + 0.7 * 0.30)) < 1e-9, "область резкости — верхние 70% лица"
    res = Q.add_eyes({"verdict": "ok", "defects": []}, rois)
    kinds = [(d["type"], d["level"], d["name"]) for d in res["defects"]]
    assert kinds == [("eyes_closed", "bad", "Глаза закрыты"), ("eyes_closed", "doubt", "Левый глаз закрыт")], kinds
    assert res["verdict"] == "bad" and len(res["faces"]) == 2
    only_second = Q.add_eyes({"verdict": "ok", "defects": []}, [dict(rois[1], face=second)])
    assert only_second["verdict"] == "bad"  # единственное лицо в списке считается главным
    res = Q.add_eyes({"verdict": "ok", "defects": []}, Q.face_rois([F.eyes_state(face(0.3, 0.3), (0.1, 0.1), (W, H)) | {"box": [0.4, 0.2, 0.6, 0.5]}]))
    assert res["verdict"] == "ok" and not res["defects"]
    tiny = F.eyes_state(face(0.05, 0.05, width_px=8), (0.9, 0.9), (W, H)) | {"box": [0.4, 0.2, 0.6, 0.5]}
    res = Q.add_eyes({"verdict": "ok", "defects": []}, Q.face_rois([tiny]))
    assert res["verdict"] == "ok" and not res["defects"], "мелкие глаза не проверяются и не бракуют"
    assert Q.face_rois([]) == []

    # модель на пустом кадре не находит лиц (если mediapipe и модель есть)
    if F.available() and (Path(__file__).resolve().parent.parent / "models" / F.MODEL_FILE).exists():
        finder = F.get_finder(Path(__file__).resolve().parent.parent / "models")
        assert finder is not None and finder.detect(np.full((480, 640, 3), 128, np.uint8)) == []
        print("модель лиц: пустой кадр — 0 лиц")
    print("лица OK")
