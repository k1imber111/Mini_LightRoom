"""Сетки обрезки: золотые пропорции, стыки дуг спирали, перпендикуляры треугольника, 45° у диагоналей.
Запуск: python tests\\test_crop_overlays.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_lightroom import crop_overlays as O  # noqa: E402

if __name__ == "__main__":
    # золотая сетка: линии на 38.2% и 61.8%
    xs = sorted({round(p[0][0], 4) for p in O.lines("phi") if p[0][0] == p[1][0]})
    assert xs == [round(1 / (1 + O.PHI), 4), round(O.PHI / (1 + O.PHI), 4)], xs
    assert abs(xs[0] - 0.382) < 1e-3 and abs(xs[1] - 0.618) < 1e-3
    # трети
    assert sorted(round(p[0][0], 4) for p in O.lines("thirds") if p[0][0] == p[1][0]) == [0.3333, 0.6667]

    # спираль: внутри квадрата, начинается в левом нижнем углу, шаги между точками убывают, нет скачков на стыках дуг
    sp = O.spiral()
    assert all(-1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9 for x, y in sp)
    assert sp[0] == (0.0, 1.0), sp[0]
    steps = [math.dist(sp[i], sp[i + 1]) for i in range(len(sp) - 1)]
    assert max(steps[:28]) < 0.1 and max(steps) < 0.1, "скачок в спирали"
    # на стыках дуг шаг не больше соседних (дуги стыкуются точно, без разрыва и без повтора точки)
    for i in range(28, len(steps), 28):
        assert steps[i] < 2.5 * max(steps[i - 1], 1e-9) and steps[i] > 0, (i, steps[i - 1], steps[i])
    # спираль сходится к «глазу»: длина дуг убывает в φ раз
    arc = [sum(steps[k * 28:(k + 1) * 28 - 1]) for k in range(8)]
    for a, b in zip(arc, arc[1:], strict=False):
        assert abs(a / b - O.PHI) < 0.05, (a, b)
    # зеркала дают другое положение и остаются в квадрате
    assert O.lines("spiral", 1)[0][0] == (1.0, 1.0) and O.lines("spiral", 2)[0][0] == (0.0, 0.0)

    # золотой треугольник: перпендикуляры действительно перпендикулярны диагонали в реальной геометрии
    for aspect in (1.5, 16 / 9, 0.8):
        diag, p1, p2 = O.lines("triangle", 0, aspect)
        dx, dy = (diag[1][0] - diag[0][0]) * aspect, diag[1][1] - diag[0][1]
        for p in (p1, p2):
            vx, vy = (p[1][0] - p[0][0]) * aspect, p[1][1] - p[0][1]
            assert abs(vx * dx + vy * dy) < 1e-9, (aspect, vx * dx + vy * dy)

    # диагонали идут под 45° в реальной геометрии
    for aspect in (1.5, 0.75):
        for s, e in O.lines("diagonal", 0, aspect):
            assert abs(abs((e[0] - s[0]) * aspect) - abs(e[1] - s[1])) < 1e-9
            assert all(-1e-9 <= v <= 1 + 1e-9 for v in (*s, *e))

    # все виды отдают линии внутри квадрата; неизвестный — ошибка
    for kind, _ in O.OVERLAYS:
        for flip in range(4):
            ls = O.lines(kind, flip, 1.5)
            assert ls and all(len(p) >= 2 for p in ls), kind
            assert all(-1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9 for p in ls for x, y in p), kind
    try:
        O.lines("нет")
    except ValueError:
        pass
    else:
        raise AssertionError("ожидалась ошибка")
    print("сетки OK:", len(O.OVERLAYS), "видов")
