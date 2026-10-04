"""Сетки-подсказки композиции для рамки обрезки: чистая геометрия без Qt.

`lines(kind, flip, aspect)` возвращает ломаные в единичном квадрате (0..1 по x и y, y вниз);
окно растягивает их на рамку. `aspect` — ширина/высота рамки в пикселях: диагонали под 45°, перпендикуляры
золотого треугольника и клетки сетки считаются в реальной геометрии, а не в растянутой.
"""
from __future__ import annotations

import math

__all__ = ["OVERLAYS", "PHI", "ROTATABLE", "lines", "spiral"]

PHI = (1 + 5 ** 0.5) / 2
# (id, подпись): порядок — как листает клавиша O
OVERLAYS = [("thirds", "Третей"), ("phi", "Золотая сетка"), ("spiral", "Золотая спираль"),
            ("triangle", "Золотой треугольник"), ("diagonal", "Диагонали"), ("center", "Центр и симметрия"),
            ("grid", "Сетка")]
ROTATABLE = ("spiral", "triangle")  # у остальных Shift+O ничего не меняет

Pt = tuple[float, float]


def _similarity(p: Pt) -> Pt:
    """Подобие, переводящее золотой прямоугольник (0,0,φ,1) в остаток после отреза левого квадрата:
    поворот на 90° и сжатие в φ раз. Каждая следующая дуга спирали — образ предыдущей."""
    return PHI - p[1] / PHI, p[0] / PHI


def spiral(arcs: int = 9, steps: int = 28) -> list[Pt]:
    """Золотая спираль в единичном квадрате: от левого нижнего угла по часовой стрелке к «глазу».
    Первая дуга — четверть окружности радиуса 1 с центром (0,0) от (0,1) до (1,0) в золотом прямоугольнике
    φ×1; дальше подобие `_similarity` даёт дуги в квадратах убывающего размера, концы стыкуются точно."""
    pts = [(math.sin(math.pi / 2 * i / steps), math.cos(math.pi / 2 * i / steps)) for i in range(steps + 1)]
    out: list[Pt] = []
    for _ in range(arcs):
        out += pts[1:] if out else pts
        pts = [_similarity(p) for p in pts]
    return [(x / PHI, y) for x, y in out]


def _flip(path: list[Pt], flip: int) -> list[Pt]:
    """flip: бит 0 — зеркало по горизонтали, бит 1 — по вертикали (4 положения для Shift+O)."""
    return [((1 - x) if flip & 1 else x, (1 - y) if flip & 2 else y) for x, y in path]


def _foot(p: Pt, a: Pt, b: Pt) -> Pt:
    """Основание перпендикуляра из точки p на прямую ab (в реальных координатах)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    return a[0] + t * dx, a[1] + t * dy


def lines(kind: str, flip: int = 0, aspect: float = 1.5) -> list[list[Pt]]:
    a = max(aspect, 1e-3)  # реальная ширина при высоте 1
    to_unit = lambda p: (p[0] / a, p[1])  # реальные координаты → единичный квадрат
    if kind == "thirds":
        out = [[(k / 3, 0.0), (k / 3, 1.0)] for k in (1, 2)] + [[(0.0, k / 3), (1.0, k / 3)] for k in (1, 2)]
    elif kind == "phi":
        g = 1 / (1 + PHI)  # 0.382 и 0.618
        out = [[(k, 0.0), (k, 1.0)] for k in (g, 1 - g)] + [[(0.0, k), (1.0, k)] for k in (g, 1 - g)]
    elif kind == "spiral":
        out = [_flip(spiral(), flip)]
    elif kind == "triangle":  # диагональ и перпендикуляры на неё из двух других углов
        pa, pb = (0.0, 0.0), (a, 1.0)
        diag = [pa, pb]
        out = [[to_unit(p) for p in diag]]
        for corner in ((a, 0.0), (0.0, 1.0)):
            out.append([to_unit(corner), to_unit(_foot(corner, pa, pb))])
        out = [_flip(path, flip) for path in out]
    elif kind == "diagonal":  # линии под 45° из каждого угла
        t = min(a, 1.0)
        out = [[to_unit(c), to_unit((c[0] + sx * t, c[1] + sy * t))]
               for c, sx, sy in (((0.0, 0.0), 1, 1), ((a, 0.0), -1, 1), ((0.0, 1.0), 1, -1), ((a, 1.0), -1, -1))]
    elif kind == "center":
        out = [[(0.5, 0.0), (0.5, 1.0)], [(0.0, 0.5), (1.0, 0.5)], [(0.0, 0.0), (1.0, 1.0)], [(1.0, 0.0), (0.0, 1.0)]]
    elif kind == "grid":  # квадратные клетки: 4 по высоте
        n = max(2, round(4 * a))
        out = [[(k * (1 / 4) / a, 0.0), (k * (1 / 4) / a, 1.0)] for k in range(1, n) if k / 4 / a < 1]
        out += [[(0.0, k / 4), (1.0, k / 4)] for k in (1, 2, 3)]
    else:
        raise ValueError(f"неизвестная сетка: {kind}")
    return out
