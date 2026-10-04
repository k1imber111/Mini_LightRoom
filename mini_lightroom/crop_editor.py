"""Рамка обрезки на кадре: затемнение снаружи, сетка-подсказка (трети, золотое сечение, спираль…), 8 ручек, перенос, пропорции.

В режиме обрезки окно показывает весь повёрнутый холст, поэтому рамка — в долях этого холста
(те же координаты, что params["crop"]). Рамка не выходит за снимок: engine.crop_valid.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from . import crop_overlays as CO
from . import engine as E

__all__ = ["CropEditor"]

HANDLE = 8


class CropEditor(QObject):
    changed = Signal()
    overlay_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.rect = [0.0, 0.0, 1.0, 1.0]
        self.angle = 0.0
        self.aspect: float | None = None   # ширина/высота в пикселях; None — свободно
        self.size = (1.0, 1.0)             # полный кадр W×H в пикселях
        self._drag = None
        self.overlay = "thirds"  # вид сетки-подсказки (crop_overlays.OVERLAYS)
        self.flip = 0            # положение спирали/треугольника: 4 зеркала

    def active(self) -> bool:
        return True

    def set_overlay(self, kind: str) -> None:
        if kind in dict(CO.OVERLAYS) and kind != self.overlay:
            self.overlay, self.flip = kind, 0
            self.overlay_changed.emit(kind)

    def cycle_overlay(self, step: int = 1) -> None:
        ids = [k for k, _ in CO.OVERLAYS]
        self.set_overlay(ids[(ids.index(self.overlay) + step) % len(ids)])

    def rotate_overlay(self) -> bool:
        """Зеркалит спираль/треугольник; у остальных сеток поворачивать нечего (False)."""
        if self.overlay not in CO.ROTATABLE:
            return False
        self.flip = (self.flip + 1) % 4
        return True

    # ---------- геометрия рамки

    def _px(self, view, xn, yn) -> QPointF:
        """Доля холста → точка виджета (в режиме обрезки холст = то, что показано)."""
        s = view.scale()
        return QPointF(view.width() / 2 + (xn * view.src_w - view.cx) * s,
                       view.height() / 2 + (yn * view.src_h - view.cy) * s)

    def _norm(self, view, pos) -> tuple[float, float]:
        s = view.scale()
        return ((view.cx + (pos.x() - view.width() / 2) / s) / view.src_w,
                (view.cy + (pos.y() - view.height() / 2) / s) / view.src_h)

    def _handles(self, view) -> dict[str, QPointF]:
        x0, y0, x1, y1 = self.rect
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        pts = {"nw": (x0, y0), "n": (xm, y0), "ne": (x1, y0), "e": (x1, ym),
               "se": (x1, y1), "s": (xm, y1), "sw": (x0, y1), "w": (x0, ym)}
        return {k: self._px(view, *v) for k, v in pts.items()}

    def _try(self, rect) -> bool:
        W, H = self.size
        if E.crop_valid(W, H, rect, self.angle):
            self.rect = list(rect)
            return True
        return False

    # ---------- мышь

    def press(self, e, view) -> bool:
        if e.button() != Qt.LeftButton or not view.pix:
            return False
        pos = e.position()
        for name, p in self._handles(view).items():
            if math.dist((p.x(), p.y()), (pos.x(), pos.y())) <= HANDLE + 4:
                self._drag = (name, self._norm(view, pos), list(self.rect))
                return True
        x, y = self._norm(view, pos)
        x0, y0, x1, y1 = self.rect
        if x0 < x < x1 and y0 < y < y1:
            self._drag = ("move", (x, y), list(self.rect))
            return True
        return False

    def move(self, e, view) -> bool:
        if not self._drag:
            return False
        what, (sx, sy), r0 = self._drag
        x, y = self._norm(view, e.position())
        x0, y0, x1, y1 = r0
        if what == "move":
            dx, dy = x - sx, y - sy
            dx = min(max(dx, -x0), 1 - x1)
            dy = min(max(dy, -y0), 1 - y1)
            for f in (1.0, 0.5, 0.25, 0.0):  # при повороте край может упереться — двигаем, сколько можно
                if self._try([x0 + dx * f, y0 + dy * f, x1 + dx * f, y1 + dy * f]):
                    break
        else:
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            if "w" in what:
                nx0 = min(x, x1 - 0.02)
            if "e" in what:
                nx1 = max(x, x0 + 0.02)
            if "n" in what:
                ny0 = min(y, y1 - 0.02)
            if "s" in what:
                ny1 = max(y, y0 + 0.02)
            if self.aspect:
                nx0, ny0, nx1, ny1 = self._keep_aspect(what, r0, [nx0, ny0, nx1, ny1])
            new = [max(0.0, nx0), max(0.0, ny0), min(1.0, nx1), min(1.0, ny1)]
            if not self._try(new) and self.aspect is None:
                # без пропорций: подвигаем каждую сторону отдельно, чтобы рамка «липла» к краю снимка
                for i in range(4):
                    trial = list(self.rect)
                    trial[i] = new[i]
                    self._try(trial)
        self.changed.emit()
        view.update()
        return True

    def _keep_aspect(self, what, r0, r):
        W, H = self.size
        k = self.aspect * H / W  # пропорция в долях холста: ширина = высота · k
        x0, y0, x1, y1 = r
        if what in ("e", "w"):  # тянем бок: высота следует за шириной, симметрично от центра
            h = (x1 - x0) / k
            cy = (r0[1] + r0[3]) / 2
            return x0, cy - h / 2, x1, cy + h / 2
        if what in ("n", "s"):
            w = (y1 - y0) * k
            cx = (r0[0] + r0[2]) / 2
            return cx - w / 2, y0, cx + w / 2, y1
        w, h = x1 - x0, y1 - y0  # угол: берём большее из движений, противоположный угол стоит
        if w / k > h:
            h = w / k
        else:
            w = h * k
        nx0 = r0[2] - w if "w" in what else r0[0]
        ny0 = r0[3] - h if "n" in what else r0[1]
        return nx0, ny0, nx0 + w, ny0 + h

    def release(self, e, view) -> bool:
        if self._drag:
            self._drag = None
            self.changed.emit()
            return True
        return False

    # ---------- рисование

    def paint(self, p: QPainter, view):
        if not view.pix:
            return
        x0, y0, x1, y1 = self.rect
        a, b = self._px(view, x0, y0), self._px(view, x1, y1)
        box = QRectF(a, b)
        full = view.to_widget(QRectF(0, 0, view.src_w, view.src_h))
        dim = QColor(0, 0, 0, 150)
        p.fillRect(QRectF(full.left(), full.top(), full.width(), box.top() - full.top()), dim)
        p.fillRect(QRectF(full.left(), box.bottom(), full.width(), full.bottom() - box.bottom()), dim)
        p.fillRect(QRectF(full.left(), box.top(), box.left() - full.left(), box.height()), dim)
        p.fillRect(QRectF(box.right(), box.top(), full.right() - box.right(), box.height()), dim)
        p.setRenderHint(QPainter.Antialiasing)
        spiral = self.overlay == "spiral"
        p.setPen(QPen(QColor(255, 255, 255, 190 if spiral else 110), 1.8 if spiral else 1))
        for path in CO.lines(self.overlay, self.flip, box.width() / max(box.height(), 1e-6)):
            p.drawPolyline(QPolygonF([QPointF(box.left() + x * box.width(), box.top() + y * box.height())
                                      for x, y in path]))
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawRect(box)
        p.setPen(QPen(QColor(0, 0, 0, 180), 1.5))
        p.setBrush(QColor("#ffffff"))
        for q in self._handles(view).values():
            p.drawRect(QRectF(q.x() - HANDLE / 2, q.y() - HANDLE / 2, HANDLE, HANDLE))
        p.setBrush(Qt.NoBrush)
