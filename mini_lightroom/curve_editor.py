"""Тональная кривая: точки тянутся мышью, щелчок — новая точка, двойной щелчок — удалить.

Точки в шкале 0..255, как в Lightroom; форма кривой — та же монотонная кубика, что
в движке (engine.curve_lut), поэтому на экране ровно то, что попадёт в кадр.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from . import engine as E

__all__ = ["CURVE_SHAPES", "CurveEditor"]

IDENTITY = [[0, 0], [255, 255]]
COLORS = {"rgb": "#e6e6e6", "r": "#ff6b6b", "g": "#6bff8a", "b": "#6b9dff"}
# Готовые формы общей кривой: подпись → точки.
CURVE_SHAPES = {
    "Линейная": IDENTITY,
    "Мягкий контраст": [[0, 0], [64, 56], [192, 200], [255, 255]],
    "Сильный контраст": [[0, 0], [64, 44], [192, 214], [255, 255]],
    "Матовая (плёнка)": [[0, 28], [64, 72], [192, 196], [255, 240]],
    "Светлая, воздушная": [[0, 16], [96, 118], [192, 214], [255, 250]],
    "Мягкие тени": [[0, 24], [48, 60], [128, 132], [255, 255]],
}


class CurveEditor(QWidget):
    changed = Signal(str, object)  # канал, точки (None — линейная)

    def __init__(self):
        super().__init__()
        self.channel = "rgb"
        self.curves: dict[str, list] = {}
        self.hist = None
        self._drag: int | None = None
        self.setMinimumHeight(190)
        self.setMouseTracking(True)
        self.setToolTip("Тяните точки. Щелчок — новая точка, двойной щелчок по точке — удалить.\n"
                        "Выше кривая — светлее этот тон, ниже — темнее.")

    # ---------- данные

    def points(self) -> list:
        return self.curves.get(self.channel) or [list(p) for p in IDENTITY]

    def set_curves(self, curves: dict):
        self.curves = {ch: [list(p) for p in pts] for ch, pts in (curves or {}).items() if pts}
        self.update()

    def set_channel(self, ch: str):
        self.channel = ch
        self.update()

    def set_points(self, pts):
        """Точки текущего канала (из готовой формы или «сброс»)."""
        self._store([list(p) for p in pts])

    def _store(self, pts):
        is_id = [list(p) for p in pts] == IDENTITY
        if is_id:
            self.curves.pop(self.channel, None)
        else:
            self.curves[self.channel] = pts
        self.update()
        self.changed.emit(self.channel, None if is_id else [list(p) for p in pts])

    # ---------- геометрия

    def _box(self) -> QRectF:
        side = min(self.width() - 12, self.height() - 12)
        return QRectF((self.width() - side) / 2, 6, side, side)

    def _to_w(self, x, y) -> QPointF:
        b = self._box()
        return QPointF(b.left() + x / 255 * b.width(), b.bottom() - y / 255 * b.height())

    def _to_v(self, pos) -> tuple[int, int]:
        b = self._box()
        x = (pos.x() - b.left()) / b.width() * 255
        y = (b.bottom() - pos.y()) / b.height() * 255
        return round(min(max(x, 0), 255)), round(min(max(y, 0), 255))

    def _hit(self, pos) -> int | None:
        for i, (x, y) in enumerate(self.points()):
            q = self._to_w(x, y)
            if math.dist((q.x(), q.y()), (pos.x(), pos.y())) <= 8:
                return i
        return None

    # ---------- мышь

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        i = self._hit(e.position())
        pts = [list(p) for p in self.points()]
        if i is None:  # новая точка на кривой под курсором
            x, _ = self._to_v(e.position())
            if any(abs(x - p[0]) < 6 for p in pts):
                return
            y = round(float(E.curve_lut(pts, 256)[x]) * 255)
            pts.append([x, y])
            pts.sort()
            i = pts.index([x, y])
            self._store(pts)
        self._drag = i

    def mouseMoveEvent(self, e):
        if self._drag is None:
            self.setCursor(Qt.PointingHandCursor if self._hit(e.position()) is not None else Qt.CrossCursor)
            return
        pts = [list(p) for p in self.points()]
        i = self._drag
        x, y = self._to_v(e.position())
        lo = pts[i - 1][0] + 4 if i > 0 else 0          # точки не перескакивают соседей
        hi = pts[i + 1][0] - 4 if i < len(pts) - 1 else 255
        if i == 0 or i == len(pts) - 1:  # концы — только вверх/вниз (чёрная и белая точка)
            x = pts[i][0]
        pts[i] = [min(max(x, lo), hi), y]
        self._store(pts)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        i = self._hit(e.position())
        pts = [list(p) for p in self.points()]
        if i is not None and 0 < i < len(pts) - 1:  # крайние точки не удаляются
            del pts[i]
            self._store(pts)

    # ---------- рисование

    def paintEvent(self, _):
        p = QPainter(self)
        b = self._box()
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        p.fillRect(b, QColor("#161616"))
        if self.hist:  # гистограмма яркости на фоне
            lum = np.sqrt(np.array(self.hist[0]) * 0.3 + np.array(self.hist[1]) * 0.59 + np.array(self.hist[2]) * 0.11)
            top = lum.max() or 1
            path = QPainterPath()
            path.moveTo(b.left(), b.bottom())
            for i, v in enumerate(lum):
                path.lineTo(b.left() + i / (len(lum) - 1) * b.width(), b.bottom() - v / top * b.height() * 0.9)
            path.lineTo(b.right(), b.bottom())
            p.fillPath(path, QColor(255, 255, 255, 28))
        p.setPen(QPen(QColor("#2e2e2e"), 1))
        for k in (1, 2, 3):
            x = b.left() + b.width() * k / 4
            y = b.top() + b.height() * k / 4
            p.drawLine(QPointF(x, b.top()), QPointF(x, b.bottom()))
            p.drawLine(QPointF(b.left(), y), QPointF(b.right(), y))
        p.setPen(QPen(QColor("#3a3a3a"), 1, Qt.DashLine))
        p.drawLine(b.bottomLeft(), b.topRight())
        p.setRenderHint(QPainter.Antialiasing)
        for ch in ("r", "g", "b", "rgb"):  # неактивные каналы бледно, активный ярко
            if ch != self.channel and ch in self.curves:
                self._draw_curve(p, self.curves[ch], QColor(COLORS[ch]), 1.0, 80)
        pts = self.points()
        self._draw_curve(p, pts, QColor(COLORS[self.channel]), 2.0, 255)
        p.setPen(QPen(QColor(0, 0, 0, 200), 1.5))
        p.setBrush(QColor(COLORS[self.channel]))
        for x, y in pts:
            p.drawEllipse(self._to_w(x, y), 4.5, 4.5)

    def _draw_curve(self, p, pts, color, width, alpha):
        lut = E.curve_lut(pts, 128)
        color.setAlpha(alpha)
        p.setPen(QPen(color, width))
        path = QPainterPath(self._to_w(0, lut[0] * 255))
        for i, v in enumerate(lut):
            path.lineTo(self._to_w(i / (len(lut) - 1) * 255, v * 255))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
