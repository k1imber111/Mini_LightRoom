"""Целевая правка: нажать на кадре и тянуть вверх/вниз — меняются ползунки HSL цвета под курсором
или точка тональной кривой для тона под курсором. Что именно менять, решает окно (колбэки).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

__all__ = ["TargetEditor"]


class TargetEditor(QObject):
    def __init__(self, on_start, on_drag):
        super().__init__()
        self._on_start = on_start   # (точка виджета) -> bool: удалось ли взять цвет/тон
        self._on_drag = on_drag     # (смещение вверх в пикселях экрана)
        self._y0: float | None = None
        self._hover: QPointF | None = None

    def active(self) -> bool:
        return True

    def press(self, e, view) -> bool:
        if e.button() != Qt.LeftButton or not view.pix:
            return False
        if self._on_start(e.position()):
            self._y0 = e.position().y()
        return True

    def move(self, e, view) -> bool:
        self._hover = e.position()
        if self._y0 is None:
            view.update()
            return False
        self._on_drag(self._y0 - e.position().y())
        return True

    def release(self, e, view) -> bool:
        was = self._y0 is not None
        self._y0 = None
        return was

    def paint(self, p: QPainter, view):
        if self._hover is None:
            return
        p.setRenderHint(QPainter.Antialiasing)
        c = self._hover
        for pen in (QPen(QColor(0, 0, 0, 170), 3), QPen(QColor("#ffffff"), 1.2)):
            p.setPen(pen)
            p.drawEllipse(c, 7, 7)
            p.drawLine(QPointF(c.x(), c.y() - 13), QPointF(c.x(), c.y() - 9))
            p.drawLine(QPointF(c.x(), c.y() + 9), QPointF(c.x(), c.y() + 13))
