"""Редактирование масок прямо на кадре: ручки градиентов, кисть, подсветка маски.

ImageView отдаёт сюда мышь и рисование; редактор меняет слой (dict из params["masks"])
и сообщает changed — окно перерисовывает кадр. Координаты слоя — доли полного кадра.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from . import masks as M

__all__ = ["MaskEditor"]

HANDLE = 7  # радиус ручки на экране, px


class MaskEditor(QObject):
    changed = Signal()          # геометрия слоя изменилась — нужен рендер
    created = Signal(object)    # новый градиент нарисован на кадре (object: Signal(dict) передал бы копию)

    def __init__(self, resolve_arr):
        super().__init__()
        self.layer: dict | None = None
        self.creating: str | None = None      # "linear" | "radial": следующий жест создаёт слой
        self.brush_size = 30                  # радиус кисти в тысячных долях ширины кадра
        self.brush_feather = 50
        self.erase = False
        self.show_overlay = False
        self._resolve_arr = resolve_arr       # слой ИИ → его маска (uint8) или None
        self._drag = None                     # (что тянем, исходная точка, копия геометрии)
        self._stroke: dict | None = None
        self._hover: QPointF | None = None
        self._overlay: QImage | None = None

    # ---------- состояние

    def set_layer(self, layer: dict | None):
        self.layer, self.creating, self._drag = layer, None, None
        self.invalidate()

    def active(self) -> bool:
        """Забирает ли редактор левую кнопку мыши (иначе она двигает кадр)."""
        return bool(self.creating) or (self.layer is not None and self.layer.get("on", True))

    def invalidate(self):
        self._overlay = None

    def _emit(self):
        self.invalidate()
        self.changed.emit()

    # ---------- мышь (координаты виджета → доли кадра через view)

    def press(self, e, view) -> bool:
        if e.button() != Qt.LeftButton or not view.pix or not self.active():
            return False
        pt = view.to_norm(e.position())
        if self.creating:
            kind = self.creating
            self.creating = None
            if kind == "linear":
                self.layer = M.new_layer("linear", a=list(pt), b=list(pt))
                self._drag = ("b", pt, None)
            else:
                self.layer = M.new_layer("radial", c=list(pt), r=[0.0, 0.0])
                self._drag = ("r", pt, None)
            self.created.emit(self.layer)
            return True
        kind = self.layer["type"]
        if kind == "brush":
            erase = self.erase or bool(e.modifiers() & Qt.AltModifier)
            self._stroke = {"pts": [list(pt)], "r": self.brush_size / 1000, "erase": erase}
            self.layer.setdefault("strokes", []).append(self._stroke)
            self._emit()
            return True
        hit = self._hit(e.position(), view)
        if hit:
            self._drag = (hit, pt, {k: list(v) for k, v in self.layer.items() if k in ("a", "b", "c", "r")})
            return True
        return kind in ("linear", "radial")  # мимо ручек: не двигаем кадр случайно

    def move(self, e, view) -> bool:
        self._hover = e.position()
        if self._stroke is not None:
            pt = view.to_norm(e.position())
            last = view.to_widget_pt(*self._stroke["pts"][-1])
            if math.dist((last.x(), last.y()), (e.position().x(), e.position().y())) >= 2:
                self._stroke["pts"].append(list(pt))
                self._emit()
            return True
        if not self._drag:
            view.update()  # курсор кисти и подсветка ручек
            return False
        what, start, geom = self._drag
        pt = view.to_norm(e.position())
        lay = self.layer
        if what in ("a", "b", "c"):
            lay[what] = list(pt)
        elif what == "r":
            c = lay["c"]
            lay["r"] = [max(abs(pt[0] - c[0]), 0.005), max(abs(pt[1] - c[1]), 0.005)]
        elif what == "rx":
            lay["r"][0] = max(abs(pt[0] - lay["c"][0]), 0.005)
        elif what == "ry":
            lay["r"][1] = max(abs(pt[1] - lay["c"][1]), 0.005)
        elif what == "mid":  # тянем весь линейный градиент
            dx, dy = pt[0] - start[0], pt[1] - start[1]
            lay["a"] = [geom["a"][0] + dx, geom["a"][1] + dy]
            lay["b"] = [geom["b"][0] + dx, geom["b"][1] + dy]
        self._emit()
        return True

    def release(self, e, view) -> bool:
        if self._stroke is not None:
            self._stroke = None
            self._emit()
            return True
        if self._drag:
            lay = self.layer
            # Щелчок без протяжки: градиент разумного размера вместо нулевого.
            if lay["type"] == "linear" and math.dist(lay["a"], lay["b"]) < 0.01:
                lay["b"] = [lay["a"][0], min(1.0, lay["a"][1] + 0.3)]
            if lay["type"] == "radial" and max(lay["r"]) < 0.01:
                lay["r"] = [0.2, 0.2 * view.src_w / view.src_h]
            self._drag = None
            self._emit()
            return True
        return False

    def _handles(self, view) -> dict[str, QPointF]:
        lay = self.layer
        if not lay or not lay.get("on", True):
            return {}
        if lay["type"] == "linear":
            a, b = lay["a"], lay["b"]
            return {"a": view.to_widget_pt(*a), "b": view.to_widget_pt(*b),
                    "mid": view.to_widget_pt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)}
        if lay["type"] == "radial":
            (cx, cy), (rx, ry) = lay["c"], lay["r"]
            return {"c": view.to_widget_pt(cx, cy), "rx": view.to_widget_pt(cx + rx, cy),
                    "ry": view.to_widget_pt(cx, cy + ry)}
        return {}

    def _hit(self, pos: QPointF, view) -> str | None:
        for name, p in self._handles(view).items():
            if math.dist((p.x(), p.y()), (pos.x(), pos.y())) <= HANDLE + 4:
                return name
        return None

    # ---------- рисование поверх кадра

    def paint(self, p: QPainter, view):
        if not view.pix:
            return
        if self.show_overlay and self.layer:
            img = self._overlay_image(view)
            if img is not None:
                p.drawImage(view.to_widget(QRectF(0, 0, view.src_w, view.src_h)), img)
        lay = self.layer
        if lay and lay.get("on", True):
            p.setRenderHint(QPainter.Antialiasing)
            pen_dark = QPen(QColor(0, 0, 0, 160), 3)
            pen = QPen(QColor("#ffffff"), 1.4)
            h = self._handles(view)
            if lay["type"] == "linear":
                a, b = h["a"], h["b"]
                d = b - a
                n = QPointF(-d.y(), d.x())
                ln = math.hypot(n.x(), n.y()) or 1
                n = n * (2000 / ln)  # линии поперёк направления градиента — «от» и «до»
                for q, style in ((a, Qt.SolidLine), (h["mid"], Qt.DashLine), (b, Qt.SolidLine)):
                    for pp in (pen_dark, pen):
                        pp.setStyle(style)
                        p.setPen(pp)
                        p.drawLine(q - n, q + n)
            elif lay["type"] == "radial":
                c = h["c"]
                rx, ry = h["rx"].x() - c.x(), h["ry"].y() - c.y()
                inner = 1 - lay.get("feather", 50) / 100
                for pp in (pen_dark, pen):
                    p.setPen(pp)
                    p.drawEllipse(c, rx, ry)
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawEllipse(c, rx * inner, ry * inner)
                pen.setStyle(Qt.SolidLine)
            for q in h.values():
                p.setPen(QPen(QColor(0, 0, 0, 180), 1.5))
                p.setBrush(QColor("#ffffff"))
                p.drawEllipse(q, HANDLE / 2 + 1, HANDLE / 2 + 1)
            p.setBrush(Qt.NoBrush)
            if lay["type"] == "brush" and self._hover is not None:
                r = self.brush_size / 1000 * view.src_w * view.scale()
                erase = self.erase
                p.setPen(QPen(QColor(0, 0, 0, 160), 3))
                p.drawEllipse(self._hover, r, r)
                p.setPen(QPen(QColor("#ff8080" if erase else "#ffffff"), 1.2))
                p.drawEllipse(self._hover, r, r)

    def _overlay_image(self, view) -> QImage | None:
        if self._overlay is None:
            lay = dict(self.layer)
            if lay["type"] == "ai":
                lay["arr"] = self._resolve_arr(lay)
            h, w = view.pix.height(), view.pix.width()
            m = M.layer_mask(lay, h, w)
            if m is None:
                return None
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0] = 255
            rgba[..., 3] = (m * 150).astype(np.uint8)
            self._overlay = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        return self._overlay
