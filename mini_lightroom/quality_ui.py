"""Интерфейс отбора брака: плитки «где именно», полоса под фото, фильтры ленты, диалог корзины.

Только показ и сигналы: считает и хранит всё MainWindow (результаты — quality.py, флаги — sidecar).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import T, tool_icon
from .widgets import fade_in, install_hover

__all__ = [
    "LEVEL_COLOR",
    "LEVEL_TEXT",
    "SHORT",
    "TYPE_TEXT",
    "FilterBar",
    "QualityBar",
    "TrashDialog",
    "ZoomTile",
    "badge_icon",
    "describe",
    "numpy_pixmap",
    "set_level",
]

LEVEL_COLOR = {"bad": T["bad"], "doubt": T["warn"], "ok": T["ok"], "unknown": T["dim"]}
LEVEL_TEXT = {"bad": "Брак", "doubt": "Сомнительно", "ok": "Резкий", "unknown": "Не проверен"}
TYPE_TEXT = {"soft": "Расфокус", "motion": "Смаз от движения", "focus_miss": "Фокус мимо объекта",
             "eyes_closed": "Глаза закрыты"}


def set_level(w: QWidget, level: str) -> None:
    """Свойство `level` для QSS-селекторов [level="bad"] — с перерисовкой стиля."""
    w.setProperty("level", level)
    w.style().unpolish(w)
    w.style().polish(w)


def numpy_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy())


def badge_icon(base: QPixmap, kind: str | None) -> QIcon:
    """Миниатюра с кружком-статусом в углу: red/amber — брак/сомнительно, reject/pick — решение пользователя."""
    pm = QPixmap(base)
    p = QPainter(pm)
    if kind in (None, "ok", "unknown"):
        p.end()
        return _icon(pm)
    color = {"bad": T["bad"], "doubt": T["warn"], "reject": T["bad"], "pick": T["ok"]}[kind]
    p.setRenderHint(QPainter.Antialiasing)
    r = max(8, pm.width() // 11)
    p.setPen(QPen(QColor(T["canvas"]), 2))
    p.setBrush(QColor(color))
    p.drawEllipse(QRectF(8, 8, r * 2, r * 2))
    if kind in ("reject", "pick"):
        p.setPen(QPen(QColor(T["on_accent"]), max(2, r // 4), Qt.SolidLine, Qt.RoundCap))
        if kind == "reject":
            p.drawLine(int(8 + r * 0.55), int(8 + r * 0.55), int(8 + r * 1.45), int(8 + r * 1.45))
            p.drawLine(int(8 + r * 1.45), int(8 + r * 0.55), int(8 + r * 0.55), int(8 + r * 1.45))
        else:
            p.drawPolyline(QPolygonF([QPointF(8 + r * 0.5, 8 + r * 1.05), QPointF(8 + r * 0.9, 8 + r * 1.45),
                                      QPointF(8 + r * 1.55, 8 + r * 0.6)]))
    p.end()
    return _icon(pm)


def _icon(pm: QPixmap) -> QIcon:
    """Иконка без тонирования при выделении: Qt иначе красит выбранную миниатюру цветом выделения."""
    ic = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):
        ic.addPixmap(pm, mode)
    return ic


SHORT = {"soft": "расфокус", "motion": "смаз", "focus_miss": "фокус мимо", "eyes_closed": "глаза"}


def describe(res: dict | None, flag: str | None) -> tuple[str, str, str]:
    """(уровень, заголовок, подробность) для полосы и подписи в ленте. Решение пользователя важнее авто-вердикта."""
    if flag == "reject":
        return "bad", "Брак (ваш выбор)", "Помечен вами; отменить — клавиша X"
    if flag == "pick":
        return "ok", "Оставлен вами", "Не попадёт в корзину; отменить — клавиша U"
    if not res:
        return "unknown", "Проверка…", ""
    verdict = res.get("verdict", "unknown")
    parts = []
    for d in res.get("defects", []):
        what = TYPE_TEXT.get(d["type"], d["type"])
        if d["type"] == "motion" and "angle" in d:
            what += f" {d['angle']:.0f}°"
        elif d.get("sigma") is not None:
            what += f": размытие {d['sigma']:.1f} px"
        parts.append(f"{what} — {d['name'].lower()}")
    if parts:
        return verdict, LEVEL_TEXT[verdict], "; ".join(parts)
    if verdict == "ok":
        sig = [r["sigma"] for r in res.get("regions", []) if r.get("sigma") is not None]
        return "ok", "Резкий", f"размытие {min(sig):.1f} px" if sig else ""
    return "unknown", "Нечем судить", "мало деталей (небо, туман) — проверьте глазами"


class ZoomTile(QWidget):
    """Плитка «где именно»: крупный (100%) вырез места с браком, заголовок цветом серьёзности, подпись."""
    clicked = Signal(object)
    W, H, IW, IH = 252, 218, 240, 160

    def __init__(self, pix: QPixmap, title: str, sub: str, level: str, roi: list[float] | None):
        super().__init__()
        self.pix, self.title, self.sub, self.level, self.roi = pix, title, sub, level, roi
        self._hover = False
        self.setFixedSize(self.W, self.H)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Щелчок — показать это место на кадре в 100%")

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.roi:
            self.clicked.emit(self.roi)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        col = QColor(LEVEL_COLOR.get(self.level, T["dim"]))
        card = QPainterPath()
        card.addRoundedRect(QRectF(0.5, 0.5, self.W - 1, self.H - 1), 12, 12)
        p.fillPath(card, QColor(T["panel"]))
        col.setAlpha(255 if self._hover else 150)
        p.setPen(QPen(col, 2 if self._hover else 1.2))
        p.drawPath(card)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(6, 6, self.IW, self.IH), 8, 8)
        p.setClipPath(clip)
        p.drawPixmap(6, 6, self.pix)
        p.setClipping(False)
        f = QFont(self.font())
        f.setPixelSize(13)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        col.setAlpha(255)
        p.setPen(col)
        p.drawText(QRectF(10, self.IH + 9, self.W - 20, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        f.setPixelSize(12)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(T["text2"]))
        p.drawText(QRectF(10, self.IH + 29, self.W - 20, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   p.fontMetrics().elidedText(self.sub, Qt.ElideRight, self.W - 20))


class QualityBar(QFrame):
    """Полоса «Контроль качества» под фото: статус, причина, плитки с крупными местами брака, решение."""
    keep = Signal()
    reject = Signal()
    zoom_to = Signal(object)

    def __init__(self):
        super().__init__()
        self.setObjectName("qbar")
        self.pill = QLabel("Проверка…")
        self.pill.setObjectName("qpill")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color:{T['text2']}")
        self.b_keep = QPushButton("Оставить")
        self.b_keep.setIcon(tool_icon("check"))
        self.b_keep.setToolTip("Не бракуем этот кадр (U)")
        self.b_keep.clicked.connect(self.keep)
        self.b_reject = QPushButton("В брак")
        self.b_reject.setIcon(tool_icon("x"))
        self.b_reject.setToolTip("Пометить браком (X). В корзину уходит отдельной кнопкой с подтверждением")
        self.b_reject.clicked.connect(self.reject)
        head = QHBoxLayout()
        head.addWidget(self.pill)
        head.addWidget(self.summary, 1)
        head.addWidget(self.b_keep)
        head.addWidget(self.b_reject)
        self.tiles = QHBoxLayout()
        self.tiles.setSpacing(10)
        self.tiles.addStretch()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(10)
        lay.addLayout(head)
        lay.addLayout(self.tiles)
        self.hide()

    def show_result(self, res: dict | None, flag: str | None) -> None:
        level, title, text = describe(res, flag)
        self.pill.setText(title)
        set_level(self.pill, level)
        self.summary.setText(text)
        self.b_reject.setEnabled(flag != "reject")
        self.b_keep.setEnabled(flag != "pick")
        if not self.isVisible():
            self.setVisible(True)
            fade_in(self)

    def set_tiles(self, tiles: list[ZoomTile]) -> None:
        while self.tiles.count() > 1:
            w = self.tiles.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for t in tiles:
            t.clicked.connect(self.zoom_to)
            self.tiles.insertWidget(self.tiles.count() - 1, t)


_FILTER_TIPS = {"all": "Все кадры папки", "bad": "Брак: расфокус, смаз, закрытые глаза и помеченные вами",
                "doubt": "Сомнительные: проверьте глазами", "ok": "Резкие и оставленные вами"}


class FilterBar(QWidget):
    """Чипы над лентой: все / брак / сомнительно / хорошие, со счётчиками."""
    changed = Signal(str)
    MODES = (("all", "Все"), ("bad", "Брак"), ("doubt", "Сомнит."), ("ok", "Резкие"))

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 2)
        lay.setSpacing(4)
        self.btns: dict[str, QPushButton] = {}
        for key, label in self.MODES:
            b = QPushButton(label)
            b.setProperty("chip", True)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(_FILTER_TIPS[key])
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            lay.addWidget(b)
            self.btns[key] = b
        self.btns["all"].setChecked(True)
        lay.addStretch()
        self.set_counts({})

    def mode(self) -> str:
        return next(k for k, b in self.btns.items() if b.isChecked())

    def set_counts(self, counts: dict[str, int]) -> None:
        for key, label in self.MODES:
            n = counts.get(key)
            self.btns[key].setText(label if key == "all" or n is None else f"{label} {n}")


class TrashDialog(QDialog):
    """Подтверждение отправки брака в корзину Windows: список с миниатюрами, галочки, размер."""

    def __init__(self, parent, rows: list[tuple[Path, QPixmap | None, str, int]]):
        super().__init__(parent)
        self.setWindowTitle("Брак в корзину")
        self.setMinimumSize(520, 460)
        self.paths = [r[0] for r in rows]
        self.sizes = {r[0]: r[3] for r in rows}
        self.head = QLabel()
        self.head.setWordWrap(True)
        self.list = QListWidget()
        self.list.setIconSize(QSize(88, 60))
        for path, pix, reason, size in rows:
            it = QListWidgetItem(QIcon(pix) if pix is not None else QIcon(), f"{path.name}\n{reason} · {size / 1e6:.0f} МБ")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, str(path))
            self.list.addItem(it)
        self.list.itemChanged.connect(self._recount)
        note = QLabel("Файлы уходят в корзину Windows — оттуда их можно вернуть. Правки этих кадров в программе "
                      "удаляются. Без вашего подтверждения ничего не удаляется.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{T['text2']}")
        self.ok = QPushButton("В корзину")
        self.ok.setProperty("primary", True)
        self.ok.setIcon(tool_icon("trash", T["on_accent"]))
        cancel = QPushButton("Отмена")
        self.ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(self.ok)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        for w in (self.head, self.list, note):
            lay.addWidget(w, 1 if w is self.list else 0)
        lay.addLayout(row)
        self._recount()
        install_hover(self)

    def chosen(self) -> list[Path]:
        return [Path(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def _recount(self, *_):
        sel = self.chosen()
        mb = sum(self.sizes[p] for p in sel) / 1e6
        self.head.setText(f"В корзину: {len(sel)} из {len(self.paths)} кадров, {mb:.0f} МБ")
        self.ok.setEnabled(bool(sel))
