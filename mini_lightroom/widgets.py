"""Общие виджеты окна: сворачиваемая секция панели и всплывающее уведомление. Без бизнес-логики."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .theme import tool_icon

__all__ = ["Section", "Toast", "set_animations"]

ANIMATIONS = True  # False — всё появляется и сворачивается мгновенно
_MAX = 16777215  # QWIDGETSIZE_MAX


def set_animations(on: bool) -> None:
    global ANIMATIONS
    ANIMATIONS = on


class Section(QFrame):
    """Группа панели: заголовок со стрелкой + тело. Содержимое кладут в `section.body` (QVBoxLayout(section.body))."""
    toggled = Signal(str, bool)

    def __init__(self, title: str, expanded: bool = True):
        super().__init__()
        self.setObjectName("section")
        self.title = title
        self._open = expanded
        self.head = QToolButton()
        self.head.setObjectName("sectionHead")
        self.head.setText(title)
        self.head.setCheckable(True)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.head.setCursor(Qt.PointingHandCursor)
        self.head.setToolTip("Свернуть / развернуть")
        self.body = QFrame()
        self.body.setObjectName("sectionBody")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.head)
        lay.addWidget(self.body)
        self._anim = QPropertyAnimation(self.body, b"maximumHeight", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._finish)
        self.set_expanded(expanded, animate=False, emit=False)
        self.head.clicked.connect(lambda: self.set_expanded(not self._open))

    def set_expanded(self, on: bool, animate: bool = True, emit: bool = True) -> None:
        self._open = on
        self.head.setChecked(on)
        self.head.setIcon(tool_icon("chevron-down" if on else "chevron-right", size=16))
        self._anim.stop()
        if animate and ANIMATIONS and self.isVisible():
            self.body.setVisible(True)
            start = self.body.height() if self.body.maximumHeight() != 0 else 0
            self._anim.setStartValue(start)
            self._anim.setEndValue(self.body.sizeHint().height() if on else 0)
            self._anim.start()
        else:
            self.body.setMaximumHeight(_MAX if on else 0)
            self.body.setVisible(on)
        if emit:
            self.toggled.emit(self.title, on)

    def _finish(self) -> None:
        if self._open:
            self.body.setMaximumHeight(_MAX)  # дальше тело растёт и сжимается вместе с содержимым
        else:
            self.body.setVisible(False)


class Toast(QLabel):
    """Короткое сообщение внизу окна: плавно появляется и исчезает, мышь не перехватывает."""

    def __init__(self, parent: QWidget, anchor: QWidget):
        """parent — окно (не QSplitter: он сделал бы сообщение своей панелью); anchor — область, внизу которой оно встаёт."""
        super().__init__(parent)
        self.anchor = anchor
        self.setObjectName("toast")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._anim = QPropertyAnimation(self._fx, b"opacity", self)
        self._anim.finished.connect(self._after)
        self._timer = QTimer(self, singleShot=True, timeout=self._fade_out)
        self._hiding = False
        anchor.installEventFilter(self)  # следим за размером области, чтобы сообщение оставалось внизу по центру
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.anchor and ev.type() == QEvent.Resize and self.isVisible():
            self.place()
        return False

    def show_text(self, text: str, ms: int = 4000) -> None:
        self._timer.stop()
        self._hiding = False
        self.setText(text)
        self.place()
        self.show()
        self.raise_()
        self._fade(1.0, 160)
        if ms > 0:  # 0 — держать, пока не придёт следующее сообщение
            self._timer.start(ms)

    def place(self) -> None:
        """Ставит по центру внизу родителя; вызывать при смене размера окна."""
        a = self.anchor.geometry()
        self.setMaximumWidth(max(240, int(a.width() * 0.7)))
        self.adjustSize()
        self.move(a.x() + (a.width() - self.width()) // 2, a.bottom() - self.height() - 28)

    def _fade(self, to: float, ms: int) -> None:
        self._anim.stop()
        if not ANIMATIONS:
            self._fx.setOpacity(to)
            self._after()
            return
        self._anim.setDuration(ms)
        self._anim.setStartValue(self._fx.opacity())
        self._anim.setEndValue(to)
        self._anim.start()

    def _fade_out(self) -> None:
        self._hiding = True
        self._fade(0.0, 240)

    def _after(self) -> None:
        if self._hiding:
            self.hide()
