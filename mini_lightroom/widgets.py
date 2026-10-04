"""Общие виджеты окна: сворачиваемая секция панели и всплывающее уведомление. Без бизнес-логики."""
from __future__ import annotations

import ctypes

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import tool_icon

__all__ = ["HoverGlow", "Section", "Toast", "fade_in", "install_hover", "set_animations", "system_animations"]

ANIMATIONS = True  # False — всё появляется и сворачивается мгновенно
_MAX = 16777215  # QWIDGETSIZE_MAX


def set_animations(on: bool) -> None:
    global ANIMATIONS
    ANIMATIONS = on


def system_animations() -> bool:
    """Включены ли анимации в Windows (Параметры → Специальные возможности → Визуальные эффекты)."""
    try:
        flag = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(flag), 0)  # SPI_GETCLIENTAREAANIMATION
        return bool(flag.value)
    except (AttributeError, OSError):  # не Windows
        return True


def fade_in(w: QWidget, ms: int = 180) -> None:
    """Мягкое появление виджета: прозрачность 0 → 1, потом эффект снимается (рисование снова обычное)."""
    if not ANIMATIONS:
        return
    fx = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(fx)
    a = QPropertyAnimation(fx, b"opacity", w)
    a.setDuration(ms)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)
    a.finished.connect(lambda: w.setGraphicsEffect(None))
    a.start(QAbstractAnimation.DeleteWhenStopped)


class HoverGlow(QWidget):
    """Плавное осветление кнопки при наведении. QSS не умеет переходов, поэтому поверх кнопки лежит прозрачный
    дочерний виджет, который сам рисует полупрозрачную заливку с анимированной силой и не ловит мышь."""

    def __init__(self, btn: QWidget):
        super().__init__(btn)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.t = 0.0
        self._anim = QVariantAnimation(self, duration=140, easingCurve=QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._set)
        btn.installEventFilter(self)
        self.setGeometry(btn.rect())
        self.show()

    def _set(self, v) -> None:
        self.t = float(v)
        self.update()

    def _to(self, target: float) -> None:
        self._anim.stop()
        if not ANIMATIONS:
            self._set(target)
            return
        self._anim.setStartValue(self.t)
        self._anim.setEndValue(target)
        self._anim.start()

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Enter:
            self._to(1.0)
        elif t in (QEvent.Leave, QEvent.Hide, QEvent.EnabledChange):
            self._to(0.0)
        elif t == QEvent.Resize:
            self.setGeometry(self.parentWidget().rect())
        return False

    def paintEvent(self, _):
        btn = self.parentWidget()
        if self.t < 0.01 or not btn.isEnabled():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, int(32 * self.t)))
        r = btn.property("glowRadius") or min(btn.height() / 2, 15)
        p.drawRoundedRect(self.rect(), r, r)


def install_hover(root: QWidget) -> None:
    """Свечение при наведении на все кнопки внутри root (повторный вызов безопасен); у QPushButton фокус только
    с клавиатуры, чтобы после щелчка мышью на кнопке не оставалось кольца фокуса."""
    for b in root.findChildren(QPushButton) + root.findChildren(QToolButton):
        if b.findChild(HoverGlow) is None:
            HoverGlow(b)
        if isinstance(b, QPushButton):
            b.setFocusPolicy(Qt.TabFocus)


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
        self.head.setProperty("glowRadius", 12)
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
        a = self.anchor
        top_left = a.mapTo(self.parentWidget(), a.rect().topLeft())  # якорь может лежать глубже окна
        self.setMaximumWidth(max(240, int(a.width() * 0.7)))
        self.adjustSize()
        self.move(top_left.x() + (a.width() - self.width()) // 2, top_left.y() + a.height() - self.height() - 24)

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
