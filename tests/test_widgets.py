"""Виджеты и анимации: свечение кнопок, мягкое появление, сворачиваемая секция, всплывающее сообщение.
Запуск: python tests\\test_widgets.py"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget  # noqa: E402

from mini_lightroom import widgets as WD  # noqa: E402
from mini_lightroom.theme import apply_theme  # noqa: E402


def pump(app, sec):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)
    WD.set_animations(True)
    assert isinstance(WD.system_animations(), bool)

    # свечение: наведение плавно доводит силу до 1, уход — до 0, выключенная кнопка не светится
    root = QWidget()
    lay = QVBoxLayout(root)
    b = QPushButton("Кнопка")
    lay.addWidget(b)
    root.show()
    WD.install_hover(root)
    WD.install_hover(root)  # повторный вызов не плодит накладки
    assert len(b.findChildren(WD.HoverGlow)) == 1
    glow = b.findChild(WD.HoverGlow)
    assert b.focusPolicy().name == "TabFocus", "фокус кнопки только с клавиатуры"
    app.sendEvent(b, QEvent(QEvent.Enter))
    pump(app, 0.04)
    assert 0 < glow.t < 1, glow.t  # анимация идёт, а не прыжок
    pump(app, 0.3)
    assert glow.t > 0.99
    app.sendEvent(b, QEvent(QEvent.Leave))
    pump(app, 0.4)
    assert glow.t < 0.01
    app.sendEvent(b, QEvent(QEvent.Enter))
    pump(app, 0.3)
    b.setEnabled(False)
    pump(app, 0.4)
    assert glow.t < 0.01, "выключенная кнопка не светится"
    b.setEnabled(True)
    root.grab()  # рисуется без ошибок
    WD.set_animations(False)
    app.sendEvent(b, QEvent(QEvent.Enter))
    assert glow.t == 1.0, "без анимаций — сразу"
    WD.set_animations(True)

    # мягкое появление: эффект снимается после показа
    lab = QLabel("полоса")
    lab.show()
    WD.fade_in(lab, 60)
    assert lab.graphicsEffect() is not None
    pump(app, 0.3)
    assert lab.graphicsEffect() is None and lab.isVisible()

    # секция: сворачивается и разворачивается плавно, состояние и сигнал
    sec = WD.Section("Заголовок", True)
    QVBoxLayout(sec.body).addWidget(QLabel("содержимое"))
    host = QWidget()
    QVBoxLayout(host).addWidget(sec)
    host.show()
    seen = []
    sec.toggled.connect(lambda t, on: seen.append((t, on)))
    sec.set_expanded(False)
    pump(app, 0.4)
    assert not sec.body.isVisible() and seen == [("Заголовок", False)]
    sec.set_expanded(True)
    pump(app, 0.4)
    assert sec.body.isVisible() and sec.body.maximumHeight() == 16777215 and seen[-1] == ("Заголовок", True)
    sec.set_expanded(False, animate=False, emit=False)
    assert not sec.body.isVisible() and len(seen) == 2

    # всплывающее сообщение: само исчезает; ms=0 держится до следующего
    win = QWidget()
    win.resize(600, 400)
    area = QWidget(win)
    area.setGeometry(0, 40, 600, 300)
    win.show()
    toast = WD.Toast(win, area)
    toast.show_text("Готово", 150)
    pump(app, 0.1)
    assert toast.isVisible()
    pump(app, 0.7)
    assert not toast.isVisible()
    toast.show_text("Идёт", 0)
    pump(app, 0.7)
    assert toast.isVisible()
    assert toast.geometry().bottom() <= area.geometry().bottom(), "сообщение внутри области якоря"
    toast.show_text("Следующее", 100)
    pump(app, 0.7)
    assert not toast.isVisible()
    print("виджеты и анимации OK")
