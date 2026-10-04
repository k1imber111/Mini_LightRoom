"""Окно: авто-кадр по композиции — варианты, примерка, применение, отмена.
Без сегментации и лиц (быстро и без сети): главное находит заметность. Запуск: python tests\\test_ui_compose.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")  # иначе offscreen рисует текст квадратиками
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mini_lightroom import engine as E  # noqa: E402
from mini_lightroom.ui import MainWindow, apply_dark_theme  # noqa: E402


def wait(app, cond, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise TimeoutError("не дождались")


def frame(cx: int = 1100, cy: int = 420) -> np.ndarray:
    """Тёмное небо с мягким градиентом и яркий диск (как луна) не в центре: главный объект находится сам."""
    h, w = 1000, 1500
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(np.array([0.02, 0.04, 0.12], np.float32) * (1 - y) + np.array([0.08, 0.08, 0.16], np.float32) * y,
                          (h, w, 3)).copy()
    cv2.circle(img, (cx, cy), 110, (0.95, 0.93, 0.85), -1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    import mini_lightroom.ui as ui
    ui.G.available = lambda: False   # без SegFormer
    ui.FC.available = lambda: False  # без mediapipe
    tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        E.save_jpeg(d / "кадр.jpg", frame(), 95)
        win = MainWindow()
        win.resize(1500, 900)
        win.show()
        win.load_folder(d)
        wait(app, lambda: win.base is not None and not win.rendering)
        win.crop_cam.setChecked(False)

        win.a_compose.trigger()
        wait(app, lambda: not win.compose_busy)
        vs = win.compose_variants
        assert win.crop_mode and vs, "авто-кадр не дал вариантов"
        assert win.crop_carousel.isVisible() and win.crop_carousel.count() == len(vs) + 1 and not win.qbar.isVisible()
        print("варианты:", [(v["label"], round(v["score"], 2), v["why"]) for v in vs])
        for v in vs:
            assert E.crop_valid(*win.full_wh, v["rect"], 0)
            assert v["rect"][0] <= 0.73 <= v["rect"][2] and v["rect"][1] <= 0.42 <= v["rect"][3], "объект выпал из кадра"
        app.processEvents()
        win.grab()

        # примерка варианта: рамка и сетка встают по варианту
        win.crop_carousel.setCurrentRow(1)
        assert win.cropper.rect == list(vs[0]["rect"]) and win.cropper.overlay == vs[0]["overlay"]
        if len(sys.argv) > 1:  # снимок окна для глаз: python tests	est_ui_compose.py папка
            out = Path(sys.argv[1])
            out.mkdir(parents=True, exist_ok=True)
            win.crop_carousel.setCurrentRow(2)
            app.processEvents()
            win.grab().save(str(out / "compose.png"))
        win.crop_carousel.setCurrentRow(0)  # исходный кадр
        assert win.cropper.rect == [0.0, 0.0, 1.0, 1.0]
        win.crop_carousel.setCurrentRow(2 if len(vs) > 1 else 1)
        picked = list(win.cropper.rect)
        # Enter: применить; карусель убирается, обрезка в правках, шаг истории
        win.a_crop.trigger()
        assert not win.crop_mode and not win.crop_carousel.isVisible()
        assert win.params["crop"] is not None and all(abs(a - b) < 1e-4 for a, b in zip(win.params["crop"], picked, strict=True))
        win.commit_history()
        win.undo()
        assert win.params["crop"] is None, "Ctrl+Z не вернул полный кадр"

        # Esc: примерили и передумали — кадр как был
        win.a_compose.trigger()
        wait(app, lambda: not win.compose_busy and win.crop_carousel.count() > 1)
        win.crop_carousel.setCurrentRow(1)
        win.cancel_crop()
        assert win.params["crop"] is None and not win.crop_mode and not win.crop_carousel.isVisible()
        # рамка во весь кадр не «заблокирована»: тянем новую; вне рамки — тоже; пропорции держатся; сетка включает обрезку
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        win.cropper.aspect = None
        win.overlay_combo.setCurrentIndex(1)
        win.on_overlay_pick(1)  # золотая сетка: режим обрезки включается сам
        assert win.crop_mode and win.cropper.overlay == "phi"
        win.cropper.rect = [0.0, 0.0, 1.0, 1.0]
        v = win.view

        def drag(a, b):
            p0, p1 = win.cropper._px(v, *a).toPoint(), win.cropper._px(v, *b).toPoint()
            QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, p0)
            for k in range(1, 6):
                QTest.mouseMove(v, p0 + (p1 - p0) * k / 5)
            QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, p1)

        drag((0.2, 0.2), (0.7, 0.6))
        r = win.cropper.rect
        assert abs(r[0] - 0.2) < 0.01 and abs(r[1] - 0.2) < 0.01 and abs(r[2] - 0.7) < 0.02 and abs(r[3] - 0.6) < 0.02, r
        before = list(r)
        drag((0.45, 0.4), (0.55, 0.45))  # теперь рамка меньше кадра — двигается, а не рисуется заново
        assert win.cropper.rect[2] - win.cropper.rect[0] > 0.45 and win.cropper.rect != before
        win.cropper.aspect = 1.0
        drag((0.9, 0.9), (0.4, 0.5))  # вне рамки, снизу справа налево вверх; квадрат по большему движению
        r = win.cropper.rect
        assert abs((r[2] - r[0]) * 1500 - (r[3] - r[1]) * 1000) < 6, r
        assert E.crop_valid(1500, 1000, r, 0)
        win.cancel_crop()

        # ползунок: при протяжке показывается быстрый кадр (уменьшенный), через SETTLE_MS — полный
        win.rows["exposure"].slider.setValue(win.rows["exposure"].slider.value() + 30)
        wait(app, lambda: win.after.width() <= ui.FAST_SIDE + 1, 10)
        assert win.settle_timer.isActive() or win.after.width() <= ui.FAST_SIDE + 1
        wait(app, lambda: win.after.width() > ui.FAST_SIDE * 1.5, 10)
        assert win.after.width() >= 1300 and not win.dirty
        # быстрый проход не теряет последнее значение: серия правок → итог соответствует последнему
        for v in (10, 20, 30, 40, 50):
            win.rows["contrast"].slider.setValue(v)
        wait(app, lambda: not win.rendering and not win.dirty and not win.settle_timer.isActive() and win.after.width() > 1300, 10)
        assert win.params["contrast"] == 50
        win.close()
    print("авто-кадр в окне OK")
