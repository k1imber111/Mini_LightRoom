"""Снимки окна в PNG для проверки внешнего вида глазами (не тест на assert).
Запуск: python tests\\test_theme_shots.py [папка_для_png]   (по умолчанию — временная, путь печатается)"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")  # без этого offscreen рисует текст квадратиками
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mini_lightroom import engine as E  # noqa: E402
from mini_lightroom.ui import MainWindow, apply_dark_theme  # noqa: E402


def pump(app, sec):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def scene(i: int) -> np.ndarray:
    """Похоже на пейзаж: небо с градиентом, горизонт, тёмный передний план, «солнце»."""
    h, w = 600, 900
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = np.array([0.25, 0.4, 0.75], np.float32) * (1 - y) + np.array([1.0, 0.65, 0.35], np.float32) * y
    img = np.broadcast_to(sky, (h, w, 3)).copy() * (0.6 + 0.08 * i)
    img[int(h * 0.62):] = np.array([0.08, 0.1, 0.07], np.float32)
    cv2.circle(img, (int(w * (0.3 + 0.1 * i)), int(h * 0.5)), 40, (1.0, 0.95, 0.8), -1)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 2), 0, 1)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="снимки_"))
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    import mini_lightroom.ui as ui
    tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        for i in range(6):
            E.save_jpeg(d / f"DSC0{2890 + i}.jpg", scene(i))
        win = MainWindow()
        win.resize(1500, 900)
        win.show()
        win.load_folder(d)
        pump(app, 3)
        win.grab().save(str(out / "1_main.png"))
        for title in ("Обрезка и горизонт", "Маски", "Стиль с референса"):
            win.sections[title].set_expanded(True, animate=False, emit=False)
        pump(app, 0.3)
        win.panel.grab().save(str(out / "2_panel_expanded.png"))
        win.toast("Продолжаем: съёмка, кадр DSC02892.jpg", 6000)
        pump(app, 0.5)
        win.a_crop.trigger()
        pump(app, 1)
        win.grab().save(str(out / "3_crop_toast.png"))
        win.close()
    print("снимки:", *sorted(str(p) for p in out.glob("*.png")), sep="\n")
