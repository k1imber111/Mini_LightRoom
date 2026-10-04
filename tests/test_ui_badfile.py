"""Нечитаемый файл в папке не задерживает остальные кадры: миниатюры и проверка брака доходят до конца.
Запуск: python tests\test_ui_badfile.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication  # noqa: E402
from test_ui_cull import save, scene  # noqa: E402

import mini_lightroom.ui as ui  # noqa: E402
from mini_lightroom import quality as Q  # noqa: E402


def wait(app, cond, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise TimeoutError("не дождались")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui.apply_dark_theme(app)
    ui.FC.available = lambda: False
    Q.CFG["norm_long_side"] = 1500  # тестовые кадры — не 6000 px
    tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d, tempfile.TemporaryDirectory(prefix="другая_") as d2:
        d, d2 = Path(d), Path(d2)
        save(d / "a.jpg", scene(1))
        save(d / "b.jpg", scene(2))
        (d / "битый.jpg").write_bytes("это не картинка".encode())
        save(d2 / "a.jpg", scene(3))  # то же имя в другой папке
        win = ui.MainWindow()
        win.show()
        win.load_folder(d)
        wait(app, lambda: len(win.quality) == 3 and not win.q_left)
        assert win.thumb_failed == {"битый.jpg"} and sorted(win.thumbs) == ["a.jpg", "b.jpg"]
        assert win.quality["битый.jpg"]["verdict"] == "unknown" and "error" in win.quality["битый.jpg"]
        assert win.quality["a.jpg"]["verdict"] == "ok"
        # результаты прежней папки с тем же именем кадра не попадают в новую
        win.load_folder(d2)
        win.on_thumb((d / "a.jpg", win.thumbs.get("a.jpg", __import__("numpy").zeros((10, 10, 3), "uint8"))))
        assert "a.jpg" not in win.thumbs or Path(win.folder) == d2
        wait(app, lambda: len(win.thumbs) == 1)
        win.close()
    print("битый файл OK")
