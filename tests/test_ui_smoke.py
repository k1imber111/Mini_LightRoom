"""Прогон окна без экрана: папка с кириллицей → миниатюры → рендер → правки → экспорт.
Запуск: python tests\\test_ui_smoke.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mini_lightroom import engine as E  # noqa: E402
from mini_lightroom.ui import MainWindow, apply_dark_theme  # noqa: E402


def wait(app, cond, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise TimeoutError("не дождались")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    import mini_lightroom.ui as ui
    ui.PREVIEW_SIDE = 400  # превью меньше кадра, чтобы проверить просмотр в масштабе
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        rng = np.random.default_rng(1)
        for i in range(4):
            img = rng.random((600, 900, 3), dtype=np.float32) * (0.2 + 0.2 * i)
            if i == 3:
                import cv2
                img = cv2.GaussianBlur(img, (0, 0), 12)  # «размытый» кадр
            E.save_jpeg(d / f"кадр_{i}.jpg", img)

        win = MainWindow()
        win.show()
        win.load_folder(d)
        wait(app, lambda: win.after is not None)
        print("первый рендер готов, авто-экспозиция:", win.params["exposure"])
        wait(app, lambda: len(win.scores) == 4)
        print("размытые:", win.blurry)
        assert "кадр_3.jpg" in win.blurry

        win.rows["contrast"].slider.setValue(40)
        win.rows["vibrance"].slider.setValue(25)
        wait(app, lambda: not win.rendering and not win.dirty)
        assert win.params["contrast"] == 40

        # стиль из «чужого» фото
        style = E.style_from_image(d / "кадр_0.jpg", "тест")
        win.styles["тест"] = style
        win.params["style"] = "тест"
        win.request_render()
        wait(app, lambda: not win.rendering and not win.dirty)

        # масштаб: превью 400 px меньше кадра 900 px → на 100% дорисовывается полное разрешение
        assert (win.view.src_w, win.view.src_h) == (900, 600)
        win.view.set_zoom(1.0)
        wait(app, lambda: win.view.detail is not None)
        dpix, rect = win.view.detail
        assert rect.width() <= 900 and dpix.width() == round(rect.width()), (dpix.width(), rect)
        win.view.step_zoom(1)
        assert win.view.zoom == 1.5 and win.zoom_combo.currentText() == "150%"
        win.view.set_zoom(None)
        assert win.zoom_combo.currentIndex() == 0

        win.strip.setCurrentRow(1)
        wait(app, lambda: win.current and win.current.name == "кадр_1.jpg" and win.base is not None)
        assert "кадр_0.jpg" in win.sidecar, "правки первого кадра не запомнились"

        jobs = [{"src": str(p), "dst": str(d / "export" / f"{p.stem}.jpg"), "params": win.sidecar["кадр_0.jpg"],
                 "style": style, "auto": True, "long_edge": 500, "quality": 90} for p in win.files]
        from mini_lightroom.ui import ExportThread
        th = ExportThread(jobs)
        result = {}
        th.finished_all.connect(lambda ok, err: result.update(ok=ok, err=err))
        th.start()
        wait(app, lambda: "ok" in result, 120)
        th.wait()
        print("экспорт:", result)
        assert result["ok"] == 4 and not result["err"]
        assert max(E.load_image(d / "export" / "кадр_0.jpg").shape[:2]) == 500
        win.close()
    print("UI OK")
