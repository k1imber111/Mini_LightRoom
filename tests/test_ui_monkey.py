"""Обезьяний тест: сотни случайных действий в окне; любое исключение в слоте или сообщение об ошибке — провал.
Запуск: python tests\\test_ui_monkey.py [число_действий] [зерно]"""
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from test_ui_cull import motion, save, scene  # noqa: E402

import mini_lightroom.ui as ui  # noqa: E402
from mini_lightroom import engine as E  # noqa: E402

ERRORS: list[str] = []
BAD_TOAST = ("Ошибка", "Не получилось", "не получилось", "Не удалось", "Traceback")


def pump(app, sec=0.0):
    end = time.time() + sec
    app.processEvents()
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def main(n_actions: int, seed: int) -> None:
    rng = random.Random(seed)
    app = QApplication.instance() or QApplication(sys.argv)
    ui.apply_dark_theme(app)
    real = os.environ.get("MONKEY_FOLDER")  # папка с настоящими кадрами: копируется во временную, ИИ включён
    if not real:
        ui.G.available = lambda: False
        ui.FC.available = lambda: False
    tmp =tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    ui.QFile.moveToTrash = staticmethod(lambda f: (Path(f).unlink(), True)[1])
    sys.excepthook = lambda t, v, tb: ERRORS.append("".join(traceback.format_exception(t, v, tb)))
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        if real:
            for f in sorted(Path(real).glob("*.ARW"))[:4]:
                shutil.copy(f, d / f.name)
        else:
            for i in range(4):
                save(d / f"кадр_{i}.jpg", scene(i))
            save(d / "кадр_4.jpg", cv2.GaussianBlur(scene(4), (0, 0), 2.5))
            save(d / "кадр_5.jpg", motion(scene(5), 41))
        win = ui.MainWindow()
        win.resize(1500, 900)
        win.show()
        real_toast = win.toast
        win.toast = lambda text, ms=4000: (ERRORS.append(f"сообщение: {text}") if any(b in text for b in BAD_TOAST)
                                           else None, real_toast(text, ms))[1]
        win.load_folder(d)
        end = time.time() + 120
        while time.time() < end and (win.base is None or win.rendering):
            pump(app, 0.02)
        v = win.view

        def slider():
            row = rng.choice(list(win.rows.values()))
            lo, hi = row.slider.minimum(), row.slider.maximum()
            row.slider.setValue(rng.randint(lo, hi))

        def mouse(kind=None):
            p0 = QPoint(rng.randint(20, max(21, v.width() - 20)), rng.randint(20, max(21, v.height() - 20)))
            p1 = QPoint(rng.randint(20, max(21, v.width() - 20)), rng.randint(20, max(21, v.height() - 20)))
            btn = rng.choice([Qt.LeftButton, Qt.LeftButton, Qt.RightButton])
            QTest.mousePress(v, btn, Qt.NoModifier, p0)
            for k in range(1, 4):
                QTest.mouseMove(v, p0 + (p1 - p0) * k / 3)
            QTest.mouseRelease(v, btn, Qt.NoModifier, p1)

        def crop_toggle():
            win.a_crop.trigger()

        def select_frame():
            win.strip.setCurrentRow(rng.randrange(win.strip.count())) if win.strip.count() else None

        actions = [
            (slider, 30), (mouse, 14), (crop_toggle, 5), (select_frame, 6),
            (lambda: win.undo(), 4), (lambda: win.redo(), 3),
            (lambda: win.a_compare.trigger(), 2), (lambda: win.a_reset.trigger(), 1), (lambda: win.a_auto.trigger(), 2),
            (lambda: v.set_zoom(rng.choice([None, 0.5, 1.0, 2.0, 4.0])), 4), (lambda: v.step_zoom(rng.choice([-1, 1])), 3),
            (lambda: win.aspect_combo.setCurrentIndex(rng.randrange(win.aspect_combo.count())) or
             win.on_aspect(win.aspect_combo.currentIndex()), 3),
            (lambda: win.on_overlay_pick(rng.randrange(len(ui.OVERLAYS))), 2),
            (lambda: win.cropper.rotate_overlay(), 1), (lambda: win.flip_aspect(), 1), (lambda: win.reset_crop(), 1),
            (lambda: win.angle_row.slider.setValue(rng.randint(-100, 100)), 3),
            (lambda: win.start_compose(), 2),
            (lambda: win.crop_carousel.setCurrentRow(rng.randrange(max(1, win.crop_carousel.count()))), 2),
            (lambda: win.add_mask(rng.choice(["linear", "radial", "brush"])), 3),
            (lambda: win.delete_mask() if win.masks() else None, 1),
            (lambda: win.on_curve("rgb", [[0, 0], [rng.randint(30, 220), rng.randint(30, 220)], [255, 255]]), 2),
            (lambda: win.on_hsl(rng.choice(["red", "green", "blue"]), rng.randrange(3), rng.randint(-60, 60)), 2),
            (lambda: win.toggle_flag(rng.choice(["reject", "pick"])), 3),
            (lambda: win.filter_bar.btns[rng.choice(["all", "bad", "doubt", "ok"])].click(), 3),
            (lambda: win.start_quality(force=rng.random() < 0.3), 2),
            (lambda: win.copy_settings(), 1), (lambda: win.paste_settings(), 1),
            (lambda: win.look_combo.setCurrentIndex(rng.randrange(win.look_combo.count())), 2),
            (lambda: win.rows["look_strength"].slider.setValue(rng.randint(0, 100)), 1),
            (lambda: win.grab(), 2),
            (lambda: win.a_tool_next.trigger(), 4), (lambda: win.a_tool_prev.trigger(), 3),
            (lambda: win.begin_pick(), 2), (lambda: win.reset_subject(), 1), (lambda: win.auto_rule.toggle(), 1),
            (lambda: win.on_subject_pick(QPointF(rng.randint(0, v.width()), rng.randint(0, v.height()))), 3),
            (lambda: win.cancel_crop(), 1), (lambda: win.a_overlay_rot.trigger(), 2),
        ]
        if real:
            actions += [
                (lambda: win.start_grade(), 2), (lambda: win.ai_enhance(), 1), (lambda: win.ai_horizon(), 1),
                (lambda: win.ai_thirds(), 1), (lambda: win.detect_scenes(), 1),
                (lambda: win.add_ai_mask(rng.choice(["sky", "people", "greenery", "water", "buildings"])), 2),
                (lambda: win.rows["denoise"].slider.setValue(rng.randint(0, 100)), 1),
                (lambda: win.rows["retouch"].slider.setValue(rng.randint(0, 100)), 1),
                (lambda: win.rows["bokeh"].slider.setValue(rng.randint(0, 100)), 1),
                (lambda: win.carousel.setCurrentRow(rng.randrange(max(1, win.carousel.count()))), 2),
            ]
        fns, weights = zip(*actions, strict=True)
        for i in range(n_actions):
            fn = rng.choices(fns, weights)[0]
            try:
                fn()
            except Exception:  # noqa: BLE001 — падение действия — тоже находка
                ERRORS.append(f"действие {i}: " + traceback.format_exc())
            pump(app, rng.choice([0.0, 0.0, 0.01, 0.05, 0.12]))
        # дать всему устояться и проверить итоговое состояние
        end = time.time() + 60
        while time.time() < end and (win.rendering or win.dirty or win.q_left or win.compose_busy or win.grade_busy
                                     or win.enhance_busy or win.dn_busy):
            pump(app, 0.05)
        pump(app, 0.5)
        if win.crop_mode:
            win.cancel_crop()
        win.grab()
        assert win.strip.count() == len(win.files) == len(win.items)
        win.close()
    if ERRORS:
        print(f"СИД {seed}: найдено проблем: {len(ERRORS)}")
        for e in dict.fromkeys(ERRORS):
            print(e[-1200:])
        raise SystemExit(1)
    print(f"сид {seed}: {n_actions} действий без ошибок")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print("обезьяна OK")
