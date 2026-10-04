"""Авто-кадрирование по выбранному правилу: выбрал сетку (мышью, O, стрелками) — кадр сразу перекадрирован, без
подтверждений; листание кадров остаётся в режиме; объекта нет — просьба указать точку, кадр строится вокруг неё.
Без сегментации и лиц: главное находит детектор яркого пятна. Запуск: python tests\\test_ui_autorule.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from test_ui_compose import frame  # noqa: E402

import mini_lightroom.ui as ui  # noqa: E402
from mini_lightroom import composition as C  # noqa: E402
from mini_lightroom import engine as E  # noqa: E402

SUN = (1100 / 1500, 420 / 1000)


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise TimeoutError("не дождались")


def rel(pt, rect):
    return (pt[0] - rect[0]) / (rect[2] - rect[0]), (pt[1] - rect[1]) / (rect[3] - rect[1])


def near_anchor(rule, pt, rect, asp, flip=None):
    rx, ry = rel(pt, rect)
    return min(np.hypot(rx - ax, ry - ay) for ax, ay, _ in C._anchors(rule, False, asp, flip)) < 0.08


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui.apply_dark_theme(app)
    ui.G.available = lambda: False
    ui.FC.available = lambda: False
    tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        E.save_jpeg(d / "a_луна.jpg", frame(), 95)
        flat = np.broadcast_to(np.linspace(0.03, 0.12, 1000, dtype=np.float32)[:, None, None], (1000, 1500, 3)).copy()
        E.save_jpeg(d / "b_пусто.jpg", flat, 95)
        win = ui.MainWindow()
        win.resize(1500, 900)
        win.show()
        win.load_folder(d)
        wait(app, lambda: win.base is not None and not win.rendering)
        win.crop_cam.setChecked(False)
        assert win.auto_rule.isChecked() and win.cropper.overlay == "thirds"

        # вход в режим обрезки: кадр сразу по выбранному правилу (третей), без единого щелчка
        win.a_crop.trigger()
        wait(app, lambda: win.cropper.rect != [0.0, 0.0, 1.0, 1.0])
        r = list(win.cropper.rect)
        assert r[0] <= SUN[0] <= r[2] and r[1] <= SUN[1] <= r[3] and near_anchor("thirds", SUN, r, 1.5), r
        assert win.params["crop"] is not None and all(abs(a - b) < 1e-4 for a, b in zip(win.params["crop"], r, strict=True))
        assert not win.pick_mode, "объект найден — точку не просим"
        assert abs((r[2] - r[0]) * 1500 / ((r[3] - r[1]) * 1000) - 1.5) < 5e-3, "формат кадра не меняется"

        # стрелка вправо — следующее правило: кадр сам перекадрирован под него
        seen = {tuple(round(v, 3) for v in r)}
        for rule in ("phi", "spiral", "triangle", "diagonal", "center", "grid"):
            before = list(win.cropper.rect)
            win.a_tool_next.trigger()
            assert win.cropper.overlay == rule, (win.cropper.overlay, rule)
            wait(app, lambda: not win.analysis_busy)
            r = list(win.cropper.rect)
            assert E.crop_valid(1500, 1000, r, 0) and r[0] <= SUN[0] <= r[2] and r[1] <= SUN[1] <= r[3], (rule, r)
            fl = win.cropper.flip if rule in ("spiral", "triangle") else None
            assert near_anchor(rule, SUN, r, 1.5, fl), (rule, rel(SUN, r))
            seen.add(tuple(round(v, 3) for v in r))
            assert win.params["crop"] is not None and abs(win.params["crop"][0] - r[0]) < 1e-4, "кадр применён сразу"
        assert len(seen) >= 4, "разные правила — разные кадрирования"
        win.a_tool_next.trigger()  # по кругу: после «Сетки» — снова «Третей»
        assert win.cropper.overlay == "thirds"
        win.a_tool_prev.trigger()
        assert win.cropper.overlay == "grid"
        win.cropper.set_overlay("phi")

        # пропорции из списка: кадр заново под правило в этом формате
        win.aspect_combo.setCurrentIndex(3)  # 1:1
        win.on_aspect(3)
        r = list(win.cropper.rect)
        assert abs((r[2] - r[0]) * 1500 / ((r[3] - r[1]) * 1000) - 1.0) < 5e-3 and near_anchor("phi", SUN, r, 1.0), r
        win.aspect_combo.setCurrentIndex(0)
        win.on_aspect(0)
        win.cropper.aspect = None

        # флажок выключен — правило не перекадрирует
        win.auto_rule.setChecked(False)
        before = list(win.cropper.rect)
        win.a_tool_next.trigger()
        app.processEvents()
        assert win.cropper.rect == before, "авто-режим выключен"
        win.auto_rule.setChecked(True)
        win.cropper.set_overlay("thirds")
        app.processEvents()

        # листаем кадры, оставаясь в режиме: у кадра без объекта — просьба указать точку, а не тишина
        win.strip.setCurrentRow(1)
        wait(app, lambda: win.current and win.current.name == "b_пусто.jpg" and win.base is not None)
        wait(app, lambda: win.crop_mode and win.pick_mode, 30)
        assert win.view.hint and win.cropper.pick_cb is not None
        # точка на кадре: рамка строится вокруг неё по выбранному правилу
        pt = (0.3, 0.6)
        win.on_subject_pick(win.view.to_widget_pt(*pt))
        wait(app, lambda: not win.pick_mode and win.manual_subject.get("b_пусто.jpg") is not None)
        wait(app, lambda: win.cropper.rect != [0.0, 0.0, 1.0, 1.0])
        r = list(win.cropper.rect)
        assert abs(win.manual_subject["b_пусто.jpg"][0] - pt[0]) < 0.01
        assert r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3] and near_anchor("thirds", pt, r, 1.5), r
        assert not win.view.hint
        # правило сменили — кадр вокруг той же точки; сброс точки — снова просьба; Esc отменяет только выбор
        win.a_tool_next.trigger()
        r2 = list(win.cropper.rect)
        assert near_anchor("phi", pt, r2, 1.5) and r2 != r
        win.reset_subject()
        wait(app, lambda: win.pick_mode)
        win.cancel_crop()
        assert not win.pick_mode and win.crop_mode, "Esc при выборе точки не выходит из обрезки"
        # вернулись на первый кадр: он уже кадрирован, режим сохранён, правка применена к кадру
        win.strip.setCurrentRow(0)
        wait(app, lambda: win.current and win.current.name == "a_луна.jpg" and win.base is not None)
        wait(app, lambda: win.crop_mode)
        win.a_crop.trigger()  # Enter
        assert not win.crop_mode and win.params["crop"] is not None
        win.close()
    print("авто-кадр по правилу OK")
