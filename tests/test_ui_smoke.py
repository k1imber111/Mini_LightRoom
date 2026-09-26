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
    styles_tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(styles_tmp.name)  # не трогаем настоящую библиотеку стилей
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

        # библиотека стилей (этап 2): миниатюра, режим, смесь, переименование, удаление
        for i, nm in ((1, "Тёплый"), (2, "Холодный")):
            st, th = ui.style_job(d / f"кадр_{i}.jpg", nm)
            E.write_json(ui.STYLES_DIR / f"{nm}.json", st)
            E.save_jpeg(ui.STYLES_DIR / f"{nm}.jpg", th / 255.0)
        win.reload_styles()
        assert not win.style_combo.itemIcon(win.style_combo.findData("Тёплый")).isNull(), "нет миниатюры стиля"
        win.style_combo.setCurrentIndex(win.style_combo.findData("Тёплый"))
        win.style2_combo.setCurrentIndex(win.style2_combo.findData("Холодный"))
        win.style_mode.setCurrentIndex(1)
        wait(app, lambda: not win.rendering and not win.dirty)
        assert win.params["style_mode"] == 1 and "+" in win.current_style()["name"], "смесь стилей не собралась"
        assert win._rename_style("Тёплый", "Персик")
        assert win.params["style"] == "Персик" and (ui.STYLES_DIR / "Персик.jpg").exists()
        assert not (ui.STYLES_DIR / "Тёплый.json").exists()
        assert not win._rename_style("Персик", "Холодный"), "переименование затёрло чужой стиль"
        win._delete_style("Холодный")
        assert win.params["style2"] == "" and not (ui.STYLES_DIR / "Холодный.json").exists()
        win.sync_controls()
        assert win.style_combo.currentText() == "Персик"
        wait(app, lambda: not win.rendering and not win.dirty)

        # маски (этап 3): жесты мышью на кадре, ползунки маски, ИИ-маска из готового PNG
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        v = win.view
        win.view.set_zoom(None)
        app.processEvents()
        at = lambda xn, yn: v.to_widget_pt(xn, yn).toPoint()

        def drag(p0, p1, mod=Qt.NoModifier):
            QTest.mousePress(v, Qt.LeftButton, mod, p0)
            for k in range(1, 6):
                QTest.mouseMove(v, p0 + (p1 - p0) * k / 5)
            QTest.mouseRelease(v, Qt.LeftButton, mod, p1)

        win.add_mask("linear")
        drag(at(0.5, 0.05), at(0.5, 0.5))
        lin = win.masks()[-1]
        assert lin["type"] == "linear" and lin["b"][1] > 0.4, lin
        win.mask_rows["exposure"].slider.setValue(-150)
        assert lin["adj"]["exposure"] == -150
        win.add_mask("radial")
        drag(at(0.5, 0.6), at(0.7, 0.8))
        rad = win.masks()[-1]
        assert rad["type"] == "radial" and rad["r"][0] > 0.1, rad
        win.mask_rows["temperature"].slider.setValue(50)
        win.add_mask("brush")
        drag(at(0.1, 0.9), at(0.4, 0.9))
        drag(at(0.2, 0.9), at(0.3, 0.9), Qt.AltModifier)
        br = win.masks()[-1]
        assert len(br["strokes"]) == 2 and br["strokes"][1]["erase"], "кисть/ластик не записались"
        win.mask_rows["saturation"].slider.setValue(-80)
        # ИИ-маска: кладём готовый PNG, как будто модель уже посчитала небо
        sky = np.zeros((win.base.shape[0], win.base.shape[1]), np.uint8)
        sky[: sky.shape[0] // 3] = 255
        ui._write_png(d / ui.MASKS_DIR / f"{win.current.name}.sky.png", sky)
        win.add_ai_mask("sky")
        assert win.masks()[-1]["type"] == "ai" and win.render_params()["masks"][-1]["arr"] is not None
        win.mask_rows["exposure"].slider.setValue(80)
        win.mask_show.setChecked(True)
        v.repaint()
        assert win.editor._overlay is not None, "подсветка маски не построилась"
        win.mask_show.setChecked(False)
        wait(app, lambda: not win.rendering and not win.dirty)
        assert len(win.masks()) == 4 and "arr" not in win.masks()[-1], "массив маски попал в правки"
        win.copy_settings()
        win.clipboard["masks"][0]["adj"]["exposure"] = 0
        assert lin["adj"]["exposure"] == -150, "копия правок делит маски с оригиналом"
        win.save_sidecar()
        saved = E.read_json(d / ".mini_lightroom.json", {})[win.current.name]["masks"]
        assert [m["type"] for m in saved] == ["linear", "radial", "brush", "ai"]
        win.mask_list.setCurrentRow(-1)

        # шумодав (этап 4): ползунок → превью без шума считается в фоне и смешивается по силе
        from mini_lightroom import enhance as N
        if N.available() and (N.MODELS_DIR / N.MODELS["denoise"][0]).exists():
            win.rows["denoise"].slider.setValue(60)
            wait(app, lambda: win.base_dn is not None, 120)
            wait(app, lambda: not win.rendering and not win.dirty)
            assert win.params["denoise"] == 60 and win.base_dn.shape == win.base.shape
            win.rows["denoise"].slider.setValue(0)
            wait(app, lambda: not win.rendering and not win.dirty)
        else:
            assert not win.rows["denoise"].isEnabled() or N.available()

        # отмена/повтор: протяжка ползунка — один шаг
        win.commit_history()
        steps = len(win._hist()["undo"])
        for val in range(1, 30):
            win.rows["clarity"].slider.setValue(val)
        wait(app, lambda: not win.rendering and not win.dirty)
        win.commit_history()
        assert len(win._hist()["undo"]) == steps + 1, "протяжка дала больше одного шага истории"
        win.undo()
        assert win.params["clarity"] == 0 and win.rows["clarity"].slider.value() == 0
        win.redo()
        assert win.params["clarity"] == 29
        n_masks = len(win.masks())
        win.masks().pop()
        win.request_render()
        win.commit_history()
        win.undo()
        assert len(win.masks()) == n_masks, "удаление маски не отменилось"
        wait(app, lambda: not win.rendering and not win.dirty)

        # обрезка: пропорции, рамка мышью, горизонт, маски поверх обрезки, масштаб
        win.a_crop.trigger()
        assert win.crop_mode and win.view.editor is win.cropper and (win.view.src_w, win.view.src_h) == (900, 600)
        win.aspect_combo.setCurrentIndex(7)
        win.on_aspect(7)  # 16:9
        x0, y0, x1, y1 = win.cropper.rect
        assert abs((x1 - x0) * 900 / ((y1 - y0) * 600) - 16 / 9) < 0.01
        win.aspect_combo.setCurrentIndex(0)
        win.on_aspect(0)  # свободно: тянем правый нижний угол внутрь
        pc = win.cropper._px(v, x1, y1).toPoint()
        drag(pc, pc - QPoint(40, 30))
        assert win.cropper.rect[2] < x1 - 0.02, "угол рамки не сдвинулся"
        win.angle_row.slider.setValue(35)  # 3.5°
        assert abs(win.params["angle"] - 3.5) < 1e-6
        assert E.crop_valid(900, 600, win.cropper.rect, 3.5), "после поворота рамка вышла за снимок"
        win.a_crop.trigger()  # Enter / повторное нажатие — применить
        assert not win.crop_mode and win.params["crop"] is not None
        cw, ch = win.view.src_w, win.view.src_h
        assert cw < 900 and ch < 600, (cw, ch)
        wait(app, lambda: not win.rendering and not win.dirty)
        assert abs(win.after.width() / win.after.height() - cw / ch) < 0.02, "превью не обрезано"
        pt = v.to_widget_pt(0.37, 0.61)  # маски поверх обрезки: доли полного кадра ↔ экран
        back = v.to_norm(pt)
        assert abs(back[0] - 0.37) < 1e-6 and abs(back[1] - 0.61) < 1e-6
        v.set_zoom(1.0)
        wait(app, lambda: win.view.detail is not None)
        dpix, rect = v.detail
        assert rect.right() <= cw + 1 and rect.bottom() <= ch + 1, "деталь вне обрезанного кадра"
        v.set_zoom(None)
        E.write_json(ui.PRESETS_DIR / "__тест_обрезка.json", {"contrast": 10})
        try:
            win.reload_presets()
            i = win.preset_combo.findText("__тест_обрезка")
            win.preset_combo.setCurrentIndex(i)
            win.apply_preset(i)
            assert win.params["crop"] is not None and abs(win.params["angle"] - 3.5) < 1e-6, "пресет сбросил обрезку"
        finally:
            (ui.PRESETS_DIR / "__тест_обрезка.json").unlink()
            win.reload_presets()
        win.undo()
        win.reset_crop()
        assert win.params["crop"] is None and (win.view.src_w, win.view.src_h) == (900, 600)
        wait(app, lambda: not win.rendering and not win.dirty)

        # тональная кривая: точка мышью, перетаскивание, удаление, каналы, готовые формы; HSL; отмена
        cv = win.curve
        cv.resize(260, 220)
        p_mid = cv._to_w(128, 128).toPoint()
        QTest.mousePress(cv, Qt.LeftButton, Qt.NoModifier, p_mid)
        QTest.mouseMove(cv, cv._to_w(128, 175).toPoint())
        QTest.mouseRelease(cv, Qt.LeftButton, Qt.NoModifier, cv._to_w(128, 175).toPoint())
        pts = win.params["curve"]["rgb"]
        assert len(pts) == 3 and pts[1][1] > 160, pts
        QTest.mouseDClick(cv, Qt.LeftButton, Qt.NoModifier, cv._to_w(*pts[1]).toPoint())
        assert "rgb" not in win.params["curve"], "двойной щелчок не удалил точку"
        win.curve_btns["b"].click()
        win.curve_shape.setCurrentIndex(list(ui.CURVE_SHAPES).index("Матовая (плёнка)") + 1)
        win.on_curve_shape(win.curve_shape.currentIndex())
        assert win.params["curve"]["b"][0][1] == 28 and "rgb" not in win.params["curve"]
        win.curve_btns["rgb"].click()
        win.hsl_rows[("green", 1)].slider.setValue(-60)
        win.hsl_rows[("orange", 2)].slider.setValue(25)
        assert win.params["hsl"] == {"green": [0, -60, 0], "orange": [0, 0, 25]}
        wait(app, lambda: not win.rendering and not win.dirty)
        win.commit_history()
        win.reset_hsl()
        win.commit_history()
        win.undo()
        assert win.params["hsl"]["green"][1] == -60 and win.hsl_rows[("green", 1)].slider.value() == -60
        wait(app, lambda: not win.rendering and not win.dirty)

        # целевая правка: тянуть по цвету/тону прямо на кадре
        saved_base = win.base
        colored = np.zeros_like(win.base)
        half = colored.shape[1] // 2
        colored[:, :half] = [0.2, 0.6, 0.2]     # зелень слева
        colored[:, half:] = [0.8, 0.5, 0.25]    # оранжевый справа
        win.base = colored
        win.hsl_tabs.setCurrentIndex(1)         # вкладка «Насыщенность»
        win.tat_hsl.setChecked(True)
        assert win.view.editor is win.targeter
        before_g = (win.params.get("hsl") or {}).get("green", [0, 0, 0])[1]
        drag(at(0.25, 0.5), at(0.25, 0.5) - QPoint(0, 60))
        assert win.params["hsl"]["green"][1] > before_g + 15, win.params["hsl"]
        assert win.hsl_rows[("green", 1)].slider.value() == win.params["hsl"]["green"][1]
        win.tat_curve.setChecked(True)
        assert not win.tat_hsl.isChecked() and win.target_mode == "curve"
        drag(at(0.75, 0.5), at(0.75, 0.5) - QPoint(0, 40))
        pts = win.params["curve"]["rgb"]
        mid_pt = [q for q in pts if 0 < q[0] < 255]
        assert mid_pt and mid_pt[0][1] > mid_pt[0][0] + 10, pts
        win.tat_curve.setChecked(False)
        assert win.view.editor is win.editor
        win.base = saved_base
        win.reset_curve()
        wait(app, lambda: not win.rendering and not win.dirty)

        # ИИ-цветокоррекция: карусель на готовых масках (без скачивания моделей)
        from mini_lightroom import grading as GR
        hh, ww = win.base.shape[:2]
        fake = {c: np.zeros((hh, ww), np.uint8) for c in GR.MASK_CATS}
        fake["sky"][: hh // 3] = 255
        fake["people"][hh // 3: hh * 2 // 3, ww // 3: ww * 2 // 3] = 255
        for c, m in fake.items():
            ui._write_png(d / ui.MASKS_DIR / f"{win.current.name}.{c}.png", m)
            win.ai_cache.pop((win.current.name, c), None)
        win.scene_of[win.current.name] = ["portrait", 0.9]
        win.start_grade()
        wait(app, lambda: not win.grade_busy and win.carousel.count() > 1, 120)
        assert win.carousel.isVisible() and win.carousel.count() == len(win.grade_variants) + 1
        win.carousel.setCurrentRow(1)
        assert win.params["ai_grade"]["id"] == win.grade_variants[0]["id"]
        win.grade_strength_row.slider.setValue(40)
        assert win.params["ai_grade"]["strength"] == 40
        subj = next(i for i, vv in enumerate(win.grade_variants) if vv["id"] == "subject")
        win.carousel.setCurrentRow(subj + 1)
        rp = win.render_params()["ai_grade"]
        assert rp["masks"] and all(m["arr"] is not None for m in rp["masks"]), "маски варианта без данных"
        assert "arr" not in win.params["ai_grade"]["masks"][0], "массив маски попал в правки"
        wait(app, lambda: not win.rendering and not win.dirty)
        win.commit_history()
        win.carousel.setCurrentRow(0)
        assert win.params["ai_grade"] is None
        win.commit_history()
        win.undo()
        assert win.params["ai_grade"]["id"] == "subject" and win.carousel.currentRow() == subj + 1
        win.carousel.setCurrentRow(0)
        win.carousel.hide()
        wait(app, lambda: not win.rendering and not win.dirty)

        # ИИ-композиция на готовых масках (люди в центре — голова встаёт на пересечение третей)
        win.ai_thirds()
        assert win.params["crop"] is not None, "кадр по третям не обрезал"
        win.ai_horizon()  # на шуме горизонта нет — поворот не меняется
        assert win.params.get("angle", 0) == 0
        win.reset_crop()
        wait(app, lambda: not win.rendering and not win.dirty)

        # пресет-образ: выбор в списке, миниатюры кадра на пунктах, сила
        assert len(win.looks) >= 20
        name = next(iter(win.looks))
        win.look_combo.setCurrentIndex(win.look_combo.findData(name))
        assert win.params["look"] == name
        wait(app, lambda: not win.look_combo.itemIcon(win.look_combo.findData(name)).isNull())
        win.rows["look_strength"].slider.setValue(60)
        wait(app, lambda: not win.rendering and not win.dirty)
        assert win.params["look_strength"] == 60

        # сцены (без модели): известная сцена → «Авто» ставит её пресет, подпись в ленте, запись в sidecar
        win.scene_of["кадр_0.jpg"] = ["sunset", 0.9]
        win.update_item("кадр_0.jpg")
        assert "Закат" in win.items["кадр_0.jpg"].text()
        win.apply_auto()
        assert win.params["look"] == win.scene_by_id["sunset"]["look"]
        win.save_sidecar()
        assert E.read_json(d / ".mini_lightroom.json", {})["__scenes__"]["кадр_0.jpg"][0] == "sunset"
        name = win.params["look"]
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

        assert win.sidecar["кадр_0.jpg"]["look"] == name, "пресет не сохранился в правках кадра"
        jobs = [{"src": str(p), "dst": str(d / "export" / f"{p.stem}.jpg"), "params": win.sidecar["кадр_0.jpg"],
                 "style": style, "look": win.looks[name], "auto": True, "long_edge": 500, "quality": 90}
                for p in win.files]
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
