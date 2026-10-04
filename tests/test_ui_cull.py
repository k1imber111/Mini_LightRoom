"""Окно: проверка брака → фильтры, значки, полоса под фото, флаги X/U, корзина, кеш в sidecar.
Удаление идёт ТОЛЬКО по временным копиям и через подменённый moveToTrash (настоящая корзина не трогается).
Запуск: python tests\\test_ui_cull.py"""
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
from mini_lightroom import quality as Q  # noqa: E402
from mini_lightroom.ui import MainWindow, apply_dark_theme  # noqa: E402


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise TimeoutError("не дождались")


def scene(seed: int) -> np.ndarray:
    """Структурный кадр 1500×1000: контрастные фигуры с антиалиасингом (как натура, но маленький)."""
    rng = np.random.default_rng(seed)
    g = np.full((1000, 1500), 110, np.uint8)
    for _ in range(260):
        c = int(rng.integers(0, 255))
        x, y = int(rng.integers(0, 1500)), int(rng.integers(0, 1000))
        if rng.integers(0, 2):
            cv2.circle(g, (x, y), int(rng.integers(8, 60)), c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(g, (x, y), (x + int(rng.integers(15, 90)), y + int(rng.integers(15, 90))), c, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(g, (0, 0), 0.5)


def save(path: Path, g: np.ndarray) -> None:
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255
    E.save_jpeg(path, rgb, 95)


def motion(g: np.ndarray, length: int) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    return cv2.filter2D(g, -1, k / k.sum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    import mini_lightroom.ui as ui
    ui.FC.available = lambda: False        # без лиц: не зависим от mediapipe и сети
    Q.CFG["norm_long_side"] = 1500         # кадры тестовые, не 6000 px: σ считаем как есть
    tmp = tempfile.TemporaryDirectory(prefix="стили_")
    ui.STYLES_DIR = Path(tmp.name)
    ui.SETTINGS_FILE = Path(tmp.name) / "settings.json"
    trashed: list[str] = []

    def fake_trash(path):  # «корзина» на временных копиях: файл просто пропадает
        trashed.append(Path(path).name)
        Path(path).unlink()
        return True

    ui.QFile.moveToTrash = staticmethod(fake_trash)
    with tempfile.TemporaryDirectory(prefix="съёмка_") as d:
        d = Path(d)
        for i in range(3):
            save(d / f"кадр_{i}.jpg", scene(i))
        save(d / "кадр_3.jpg", cv2.GaussianBlur(scene(3), (0, 0), 2.5))   # расфокус
        save(d / "кадр_4.jpg", motion(scene(4), 41))                       # смаз

        win = MainWindow()
        win.resize(1500, 900)
        win.show()
        win.load_folder(d)
        wait(app, lambda: len(win.quality) == 5 and not win.q_left, 180)
        v = {n: r["verdict"] for n, r in win.quality.items()}
        print("вердикты:", v)
        assert [v[f"кадр_{i}.jpg"] for i in range(3)] == ["ok"] * 3, v
        assert v["кадр_3.jpg"] == "bad" and win.quality["кадр_3.jpg"]["defects"][0]["type"] == "soft", win.quality["кадр_3.jpg"]
        assert v["кадр_4.jpg"] == "bad" and win.quality["кадр_4.jpg"]["defects"][0]["type"] == "motion"
        assert "брак 2" in win.filter_bar.btns["bad"].text().lower() or win.filter_bar.btns["bad"].text() == "Брак 2"
        assert win.trash_btn.isVisible() and "(2)" in win.trash_btn.text()
        assert not win.items["кадр_3.jpg"].icon().isNull() and "Брак" in win.items["кадр_3.jpg"].text()

        # фильтр «Брак»: видны только два кадра
        win.filter_bar.btns["bad"].click()
        assert [n for n, it in win.items.items() if not it.isHidden()] == ["кадр_3.jpg", "кадр_4.jpg"]
        if len(sys.argv) > 1:
            app.processEvents()
            win.grab().save(str(Path(sys.argv[1]) / "cull_0_filter.png"))
        win.filter_bar.btns["all"].click()
        assert not any(it.isHidden() for it in win.items.values())

        # выбор бракованного кадра: полоса «Контроль качества», рамки на кадре, плитка с крупным местом брака
        win.strip.setCurrentItem(win.items["кадр_3.jpg"])
        wait(app, lambda: win.current and win.current.name == "кадр_3.jpg" and win.base is not None)
        wait(app, lambda: win.qbar.tiles.count() >= 2)  # плитка (место брака = лучшее место кадра) + растяжка
        assert win.qbar.isVisible() and win.qbar.pill.text() == "Брак", win.qbar.pill.text()
        assert win.view.marks and win.view.marks[0][1] == "bad"
        tile = win.qbar.tiles.itemAt(0).widget()
        assert tile.roi and tile.title.startswith("Расфокус")
        win.qbar.zoom_to.emit(tile.roi)  # щелчок по плитке → кадр в 100% на этом месте
        assert win.view.zoom == 1.0
        win.view.set_zoom(None)
        if len(sys.argv) > 1:  # снимки окна для глаз: python tests	est_ui_cull.py папка
            out = Path(sys.argv[1])
            out.mkdir(parents=True, exist_ok=True)
            app.processEvents()
            win.grab().save(str(out / "cull_1_bad.png"))
        win.grab()

        # решение пользователя важнее авто-вердикта: «оставить» снимает кадр из кандидатов, X помечает хороший
        win.toggle_flag("pick")
        assert win.effective("кадр_3.jpg") == "ok" and "кадр_3.jpg" not in win.cull_candidates()
        win.strip.setCurrentItem(win.items["кадр_0.jpg"])
        wait(app, lambda: win.current and win.current.name == "кадр_0.jpg" and win.base is not None)
        win.toggle_flag("reject")
        assert sorted(win.cull_candidates()) == ["кадр_0.jpg", "кадр_4.jpg"] and win.cull_rejects() == set(win.cull_candidates())
        win.toggle_flag("reject")  # повторный X снимает пометку
        assert win.flags.get("кадр_0.jpg") is None and win.cull_candidates() == ["кадр_4.jpg"]
        win.toggle_flag("reject")
        win.save_sidecar()
        side = E.read_json(d / ".mini_lightroom.json", {})
        assert side["__flags__"] == {"кадр_3.jpg": "pick", "кадр_0.jpg": "reject"} and "кадр_4.jpg" in side["__quality__"]
        assert all(k.startswith("__") or isinstance(v, dict) and "exposure" in v for k, v in side.items() if k != "__scenes__"
                   and not k.startswith("__")), "правки кадров не должны мешаться со служебными ключами"

        # корзина: отмена диалога ничего не удаляет
        ui.TrashDialog.exec = lambda self: 0
        win.trash_rejected()
        assert not trashed and len(win.files) == 5 and (d / "кадр_4.jpg").exists()
        # подтверждение: файлы уходят в корзину, из ленты, кешей и sidecar исчезают, остальные целы
        ui.TrashDialog.exec = lambda self: 1
        win.trash_rejected()
        assert sorted(trashed) == ["кадр_0.jpg", "кадр_4.jpg"], trashed
        assert sorted(p.name for p in win.files) == ["кадр_1.jpg", "кадр_2.jpg", "кадр_3.jpg"]
        assert sorted(win.items) == ["кадр_1.jpg", "кадр_2.jpg", "кадр_3.jpg"] and win.strip.count() == 3
        side = E.read_json(d / ".mini_lightroom.json", {})
        assert "кадр_4.jpg" not in side["__quality__"] and "кадр_0.jpg" not in side["__flags__"] and "кадр_0.jpg" not in side
        assert not win.trash_btn.isVisible() and (d / "кадр_3.jpg").exists() and not (d / "кадр_0.jpg").exists()
        wait(app, lambda: win.current is not None and win.base is not None)  # на кадр рядом перешли сами

        # перезапуск: результаты берутся из sidecar, повторно не считаются
        win.close()
        win2 = MainWindow()
        win2.show()
        win2.load_folder(d)
        wait(app, lambda: len(win2.thumbs) == 3)
        app.processEvents()
        assert win2.q_total == 0 and len(win2.quality) == 3 and win2.effective("кадр_3.jpg") == "ok", \
            (win2.q_total, win2.quality.keys())
        win2.close()
    print("отбор брака OK")
