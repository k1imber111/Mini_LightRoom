"""Тема: контраст токенов (WCAG), QSS без неподставленных токенов, все иконки рисуются.
Запуск: python tests\\test_theme.py"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from mini_lightroom import theme as TH  # noqa: E402

if __name__ == "__main__":
    app = QApplication(sys.argv)
    T = TH.T
    # основной текст ≥ 7:1 на всех поверхностях, вторичный ≥ 4.5:1
    for bg in ("canvas", "window", "panel", "control"):
        assert TH.contrast(T["text"], T[bg]) >= 7, (bg, TH.contrast(T["text"], T[bg]))
        assert TH.contrast(T["text2"], T[bg]) >= 4.5, (bg, TH.contrast(T["text2"], T[bg]))
    # акцент и статусы читаются на панели; текст на акцентной заливке
    for key in ("accent", "ok", "warn", "bad"):
        assert TH.contrast(T[key], T["panel"]) >= 4.5, (key, TH.contrast(T[key], T["panel"]))
    assert TH.contrast(T["on_accent"], T["accent"]) >= 6, TH.contrast(T["on_accent"], T["accent"])  # AA для текста — 4.5
    assert TH.contrast(T["on_accent"], T["accent_press"]) >= 4.5
    # у нейтральных серых нет оттенка: r == g == b
    for key in ("canvas", "window", "panel", "control", "hover", "press", "text", "text2", "dim"):
        v = T[key].lstrip("#")
        assert v[0:2] == v[2:4] == v[4:6], key

    qss = TH.build_qss()
    assert "@" not in re.sub(r"/\*.*?\*/", "", qss), "в QSS остались неподставленные токены"
    assert "chevron-down" in qss and "check" in qss, "картинки стрелки и галочки не подставились"

    names = sorted(p.stem for p in TH.ICONS_DIR.glob("*.svg"))
    assert len(names) >= 20, names
    for name in names:
        pm = TH.icon_pixmap(name, T["text"])
        assert not pm.isNull() and pm.width() == 36, name
        img = pm.toImage()
        assert any(img.pixelColor(x, y).alpha() > 0 for x in range(0, 36, 3) for y in range(0, 36, 3)), \
            f"иконка {name} пустая"
    assert not TH.tool_icon("crop").isNull()

    TH.apply_theme(app)
    print("тема OK:", len(names), "иконок, контраст текста на панели",
          round(TH.contrast(T["text"], T["panel"]), 1), "/", round(TH.contrast(T["text2"], T["panel"]), 1))
