"""Тема окна: токены цвета, палитра, QSS и линейные иконки (Tabler, MIT). Без бизнес-логики.

Серые нейтральные, без оттенка: цвет вокруг снимка влияет на восприятие цвета самого снимка.
Акцент один (`accent`) и только на активном: режим, ползунок, выделение. ok/warn/bad — только статусы.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

__all__ = ["T", "apply_theme", "build_palette", "build_qss", "contrast", "icon_pixmap", "tool_icon"]

APP_DIR = Path(__file__).resolve().parent.parent
ICONS_DIR = APP_DIR / "assets" / "icons"
CACHE_DIR = APP_DIR / ".cache" / "theme"  # цветные копии SVG для url() в QSS (в .gitignore)

T = {
    "canvas": "#161616",   # фон вокруг фото
    "window": "#1e1e1e",   # панели, тулбар
    "panel": "#262626",    # карточки, секции
    "control": "#2f2f2f",  # поля, кнопки
    "hover": "#3a3a3a",
    "press": "#444444",
    "text": "#dcdcdc",
    "text2": "#9a9a9a",
    "dim": "#666666",      # только декор и disabled
    "accent": "#6c9bf2",
    "accent_hover": "#85adf5",
    "accent_press": "#5887d8",
    "on_accent": "#101010",
    "ok": "#6dbb86",
    "warn": "#e3b04b",
    "bad": "#e2685f",
    "border": "rgba(255,255,255,18)",        # белый 7%
    "accent_soft": "rgba(108,155,242,56)",   # акцент 22% — «включено» без заливки
    "bad_soft": "rgba(226,104,95,46)",
    "warn_soft": "rgba(227,176,75,46)",
    "ok_soft": "rgba(109,187,134,46)",
}

_QSS = """
QToolTip { background: @panel@; color: @text@; border: 1px solid @border@; border-radius: 8px; padding: 6px 8px; }
QMainWindow, QDialog { background: @window@; }
QScrollArea { border: none; background: transparent; }

QToolBar { background: @window@; border: none; border-bottom: 1px solid @border@; spacing: 4px; padding: 6px 10px; }
QToolBar::separator { background: @border@; width: 1px; margin: 6px 8px; }
QToolButton { background: transparent; color: @text2@; border: none; border-radius: 15px; padding: 6px 12px; }
QToolButton:hover { background: @hover@; color: @text@; }
QToolButton:pressed { background: @press@; }
QToolButton:checked { background: @accent_soft@; color: @accent@; }
QToolButton:disabled { color: @dim@; background: transparent; }
QToolButton::menu-indicator { image: none; }
QToolButton#primary { background: @accent@; color: @on_accent@; font-weight: 600; padding: 6px 16px; }
QToolButton#primary:hover { background: @accent_hover@; }
QToolButton#primary:pressed { background: @accent_press@; }
QToolButton#primary:disabled { background: @control@; color: @dim@; }

QFrame#section { background: @panel@; border: 1px solid @border@; border-radius: 12px; }
QToolButton#sectionHead { background: transparent; color: @text@; border: none; border-radius: 12px;
    padding: 9px 12px; font-weight: 600; text-align: left; }
QToolButton#sectionHead:hover { background: @hover@; }
QToolButton#sectionHead:checked { background: transparent; color: @text@; }
QFrame#sectionBody { background: transparent; border: none; padding: 2px 12px 12px 12px; }

QPushButton { background: @control@; color: @text@; border: 1px solid @border@; border-radius: 15px;
    padding: 5px 10px; min-height: 20px; }
QPushButton:hover { background: @hover@; }
QPushButton:pressed { background: @press@; }
QPushButton:checked { background: @accent_soft@; color: @accent@; border-color: @accent@; }
QPushButton:disabled { color: @dim@; background: @window@; }
QPushButton[primary="true"] { background: @accent@; color: @on_accent@; border: none; font-weight: 600; }
QPushButton[primary="true"]:hover { background: @accent_hover@; }
QPushButton[primary="true"]:pressed { background: @accent_press@; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: @control@; color: @text@; border: 1px solid @border@;
    border-radius: 10px; padding: 4px 10px; min-height: 22px; selection-background-color: @accent@;
    selection-color: @on_accent@; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: @accent@; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled { color: @dim@; background: @window@; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow { image: url(@chev_down@); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: @panel@; color: @text@; border: 1px solid @border@; padding: 4px;
    selection-background-color: @accent_soft@; selection-color: @text@; outline: 0; }

QCheckBox, QRadioButton { spacing: 8px; color: @text@; }
QCheckBox:disabled { color: @dim@; }
QCheckBox::indicator, QListView::indicator { width: 16px; height: 16px; border-radius: 5px; border: 1px solid #5a5a5a;
    background: @control@; }
QCheckBox::indicator:hover, QListView::indicator:hover { border-color: @text2@; }
QCheckBox::indicator:checked, QListView::indicator:checked { background: @accent@; border-color: @accent@;
    image: url(@check@); }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 9px; border: 1px solid #5a5a5a;
    background: @control@; }
QRadioButton::indicator:checked { background: @accent@; border-color: @accent@; }

QSlider::groove:horizontal { height: 4px; background: #3a3a3a; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent@; border-radius: 2px; }
QSlider::add-page:horizontal { background: #3a3a3a; border-radius: 2px; }
QSlider::handle:horizontal { background: #e8e8e8; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider::handle:horizontal:disabled { background: @dim@; }
QSlider::sub-page:horizontal:disabled { background: #4a4a4a; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #3a3a3a; border-radius: 3px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #4d4d4d; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #3a3a3a; border-radius: 3px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #4d4d4d; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QListWidget { background: @window@; border: none; outline: 0; }
QListWidget::item { border-radius: 10px; padding: 4px; margin: 2px 6px; color: @text2@; }
QListWidget::item:hover { background: @panel@; }
QListWidget::item:selected { background: @accent_soft@; color: @text@; }

QTabWidget::pane { border: none; top: 4px; }
QTabBar::tab { background: transparent; color: @text2@; padding: 5px 12px; border-radius: 13px; margin-right: 4px; }
QTabBar::tab:hover { background: @hover@; color: @text@; }
QTabBar::tab:selected { background: @accent_soft@; color: @accent@; }

QProgressBar { background: @control@; border: none; border-radius: 6px; text-align: center; color: @text@;
    min-height: 12px; max-height: 14px; }
QProgressBar::chunk { background: @accent@; border-radius: 6px; }

QMenu { background: @panel@; color: @text@; border: 1px solid @border@; padding: 6px; }
QMenu::item { padding: 6px 20px 6px 12px; border-radius: 6px; }
QMenu::item:selected { background: @accent_soft@; }
QMenu::separator { height: 1px; background: @border@; margin: 4px 6px; }

QStatusBar { background: @window@; color: @text2@; border-top: 1px solid @border@; }
QStatusBar::item { border: none; }
QSplitter::handle { background: transparent; }
QSplitter::handle:horizontal { width: 6px; }

QFrame#qbar { background: @window@; border: 1px solid @border@; border-radius: 12px; }
QLabel#qpill { border-radius: 12px; padding: 3px 12px; font-weight: 600; background: @control@; color: @text2@; }
QLabel#qpill[level="bad"] { background: @bad_soft@; color: @bad@; }
QLabel#qpill[level="doubt"] { background: @warn_soft@; color: @warn@; }
QLabel#qpill[level="ok"] { background: @ok_soft@; color: @ok@; }
QPushButton[chip="true"] { min-height: 16px; padding: 3px 10px; border-radius: 12px; background: transparent;
    border: 1px solid @border@; color: @text2@; }
QPushButton[chip="true"]:hover { background: @hover@; color: @text@; }
QPushButton[chip="true"]:checked { background: @accent_soft@; border-color: transparent; color: @accent@; }

QLabel#toast { background: @panel@; color: @text@; border: 1px solid @border@; border-radius: 16px; padding: 8px 16px; }
"""


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lum(h: str) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in _hex_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """Контраст WCAG двух цветов `#rrggbb` (1…21)."""
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _svg_text(name: str, color: str) -> str:
    return (ICONS_DIR / f"{name}.svg").read_text(encoding="utf-8").replace("currentColor", color)


def icon_pixmap(name: str, color: str, size: int = 18, dpr: float = 2.0) -> QPixmap:
    """Иконка Tabler нужного цвета; рисуется с запасом по плотности пикселей — чёткая на любом экране."""
    px = int(size * dpr)
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    QSvgRenderer(QByteArray(_svg_text(name, color).encode("utf-8"))).render(p)
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    return pm


@lru_cache(maxsize=128)
def tool_icon(name: str, color: str | None = None, size: int = 18) -> QIcon:
    """Иконка для кнопки: в покое серая, при наведении светлая, включённая — акцентом, выключенная — тусклая.
    `color` задаёт один цвет для всех состояний (кроме выключенной), например на акцентной заливке."""
    ic = QIcon()
    states = ((QIcon.Normal, QIcon.Off, color or T["text2"]), (QIcon.Active, QIcon.Off, color or T["text"]),
              (QIcon.Normal, QIcon.On, color or T["accent"]), (QIcon.Active, QIcon.On, color or T["accent"]),
              (QIcon.Disabled, QIcon.Off, T["dim"]), (QIcon.Disabled, QIcon.On, T["dim"]))
    for mode, state, col in states:
        ic.addPixmap(icon_pixmap(name, col, size), mode, state)
    return ic


def _svg_file(name: str, color: str) -> str:
    """Цветная копия SVG в кеше: QSS не умеет перекрашивать картинки. Пустая строка — не удалось записать."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{name}-{color.lstrip('#')}.svg"
        if not path.exists():
            path.write_text(_svg_text(name, color), encoding="utf-8")
        return path.as_posix()
    except OSError:
        return ""


def build_qss() -> str:
    qss = _QSS
    tokens = dict(T, chev_down=_svg_file("chevron-down", T["text2"]), check=_svg_file("check", T["on_accent"]))
    for key, value in tokens.items():
        qss = qss.replace(f"@{key}@", value)
    return qss


def build_palette() -> QPalette:
    pal = QPalette()
    for role, key in ((QPalette.Window, "window"), (QPalette.WindowText, "text"), (QPalette.Base, "control"),
                      (QPalette.AlternateBase, "panel"), (QPalette.Text, "text"), (QPalette.Button, "control"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
                      (QPalette.HighlightedText, "on_accent"), (QPalette.ToolTipBase, "panel"),
                      (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "dim"), (QPalette.Link, "accent")):
        pal.setColor(role, QColor(T[key]))
    for role in (QPalette.WindowText, QPalette.ButtonText, QPalette.Text):
        pal.setColor(QPalette.Disabled, role, QColor(T["dim"]))
    return pal


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
    font.setPointSizeF(10)
    app.setFont(font)
    app.setPalette(build_palette())
    app.setStyleSheet(build_qss())
