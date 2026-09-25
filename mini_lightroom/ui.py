"""Окно Mini LightRoom (PySide6)."""
from __future__ import annotations

import copy
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import engine as E
from . import masks as MK
from . import scene as S
from . import segment as G
from .mask_editor import MaskEditor

APP_DIR = Path(__file__).resolve().parent.parent
STYLES_DIR = APP_DIR / "styles"
LOOKS_DIR = APP_DIR / "looks"
SCENES_FILE = APP_DIR / "scenes.json"
MODELS_DIR = APP_DIR / "models"
SCENES_KEY = "__scenes__"   # в sidecar: {имя файла: [id сцены, уверенность]}
PRESETS_DIR = APP_DIR / "presets"
SIDECAR = ".mini_lightroom.json"   # настройки кадров лежат рядом со снимками
MASKS_DIR = ".mini_lightroom_masks"  # ИИ-маски кадров (PNG) рядом со снимками
PREVIEW_SIDE = 1400
PATH_ROLE = Qt.UserRole
ZOOM_STEPS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 16]  # 1 = 100% (пиксель снимка = пиксель экрана)


# ---------------------------------------------------------------- фоновые задачи

class _Signals(QObject):
    done = Signal(object)
    fail = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn, args):
        super().__init__()
        self.fn, self.args, self.sig = fn, args, _Signals()
        self.setAutoDelete(False)

    def run(self):
        try:
            res = self.fn(*self.args)
        except Exception as e:  # noqa: BLE001 — показываем любую ошибку пользователю
            self.sig.fail.emit(f"{type(e).__name__}: {e}")
        else:
            self.sig.done.emit(res)


_alive: set = set()


def run_task(fn, *args, done=None, fail=None, pool: QThreadPool | None = None):
    t = _Task(fn, args)
    _alive.add(t)
    if done:
        t.sig.done.connect(done)
    if fail:
        t.sig.fail.connect(fail)
    t.sig.done.connect(lambda *_: _alive.discard(t))
    t.sig.fail.connect(lambda *_: _alive.discard(t))
    (pool or QThreadPool.globalInstance()).start(t)


def to_qimage(rgb_u8: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgb_u8)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def thumb_job(path):
    th = E.load_thumb(path, 512)
    return path, E.resize_max(th, 200), E.sharpness(th)


def preview_job(path):
    return path, E.load_image(path, half=True, max_side=PREVIEW_SIDE), E.full_size(path)


def look_icons_job(path, base, params, style, looks):
    """Миниатюры текущего кадра под каждым пресетом — видно, что выбираешь."""
    small = E.process(E.resize_max(base, 84), {**params, "look": "", "vignette": 0, "sharpness": 0}, style)
    return path, {name: to_u8(E.apply_look(small, look)) for name, look in looks.items()}


def scene_job(classifier, scenes, names, thumbs):
    if classifier is None:  # первый раз: загрузка модели (и скачивание весов)
        classifier = S.SceneClassifier(scenes, MODELS_DIR)
    return classifier, names, classifier.classify(thumbs)


def style_job(path, name):
    img = E.load_image(path, half=True, max_side=1024)
    return {"name": name, **E.lab_stats(img)}, to_u8(E.resize_max(img, 96))


def _seg_input(base):
    """Кадр для сегментации: с авто-тоном, чтобы тёмный RAW не путал модель."""
    return E.process(base, E.normalize_params(E.auto_params(base)))


def segment_job(segmenter, path, base, cats, create):
    if segmenter is None:  # первый раз: загрузка модели (и скачивание весов)
        segmenter = G.Segmenter(MODELS_DIR)
    maps = segmenter.masks(_seg_input(base), cats)
    return segmenter, path, {c: (m * 255 + 0.5).astype(np.uint8) for c, m in maps.items()}, create


def prepare_ai_job(segmenter, items, out_dir):
    """ИИ-маски для кадров, которые ещё не открывали (перед экспортом)."""
    if segmenter is None:
        segmenter = G.Segmenter(MODELS_DIR)
    for path, cats in items:
        base = E.load_image(path, half=True, max_side=PREVIEW_SIDE)
        for cat, m in segmenter.masks(_seg_input(base), cats).items():
            _write_png(Path(out_dir) / f"{Path(path).name}.{cat}.png", (m * 255 + 0.5).astype(np.uint8))
    return segmenter


def _write_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", arr)[1].tobytes())  # через Python: кириллица в пути


def full_job(path):
    return path, E.load_image(path, half=False)


def detail_job(gen, full, base, rect, scale, params, style, look):
    src_stats = E.style_source_stats(base, params) if style else None  # стиль — как у всего кадра
    before, after = E.process_region(full, rect, scale, params, style, src_stats, look)
    return gen, rect, to_u8(before), to_u8(after)


def render_job(gen, base, params, style, look):
    out = to_u8(E.process(base, params, style, look=look))
    hist = [np.histogram(out[..., c], bins=64, range=(0, 256))[0] for c in range(3)]
    return gen, out, hist


# ---------------------------------------------------------------- виджеты

class ImageView(QWidget):
    """Кадр по размеру окна или в масштабе. Масштаб — доля от полного разрешения снимка.
    Поверх растянутого превью рисуется «деталь»: видимая область в полном разрешении."""

    view_changed = Signal()

    def __init__(self):
        super().__init__()
        self.pix: QPixmap | None = None
        self.detail: tuple[QPixmap, QRectF] | None = None  # картинка и её место в пикселях кадра
        self.src_w, self.src_h = 1, 1                        # полный размер снимка
        self.zoom: float | None = None                       # None — вписать в окно
        self.cx = self.cy = 0.0                              # центр обзора в пикселях кадра
        self._drag = None
        self.message = "Откройте папку со снимками\nCtrl+O или кнопка «Открыть папку»"
        self.badge = ""
        self.editor = None  # MaskEditor: забирает левую кнопку, когда выбрана маска
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_image(self, img: QImage | None):
        self.pix = QPixmap.fromImage(img) if img is not None else None
        self.update()

    def set_detail(self, detail: tuple[QPixmap, QRectF] | None):
        self.detail = detail
        self.update()

    def set_source_size(self, w: int, h: int):
        self.src_w, self.src_h = max(1, w), max(1, h)
        self.cx, self.cy = w / 2, h / 2
        self.detail = None
        self._clamp()
        self.update()

    # ---------- геометрия

    def fit_scale(self) -> float:
        return min(self.width() * 0.97 / self.src_w, self.height() * 0.97 / self.src_h)

    def scale(self) -> float:
        return self.zoom or self.fit_scale()

    def _clamp(self):
        s = self.scale()
        for attr, size, view in (("cx", self.src_w, self.width()), ("cy", self.src_h, self.height())):
            half = view / 2 / s
            v = size / 2 if half * 2 >= size else min(max(getattr(self, attr), half), size - half)
            setattr(self, attr, v)

    def to_widget(self, r: QRectF) -> QRectF:
        s = self.scale()
        return QRectF(self.width() / 2 + (r.x() - self.cx) * s, self.height() / 2 + (r.y() - self.cy) * s,
                      r.width() * s, r.height() * s)

    def to_norm(self, pos: QPointF) -> tuple[float, float]:
        """Точка виджета → доли полного кадра."""
        s = self.scale()
        return ((self.cx + (pos.x() - self.width() / 2) / s) / self.src_w,
                (self.cy + (pos.y() - self.height() / 2) / s) / self.src_h)

    def to_widget_pt(self, xn: float, yn: float) -> QPointF:
        s = self.scale()
        return QPointF(self.width() / 2 + (xn * self.src_w - self.cx) * s,
                       self.height() / 2 + (yn * self.src_h - self.cy) * s)

    def visible_rect(self) -> QRectF:
        """Видимая часть кадра в его пикселях."""
        s = self.scale()
        r = QRectF(self.cx - self.width() / 2 / s, self.cy - self.height() / 2 / s,
                   self.width() / s, self.height() / s)
        return r.intersected(QRectF(0, 0, self.src_w, self.src_h))

    def set_zoom(self, z: float | None, anchor: QPointF | None = None):
        """Меняет масштаб, оставляя точку под anchor (координаты виджета) на месте."""
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        s0 = self.scale()
        ix = self.cx + (anchor.x() - self.width() / 2) / s0
        iy = self.cy + (anchor.y() - self.height() / 2) / s0
        self.zoom = z if z is None or z > self.fit_scale() * 1.001 else None
        s1 = self.scale()
        self.cx = ix - (anchor.x() - self.width() / 2) / s1
        self.cy = iy - (anchor.y() - self.height() / 2) / s1
        self._clamp()
        self.update()
        self.view_changed.emit()

    def step_zoom(self, direction: int, anchor: QPointF | None = None):
        s, fit = self.scale(), self.fit_scale()
        if direction > 0:
            nxt = next((z for z in ZOOM_STEPS if z > s * 1.001 and z > fit), ZOOM_STEPS[-1])
        else:
            nxt = next((z for z in reversed(ZOOM_STEPS) if z < s * 0.999), None)
            nxt = nxt if nxt is not None and nxt > fit else None
        self.set_zoom(nxt, anchor)

    # ---------- мышь

    def wheelEvent(self, e):
        if self.pix and e.angleDelta().y():
            self.step_zoom(1 if e.angleDelta().y() > 0 else -1, e.position())

    def _editing(self) -> bool:
        return self.editor is not None and self.editor.active()

    def mouseDoubleClickEvent(self, e):
        if self.pix and not self._editing():  # при рисовании двойной щелчок не прыгает масштабом
            self.set_zoom(None if self.zoom else 1.0, e.position())

    def mousePressEvent(self, e):
        # Правая и средняя кнопки всегда двигают кадр; левая — если не занята маской.
        pan = e.button() in (Qt.RightButton, Qt.MiddleButton) or (e.button() == Qt.LeftButton and not self._editing())
        if pan and self.zoom:
            self._drag = (e.position(), self.cx, self.cy)
            self.setCursor(Qt.ClosedHandCursor)
        elif self.editor is not None and self.editor.press(e, self):
            self.update()

    def mouseMoveEvent(self, e):
        if self._drag:
            p0, cx, cy = self._drag
            s = self.scale()
            self.cx, self.cy = cx - (e.position().x() - p0.x()) / s, cy - (e.position().y() - p0.y()) / s
            self._clamp()
            self.update()
            return
        if self.editor is not None and self.editor.move(e, self):
            self.update()
        if self._editing():
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.OpenHandCursor if self.zoom else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = None
            self.setCursor(Qt.OpenHandCursor)
            self.view_changed.emit()
        elif self.editor is not None and self.editor.release(e, self):
            self.update()

    def leaveEvent(self, e):
        if self.editor is not None:
            self.editor._hover = None
            self.update()
        super().leaveEvent(e)

    def resizeEvent(self, e):
        self._clamp()
        super().resizeEvent(e)
        self.view_changed.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#161616"))
        if self.pix:
            self._clamp()
            vis = self.visible_rect()
            k = self.pix.width() / self.src_w
            src = QRectF(vis.x() * k, vis.y() * k, vis.width() * k, vis.height() * k)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawPixmap(self.to_widget(vis), self.pix, src)
            if self.detail:
                dpix, rect = self.detail
                # От 200% показываем пиксели как есть, без сглаживания: так видна реальная резкость.
                p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale() * rect.width() < dpix.width() * 2)
                p.drawPixmap(self.to_widget(rect), dpix, QRectF(dpix.rect()))
            if self.editor is not None:
                self.editor.paint(p, self)
        if self.badge:
            p.setPen(QColor("#ffffff"))
            r = p.fontMetrics().boundingRect(self.badge).adjusted(-8, -4, 8, 4)
            r.moveTo(12, 12)
            p.fillRect(r, QColor(0, 0, 0, 170))
            p.drawText(r, Qt.AlignCenter, self.badge)
        if self.message:
            p.setPen(QColor("#9a9a9a"))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)


class Histogram(QWidget):
    def __init__(self):
        super().__init__()
        self.hist = None
        self.setFixedHeight(80)

    def set_hist(self, hist):
        self.hist = hist
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        if not self.hist:
            return
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height() - 2
        top = max(float(np.sqrt(ch).max()) for ch in self.hist) or 1
        for ch, color in zip(self.hist, ("#ff5a5a", "#5aff7a", "#5a8cff")):
            path = QPainterPath()
            path.moveTo(0, h)
            n = len(ch)
            for i, v in enumerate(ch):
                path.lineTo(i * w / (n - 1), h - np.sqrt(v) / top * (h - 4))
            path.lineTo(w, h)
            c = QColor(color)
            c.setAlpha(90)
            p.fillPath(path, c)


class ResetSlider(QSlider):
    reset = Signal()

    def mouseDoubleClickEvent(self, e):
        self.reset.emit()


class SliderRow(QWidget):
    changed = Signal(str, int)

    def __init__(self, key, label, lo, hi):
        super().__init__()
        self.key = key
        self.name = QLabel(label)
        self.value = QLabel()
        self.value.setAlignment(Qt.AlignRight)
        self.value.setMinimumWidth(70)
        self.slider = ResetSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setToolTip("Двойной щелчок — сбросить")
        self.slider.valueChanged.connect(self._on_change)
        self.slider.reset.connect(lambda: self.slider.setValue(0))
        top = QHBoxLayout()
        top.addWidget(self.name)
        top.addStretch()
        top.addWidget(self.value)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(0)
        lay.addLayout(top)
        lay.addWidget(self.slider)
        self._show(0)

    def _show(self, v):
        if self.key == "exposure":
            self.value.setText(f"{v / 100:+.2f} EV")
        elif self.key in ("sharpness", "brush_size", "mask_feather") or self.key.startswith(("style_", "look_")):
            self.value.setText(f"{v}")
        else:
            self.value.setText(f"{v:+d}" if v else "0")

    def _on_change(self, v):
        self._show(v)
        self.changed.emit(self.key, v)

    def set_value(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(v))
        self.slider.blockSignals(False)
        self._show(int(v))


# ---------------------------------------------------------------- экспорт

class ExportDialog(QDialog):
    def __init__(self, parent, n_selected, n_all, folder: Path, has_scenes: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Экспорт в JPEG")
        self.r_current = QRadioButton("Текущий кадр")
        self.r_selected = QRadioButton(f"Выделенные кадры ({n_selected})")
        self.r_all = QRadioButton(f"Все кадры в папке ({n_all})")
        self.r_selected.setEnabled(n_selected > 1)
        (self.r_selected if n_selected > 1 else self.r_current).setChecked(True)
        self.auto = QCheckBox("Авто-тон для каждого кадра")
        self.auto.setToolTip("Экспозицию, света/тени и баланс белого программа подберёт под каждый снимок,\n"
                             "а стиль, сочность, чёткость и резкость возьмёт ваши.")
        self.scene_look = QCheckBox("Пресет по сцене для каждого кадра")
        self.scene_look.setEnabled(has_scenes)
        self.scene_look.setToolTip("Каждый кадр получит авто-тон и пресет своей сцены (закат, портрет, лес…).\n"
                                   "Доступно после кнопки «Сцены» на панели.")
        self.skip_blur = QCheckBox("Пропускать размытые кадры")
        self.edge = QSpinBox()
        self.edge.setRange(0, 12000)
        self.edge.setSingleStep(500)
        self.edge.setSpecialValueText("как в оригинале")
        self.edge.setSuffix(" px")
        self.quality = QSpinBox()
        self.quality.setRange(60, 100)
        self.quality.setValue(92)
        self.out = QLineEdit(str(folder / "export"))
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out)
        out_row.addWidget(browse)

        hint = QLabel("Кадры без своих настроек получат настройки текущего кадра.")
        hint.setStyleSheet("color:#9a9a9a")
        form = QFormLayout()
        form.addRow("Длинная сторона:", self.edge)
        form.addRow("Качество JPEG:", self.quality)
        form.addRow("Папка:", out_row)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Экспортировать")
        btns.button(QDialogButtonBox.Cancel).setText("Отмена")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        for w in (self.r_current, self.r_selected, self.r_all, hint, self.auto, self.scene_look, self.skip_blur):
            lay.addWidget(w)
        lay.addLayout(form)
        lay.addWidget(btns)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Куда сохранить", self.out.text())
        if d:
            self.out.setText(d)


class ExportThread(QThread):
    progress = Signal(int, int)
    finished_all = Signal(int, list)

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs
        self.stop = False

    def run(self):
        workers = max(1, min(2, (os.cpu_count() or 2) // 4))  # полный RAW занимает ~1 ГБ памяти
        errors, done = [], 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(E.export_one, j): j for j in self.jobs}
            for f in as_completed(futures):
                if self.stop:
                    ex.shutdown(cancel_futures=True)
                    break
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{Path(futures[f]['src']).name}: {e}")
                done += 1
                self.progress.emit(done, len(self.jobs))
        self.finished_all.emit(done - len(errors), errors)


# ---------------------------------------------------------------- главное окно

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini LightRoom")
        self.resize(1500, 900)
        self.folder: Path | None = None
        self.files: list[Path] = []
        self.items: dict[str, QListWidgetItem] = {}
        self.sidecar: dict[str, dict] = {}
        self.scores: dict[str, float] = {}
        self.blurry: set[str] = set()
        self.current: Path | None = None
        self.base: np.ndarray | None = None
        self.before: QImage | None = None
        self.after: QImage | None = None
        self.params = E.default_params()
        self.clipboard: dict | None = None
        self.cache: dict[Path, tuple] = {}  # превью и полный размер
        self.gen = 0
        self.rendering = False
        self.dirty = False
        self.full: np.ndarray | None = None       # полное разрешение текущего кадра (для масштаба)
        self.full_path: Path | None = None
        self.full_loading: Path | None = None
        self.detail_gen = 0
        self.detail_busy = False
        self.detail_dirty = False
        self.detail_pair: tuple | None = None     # (до, после, место в кадре)
        self.detail_timer = QTimer(self, singleShot=True, interval=150, timeout=self.request_detail)
        self.export_thread: ExportThread | None = None
        self.thumbs: dict[str, np.ndarray] = {}
        self.scene_defs: list[dict] = E.read_json(SCENES_FILE, [])
        self.scene_by_id = {sc["id"]: sc for sc in self.scene_defs}
        self.scene_of: dict[str, list] = {}
        self.classifier = None
        self.segmenter = None
        self.ai_cache: dict[tuple[str, str], np.ndarray] = {}
        self.editor = MaskEditor(self.ai_arr_for_layer)
        self.editor.changed.connect(self.on_mask_geom)
        self.editor.created.connect(self.on_mask_created)
        self.thumb_pool = QThreadPool()
        self.thumb_pool.setMaxThreadCount(2)
        STYLES_DIR.mkdir(exist_ok=True)
        PRESETS_DIR.mkdir(exist_ok=True)

        self._build_toolbar()
        self._build_body()
        self._build_status()
        self.reload_styles()
        self.reload_presets()
        self.reload_looks()
        self._set_enabled(False)

    # ---------- построение интерфейса

    def _action(self, text, slot, shortcut=None, tip=""):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip(f"{tip or text} ({shortcut})" if shortcut else (tip or text))
        self.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction(self._action("📂  Открыть папку", self.open_folder, "Ctrl+O"))
        tb.addSeparator()
        self.a_auto = self._action("✨ Авто", self.apply_auto, "A", "Подобрать тон и баланс белого под кадр")
        self.a_reset = self._action("↺ Сброс", self.reset_all, "Ctrl+R", "Сбросить все правки кадра")
        self.a_compare = self._action("◐ До / после", self.toggle_compare, "\\", "Показать исходник")
        self.a_compare.setCheckable(True)
        self.a_copy = self._action("Копировать правки", self.copy_settings, "Ctrl+C")
        self.a_paste = self._action("Вставить в выделенные", self.paste_settings, "Ctrl+V")
        for a in (self.a_auto, self.a_reset, self.a_compare, self.a_copy, self.a_paste):
            tb.addAction(a)
        tb.addSeparator()
        self.auto_on_open = QCheckBox("Авто для новых кадров")
        self.auto_on_open.setChecked(True)
        self.auto_on_open.setToolTip("Кадр без правок при открытии сразу получает «Авто»")
        tb.addWidget(self.auto_on_open)
        tb.addSeparator()
        self.a_scenes = self._action("🔍 Сцены", self.detect_scenes, "Ctrl+Shift+S",
                                     "Определить сцену каждого кадра: закат, портрет, лес… (ИИ локально на видеокарте)")
        tb.addAction(self.a_scenes)
        self.scene_auto = QCheckBox("Пресет по сцене")
        self.scene_auto.setChecked(True)
        self.scene_auto.setToolTip("«Авто» и новые кадры получают пресет своей сцены.\n"
                                   "Работает для кадров, у которых сцена уже определена")
        tb.addWidget(self.scene_auto)
        tb.addSeparator()
        self.a_zoom_out = self._action("−", lambda: self.view.step_zoom(-1), "Ctrl+-", "Уменьшить масштаб")
        self.a_zoom_in = self._action("+", lambda: self.view.step_zoom(1), "Ctrl+=", "Увеличить масштаб")
        self.a_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        self.a_fit = self._action("Вписать", lambda: self.view.set_zoom(None), "Ctrl+0")
        self.a_100 = self._action("100%", lambda: self.view.set_zoom(1.0), "Ctrl+1")
        self._action("Показать маску", lambda: self.mask_show.toggle(), "O")
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("Вписать")
        self.zoom_combo.addItems([f"{round(z * 100)}%" for z in ZOOM_STEPS])
        self.zoom_combo.setToolTip("Масштаб от полного разрешения снимка.\nКолёсико — приблизить к курсору, "
                                   "перетаскивание — сдвиг,\nдвойной щелчок — 100% / вписать")
        self.zoom_combo.activated.connect(
            lambda i: self.view.set_zoom(None if i == 0 else ZOOM_STEPS[i - 1]))
        tb.addAction(self.a_zoom_out)
        tb.addWidget(self.zoom_combo)
        tb.addAction(self.a_zoom_in)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_export = self._action("⬇  Экспорт…", self.export, "Ctrl+E")
        tb.addAction(self.a_export)

    def _build_body(self):
        self.strip = QListWidget()
        self.strip.setViewMode(QListWidget.ListMode)
        self.strip.setIconSize(QSize(160, 110))
        self.strip.setSelectionMode(QListWidget.ExtendedSelection)
        self.strip.setMinimumWidth(230)
        self.strip.currentItemChanged.connect(self.on_select)

        self.view = ImageView()
        self.view.view_changed.connect(self.on_view_changed)
        self.view.editor = self.editor

        panel = QWidget()
        pl = QVBoxLayout(panel)
        self.hist = Histogram()
        pl.addWidget(self.hist)
        self.rows: dict[str, SliderRow] = {}
        groups: dict[str, QVBoxLayout] = {}
        for key, label, lo, hi, group in E.SLIDERS:
            if group not in groups:
                box = QGroupBox(group)
                groups[group] = QVBoxLayout(box)
                pl.addWidget(box)
            row = SliderRow(key, label, lo, hi)
            row.changed.connect(self.on_param)
            groups[group].addWidget(row)
            self.rows[key] = row

        # Маски: локальные правки
        box = QGroupBox("Маски")
        ml = QVBoxLayout(box)
        row = QHBoxLayout()
        for text, kind, tip in (
                ("🖌 Кисть", "brush", "Рисуйте по кадру, где нужна правка. Alt — стереть"),
                ("▤ Линейный", "linear", "Протяните по кадру: сила от начала линии до нуля в конце"),
                ("◎ Радиальный", "radial", "Протяните от центра: овал, внутри — правка")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=kind: self.add_mask(k))
            row.addWidget(b)
        ai_btn = QToolButton()
        ai_btn.setText("✨ ИИ")
        ai_btn.setToolTip("Маска по содержимому кадра: небо, люди, зелень, вода, здания (локально на видеокарте)")
        ai_btn.setPopupMode(QToolButton.InstantPopup)
        ai_menu = QMenu(ai_btn)
        for cat, (label, _) in G.CATEGORIES.items():
            ai_menu.addAction(label, lambda c=cat: self.add_ai_mask(c))
        ai_btn.setMenu(ai_menu)
        row.addWidget(ai_btn)
        ml.addLayout(row)
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(96)
        self.mask_list.setToolTip("Галочка — включить/выключить, двойной щелчок — переименовать")
        self.mask_list.currentRowChanged.connect(self.on_mask_select)
        self.mask_list.itemChanged.connect(self.on_mask_item)
        ml.addWidget(self.mask_list)
        self.mask_box = QWidget()
        mb = QVBoxLayout(self.mask_box)
        mb.setContentsMargins(0, 0, 0, 0)
        opts = QHBoxLayout()
        self.mask_invert = QCheckBox("Инверсия")
        self.mask_invert.toggled.connect(self.on_mask_invert)
        self.mask_show = QCheckBox("Показать (O)")
        self.mask_show.setToolTip("Подсветить маску красным")
        self.mask_show.toggled.connect(self.on_mask_show)
        del_btn = QPushButton("Удалить")
        del_btn.clicked.connect(self.delete_mask)
        for w in (self.mask_invert, self.mask_show):
            opts.addWidget(w)
        opts.addStretch()
        opts.addWidget(del_btn)
        mb.addLayout(opts)
        self.brush_size_row = SliderRow("brush_size", "Размер кисти", 2, 200)
        self.brush_size_row.set_value(self.editor.brush_size)
        self.brush_size_row.changed.connect(lambda _k, v: setattr(self.editor, "brush_size", v))
        self.brush_erase = QCheckBox("Стирать (или удерживайте Alt)")
        self.brush_erase.toggled.connect(lambda on: setattr(self.editor, "erase", on))
        self.mask_feather_row = SliderRow("mask_feather", "Растушёвка", 0, 100)
        self.mask_feather_row.changed.connect(self.on_mask_feather)
        for w in (self.brush_size_row, self.brush_erase, self.mask_feather_row):
            mb.addWidget(w)
        limits = {key: (label, lo, hi) for key, label, lo, hi, _ in E.SLIDERS}
        self.mask_rows: dict[str, SliderRow] = {}
        for key in MK.LOCAL_KEYS:
            label, lo, hi = limits[key]
            r = SliderRow(key, label, lo, hi)
            r.changed.connect(self.on_mask_adj)
            mb.addWidget(r)
            self.mask_rows[key] = r
        ml.addWidget(self.mask_box)
        self.mask_box.hide()
        pl.addWidget(box)

        # Готовые пресеты-образы с силой
        box = QGroupBox("Пресеты")
        ll = QVBoxLayout(box)
        self.look_combo = QComboBox()
        self.look_combo.setIconSize(QSize(64, 44))
        self.look_combo.setMaxVisibleItems(14)
        self.look_combo.currentIndexChanged.connect(self.on_look)
        ll.addWidget(self.look_combo)
        self.rows["look_strength"] = SliderRow("look_strength", "Сила пресета", 0, 100)
        self.rows["look_strength"].changed.connect(self.on_param)
        ll.addWidget(self.rows["look_strength"])
        pl.addWidget(box)

        # Стиль с чужого фото
        box = QGroupBox("Стиль с референса")
        sl = QVBoxLayout(box)
        top = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.setIconSize(QSize(48, 32))
        self.style_combo.currentIndexChanged.connect(self.on_style)
        top.addWidget(self.style_combo, 1)
        self.style_menu_btn = QToolButton()
        self.style_menu_btn.setText("⋯")
        self.style_menu_btn.setToolTip("Переименовать или удалить стиль")
        self.style_menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.style_menu_btn)
        menu.addAction("Переименовать…", self.rename_style)
        menu.addAction("Удалить…", self.delete_style)
        self.style_menu_btn.setMenu(menu)
        top.addWidget(self.style_menu_btn)
        sl.addLayout(top)
        b = QPushButton("＋ Новый стиль из фото…")
        b.setToolTip("Выберите чужой снимок, цвета и настроение которого нравятся")
        b.clicked.connect(self.new_style)
        sl.addWidget(b)
        self.style_mode = QComboBox()
        self.style_mode.addItems(["Мягкий перенос", "Точный перенос"])
        self.style_mode.setToolTip("Мягкий — переносит общий цвет и контраст референса.\n"
                                   "Точный — повторяет распределение цветов и тонов референса целиком.")
        self.style_mode.currentIndexChanged.connect(lambda i: self.on_param("style_mode", i))
        sl.addWidget(self.style_mode)
        for key, label, tip in (
                ("style_strength", "Цвет стиля", "Сколько цвета брать у референса. 0 — только тон"),
                ("style_tone", "Свет стиля", "Сколько света и контраста брать у референса. 0 — только цвет"),
                ("style_skin", "Защита кожи", "Не перекрашивать тона кожи: лица остаются естественными")):
            self.rows[key] = SliderRow(key, label, 0, 100)
            self.rows[key].setToolTip(tip)
            self.rows[key].changed.connect(self.on_param)
            sl.addWidget(self.rows[key])
        self.style2_combo = QComboBox()
        self.style2_combo.setIconSize(QSize(48, 32))
        self.style2_combo.setToolTip("Смешать с другим стилем: доля — ползунком ниже")
        self.style2_combo.currentIndexChanged.connect(self.on_style2)
        sl.addWidget(self.style2_combo)
        self.rows["style_mix"] = SliderRow("style_mix", "Доля второго стиля", 0, 100)
        self.rows["style_mix"].changed.connect(self.on_param)
        sl.addWidget(self.rows["style_mix"])
        b = QPushButton("Сохранить как LUT (.cube)…")
        b.setToolTip("Цвет, стиль и пресет в файле .cube для Lightroom, DaVinci или телефона.\n"
                     "Можно сохранить и только стиль, без ваших ползунков")
        b.clicked.connect(self.export_lut)
        sl.addWidget(b)
        pl.addWidget(box)

        # Пресеты
        box = QGroupBox("Мои пресеты")
        prl = QHBoxLayout(box)
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self.apply_preset)
        prl.addWidget(self.preset_combo, 1)
        b = QPushButton("Сохранить")
        b.clicked.connect(self.save_preset)
        prl.addWidget(b)
        pl.addWidget(box)
        pl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        self.panel = panel

        split = QSplitter()
        split.addWidget(self.strip)
        split.addWidget(self.view)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 950, 320])
        self.setCentralWidget(split)

    def _build_status(self):
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.cancel_btn = QPushButton("Остановить")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self.cancel_export)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

    def _set_enabled(self, on):
        self.panel.setEnabled(on)
        for a in (self.a_auto, self.a_reset, self.a_compare, self.a_copy, self.a_paste, self.a_export,
                  self.a_zoom_in, self.a_zoom_out, self.a_fit, self.a_100, self.a_scenes):
            a.setEnabled(on)
        self.zoom_combo.setEnabled(on)

    def toast(self, text, ms=4000):
        self.statusBar().showMessage(text, ms)

    # ---------- папка и миниатюры

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Папка со снимками", str(self.folder or Path.home()))
        if d:
            self.load_folder(Path(d))

    def load_folder(self, folder: Path):
        self.save_sidecar()
        self.folder = folder
        self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in E.PHOTO_EXT)
        self.sidecar = E.read_json(folder / SIDECAR, {})
        self.scene_of = dict(self.sidecar.get(SCENES_KEY, {}))
        self.thumbs.clear()
        self.scores.clear()
        self.blurry.clear()
        self.items.clear()
        self.cache.clear()
        self.current = None
        self.strip.clear()
        self.setWindowTitle(f"Mini LightRoom — {folder}")
        if not self.files:
            self.view.set_image(None)
            self.view.message = "В этой папке нет снимков (ARW, DNG, NEF, CR3, JPEG…)\nОткройте другую папку"
            self.view.update()
            self._set_enabled(False)
            return
        placeholder = QPixmap(160, 106)
        placeholder.fill(QColor("#2a2a2a"))
        for p in self.files:
            it = QListWidgetItem(QIcon(placeholder), p.name)
            it.setData(PATH_ROLE, str(p))
            self.strip.addItem(it)
            self.items[p.name] = it
            run_task(thumb_job, p, done=self.on_thumb, pool=self.thumb_pool)
        self.toast(f"Снимков в папке: {len(self.files)}. Проверяю резкость…")
        self.strip.setCurrentRow(0)

    def on_thumb(self, res):
        path, th, score = res
        it = self.items.get(Path(path).name)
        if it is None:
            return
        it.setIcon(QIcon(QPixmap.fromImage(to_qimage(th))))
        self.scores[Path(path).name] = score
        self.thumbs[Path(path).name] = th
        self.update_item(Path(path).name)
        if len(self.scores) == len(self.files) and len(self.files) >= 3:
            med = statistics.median(self.scores.values())
            self.blurry = {n for n, s in self.scores.items() if s < med * 0.3}
            for n in self.items:
                self.update_item(n)
            self.toast(f"Резкость проверена: подозрительно размытых {len(self.blurry)}")

    def update_item(self, name: str):
        """Подпись кадра в ленте: имя, пометка размытости, сцена."""
        it = self.items.get(name)
        if it is None:
            return
        lines = [f"⚠ {name}", "возможно, размыт"] if name in self.blurry else [name]
        sc = self.scene_by_id.get(self.scene_of.get(name, [None])[0])
        if sc:
            lines.append(f"{sc.get('icon', '')} {sc['name']}")
        it.setText("\n".join(lines))
        if name in self.blurry:
            it.setForeground(QColor("#ff9b73"))

    # ---------- выбор кадра

    def on_select(self, item, _prev=None):
        if item is None:
            return
        self.store_current()
        self.current = Path(item.data(PATH_ROLE))
        self.base = None
        self.full = self.full_path = None
        self.view.message = "Проявляю RAW…"
        self.view.badge = ""
        self.view.update()
        if self.current in self.cache:
            self.on_preview((self.current, *self.cache[self.current]))
        else:
            run_task(preview_job, self.current, done=self.on_preview, fail=self.on_error)

    def on_preview(self, res):
        path, img, size = res
        self.cache[path] = (img, size)
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        if path != self.current:
            return
        self.base = img
        self.view.set_source_size(*size)
        self.before = to_qimage(to_u8(img))
        saved = self.sidecar.get(path.name)
        self.params = E.normalize_params(saved)
        if not saved and self.auto_on_open.isChecked():
            self.params.update(self.auto_for(path.name, img)[0])
        self.sync_controls()
        self._set_enabled(True)
        self.view.message = ""
        self.request_render()
        self.refresh_look_icons()
        self.ensure_ai_masks()

    def on_error(self, msg):
        self.view.message = f"Не получилось открыть кадр:\n{msg}"
        self.view.set_image(None)
        self.toast("Ошибка чтения файла — выберите другой кадр", 8000)

    def store_current(self):
        if self.current is not None and self.base is not None:
            self.sidecar[self.current.name] = dict(self.params)

    def save_sidecar(self):
        self.store_current()
        if self.scene_of:
            self.sidecar[SCENES_KEY] = self.scene_of
        if self.folder and self.sidecar:
            try:
                E.write_json(self.folder / SIDECAR, self.sidecar)
            except OSError as e:
                self.toast(f"Не удалось сохранить правки: {e}", 8000)

    # ---------- правки и рендер

    def sync_controls(self):
        for key, row in self.rows.items():
            row.set_value(self.params.get(key, 0))
        self.refresh_mask_list()
        for combo, key in ((self.style_combo, "style"), (self.style2_combo, "style2")):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(self.params.get(key) or "")))
            combo.blockSignals(False)
        self.style_mode.blockSignals(True)
        self.style_mode.setCurrentIndex(int(self.params.get("style_mode", 0)))
        self.style_mode.blockSignals(False)
        self.look_combo.blockSignals(True)
        self.look_combo.setCurrentIndex(max(0, self.look_combo.findData(self.params.get("look") or "")))
        self.look_combo.blockSignals(False)

    def on_param(self, key, value):
        self.params[key] = value
        self.request_render()

    def current_style(self):
        return self.style_for(self.params)

    def style_for(self, p: dict) -> dict | None:
        """Стиль кадра с учётом смеси со вторым стилем."""
        a = self.styles.get(p.get("style") or "")
        b = self.styles.get(p.get("style2") or "")
        if a and b and b is not a and p.get("style_mix", 0):
            return E.mix_styles(a, b, p["style_mix"] / 100)
        return a

    def current_look(self):
        return self.looks.get(self.params.get("look") or "")

    def request_render(self):
        self.detail_gen += 1  # деталь с прежними правками больше не годится
        self.detail_pair = None
        if self.base is None:
            return
        if self.rendering:
            self.dirty = True
            return
        self.rendering = True
        self.gen += 1
        run_task(render_job, self.gen, self.base, self.render_params(), self.current_style(), self.current_look(),
                 done=self.on_rendered, fail=self.on_render_fail)

    def on_rendered(self, res):
        _gen, out, hist = res
        self.rendering = False
        self.after = to_qimage(out)
        self.hist.set_hist(hist)
        self.show_current()
        self.detail_timer.start()
        if self.dirty:
            self.dirty = False
            self.request_render()

    def on_render_fail(self, msg):
        self.rendering = False
        self.toast(f"Ошибка обработки: {msg}", 8000)

    def show_current(self):
        before = self.a_compare.isChecked()
        self.view.badge = "ДО" if before else ""
        self.view.set_image(self.before if before else self.after)
        if self.detail_pair:
            b, a, rect = self.detail_pair
            self.view.set_detail((b if before else a, rect))
        else:
            self.view.set_detail(None)

    # ---------- масштаб

    def on_view_changed(self):
        v = self.view
        self.zoom_combo.setItemText(0, f"Вписать ({round(v.fit_scale() * 100)}%)")
        idx = 0 if v.zoom is None else next((i + 1 for i, z in enumerate(ZOOM_STEPS) if abs(z - v.zoom) < 1e-6), 0)
        self.zoom_combo.setCurrentIndex(idx)
        self.detail_timer.start()

    def request_detail(self):
        """Видимая область в полном разрешении, когда превью растянуто больше своего размера."""
        v = self.view
        if self.base is None or v.scale() * v.src_w <= self.base.shape[1] * 1.05:
            return
        if self.full_path != self.current:
            if self.full_loading != self.current:
                self.full_loading = self.current
                self.toast("Загружаю полное разрешение для просмотра в масштабе…", 0)
                run_task(full_job, self.current, done=self.on_full, fail=self.on_full_fail)
            return
        if self.detail_busy:
            self.detail_dirty = True
            return
        r = v.visible_rect()
        h, w = self.full.shape[:2]
        rect = (max(0, int(r.left())), max(0, int(r.top())),
                min(w, int(np.ceil(r.right()))), min(h, int(np.ceil(r.bottom()))))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            return
        self.detail_busy = True
        run_task(detail_job, self.detail_gen, self.full, self.base, rect, min(1.0, v.scale()),
                 self.render_params(), self.current_style(), self.current_look(), done=self.on_detail, fail=self.on_detail_fail)

    def on_full(self, res):
        path, img = res
        if self.full_loading == path:
            self.full_loading = None
        if path != self.current:
            return
        self.full, self.full_path = img, path
        h, w = img.shape[:2]
        v = self.view
        if (w, h) != (v.src_w, v.src_h):  # размер из заголовка RAW может чуть отличаться
            v.cx, v.cy = v.cx * w / v.src_w, v.cy * h / v.src_h
            v.src_w, v.src_h = w, h
        self.toast("Полное разрешение загружено", 2000)
        self.request_detail()

    def on_full_fail(self, msg):
        self.full_loading = None
        self.toast(f"Не удалось загрузить полное разрешение: {msg}", 8000)

    def on_detail(self, res):
        gen, (x0, y0, x1, y1), before, after = res
        self.detail_busy = False
        if gen == self.detail_gen and self.full_path == self.current:
            self.detail_pair = (QPixmap.fromImage(to_qimage(before)), QPixmap.fromImage(to_qimage(after)),
                                QRectF(x0, y0, x1 - x0, y1 - y0))
            self.show_current()
        if self.detail_dirty:
            self.detail_dirty = False
            self.request_detail()

    def on_detail_fail(self, msg):
        self.detail_busy = False
        self.toast(f"Ошибка обработки в масштабе: {msg}", 8000)

    def toggle_compare(self):
        self.show_current()

    def apply_auto(self):
        if self.base is None:
            return
        auto, msg = self.auto_for(self.current.name, self.base)
        self.params.update(auto)
        self.sync_controls()
        self.request_render()
        self.toast(msg)

    # ---------- маски (этап 3)

    def masks(self) -> list:
        return self.params.setdefault("masks", [])

    def ai_arr(self, name: str, cat: str) -> np.ndarray | None:
        """ИИ-маска кадра (uint8, весь кадр) из памяти или из PNG рядом со снимками."""
        key = (name, cat)
        if key not in self.ai_cache and self.folder:
            f = self.folder / MASKS_DIR / f"{name}.{cat}.png"
            if f.exists():
                self.ai_cache[key] = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        return self.ai_cache.get(key)

    def ai_arr_for_layer(self, layer: dict) -> np.ndarray | None:
        return self.ai_arr(self.current.name, layer["cat"]) if self.current else None

    def render_params(self, p: dict | None = None, name: str | None = None) -> dict:
        """Копия правок для фоновой обработки: маски отдельно от окна, ИИ-слои с массивами."""
        p = self.params if p is None else p
        name = name or (self.current.name if self.current else "")
        layers = []
        for layer in p.get("masks", []):
            c = copy.deepcopy(layer)
            if c["type"] == "ai":
                c["arr"] = self.ai_arr(name, c["cat"])
            layers.append(c)
        return {**p, "masks": layers}

    def refresh_mask_list(self, select: int | None = None):
        layers = self.masks()
        if select is None:  # сохраняем выбранный слой, если он есть в новом списке
            select = next((i for i, lay in enumerate(layers) if lay is self.editor.layer), -1)
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for lay in layers:
            it = QListWidgetItem(lay["name"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if lay.get("on", True) else Qt.Unchecked)
            self.mask_list.addItem(it)
        self.mask_list.setCurrentRow(select)
        self.mask_list.blockSignals(False)
        self.on_mask_select(select)

    def on_mask_select(self, row: int):
        layers = self.masks()
        layer = layers[row] if 0 <= row < len(layers) else None
        if layer is not self.editor.layer:
            self.editor.set_layer(layer)
        self.mask_box.setVisible(layer is not None)
        if layer is not None:
            kind = layer["type"]
            self.brush_size_row.setVisible(kind == "brush")
            self.brush_erase.setVisible(kind == "brush")
            self.mask_feather_row.setVisible(kind in ("brush", "radial"))
            self.mask_feather_row.set_value(layer.get("feather", 50))
            self.mask_invert.blockSignals(True)
            self.mask_invert.setChecked(layer.get("invert", False))
            self.mask_invert.blockSignals(False)
            for key, r in self.mask_rows.items():
                r.set_value(layer["adj"].get(key, 0))
        self.view.update()

    def add_mask(self, kind: str):
        if self.base is None:
            return
        if kind == "brush":
            self.masks().append(MK.new_layer("brush", feather=self.mask_feather_row.slider.value() or 50))
            self.refresh_mask_list(select=len(self.masks()) - 1)
            self.toast("Рисуйте по кадру левой кнопкой. Alt — стереть, правая кнопка — сдвинуть кадр", 8000)
            return
        self.mask_list.setCurrentRow(-1)
        self.editor.set_layer(None)
        self.editor.creating = kind
        self.mask_box.hide()
        what = "от места, где правка полная, до места, где она сходит на нет" if kind == "linear" \
            else "от центра наружу — получится овал"
        self.toast(f"Протяните мышью по кадру {what}", 8000)

    def on_mask_created(self, layer: dict):
        self.masks().append(layer)
        self.refresh_mask_list(select=len(self.masks()) - 1)
        self.toast("Готово. Теперь двигайте ползунки маски; ручки на кадре меняют форму")

    def on_mask_geom(self):
        self.request_render()
        self.view.update()

    def on_mask_adj(self, key: str, value: int):
        if self.editor.layer is not None:
            self.editor.layer["adj"][key] = value
            self.request_render()

    def on_mask_invert(self, on: bool):
        if self.editor.layer is not None:
            self.editor.layer["invert"] = on
            self.editor.invalidate()
            self.on_mask_geom()

    def on_mask_feather(self, _key: str, value: int):
        if self.editor.layer is not None:
            self.editor.layer["feather"] = value
            self.editor.brush_feather = value
            self.editor.invalidate()
            self.on_mask_geom()

    def on_mask_show(self, on: bool):
        self.editor.show_overlay = on
        self.editor.invalidate()
        self.view.update()

    def on_mask_item(self, it: QListWidgetItem):
        row = self.mask_list.row(it)
        layer = self.masks()[row]
        layer["on"] = it.checkState() == Qt.Checked
        layer["name"] = it.text().strip() or layer["name"]
        self.editor.invalidate()
        self.on_mask_geom()

    def delete_mask(self):
        layer = self.editor.layer
        if layer is None:
            return
        self.masks().remove(layer)
        self.editor.set_layer(None)
        self.refresh_mask_list(select=-1)
        self.request_render()
        self.toast(f"Маска «{layer['name']}» удалена")

    def add_ai_mask(self, cat: str):
        if not G.available():
            QMessageBox.information(self, "Нужны библиотеки ИИ",
                                    "Для масок ИИ нужен PyTorch и transformers (один раз).\n"
                                    "Запустите install_ai.bat в папке программы, затем перезапустите её.")
            return
        if self.base is None:
            return
        label = G.CATEGORIES[cat][0]
        if self.ai_arr(self.current.name, cat) is not None:
            self._create_ai_layer(cat)
            return
        first = self.segmenter is None
        self.toast(f"Ищу «{label}» на кадре…" + (" В первый раз загружается модель (~110 МБ)" if first else ""), 0)
        run_task(segment_job, self.segmenter, self.current, self.base, [cat], True,
                 done=self.on_segmented, fail=lambda m: self.toast(f"Маска ИИ не получилась: {m}", 10000))

    def ensure_ai_masks(self):
        """У кадра есть ИИ-слои без готовой маски (например, после вставки правок) — досчитать в фоне."""
        cats = sorted({m["cat"] for m in self.masks() if m["type"] == "ai"
                       and self.ai_arr(self.current.name, m["cat"]) is None})
        if cats and G.available():
            run_task(segment_job, self.segmenter, self.current, self.base, cats, False,
                     done=self.on_segmented, fail=lambda m: self.toast(f"Маска ИИ не получилась: {m}", 10000))

    def on_segmented(self, res):
        self.segmenter, path, maps, create = res
        for cat, arr in maps.items():
            self.ai_cache[(path.name, cat)] = arr
            try:
                _write_png(path.parent / MASKS_DIR / f"{path.name}.{cat}.png", arr)
            except OSError as e:
                self.toast(f"Не удалось сохранить маску: {e}", 8000)
        if path != self.current:
            return
        if create:
            for cat in maps:
                self._create_ai_layer(cat)
        else:
            self.editor.invalidate()
            self.request_render()

    def _create_ai_layer(self, cat: str):
        label = G.CATEGORIES[cat][0]
        arr = self.ai_arr(self.current.name, cat)
        self.masks().append(MK.new_layer("ai", cat=cat, name=label))
        self.refresh_mask_list(select=len(self.masks()) - 1)
        share = float((arr > 127).mean()) if arr is not None else 0
        if share < 0.005:
            self.toast(f"«{label}» на кадре почти не найдено — маска пустая. Попробуйте другую", 10000)
        else:
            self.toast(f"Маска «{label}»: {share:.0%} кадра. Включите «Показать (O)», чтобы увидеть её")

    # ---------- сцены (этап 1)

    def scene_for(self, name: str) -> dict | None:
        return self.scene_by_id.get(self.scene_of.get(name, [None])[0])

    def auto_for(self, name: str, img: np.ndarray) -> tuple[dict, str]:
        """Авто-параметры кадра: с пресетом сцены, если она известна и галочка включена."""
        sc = self.scene_for(name) if self.scene_auto.isChecked() else None
        if sc and sc.get("look") in self.looks:
            return E.scene_preset(img, sc), f"Авто по сцене «{sc['name']}»: пресет «{sc['look']}»"
        return E.auto_params(img), "Авто: тон и баланс белого подобраны под кадр"

    def detect_scenes(self):
        if not S.available():
            QMessageBox.information(
                self, "Нужны библиотеки ИИ",
                "Для распознавания сцен нужен PyTorch (≈3 ГБ, один раз).\n"
                "Запустите install_ai.bat в папке программы, затем перезапустите её.")
            return
        names = [n for n in self.items if n in self.thumbs]
        if not names:
            self.toast("Миниатюры ещё загружаются, попробуйте через пару секунд")
            return
        self.a_scenes.setEnabled(False)
        first = self.classifier is None
        self.toast("Загружаю модель сцен (в первый раз скачивается ~600 МБ)…" if first
                   else f"Распознаю сцены: {len(names)} кадров…", 0)
        run_task(scene_job, self.classifier, self.scene_defs, names, [self.thumbs[n] for n in names],
                 done=self.on_scenes, fail=self.on_scenes_fail)

    def on_scenes(self, res):
        self.classifier, names, results = res
        self.a_scenes.setEnabled(True)
        counts: dict[str, int] = {}
        for n, (sid, prob) in zip(names, results):
            self.scene_of[n] = [sid, round(prob, 2)]
            self.update_item(n)
            counts[sid] = counts.get(sid, 0) + 1
        self.save_sidecar()
        top = ", ".join(f"{self.scene_by_id[k]['name'].lower()} {v}" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1])[:4])
        self.toast(f"Сцены определены ({top}). «Авто» (A) применит пресет сцены", 10000)

    def on_scenes_fail(self, msg):
        self.a_scenes.setEnabled(True)
        self.toast(f"Не удалось распознать сцены: {msg}", 10000)

    def reset_all(self):
        self.params = E.default_params()
        self.sync_controls()
        self.request_render()
        self.toast("Правки сброшены")

    def copy_settings(self):
        self.clipboard = copy.deepcopy(self.params)
        self.toast("Правки скопированы. Выделите кадры (Ctrl/Shift+щелчок) и нажмите Ctrl+V")

    def paste_settings(self):
        if not self.clipboard:
            self.toast("Сначала скопируйте правки (Ctrl+C)")
            return
        items = self.strip.selectedItems()
        for it in items:
            self.sidecar[Path(it.data(PATH_ROLE)).name] = copy.deepcopy(self.clipboard)
        if self.current and self.current.name in {Path(i.data(PATH_ROLE)).name for i in items}:
            self.params = self.sidecar[self.current.name]
            self.sync_controls()
            self.request_render()
        self.save_sidecar()
        self.toast(f"Правки вставлены в кадров: {len(items)}")

    # ---------- стили

    def reload_styles(self, select: str | None = None):
        self.styles, self.style_files = {}, {}
        for f in sorted(STYLES_DIR.glob("*.json")):
            st = E.read_json(f, None)
            if st and "mean" in st:
                name = st.get("name", f.stem)
                self.styles[name], self.style_files[name] = st, f
        for combo, empty in ((self.style_combo, "— без стиля —"), (self.style2_combo, "— смешать со стилем… —")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(empty, "")
            for name, f in self.style_files.items():
                thumb = f.with_suffix(".jpg")
                icon = QIcon(str(thumb)) if thumb.exists() else QIcon()
                combo.addItem(icon, name, name)
            combo.blockSignals(False)
        if select:
            self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(select)))
        self.style_menu_btn.setEnabled(bool(self.styles))

    def on_style(self, idx):
        self.params["style"] = self.style_combo.itemData(idx) or ""
        self.request_render()

    def on_style2(self, idx):
        self.params["style2"] = self.style2_combo.itemData(idx) or ""
        if self.params["style2"] and not self.params.get("style"):
            self.toast("Сначала выберите основной стиль, затем второй для смеси")
        self.request_render()

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in name).strip() or "стиль"

    def new_style(self):
        exts = " ".join(f"*{e}" for e in sorted(E.PHOTO_EXT))
        path, _ = QFileDialog.getOpenFileName(self, "Фото-референс", str(self.folder or Path.home()),
                                              f"Снимки ({exts})")
        if not path:
            return
        name, ok = QInputDialog.getText(self, "Новый стиль", "Название стиля:", text=Path(path).stem)
        name = name.strip()
        if not ok or not name:
            return
        if name in self.styles and QMessageBox.question(
                self, "Стиль уже есть", f"Стиль «{name}» уже есть. Заменить его?") != QMessageBox.Yes:
            return
        self.toast("Анализирую цвета референса…", 0)

        def done(res):
            style, thumb = res
            old = self.style_files.get(name)
            dst = old or STYLES_DIR / f"{self._safe(name)}.json"
            E.write_json(dst, style)
            E.save_jpeg(dst.with_suffix(".jpg"), thumb / 255.0, 90)
            self.reload_styles(select=name)
            self.params["style"] = name
            self.request_render()
            self.refresh_look_icons()
            self.toast(f"Стиль «{name}» сохранён и применён. Цвет и свет стиля регулируются ползунками ниже")

        run_task(style_job, path, name, done=done,
                 fail=lambda m: self.toast(f"Не удалось прочитать референс: {m}", 8000))

    def _replace_style_refs(self, old: str, new: str):
        """Меняет имя стиля в правках текущего кадра и всех кадров открытой папки."""
        for p in [self.params, *(v for k, v in self.sidecar.items() if k != SCENES_KEY)]:
            for key in ("style", "style2"):
                if p.get(key) == old:
                    p[key] = new

    def rename_style(self):
        old = self.style_combo.itemData(self.style_combo.currentIndex()) or ""
        if not old:
            self.toast("Выберите в списке стиль, который нужно переименовать")
            return
        new, ok = QInputDialog.getText(self, "Переименовать стиль", "Новое название:", text=old)
        new = new.strip()
        if ok and new and new != old:
            self._rename_style(old, new)

    def _rename_style(self, old: str, new: str) -> bool:
        src = self.style_files[old]
        dst = STYLES_DIR / f"{self._safe(new)}.json"
        if new in self.styles or (dst.exists() and dst != src):
            self.toast(f"Стиль «{new}» уже есть — выберите другое название")
            return False
        try:
            E.write_json(dst, {**self.styles[old], "name": new})
            if src.with_suffix(".jpg").exists() and dst != src:
                src.with_suffix(".jpg").replace(dst.with_suffix(".jpg"))
            if dst != src:
                src.unlink()
        except OSError as e:
            self.toast(f"Не удалось переименовать: {e}", 8000)
            return False
        self._replace_style_refs(old, new)
        self.save_sidecar()
        self.reload_styles()
        self.sync_controls()
        self.toast(f"Стиль «{old}» переименован в «{new}»")
        return True

    def delete_style(self):
        name = self.style_combo.itemData(self.style_combo.currentIndex()) or ""
        if not name:
            self.toast("Выберите в списке стиль, который нужно удалить")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Удалить стиль")
        box.setText(f"Удалить стиль «{name}»?\nКадры с этим стилем останутся без него. Фото-референс не удаляется.")
        yes = box.addButton("✅ Удалить", QMessageBox.AcceptRole)
        box.addButton("❌ Отмена", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self._delete_style(name)

    def _delete_style(self, name: str):
        f = self.style_files[name]
        try:
            f.unlink()
            f.with_suffix(".jpg").unlink(missing_ok=True)
        except OSError as e:
            self.toast(f"Не удалось удалить: {e}", 8000)
            return
        self._replace_style_refs(name, "")
        self.save_sidecar()
        self.reload_styles()
        self.sync_controls()
        self.request_render()
        self.toast(f"Стиль «{name}» удалён")

    def export_lut(self):
        if self.base is None:
            return
        name = self.params.get("style") or "мой_цвет"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить LUT", str(APP_DIR / f"{name}.cube"),
                                              "3D LUT (*.cube)")
        if not path:
            return
        style_only = False
        if self.current_style():
            box = QMessageBox(self)
            box.setWindowTitle("Что сохранить в LUT")
            box.setText("Сохранить в LUT всё (ползунки, пресет и стиль) или только стиль референса?")
            all_btn = box.addButton("Всё вместе", QMessageBox.AcceptRole)
            box.addButton("Только стиль", QMessageBox.AcceptRole)
            box.exec()
            style_only = box.clickedButton() is not all_btn
        if style_only:  # только перенос стиля: ползунки и пресет нейтральны
            keys = ("style", "style2", "style_mix", "style_mode", "style_strength", "style_tone", "style_skin")
            params = {**E.default_params(), **{k: self.params[k] for k in keys}}
            look = None
        else:
            params, look = self.params, self.current_look()
        src = E.lab_stats(E.process(self.base, {**params, "style": ""}, None, local=False))
        E.export_cube(path, params, self.current_style(), src, look=look)
        what = "только стиль" if style_only else "света/тени, чёткость и резкость в LUT не входят"
        self.toast(f"LUT сохранён: {path} ({what})", 8000)

    # ---------- пресеты-образы

    def reload_looks(self):
        self.looks = E.load_looks(LOOKS_DIR)
        c = self.look_combo
        c.blockSignals(True)
        c.clear()
        c.addItem("— без пресета —", "")
        group = None
        for name, look in self.looks.items():
            if look.get("group") != group:
                group = look.get("group")
                c.addItem(f"— {group or 'Другие'} —")
                c.model().item(c.count() - 1).setEnabled(False)
            c.addItem(name, name)
            c.setItemData(c.count() - 1, look.get("hint", ""), Qt.ToolTipRole)
        c.blockSignals(False)

    def on_look(self, idx):
        name = self.look_combo.itemData(idx) or ""
        self.params["look"] = name
        self.look_combo.setToolTip(self.looks.get(name, {}).get("hint", "Готовые мягкие образы; сила — ползунок ниже"))
        self.request_render()
        if name:
            self.toast(f"Пресет «{name}». Силу меняет ползунок «Сила пресета»")

    def refresh_look_icons(self):
        if self.base is None or not self.looks:
            return
        run_task(look_icons_job, self.current, self.base, dict(self.params), self.current_style(), self.looks,
                 done=self.on_look_icons)

    def on_look_icons(self, res):
        path, icons = res
        if path != self.current:
            return
        for name, u8 in icons.items():
            i = self.look_combo.findData(name)
            if i >= 0:
                self.look_combo.setItemIcon(i, QIcon(QPixmap.fromImage(to_qimage(u8))))

    # ---------- пресеты

    def reload_presets(self):
        self.preset_combo.clear()
        self.preset_combo.addItem("— выбрать пресет —")
        self.preset_combo.addItems([f.stem for f in sorted(PRESETS_DIR.glob("*.json"))])

    def apply_preset(self, idx):
        if idx <= 0 or self.base is None:
            return
        name = self.preset_combo.currentText()
        self.params = E.normalize_params(E.read_json(PRESETS_DIR / f"{name}.json", {}))
        self.sync_controls()
        self.request_render()
        self.preset_combo.setCurrentIndex(0)
        self.toast(f"Пресет «{name}» применён")

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:")
        name = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in name.strip())
        if ok and name:
            E.write_json(PRESETS_DIR / f"{name}.json", self.params)
            self.reload_presets()
            self.toast(f"Пресет «{name}» сохранён")

    # ---------- экспорт

    def export(self):
        if self.export_thread or not self.files:
            return
        self.store_current()
        selected = [Path(i.data(PATH_ROLE)) for i in self.strip.selectedItems()]
        dlg = ExportDialog(self, len(selected), len(self.files), self.folder, bool(self.scene_of))
        if not dlg.exec():
            return
        if dlg.r_all.isChecked():
            paths = list(self.files)
        elif dlg.r_selected.isChecked():
            paths = selected
        else:
            paths = [self.current]
        if dlg.skip_blur.isChecked():
            paths = [p for p in paths if p.name not in self.blurry]
        if not paths:
            self.toast("Нечего экспортировать: все выбранные кадры отмечены как размытые")
            return
        opts = {"out": Path(dlg.out.text()), "scene": dlg.scene_look.isChecked(), "auto": dlg.auto.isChecked(),
                "edge": dlg.edge.value(), "quality": dlg.quality.value()}
        self.save_sidecar()
        missing = []
        for p in paths:
            layers = E.normalize_params(self.sidecar.get(p.name, self.params)).get("masks", [])
            cats = sorted({m["cat"] for m in layers if m["type"] == "ai" and self.ai_arr(p.name, m["cat"]) is None})
            if cats:
                missing.append((str(p), cats))
        if missing and G.available():
            self.toast(f"Готовлю маски ИИ для кадров: {len(missing)}…", 0)
            self.a_export.setEnabled(False)

            def ready(seg):
                self.segmenter = seg
                self._run_export(paths, opts)

            def failed(msg):
                self.a_export.setEnabled(True)
                self.toast(f"Маски ИИ не подготовлены ({msg}). Экспорт без них", 8000)
                self._run_export(paths, opts)

            run_task(prepare_ai_job, self.segmenter, missing, str(self.folder / MASKS_DIR), done=ready, fail=failed)
            return
        self._run_export(paths, opts)

    def _run_export(self, paths: list[Path], opts: dict):
        out_dir = opts["out"]
        jobs = []
        for p in paths:
            params = E.normalize_params(self.sidecar.get(p.name, self.params))
            sc = self.scene_for(p.name) if opts["scene"] else None
            look = self.looks.get((sc or params).get("look") or "")
            jobs.append({"scene": sc, "src": str(p), "dst": str(out_dir / f"{p.stem}.jpg"),
                         "params": self.render_params(params, p.name),
                         "style": self.style_for(params),
                         "look": look,
                         "auto": opts["auto"], "long_edge": opts["edge"],
                         "quality": opts["quality"]})
        self.export_dir = out_dir
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.progress.setFormat("Экспорт %v из %m")
        self.progress.show()
        self.cancel_btn.show()
        self.a_export.setEnabled(False)
        self.export_thread = ExportThread(jobs)
        self.export_thread.progress.connect(lambda d, n: self.progress.setValue(d))
        self.export_thread.finished_all.connect(self.on_export_done)
        self.export_thread.start()

    def cancel_export(self):
        if self.export_thread:
            self.export_thread.stop = True
            self.cancel_btn.setEnabled(False)
            self.toast("Останавливаю после текущих кадров…", 0)

    def on_export_done(self, ok, errors):
        self.export_thread.wait()
        self.export_thread = None
        self.progress.hide()
        self.cancel_btn.hide()
        self.cancel_btn.setEnabled(True)
        self.a_export.setEnabled(True)
        if errors:
            QMessageBox.warning(self, "Экспорт завершён с ошибками",
                                f"Готово: {ok}. Ошибки:\n" + "\n".join(errors[:15]))
        box = QMessageBox(self)
        box.setWindowTitle("Экспорт готов")
        box.setText(f"Сохранено снимков: {ok}\n{self.export_dir}")
        open_btn = box.addButton("Открыть папку", QMessageBox.AcceptRole)
        box.addButton("Закрыть", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_dir)))

    def closeEvent(self, e):
        self.save_sidecar()
        if self.export_thread:
            self.export_thread.stop = True
            self.export_thread.wait()
        super().closeEvent(e)


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in [(QPalette.Window, "#232323"), (QPalette.WindowText, "#e6e6e6"),
                        (QPalette.Base, "#1c1c1c"), (QPalette.AlternateBase, "#262626"),
                        (QPalette.Text, "#e6e6e6"), (QPalette.Button, "#2e2e2e"),
                        (QPalette.ButtonText, "#e6e6e6"), (QPalette.Highlight, "#3d7eff"),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#2e2e2e"),
                        (QPalette.ToolTipText, "#e6e6e6")]:
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#6a6a6a"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6a6a6a"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#6a6a6a"))
    app.setPalette(pal)
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 14px; padding: 8px 8px 4px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #bdbdbd; }
        QToolBar { spacing: 4px; padding: 4px; border: none; }
        QToolButton { padding: 5px 10px; border-radius: 5px; }
        QToolButton:hover { background: #353535; }
        QToolButton:checked { background: #3d7eff; }
        QListWidget { border: none; }
        QListWidget::item { padding: 4px; }
        QPushButton { padding: 5px 10px; }
    """)
