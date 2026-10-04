"""Окно Mini LightRoom (PySide6)."""
from __future__ import annotations

import copy
import json
import os
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QFile,
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
    QPen,
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
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import engine as E
from . import enhance as N
from . import faces as FC
from . import grading as GR
from . import masks as MK
from . import quality as Q
from . import scene as S
from . import segment as G
from .crop_editor import CropEditor
from .crop_overlays import OVERLAYS, ROTATABLE
from .curve_editor import CURVE_SHAPES, CurveEditor
from .mask_editor import MaskEditor
from .quality_ui import (
    LEVEL_COLOR,
    SHORT,
    TYPE_TEXT,
    FilterBar,
    QualityBar,
    TrashDialog,
    ZoomTile,
    badge_icon,
    describe,
    numpy_pixmap,
)
from .target_editor import TargetEditor
from .theme import T, apply_theme, tool_icon
from .widgets import Section, Toast

APP_DIR = Path(__file__).resolve().parent.parent
STYLES_DIR = APP_DIR / "styles"
LOOKS_DIR = APP_DIR / "looks"
SCENES_FILE = APP_DIR / "scenes.json"
MODELS_DIR = APP_DIR / "models"
SCENES_KEY = "__scenes__"   # в sidecar: {имя файла: [id сцены, уверенность]}
QUALITY_KEY = "__quality__"  # в sidecar: {имя файла: результат quality.analyze_file}
FLAGS_KEY = "__flags__"      # в sidecar: {имя файла: "reject" | "pick"} — решение пользователя
META_KEYS = (SCENES_KEY, QUALITY_KEY, FLAGS_KEY)  # служебные ключи sidecar: это не правки кадров
PRESETS_DIR = APP_DIR / "presets"
SETTINGS_FILE = APP_DIR / "settings.json"  # последняя папка и кадр, окно, галочки — между запусками
SIDECAR = ".mini_lightroom.json"   # настройки кадров лежат рядом со снимками
HISTORY_FILE = ".mini_lightroom_history.json"  # шаги отмены кадров — тоже рядом, чтобы Ctrl+Z пережил перезапуск
HISTORY_KEEP = 50  # шагов на кадр в файле (в памяти — 100)
MASKS_DIR = ".mini_lightroom_masks"  # ИИ-маски кадров (PNG) рядом со снимками
PREVIEW_SIDE = 1400
PATH_ROLE = Qt.UserRole
# Цвета HSL: ключ движка, подпись, цвет метки.
HSL_COLORS = [("red", "Красный", "#ff4d4d"), ("orange", "Оранжевый", "#ff9a3d"), ("yellow", "Жёлтый", "#f2d33b"),
              ("green", "Зелёный", "#4fd463"), ("aqua", "Голубой", "#3dd6d0"), ("blue", "Синий", "#4d7dff"),
              ("purple", "Фиолетовый", "#9b5cff"), ("magenta", "Пурпурный", "#ff4fd0")]
# Пропорции обрезки: подпись → ширина/высота; "cam" — как снимала камера, "orig" — как у RAW.
ASPECTS = [("Свободно", None), ("Как в камере", "cam"), ("Исходное", "orig"), ("1:1", 1.0), ("4:5", 0.8),
           ("3:2", 1.5), ("4:3", 4 / 3), ("16:9", 16 / 9)]
SECTIONS_OPEN = ("ИИ-цветокоррекция", "Автоматика", "Свет", "Цвет")  # остальные группы панели свёрнуты, пока их не раскрыли
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

# Всё, что трогает torch, — строго по очереди в ОДНОМ постоянном потоке Python. В потоках QThreadPool
# torch нельзя: PySide после каждой задачи уничтожает состояние Python-потока, а torch (pybind11)
# держит на него указатель — второй вызов torch в том же потоке Qt вешал программу или ронял её
# (0xc0000374). Так и получалось «нажал две-три ИИ-функции — всё встало».
AI_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai")


def run_task(fn, *args, done=None, fail=None, pool: QThreadPool | Executor | None = None):
    t = _Task(fn, args)
    _alive.add(t)
    if done:
        t.sig.done.connect(done)
    if fail:
        t.sig.fail.connect(fail)
    t.sig.done.connect(lambda *_: _alive.discard(t))
    t.sig.fail.connect(lambda *_: _alive.discard(t))
    if isinstance(pool, Executor):
        pool.submit(t.run)  # сигналы из потока Python приходят в окно очередью, как из QThreadPool
    else:
        (pool or QThreadPool.globalInstance()).start(t)


def aspect_label(a: float) -> str:
    """1.777 → «16:9»: подпись пропорции для людей."""
    for w, h in ((16, 9), (3, 2), (4, 3), (1, 1), (5, 4), (4, 5), (9, 16), (2, 3), (3, 4)):
        if abs(a - w / h) < 0.01:
            return f"{w}:{h}"
    return f"{a:.2f}:1"


def to_qimage(rgb_u8: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgb_u8)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def thumb_job(path):
    return path, E.resize_max(E.load_thumb(path, 512), 200)


def preview_job(path):
    return (path, E.load_image(path, half=True, max_side=PREVIEW_SIDE), E.full_size(path), N.iso_of(path),
            E.camera_aspect(path))


def denoise_job(path, base):
    """Превью без шума на полной силе; ползунок потом только смешивает его с исходником."""
    return path, N.denoise(base, 1.0)


def _denoise_box(crop, box, strength):
    """Шумодав только видимой части вырезки (+16 px): поля под размытия шум не портит, а время экономится."""
    x0, y0, x1, y1 = box
    h, w = crop.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - 16), max(0, y0 - 16), min(w, x1 + 16), min(h, y1 + 16)
    out = crop.copy()
    out[Y0:Y1, X0:X1] = N.denoise(crop[Y0:Y1, X0:X1], strength)
    return out


def look_icons_job(path, base, params, style, looks):
    """Миниатюры текущего кадра под каждым пресетом — видно, что выбираешь."""
    small = E.process(E.resize_max(base, 84), {**params, "look": "", "vignette": 0, "sharpness": 0}, style)
    return path, {name: to_u8(E.apply_look(small, look)) for name, look in looks.items()}


def scene_job(scenes, names, thumbs):
    return names, S.get_classifier(scenes, MODELS_DIR).classify(thumbs)


def faces_job(path, models_dir):
    """Лица и глаза кадра (MediaPipe) — только из AI_POOL. None — нет mediapipe или модели."""
    finder = FC.get_finder(models_dir)
    return path, (finder.detect(E.load_thumb(path, 2400)) if finder else None)


def quality_job(path, faces):
    """Брак кадра: резкость и смаз по области лица/точки АФ/лучшим местам + закрытые глаза."""
    rois = Q.face_rois(faces or [])
    res = Q.analyze_file(path, rois)
    return path, (Q.add_eyes(res, rois) if rois else res)


def tiles_job(path, boxes):
    """Крупные (100%) вырезки мест брака из полноразмерного кадра."""
    return path, Q.crop_tiles(Q.load_rgb(path)[0], boxes)


def style_job(path, name):
    img = E.load_image(path, half=True, max_side=1024)
    return {"name": name, **E.lab_stats(img)}, to_u8(E.resize_max(img, 96))


def _seg_input(base):
    """Кадр для сегментации: с авто-тоном, чтобы тёмный RAW не путал модель."""
    return E.process(base, E.normalize_params(E.auto_params(base)))


def _ai_masks(base, cats) -> dict[str, np.ndarray]:
    """Маски ИИ кадра (uint8). «subject» — главный объект: крупный человек (без толпы за спиной),
    а если такого нет — самое заметное пятно."""
    img = _seg_input(base)
    want = [c for c in cats if c != "subject"]
    if "subject" in cats and "people" not in want:
        want.append("people")
    maps = G.get_segmenter(MODELS_DIR).masks(img, want) if want else {}
    if "subject" in cats:
        main = E.main_people(maps["people"])
        if main is None:
            main = G.refine(E.saliency_mask(img), img)
            if main.mean() < 0.02:  # заметное пятно меньше 2% кадра — это не объект съёмки (пейзаж, город)
                main = np.zeros_like(main)
        maps["subject"] = main
    return {c: (m * 255 + 0.5).astype(np.uint8) for c, m in maps.items()}


def segment_job(path, base, cats, create):
    return path, _ai_masks(base, cats), create


def prepare_ai_job(items, out_dir):
    """ИИ-маски для кадров, которые ещё не открывали (перед экспортом)."""
    for path, cats in items:
        base = E.load_image(path, half=True, max_side=PREVIEW_SIDE)
        for cat, m in _ai_masks(base, cats).items():
            _write_png(Path(out_dir) / f"{Path(path).name}.{cat}.png", m)


def _write_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", arr)[1].tobytes())  # через Python: кириллица в пути


def _grade_analysis(scene_defs, base, params, maps, scene, thumb):
    """Маски объектов → сцена → анализ цвета → варианты по правилам. Общая часть карусели и «ИИ-улучшить»."""
    new = {}
    if G.available():
        missing = [c for c in GR.MASK_CATS if maps.get(c) is None]
        if missing:
            new = _ai_masks(base, missing)
    allm = {**{c: m for c, m in maps.items() if m is not None}, **new}
    if scene is None and thumb is not None and S.available():
        scene = S.get_classifier(scene_defs, MODELS_DIR).classify([thumb])[0][0]
    small = E.resize_max(base, 512)
    pre = E.process(small, {**params, "ai_grade": None, "style": "", "look": "", "masks": []})
    seg = {c: cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
           for c, m in allm.items()} or None
    an = GR.analyze(pre, seg, scene)
    return new, allm, scene, an, GR.variants(an)


def grade_job(scene_defs, path, base, params, style, look, geo, maps, scene, thumb):
    """ИИ-цветокоррекция: анализ → варианты → миниатюры карусели → оценка CLIP (★ самому удачному)."""
    new, allm, scene, an, vs = _grade_analysis(scene_defs, base, params, maps, scene, thumb)
    tiny = E.resize_max(base, 260)
    crop, angle, full = geo
    strength = (params.get("ai_grade") or {}).get("strength", 80)

    def render(grade):
        p = {**params, "ai_grade": None}
        if grade:
            p["ai_grade"] = {**grade, "strength": strength,
                             "masks": [{**m, "arr": allm.get(m["cat"])} for m in grade["masks"]]}
        return to_u8(E.apply_crop(E.process(tiny, p, style, look=look), crop, angle, full))

    thumbs = [render(None)] + [render(v) for v in vs]
    scores = S.get_classifier(scene_defs, MODELS_DIR).aesthetic(thumbs) if S.available() else None
    return path, new, scene, an, vs, thumbs, scores


def enhance_job(scene_defs, path, base, params, auto, maps, scene, thumb):
    """«ИИ-улучшить»: вариант «Естественные цвета» после авто-тона + угол горизонта."""
    new, allm, scene, _an, vs = _grade_analysis(scene_defs, base, {**params, **auto}, maps, scene, thumb)
    sky = allm.get("sky")
    angle = E.auto_horizon(_seg_input(base), None if sky is None else sky.astype(np.float32) / 255)
    return path, new, scene, auto, next((v for v in vs if v["id"] == "natural"), None), angle


def match_job(ref_params, ref_img, style, look, targets):
    """Единый цвет серии: эталон — готовый кадр; каждой цели — правки, чтобы выглядела так же."""
    ref_stats = E.match_stats(E.process(E.resize_max(ref_img, 400), ref_params, style, local=False, look=look))
    ref_auto = E.auto_params(ref_img)
    out = {}
    for path, own in targets:
        img = E.load_image(path, half=True, max_side=800)
        out[Path(path).name] = E.match_params(img, ref_params, ref_stats, own, style, look, ref_auto)
    return out


def horizon_job(path, base, sky):
    return path, E.auto_horizon(_seg_input(base), sky)


def saliency_job(path, base):
    return path, E.saliency_point(_seg_input(base))


def full_job(path):
    return path, E.load_image(path, half=False)


def detail_job(gen, full, base, rect, scale, params, style, look):
    src_stats = E.style_source_stats(base, params) if style else None  # стиль — как у всего кадра
    s = params.get("denoise", 0) / 100
    prep = (lambda c, box: _denoise_box(c, box, s)) if s > 0 and N.available() else None
    before, after = E.process_view_region(full, rect, scale, params, style, src_stats, look, prep)
    return gen, rect, to_u8(before), to_u8(after)


def render_job(gen, base, params, style, look, dn=None, geo=None):
    if dn is not None:  # превью без шума готово — смешиваем по силе шумодава
        base = base + (dn - base) * (params.get("denoise", 0) / 100)
    out = E.process(base, params, style, look=look)
    if geo:  # обрезка и горизонт — последним шагом; гистограмма уже по обрезанному кадру
        crop, angle, full = geo
        out = E.apply_crop(out, crop, angle, full)
    out = to_u8(out)
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
        self.editor = None  # MaskEditor / CropEditor: забирает левую кнопку
        self.marks: list[tuple] = []  # рамки брака: (рамка в долях кадра, уровень, подпись)
        # Обрезка: A — полный кадр (пиксели) → показанный кадр; src_w×src_h — размер показанного кадра.
        self.A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        self.Ainv = self.A.copy()
        self.full_w, self.full_h = 1, 1
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_image(self, img: QImage | None):
        self.pix = QPixmap.fromImage(img) if img is not None else None
        self.update()

    def set_detail(self, detail: tuple[QPixmap, QRectF] | None):
        self.detail = detail
        self.update()

    def set_source_size(self, w: int, h: int):
        self.src_w = self.src_h = 0  # размер «изменился» — центр встанет посередине
        self.set_geometry(np.array([[1.0, 0, 0], [0, 1.0, 0]]), (w, h), (w, h))

    def set_geometry(self, A, full: tuple, out: tuple):
        """Показанный кадр = полный кадр после поворота и обрезки (матрица A)."""
        self.A = np.asarray(A, np.float64)
        self.Ainv = cv2.invertAffineTransform(self.A)
        self.full_w, self.full_h = max(1, full[0]), max(1, full[1])
        ow, oh = max(1.0, float(out[0])), max(1.0, float(out[1]))
        if abs(ow - self.src_w) > 0.5 or abs(oh - self.src_h) > 0.5:
            self.src_w, self.src_h = ow, oh
            self.cx, self.cy = ow / 2, oh / 2
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
        """Точка виджета → доли полного кадра (с учётом обрезки и поворота)."""
        s = self.scale()
        ox, oy = self.cx + (pos.x() - self.width() / 2) / s, self.cy + (pos.y() - self.height() / 2) / s
        fx, fy = self.Ainv @ (ox, oy, 1.0)
        return fx / self.full_w, fy / self.full_h

    def to_widget_pt(self, xn: float, yn: float) -> QPointF:
        s = self.scale()
        ox, oy = self.A @ (xn * self.full_w, yn * self.full_h, 1.0)
        return QPointF(self.width() / 2 + (ox - self.cx) * s, self.height() / 2 + (oy - self.cy) * s)

    def visible_rect(self) -> QRectF:
        """Видимая часть кадра в его пикселях."""
        s = self.scale()
        r = QRectF(self.cx - self.width() / 2 / s, self.cy - self.height() / 2 / s,
                   self.width() / s, self.height() / s)
        return r.intersected(QRectF(0, 0, self.src_w, self.src_h))

    def focus_on(self, xn: float, yn: float, zoom: float = 1.0):
        """Показать точку кадра (в долях полного кадра) в центре на заданном масштабе (1.0 — 100%)."""
        self.zoom = zoom if zoom > self.fit_scale() * 1.001 else None
        self.cx, self.cy = (float(v) for v in self.A @ (xn * self.full_w, yn * self.full_h, 1.0))
        self._clamp()
        self.update()
        self.view_changed.emit()

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

    def _paint_marks(self, p: QPainter):
        """Рамки мест брака: цвет — серьёзность, подпись над рамкой."""
        p.setRenderHint(QPainter.Antialiasing)
        for box, level, label in self.marks:
            r = QRectF(self.to_widget_pt(box[0], box[1]), self.to_widget_pt(box[2], box[3])).normalized()
            col = QColor(LEVEL_COLOR.get(level, "#e2685f"))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 2))
            p.drawRoundedRect(r, 6, 6)
            tag = p.fontMetrics().boundingRect(label).adjusted(-7, -3, 7, 3)
            tag.moveBottomLeft(r.topLeft().toPoint())
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(tag, 6, 6)
            p.setPen(QColor("#101010"))
            p.drawText(tag, Qt.AlignCenter, label)

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
            self._paint_marks(p)
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
        elif self.key == "angle_x10":
            self.value.setText(f"{v / 10:+.1f}°" if v else "0°")
        elif self.key in ("sharpness", "brush_size", "mask_feather") or self.key.startswith(("style_", "look_",
                                                                                              "grade_")):
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
        self.skip_blur = QCheckBox("Пропускать брак (расфокус, смаз, закрытые глаза)")
        self.edge = QSpinBox()
        self.edge.setRange(0, 12000)
        self.edge.setSingleStep(500)
        self.edge.setSpecialValueText("как в оригинале")
        self.edge.setSuffix(" px")
        self.quality = QSpinBox()
        self.quality.setRange(60, 100)
        self.quality.setValue(92)
        self.upscale = QComboBox()
        self.upscale.addItems(["нет", "×2", "×4"])
        self.upscale.setEnabled(N.available())
        self.upscale.setToolTip("ИИ-увеличение (Real-ESRGAN) после уменьшения по длинной стороне.\n"
                                "Для небольших кадров и кропов: 24 Мп ×2 — это 96 Мп и несколько минут."
                                if N.available() else "Нужны библиотеки ИИ: install_ai.bat")
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
        form.addRow("Увеличение (ИИ):", self.upscale)
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
        if any(j.get("upscale") or j["params"].get("denoise") for j in self.jobs):
            workers = 1  # видеокарта одна: второй процесс только делил бы её и память
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
        self.quality: dict[str, dict] = {}   # результаты проверки брака (quality.py); в sidecar под QUALITY_KEY
        self.flags: dict[str, str] = {}      # решение пользователя: "reject" | "pick"
        self.thumb_pix: dict[str, QPixmap] = {}
        self.q_gen = 0                       # поколение проверки: результаты прежней папки отбрасываются
        self.q_left = self.q_total = 0       # сколько кадров ещё проверяется / всего в этом запуске
        self.tile_cache: dict[tuple, list] = {}
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
        self.isos: dict[str, int | None] = {}
        self.base_dn: np.ndarray | None = None   # превью текущего кадра без шума (полная сила)
        self.dn_busy = False
        self.ai_cache: dict[tuple[str, str], np.ndarray] = {}
        self.seg_pending: set[tuple[str, str]] = set()  # (кадр, маска), которые уже считаются
        self.full_wh: tuple[int, int] = (1, 1)
        self.cam_aspects: dict[str, float | None] = {}
        self.crop_mode = False
        self.crop_backup: dict | None = None
        self._after_seg = None  # что сделать, когда досчитаются маски (кадр по третям)
        self.grade_variants: list[dict] = []
        self.grade_busy = False
        self.enhance_busy = False
        self.grade_cache: dict[str, tuple] = {}  # {кадр: (ключ правок, варианты, миниатюры, оценки)}
        self.target_mode: str | None = None
        self._tat: dict | None = None
        self.targeter = TargetEditor(self.tat_start, self.tat_drag)
        self.cropper = CropEditor()
        self.cropper.changed.connect(self.on_crop_rect)
        self.history: dict[str, dict] = {}   # {кадр: {"undo": [снимки правок], "redo": [...]}}
        self.history_dirty = False  # история изменилась с последней записи в файл
        self.history_timer = QTimer(self, singleShot=True, interval=400, timeout=self.commit_history)
        # Автосохранение правок: через 1.5 с после шага истории файл правок уже на диске.
        self.save_timer = QTimer(self, singleShot=True, interval=1500, timeout=self.save_sidecar)
        self.settings: dict = E.read_json(SETTINGS_FILE, {})
        self.sections: dict[str, Section] = {}
        self.editor = MaskEditor(self.ai_arr_for_layer,
                                 (lambda: self.base.shape[:2], lambda m: E.apply_crop(m, *self.display_geo())))
        self.editor.changed.connect(self.on_mask_geom)
        self.editor.created.connect(self.on_mask_created)
        self.thumb_pool = QThreadPool()
        self.thumb_pool.setMaxThreadCount(2)
        self.quality_pool = QThreadPool()  # резкость и смаз: numpy/OpenCV, без torch и mediapipe
        self.quality_pool.setMaxThreadCount(2)
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

    def _section(self, title: str) -> Section:
        """Группа панели правок; раскрытое/свёрнутое состояние помним между запусками."""
        saved = self.settings.get("sections", {})
        box = Section(title, saved.get(title, title in SECTIONS_OPEN))
        box.toggled.connect(self._on_section)
        self.sections[title] = box
        return box

    def _on_section(self, title: str, on: bool):
        self.remember(sections={**self.settings.get("sections", {}), title: on})

    def _action(self, text, slot, shortcut=None, tip="", icon=None):
        a = QAction(text, self)
        if icon:
            a.setIcon(tool_icon(icon))
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip(f"{tip or text} ({shortcut})" if shortcut else (tip or text))
        self.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        tb.addAction(self._action("Открыть папку", self.open_folder, "Ctrl+O", icon="folder-open"))
        tb.addSeparator()
        self.a_undo = self._action("", self.undo, "Ctrl+Z", "Отменить", icon="arrow-back-up")
        self.a_redo = self._action("", self.redo, "Ctrl+Y", "Вернуть", icon="arrow-forward-up")
        self.a_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        tb.addAction(self.a_undo)
        tb.addAction(self.a_redo)
        self.a_grade = self._action("ИИ-цвет", self.start_grade, "G",
                                    "Варианты цветокоррекции по законам фотографии: ИИ видит, что на кадре",
                                    icon="palette")
        tb.addAction(self.a_grade)
        self.a_enhance = self._action("ИИ-улучшить", self.ai_enhance, "Shift+A",
                                      "Одной кнопкой: тон и баланс белого, естественные цвета, ровный горизонт",
                                      icon="sparkles")
        tb.addAction(self.a_enhance)
        self.a_crop = self._action("Обрезка", self.toggle_crop, "C", "Обрезка и горизонт: рамка на кадре", icon="crop")
        self.a_crop.setCheckable(True)
        tb.addAction(self.a_crop)
        # Enter/Esc включены только в режиме обрезки: иначе они перехватывали бы Enter в полях ввода.
        self.a_crop_done = self._action("Готово", lambda: self.a_crop.trigger(), "Return")
        self.a_crop_cancel = self._action("Отменить обрезку", self.cancel_crop, "Esc")
        self.a_crop_done.setEnabled(False)
        self.a_crop_cancel.setEnabled(False)
        tb.addSeparator()
        self.a_auto = self._action("Авто", self.apply_auto, "A", "Подобрать тон и баланс белого под кадр", icon="wand")
        self.a_reset = self._action("", self.reset_all, "Ctrl+R", "Сбросить все правки кадра", icon="refresh")
        self.a_compare = self._action("", self.toggle_compare, "\\", "До / после: показать исходник", icon="contrast-2")
        self.a_compare.setCheckable(True)
        self.a_copy = self._action("", self.copy_settings, "Ctrl+C", "Копировать правки", icon="copy")
        self.a_paste = self._action("", self.paste_settings, "Ctrl+V", "Вставить правки в выделенные кадры", icon="clipboard")
        self.a_match = self._action("Единый цвет", self.match_series, "Ctrl+Shift+M",
                                    "Выделенные кадры (или всю папку) подогнать под этот: тот же стиль и пресет,\n"
                                    "а экспозиция, баланс белого, контраст и насыщенность — под каждый кадр,\n"
                                    "чтобы серия выглядела одинаково", icon="adjustments-horizontal")
        for a in (self.a_auto, self.a_reset, self.a_compare, self.a_copy, self.a_paste, self.a_match):
            tb.addAction(a)
        self.auto_on_open = QCheckBox("Авто для новых кадров")  # живут в группе «Автоматика» панели, не в тулбаре
        self.auto_on_open.setChecked(True)
        self.auto_on_open.setToolTip("Кадр без правок при открытии сразу получает «Авто»")
        self.a_scenes = self._action("Сцены", self.detect_scenes, "Ctrl+Shift+S",
                                     "Определить сцену каждого кадра: закат, портрет, лес… (ИИ локально на видеокарте)",
                                     icon="scan")
        tb.addAction(self.a_scenes)
        self.scene_auto = QCheckBox("Пресет по сцене")
        self.scene_auto.setChecked(True)
        self.scene_auto.setToolTip("«Авто» и новые кадры получают пресет своей сцены.\n"
                                   "Работает для кадров, у которых сцена уже определена")
        self.a_quality = self._action("Фокус", lambda: self.start_quality(force=True), "Ctrl+Shift+F",
                                      "Проверить резкость, смаз и закрытые глаза у всех кадров папки заново",
                                      icon="focus-2")
        tb.addAction(self.a_quality)
        self.quality_auto = QCheckBox("Проверять фокус при открытии папки")
        self.quality_auto.setChecked(True)
        self.quality_auto.setToolTip("После загрузки миниатюр в фоне ищется брак: расфокус, смаз, закрытые глаза.\n"
                                     "Результат хранится рядом со снимками; кадры, которые не менялись, не пересчитываются")
        self.a_reject = self._action("Брак", lambda: self.toggle_flag("reject"), "X", "Пометить кадр браком / снять")
        self.a_pick = self._action("Оставить", lambda: self.toggle_flag("pick"), "U", "Оставить кадр (не бракуем) / снять")
        self.a_zoom_out = self._action("", lambda: self.view.step_zoom(-1), "Ctrl+-", "Уменьшить масштаб", icon="minus")
        self.a_zoom_in = self._action("", lambda: self.view.step_zoom(1), "Ctrl+=", "Увеличить масштаб", icon="plus")
        self.a_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        self.a_fit = self._action("Вписать", lambda: self.view.set_zoom(None), "Ctrl+0")
        self.a_100 = self._action("100%", lambda: self.view.set_zoom(1.0), "Ctrl+1")
        # O: в обрезке — вид сетки, вне её — подсветка маски (две одинаковые клавиши включены по очереди)
        self.a_mask_show = self._action("Показать маску", lambda: self.mask_show.toggle(), "O")
        self.a_overlay = self._action("Вид сетки", self.cycle_overlay, "O", "Сетка-подсказка: трети, золотое сечение, спираль…")
        self.a_overlay_rot = self._action("Повернуть сетку", self.rotate_overlay, "Shift+O")
        self.a_overlay.setEnabled(False)
        self.a_overlay_rot.setEnabled(False)
        self.cropper.overlay = self.settings.get("overlay", "thirds") if self.settings.get("overlay") in dict(OVERLAYS)             else "thirds"
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("Вписать")
        self.zoom_combo.addItems([f"{round(z * 100)}%" for z in ZOOM_STEPS])
        self.zoom_combo.setToolTip("Масштаб от полного разрешения снимка.\nКолёсико — приблизить к курсору, "
                                   "перетаскивание — сдвиг,\nдвойной щелчок — 100% / вписать")
        self.zoom_combo.activated.connect(
            lambda i: self.view.set_zoom(None if i == 0 else ZOOM_STEPS[i - 1]))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_export = self._action("Экспорт…", self.export, "Ctrl+E")
        self.a_export.setIcon(tool_icon("download", T["on_accent"]))
        tb.addAction(self.a_export)
        tb.widgetForAction(self.a_export).setObjectName("primary")  # единственная кнопка с заливкой акцентом

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

        box = self._section("ИИ-цветокоррекция")
        gl = QVBoxLayout(box.body)
        b = QPushButton("Подобрать цвет по правилам (G)")
        b.setIcon(tool_icon("palette"))
        b.setToolTip("ИИ определяет сцену и объекты (кожа, небо, зелень, вода) и строит варианты:\n"
                     "цвета памяти, гармонии цветового круга, тёплое/холодное, фигура и фон, 60-30-10, ч/б.\n"
                     "Варианты — в карусели под кадром, щелчок применяет")
        b.clicked.connect(self.start_grade)
        gl.addWidget(b)
        b = QPushButton("ИИ-улучшить (Shift+A)")
        b.setIcon(tool_icon("sparkles"))
        b.setToolTip("Авто-тон и баланс белого (с пресетом сцены, если включён), цвета памяти — кожа, небо,\n"
                     "зелень — к эталону, горизонт выровнен, если он уверенно найден. Одно Ctrl+Z отменяет всё")
        b.clicked.connect(self.ai_enhance)
        gl.addWidget(b)
        self.grade_rule = QLabel("Нажмите кнопку — под кадром появятся варианты")
        self.grade_rule.setWordWrap(True)
        self.grade_rule.setStyleSheet("color:#9a9a9a")
        gl.addWidget(self.grade_rule)
        self.grade_strength_row = SliderRow("grade_strength", "Сила", 0, 100)
        self.grade_strength_row.set_value(80)
        self.grade_strength_row.changed.connect(self.on_grade_strength)
        gl.addWidget(self.grade_strength_row)
        b = QPushButton("Убрать ИИ-цветокоррекцию")
        b.clicked.connect(lambda: self.carousel.setCurrentRow(0) if self.carousel.count() else self.clear_grade())
        gl.addWidget(b)
        pl.addWidget(box)

        box = self._section("Автоматика")
        al = QVBoxLayout(box.body)
        al.addWidget(self.auto_on_open)
        al.addWidget(self.scene_auto)
        al.addWidget(self.quality_auto)
        pl.addWidget(box)

        box = self._section("Обрезка и горизонт")
        cl = QVBoxLayout(box.body)
        row = QHBoxLayout()
        self.aspect_combo = QComboBox()
        for label, _ in ASPECTS:
            self.aspect_combo.addItem(label)
        self.aspect_combo.setToolTip("Пропорции рамки. «Как в камере» — формат, выбранный в камере (например 16:9)")
        self.aspect_combo.activated.connect(self.on_aspect)
        row.addWidget(self.aspect_combo, 1)
        b = QPushButton()
        b.setIcon(tool_icon("arrows-left-right"))
        b.setToolTip("Повернуть рамку: горизонтальная ↔ вертикальная")
        b.setFixedWidth(34)
        b.clicked.connect(self.flip_aspect)
        row.addWidget(b)
        cl.addLayout(row)
        self.overlay_combo = QComboBox()
        for _, label in OVERLAYS:
            self.overlay_combo.addItem(f"Сетка: {label}")
        self.overlay_combo.setCurrentIndex([k for k, _ in OVERLAYS].index(self.cropper.overlay))
        self.overlay_combo.setToolTip("Подсказка композиции на рамке обрезки (клавиша O листает, Shift+O поворачивает)")
        self.overlay_combo.activated.connect(lambda i: self.cropper.set_overlay(OVERLAYS[i][0]))
        self.cropper.overlay_changed.connect(self.on_overlay)
        cl.addWidget(self.overlay_combo)
        self.angle_row = SliderRow("angle_x10", "Горизонт", -450, 450)
        self.angle_row.setToolTip("Поворот для выравнивания горизонта. Рамка сама сжимается, чтобы не было пустых углов")
        self.angle_row.changed.connect(lambda _k, v: self.on_angle(v / 10))
        cl.addWidget(self.angle_row)
        row = QHBoxLayout()
        b = QPushButton("Выровнять горизонт")
        b.setIcon(tool_icon("wand"))
        b.setToolTip("ИИ ищет горизонт, границу неба и вертикали зданий и выравнивает кадр.\n"
                     "Если линии противоречат друг другу — ничего не поворачивает")
        b.clicked.connect(self.ai_horizon)
        row.addWidget(b)
        b = QPushButton("Кадр по третям")
        b.setIcon(tool_icon("crop"))
        b.setToolTip("Правило третей: главный объект (голова человека или самое заметное место)\n"
                     "встаёт на пересечение третей, горизонт — на треть по высоте")
        b.clicked.connect(self.ai_thirds)
        row.addWidget(b)
        cl.addLayout(row)
        row = QHBoxLayout()
        self.crop_cam = QCheckBox("Новые кадры — как в камере")
        self.crop_cam.setChecked(True)
        self.crop_cam.setToolTip("Если камера снимала в другом формате (например 16:9), новый кадр сразу обрезается так же")
        row.addWidget(self.crop_cam, 1)
        b = QPushButton("Сброс")
        b.setToolTip("Полный кадр без поворота")
        b.clicked.connect(self.reset_crop)
        row.addWidget(b)
        cl.addLayout(row)
        pl.addWidget(box)
        self.rows: dict[str, SliderRow] = {}
        groups: dict[str, QVBoxLayout] = {}
        for key, label, lo, hi, group in E.SLIDERS:
            if group not in groups:
                box = self._section(group)
                groups[group] = QVBoxLayout(box.body)
                pl.addWidget(box)
            row = SliderRow(key, label, lo, hi)
            row.changed.connect(self.on_param)
            groups[group].addWidget(row)
            self.rows[key] = row
        self.rows["denoise"].setToolTip(
            "ИИ-шумодав (SCUNet, на видеокарте). Лучше всего виден в масштабе 100%.\n"
            "Кадры с высоким ISO получают его сами при открытии." if N.available()
            else "Нужны библиотеки ИИ: запустите install_ai.bat")
        self.rows["denoise"].setEnabled(N.available())
        no_ai = "Нужны библиотеки ИИ: запустите install_ai.bat"
        self.rows["retouch"].setToolTip(
            "ИИ находит людей и разглаживает только кожу: пятна и неровности тона уходят,\n"
            "поры и текстура остаются. Глаза, губы и волосы не трогаются" if G.available() else no_ai)
        self.rows["bokeh"].setToolTip(
            "Размытие фона, как у светосильного объектива. Резким остаётся главный объект:\n"
            "человек, а если людей нет — самое заметное место кадра" if G.available() else no_ai)
        for k in E.AI_SLIDER_CATS:
            self.rows[k].setEnabled(G.available())

        # Тональная кривая
        box = self._section("Тональная кривая")
        cvl = QVBoxLayout(box.body)
        row = QHBoxLayout()
        self.curve_btns: dict[str, QPushButton] = {}
        for ch, label in (("rgb", "RGB"), ("r", "R"), ("g", "G"), ("b", "B")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setFixedWidth(48)
            b.setToolTip("Общая кривая" if ch == "rgb" else f"Кривая канала {label}: оттенок светов и теней")
            b.clicked.connect(lambda _=False, c=ch: self.curve.set_channel(c))
            row.addWidget(b)
            self.curve_btns[ch] = b
        self.curve_btns["rgb"].setChecked(True)
        self.curve_shape = QComboBox()
        self.curve_shape.addItem("Форма…")
        self.curve_shape.addItems(list(CURVE_SHAPES))
        self.curve_shape.setToolTip("Готовая форма для выбранного канала")
        self.curve_shape.activated.connect(self.on_curve_shape)
        row.addWidget(self.curve_shape, 1)
        self.tat_curve = QPushButton()
        self.tat_curve.setIcon(tool_icon("target"))
        self.tat_curve.setCheckable(True)
        self.tat_curve.setFixedWidth(34)
        self.tat_curve.setToolTip("Целевая правка: нажмите на кадре и тяните вверх/вниз —\n"
                                  "двигается точка кривой для тона под курсором")
        self.tat_curve.toggled.connect(lambda on: self.set_target_mode("curve" if on else None))
        row.addWidget(self.tat_curve)
        cvl.addLayout(row)
        self.curve = CurveEditor()
        self.curve.changed.connect(self.on_curve)
        cvl.addWidget(self.curve)
        b = QPushButton("Сброс кривой")
        b.setToolTip("Все каналы — прямая линия")
        b.clicked.connect(self.reset_curve)
        cvl.addWidget(b)
        pl.addWidget(box)

        # HSL: оттенок, насыщенность, яркость по 8 цветам
        box = self._section("HSL / Цвет")
        hl = QVBoxLayout(box.body)
        tabs = self.hsl_tabs = QTabWidget()
        self.hsl_rows: dict[tuple[str, int], SliderRow] = {}
        tips = ("Сдвиг оттенка цвета: например, зелень в сторону мяты или жёлтого",
                "Насыщенность только этого цвета", "Яркость только этого цвета: светлее кожа, темнее небо")
        for idx, tab_name in enumerate(("Оттенок", "Насыщенность", "Яркость")):
            w = QWidget()
            tl = QVBoxLayout(w)
            tl.setContentsMargins(4, 4, 4, 4)
            for color, label, hexc in HSL_COLORS:
                r = SliderRow(f"hsl_{color}_{idx}", f'<span style="color:{hexc}">●</span> {label}', -100, 100)
                r.setToolTip(tips[idx])
                r.changed.connect(lambda _k, v, c=color, i=idx: self.on_hsl(c, i, v))
                tl.addWidget(r)
                self.hsl_rows[(color, idx)] = r
            tabs.addTab(w, tab_name)
        hl.addWidget(tabs)
        row = QHBoxLayout()
        self.tat_hsl = QPushButton("Тянуть по цвету на кадре")
        self.tat_hsl.setIcon(tool_icon("target"))
        self.tat_hsl.setCheckable(True)
        self.tat_hsl.setToolTip("Целевая правка: нажмите на цвет на кадре и тяните вверх/вниз —\n"
                                "двигаются ползунки этого цвета на открытой вкладке")
        self.tat_hsl.toggled.connect(lambda on: self.set_target_mode("hsl" if on else None))
        row.addWidget(self.tat_hsl, 1)
        hl.addLayout(row)
        b = QPushButton("Сброс HSL")
        b.clicked.connect(self.reset_hsl)
        hl.addWidget(b)
        pl.addWidget(box)

        # Маски: локальные правки
        box = self._section("Маски")
        ml = QVBoxLayout(box.body)
        row = QHBoxLayout()
        for text, kind, tip in (
                ("Кисть", "brush", "Рисуйте по кадру, где нужна правка. Alt — стереть"),
                ("Линейный", "linear", "Протяните по кадру: сила от начала линии до нуля в конце"),
                ("Радиальный", "radial", "Протяните от центра: овал, внутри — правка")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=kind: self.add_mask(k))
            row.addWidget(b)
        ai_btn = QToolButton()
        ai_btn.setText("ИИ")
        ai_btn.setIcon(tool_icon("sparkles"))
        ai_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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
        box = self._section("Пресеты")
        ll = QVBoxLayout(box.body)
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
        box = self._section("Стиль с референса")
        sl = QVBoxLayout(box.body)
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
        b = QPushButton("Новый стиль из фото…")
        b.setIcon(tool_icon("plus"))
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
        box = self._section("Мои пресеты")
        prl = QHBoxLayout(box.body)
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
        scroll.setMinimumWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel = panel

        self.filter_bar = FilterBar()
        self.filter_bar.changed.connect(lambda _m: self.apply_filter())
        self.trash_btn = QPushButton("В корзину…")
        self.trash_btn.setIcon(tool_icon("trash"))
        self.trash_btn.setToolTip("Отправить брак в корзину Windows. Только после вашего подтверждения; из корзины можно вернуть")
        self.trash_btn.clicked.connect(self.trash_rejected)
        self.trash_btn.hide()
        left = QWidget()
        left.setMinimumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 8)
        ll.setSpacing(0)
        ll.addWidget(self.filter_bar)
        ll.addWidget(self.strip, 1)
        trash_row = QHBoxLayout()
        trash_row.setContentsMargins(8, 6, 8, 0)
        trash_row.addWidget(self.trash_btn)
        ll.addLayout(trash_row)
        split = QSplitter()
        split.addWidget(left)
        center = QWidget()
        cvl2 = QVBoxLayout(center)
        cvl2.setContentsMargins(0, 0, 0, 0)
        cvl2.setSpacing(4)
        cvl2.addWidget(self.view, 1)
        self.qbar = QualityBar()
        self.qbar.keep.connect(lambda: self.toggle_flag("pick"))
        self.qbar.reject.connect(lambda: self.toggle_flag("reject"))
        self.qbar.zoom_to.connect(self.zoom_to_roi)
        cvl2.addWidget(self.qbar)
        self.carousel = QListWidget()
        self.carousel.setViewMode(QListWidget.IconMode)
        self.carousel.setFlow(QListWidget.LeftToRight)
        self.carousel.setWrapping(False)
        self.carousel.setMovement(QListWidget.Static)
        self.carousel.setIconSize(QSize(168, 112))
        self.carousel.setGridSize(QSize(184, 150))
        self.carousel.setWordWrap(True)
        self.carousel.setFixedHeight(172)
        self.carousel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.carousel.setToolTip("Варианты ИИ-цветокоррекции: щелчок — применить, ← → — листать")
        self.carousel.currentRowChanged.connect(self.on_grade_pick)
        self.carousel.hide()
        cvl2.addWidget(self.carousel)
        split.addWidget(center)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 810, 390])
        self.setCentralWidget(split)
        self.toaster = Toast(self, self.view)

    def _build_status(self):
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.cancel_btn = QPushButton("Остановить")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self.cancel_export)
        self.iso_label = QLabel()
        self.iso_label.setStyleSheet("color:#9a9a9a; padding: 0 8px")
        self.statusBar().addPermanentWidget(self.iso_label)
        for w in (self.a_zoom_out, self.zoom_combo, self.a_zoom_in):  # масштаб — внизу справа, как в фоторедакторах
            if isinstance(w, QAction):
                btn = QToolButton()
                btn.setDefaultAction(w)
                w = btn
            self.statusBar().addPermanentWidget(w)
        self.zoom_combo.setFixedWidth(120)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

    def _set_enabled(self, on):
        self.panel.setEnabled(on)
        for a in (self.a_auto, self.a_reset, self.a_compare, self.a_copy, self.a_paste, self.a_export,
                  self.a_zoom_in, self.a_zoom_out, self.a_fit, self.a_100, self.a_scenes,
                  self.a_undo, self.a_redo, self.a_crop, self.a_grade, self.a_enhance, self.a_match,
                  self.a_quality, self.a_reject, self.a_pick):
            a.setEnabled(on)
        self.zoom_combo.setEnabled(on)

    def toast(self, text, ms=4000):
        self.toaster.show_text(text, ms)

    # ---------- сессия: последняя папка и кадр, окно, галочки

    def remember(self, **values):
        """Обновить settings.json (маленький файл — пишем сразу, чтобы пережить и аварийное закрытие)."""
        self.settings.update(values)
        try:
            E.write_json(SETTINGS_FILE, self.settings)
        except OSError:
            pass  # папка программы только для чтения — работаем без памяти сессии

    def restore_session(self) -> bool:
        """Открыть последнюю папку на последнем кадре; окно и галочки — как были. False — нечего открывать."""
        st = self.settings
        if st.get("geometry"):
            self.restoreGeometry(QByteArray.fromHex(st["geometry"].encode()))
        for box, key in ((self.auto_on_open, "auto_on_open"), (self.scene_auto, "scene_auto"),
                         (self.crop_cam, "crop_cam"), (self.quality_auto, "quality_auto")):
            if key in st:
                box.setChecked(bool(st[key]))
        folder = Path(st["last_folder"]) if st.get("last_folder") else None
        if folder is None or not folder.is_dir():
            if folder is not None:
                self.toast(f"Последняя папка не найдена: {folder}", 8000)
            return False
        last = st.get("last_file")
        self.load_folder(folder)
        if last in self.items:
            self.strip.setCurrentItem(self.items[last])
            self.strip.scrollToItem(self.items[last])
        self.toast(f"Продолжаем: {folder.name}" + (f", кадр {last}" if last in self.items else ""), 6000)
        return True

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
        self.history = {name: {k: [E.normalize_params(p) for p in h.get(k, [])] for k in ("undo", "redo")}
                        for name, h in E.read_json(folder / HISTORY_FILE, {}).items() if isinstance(h, dict)}
        self.history_dirty = False
        self.scene_of = dict(self.sidecar.get(SCENES_KEY, {}))
        self.thumbs.clear()
        self.q_gen += 1
        self.q_left = self.q_total = 0
        self.thumb_pix.clear()
        self.tile_cache.clear()
        self.quality = dict(self.sidecar.get(QUALITY_KEY, {}))
        self.flags = dict(self.sidecar.get(FLAGS_KEY, {}))
        self.qbar.hide()
        self.view.marks = []
        self.items.clear()
        self.cache.clear()
        self.grade_cache.clear()  # варианты карусели — по именам кадров этой папки
        self.current = None
        self.strip.clear()
        self.setWindowTitle(f"Mini LightRoom — {folder}")
        self.remember(last_folder=str(folder))
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
        self.toast(f"Снимков в папке: {len(self.files)}. Загружаю миниатюры…")
        self.strip.setCurrentRow(0)

    def on_thumb(self, res):
        path, th = res
        name = Path(path).name
        if name not in self.items:
            return
        self.thumbs[name] = th
        self.thumb_pix[name] = QPixmap.fromImage(to_qimage(th))
        self.update_item(name)
        if len(self.thumbs) == len(self.files):  # миниатюры готовы — теперь можно искать брак
            self.refresh_cull()
            if self.quality_auto.isChecked():
                self.start_quality()

    def effective(self, name: str) -> str:
        """Итог по кадру: решение пользователя важнее авто-вердикта. ok | doubt | bad | unknown."""
        flag = self.flags.get(name)
        if flag:
            return "bad" if flag == "reject" else "ok"
        return (self.quality.get(name) or {}).get("verdict", "unknown")

    def cull_candidates(self) -> list[str]:
        """Кадры, которые предлагаем в корзину: помеченные вами и найденные браком, кроме оставленных вами."""
        return [n for n in self.items if self.effective(n) == "bad"]

    def cull_rejects(self) -> set[str]:
        return set(self.cull_candidates())

    def update_item(self, name: str):
        """Подпись и значок кадра в ленте: имя, статус брака или решение, сцена."""
        it = self.items.get(name)
        if it is None:
            return
        level, title, _ = describe(self.quality.get(name), self.flags.get(name))
        lines = [name]
        if level in ("bad", "doubt") or self.flags.get(name):
            defects = (self.quality.get(name) or {}).get("defects") or []
            why = f" · {SHORT.get(defects[0]['type'], '')}" if defects and not self.flags.get(name) else ""
            lines.append({"Сомнительно": "Сомнит."}.get(title, title) + why)  # короче: лента узкая
        sc = self.scene_by_id.get(self.scene_of.get(name, [None])[0])
        if sc:
            lines.append(f"{sc.get('icon', '')} {sc['name']}")
        it.setText("\n".join(lines))
        pix = self.thumb_pix.get(name)
        if pix is not None:
            flag = self.flags.get(name)
            it.setIcon(badge_icon(pix, flag or (level if level in ("bad", "doubt") else None)))

    # ---------- проверка брака

    def start_quality(self, force: bool = False):
        """Брак всех кадров папки в фоне: лица (AI_POOL, mediapipe) → резкость и смаз (2 потока). Готовые
        результаты из sidecar берутся, если файл не менялся и метрика та же."""
        if not self.files or self.q_left:
            if self.q_left:
                self.toast("Проверка фокуса уже идёт", 3000)
            return
        todo = []
        for p in self.files:
            try:
                st = p.stat()
            except OSError:
                continue
            res = self.quality.get(p.name)
            if force or not res or res.get("file") != [st.st_size, int(st.st_mtime)] or res.get("v") != Q.CFG["version"]:
                todo.append(p)
        if not todo:
            self.refresh_cull()
            self.apply_filter()
            return
        todo.sort(key=lambda p: p != self.current)  # открытый кадр — первым
        self.q_gen += 1
        gen = self.q_gen
        self.q_left = self.q_total = len(todo)
        self.toast(f"Проверка фокуса: 0 из {len(todo)}…", 0)
        for p in todo:
            if FC.available():
                run_task(faces_job, p, MODELS_DIR, done=lambda r, g=gen: self._q_after_faces(r, g),
                         fail=lambda _m, g=gen, p=p: self._q_after_faces((p, None), g), pool=AI_POOL)
            else:
                self._q_after_faces((p, None), gen)

    def _q_after_faces(self, res, gen):
        if gen != self.q_gen:
            return
        path, faces = res
        run_task(quality_job, path, faces, done=lambda r, g=gen: self.on_quality(r, g),
                 fail=lambda m, g=gen, p=path: self.on_quality(
                     (p, {"verdict": "unknown", "defects": [], "regions": [], "error": m}), g),
                 pool=self.quality_pool)

    def on_quality(self, res, gen):
        if gen != self.q_gen:
            return
        path, r = res
        name = Path(path).name
        self.quality[name] = r
        self.q_left -= 1
        self.update_item(name)
        if self.current is not None and self.current.name == name:
            self.show_quality()
        done = self.q_total - self.q_left
        if self.q_left:
            if done % 4 == 0:
                self.toast(f"Проверка фокуса: {done} из {self.q_total}…", 0)
            return
        bad = sum(self.effective(n) == "bad" for n in self.items)
        doubt = sum(self.effective(n) == "doubt" for n in self.items)
        self.refresh_cull()
        self.apply_filter()
        self.toast(f"Проверка фокуса готова: брак {bad}, сомнительно {doubt}, из {len(self.items)}", 8000)
        self.save_timer.start()

    def refresh_cull(self):
        """Счётчики фильтров и кнопка корзины."""
        counts = {"all": len(self.items), "bad": 0, "doubt": 0, "ok": 0}
        for n in self.items:
            v = self.effective(n)
            if v in counts:
                counts[v] += 1
        self.filter_bar.set_counts(counts)
        n = counts["bad"]
        self.trash_btn.setVisible(n > 0)
        self.trash_btn.setText(f"В корзину ({n})…")

    def apply_filter(self):
        mode = self.filter_bar.mode()
        for name, it in self.items.items():
            it.setHidden(mode != "all" and self.effective(name) != mode)

    def toggle_flag(self, flag: str):
        if self.current is None:
            return
        name = self.current.name
        if self.flags.get(name) == flag:
            self.flags.pop(name)
        else:
            self.flags[name] = flag
        self.update_item(name)
        self.show_quality()
        self.refresh_cull()
        self.apply_filter()
        self.save_timer.start()

    def _tile_specs(self, res: dict) -> list[tuple]:
        """Плитки полосы: (рамка, заголовок, подпись, уровень) — места брака и для сравнения самое резкое место кадра."""
        specs = []
        for d in (res.get("defects") or [])[:3]:
            if not d.get("roi"):
                continue
            title = TYPE_TEXT.get(d["type"], d["type"]) + (f" · {d['name']}" if d["name"] != "Кадр" else "")
            if d["type"] == "motion":
                sub = f"направление {d['angle']:.0f}°, размытие {d['sigma']:.1f} px"
            elif d["type"] == "eyes_closed":
                sub = "веки опущены"
            else:
                best = "лучшее место · " if d["name"] == "Кадр" else "размытие "
                sub = f"{best}{d['sigma']:.1f} px (норма ≤ {Q.CFG['soft_doubt']:.1f})"
            specs.append((d["roi"], title, sub, d["level"]))
        ref = res.get("ref")
        if specs and ref and all(abs(sp[0][0] - ref[0]) > 1e-6 for sp in specs):
            specs.append((ref, "Самое резкое место кадра", f"размытие {res.get('best_sigma', 0):.1f} px", "ok"))
        return specs

    def show_quality(self):
        """Полоса под фото и рамки на кадре для текущего снимка."""
        name = self.current.name if self.current else None
        res, flag = self.quality.get(name), self.flags.get(name)
        marks = []
        for d in (res or {}).get("defects", []):
            if d.get("roi") and not flag:
                marks.append((d["roi"], d["level"], TYPE_TEXT.get(d["type"], d["type"])))
        self.view.marks = marks
        self.view.update()
        if name is None or (res is None and not flag and not self.q_left):
            self.qbar.hide()
            return
        self.qbar.show_result(res, flag)
        specs = self._tile_specs(res) if res else []
        if not specs:
            self.qbar.set_tiles([])
            return
        key = (name, tuple(res.get("file") or ()), res.get("v"))
        if key in self.tile_cache:
            self._put_tiles(specs, self.tile_cache[key])
        else:
            run_task(tiles_job, self.current, [sp[0] for sp in specs],
                     done=lambda r, k=key: self.on_tiles(r, k), fail=lambda _m: self.qbar.set_tiles([]))

    def on_tiles(self, res, key):
        _path, imgs = res
        self.tile_cache[key] = imgs
        if len(self.tile_cache) > 12:
            self.tile_cache.pop(next(iter(self.tile_cache)))
        if self.current is not None and self.current.name == key[0]:
            self._put_tiles(self._tile_specs(self.quality.get(key[0]) or {}), imgs)

    def _put_tiles(self, specs: list[tuple], imgs: list):
        self.qbar.set_tiles([ZoomTile(numpy_pixmap(img), t, sub, lvl, box)
                             for (box, t, sub, lvl), img in zip(specs, imgs, strict=False)])

    def zoom_to_roi(self, box):
        self.view.focus_on((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0)

    def _forget_frame(self, p: Path):
        """Кадр ушёл в корзину: убираем его из ленты, кешей и правок."""
        name = p.name
        if self.current == p:
            self.history_timer.stop()
            self.current, self.base = None, None
        it = self.items.pop(name, None)
        if it is not None:
            self.strip.takeItem(self.strip.row(it))
        self.files = [f for f in self.files if f != p]
        for d in (self.thumbs, self.thumb_pix, self.sidecar, self.scene_of, self.quality, self.flags, self.history,
                  self.isos, self.cam_aspects, self.grade_cache):
            d.pop(name, None)
        self.cache.pop(p, None)
        self.history_dirty = True
        for f in (self.folder / MASKS_DIR).glob(f"{name}.*.png"):  # ИИ-маски этого кадра — туда же
            QFile.moveToTrash(str(f))

    def trash_rejected(self):
        """Брак в корзину Windows. RAW уходит только после подтверждения в диалоге и только в корзину."""
        names = self.cull_candidates()
        if not names or self.folder is None:
            return
        rows = []
        for n in names:
            path = self.folder / n
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            rows.append((path, self.thumb_pix.get(n), describe(self.quality.get(n), self.flags.get(n))[1], size))
        dlg = TrashDialog(self, rows)
        if not dlg.exec():
            return
        moved, failed = [], []
        for path in dlg.chosen():
            r = QFile.moveToTrash(str(path))
            (moved if (r[0] if isinstance(r, tuple) else r) else failed).append(path)
        for path in moved:
            self._forget_frame(path)
        if moved and not self.files:
            self.view.set_image(None)
            self.view.message = "В папке не осталось снимков"
            self.view.update()
            self._set_enabled(False)
        self.save_sidecar()
        self.refresh_cull()
        self.apply_filter()
        self.show_quality()
        msg = f"В корзину отправлено кадров: {len(moved)}. Вернуть можно из корзины Windows"
        self.toast(msg + (f". Не удалось: {len(failed)}" if failed else ""), 10000)

    # ---------- выбор кадра

    def on_select(self, item, _prev=None):
        if item is None:
            return
        if self.crop_mode:
            self.a_crop.setChecked(False)
            self.toggle_crop()
        self.commit_history()
        self.store_current()
        self.current = Path(item.data(PATH_ROLE))
        self.remember(last_file=self.current.name)
        self.show_quality()
        self.base = None
        self.full = self.full_path = None
        self.base_dn = None
        self.view.message = "Проявляю RAW…"
        self.view.badge = ""
        self.view.update()
        if self.current in self.cache:
            self.on_preview((self.current, *self.cache[self.current]))
        else:
            run_task(preview_job, self.current, done=self.on_preview, fail=self.on_error)

    def on_preview(self, res):
        path, img, size, iso, cam = res
        self.cache[path] = (img, size, iso, cam)
        self.isos[path.name] = iso
        self.cam_aspects[path.name] = cam
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        if path != self.current:
            return
        self.base = img
        self.full_wh = size
        self.view.set_source_size(*size)
        self.iso_label.setText(f"ISO {iso}" if iso else "")
        saved = self.sidecar.get(path.name)
        self.params = E.normalize_params(saved)
        if not saved and self.auto_on_open.isChecked():
            self.params.update(self.auto_for(path.name, img)[0])
        h = self.history.get(path.name)
        if not h or not h["undo"]:
            self.history[path.name] = {"undo": [copy.deepcopy(self.params)], "redo": []}
        elif json.dumps(h["undo"][-1], sort_keys=True) != json.dumps(self.params, sort_keys=True):
            h["undo"].append(copy.deepcopy(self.params))  # правки меняли в обход истории — это тоже шаг
            h["redo"].clear()
        if not saved and self.crop_cam.isChecked() and self._cam_differs(path.name):
            W, H = size
            self.params["crop"] = E.aspect_crop(W, H, cam)
            self.toast(f"Кадр обрезан до {aspect_label(cam)}, как снимала камера. Ctrl+Z — полный кадр", 8000)
        self.sync_controls()
        self._set_enabled(True)
        self.view.message = ""
        self.request_render()
        self.refresh_look_icons()
        self.ensure_ai_masks()
        if self.carousel.isVisible():  # карусель открыта — варианты сразу для нового кадра
            self.grade_variants = []
            self.carousel.clear()
            self.start_grade()

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
        for key, data in ((QUALITY_KEY, self.quality), (FLAGS_KEY, self.flags)):
            if data:
                self.sidecar[key] = data
            else:
                self.sidecar.pop(key, None)
        if self.folder and self.sidecar:
            try:
                E.write_json(self.folder / SIDECAR, self.sidecar)
            except OSError as e:
                self.toast(f"Не удалось сохранить правки: {e}", 8000)
        if self.folder and self.history_dirty:
            data = {name: {k: h[k][-HISTORY_KEEP:] for k in ("undo", "redo")}
                    for name, h in self.history.items() if len(h["undo"]) > 1 or h["redo"]}
            try:
                E.write_json(self.folder / HISTORY_FILE, data)
                self.history_dirty = False
            except OSError as e:
                self.toast(f"Не удалось сохранить историю отмены: {e}", 8000)

    # ---------- правки и рендер

    def sync_controls(self):
        for key, row in self.rows.items():
            row.set_value(self.params.get(key, 0))
        self.angle_row.set_value(round(self.params.get("angle", 0) * 10))
        self.curve.set_curves(self.params.get("curve"))
        self._sync_grade()
        hsl = self.params.get("hsl") or {}
        for (color, idx), r in self.hsl_rows.items():
            r.set_value(hsl.get(color, [0, 0, 0])[idx])
        self.refresh_mask_list()
        self.update_geometry()
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
        if key in E.AI_SLIDER_CATS and value:
            self.ensure_ai_masks()  # маска для ретуши/боке досчитается в фоне, дальше кадр перерисуется
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
        self.history_timer.start()  # правки «устоялись» 0.4 с — шаг истории (протяжка = один шаг)
        if self.rendering:
            self.dirty = True
            return
        self.rendering = True
        self.gen += 1
        dn = self.base_dn if self.params.get("denoise") else None
        if self.params.get("denoise") and dn is None:
            self.request_denoise()
        run_task(render_job, self.gen, self.base, self.render_params(), self.current_style(), self.current_look(), dn,
                 self.display_geo(),
                 done=self.on_rendered, fail=self.on_render_fail)

    def on_rendered(self, res):
        _gen, out, hist = res
        self.rendering = False
        self.after = to_qimage(out)
        self.hist.set_hist(hist)
        self.curve.hist = hist
        self.curve.update()
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
        if self.base is None or v.scale() <= self.base.shape[1] / self.full_wh[0] * 1.05:
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
        r = v.visible_rect()  # в пикселях показанного (обрезанного) кадра
        rect = (max(0, int(r.left())), max(0, int(r.top())),
                min(int(v.src_w), int(np.ceil(r.right()))), min(int(v.src_h), int(np.ceil(r.bottom()))))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            return
        self.detail_busy = True
        rp = self.render_params()
        rp["crop"] = self.display_geo()[0]
        run_task(detail_job, self.detail_gen, self.full, self.base, rect, min(1.0, v.scale()),
                 rp, self.current_style(), self.current_look(), done=self.on_detail, fail=self.on_detail_fail,
                 pool=AI_POOL if rp.get("denoise") and N.available() else None)  # шумодав — это torch

    def on_full(self, res):
        path, img = res
        if self.full_loading == path:
            self.full_loading = None
        if path != self.current:
            return
        self.full, self.full_path = img, path
        h, w = img.shape[:2]
        if (w, h) != tuple(self.full_wh):  # размер из заголовка RAW может чуть отличаться
            self.full_wh = (w, h)
            self.update_geometry()
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

    # ---------- тональная кривая и HSL
    # Словари заменяются новыми, а не меняются на месте: фоновая обработка держит свою копию params.

    def on_curve(self, ch: str, pts):
        curves = {k: v for k, v in (self.params.get("curve") or {}).items() if k != ch}
        if pts:
            curves[ch] = pts
        self.params["curve"] = curves
        self.request_render()

    def on_curve_shape(self, idx: int):
        if idx > 0:
            self.curve.set_points(CURVE_SHAPES[self.curve_shape.currentText()])
            self.curve_shape.setCurrentIndex(0)

    def reset_curve(self):
        self.params["curve"] = {}
        self.curve.set_curves({})
        self.request_render()

    def on_hsl(self, color: str, idx: int, value: int):
        hsl = {c: list(v) for c, v in (self.params.get("hsl") or {}).items()}
        vals = hsl.get(color, [0, 0, 0])
        vals[idx] = value
        if any(vals):
            hsl[color] = vals
        else:
            hsl.pop(color, None)
        self.params["hsl"] = hsl
        self.request_render()

    def reset_hsl(self):
        self.params["hsl"] = {}
        for r in self.hsl_rows.values():
            r.set_value(0)
        self.request_render()

    # ---------- ИИ-композиция: горизонт и трети

    def _mask01(self, cat: str) -> np.ndarray | None:
        arr = self.ai_arr(self.current.name, cat) if self.current else None
        return None if arr is None else arr.astype(np.float32) / 255

    def ai_horizon(self):
        if self.base is None:
            return
        self.toast("ИИ ищет горизонт…", 0)
        run_task(horizon_job, self.current, self.base, self._mask01("sky"), done=self.on_horizon,
                 fail=lambda m: self.toast(f"Горизонт не найден: {m}", 8000))

    def on_horizon(self, res):
        path, ang = res
        if path != self.current:
            return
        if ang is None:
            self.toast("Не нашёл уверенного горизонта или вертикалей — выровняйте ползунком «Горизонт»", 8000)
            return
        ang = float(np.clip(ang, -45, 45))
        self.angle_row.set_value(round(ang * 10))
        self.on_angle(ang)
        self.toast("Горизонт уже ровный" if ang == 0 else f"Горизонт выровнен: {ang:+.1f}°. Ctrl+Z — вернуть")

    def ai_thirds(self):
        if self.base is None:
            return
        people, sky = self._mask01("people"), self._mask01("sky")
        if (people is None or sky is None) and G.available() and self._after_seg is None:
            missing = [c for c, m in (("people", people), ("sky", sky)) if m is None]
            self._after_seg = self.ai_thirds
            self.toast("ИИ ищет людей и небо на кадре…", 0)
            run_task(segment_job, self.current, self.base, missing, False, pool=AI_POOL,
                     done=self.on_segmented, fail=lambda m: (setattr(self, "_after_seg", None),
                                                             self.toast(f"Маски не получились: {m}", 8000)))
            return
        pt = E.subject_point(people)
        if pt is not None:
            self._apply_thirds(pt, "человек", None)
            return
        self.toast("ИИ ищет главное в кадре…", 0)
        run_task(saliency_job, self.current, self.base, fail=lambda m: self.toast(f"Не получилось: {m}", 8000),
                 done=lambda r: r[0] == self.current and self._apply_thirds(r[1], "самое заметное место",
                                                                            E.horizon_level(sky)))

    def _apply_thirds(self, pt: tuple, what: str, horizon: float | None):
        W, H = self.full_wh
        angle = self.params.get("angle", 0)
        A, _ = E.crop_matrix(W, H, None, angle)  # точка полного кадра → повёрнутый холст, где живёт рамка
        rx, ry = A @ (pt[0] * W, pt[1] * H, 1.0)
        crop = self.params.get("crop")
        if crop:
            aspect = (crop[2] - crop[0]) * W / ((crop[3] - crop[1]) * H)
        else:
            aspect = self.cam_aspects.get(self.current.name) if self._cam_differs(self.current.name) else W / H
        rect = E.thirds_crop(W, H, aspect, angle, (rx / W, ry / H), horizon, sure=what == "человек")
        if self.crop_mode:
            self.cropper.rect = rect
            self.view.update()
        self._set_crop(rect)
        self.update_geometry()
        self.request_render()
        self.toast(f"Правило третей: {what} на пересечении третей" + (", горизонт на трети" if horizon else "")
                   + ". Ctrl+Z — вернуть", 8000)

    # ---------- ИИ-цветокоррекция: карусель вариантов

    def start_grade(self):
        if self.base is None:
            return
        self.carousel.show()
        if self.grade_busy:
            return
        name = self.current.name
        cached = self.grade_cache.get(name)
        if cached and cached[0] == self._grade_key():  # правки те же — карусель сразу, без пересчёта
            self._fill_carousel(*cached[1:])
            return
        self.grade_busy = True
        maps = {c: self.ai_arr(name, c) for c in GR.MASK_CATS}
        scene = (self.scene_of.get(name) or [None])[0]
        first = not G.loaded() and G.available() and any(m is None for m in maps.values())
        self.toast("ИИ смотрит на кадр: объекты, цвета, гармонии…"
                   + (" В первый раз загружаются модели" if first else ""), 0)
        run_task(grade_job, self.scene_defs, self.current, self.base,
                 self.render_params(), self.current_style(), self.current_look(), self.display_geo(), maps, scene,
                 self.thumbs.get(name), done=self.on_graded, fail=self.on_grade_fail, pool=AI_POOL)

    def on_graded(self, res):
        path, new, scene, an, vs, thumbs, scores = res
        self.grade_busy = False
        self._store_ai(path, new, scene)
        if path != self.current:  # пока считали, открыли другой кадр
            self.start_grade()
            return
        self.grade_cache[path.name] = (self._grade_key(), vs, thumbs, scores)
        if len(self.grade_cache) > 12:
            self.grade_cache.pop(next(iter(self.grade_cache)))
        self._fill_carousel(vs, thumbs, scores)
        found = [GR.LABELS.get(r, "") for r in an["regions"] if r in GR.LABELS]
        sc = self.scene_by_id.get(scene or "", {}).get("name", "не определена")
        self.toast(f"Вариантов: {len(vs)}. Сцена: {sc.lower()}. Найдено: {', '.join(found) or 'без масок объектов'}."
                   " Щёлкайте по карусели под кадром", 10000)

    def _grade_key(self) -> str:
        """Правки, от которых зависят варианты карусели (всё, кроме самой ИИ-цветокоррекции)."""
        return json.dumps({k: v for k, v in self.params.items() if k != "ai_grade"}, sort_keys=True, default=str)

    def _store_ai(self, path: Path, masks: dict, scene: str | None):
        """Маски и сцену, досчитанные ИИ-задачей, — в кэш и на диск."""
        for cat, arr in masks.items():
            self.ai_cache[(path.name, cat)] = arr
            try:
                _write_png(path.parent / MASKS_DIR / f"{path.name}.{cat}.png", arr)
            except OSError as e:
                self.toast(f"Не удалось сохранить маску: {e}", 8000)
        if scene and path.name not in self.scene_of:
            self.scene_of[path.name] = [scene, 0.0]
            self.update_item(path.name)

    def _fill_carousel(self, vs: list, thumbs: list, scores: list | None):
        self.grade_variants = vs
        best = int(np.argmax(scores)) if scores else -1
        c = self.carousel
        c.blockSignals(True)
        c.clear()
        for i, th in enumerate(thumbs):
            name = "Исходный" if i == 0 else vs[i - 1]["name"]
            tip = "Без ИИ-цветокоррекции" if i == 0 else vs[i - 1]["rule"]
            if i == best:
                name, tip = f"★ {name}", f"{tip}\n★ ИИ (CLIP) считает этот вариант самым удачным — это подсказка"
            it = QListWidgetItem(QIcon(QPixmap.fromImage(to_qimage(th))), name)
            it.setToolTip(tip + (f"\nОценка ИИ: {scores[i]:+.1f}" if scores else ""))
            c.addItem(it)
        c.blockSignals(False)
        self._sync_grade()

    # ---------- «ИИ-улучшить»: всё лучшее одной кнопкой

    def ai_enhance(self):
        if self.base is None or self.enhance_busy:
            return
        self.enhance_busy = True
        name = self.current.name
        auto, _ = self.auto_for(name, self.base)
        maps = {c: self.ai_arr(name, c) for c in GR.MASK_CATS}
        self.toast("ИИ улучшает кадр: тон, цвета памяти, горизонт…"
                   + (" В первый раз загружаются модели" if not G.loaded() and G.available() else ""), 0)
        run_task(enhance_job, self.scene_defs, self.current, self.base, self.render_params({**self.params, "ai_grade": None}),
                 auto, maps, (self.scene_of.get(name) or [None])[0], self.thumbs.get(name), pool=AI_POOL,
                 done=self.on_enhanced,
                 fail=lambda m: (setattr(self, "enhance_busy", False), self.toast(f"Не получилось: {m}", 10000)))

    def on_enhanced(self, res):
        path, new, scene, auto, natural, angle = res
        self.enhance_busy = False
        self._store_ai(path, new, scene)
        if path != self.current or self.base is None:
            return
        self.commit_history()  # всё ниже — один шаг: одно Ctrl+Z возвращает кадр как был
        self.params.update(auto)
        done = ["тон и баланс белого"]
        self.params["ai_grade"] = {**copy.deepcopy(natural), "strength": 80} if natural else None
        if natural:
            done.append("естественные цвета")
        if angle is not None and 0.3 <= abs(angle) <= 5 and not self.params.get("angle"):
            self.on_angle(float(angle))
            done.append(f"горизонт {angle:+.1f}°")
        self.sync_controls()
        self._sync_grade()
        self.request_render()
        self.toast(f"ИИ-улучшить: {', '.join(done)}. Ctrl+Z — вернуть", 8000)

    def on_grade_fail(self, msg):
        self.grade_busy = False
        self.toast(f"ИИ-цветокоррекция не получилась: {msg}", 10000)

    def on_grade_pick(self, row: int):
        if row < 0 or self.base is None:
            return
        if row == 0 or row - 1 >= len(self.grade_variants):
            self.clear_grade()
            return
        v = self.grade_variants[row - 1]
        self.params["ai_grade"] = {**copy.deepcopy(v), "strength": self.grade_strength_row.slider.value()}
        self.grade_rule.setText(f"<b>{v['name']}</b><br>{v['rule']}")
        self.request_render()

    def clear_grade(self):
        self.params["ai_grade"] = None
        self.grade_rule.setText("Без ИИ-цветокоррекции")
        self.request_render()

    def on_grade_strength(self, _key: str, value: int):
        g = self.params.get("ai_grade")
        if g:
            self.params["ai_grade"] = {**g, "strength": value}
            self.request_render()

    def _sync_grade(self):
        g = self.params.get("ai_grade")
        self.grade_strength_row.set_value(g.get("strength", 80) if g else self.grade_strength_row.slider.value())
        if g:
            self.grade_rule.setText(f"<b>{g['name']}</b><br>{g.get('rule', '')}")
        row = 0
        if g:
            row = next((i + 1 for i, v in enumerate(self.grade_variants) if v["id"] == g["id"]), -1)
        self.carousel.blockSignals(True)
        self.carousel.setCurrentRow(row if self.carousel.count() else -1)
        self.carousel.blockSignals(False)

    # ---------- целевая правка (тянуть по цвету/тону прямо на кадре)

    def set_target_mode(self, mode: str | None):
        if mode and (self.base is None or self.crop_mode):
            mode = None
        self.target_mode = mode
        for btn, m in ((self.tat_hsl, "hsl"), (self.tat_curve, "curve")):
            btn.blockSignals(True)
            btn.setChecked(mode == m)
            btn.blockSignals(False)
        if mode:
            self.mask_list.setCurrentRow(-1)
            self.editor.set_layer(None)
            self.view.editor = self.targeter
            what = (f"ползунки «{self.hsl_tabs.tabText(self.hsl_tabs.currentIndex())}» цвета под курсором"
                    if mode == "hsl" else "точка кривой для тона под курсором")
            self.toast(f"Нажмите на кадре и тяните вверх/вниз — меняется {what}", 10000)
        elif not self.crop_mode:
            self.view.editor = self.editor
        self.view.update()

    def sample_input(self, pos, drop: tuple) -> np.ndarray | None:
        """Цвет точки кадра на входе HSL/кривой: патч 5×5 превью через те же правки, но без
        перечисленных в drop (и без стиля, пресета, масок), — как берёт цвет Lightroom."""
        if self.base is None:
            return None
        xn, yn = self.view.to_norm(pos)
        if not (0 <= xn < 1 and 0 <= yn < 1):
            return None
        h, w = self.base.shape[:2]
        x, y = int(xn * w), int(yn * h)
        patch = self.base[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        p = {**self.params, "masks": [], "ai_grade": None, **{k: {} for k in drop}}
        return E.process(patch, p, None, local=False).reshape(-1, 3).mean(0)

    def tat_start(self, pos) -> bool:
        if self.target_mode == "hsl":
            rgb = self.sample_input(pos, ("hsl",))
            if rgb is None:
                return False
            hsv = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV)[0, 0]
            if hsv[1] < 0.06:
                self.toast("Здесь почти нет цвета — целевая правка HSL работает по цветным участкам")
                return False
            weights = E.hsl_weights(hsv[0])
            idx = self.hsl_tabs.currentIndex()
            cur = self.params.get("hsl") or {}
            names = list(E.HSL_CENTERS)
            self._tat = {"w": {names[i]: float(v) for i, v in enumerate(weights) if v > 0.02}, "idx": idx,
                         "start": {c: list(cur.get(c, [0, 0, 0])) for c in names}}
            main = max(self._tat["w"], key=self._tat["w"].get)
            label = next(lbl for key, lbl, _ in HSL_COLORS if key == main)
            self.toast(f"{label}: тяните вверх — больше, вниз — меньше", 6000)
            return True
        rgb = self.sample_input(pos, ("hsl", "curve"))
        if rgb is None:
            return False
        ch = self.curve.channel
        x = float(rgb @ E.LUM) if ch == "rgb" else float(rgb["rgb".index(ch)])
        x = round(min(max(x, 0.0), 1.0) * 255)
        pts = [list(q) for q in self.curve.points()]
        i = next((k for k, q in enumerate(pts) if abs(q[0] - x) <= 10), None)
        if i is None:  # новая точка на кривой в этом тоне
            y = round(float(E.curve_lut(pts, 256)[x]) * 255)
            pts.append([x, y])
            pts.sort()
            i = pts.index([x, y])
            self.curve.set_points(pts)
        self._tat = {"i": i, "y0": pts[i][1]}
        return True

    def tat_drag(self, up_px: float):
        t = self._tat
        if not t:
            return
        if self.target_mode == "hsl":
            hsl = {c: list(v) for c, v in (self.params.get("hsl") or {}).items()}
            for color, w in t["w"].items():
                vals = list(t["start"][color])
                vals[t["idx"]] = int(np.clip(round(vals[t["idx"]] + up_px * 0.5 * w), -100, 100))
                if any(vals):
                    hsl[color] = vals
                else:
                    hsl.pop(color, None)
                self.hsl_rows[(color, t["idx"])].set_value(vals[t["idx"]])
            self.params["hsl"] = hsl
            self.request_render()
        else:
            pts = [list(q) for q in self.curve.points()]
            i = min(t["i"], len(pts) - 1)
            pts[i][1] = int(np.clip(round(t["y0"] + up_px * 0.5), 0, 255))
            self.curve.set_points(pts)

    # ---------- история: отмена и повтор

    def _hist(self) -> dict:
        return self.history.setdefault(self.current.name, {"undo": [], "redo": []})

    def commit_history(self):
        """Снимок правок кадра, если они изменились с прошлого снимка."""
        self.history_timer.stop()
        if self.base is None or self.current is None:
            return
        h = self._hist()
        snap = copy.deepcopy(self.params)
        if not h["undo"] or h["undo"][-1] != snap:
            h["undo"].append(snap)
            h["redo"].clear()
            del h["undo"][:-100]  # храним последние 100 шагов на кадр
            if len(h["undo"]) > 1:  # это настоящая правка, а не открытие кадра
                self.history_dirty = True
                self.save_timer.start()

    def _restore(self, snap: dict):
        self.params = copy.deepcopy(snap)
        self.editor.set_layer(None)
        self.sync_controls()
        self.request_render()

    def undo(self):
        if self.base is None:
            return
        if self.crop_mode:
            self.a_crop.setChecked(False)
            self.toggle_crop()
        self.commit_history()
        h = self._hist()
        if len(h["undo"]) < 2:
            self.toast("Отменять больше нечего")
            return
        h["redo"].append(h["undo"].pop())
        self._restore(h["undo"][-1])
        self.history_dirty = True
        self.save_timer.start()
        self.toast(f"Отменено. Ещё шагов назад: {len(h['undo']) - 1}. Ctrl+Y — вернуть")

    def redo(self):
        if self.base is None:
            return
        self.commit_history()
        h = self._hist()
        if not h["redo"]:
            self.toast("Возвращать нечего")
            return
        h["undo"].append(h["redo"].pop())
        self._restore(h["undo"][-1])
        self.history_dirty = True
        self.save_timer.start()
        self.toast("Возвращено")

    # ---------- обрезка и горизонт

    def display_geo(self) -> tuple:
        """(обрезка, угол, полный размер) для показа: в режиме обрезки — весь повёрнутый холст."""
        return (None if self.crop_mode else self.params.get("crop"), self.params.get("angle", 0), self.full_wh)

    def update_geometry(self):
        if self.base is None:
            return
        crop, angle, full = self.display_geo()
        A, out = E.crop_matrix(*full, crop, angle)
        self.view.set_geometry(A, full, out)
        self.before = to_qimage(to_u8(E.apply_crop(self.base, crop, angle, full)))
        self.editor.invalidate()
        self.on_view_changed()

    def _cam_differs(self, name: str) -> bool:
        cam = self.cam_aspects.get(name)
        W, H = self.full_wh
        return bool(cam) and abs(np.log(cam / (W / H))) > 0.03

    def _aspect_value(self, idx: int) -> float | None:
        W, H = self.full_wh
        a = ASPECTS[idx][1]
        if a == "cam":
            a = self.cam_aspects.get(self.current.name) if self.current else None
            a = a or W / H
        elif a == "orig":
            a = W / H
        if a and H > W and ASPECTS[idx][1] not in ("cam", "orig") and a > 1:
            a = 1 / a  # вертикальный кадр — вертикальная рамка
        return a

    def _set_crop(self, rect):
        full = rect is None or (rect[0] <= 1e-4 and rect[1] <= 1e-4 and rect[2] >= 1 - 1e-4 and rect[3] >= 1 - 1e-4)
        self.params["crop"] = None if full else [round(v, 5) for v in rect]

    def on_aspect(self, idx: int):
        if self.base is None:
            return
        a = self._aspect_value(idx)
        self._apply_aspect(a)

    def flip_aspect(self):
        if self.base is None:
            return
        x0, y0, x1, y1 = self.cropper.rect if self.crop_mode else (self.params.get("crop") or (0, 0, 1, 1))
        W, H = self.full_wh
        cur = (x1 - x0) * W / ((y1 - y0) * H)
        self._apply_aspect(1 / cur)

    def _apply_aspect(self, a: float | None):
        W, H = self.full_wh
        angle = self.params.get("angle", 0)
        self.cropper.aspect = a
        if a is None:
            self.toast("Пропорции свободные")
            return
        x0, y0, x1, y1 = self.cropper.rect if self.crop_mode else (self.params.get("crop") or (0, 0, 1, 1))
        rect = E.aspect_crop(W, H, a, angle, around=((x0 + x1) / 2, (y0 + y1) / 2))
        if self.crop_mode:
            self.cropper.rect = rect
            self.view.update()
        self._set_crop(rect)
        self.update_geometry()
        self.request_render()

    def on_angle(self, angle: float):
        if self.base is None:
            return
        W, H = self.full_wh
        self.params["angle"] = angle
        rect = self.cropper.rect if self.crop_mode else (self.params.get("crop") or [0, 0, 1, 1])
        a = self.cropper.aspect
        if a:  # с пропорцией: наибольшая рамка этой пропорции вокруг того же центра
            rect = E.aspect_crop(W, H, a, angle, around=((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2))
        else:
            rect = E.fit_crop(W, H, rect, angle)
        if self.crop_mode:
            self.cropper.angle, self.cropper.rect = angle, rect
        self._set_crop(rect)
        self.update_geometry()
        self.request_render()

    def on_crop_rect(self):
        self._set_crop(self.cropper.rect)
        self.history_timer.start()

    def toggle_crop(self):
        if self.base is None:
            self.a_crop.setChecked(False)
            return
        on = self.a_crop.isChecked()
        if on == self.crop_mode:
            return
        if on:
            self.set_target_mode(None)
            self.crop_backup = {k: copy.deepcopy(self.params.get(k)) for k in ("crop", "angle")}
            W, H = self.full_wh
            self.cropper.size = (W, H)
            self.cropper.angle = self.params.get("angle", 0)
            self.cropper.rect = list(self.params.get("crop") or [0, 0, 1, 1])
            self.mask_list.setCurrentRow(-1)
            self.editor.set_layer(None)
            self.crop_mode = True
            self.view.editor = self.cropper
            for a in (self.a_crop_done, self.a_crop_cancel, self.a_overlay, self.a_overlay_rot):
                a.setEnabled(True)
            self.a_mask_show.setEnabled(False)
            self.sections["Обрезка и горизонт"].set_expanded(True, emit=False)  # элементы рамки — внутри этой группы
            self.toast("Тяните рамку и её углы; «Горизонт» — поворот. Enter — готово, Esc — отмена", 10000)
        else:
            self.crop_mode = False
            self.view.editor = self.editor
            for a in (self.a_crop_done, self.a_crop_cancel, self.a_overlay, self.a_overlay_rot):
                a.setEnabled(False)
            self.a_mask_show.setEnabled(True)
            self._set_crop(self.cropper.rect)
            self.toast("Обрезка применена. Ctrl+Z — отменить")
        self.update_geometry()
        self.request_render()

    def cycle_overlay(self):
        self.cropper.cycle_overlay()

    def rotate_overlay(self):
        if self.cropper.rotate_overlay():
            self.view.update()

    def on_overlay(self, kind: str):
        """Вид сетки сменился (клавиша O или список): перерисовать, синхронизировать список, запомнить."""
        self.view.update()
        self.overlay_combo.blockSignals(True)
        self.overlay_combo.setCurrentIndex([k for k, _ in OVERLAYS].index(kind))
        self.overlay_combo.blockSignals(False)
        self.remember(overlay=kind)
        if self.crop_mode:
            self.toast(f"Сетка: {dict(OVERLAYS)[kind]}" + (" · Shift+O — повернуть" if kind in ROTATABLE else ""), 3000)

    def cancel_crop(self):
        if not self.crop_mode:
            return
        self.params.update(self.crop_backup or {})
        self.cropper.rect = list(self.params.get("crop") or [0, 0, 1, 1])
        self.a_crop.setChecked(False)
        self.toggle_crop()
        self.angle_row.set_value(round(self.params.get("angle", 0) * 10))
        self.toast("Обрезка отменена")

    def reset_crop(self):
        if self.base is None:
            return
        self.params["crop"], self.params["angle"] = None, 0.0
        self.cropper.rect, self.cropper.angle = [0.0, 0.0, 1.0, 1.0], 0.0
        self.angle_row.set_value(0)
        self.update_geometry()
        self.request_render()
        self.toast("Полный кадр без поворота")

    # ---------- шумодав (этап 4)

    def request_denoise(self):
        if self.dn_busy or not N.available() or self.base is None:
            return
        self.dn_busy = True
        self.toast("Шумодав: считаю превью… (в первый раз загружается модель)", 0)
        run_task(denoise_job, self.current, self.base, done=self.on_denoised, fail=self.on_denoise_fail,
                 pool=AI_POOL)

    def on_denoised(self, res):
        path, dn = res
        self.dn_busy = False
        if path != self.current:  # пока считали, открыли другой кадр
            if self.params.get("denoise"):
                self.request_denoise()
            return
        self.base_dn = dn
        self.toast("Шумодав готов. В масштабе 100% видно лучше всего", 4000)
        self.request_render()

    def on_denoise_fail(self, msg):
        self.dn_busy = False
        self.toast(f"Шумодав не сработал: {msg}", 10000)

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
        out = {**p, "masks": layers}
        ai = {cat: self.ai_arr(name, cat) for key, cat in E.AI_SLIDER_CATS.items() if p.get(key)}
        if ai:
            out["ai_arr"] = ai
        grade = p.get("ai_grade")
        if grade and grade.get("masks"):
            out["ai_grade"] = {**grade, "masks": [{**copy.deepcopy(m), "arr": self.ai_arr(name, m["cat"])}
                                                  for m in grade["masks"]]}
        return out

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
        if layer is not None and self.target_mode:
            self.set_target_mode(None)
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
        if self.crop_mode:
            self.a_crop.setChecked(False)
            self.toggle_crop()
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
        first = not G.loaded()
        self.toast(f"Ищу «{label}» на кадре…" + (" В первый раз загружается модель (~110 МБ)" if first else ""), 0)
        run_task(segment_job, self.current, self.base, [cat], True, pool=AI_POOL,
                 done=self.on_segmented, fail=lambda m: self.toast(f"Маска ИИ не получилась: {m}", 10000))

    @staticmethod
    def ai_cats(p: dict) -> set[str]:
        """Маски ИИ, которые нужны правкам: ИИ-слои, маски варианта ИИ-цвета, ползунки ИИ-ретуши."""
        layers = list(p.get("masks") or []) + list((p.get("ai_grade") or {}).get("masks", []))
        return ({m["cat"] for m in layers if m["type"] == "ai"}
                | {cat for key, cat in E.AI_SLIDER_CATS.items() if p.get(key)})

    def ensure_ai_masks(self):
        """У правок кадра есть маски ИИ, которых ещё нет (вставка правок, ползунок ретуши) — досчитать в фоне."""
        if self.base is None or not G.available():
            return
        name = self.current.name
        cats = sorted(c for c in self.ai_cats(self.params)
                      if self.ai_arr(name, c) is None and (name, c) not in self.seg_pending)
        if not cats:
            return
        self.seg_pending |= {(name, c) for c in cats}  # второй раз ту же маску в очередь не ставим
        if not G.loaded():
            self.toast("ИИ ищет на кадре людей и объекты… В первый раз загружается модель", 0)

        def failed(msg):
            self.seg_pending -= {(name, c) for c in cats}
            self.toast(f"Маска ИИ не получилась: {msg}", 10000)

        run_task(segment_job, self.current, self.base, cats, False, pool=AI_POOL,
                 done=self.on_segmented, fail=failed)

    def on_segmented(self, res):
        path, maps, create = res
        self.seg_pending -= {(path.name, c) for c in maps}
        for cat, arr in maps.items():
            self.ai_cache[(path.name, cat)] = arr
            try:
                _write_png(path.parent / MASKS_DIR / f"{path.name}.{cat}.png", arr)
            except OSError as e:
                self.toast(f"Не удалось сохранить маску: {e}", 8000)
        if path != self.current:
            return
        if "subject" in maps and maps["subject"].max() < 128 and self.params.get("bokeh"):
            self.toast("Главный объект не найден (нет крупного человека или явного объекта) — "
                       "размытие фона на этом кадре не применяется", 10000)
        if self._after_seg:
            then, self._after_seg = self._after_seg, None
            then()
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
            return (self._with_denoise(name, E.scene_preset(img, sc)),
                    f"Авто по сцене «{sc['name']}»: пресет «{sc['look']}»")
        return self._with_denoise(name, E.auto_params(img)), "Авто: тон и баланс белого подобраны под кадр"

    def _with_denoise(self, name: str, auto: dict) -> dict:
        """Высокий ISO → шумодав в авто (низкий ISO не сбрасывает уже выставленный вручную)."""
        strength = N.iso_strength(self.isos.get(name)) if N.available() else 0
        return {**auto, "denoise": strength} if strength else auto

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
        first = not S.loaded()
        self.toast("Загружаю модель сцен (в первый раз скачивается ~600 МБ)…" if first
                   else f"Распознаю сцены: {len(names)} кадров…", 0)
        run_task(scene_job, self.scene_defs, names, [self.thumbs[n] for n in names],
                 done=self.on_scenes, fail=self.on_scenes_fail, pool=AI_POOL)

    def on_scenes(self, res):
        names, results = res
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

    def match_series(self):
        if self.base is None:
            return
        names = [Path(it.data(PATH_ROLE)).name for it in self.strip.selectedItems()]
        names = [n for n in names if n != self.current.name]
        if not names:
            others = [p.name for p in self.files if p.name != self.current.name]
            if not others:
                return
            box = QMessageBox(self)
            box.setWindowTitle("Единый цвет серии")
            box.setText(f"Кадры не выделены. Подогнать под этот кадр все остальные кадры папки ({len(others)})?\n"
                        "Их правки цвета заменятся, обрезка и маски останутся. Ctrl+Z на каждом кадре вернёт.")
            yes = box.addButton("Подогнать все", QMessageBox.AcceptRole)
            box.addButton("Отмена", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is not yes:
                return
            names = others
        self.commit_history()
        self.store_current()
        ref = dict(self.params)
        targets = [(str(self.folder / n), E.normalize_params(self.sidecar.get(n))) for n in names]
        self.a_match.setEnabled(False)
        self.toast(f"Подгоняю цвет {len(names)} кадров под «{self.current.name}»…", 0)
        run_task(match_job, ref, self.base, self.current_style(), self.current_look(), targets,
                 done=self.on_matched,
                 fail=lambda m: (self.a_match.setEnabled(True), self.toast(f"Не получилось: {m}", 10000)))

    def on_matched(self, res: dict):
        self.a_match.setEnabled(True)
        for name, p in res.items():
            h = self.history.setdefault(
                name, {"undo": [copy.deepcopy(E.normalize_params(self.sidecar.get(name)))], "redo": []})
            self.sidecar[name] = p
            h["undo"].append(copy.deepcopy(p))
            h["redo"].clear()
            if self.current and name == self.current.name:
                self.params = copy.deepcopy(p)
                self.sync_controls()
                self.request_render()
        self.history_dirty |= bool(res)
        self.save_sidecar()
        self.refresh_look_icons()
        self.toast(f"Единый цвет: подогнано кадров — {len(res)}. На каждом кадре Ctrl+Z вернёт прежний цвет", 8000)

    def paste_settings(self):
        if not self.clipboard:
            self.toast("Сначала скопируйте правки (Ctrl+C)")
            return
        items = self.strip.selectedItems()
        for it in items:
            name = Path(it.data(PATH_ROLE)).name
            h = self.history.setdefault(
                name, {"undo": [copy.deepcopy(E.normalize_params(self.sidecar.get(name)))], "redo": []})
            self.sidecar[name] = copy.deepcopy(self.clipboard)
            h["undo"].append(copy.deepcopy(self.sidecar[name]))
            h["redo"].clear()
        if self.current and self.current.name in {Path(i.data(PATH_ROLE)).name for i in items}:
            self.params = self.sidecar[self.current.name]
            self.sync_controls()
            self.request_render()
        self.history_dirty |= bool(items)
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
        for p in [self.params, *(v for k, v in self.sidecar.items() if k not in META_KEYS)]:
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
        yes = box.addButton("Удалить", QMessageBox.AcceptRole)
        box.addButton("Отмена", QMessageBox.RejectRole)
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
        keep = {k: self.params.get(k) for k in ("crop", "angle")}  # пресет не меняет обрезку
        self.params = {**E.normalize_params(E.read_json(PRESETS_DIR / f"{name}.json", {})), **keep}
        self.sync_controls()
        self.request_render()
        self.preset_combo.setCurrentIndex(0)
        self.toast(f"Пресет «{name}» применён")

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:")
        name = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in name.strip())
        if ok and name:
            E.write_json(PRESETS_DIR / f"{name}.json",
                         {k: v for k, v in self.params.items() if k not in ("crop", "angle")})
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
            paths = [p for p in paths if p.name not in self.cull_rejects()]
        if not paths:
            self.toast("Нечего экспортировать: все выбранные кадры отмечены как брак")
            return
        opts = {"out": Path(dlg.out.text()), "scene": dlg.scene_look.isChecked(), "auto": dlg.auto.isChecked(),
                "upscale": (0, 2, 4)[dlg.upscale.currentIndex()],
                "edge": dlg.edge.value(), "quality": dlg.quality.value()}
        self.save_sidecar()
        missing = []
        for p in paths:
            fp = E.normalize_params(self.sidecar.get(p.name, self.params))
            cats = sorted(c for c in self.ai_cats(fp) if self.ai_arr(p.name, c) is None)
            if cats:
                missing.append((str(p), cats))
        if missing and G.available():
            self.toast(f"Готовлю маски ИИ для кадров: {len(missing)}…", 0)
            self.a_export.setEnabled(False)

            def ready(_):
                self._run_export(paths, opts)

            def failed(msg):
                self.a_export.setEnabled(True)
                self.toast(f"Маски ИИ не подготовлены ({msg}). Экспорт без них", 8000)
                self._run_export(paths, opts)

            run_task(prepare_ai_job, missing, str(self.folder / MASKS_DIR), done=ready, fail=failed,
                     pool=AI_POOL)
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
                         "auto": opts["auto"], "long_edge": opts["edge"], "upscale": opts["upscale"],
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
        self.commit_history()
        self.save_sidecar()
        self.remember(geometry=bytes(self.saveGeometry().toHex()).decode(),
                      auto_on_open=self.auto_on_open.isChecked(), scene_auto=self.scene_auto.isChecked(),
                      crop_cam=self.crop_cam.isChecked(), quality_auto=self.quality_auto.isChecked())
        if self.export_thread:
            self.export_thread.stop = True
            self.export_thread.wait()
        super().closeEvent(e)


def apply_dark_theme(app: QApplication):
    """Тема окна: токены и QSS — в theme.py (имя оставлено для main.py и тестов)."""
    apply_theme(app)
