"""Лица, глаза и закрытые глаза: MediaPipe Face Landmarker, локально на процессоре. Без Qt.

mediapipe необязателен: без него `available()` — False, проверка брака работает без лиц и глаз.
Модель (3.7 МБ) скачивается в `models/` один раз при первом использовании; вызовы — только из `ui.AI_POOL`
(pybind11, как torch: постоянный поток) и под замком `_LOCK`.
"""
from __future__ import annotations

import atexit
import contextlib
import threading
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["MODEL_FILE", "FaceFinder", "available", "eyes_state", "get_finder"]

MODEL_FILE = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/" + MODEL_FILE
MODEL_MIN_BYTES = 3_000_000
# Контуры глаз в сетке MediaPipe (478 точек). «Левый» и «правый» — как на снимке, слева направо.
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
EYE_B = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Точки для EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)
EAR_A = (33, 160, 158, 133, 153, 144)
EAR_B = (362, 385, 387, 263, 373, 380)
BLINK_MIN = 0.6    # blendshape eyeBlink*: выше — веко опущено
EAR_MAX = 0.18     # ниже — глаз почти закрыт (открытый ≈ 0.25–0.35)
MIN_EYE_PX = 14    # глаз у́же этого на анализируемом изображении: судить нельзя («не проверено»)
_LOCK = threading.Lock()
_finder: FaceFinder | None = None
_download_failed = False


def available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


def _ear(pts: np.ndarray, idx: tuple) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9))


def _box(pts: np.ndarray, idx: list, pad: float) -> list[float]:
    p = pts[idx]
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    w, h = x1 - x0, max(y1 - y0, (x1 - x0) * 0.5)  # у закрытого глаза контур плоский — рамка не меньше половины ширины
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [float(max(0, cx - w * pad)), float(max(0, cy - h * pad)), float(min(1, cx + w * pad)),
            float(min(1, cy + h * pad))]


def eyes_state(pts: np.ndarray, blink: tuple[float, float], size: tuple[int, int]) -> dict:
    """Состояние глаз одного лица. pts — (N, 2) точки сетки в долях кадра, blink — (левый, правый) eyeBlink,
    size — (ширина, высота) анализируемого изображения в пикселях. closed: list имён закрытых глаз
    ("left"/"right" — как на снимке) или None, если глаза слишком мелкие и судить нельзя."""
    W, H = size
    out = {"eyes": {"left": _box(pts, EYE_A, 1.6), "right": _box(pts, EYE_B, 1.6)}, "blink": list(blink)}
    ears = [_ear(pts * (W, H), EAR_A), _ear(pts * (W, H), EAR_B)]
    out["ear"] = ears
    # куда смотрит: кончик носа (точка 1) правее середины между глазами — вправо (доля расстояния между глазами)
    mid = (pts[EYE_A, 0].mean() + pts[EYE_B, 0].mean()) / 2
    out["yaw"] = float((pts[1, 0] - mid) / max(abs(pts[EYE_B, 0].mean() - pts[EYE_A, 0].mean()), 1e-6))
    width_px = [float(np.ptp(pts[EYE_A, 0]) * W), float(np.ptp(pts[EYE_B, 0]) * W)]
    out["eye_px"] = width_px
    if min(width_px) < MIN_EYE_PX:
        out["closed"] = None
        return out
    # Веко опущено по обоим признакам. Blendshape берём максимальный из двух: не зависим от того, какой из них
    # к какому глазу относится (не проверено на живом лице), а EAR считается по контуру именно этого глаза.
    closed_by_blink = max(blink) >= BLINK_MIN
    out["closed"] = [n for n, e in (("left", ears[0]), ("right", ears[1])) if closed_by_blink and e < EAR_MAX]
    return out


class FaceFinder:
    def __init__(self, model_bytes: bytes):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_buffer=model_bytes), output_face_blendshapes=True, num_faces=8,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5))

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Лица на RGB uint8: [{box, eyes: {left,right}, blink, ear, eye_px, closed}] по убыванию площади."""
        H, W = rgb.shape[:2]
        with _LOCK:
            res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(rgb)))
        faces = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x, p.y] for p in lms], np.float32)
            blend = {c.category_name: c.score for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            # eyeBlinkLeft — левый глаз человека, то есть правый на снимке: порядок слева направо наоборот
            st = eyes_state(pts, (blend.get("eyeBlinkRight", 0.0), blend.get("eyeBlinkLeft", 0.0)), (W, H))
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            st["box"] = [float(max(0, x0)), float(max(0, y0)), float(min(1, x1)), float(min(1, y1))]
            faces.append(st)
        faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        return faces


def _download(dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if tmp.stat().st_size < MODEL_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except OSError:
        return False


def get_finder(models_dir: Path) -> FaceFinder | None:
    """Один поисковик лиц на программу. None — нет mediapipe или не удалось скачать модель."""
    global _finder
    if _finder is not None:
        return _finder
    if not available():
        return None
    global _download_failed
    path = Path(models_dir) / MODEL_FILE
    if not path.exists() and (_download_failed or not _download(path)):
        _download_failed = True  # нет сети — не пытаться заново на каждом кадре
        return None
    _finder = FaceFinder(path.read_bytes())
    atexit.register(_close)
    return _finder


def _close() -> None:
    """Закрыть модель до остановки интерпретатора: иначе её деструктор mediapipe печатает исключение при выходе."""
    global _finder
    if _finder is not None:
        with contextlib.suppress(Exception):  # выходим: ошибка закрытия уже ничего не меняет
            _finder._lm.close()
        _finder = None
