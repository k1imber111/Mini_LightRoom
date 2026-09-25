"""Проверка на настоящем RAW (файл только читается): python tests\\test_real_raw.py путь\\к\\снимку.ARW"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_lightroom import engine as E  # noqa: E402

src = Path(sys.argv[1])
t = time.perf_counter()
th = E.load_thumb(src)
print(f"миниатюра {th.shape} за {time.perf_counter() - t:.2f} с, резкость {E.sharpness(th):.0f}")
fw, fh = E.full_size(src)
assert (th.shape[1] > th.shape[0]) == (fw > fh), "миниатюра повёрнута не так, как кадр (двойной поворот?)"
t = time.perf_counter()
prev = E.load_image(src, half=True, max_side=1400)
print(f"превью {prev.shape} за {time.perf_counter() - t:.2f} с")
auto = E.auto_params(prev)
print("авто:", auto)
p = {**E.default_params(), **auto, "vibrance": 20, "clarity": 15, "sharpness": 30}
t = time.perf_counter()
E.process(prev, p)
print(f"рендер превью за {(time.perf_counter() - t) * 1000:.0f} мс")
with tempfile.TemporaryDirectory() as d:
    t = time.perf_counter()
    E.export_one({"src": str(src), "dst": str(Path(d) / "out.jpg"), "params": p, "style": None,
                  "auto": False, "long_edge": 0, "quality": 92})
    out = E.load_image(Path(d) / "out.jpg")
    print(f"полный экспорт {out.shape} за {time.perf_counter() - t:.1f} с")
print("RAW OK")
