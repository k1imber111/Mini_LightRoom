# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Mini LightRoom — инструкции для Claude

Локальный проявщик RAW (Sony ARW) на Python + PySide6 для Ивана. Отвечать только по-русски.

**Перед любой работой прочитай `PROJECT_STATE.md`**: там что сделано, что не проверено и пошаговый план (этапы 0–8).

## Правила
- Всё локально, без платных API. DeepSeek/GigaChat допустимы только для текстовых команд, картинки туда не отправлять.
- Исходные RAW не менять и не удалять (тестовые ARW: `E:\Canon\Исходники\...`, только чтение). `.env` не читать.
- После каждого этапа обновлять `PROJECT_STATE.md` (разделы 4–5) и `README.md`.
- Git: ветка `main`, атомарные коммиты по этапам.

## Команды
Python 3.12 в `.venv` (не 3.13/3.14: будущие ML-библиотеки под 3.12).
```
.venv\Scripts\python.exe main.py [папка]                      # запуск (или start.bat — сам создаёт .venv)
.venv\Scripts\python.exe tests\test_engine.py                # движок на синтетике
.venv\Scripts\python.exe tests\test_ui_smoke.py              # окно в QT_QPA_PLATFORM=offscreen
.venv\Scripts\python.exe tests\test_real_raw.py путь\к\кадр.ARW  # замеры на настоящем RAW
.venv\Scripts\ruff.exe check --line-length 120 .
```
Библиотеки ИИ (этап 1) — отдельно: `install_ai.bat` / `requirements-ai.txt` (torch cu128 + open_clip_torch).
Тесты — обычные скрипты на `assert` (не pytest), каждый файл запускается отдельно и печатает `OK`. Живое окно глазами пока не проверялось — offscreen-тест этого не заменяет.

## Архитектура
- `mini_lightroom/engine.py` — чистый numpy/OpenCV/rawpy, **без Qt**. `mini_lightroom/ui.py` — всё окно. Новую обработку класть в engine, UI только вызывает.
- **Параметры кадра — плоский dict**: ключи из `engine.SLIDERS` (единый источник: ключ, подпись, диапазон, группа панели) + `style`, `style_strength`, `look`, `look_strength`. Новый ползунок = строка в `SLIDERS` + его обработка в `process()`; UI-панель строится из `SLIDERS` автоматически. Значения целые (экспозиция в сотых EV).
- `process(img, p, style, src_stats, local)` — вход float32 RGB 0..1. `local=False` отключает пространственные операции (сглаживание, чёткость, резкость, виньетка), чтобы та же функция годилась для 3D LUT (`export_cube`). Пространственная правка обязана уважать этот флаг.
- Стиль = статистики LAB референса (`lab_stats`), перенос по Рейнхарду; хранится в `styles/*.json`. Пресеты — `presets/*.json`. Пути к ним от `APP_DIR` (корень проекта).
- Правки кадров — sidecar `.mini_lightroom.json` в папке со снимками (`{имя_файла: params}`).
- **Фон**:
  - превью/миниатюры/рендер/стиль — `run_task(fn, ..., done=, fail=)` на `QThreadPool` (миниатюры в отдельном пуле на 2 потока);
  - рендер превью «схлопывается» флагами `rendering`/`dirty`: пока идёт рендер, новые запросы только ставят `dirty`;
  - экспорт — `ExportThread` → `ProcessPoolExecutor` (≤2 процесса, полный RAW ≈ 1 ГБ). Поэтому `engine.export_one(job)` принимает только picklable dict и должен оставаться функцией верхнего уровня; `main.py` вызывает `multiprocessing.freeze_support()`.
- **Масштаб**: `ImageView` хранит масштаб как долю от полного разрешения. Поверх растянутого превью рисуется «деталь» — видимая область, обработанная `engine.process_region` из полного кадра (`full_job`, один на текущий снимок). Поэтому `process` принимает `frame=`: у вырезки размытия и виньетка считаются от размеров целого кадра. Новая пространственная правка должна брать размеры из `fw/fh/x0/y0`, а не из `img.shape`, иначе вырезка разойдётся с превью (тест в `test_engine.py`).
- **Пресеты-образы и сцены**: рецепты пресетов — данные в `looks/*.json`, применяет `engine.apply_look` (только поточечные операции, чтобы работал LUT). Сцены — `scenes.json` + `mini_lightroom/scene.py` (CLIP). torch/open_clip необязательны и импортируются только внутри `SceneClassifier`; без них программа обязана работать. Результаты сцен лежат в sidecar под ключом `__scenes__`, поэтому sidecar читать через `.get(имя_файла)`, а не перебором значений.
- **Кириллица в путях**: файлы читаются через `Path.read_bytes()` (`_read_bytes`) и декодируются из памяти, а не передаются путём в C-библиотеки (rawpy/cv2.imread ломаются). Новые загрузчики делать так же. Тесты специально используют кириллические временные папки.
- Правка файлов с кириллицей из PowerShell ненадёжна — лучше Edit или небольшой Python-скрипт. Предупреждение Qt `Cannot find font directory` безвредно.
