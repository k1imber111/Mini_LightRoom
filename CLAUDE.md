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
- **Параметры кадра — плоский dict**: ключи из `engine.SLIDERS` (единый источник: ключ, подпись, диапазон, группа панели) + стиль (`style`, `style_strength` — цвет, `style_tone` — свет, `style_skin`, `style_mode`, `style2`, `style_mix`) + `look`, `look_strength` + `curve`/`hsl` (кривая и HSL, заменять словари новыми, не мутировать) + `masks`, `crop`, `angle`. Сохранённые правки загружать через `engine.normalize_params` (совместимость со старыми). Новый ползунок = строка в `SLIDERS` + его обработка в `process()`; UI-панель строится из `SLIDERS` автоматически. Значения целые (экспозиция в сотых EV).
- `process(img, p, style, src_stats, local)` — вход float32 RGB 0..1. `local=False` отключает пространственные операции (сглаживание, чёткость, резкость, виньетка), чтобы та же функция годилась для 3D LUT (`export_cube`). Пространственная правка обязана уважать этот флаг.
- Стиль = статистики LAB референса (`lab_stats`), перенос по Рейнхарду; хранится в `styles/*.json`. Пресеты — `presets/*.json`. Пути к ним от `APP_DIR` (корень проекта).
- Правки кадров — sidecar `.mini_lightroom.json` в папке со снимками (`{имя_файла: params}`); история отмены — `.mini_lightroom_history.json` там же (пишется в `save_sidecar` по флагу `history_dirty`: кто меняет `self.history` в обход `commit_history`/`undo`/`redo`, ставит флаг сам).
- **Фон**:
  - превью/миниатюры/рендер/стиль — `run_task(fn, ..., done=, fail=)` на `QThreadPool` (миниатюры в отдельном пуле на 2 потока);
  - рендер превью «схлопывается» флагами `rendering`/`dirty`: пока идёт рендер, новые запросы только ставят `dirty`;
  - **torch — только в `ui.AI_POOL`** (`ThreadPoolExecutor(1)`, постоянный поток Python): в потоках `QThreadPool` второй вызов torch вешает или роняет программу (PySide уничтожает состояние потока после задачи, pybind11 держит указатель). Модели — синглтоны `segment.get_segmenter` / `scene.get_classifier` под `enhance.GPU_LOCK`. В тестах не ждать фоновые задачи через `QTest.qWait` — он держит GIL;
  - экспорт — `ExportThread` → `ProcessPoolExecutor` (≤2 процесса, полный RAW ≈ 1 ГБ). Поэтому `engine.export_one(job)` принимает только picklable dict и должен оставаться функцией верхнего уровня; `main.py` вызывает `multiprocessing.freeze_support()`.
- **Масштаб**: `ImageView` хранит масштаб как долю от полного разрешения. Поверх растянутого превью рисуется «деталь» — видимая область, обработанная `engine.process_region` из полного кадра (`full_job`, один на текущий снимок). Поэтому `process` принимает `frame=`: у вырезки размытия и виньетка считаются от размеров целого кадра. Новая пространственная правка должна брать размеры из `fw/fh/x0/y0`, а не из `img.shape`, иначе вырезка разойдётся с превью (тест в `test_engine.py`).
- **Пресеты-образы и сцены**: рецепты пресетов — данные в `looks/*.json`, применяет `engine.apply_look` (только поточечные операции, чтобы работал LUT). Сцены — `scenes.json` + `mini_lightroom/scene.py` (CLIP). torch/open_clip необязательны и импортируются только внутри `SceneClassifier`; без них программа обязана работать. Результаты сцен лежат в sidecar под ключом `__scenes__`, поэтому sidecar читать через `.get(имя_файла)`, а не перебором значений.
- **Маски**: слои в `params["masks"]`, геометрия в долях кадра, растеризация — `mini_lightroom/masks.py` (без Qt), применение — `engine.apply_masks` (только `local=True`). ИИ-маски — `segment.py` (SegFormer, лениво), PNG в `<папка>/.mini_lightroom_masks/`; в обработку массив идёт ключом `arr` только через `ui.render_params()` — всегда передавай в фоновые задачи её результат, а не `self.params`. Жесты на кадре — `mask_editor.py`; сигналы с dict в PySide объявлять `Signal(object)` (`Signal(dict)` передаёт копию).
- **Шумодав и увеличение**: `enhance.py` (torch + spandrel, лениво, свой кэш моделей в каждом процессе). Шумодав — ключ `denoise` в `SLIDERS`, но применяется к исходнику ДО `process()` (превью: `base_dn` + смешивание в `render_job`; масштаб: `process_region(prep=)`; экспорт: `export_one`), а `process` его игнорирует. Увеличение — только в `export_one` (`job["upscale"]`).
- **Обрезка и история**: обрезка/горизонт (`params["crop"]`, `params["angle"]`) — последний шаг после всех правок, одна матрица `engine.crop_matrix` для превью (`render_job`), масштаба (`process_view_region`) и экспорта; всё остальное (маски, стиль, шумодав) живёт в координатах полного кадра. Отмена: снимок `params` через 0.4 с после `request_render` — любая правка, прошедшая через `request_render`, попадает в историю автоматически.
- **ИИ-цветокоррекция**: `grading.py` (без Qt) даёт варианты `{recipe, masks}`; применяются через `params["ai_grade"]` в `process` (рецепт — `apply_look`, маски — в `apply_masks` с `amount`). Новое правило = функция в `variants` + проверка заметности/«кислоты» скриптом по реальным кадрам (ΔE к исходнику 4–15, без добавочного пересвета).
- **Кириллица в путях**: файлы читаются через `Path.read_bytes()` (`_read_bytes`) и декодируются из памяти, а не передаются путём в C-библиотеки (rawpy/cv2.imread ломаются). Новые загрузчики делать так же. Тесты специально используют кириллические временные папки.
- Правка файлов с кириллицей из PowerShell ненадёжна — лучше Edit или небольшой Python-скрипт. Предупреждение Qt `Cannot find font directory` безвредно.
