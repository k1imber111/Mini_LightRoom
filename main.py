"""Запуск Mini LightRoom: python main.py [папка_со_снимками]"""
import faulthandler
import multiprocessing
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from mini_lightroom.ui import MainWindow, apply_dark_theme

LOGS = Path(__file__).resolve().parent / "logs"
RUN_LOG = LOGS / "crash_current.log"  # журнал этого запуска: при нормальном выходе удаляется
_crash_log = None  # файл держим открытым: faulthandler пишет в него при аварийном падении


def watch_hangs(app: QApplication) -> None:
    """Падения и зависания — в logs/: стеки всех потоков показывают, на чём именно встало.
    faulthandler в Windows пишет и безобидные исключения (COM 0x8001010d), поэтому журнал запуска
    оставляем, только если программа не дошла до нормального выхода: тогда он — crash_<время>.log."""
    global _crash_log
    try:
        LOGS.mkdir(exist_ok=True)
        if RUN_LOG.exists() and RUN_LOG.stat().st_size:  # прошлый запуск оборвался — сохранить его журнал
            RUN_LOG.replace(LOGS / time.strftime("crash_%Y-%m-%d_%H-%M-%S.log", time.localtime(RUN_LOG.stat().st_mtime)))
        _crash_log = open(RUN_LOG, "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        return  # папка программы только для чтения — без журнала
    faulthandler.enable(_crash_log, all_threads=True)
    beat = [time.monotonic()]
    timer = QTimer(app, interval=1000, timeout=lambda: beat.__setitem__(0, time.monotonic()))
    timer.start()

    def watch():
        dumped = False
        while True:
            time.sleep(5)
            stale = time.monotonic() - beat[0] > 15  # окно не обрабатывало события 15 с
            if stale and not dumped:
                with open(LOGS / "hang.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== окно не отвечает 15 с: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    fh.flush()
                    faulthandler.dump_traceback(fh, all_threads=True)
            dumped = stale

    threading.Thread(target=watch, daemon=True, name="сторож").start()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mini LightRoom")
    apply_dark_theme(app)
    watch_hangs(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        win.load_folder(Path(sys.argv[1]))
    else:
        win.restore_session()  # последняя папка на последнем кадре — продолжаем, где остановились
    code = app.exec()
    if _crash_log is not None:  # дошли до нормального выхода — записи журнала были безобидными
        faulthandler.disable()
        _crash_log.close()
        RUN_LOG.unlink(missing_ok=True)
    sys.exit(code)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # нужно для параллельного экспорта в Windows
    main()
