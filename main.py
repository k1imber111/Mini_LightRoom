"""Запуск Mini LightRoom: python main.py [папка_со_снимками]"""
import multiprocessing
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from mini_lightroom.ui import MainWindow, apply_dark_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mini LightRoom")
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        win.load_folder(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()  # нужно для параллельного экспорта в Windows
    main()
