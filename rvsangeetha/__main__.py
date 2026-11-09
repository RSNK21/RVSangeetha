"""Entry point: ``python -m rvsangeetha``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from . import APP_NAME
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    # optional: open a file passed on the command line
    if len(sys.argv) > 1:
        try:
            win.doc.load(sys.argv[1])
        except Exception as e:
            print(f"Could not open {sys.argv[1]}: {e}", file=sys.stderr)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
