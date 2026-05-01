from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is not installed. Install the desktop extra with: python3 -m pip install -e '.[desktop]'"
        ) from exc

    from .main_window import MainWindow, apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName("AsteroidFinder")
    apply_dark_theme(app)
    window = MainWindow()
    window.resize(1360, 860)
    window.show()
    return app.exec()
