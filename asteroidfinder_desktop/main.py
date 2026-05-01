from __future__ import annotations

import os
import sys


def main() -> int:
    # Qt worker threads have smaller stacks than the main Python thread. Some
    # NumPy/OpenBLAS alignment paths can overflow those stacks if BLAS fans out
    # internally, so keep native math single-threaded inside the desktop app.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
