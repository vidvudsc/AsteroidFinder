from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import shutil
from time import perf_counter
import webbrowser

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QRectF, QThreadPool, QTimer, Qt
from PIL import Image as PILImage
from PIL import ImageOps
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from asteroidfinder.alignment import align_images
from asteroidfinder.calibration import calibrate_images_with_persistent_hot_pixels
from asteroidfinder.diagnostics import plot_track_diagnostics
from asteroidfinder.io import load_image
from asteroidfinder.known_objects import KnownObject, query_known_objects_with_motion_cache, write_known_objects_csv
from asteroidfinder.mpc import write_detected_track_ades, write_detected_track_mpc
from asteroidfinder.platesolve import solve_image
from asteroidfinder.report import generate_html_report
from asteroidfinder.tracking import Track, track_moving_objects
from asteroidfinder.workflow import write_tracks_csv

from .session import FITS_EXTENSIONS, IMAGE_EXTENSIONS, FrameInfo, SessionState, filter_image_files, natural_sorted, save_session
from .viewer import FitsViewer, PreparedDisplayFrame, prepare_display_frame
from .workers import FunctionWorker


ANALYSIS_PANEL_WIDTH = 380
WORKFLOW_PANEL_WIDTH = 320


def _parse_optional_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_LAT_KEYS = ("SITELAT", "OBS-LAT", "OBSLAT", "OBSGEO-B", "LATITUDE", "LAT-OBS", "TEL_LAT")
_LON_KEYS = ("SITELONG", "OBS-LONG", "OBSLON", "OBSGEO-L", "LONGITUD", "LONG-OBS", "TEL_LONG")
_ELEV_KEYS = ("SITEELEV", "OBS-ELEV", "OBSALT", "OBSGEO-H", "ELEVATIO", "ELEV-OBS", "ALTITUDE", "TEL_ELEV")
_MPC_KEYS = ("MPC-CODE", "MPCCODE", "OBSCODE", "MPC_CODE")


def _read_observer_from_header(header: fits.Header) -> tuple[float | None, float | None, float | None, str | None]:
    def first(keys: tuple[str, ...]) -> Any:
        for key in keys:
            value = header.get(key)
            if value not in (None, ""):
                return value
        return None

    def parse_angle(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            from astropy.coordinates import Angle
            return float(Angle(text, unit=u.deg).deg)
        except Exception:
            return None

    def parse_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    lat_raw = first(_LAT_KEYS)
    lon_raw = first(_LON_KEYS)
    elev_raw = first(_ELEV_KEYS)
    mpc_raw = first(_MPC_KEYS)

    lat = parse_angle(lat_raw)
    lon = parse_angle(lon_raw)
    elev = parse_float(elev_raw)
    mpc = str(mpc_raw).strip() if mpc_raw not in (None, "") else None
    return lat, lon, elev, mpc


def _build_observer_location(settings: Any) -> EarthLocation | None:
    lat = settings.observer_lat_deg
    lon = settings.observer_lon_deg
    if lat is None or lon is None:
        return None
    elev = settings.observer_elev_m if settings.observer_elev_m is not None else 0.0
    try:
        return EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
    except Exception:
        return None


class WorkflowStepDialog(QDialog):
    """Small dialog that shows relevant settings for a single workflow step."""

    def __init__(self, title: str, fields: list[tuple[str, QWidget]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        if fields:
            form = QFormLayout()
            for label, widget in fields:
                form.addRow(label, widget)
            layout.addLayout(form)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.accept)
        button_row.addWidget(cancel)
        button_row.addWidget(self.run_button)
        layout.addLayout(button_row)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AsteroidFinder")
        self.session = SessionState()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)
        self.thread_pool.setStackSize(32 * 1024 * 1024)
        self.frame_thread_pool = QThreadPool(self)
        self.frame_thread_pool.setMaxThreadCount(1)
        self.frame_thread_pool.setStackSize(32 * 1024 * 1024)
        self._workers: list[FunctionWorker] = []
        self._frame_workers: list[FunctionWorker] = []
        self._diagnostic_windows: list[QDialog] = []
        self._report_windows: list[QDialog] = []
        self._qa_windows: list[QDialog] = []
        self._known_object_windows: list[QDialog] = []
        self._prediction_windows: list[QDialog] = []
        self._submission_windows: list[QDialog] = []
        self.tracks: list[Track] = []
        self.known_objects: list[KnownObject] = []
        self.track_known_matches: dict[int, str] = {}
        self.track_known_objects: dict[int, KnownObject] = {}
        self.visible_track_indices: set[int] = set()
        self._last_skipped_images = 0
        self._worker_started_at: dict[str, float] = {}
        self._worker_progress_totals: dict[str, int] = {}
        self._plate_info_cache: dict[Path, tuple[str, str, str, str]] = {}
        self._updating_track_table = False
        self._current_frame_index = 0
        self._frame_load_serial = 0
        self._frame_load_in_progress = False
        self._blink_pending_advance = False
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._advance_blink)

        self.viewer = FitsViewer()
        self.viewer.on_frame_step = self._step_frame
        self.frame_table = QTableWidget(0, 5)
        self.track_table = QTableWidget(0, 7)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("MutedText")
        self.progress_label.setMinimumWidth(130)
        self.progress_label.setMaximumWidth(190)
        self.plate_status = QLabel("No frame selected")
        self.plate_center = QLabel("")
        self.plate_scale = QLabel("")
        self.plate_path = QLabel("")
        self.plate_path.setWordWrap(True)

        self._workflow_done: dict[str, bool] = {}
        self._workflow_actions: dict[str, QAction] = {}
        self.path_label = QLabel("No images loaded")
        self.path_label.setObjectName("MutedText")
        self.path_label.setToolTip("Click a workflow step to begin")
        self.show_full_track = QCheckBox("Show full track")
        self.show_full_track.setToolTip("Off shows one circle on the current frame. On shows every detection and the motion line.")
        self.show_full_track.toggled.connect(lambda _: self._draw_checked_tracks())
        self.show_known_objects = QCheckBox("Show known objects")
        self.show_known_objects.setToolTip("Overlay known SkyBoT/MPC object predictions for the selected frame.")
        self.show_known_objects.toggled.connect(lambda _: self._draw_checked_tracks())
        self.invert_check = QCheckBox("Invert")
        self.blink_slider = QSlider(Qt.Orientation.Horizontal)
        self.blink_slider.setRange(1, 20)
        self.blink_slider.setValue(15)
        self.blink_slider.setFixedWidth(92)
        self.blink_slider.setToolTip("Blink speed: left is slower, right is faster")

        self._build_ui()
        self._wire_actions()

    def _build_ui(self) -> None:
        open_action = QAction("Import Images", self)
        open_action.setShortcut("Meta+O")
        open_action.triggered.connect(self.import_images)
        self.addAction(open_action)
        self._build_menus(open_action)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        save_action = QAction("Save Session", self)
        save_action.setShortcut("Meta+S")
        save_action.triggered.connect(self.save_session_file)
        toolbar.addAction(save_action)
        toolbar.addSeparator()

        for key, label, handler in [
            ("calibration", "Calibrate", self._open_calibration_dialog),
            ("plate solve", "Solve", self._open_plate_solve_dialog),
            ("alignment", "Align", self._open_alignment_dialog),
            ("tracking", "Detect", self._open_tracking_dialog),
            ("known objects", "Known", self._open_known_objects_dialog),
            ("prediction", "Predict", self.open_prediction_window),
            ("submission export", "Submit", self.open_submission_window),
            ("report", "Report", self.open_report_window),
        ]:
            action = QAction(label, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)
            self._workflow_actions[key] = action

        toolbar.addSeparator()
        toolbar.addWidget(self.path_label)

        self._analysis_widget = self._analysis_panel()
        root = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._center_panel())
        root.addWidget(self._analysis_widget)
        root.setStretchFactor(0, 1)
        root.setStretchFactor(1, 0)
        self.setCentralWidget(root)

    def _center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self.viewer, 1)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.progress_bar, 1)
        layout.addLayout(progress_row)
        return panel

    def _build_menus(self, open_action: QAction) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction("Choose Output Folder", self.choose_output_folder)
        save_action = QAction("Save Session", self)
        save_action.setShortcut("Meta+S")
        save_action.triggered.connect(self.save_session_file)
        file_menu.addAction(save_action)
        file_menu.addAction("Open Report Window", self.open_report_window)
        file_menu.addSeparator()
        quit_action = QAction("Quit AsteroidFinder", self)
        quit_action.setShortcut("Meta+Q")
        quit_action.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("View")
        fit_action = QAction("Fit Image To View", self)
        fit_action.setShortcut("Meta+0")
        fit_action.triggered.connect(self.viewer.fit_to_view)
        view_menu.addAction(fit_action)
        invert_action = QAction("Invert Image", self)
        invert_action.setCheckable(True)
        invert_action.setShortcut("Meta+I")
        invert_action.toggled.connect(self.invert_check.setChecked)
        self.invert_check.toggled.connect(invert_action.setChecked)
        view_menu.addAction(invert_action)
        view_menu.addSeparator()
        full_track_action = QAction("Show Full Track", self)
        full_track_action.setCheckable(True)
        full_track_action.toggled.connect(self.show_full_track.setChecked)
        self.show_full_track.toggled.connect(full_track_action.setChecked)
        view_menu.addAction(full_track_action)
        known_action = QAction("Show Known Objects", self)
        known_action.setCheckable(True)
        known_action.toggled.connect(self.show_known_objects.setChecked)
        self.show_known_objects.toggled.connect(known_action.setChecked)
        view_menu.addAction(known_action)
        view_menu.addAction("Known Object Info", self.open_known_objects_window)
        view_menu.addSeparator()
        view_menu.addAction("Previous Frame", lambda: self._step_frame(-1))
        view_menu.addAction("Next Frame", lambda: self._step_frame(1))
        view_menu.addAction("Start/Pause Blink", self.toggle_blink)

        workflow_menu = self.menuBar().addMenu("Workflow")
        self._menu_workflow_actions: dict[str, QAction] = {}
        for key, label, handler in [
            ("calibration", "Clean Hot Pixels", self._open_calibration_dialog),
            ("plate solve", "Plate Solve", self._open_plate_solve_dialog),
            ("alignment", "Align Stars", self._open_alignment_dialog),
            ("tracking", "Detect Movers", self._open_tracking_dialog),
            ("known objects", "Identify Known", self._open_known_objects_dialog),
            ("prediction", "Predict Track Path", self.open_prediction_window),
            ("submission export", "Submission Window", self.open_submission_window),
            ("report", "Report", self.open_report_window),
        ]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setEnabled(True)
            action.triggered.connect(lambda checked, h=handler: h())
            workflow_menu.addAction(action)
            self._menu_workflow_actions[key] = action
        workflow_menu.addSeparator()
        workflow_menu.addAction("Run Full Pipeline", self.run_full_workflow)

        analysis_menu = self.menuBar().addMenu("Analysis")
        analysis_menu.addAction("Movement Chart", self.open_or_generate_movement_graph)
        analysis_menu.addAction("Track Cutouts", self._open_cutout_window)
        analysis_menu.addAction("Prediction Window", self.open_prediction_window)
        analysis_menu.addAction("Known Object Info", self.open_known_objects_window)
        analysis_menu.addSeparator()
        analysis_menu.addAction("Hot Pixel QA", self.open_hot_pixel_qa)
        analysis_menu.addAction("Alignment QA", self.open_alignment_qa)
        analysis_menu.addAction("PNG Diagnostics", self.write_png_diagnostics)
        analysis_menu.addSeparator()
        analysis_menu.addAction("Submission Window", self.open_submission_window)
        analysis_menu.addAction("Export MPC (80-col)", self._open_export_dialog)
        analysis_menu.addAction("Export ADES (PSV)", self._open_ades_export_dialog)
        analysis_menu.addAction("Report", self.open_report_window)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About AsteroidFinder", self._show_about)

    def _analysis_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("SidePanel")
        panel.setMinimumWidth(ANALYSIS_PANEL_WIDTH)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)

        blink_row = QHBoxLayout()
        previous = _icon_button("⏮", "Previous frame")
        previous.clicked.connect(lambda: self._step_frame(-1))
        self.play_button = _icon_button("▶", "Start blink")
        self.play_button.clicked.connect(self.toggle_blink)
        next_button = _icon_button("⏭", "Next frame")
        next_button.clicked.connect(lambda: self._step_frame(1))
        fit_button = _icon_button("⛶", "Fit image to view")
        fit_button.clicked.connect(self.viewer.fit_to_view)
        self.invert_check.toggled.connect(self.viewer.set_inverted)
        blink_row.addWidget(previous)
        blink_row.addWidget(self.play_button)
        blink_row.addWidget(next_button)
        blink_row.addWidget(fit_button)
        blink_row.addWidget(self.invert_check)
        blink_row.addStretch(1)
        speed_label = QLabel("Speed")
        speed_label.setObjectName("MutedText")
        blink_row.addWidget(speed_label)
        blink_row.addWidget(self.blink_slider)
        layout.addLayout(blink_row)

        self.frame_table.setHorizontalHeaderLabels(["Frame", "Time", "Filter", "Size", "WCS"])
        self.frame_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.frame_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_table.itemSelectionChanged.connect(self._selected_frame_changed)
        _install_table_copy_shortcut(self.frame_table)
        layout.addWidget(QLabel("Frames"))
        layout.addWidget(self.frame_table, 2)

        plate_box = QGroupBox("Astrometry")
        plate_layout = QFormLayout(plate_box)
        plate_layout.addRow("Status", self.plate_status)
        plate_layout.addRow("Center", self.plate_center)
        plate_layout.addRow("Scale", self.plate_scale)
        plate_layout.addRow("File", self.plate_path)
        layout.addWidget(plate_box)

        self.track_table.setHorizontalHeaderLabels(["Show", "ID", "Hits", "Known", "Speed", "PA", "Score"])
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.track_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.track_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.track_table.itemSelectionChanged.connect(self._selected_track_changed)
        _install_table_copy_shortcut(self.track_table)
        layout.addWidget(QLabel("Detected Tracks"))
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.show_full_track)
        overlay_row.addWidget(self.show_known_objects)
        self.show_stars = QCheckBox("Stars")
        self.show_stars.setToolTip("Overlay detected catalog stars from plate solve")
        self.show_stars.toggled.connect(lambda _: self._draw_checked_tracks())
        overlay_row.addWidget(self.show_stars)
        layout.addLayout(overlay_row)
        layout.addWidget(self.track_table, 1)
        track_buttons = QHBoxLayout()
        graph_button = QPushButton("Movement Chart")
        graph_button.clicked.connect(self.open_or_generate_movement_graph)
        cutout_button = QPushButton("Cutouts")
        cutout_button.setToolTip("Show thumbnail cutouts around selected track")
        cutout_button.clicked.connect(self._open_cutout_window)
        track_buttons.addWidget(graph_button)
        track_buttons.addWidget(cutout_button)
        layout.addLayout(track_buttons)

        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log, 1)
        return panel

    def _wire_actions(self) -> None:
        self.blink_slider.valueChanged.connect(lambda _: self._blink_timer.setInterval(self._blink_interval_ms()))
        self._copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        self._copy_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._copy_shortcut.activated.connect(self._copy_active_content)
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._space_shortcut.activated.connect(self.toggle_blink)
        self._previous_frame_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._previous_frame_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._previous_frame_shortcut.activated.connect(lambda: self._step_frame(-1))
        self._next_frame_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._next_frame_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._next_frame_shortcut.activated.connect(lambda: self._step_frame(1))

    def import_images(self) -> None:
        image_suffixes = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_EXTENSIONS))
        fits_suffixes = " ".join(f"*{suffix}" for suffix in sorted(FITS_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import images",
            str(Path.home()),
            f"Supported images ({image_suffixes});;FITS images ({fits_suffixes});;All files (*)",
        )
        if not files:
            return
        self.load_paths(filter_image_files(files))

    def load_paths(self, paths: list[Path]) -> None:
        if not paths:
            self._error("No supported images", "Choose FITS, FIT, FTS, or solved .new files.")
            return
        paths = natural_sorted(paths)
        selected_count = len(paths)
        paths = self._filter_readable_paths(paths, show_message=True)
        if not paths:
            self._error("No readable images", "All selected images were empty, corrupt, or unsupported.")
            return
        common_parent = paths[0].parent
        self.session.input_dir = str(common_parent)
        self.session.output_dir = self.session.output_dir or str(common_parent / "asteroidfinder_output")
        skipped = selected_count - len(paths)
        self._update_path_label(len(paths), skipped)
        self.viewer.clear_cache()
        self._plate_info_cache.clear()
        self.session.frames = [self._frame_info(path) for path in paths]
        self._populate_frames()
        self._log(_import_status_text(len(paths), skipped))
        self._auto_detect_observer_from_headers(paths)
        self._show_frame(0, keep_view=False)

    def _auto_detect_observer_from_headers(self, paths: list[Path]) -> None:
        settings = self.session.settings
        already_have_loc = settings.observer_lat_deg is not None and settings.observer_lon_deg is not None
        already_have_code = bool(settings.observatory_code) and settings.observatory_code != "500"
        if already_have_loc and already_have_code:
            return
        for path in paths:
            try:
                with fits.open(path, memmap=False) as hdul:
                    header = hdul[0].header
            except Exception:
                continue
            lat, lon, elev, mpc = _read_observer_from_header(header)
            if not already_have_loc and lat is not None and lon is not None:
                settings.observer_lat_deg = lat
                settings.observer_lon_deg = lon
                if elev is not None:
                    settings.observer_elev_m = elev
                self._log(
                    f"Detected observer location from FITS header: lat={lat:.4f}°, lon={lon:.4f}°"
                    + (f", elev={elev:.0f}m" if elev is not None else "")
                )
                already_have_loc = True
            if not already_have_code and mpc:
                settings.observatory_code = mpc
                self._log(f"Detected MPC observatory code from FITS header: {mpc}")
                already_have_code = True
            if already_have_loc and already_have_code:
                break

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.session.output_dir or str(Path.home()))
        if folder:
            self.session.output_dir = folder
            self._update_path_label(len(self.session.frames), 0)

    def save_session_file(self) -> None:
        if not self.session.output_dir:
            self.choose_output_folder()
        if not self.session.output_dir:
            return
        self._sync_settings()
        path = save_session(self.session, Path(self.session.output_dir) / "session.json")
        self._log(f"Saved session: {path}")

    def _open_calibration_dialog(self) -> None:
        hot_sigma = _double_spin(self.session.settings.hot_sigma, 0.0, 50.0, 0.5)
        dlg = WorkflowStepDialog("Calibrate Hot Pixels", [("Hot sigma", hot_sigma)], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.hot_sigma = hot_sigma.value()
        self.run_calibration()

    def _open_plate_solve_dialog(self) -> None:
        index_dir = QLineEdit(self.session.settings.index_dir)
        scale_low = _optional_double_spin()
        if self.session.settings.scale_low:
            scale_low.setValue(self.session.settings.scale_low)
        scale_high = _optional_double_spin()
        if self.session.settings.scale_high:
            scale_high.setValue(self.session.settings.scale_high)
        timeout = QSpinBox()
        timeout.setRange(30, 1800)
        timeout.setSingleStep(30)
        timeout.setValue(self.session.settings.solve_timeout if hasattr(self.session.settings, "solve_timeout") else 300)
        timeout.setToolTip("Maximum solve-field time per image, in seconds.")
        dlg = WorkflowStepDialog("Plate Solve", [
            ("Index dir", index_dir),
            ("Scale low", scale_low),
            ("Scale high", scale_high),
            ("Timeout", timeout),
        ], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.index_dir = index_dir.text().strip()
        self.session.settings.scale_low = scale_low.value() if scale_low.value() > 0 else None
        self.session.settings.scale_high = scale_high.value() if scale_high.value() > 0 else None
        self._solve_timeout = timeout.value()
        self.run_plate_solve()

    def _open_alignment_dialog(self) -> None:
        alignment_output = QComboBox()
        alignment_output.addItem("Crop clean overlap", True)
        alignment_output.addItem("Keep reference canvas", False)
        dlg = WorkflowStepDialog("Align Stars", [("Output mode", alignment_output)], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._alignment_crop = bool(alignment_output.currentData())
        self.run_alignment()

    def _open_tracking_dialog(self) -> None:
        detect_sigma = _double_spin(self.session.settings.detect_sigma, 1.0, 20.0, 0.25)
        min_detections = QSpinBox()
        min_detections.setRange(2, 20)
        min_detections.setValue(self.session.settings.min_detections)
        dlg = WorkflowStepDialog("Detect Movers", [
            ("Detect sigma", detect_sigma),
            ("Min detections", min_detections),
        ], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.detect_sigma = detect_sigma.value()
        self.session.settings.min_detections = min_detections.value()
        self.run_tracking()

    def _open_known_objects_dialog(self) -> None:
        observatory = QLineEdit(self.session.settings.observatory_code)
        lat = QLineEdit("" if self.session.settings.observer_lat_deg is None else f"{self.session.settings.observer_lat_deg:.5f}")
        lon = QLineEdit("" if self.session.settings.observer_lon_deg is None else f"{self.session.settings.observer_lon_deg:.5f}")
        elev = QLineEdit("" if self.session.settings.observer_elev_m is None else f"{self.session.settings.observer_elev_m:.0f}")
        lat.setPlaceholderText("e.g. 56.95 (used only if observatory code is 500)")
        lon.setPlaceholderText("e.g. 24.10 (east positive)")
        elev.setPlaceholderText("meters above sea level")
        dlg = WorkflowStepDialog(
            "Identify Known Objects",
            [
                ("Observatory code", observatory),
                ("Observer latitude (deg)", lat),
                ("Observer longitude (deg)", lon),
                ("Observer elevation (m)", elev),
            ],
            self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.observatory_code = observatory.text().strip() or "500"
        self.session.settings.observer_lat_deg = _parse_optional_float(lat.text())
        self.session.settings.observer_lon_deg = _parse_optional_float(lon.text())
        self.session.settings.observer_elev_m = _parse_optional_float(elev.text())
        self.query_known_objects()

    def _open_export_dialog(self) -> None:
        observatory = QLineEdit(self.session.settings.observatory_code)
        dlg = WorkflowStepDialog("Export MPC Observations", [("Observatory code", observatory)], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.observatory_code = observatory.text().strip() or "500"
        self.export_mpc()

    def _open_ades_export_dialog(self) -> None:
        observatory = QLineEdit(self.session.settings.observatory_code)
        dlg = WorkflowStepDialog("Export ADES (PSV)", [("Observatory code", observatory)], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.session.settings.observatory_code = observatory.text().strip() or "500"
        self.export_ades()

    def _open_cutout_window(self) -> None:
        index = self._selected_track_index()
        if index is None or index >= len(self.tracks):
            self._error("No track selected", "Select a detected track first.")
            return
        track = self.tracks[index]
        paths = self.session.frame_paths()
        if not paths:
            self._error("No frames", "Import images first.")
            return
        window = CutoutWindow(track, index, paths, parent=self)
        window.show()

    def run_calibration(self) -> None:
        paths = self._require_paths()
        if not paths:
            return
        out_dir = self._output_dir() / "calibrated"
        self._start_worker(
            "calibration",
            calibrate_images_with_persistent_hot_pixels,
            paths,
            output_dir=out_dir,
            hot_sigma=self.session.settings.hot_sigma,
        )

    def run_plate_solve(self) -> None:
        paths = self._require_paths()
        if not paths:
            return
        self._sync_settings()
        out_dir = self._output_dir() / "solved"
        timeout = getattr(self, "_solve_timeout", 300)

        def solve_all(progress_callback: object | None = None) -> list[Path]:
            solved = []
            total = len(paths)
            for index, path in enumerate(paths, start=1):
                if callable(progress_callback):
                    progress_callback(index - 1, total, f"Solving {Path(path).name}")
                result = solve_image(
                    path,
                    output_dir=out_dir,
                    index_dir=self.session.settings.index_dir or None,
                    scale_low=self.session.settings.scale_low,
                    scale_high=self.session.settings.scale_high,
                    timeout=timeout,
                )
                solved.append(result.solved_fits or result.path)
                if callable(progress_callback):
                    progress_callback(index, total, f"Solved {Path(path).name}")
            return solved

        self._start_worker("plate solve", solve_all)

    def run_alignment(self) -> None:
        paths = self._require_paths(prefer_solved=True)
        if not paths:
            return
        crop_overlap = getattr(self, "_alignment_crop", True)
        aligned_dir = self._output_dir() / "aligned"
        self._clear_generated_aligned_outputs(aligned_dir)
        self._start_worker(
            "alignment",
            align_images,
            paths,
            output_dir=aligned_dir,
            crop_overlap=crop_overlap,
            prefer_translation=False,
        )

    def run_tracking(self) -> None:
        paths = self._require_paths(prefer_aligned=True, require_same_shape=True)
        if not paths:
            return
        aligned_dir = self._output_dir() / "aligned"
        assume_aligned = bool(paths) and all(Path(path).parent == aligned_dir for path in paths)

        def detect(progress_callback: object | None = None) -> list[Track]:
            tracks = track_moving_objects(
                paths,
                sigma=self.session.settings.detect_sigma,
                min_detections=self.session.settings.min_detections,
                assume_aligned=assume_aligned,
                max_sources=500,
            )
            write_tracks_csv(tracks, self._output_dir() / "tracks.csv")
            if callable(progress_callback):
                progress_callback(1, 1, f"Detected {len(tracks)} track(s)")
            return tracks

        self._start_worker(
            "tracking",
            detect,
        )

    def generate_movement_graphs(self) -> None:
        self.write_png_diagnostics()

    def write_png_diagnostics(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before writing PNG diagnostics.")
            return
        self._start_worker("PNG diagnostics", plot_track_diagnostics, self.tracks, self._output_dir() / "diagnostics")

    def open_or_generate_movement_graph(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before opening movement charts.")
            return
        index = self._selected_track_index()
        if index is None:
            index = 0
        self._open_png_diagnostic_window(start_index=index)

    def query_known_objects(self) -> None:
        paths = self._require_paths(prefer_solved=True)
        if not paths:
            return
        location = self.session.settings.observatory_code or "500"
        observer = _build_observer_location(self.session.settings)
        if location == "500" and observer is None:
            self._log(
                "Known-object query is geocentric (loc=500) — set observer lat/lon to remove parallax (~3-5\")."
            )

        def query() -> list[KnownObject]:
            objects = query_known_objects_with_motion_cache(
                paths, location=location, observer_location=observer
            )
            write_known_objects_csv(objects, self._output_dir() / "known_objects.csv")
            return objects

        self._start_worker("known objects", query)

    def open_report_window(self) -> None:
        self._start_worker("report", generate_html_report, self._output_dir())

    def open_known_objects_window(self) -> None:
        if not self.known_objects:
            self._error("No known objects", "Run Identify Known before opening known-object info.")
            return
        window = KnownObjectsWindow(self.known_objects, parent=self)
        window.destroyed.connect(lambda *_: self._forget_known_object_window(window))
        self._known_object_windows.append(window)
        window.show()

    def open_prediction_window(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run Detect Movers before opening prediction tools.")
            return
        index = self._selected_track_index()
        if index is None:
            index = 0
        window = PredictionWindow(
            self.tracks,
            self.session.frames,
            start_index=index,
            known_matches=self.track_known_matches,
            known_objects=self.known_objects,
            matched_known_objects=self.track_known_objects,
            parent=self,
        )
        window.destroyed.connect(lambda *_: self._forget_prediction_window(window))
        self._prediction_windows.append(window)
        window.show()

    def open_submission_window(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run Detect Movers before opening submission tools.")
            return
        window = SubmissionWindow(
            self.tracks,
            self.track_known_matches,
            observatory_code=self.session.settings.observatory_code or "500",
            output_dir=self._output_dir(),
            export_callback=self.export_selected_submission,
            parent=self,
        )
        window.destroyed.connect(lambda *_: self._forget_submission_window(window))
        self._submission_windows.append(window)
        window.show()

    def run_full_workflow(self) -> None:
        paths = self._require_paths()
        if not paths:
            return

        out_dir = self._output_dir()
        calibrated_dir = out_dir / "calibrated"
        aligned_dir = out_dir / "aligned"
        crop_overlap = getattr(self, "_alignment_crop", True)

        def full_pipeline(progress_callback: object | None = None) -> dict[str, object]:
            def progress(done: int, total: int, text: str) -> None:
                if callable(progress_callback):
                    progress_callback(done, total, text)

            progress(0, 3, "Calibrating frames")
            calibration = calibrate_images_with_persistent_hot_pixels(
                paths,
                output_dir=calibrated_dir,
                hot_sigma=self.session.settings.hot_sigma,
            )
            calibrated_paths = [
                calibrated_dir / f"{item.image.path.stem}_calibrated.fits"
                for item in calibration
                if (calibrated_dir / f"{item.image.path.stem}_calibrated.fits").exists()
            ]
            progress(1, 3, "Aligning calibrated frames")
            self._clear_generated_aligned_outputs(aligned_dir)
            aligned_frames = align_images(
                calibrated_paths,
                output_dir=aligned_dir,
                crop_overlap=crop_overlap,
                prefer_translation=False,
            )
            aligned_paths = [
                aligned_dir / f"{frame.image.path.stem}_aligned.fits"
                for frame in aligned_frames
                if (aligned_dir / f"{frame.image.path.stem}_aligned.fits").exists()
            ]
            progress(2, 3, "Detecting movers on aligned frames")
            tracks = track_moving_objects(
                aligned_paths,
                sigma=self.session.settings.detect_sigma,
                min_detections=self.session.settings.min_detections,
                assume_aligned=True,
                max_sources=500,
            )
            write_tracks_csv(tracks, out_dir / "tracks.csv")
            progress(3, 3, f"Detected {len(tracks)} track(s)")
            return {
                "calibration": calibration,
                "aligned_frames": aligned_frames,
                "aligned_paths": aligned_paths,
                "tracks": tracks,
            }

        self._start_worker(
            "basic workflow",
            full_pipeline,
        )

    def export_mpc(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before exporting MPC observations.")
            return
        paths = self._require_paths(prefer_aligned=True)
        if not paths:
            return
        out_dir = self._output_dir()
        mpc_path = out_dir / "detected_track_mpc.txt"
        csv_path = out_dir / "detected_track_observations.csv"
        observatory = self.session.settings.observatory_code or "500"

        def export() -> Path:
            from asteroidfinder.alignment import AlignedFrame

            images = [load_image(path) for path in paths]
            frames = [AlignedFrame(image, image.data, None, None) for image in images]
            return write_detected_track_mpc(
                self.tracks,
                frames,
                mpc_path,
                observatory_code=observatory,
                csv_path=csv_path,
            )

        self._start_worker("MPC export", export)

    def export_ades(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before exporting ADES observations.")
            return
        paths = self._require_paths(prefer_aligned=True)
        if not paths:
            return
        out_dir = self._output_dir()
        ades_path = out_dir / "detected_track_ades.psv"
        observatory = self.session.settings.observatory_code or "500"

        def export() -> Path:
            from asteroidfinder.alignment import AlignedFrame

            images = [load_image(path) for path in paths]
            frames = [AlignedFrame(image, image.data, None, None) for image in images]
            return write_detected_track_ades(
                self.tracks,
                frames,
                ades_path,
                observatory_code=observatory,
            )

        self._start_worker("ADES export", export)

    def export_selected_submission(
        self,
        indices: list[int],
        *,
        observatory_code: str,
        object_prefix: str,
        export_format: str,
    ) -> None:
        selected = [index for index in indices if 0 <= index < len(self.tracks)]
        if not selected:
            self._error("No tracks selected", "Select at least one detected track for export.")
            return
        paths = self._require_paths(prefer_aligned=True)
        if not paths:
            return
        self.session.settings.observatory_code = observatory_code.strip() or "500"
        prefix = object_prefix.strip() or "AF"
        out_dir = self._output_dir()
        selected_tracks = [self.tracks[index] for index in selected]
        track_ids = [index + 1 for index in selected]

        def export() -> list[Path]:
            from asteroidfinder.alignment import AlignedFrame

            images = [load_image(path) for path in paths]
            frames = [AlignedFrame(image, image.data, None, None) for image in images]
            written: list[Path] = []
            if export_format in {"MPC", "Both"}:
                written.append(
                    write_detected_track_mpc(
                        selected_tracks,
                        frames,
                        out_dir / "submission_mpc.txt",
                        observatory_code=self.session.settings.observatory_code,
                        object_prefix=prefix,
                        csv_path=out_dir / "submission_observations.csv",
                        track_ids=track_ids,
                    )
                )
            if export_format in {"ADES", "Both"}:
                written.append(
                    write_detected_track_ades(
                        selected_tracks,
                        frames,
                        out_dir / "submission_ades.psv",
                        observatory_code=self.session.settings.observatory_code,
                        object_prefix=prefix,
                        track_ids=track_ids,
                    )
                )
            return written

        self._start_worker("submission export", export)

    def toggle_blink(self) -> None:
        if self._blink_timer.isActive():
            self._blink_timer.stop()
            self.play_button.setText("▶")
            self.play_button.setToolTip("Start blink")
        else:
            self._blink_timer.start(self._blink_interval_ms())
            self.play_button.setText("⏸")
            self.play_button.setToolTip("Pause blink")

    def _blink_interval_ms(self) -> int:
        speed = self.blink_slider.value()
        slow_ms = 2000
        fast_ms = 90
        fraction = (speed - self.blink_slider.minimum()) / (self.blink_slider.maximum() - self.blink_slider.minimum())
        return int(slow_ms - fraction * (slow_ms - fast_ms))

    def _start_worker(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        total = _initial_progress_total(name, args)
        if total is not None:
            self._worker_progress_totals[name] = total
        worker = FunctionWorker(name, fn, *args, **kwargs)
        worker.signals.started.connect(self._worker_started)
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.failed.connect(self._worker_failed)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.finished.connect(lambda *_: self._forget_worker(worker))
        worker.signals.failed.connect(lambda *_: self._forget_worker(worker))
        self._workers.append(worker)
        self.thread_pool.start(worker)

    def _worker_started(self, name: str) -> None:
        self._worker_started_at[name] = perf_counter()
        self._log(f"Started {name}")
        total = self._worker_progress_totals.get(name)
        if total is None:
            self.progress_label.setText(f"Running {name}")
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("%p%")
        else:
            self.progress_label.setText(f"{name} 0/{total}")
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)

    def _worker_progress(self, name: str, done: int, total: int, text: str) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText(f"Running {name}")
            self.progress_bar.setFormat("%p%")
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(max(0, min(done, total)))
        self.progress_label.setText(f"{name} {done}/{total}")
        self.progress_bar.setFormat(_progress_bar_text(text))

    def _worker_failed(self, name: str, details: str) -> None:
        elapsed = self._worker_elapsed_text(name)
        self._log(f"{name} failed {elapsed}")
        self.progress_label.setText(f"{name} failed {elapsed}")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Failed")
        self.progress_bar.setVisible(True)
        self._error(f"{name} failed", _short_error(details))

    def _worker_finished(self, name: str, result: object) -> None:
        elapsed = self._worker_elapsed_text(name)
        self._log(f"Finished {name} {elapsed}")
        self.progress_label.setText(f"Finished {name} {elapsed}")
        self._mark_workflow_done(name)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("Done")
        self.progress_bar.setVisible(True)
        if name == "calibration":
            results = list(result) if isinstance(result, list) else []
            replaced = sum(int(item.hot_pixel_mask.sum()) for item in results if hasattr(item, "hot_pixel_mask"))
            self._log(f"Hot pixels replaced: {replaced} across {len(results)} frame(s)")
            self._log_hot_pixel_qa()
            calibrated_dir = self._output_dir() / "calibrated"
            calibrated_paths = [
                calibrated_dir / f"{item.image.path.stem}_calibrated.fits"
                for item in results
                if hasattr(item, "image")
            ]
            existing = [path for path in calibrated_paths if path.exists()]
            if existing:
                self.session.frames = [self._frame_info(path) for path in existing]
                self._populate_frames()
                self._show_frame(0, keep_view=False)
        elif name == "tracking":
            self.tracks = list(result)  # type: ignore[arg-type]
            self.visible_track_indices = {0} if self.tracks else set()
            self._match_known_objects_to_tracks()
            self._populate_tracks()
        elif name == "basic workflow":
            data = result if isinstance(result, dict) else {}
            tracks = data.get("tracks", getattr(result, "tracks", []))
            aligned_paths = [Path(path) for path in data.get("aligned_paths", [])] if isinstance(data, dict) else []
            if aligned_paths:
                self.session.frames = [self._frame_info(path) for path in aligned_paths]
                self._populate_frames()
                self._show_frame(0, keep_view=False)
            self.tracks = list(tracks)
            self.visible_track_indices = {0} if self.tracks else set()
            self._match_known_objects_to_tracks()
            self._populate_tracks()
            self._log_hot_pixel_qa()
            self._log_alignment_qa()
            for completed in ("calibration", "alignment", "tracking"):
                self._mark_workflow_done(completed)
        elif name == "alignment":
            aligned_dir = self._output_dir() / "aligned"
            self._log(f"Aligned FITS written to {aligned_dir}")
            self._log_alignment_qa()
            aligned_paths = [aligned_dir / f"{frame.image.path.stem}_aligned.fits" for frame in result] if isinstance(result, list) else []
            existing = [path for path in aligned_paths if path.exists()]
            if existing:
                self.session.frames = [self._frame_info(path) for path in existing]
                self._populate_frames()
                self._show_frame(0, keep_view=False)
        elif name == "plate solve":
            self._log(f"Solved FITS written to {self._output_dir() / 'solved'}")
            solved_paths = [Path(path) for path in result] if isinstance(result, list) else []
            if solved_paths:
                self.session.frames = [self._frame_info(path) for path in solved_paths]
                self._populate_frames()
                self._show_frame(0, keep_view=False)
        elif name == "known objects":
            self.known_objects = list(result)  # type: ignore[arg-type]
            self._match_known_objects_to_tracks()
            self._populate_tracks()
            self._log(f"Known objects predicted: {len(self.known_objects)} from one SkyBoT anchor query")
            self._draw_checked_tracks()
        elif name == "PNG diagnostics":
            paths = [Path(path) for path in result] if isinstance(result, list) else []
            self._log(f"PNG diagnostics written: {len(paths)} file(s) in {self._output_dir() / 'diagnostics'}")
        elif name == "submission export":
            paths = [Path(path) for path in result] if isinstance(result, list) else []
            self._log("Submission files written: " + ", ".join(path.name for path in paths))
        elif name == "report":
            path = Path(result) if isinstance(result, (str, Path)) else self._output_dir() / "report.html"
            self._open_report_dialog(path)

    def _forget_worker(self, worker: FunctionWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def _worker_elapsed_text(self, name: str) -> str:
        self._worker_progress_totals.pop(name, None)
        started = self._worker_started_at.pop(name, None)
        if started is None:
            return ""
        return f"({perf_counter() - started:.2f} seconds)"

    def _populate_frames(self) -> None:
        self.frame_table.setRowCount(len(self.session.frames))
        for row, frame in enumerate(self.session.frames):
            values = [
                frame.name,
                frame.date_obs or "",
                frame.filter_name or "",
                _size_text(frame),
                "yes" if frame.has_wcs else "no",
            ]
            for column, value in enumerate(values):
                self.frame_table.setItem(row, column, QTableWidgetItem(value))

    def _populate_tracks(self) -> None:
        self._updating_track_table = True
        self.track_table.setRowCount(len(self.tracks))
        for row, track in enumerate(self.tracks):
            show = QCheckBox()
            show.setChecked(row in self.visible_track_indices)
            show.setToolTip("Show this detected track on the image")
            show.toggled.connect(lambda checked, index=row: self._set_track_visible(index, checked))
            self.track_table.setCellWidget(row, 0, show)
            known_name = self.track_known_matches.get(row, "")
            values = [
                f"AF{row + 1:04d}" + (f"\n{known_name}" if known_name else ""),
                str(len(track.detections)),
                known_name or "unknown",
                "" if track.angular_rate_arcsec_per_frame is None else f"{track.angular_rate_arcsec_per_frame:.3f}",
                "" if track.position_angle_deg is None else f"{track.position_angle_deg:.1f}",
                f"{track.score:.3f}",
            ]
            for column, value in enumerate(values):
                self.track_table.setItem(row, column + 1, QTableWidgetItem(value))
            if known_name:
                self.track_table.setRowHeight(row, max(self.track_table.rowHeight(row), 46))
        self._updating_track_table = False
        self._log(f"Tracking found {len(self.tracks)} candidate tracks")
        if self.tracks:
            self.track_table.selectRow(0)
            self._draw_selected_track()

    def _selected_frame_changed(self) -> None:
        rows = self.frame_table.selectionModel().selectedRows()
        if rows:
            self._show_frame(rows[0].row())

    def _selected_track_changed(self) -> None:
        rows = self.track_table.selectionModel().selectedRows()
        if not rows:
            return
        index = rows[0].row()
        if index >= len(self.tracks):
            return
        self._draw_selected_track()

    def _draw_selected_track(self) -> None:
        index = self._selected_track_index()
        if index is None:
            return
        self._draw_track_indices([index])

    def _draw_checked_tracks(self) -> None:
        index = self._selected_track_index()
        self.viewer.clear_overlays()
        indices = sorted(self.visible_track_indices)
        if not indices and index is not None:
            indices = [index]
        if indices:
            self._draw_track_indices(indices, clear_first=False)
        if self.show_known_objects.isChecked():
            self._draw_known_object_overlays()
        if self.show_stars.isChecked():
            self._draw_star_overlay()

    def _set_track_visible(self, index: int, checked: bool) -> None:
        if checked:
            self.visible_track_indices.add(index)
        else:
            self.visible_track_indices.discard(index)
        self._draw_checked_tracks()

    def _draw_track_indices(self, indices: list[int], *, clear_first: bool = True, mode: str | None = None) -> None:
        if clear_first:
            self.viewer.clear_overlays()
        colors = ["#38bdf8", "#fbbf24", "#a78bfa", "#34d399", "#fb7185", "#f97316"]
        for index in indices:
            if index >= len(self.tracks):
                continue
            track = self.tracks[index]
            points = [(det.source.x, det.source.y) for det in track.detections]
            current_detection_index = next(
                (det_index for det_index, det in enumerate(track.detections) if det.frame_index == self._current_frame_index),
                None,
            )
            self.viewer.show_track_overlay(
                points,
                f"AF{index + 1:04d}" + (f"\n{self.track_known_matches[index]}" if index in self.track_known_matches else ""),
                mode=mode or ("path" if self.show_full_track.isChecked() else "circle"),
                current_index=current_detection_index,
                color=colors[index % len(colors)],
            )

    def _draw_known_object_overlays(self) -> None:
        colors = ["#fbbf24", "#34d399", "#a78bfa", "#fb7185", "#38bdf8", "#f97316"]
        for index, obj in enumerate(self._known_objects_for_current_frame()):
            label = obj.name or obj.number or obj.object_type or "known"
            self.viewer.show_prediction_overlay(obj.x, obj.y, label, color=colors[index % len(colors)])

    def _draw_star_overlay(self) -> None:
        if not self.session.frames:
            return
        frame = self.session.frames[self._current_frame_index]
        path = Path(frame.path)
        try:
            from asteroidfinder.detection import detect_sources
            from asteroidfinder.io import load_image as _load

            image = _load(path)
            if image.header is None:
                return
            wcs = WCS(image.header)
            if not wcs.has_celestial:
                return
            sources = detect_sources(image.data, sigma=5.0, max_sources=200)
            pen = QPen(QColor("#4a6a8a"), 1.0)
            for src in sources:
                radius_x = max(3.0, min(24.0, src.a * 3.0))
                radius_y = max(3.0, min(24.0, src.b * 3.0))
                marker = QGraphicsEllipseItem(QRectF(src.x - radius_x, src.y - radius_y, radius_x * 2, radius_y * 2))
                marker.setPen(pen)
                marker.setBrush(Qt.BrushStyle.NoBrush)
                marker.setZValue(5)
                self.viewer._scene.addItem(marker)
                self.viewer._overlay_items.append(marker)
        except Exception:
            pass

    def open_selected_movement_graph(self) -> None:
        index = self._selected_track_index()
        if index is None or index >= len(self.tracks):
            self._error("No track", "Select a detected track first.")
            return
        self._open_png_diagnostic_window(start_index=index)

    def open_all_movement_graphs(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before opening movement charts.")
            return
        self._open_png_diagnostic_window()

    def _open_graph_window(self, tracks: list[Track], title: str, *, start_index: int = 0) -> None:
        self._open_png_diagnostic_window(start_index=start_index)

    def _open_png_diagnostic_window(self, *, start_index: int = 0) -> None:
        diagnostics_dir = self._output_dir() / "diagnostics"
        paths = natural_sorted(diagnostics_dir.glob("track_*_diagnostic.png"))
        if len(paths) < len(self.tracks):
            paths = plot_track_diagnostics(self.tracks, diagnostics_dir)
        if not paths:
            self._error("No diagnostics", "Run tracking before opening movement charts.")
            return
        window = ImageDiagnosticWindow(
            "Movement Diagnostics",
            [Path(path) for path in paths],
            start_index=min(max(start_index, 0), len(paths) - 1),
            parent=self,
        )
        window.destroyed.connect(lambda *_: self._forget_diagnostic_window(window))
        self._diagnostic_windows.append(window)
        window.show()

    def _show_track_from_chart(self, index: int, frame_index: int | None = None) -> None:
        if 0 <= index < len(self.tracks):
            if frame_index is not None:
                self._show_frame(frame_index)
            self.track_table.selectRow(index)
            self.viewer.clear_overlays()
            self._draw_track_indices([index], clear_first=False, mode="both")

    def _forget_diagnostic_window(self, window: QDialog) -> None:
        if window in self._diagnostic_windows:
            self._diagnostic_windows.remove(window)

    def _forget_known_object_window(self, window: QDialog) -> None:
        if window in self._known_object_windows:
            self._known_object_windows.remove(window)

    def _forget_prediction_window(self, window: QDialog) -> None:
        if window in self._prediction_windows:
            self._prediction_windows.remove(window)

    def _forget_submission_window(self, window: QDialog) -> None:
        if window in self._submission_windows:
            self._submission_windows.remove(window)

    def _open_report_dialog(self, path: Path) -> None:
        window = ReportWindow(path, parent=self)
        window.destroyed.connect(lambda *_: self._forget_report_window(window))
        self._report_windows.append(window)
        window.show()

    def _forget_report_window(self, window: QDialog) -> None:
        if window in self._report_windows:
            self._report_windows.remove(window)

    def open_hot_pixel_qa(self) -> None:
        path = self._hot_pixel_qa_dir() / "hot_pixel_summary.csv"
        if not path.exists():
            self._error("No hot pixel QA", "Run Clean Hot Pixels first.")
            return
        self._open_qa_window("Hot Pixel QA", path, self._hot_pixel_qa_dir(), mask_controls=True)

    def open_alignment_qa(self) -> None:
        path = self._alignment_qa_path()
        if not path.exists():
            self._error("No alignment QA", "Run Align Stars first.")
            return
        self._open_qa_window("Alignment QA", path, path.parent, mask_controls=False)

    def _open_qa_window(self, title: str, csv_path: Path, folder: Path, *, mask_controls: bool) -> None:
        window = QaWindow(title, csv_path, folder, mask_controls=mask_controls, parent=self)
        window.destroyed.connect(lambda *_: self._forget_qa_window(window))
        self._qa_windows.append(window)
        window.show()

    def _forget_qa_window(self, window: QDialog) -> None:
        if window in self._qa_windows:
            self._qa_windows.remove(window)

    def _hot_pixel_qa_dir(self) -> Path:
        return self._output_dir() / "calibrated" / "hot_pixel_qa"

    def _alignment_qa_path(self) -> Path:
        return self._output_dir() / "aligned" / "alignment_qa.csv"

    def _log_hot_pixel_qa(self) -> None:
        rows = _read_csv_file(self._hot_pixel_qa_dir() / "hot_pixel_summary.csv")
        if not rows:
            return
        persistent_total = rows[0].get("persistent_mask_total", "0")
        transient_total = rows[0].get("transient_union_total", "0")
        worst = max(rows, key=lambda row: int(row.get("transient_hits_in_frame") or 0))
        self._log(
            "Hot pixel QA: "
            f"persistent={persistent_total}, transient union={transient_total}, "
            f"worst transient frame={worst.get('frame', '')} ({worst.get('transient_hits_in_frame', '0')})"
        )
        self._log(f"Hot pixel masks written to {self._hot_pixel_qa_dir()}")

    def _log_alignment_qa(self) -> None:
        rows = _read_csv_file(self._alignment_qa_path())
        measured = [row for row in rows if row.get("rms_error_px")]
        if not measured:
            return
        worst = max(measured, key=lambda row: float(row.get("rms_error_px") or 0.0))
        good = sum(1 for row in measured if row.get("quality") == "good")
        warning = sum(1 for row in measured if row.get("quality") == "warning")
        bad = sum(1 for row in measured if row.get("quality") == "bad")
        self._log(
            "Alignment QA: "
            f"good={good}, warning={warning}, bad={bad}, "
            f"worst={worst.get('frame', '')} rms={worst.get('rms_error_px', '')} px "
            f"matches={worst.get('matched_sources', '')}"
        )

    def _match_known_objects_to_tracks(self, *, radius_px: float = 20.0) -> None:
        self.track_known_matches = {}
        self.track_known_objects = {}
        if not self.tracks or not self.known_objects:
            return
        for track_index, track in enumerate(self.tracks):
            best_name = ""
            best_object: KnownObject | None = None
            best_distance = radius_px
            for det in track.detections:
                if det.frame_index >= len(self.session.frames):
                    continue
                frame = self.session.frames[det.frame_index]
                for obj in _known_objects_matching_frame(self.known_objects, Path(frame.path), det.frame_index):
                    distance = ((det.source.x - obj.x) ** 2 + (det.source.y - obj.y) ** 2) ** 0.5
                    if distance <= best_distance:
                        best_distance = distance
                        best_object = obj
                        best_name = obj.name or obj.number
            if best_name:
                self.track_known_matches[track_index] = best_name
            if best_object is not None:
                self.track_known_objects[track_index] = best_object

    def _known_objects_for_current_frame(self) -> list[KnownObject]:
        if not self.known_objects or not self.session.frames:
            return []
        frame = self.session.frames[self._current_frame_index]
        matched_identities = {_known_identity(obj) for obj in self.track_known_objects.values()}
        objects = _known_objects_matching_frame(self.known_objects, Path(frame.path), self._current_frame_index)
        return [obj for obj in objects if _known_identity(obj) not in matched_identities]

    def _selected_track_index(self) -> int | None:
        rows = self.track_table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return index if 0 <= index < len(self.tracks) else None

    def _show_frame(self, index: int, *, keep_view: bool = True) -> None:
        if not self.session.frames:
            return
        self._current_frame_index = index % len(self.session.frames)
        frame = self.session.frames[self._current_frame_index]
        signals_blocked = self.frame_table.blockSignals(True)
        self.frame_table.selectRow(self._current_frame_index)
        self.frame_table.blockSignals(signals_blocked)
        self._update_plate_info(Path(frame.path))
        self.viewer.clear_overlays()
        if self.viewer.can_show_cached(frame.path):
            self._frame_load_serial += 1
            self._frame_load_in_progress = False
            self.viewer.load_path(frame.path, keep_view=keep_view)
            self._draw_checked_tracks()
            return
        self._load_frame_async(Path(frame.path), keep_view=keep_view)

    def _step_frame(self, delta: int, *, keep_view: bool = True) -> None:
        if self.session.frames:
            self._show_frame(self._current_frame_index + delta, keep_view=keep_view)

    def _advance_blink(self) -> None:
        if self._frame_load_in_progress:
            self._blink_pending_advance = True
            return
        self._step_frame(1)

    def _load_frame_async(self, path: Path, *, keep_view: bool) -> None:
        self._frame_load_serial += 1
        serial = self._frame_load_serial
        self._frame_load_in_progress = True
        self.progress_label.setText(f"Loading {path.name[:32]}")
        worker = FunctionWorker(
            "frame display",
            prepare_display_frame,
            path,
            inverted=self.viewer.inverted,
            percentile_low=self.viewer.percentile_low,
            percentile_high=self.viewer.percentile_high,
        )
        worker.signals.finished.connect(
            lambda _name, result, request_serial=serial, request_keep_view=keep_view: self._frame_display_loaded(
                request_serial,
                result,
                keep_view=request_keep_view,
            )
        )
        worker.signals.failed.connect(
            lambda _name, details, request_serial=serial, request_path=path: self._frame_display_failed(
                request_serial,
                request_path,
                details,
            )
        )
        worker.signals.finished.connect(lambda *_: self._forget_frame_worker(worker))
        worker.signals.failed.connect(lambda *_: self._forget_frame_worker(worker))
        self._frame_workers.append(worker)
        self.frame_thread_pool.start(worker)

    def _frame_display_loaded(self, serial: int, result: object, *, keep_view: bool) -> None:
        if serial != self._frame_load_serial:
            return
        self._frame_load_in_progress = False
        if not isinstance(result, PreparedDisplayFrame):
            self._log("Frame display failed: unexpected loader result")
            return
        self.viewer.show_prepared_frame(result, keep_view=keep_view)
        if not self._workers:
            self.progress_label.setText("Idle")
        self._draw_checked_tracks()
        if self._blink_timer.isActive() and self._blink_pending_advance:
            self._blink_pending_advance = False
            QTimer.singleShot(0, self._advance_blink)

    def _frame_display_failed(self, serial: int, path: Path, details: str) -> None:
        if serial != self._frame_load_serial:
            return
        self._frame_load_in_progress = False
        self._log(f"Frame display failed for {path.name}: {_short_error(details)}")
        if not self._workers:
            self.progress_label.setText("Idle")

    def _forget_frame_worker(self, worker: FunctionWorker) -> None:
        if worker in self._frame_workers:
            self._frame_workers.remove(worker)

    def _require_paths(
        self,
        *,
        prefer_solved: bool = False,
        prefer_aligned: bool = False,
        require_same_shape: bool = False,
    ) -> list[Path]:
        if prefer_aligned:
            aligned = natural_sorted((self._output_dir() / "aligned").glob("*_aligned.fits"))
            if aligned:
                aligned = self._filter_readable_paths(aligned)
                valid_aligned = self._validate_paths(aligned, require_same_shape=require_same_shape, show_error=False)
                if valid_aligned:
                    return valid_aligned
                self._log("Ignoring stale aligned outputs with mixed dimensions; using current frame list instead.")
        if prefer_solved:
            solved = natural_sorted((self._output_dir() / "solved").glob("*.new"))
            if solved:
                solved = self._filter_readable_paths(solved)
                return self._validate_paths(solved, require_same_shape=require_same_shape)
        paths = self.session.frame_paths()
        paths = self._filter_readable_paths(paths, show_message=True)
        if not paths:
            self._error("No frames", "Import FITS images first.")
            return []
        return self._validate_paths(paths, require_same_shape=require_same_shape)

    def _filter_readable_paths(self, paths: list[Path], *, show_message: bool = False) -> list[Path]:
        readable: list[Path] = []
        skipped: list[tuple[Path, str]] = []
        for path in paths:
            try:
                _image_metadata(path)
            except Exception as exc:
                skipped.append((path, str(exc)))
                continue
            readable.append(path)
        for path, detail in skipped:
            self._log(f"Skipped unreadable image: {path.name} ({detail})")
        if skipped and show_message:
            self._log(f"Skipped {len(skipped)} unreadable image(s); continuing with {len(readable)} readable image(s).")
        self._last_skipped_images = len(skipped)
        return readable

    def _validate_paths(self, paths: list[Path], *, require_same_shape: bool, show_error: bool = True) -> list[Path]:
        if not require_same_shape:
            return paths
        groups = self._shape_groups(paths)
        if len(groups) <= 1:
            return paths
        if not show_error:
            return []
        lines = [f"{shape[1]} x {shape[0]}: {len(shape_paths)} image(s)" for shape, shape_paths in groups.items()]
        self._error(
            "Image sizes do not match",
            "This step needs all selected images to have the same dimensions.\n\n"
            + "\n".join(lines)
            + "\n\nImport one matching image set, or plate solve/calibrate matching frames separately.",
        )
        return []

    def _clear_generated_aligned_outputs(self, aligned_dir: Path) -> None:
        aligned_dir.mkdir(parents=True, exist_ok=True)
        for path in aligned_dir.glob("*"):
            if path.is_file() and (path.name.endswith("_aligned.fits") or path.name == "alignment_qa.csv"):
                path.unlink()
            elif path.is_dir() and path.name == "diagnostics":
                shutil.rmtree(path)

    def _shape_groups(self, paths: list[Path]) -> dict[tuple[int, int], list[Path]]:
        groups: dict[tuple[int, int], list[Path]] = {}
        for path in paths:
            try:
                shape, _ = _image_metadata(path)
            except Exception:
                shape = (-1, -1)
            groups.setdefault(shape, []).append(path)
        return groups

    def _output_dir(self) -> Path:
        if not self.session.output_dir:
            if self.session.input_dir:
                self.session.output_dir = str(Path(self.session.input_dir) / "asteroidfinder_output")
            else:
                self.session.output_dir = str(Path.cwd() / "asteroidfinder_output")
        output = Path(self.session.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _sync_settings(self) -> None:
        pass

    def _frame_info(self, path: Path) -> FrameInfo:
        try:
            shape, header = _image_metadata(path)
            has_wcs = False
            if header is not None:
                try:
                    has_wcs = WCS(header).has_celestial
                except Exception:
                    has_wcs = False
            self._plate_info_cache[path] = _plate_info_for(path, shape, header)
            return FrameInfo(
                path=str(path),
                width=int(shape[1]),
                height=int(shape[0]),
                date_obs=None if header is None else _header_observation_time_text(header),
                filter_name=None if header is None else str(header.get("FILTER", "")) or None,
                has_wcs=has_wcs,
            )
        except Exception as exc:
            self._log(f"Could not inspect {path}: {exc}")
            return FrameInfo(path=str(path))

    def _update_path_label(self, count: int, skipped: int) -> None:
        parts = [f"{count} images"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if self.session.input_dir:
            short = Path(self.session.input_dir).name
            parts.append(f"from {short}")
        if self.session.output_dir:
            short = Path(self.session.output_dir).name
            parts.append(f"\u2192 {short}")
        self.path_label.setText("  ".join(parts))
        self.path_label.setToolTip(
            f"Input: {self.session.input_dir or '(none)'}\nOutput: {self.session.output_dir or '(none)'}"
        )

    def _mark_workflow_done(self, name: str) -> None:
        self._workflow_done[name] = True
        action = self._workflow_actions.get(name)
        if action is not None:
            base = action.text().lstrip("\u2713 ")
            action.setText(f"\u2713 {base}")
        menu_action = self._menu_workflow_actions.get(name)
        if menu_action is not None:
            menu_action.setChecked(True)

    def _log(self, text: str) -> None:
        self.log.append(text)

    def _copy_active_content(self) -> None:
        focus = QApplication.focusWidget()
        table = _ancestor_table_widget(focus)
        if table is not None:
            _copy_table_to_clipboard(table)
            return
        if focus is not None and (focus is self.log or self.log.isAncestorOf(focus)):
            _copy_text_edit_to_clipboard(self.log)
            return
        if self.track_table.selectionModel().hasSelection():
            _copy_table_to_clipboard(self.track_table)
        elif self.frame_table.selectionModel().hasSelection():
            _copy_table_to_clipboard(self.frame_table)
        else:
            _copy_text_edit_to_clipboard(self.log)

    def _error(self, title: str, detail: str) -> None:
        QMessageBox.critical(self, title, detail)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About AsteroidFinder",
            "AsteroidFinder desktop preview\n\nReal FITS loading, plate solving, alignment, tracking, and MPC export.",
        )

    def _update_plate_info(self, path: Path) -> None:
        cached = self._plate_info_cache.get(path)
        if cached is not None:
            self._set_plate_info(*cached)
            return
        try:
            shape, header = _image_metadata(path)
            info = _plate_info_for(path, shape, header)
            self._plate_info_cache[path] = info
            self._set_plate_info(*info)
        except Exception as exc:
            self._set_unsolved_plate_info(path, f"Read failed: {exc}")

    def _set_unsolved_plate_info(self, path: Path, status: str) -> None:
        self._set_plate_info(status, "", "", path.name)

    def _set_plate_info(self, status: str, center: str, scale: str, filename: str) -> None:
        self.plate_status.setText(status)
        self.plate_center.setText(center)
        self.plate_scale.setText(scale)
        self.plate_path.setText(filename)


def apply_dark_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0b111a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#dbe7f3"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#070b11"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#101824"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#101824"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#dbe7f3"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#dbe7f3"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#12365a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#eef7ff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2587d9"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget {
            background: #0b111a;
            color: #dbe7f3;
            font-size: 13px;
        }
        #SidePanel {
            background: #0f1722;
            border-left: 1px solid #1d2a3a;
            border-right: 1px solid #1d2a3a;
        }
        QPushButton {
            background: #1663a9;
            color: #f6fbff;
            border: 1px solid #2a8cdc;
            border-radius: 4px;
            min-height: 26px;
            padding: 5px 9px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #1d7ccc;
        }
        QPushButton:pressed {
            background: #0e4f89;
        }
        QPushButton#IconButton {
            color: #ffffff;
            font-size: 16px;
            font-weight: 700;
            padding: 0;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
            background: #070b11;
            border: 1px solid #26364a;
            border-radius: 3px;
            selection-background-color: #1d7ccc;
        }
        QHeaderView::section {
            background: #111d2a;
            color: #dbe7f3;
            border: 0;
            border-right: 1px solid #26364a;
            padding: 5px;
        }
        QToolBar {
            background: #0f1722;
            border-bottom: 1px solid #26364a;
        }
        QSlider::groove:horizontal {
            background: #26364a;
            height: 5px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #2a8cdc;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QGroupBox {
            border: 1px solid #26364a;
            border-radius: 4px;
            margin-top: 8px;
            padding: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #9fc7ea;
        }
        QProgressBar {
            background: #070b11;
            border: 1px solid #26364a;
            border-radius: 3px;
            height: 14px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #2a8cdc;
        }
        #MutedText {
            color: #9fb2c5;
        }
        """
    )


def _double_spin(value: float, low: float, high: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setDecimals(3)
    return spin


def _optional_double_spin() -> QDoubleSpinBox:
    spin = _double_spin(0.0, 0.0, 10000.0, 0.1)
    spin.setSpecialValueText("auto")
    return spin


def _size_text(frame: FrameInfo) -> str:
    if frame.width is None or frame.height is None:
        return ""
    return f"{frame.width} x {frame.height}"


def _import_status_text(imported: int, skipped: int) -> str:
    if skipped:
        return f"{imported} images imported, {skipped} skipped"
    return f"{imported} images imported"


def _icon_button(text: str, tooltip: str) -> QPushButton:
    button = QPushButton(text)
    button.setFixedSize(34, 30)
    button.setToolTip(tooltip)
    button.setObjectName("IconButton")
    return button


def _short_error(details: str) -> str:
    lines = [line.strip() for line in details.splitlines() if line.strip()]
    if not lines:
        return details
    for line in reversed(lines):
        if "Error:" in line or "Exception:" in line or line.startswith("ValueError"):
            return line
    return lines[-1]


def _frame_match_key(path: str | Path) -> str:
    stem = Path(path).stem
    changed = True
    while changed:
        changed = False
        for suffix in ("-solveinput", "_solveinput", "_aligned", "_calibrated"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def _unique_known_frames(objects: list[KnownObject]) -> list[Path]:
    frames: list[Path] = []
    seen: set[Path] = set()
    for obj in objects:
        if obj.frame not in seen:
            seen.add(obj.frame)
            frames.append(obj.frame)
    return frames


def _known_objects_matching_frame(objects: list[KnownObject], frame_path: Path, frame_index: int) -> list[KnownObject]:
    current_key = _frame_match_key(frame_path)
    matches = [obj for obj in objects if _frame_match_key(obj.frame) == current_key]
    if matches:
        return matches

    # Solved files may be generated beside the imported images without
    # replacing the current frame list. In that case, preserve the query
    # order and map frame N in the viewer to frame N in known-object data.
    known_frames = _unique_known_frames(objects)
    if frame_index < len(known_frames):
        fallback_frame = known_frames[frame_index]
        return [obj for obj in objects if obj.frame == fallback_frame]
    return []


def _initial_progress_total(name: str, args: tuple[Any, ...]) -> int | None:
    if not args:
        return None
    try:
        count = len(args[0])
    except TypeError:
        return None
    if count <= 0:
        return None
    if name == "plate solve":
        return count
    if name == "calibration":
        return count
    if name == "alignment":
        return count
    return None


def _progress_bar_text(text: str) -> str:
    if not text:
        return "%p%"
    compact = text
    for prefix in ("Scanning ", "Solving ", "Solved ", "Aligning ", "Aligned ", "Wrote "):
        if compact.startswith(prefix):
            compact = prefix + Path(compact[len(prefix) :]).name
            break
    if len(compact) > 64:
        compact = compact[:28] + "..." + compact[-30:]
    return f"{compact} - %p%"


def _pixel_scale_arcsec(wcs: WCS) -> float | None:
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
        return float(sum(abs(value) for value in scales) / len(scales))
    except Exception:
        return None


def _image_metadata(path: Path) -> tuple[tuple[int, int], fits.Header | None]:
    suffix = path.suffix.lower()
    if suffix in FITS_EXTENSIONS:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError("Empty FITS file")
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            for hdu in hdul:
                header = hdu.header.copy()
                shape = _display_shape_from_header(header)
                if shape is not None:
                    return shape, header
        raise ValueError("No image data found in FITS file")
    if suffix in IMAGE_EXTENSIONS:
        with PILImage.open(path) as image:
            width, height = image.size
        return (int(height), int(width)), None
    raise ValueError(f"Unsupported image format: {path}")


def _display_shape_from_header(header: fits.Header) -> tuple[int, int] | None:
    naxis = int(header.get("NAXIS", 0) or 0)
    if naxis < 2:
        return None
    width = int(header.get("NAXIS1", 0) or 0)
    height = int(header.get("NAXIS2", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    if naxis == 2:
        return (height, width)
    depth = int(header.get("NAXIS3", 0) or 0)
    if depth in {3, 4}:
        return (height, width)
    if width in {3, 4} and depth > 0:
        return (depth, height)
    raise ValueError(
        "Unsupported FITS cube shape from header: "
        f"NAXIS1={width}, NAXIS2={height}, NAXIS3={depth}"
    )


def _plate_info_for(path: Path, shape: tuple[int, ...], header: object | None) -> tuple[str, str, str, str]:
    if header is None:
        return "No FITS header", "", "", path.name
    try:
        wcs = WCS(header)
        if not wcs.has_celestial:
            return "No WCS", "", "", path.name
        height, width = shape
        ra, dec = wcs.pixel_to_world_values(width / 2, height / 2)
        scale = _pixel_scale_arcsec(wcs)
        return (
            "Solved WCS",
            f"RA {float(ra):.6f}, Dec {float(dec):.6f}",
            "unknown" if scale is None else f"{scale:.3f} arcsec/px",
            path.name,
        )
    except Exception as exc:
        return f"WCS read failed: {exc}", "", "", path.name


def _header_observation_time_text(header: fits.Header) -> str | None:
    for key in ("DATE-OBS", "DATEOBS", "DATE"):
        value = header.get(key)
        if value not in {None, ""}:
            return str(value)
    for key in ("OBSJD", "JD", "JULDATE"):
        value = header.get(key)
        if value not in {None, ""}:
            try:
                return Time(float(value), format="jd", scale="utc").isot
            except Exception:
                pass
    for key in ("MJD-OBS", "MJD"):
        value = header.get(key)
        if value not in {None, ""}:
            try:
                return Time(float(value), format="mjd", scale="utc").isot
            except Exception:
                pass
    return None


def _track_summary(track: Track, index: int, known_name: str = "") -> str:
    pixel_speed = (track.velocity_x**2 + track.velocity_y**2) ** 0.5
    sky_speed = "unknown" if track.angular_rate_arcsec_per_frame is None else f"{track.angular_rate_arcsec_per_frame:.3f} arcsec/frame"
    pa = "unknown" if track.position_angle_deg is None else f"{track.position_angle_deg:.1f} deg"
    lines = [
        f"Track AF{index + 1:04d}",
        f"Known match: {known_name or 'none'}",
        f"Detections: {len(track.detections)}",
        f"Pixel velocity: vx={track.velocity_x:.4f}, vy={track.velocity_y:.4f}, speed={pixel_speed:.4f} px/frame",
        f"Sky speed: {sky_speed}",
        f"Position angle: {pa}",
        f"Score: {track.score:.4f}",
        "Frame,x,y,snr,flux",
    ]
    for det in sorted(track.detections, key=lambda item: item.frame_index):
        src = det.source
        lines.append(f"{det.frame_index},{src.x:.3f},{src.y:.3f},{src.snr:.3f},{src.flux:.3f}")
    return "\n".join(lines)


def _copy_table_to_clipboard(table: QTableWidget) -> None:
    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    lines = ["\t".join(headers)]
    selected_rows = sorted({index.row() for index in table.selectedIndexes()})
    rows = selected_rows or list(range(table.rowCount()))
    for row in rows:
        values = []
        for column in range(table.columnCount()):
            widget = table.cellWidget(row, column)
            if isinstance(widget, QCheckBox):
                values.append("yes" if widget.isChecked() else "no")
                continue
            item = table.item(row, column)
            values.append("" if item is None else item.text())
        lines.append("\t".join(values))
    QApplication.clipboard().setText("\n".join(lines))


def _copy_text_edit_to_clipboard(edit: QTextEdit) -> None:
    cursor = edit.textCursor()
    selected = cursor.selectedText().replace("\u2029", "\n")
    QApplication.clipboard().setText(selected or edit.toPlainText())


def _ancestor_table_widget(widget: QWidget | None) -> QTableWidget | None:
    while widget is not None:
        if isinstance(widget, QTableWidget):
            return widget
        parent = widget.parentWidget()
        widget = parent if isinstance(parent, QWidget) else None
    return None


def _install_table_copy_shortcut(table: QTableWidget) -> None:
    shortcut = QShortcut(QKeySequence.StandardKey.Copy, table)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.activated.connect(lambda: _copy_table_to_clipboard(table))
    table._asteroidfinder_copy_shortcut = shortcut  # type: ignore[attr-defined]


def _unique_track_count(rows: list[dict[str, str]]) -> int:
    ids = {row.get("track_id", "").strip() for row in rows if row.get("track_id", "").strip()}
    return len(ids) if ids else len(rows)


def _parse_frame_time(value: str | None) -> Time | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for kwargs in ({"format": "isot", "scale": "utc"}, {"scale": "utc"}):
        try:
            return Time(text, **kwargs)
        except Exception:
            continue
    return None


def _frame_time(frames: list[FrameInfo], frame_index: int) -> Time | None:
    if frame_index < 0 or frame_index >= len(frames):
        return None
    return _parse_frame_time(frames[frame_index].date_obs)


def _prediction_rows(
    track: Track,
    frames: list[FrameInfo],
    *,
    start_after: float,
    duration: float,
    step: float,
) -> list[list[str]]:
    detections = sorted(track.detections, key=lambda item: item.frame_index)
    if len(detections) < 2:
        return [["", "", "", "", "", "", "", "Need at least two detections to predict a path"]]

    times = [_frame_time(frames, det.frame_index) for det in detections]
    has_times = all(time is not None for time in times)
    if has_times:
        base_time = times[0]
        assert base_time is not None
        t_values = np.array([(time.jd - base_time.jd) * 1440.0 for time in times if time is not None], dtype=float)
        last_time = times[-1]
        assert last_time is not None
        unit = "min"
    else:
        base_time = None
        last_time = None
        t_values = np.array([float(det.frame_index) for det in detections], dtype=float)
        unit = "frame"

    xs = np.array([float(det.source.x) for det in detections], dtype=float)
    ys = np.array([float(det.source.y) for det in detections], dtype=float)
    fit_x = np.polyfit(t_values, xs, 1)
    fit_y = np.polyfit(t_values, ys, 1)
    residual = np.hypot(xs - np.polyval(fit_x, t_values), ys - np.polyval(fit_y, t_values))
    rms_px = float(np.sqrt(np.mean(residual**2))) if len(residual) else 0.0

    ra_fit: np.ndarray | None = None
    dec_fit: np.ndarray | None = None
    if all(det.ra_deg is not None and det.dec_deg is not None for det in detections):
        ra_values = np.rad2deg(np.unwrap(np.deg2rad([float(det.ra_deg) for det in detections])))
        dec_values = np.array([float(det.dec_deg) for det in detections], dtype=float)
        ra_fit = np.polyfit(t_values, ra_values, 1)
        dec_fit = np.polyfit(t_values, dec_values, 1)

    step_value = max(float(step), 0.1)
    first_offset = max(float(start_after), 0.0)
    last_offset = max(first_offset, float(duration))
    offsets = np.arange(first_offset, last_offset + step_value * 0.5, step_value)
    last_t = float(t_values[-1])
    rows: list[list[str]] = []
    for offset in offsets:
        t = last_t + float(offset)
        pred_x = float(np.polyval(fit_x, t))
        pred_y = float(np.polyval(fit_y, t))
        if has_times and last_time is not None:
            utc = Time(last_time.jd + float(offset) / 1440.0, format="jd", scale="utc").isot
            offset_text = f"+{offset:.1f} min"
        else:
            utc = f"frame {t:.2f}"
            offset_text = f"+{offset:.1f} frame"
        if ra_fit is not None and dec_fit is not None:
            ra = float(np.polyval(ra_fit, t)) % 360.0
            dec = float(np.polyval(dec_fit, t))
            ra_text = f"{ra:.7f}"
            dec_text = f"{dec:.7f}"
        else:
            ra_text = ""
            dec_text = ""
        note = "time-based linear fit" if has_times else "no DATE-OBS; frame-index prediction only"
        if unit == "min" and offset > 180:
            note += "; orbit fit recommended"
        rows.append([offset_text, utc, ra_text, dec_text, f"{pred_x:.2f}", f"{pred_y:.2f}", f"{rms_px:.3f}", note])
    return rows


def _known_residual_rows(
    track: Track,
    frames: list[FrameInfo],
    known_objects: list[KnownObject],
    matched: KnownObject,
) -> list[list[str]]:
    detections = sorted(track.detections, key=lambda item: item.frame_index)
    if not detections:
        return [["", "", "", "", "", "", "", "", "", "", "", "", "", "No measured detections"]]
    rows: list[list[str]] = []
    target_identity = _known_identity(matched)
    for det in detections:
        if det.frame_index >= len(frames):
            rows.append([str(det.frame_index), "", "", "", f"{det.source.x:.2f}", f"{det.source.y:.2f}", "", "", "", "", "", "", "", "No frame"])
            continue
        frame = frames[det.frame_index]
        predicted = _matching_known_object_for_detection(
            known_objects,
            target_identity,
            Path(frame.path),
            det.frame_index,
            det.source.x,
            det.source.y,
        )
        utc = frame.date_obs or ""
        actual_ra = "" if det.ra_deg is None else f"{det.ra_deg:.7f}"
        actual_dec = "" if det.dec_deg is None else f"{det.dec_deg:.7f}"
        if predicted is None:
            rows.append(
                [
                    str(det.frame_index),
                    utc,
                    "",
                    "",
                    f"{det.source.x:.2f}",
                    f"{det.source.y:.2f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    actual_ra,
                    actual_dec,
                ]
            )
            continue
        delta_px = float(((det.source.x - predicted.x) ** 2 + (det.source.y - predicted.y) ** 2) ** 0.5)
        if det.ra_deg is None or det.dec_deg is None:
            ra_residual = ""
            dec_residual = ""
            delta_arcsec = ""
        else:
            ra_arcsec, dec_arcsec, total_arcsec = _signed_sky_residual_arcsec(
                predicted.ra_deg,
                predicted.dec_deg,
                det.ra_deg,
                det.dec_deg,
            )
            ra_residual = f"{ra_arcsec:.2f}"
            dec_residual = f"{dec_arcsec:.2f}"
            delta_arcsec = f"{total_arcsec:.2f}"
        rows.append(
            [
                str(det.frame_index),
                utc,
                f"{predicted.x:.2f}",
                f"{predicted.y:.2f}",
                f"{det.source.x:.2f}",
                f"{det.source.y:.2f}",
                f"{delta_px:.2f}",
                ra_residual,
                dec_residual,
                delta_arcsec,
                f"{predicted.ra_deg:.7f}",
                f"{predicted.dec_deg:.7f}",
                actual_ra,
                actual_dec,
            ]
        )
    return rows


def _known_identity(obj: KnownObject) -> str:
    for value in (obj.number, obj.name):
        text = str(value).strip()
        if text:
            return text.lower()
    return f"{obj.ra_deg:.6f}:{obj.dec_deg:.6f}"


def _matching_known_object_for_detection(
    objects: list[KnownObject],
    target_identity: str,
    frame_path: Path,
    frame_index: int,
    measured_x: float,
    measured_y: float,
) -> KnownObject | None:
    candidates = [
        obj
        for obj in _known_objects_matching_frame(objects, frame_path, frame_index)
        if _known_identity(obj) == target_identity
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda obj: (obj.x - measured_x) ** 2 + (obj.y - measured_y) ** 2)


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    return _signed_sky_residual_arcsec(ra1, dec1, ra2, dec2)[2]


def _signed_sky_residual_arcsec(
    predicted_ra: float,
    predicted_dec: float,
    actual_ra: float,
    actual_dec: float,
) -> tuple[float, float, float]:
    dra = ((actual_ra - predicted_ra + 180.0) % 360.0) - 180.0
    dec_mid = np.deg2rad((predicted_dec + actual_dec) / 2.0)
    ra_arcsec = dra * np.cos(dec_mid) * 3600.0
    dec_arcsec = (actual_dec - predicted_dec) * 3600.0
    total_arcsec = float(np.hypot(ra_arcsec, dec_arcsec))
    return float(ra_arcsec), float(dec_arcsec), total_arcsec


def _linear_fit(values: list[tuple[float, float]]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, values[0][1] if values else 0.0
    n = float(len(values))
    sum_x = sum(x for x, _ in values)
    sum_y = sum(y for _, y in values)
    sum_xx = sum(x * x for x, _ in values)
    sum_xy = sum(x * y for x, y in values)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _range_with_padding(values: list[float], padding_fraction: float = 0.12) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        return low - 1.0, high + 1.0
    pad = (high - low) * padding_fraction
    return low - pad, high + pad



class TrackChartWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.track: Track | None = None
        self.track_index = 0
        self.known_name = ""
        self.frame_size: tuple[int, int] | None = None
        self.selected_offset = 0
        self.on_selected_offset: Any | None = None
        self.zoom = 100
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor="#0b111a")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.axes: list[Any] = []
        self._points: list[tuple[float, float, float, float]] = []
        self.canvas.mpl_connect("button_press_event", self._pick_detection)
        self.canvas.mpl_connect("motion_notify_event", self._hover_detection)

    def set_track(
        self,
        track: Track,
        track_index: int,
        known_name: str,
        frame_size: tuple[int, int] | None,
        selected_offset: int | None,
    ) -> None:
        self.track = track
        self.track_index = track_index
        self.known_name = known_name
        self.frame_size = frame_size
        self.selected_offset = selected_offset or 0
        self._draw()

    def set_zoom(self, zoom: int) -> None:
        self.zoom = 100

    def _draw(self) -> None:
        self.figure.clear()
        self.figure.set_facecolor("#0b111a")
        if self.track is None:
            self.canvas.draw_idle()
            return
        detections = sorted(self.track.detections, key=lambda item: item.frame_index)
        self._points = [(float(det.frame_index), float(det.source.x), float(det.source.y), float(det.source.snr)) for det in detections]
        if not self._points:
            self.canvas.draw_idle()
            return

        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)
        ax_path = self.figure.add_subplot(grid[:, 0])
        ax_x = self.figure.add_subplot(grid[0, 1])
        ax_y = self.figure.add_subplot(grid[1, 1], sharex=ax_x)
        self.axes = [ax_path, ax_x, ax_y]
        self._style_axis(ax_path)
        self._style_axis(ax_x)
        self._style_axis(ax_y)

        frames = [point[0] for point in self._points]
        xs = [point[1] for point in self._points]
        ys = [point[2] for point in self._points]
        fit_x = _linear_fit(list(zip(frames, xs)))
        fit_y = _linear_fit(list(zip(frames, ys)))
        fit_xs = [fit_x[0] * frame + fit_x[1] for frame in frames]
        fit_ys = [fit_y[0] * frame + fit_y[1] for frame in frames]

        ax_path.plot(xs, ys, color="#38bdf8", linewidth=2.0, marker="o", markersize=5, label="detections")
        ax_path.plot(fit_xs, fit_ys, color="#fbbf24", linestyle="--", linewidth=1.5, label="linear fit")
        selected = self._selected_point()
        ax_path.scatter([selected[1]], [selected[2]], s=130, facecolors="none", edgecolors="#ff4f8b", linewidths=2.2, zorder=6)
        for frame, x, y, _snr in self._points:
            ax_path.annotate(f"F{int(frame)}", (x, y), xytext=(5, 5), textcoords="offset points", color="#dbe7f3", fontsize=8)
        if self.frame_size is not None:
            ax_path.set_xlim(0, self.frame_size[0])
            ax_path.set_ylim(0, self.frame_size[1])
        else:
            ax_path.set_xlim(*_range_with_padding(xs))
            ax_path.set_ylim(*_range_with_padding(ys))
        ax_path.set_aspect("equal", adjustable="box")
        ax_path.set_title("Full-frame motion", color="#e8f3ff")
        ax_path.set_xlabel("x pixel", color="#9fb2c5")
        ax_path.set_ylabel("y pixel", color="#9fb2c5")
        ax_path.legend(facecolor="#0f1722", edgecolor="#26364a", labelcolor="#dbe7f3", fontsize=8)

        self._plot_series(ax_x, frames, xs, fit_x, "X position", "#34d399")
        self._plot_series(ax_y, frames, ys, fit_y, "Y position", "#a78bfa")
        self.figure.suptitle(self._title(), color="#e8f3ff", fontsize=12)
        self.canvas.draw_idle()

    def _plot_series(self, axis: Any, frames: list[float], values: list[float], fit: tuple[float, float], title: str, color: str) -> None:
        fitted = [fit[0] * frame + fit[1] for frame in frames]
        axis.plot(frames, values, color=color, linewidth=1.8, marker="o", markersize=4)
        axis.plot(frames, fitted, color="#fbbf24", linestyle="--", linewidth=1.3)
        selected = self._selected_point()
        value = selected[1] if title.startswith("X") else selected[2]
        axis.axvline(selected[0], color="#ff4f8b", linewidth=1.1, alpha=0.75)
        axis.scatter([selected[0]], [value], s=80, facecolors="none", edgecolors="#ff4f8b", linewidths=1.8, zorder=6)
        residuals = [value - (fit[0] * frame + fit[1]) for frame, value in zip(frames, values)]
        rms = (sum(value * value for value in residuals) / len(residuals)) ** 0.5 if residuals else 0.0
        axis.set_title(f"{title}   slope {fit[0]:.4f} px/frame   RMS {rms:.3f}", color="#e8f3ff", fontsize=10)
        axis.set_xlabel("frame", color="#9fb2c5")
        axis.set_ylabel("pixel", color="#9fb2c5")
        axis.set_xlim(*_range_with_padding(frames, 0.08))
        axis.set_ylim(*_range_with_padding(values))

    def _style_axis(self, axis: Any) -> None:
        axis.set_facecolor("#0f1722")
        axis.tick_params(colors="#9fb2c5", labelsize=8)
        axis.grid(True, color="#26364a", linewidth=0.7, alpha=0.8)
        for spine in axis.spines.values():
            spine.set_color("#3b4c62")

    def _selected_point(self) -> tuple[float, float, float, float]:
        index = min(max(self.selected_offset, 0), len(self._points) - 1)
        return self._points[index]

    def _pick_detection(self, event: Any) -> None:
        self._select_from_event(event)

    def _hover_detection(self, event: Any) -> None:
        self._select_from_event(event)

    def _select_from_event(self, event: Any) -> None:
        if event.inaxes is None or not self._points:
            return
        candidates: list[tuple[float, int]] = []
        for index, point in enumerate(self._points):
            if event.inaxes == self.axes[0]:
                x_data, y_data = point[1], point[2]
            elif event.inaxes == self.axes[1]:
                x_data, y_data = point[0], point[1]
            elif event.inaxes == self.axes[2]:
                x_data, y_data = point[0], point[2]
            else:
                continue
            screen_x, screen_y = event.inaxes.transData.transform((x_data, y_data))
            candidates.append(((screen_x - event.x) ** 2 + (screen_y - event.y) ** 2, index))
        if not candidates:
            return
        distance_sq, index = min(candidates)
        if distance_sq <= 14**2 and index != self.selected_offset and self.on_selected_offset is not None:
            self.on_selected_offset(index)

    def _title(self) -> str:
        if self.track is None:
            return ""
        pixel_speed = (self.track.velocity_x**2 + self.track.velocity_y**2) ** 0.5
        known = f"  matched {self.known_name}" if self.known_name else ""
        return (
            f"AF{self.track_index + 1:04d}{known}   "
            f"{len(self.track.detections)} detections   "
            f"speed {pixel_speed:.3f} px/frame   "
            f"score {self.track.score:.3f}"
        )


class DiagnosticWindow(QDialog):
    def __init__(
        self,
        tracks: list[Track],
        title: str,
        *,
        start_index: int = 0,
        known_matches: dict[int, str] | None = None,
        frame_size: tuple[int, int] | None = None,
        show_track_callback: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tracks = tracks
        self.index = min(max(start_index, 0), max(len(tracks) - 1, 0))
        self.known_matches = known_matches or {}
        self.frame_size = frame_size
        self.show_track_callback = show_track_callback
        self.selected_offset = 0
        self._updating_measurements = False
        self.zoom = 100
        self.setWindowTitle(title)
        self.resize(1220, 860)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        previous = _icon_button("◀", "Previous track")
        previous.clicked.connect(lambda: self._step(-1))
        next_button = _icon_button("▶", "Next track")
        next_button.clicked.connect(lambda: self._step(1))
        show_button = QPushButton("Show On Image")
        show_button.clicked.connect(self._show_on_image)
        copy_button = QPushButton("Copy Summary")
        copy_button.clicked.connect(self._copy_summary)
        self.caption = QLabel()
        self.caption.setObjectName("MutedText")
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(show_button)
        controls.addWidget(copy_button)
        controls.addWidget(self.caption, 1)
        layout.addLayout(controls)

        scrub = QHBoxLayout()
        prev_detection = _icon_button("◁", "Previous detection")
        prev_detection.clicked.connect(lambda: self._step_detection(-1))
        next_detection = _icon_button("▷", "Next detection")
        next_detection.clicked.connect(lambda: self._step_detection(1))
        self.frame_label = QLabel()
        self.frame_label.setObjectName("MutedText")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._set_detection_offset)
        scrub.addWidget(prev_detection)
        scrub.addWidget(next_detection)
        scrub.addWidget(QLabel("Detection"))
        scrub.addWidget(self.frame_slider, 1)
        scrub.addWidget(self.frame_label)
        layout.addLayout(scrub)

        self.chart = TrackChartWidget()
        self.chart.on_selected_offset = self._chart_selected_offset
        self.chart.on_zoom_changed = self._chart_zoom_changed
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.chart, 1)

        self.measurement_table = QTableWidget(0, 8)
        self.measurement_table.setHorizontalHeaderLabels(["Frame", "X", "Y", "SNR", "Flux", "RA", "Dec", "Residual"])
        self.measurement_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.measurement_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.measurement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.measurement_table.itemSelectionChanged.connect(self._measurement_selection_changed)
        _install_table_copy_shortcut(self.measurement_table)

        self.tabs = QTabWidget()
        self.tabs.addTab(chart_tab, "Motion")
        self.tabs.addTab(self.measurement_table, "Measurements")
        layout.addWidget(self.tabs, 1)
        self._render()

    def _step(self, delta: int) -> None:
        if not self.tracks:
            return
        self.index = (self.index + delta) % len(self.tracks)
        self.selected_offset = 0
        self._render()

    def _step_detection(self, delta: int) -> None:
        detections = self._detections()
        if not detections:
            return
        self.selected_offset = (self.selected_offset + delta) % len(detections)
        self._sync_detection_controls()
        self._render_chart()

    def _set_detection_offset(self, value: int) -> None:
        if self._updating_measurements:
            return
        self.selected_offset = value
        self._sync_detection_controls()
        self._render_chart()

    def _set_zoom(self, value: int) -> None:
        self.zoom = value
        self._render_chart()

    def _chart_selected_offset(self, value: int) -> None:
        self.selected_offset = value
        self._sync_detection_controls()
        self._render_chart()

    def _chart_zoom_changed(self, value: int) -> None:
        self.zoom = value

    def _render(self) -> None:
        if not self.tracks:
            self.caption.setText("No tracks")
            return
        track = self.tracks[self.index]
        self.caption.setText(f"{self.index + 1} / {len(self.tracks)}  AF{self.index + 1:04d}")
        detections = self._detections()
        self.selected_offset = min(self.selected_offset, max(len(detections) - 1, 0))
        self._populate_measurements()
        self._sync_detection_controls()
        self._render_chart()

    def _render_chart(self) -> None:
        if not self.tracks:
            return
        track = self.tracks[self.index]
        self.chart.set_zoom(self.zoom)
        self.chart.set_track(track, self.index, self.known_matches.get(self.index, ""), self.frame_size, self.selected_offset)

    def _show_on_image(self) -> None:
        if self.show_track_callback is not None:
            detection = self._selected_detection()
            self.show_track_callback(self.index, None if detection is None else detection.frame_index)

    def _copy_summary(self) -> None:
        if not self.tracks:
            return
        track = self.tracks[self.index]
        QApplication.clipboard().setText(_track_summary(track, self.index, self.known_matches.get(self.index, "")))

    def _detections(self) -> list[Any]:
        if not self.tracks:
            return []
        return sorted(self.tracks[self.index].detections, key=lambda item: item.frame_index)

    def _selected_detection(self) -> Any | None:
        detections = self._detections()
        if not detections:
            return None
        return detections[min(self.selected_offset, len(detections) - 1)]

    def _sync_detection_controls(self) -> None:
        detections = self._detections()
        self._updating_measurements = True
        self.frame_slider.setEnabled(bool(detections))
        self.frame_slider.setRange(0, max(len(detections) - 1, 0))
        self.frame_slider.setValue(min(self.selected_offset, max(len(detections) - 1, 0)))
        detection = self._selected_detection()
        if detection is None:
            self.frame_label.setText("No detections")
        else:
            src = detection.source
            self.frame_label.setText(f"frame {detection.frame_index}   x {src.x:.2f}   y {src.y:.2f}   SNR {src.snr:.1f}")
            self.measurement_table.selectRow(self.selected_offset)
        self._updating_measurements = False

    def _populate_measurements(self) -> None:
        detections = self._detections()
        fit_x = _linear_fit([(float(det.frame_index), float(det.source.x)) for det in detections])
        fit_y = _linear_fit([(float(det.frame_index), float(det.source.y)) for det in detections])
        self._updating_measurements = True
        self.measurement_table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            src = det.source
            fitted_x = fit_x[0] * det.frame_index + fit_x[1]
            fitted_y = fit_y[0] * det.frame_index + fit_y[1]
            residual = ((src.x - fitted_x) ** 2 + (src.y - fitted_y) ** 2) ** 0.5
            values = [
                str(det.frame_index),
                f"{src.x:.3f}",
                f"{src.y:.3f}",
                f"{src.snr:.2f}",
                f"{src.flux:.1f}",
                "" if det.ra_deg is None else f"{det.ra_deg:.7f}",
                "" if det.dec_deg is None else f"{det.dec_deg:.7f}",
                f"{residual:.3f}",
            ]
            for column, value in enumerate(values):
                self.measurement_table.setItem(row, column, QTableWidgetItem(value))
        self._updating_measurements = False

    def _measurement_selection_changed(self) -> None:
        if self._updating_measurements:
            return
        rows = self.measurement_table.selectionModel().selectedRows()
        if not rows:
            return
        self.selected_offset = rows[0].row()
        self._sync_detection_controls()
        self._render_chart()


class ImageDiagnosticWindow(QDialog):
    def __init__(self, title: str, image_paths: list[Path], *, start_index: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_paths = image_paths
        self.index = min(max(start_index, 0), max(len(image_paths) - 1, 0))
        self._pixmap = QPixmap()
        self.setWindowTitle(title)
        self.resize(960, 680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        controls = QHBoxLayout()
        previous = _icon_button("◀", "Previous diagnostic")
        previous.clicked.connect(lambda: self._step(-1))
        next_button = _icon_button("▶", "Next diagnostic")
        next_button.clicked.connect(lambda: self._step(1))
        open_file = QPushButton("Open PNG")
        open_file.clicked.connect(self._open_current)
        self.caption = QLabel()
        self.caption.setObjectName("MutedText")
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(open_file)
        controls.addWidget(self.caption, 1)
        layout.addLayout(controls)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #0b111a;")
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label, 1)
        self._load_current()

    def _step(self, delta: int) -> None:
        if not self.image_paths:
            return
        self.index = (self.index + delta) % len(self.image_paths)
        self._load_current()

    def _open_current(self) -> None:
        if self.image_paths:
            webbrowser.open(self.image_paths[self.index].resolve().as_uri())

    def _load_current(self) -> None:
        if not self.image_paths:
            return
        path = self.image_paths[self.index]
        self._pixmap = QPixmap(str(path))
        self.caption.setText(f"{self.index + 1} / {len(self.image_paths)}   {path.name}")
        self._update_preview()

    def _update_preview(self) -> None:
        if self._pixmap.isNull():
            self.image_label.setText("No preview")
            return
        available = self.image_label.size()
        scaled = self._pixmap.scaled(
            max(64, available.width()),
            max(64, available.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_preview()


class CutoutWindow(QDialog):
    """Shows thumbnail cutouts from each frame around a track detection."""

    CUTOUT_RADIUS = 40

    def __init__(self, track: Track, track_index: int, frame_paths: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.track = track
        self.track_index = track_index
        self.frame_paths = frame_paths
        self.setWindowTitle(f"Track AF{track_index + 1:04d} Cutouts")
        self.resize(820, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel(
            f"AF{track_index + 1:04d}   {len(track.detections)} detections   "
            f"score {track.score:.3f}"
        )
        header.setStyleSheet("font-size: 14px; font-weight: 600; color: #e8f3ff;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: #0b111a;")
        container = QWidget()
        self._grid = QHBoxLayout(container)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        self._load_cutouts()

    def _load_cutouts(self) -> None:
        detections = sorted(self.track.detections, key=lambda d: d.frame_index)
        for det in detections:
            if det.frame_index >= len(self.frame_paths):
                continue
            path = self.frame_paths[det.frame_index]
            try:
                image = load_image(path)
                data = image.data
                cx, cy = int(det.source.x), int(det.source.y)
                r = self.CUTOUT_RADIUS
                y0 = max(0, cy - r)
                y1 = min(data.shape[0], cy + r)
                x0 = max(0, cx - r)
                x1 = min(data.shape[1], cx + r)
                cutout = data[y0:y1, x0:x1]
                from asteroidfinder.io import stretch_to_uint8

                pixels = stretch_to_uint8(cutout.astype(float), percentile=(1.0, 99.0))
                h, w = pixels.shape
                qimg = QImage(np.ascontiguousarray(pixels).data, w, h, w, QImage.Format.Format_Grayscale8).copy()
                pixmap = QPixmap.fromImage(qimg).scaled(
                    160, 160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                card = QVBoxLayout()
                thumb = QLabel()
                thumb.setPixmap(pixmap)
                thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb.setStyleSheet("border: 1px solid #26364a; border-radius: 4px;")
                info = QLabel(
                    f"F{det.frame_index}  ({det.source.x:.1f}, {det.source.y:.1f})\n"
                    f"SNR {det.source.snr:.1f}"
                )
                info.setObjectName("MutedText")
                info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                card.addWidget(thumb)
                card.addWidget(info)
                frame = QWidget()
                frame.setLayout(card)
                self._grid.addWidget(frame)
            except Exception:
                error_label = QLabel(f"F{det.frame_index}\n(load error)")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setObjectName("MutedText")
                self._grid.addWidget(error_label)
        self._grid.addStretch(1)


class PredictionWindow(QDialog):
    """Prediction/residual table for measured moving-object tracks."""

    def __init__(
        self,
        tracks: list[Track],
        frames: list[FrameInfo],
        *,
        start_index: int = 0,
        known_matches: dict[int, str] | None = None,
        known_objects: list[KnownObject] | None = None,
        matched_known_objects: dict[int, KnownObject] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tracks = tracks
        self.frames = frames
        self.known_matches = known_matches or {}
        self.known_objects = known_objects or []
        self.matched_known_objects = matched_known_objects or {}
        self.index = min(max(start_index, 0), max(len(tracks) - 1, 0))
        self.setWindowTitle("Prediction And Residuals")
        self.resize(1040, 640)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        previous = _icon_button("◀", "Previous track")
        previous.clicked.connect(lambda: self._step_track(-1))
        next_button = _icon_button("▶", "Next track")
        next_button.clicked.connect(lambda: self._step_track(1))
        self.caption = QLabel()
        self.caption.setObjectName("MutedText")
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(self.caption, 1)
        layout.addLayout(controls)

        settings = QHBoxLayout()
        self.start_after = _double_spin(5.0, 0.0, 10080.0, 1.0)
        self.start_after.setDecimals(1)
        self.duration = _double_spin(120.0, 1.0, 10080.0, 5.0)
        self.duration.setDecimals(1)
        self.step_minutes = _double_spin(10.0, 0.5, 1440.0, 1.0)
        self.step_minutes.setDecimals(1)
        for widget in (self.start_after, self.duration, self.step_minutes):
            widget.valueChanged.connect(lambda *_: self._render())
        settings.addWidget(QLabel("Start after last"))
        settings.addWidget(self.start_after)
        settings.addWidget(QLabel("Duration"))
        settings.addWidget(self.duration)
        settings.addWidget(QLabel("Step"))
        settings.addWidget(self.step_minutes)
        settings.addStretch(1)
        layout.addLayout(settings)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("MutedText")
        layout.addWidget(self.note)

        self.table = QTableWidget(0, 8)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        self.copy_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.copy_shortcut.activated.connect(lambda: _copy_table_to_clipboard(self.table))
        layout.addWidget(self.table, 1)
        self._render()

    def _step_track(self, delta: int) -> None:
        if not self.tracks:
            return
        self.index = (self.index + delta) % len(self.tracks)
        self._render()

    def _render(self) -> None:
        if not self.tracks:
            self.caption.setText("No tracks")
            self.table.setRowCount(0)
            return
        track = self.tracks[self.index]
        known = self.known_matches.get(self.index, "")
        known_text = f"   known match: {known}" if known else "   unknown candidate"
        speed = "unknown" if track.angular_rate_arcsec_per_frame is None else f"{track.angular_rate_arcsec_per_frame:.3f} arcsec/frame"
        self.caption.setText(
            f"AF{self.index + 1:04d}{known_text}   {len(track.detections)} detections   speed {speed}   score {track.score:.3f}"
        )
        matched = self.matched_known_objects.get(self.index)
        if matched is not None:
            self.note.setText(
                "Known-object O-C residuals: observed measured centroids minus calculated SkyBoT/MPC positions. "
                "Signed RA/Dec residuals show the error direction; total delta is the scalar separation."
            )
            self.table.setColumnCount(14)
            self.table.setHorizontalHeaderLabels(
                [
                    "Frame",
                    "UTC",
                    "Pred X",
                    "Pred Y",
                    "Actual X",
                    "Actual Y",
                    "Delta px",
                    "O-C RA arcsec",
                    "O-C Dec arcsec",
                    "Delta arcsec",
                    "Pred RA",
                    "Pred Dec",
                    "Actual RA",
                    "Actual Dec",
                ]
            )
            stretch_column = 1
            rows = _known_residual_rows(track, self.frames, self.known_objects, matched)
        else:
            self.note.setText(
                "Unknown-candidate short-arc prediction from measured centroids. Good for near-term pointing checks; "
                "not a real orbit solution for next-night recovery."
            )
            self.table.setColumnCount(8)
            self.table.setHorizontalHeaderLabels(["Offset", "UTC / frame", "RA deg", "Dec deg", "X", "Y", "RMS px", "Notes"])
            stretch_column = 7
            rows = _prediction_rows(
                track,
                self.frames,
                start_after=float(self.start_after.value()),
                duration=float(self.duration.value()),
                step=float(self.step_minutes.value()),
            )
        for column in range(self.table.columnCount()):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))


class SubmissionWindow(QDialog):
    """Review detected tracks and write measured MPC/ADES submission drafts."""

    def __init__(
        self,
        tracks: list[Track],
        known_matches: dict[int, str],
        *,
        observatory_code: str,
        output_dir: Path,
        export_callback: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tracks = tracks
        self.known_matches = known_matches
        self.output_dir = output_dir
        self.export_callback = export_callback
        self.checkboxes: list[QCheckBox] = []
        self.setWindowTitle("Submission Review")
        self.resize(1080, 680)
        layout = QVBoxLayout(self)

        note = QLabel(
            "Exports measured track centroids from your frames. Known matches are labels for review; "
            "the exported positions are measured, not copied from predictions."
        )
        note.setWordWrap(True)
        note.setObjectName("MutedText")
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.observatory = QLineEdit(observatory_code or "500")
        self.observatory.setMaximumWidth(90)
        self.prefix = QLineEdit("AF")
        self.prefix.setMaximumWidth(70)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Both", "MPC", "ADES"])
        controls.addWidget(QLabel("Observatory"))
        controls.addWidget(self.observatory)
        controls.addWidget(QLabel("Track prefix"))
        controls.addWidget(self.prefix)
        controls.addWidget(QLabel("Format"))
        controls.addWidget(self.format_combo)
        controls.addStretch(1)
        select_all = QPushButton("Select All")
        select_all.clicked.connect(lambda: self._set_all(True))
        select_none = QPushButton("Select None")
        select_none.clicked.connect(lambda: self._set_all(False))
        controls.addWidget(select_all)
        controls.addWidget(select_none)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Use", "Track", "Known match", "Hits", "Speed", "PA", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        _install_table_copy_shortcut(self.table)
        layout.addWidget(self.table, 1)
        self._populate()

        buttons = QHBoxLayout()
        copy_button = QPushButton("Copy Table")
        copy_button.clicked.connect(lambda: _copy_table_to_clipboard(self.table))
        open_folder = QPushButton("Open Output Folder")
        open_folder.clicked.connect(lambda: webbrowser.open(self.output_dir.resolve().as_uri()))
        export_button = QPushButton("Export Selected")
        export_button.clicked.connect(self._export_selected)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(copy_button)
        buttons.addWidget(open_folder)
        buttons.addStretch(1)
        buttons.addWidget(export_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _populate(self) -> None:
        self.table.setRowCount(len(self.tracks))
        self.checkboxes = []
        for row, track in enumerate(self.tracks):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setToolTip("Include this measured track in the export")
            self.checkboxes.append(checkbox)
            self.table.setCellWidget(row, 0, checkbox)
            speed = "" if track.angular_rate_arcsec_per_frame is None else f"{track.angular_rate_arcsec_per_frame:.3f}"
            pa = "" if track.position_angle_deg is None else f"{track.position_angle_deg:.1f}"
            values = [
                f"AF{row + 1:04d}",
                self.known_matches.get(row, "unknown"),
                str(len(track.detections)),
                speed,
                pa,
                f"{track.score:.3f}",
            ]
            for column, value in enumerate(values, start=1):
                self.table.setItem(row, column, QTableWidgetItem(value))

    def _set_all(self, checked: bool) -> None:
        for checkbox in self.checkboxes:
            checkbox.setChecked(checked)

    def _selected_indices(self) -> list[int]:
        return [index for index, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

    def _export_selected(self) -> None:
        self.export_callback(
            self._selected_indices(),
            observatory_code=self.observatory.text().strip() or "500",
            object_prefix=self.prefix.text().strip() or "AF",
            export_format=self.format_combo.currentText(),
        )


class KnownObjectsWindow(QDialog):
    def __init__(self, objects: list[KnownObject], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.objects = objects
        self.setWindowTitle("Known Objects")
        self.resize(1180, 720)
        layout = QVBoxLayout(self)
        summary = QLabel(f"{len(objects)} known object prediction(s)")
        summary.setObjectName("MutedText")
        layout.addWidget(summary)
        rows = [_known_object_row(obj) for obj in objects]
        self.table = _table_widget(rows)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        controls.addStretch(1)
        controls.addWidget(close_button)
        layout.addLayout(controls)
        self.table.setFocus()


class ReportWindow(QDialog):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("AsteroidFinder Report")
        self.resize(1020, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("Session Report")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #e8f3ff;")
        folder_label = QLabel(str(path.parent))
        folder_label.setObjectName("MutedText")
        folder_label.setWordWrap(True)
        header.addWidget(title)
        header.addStretch(1)
        open_browser = QPushButton("Open In Browser")
        open_browser.clicked.connect(lambda: webbrowser.open(self.path.resolve().as_uri()))
        open_folder = QPushButton("Open Folder")
        open_folder.clicked.connect(lambda: webbrowser.open(self.path.parent.resolve().as_uri()))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._load_report)
        header.addWidget(refresh)
        header.addWidget(open_folder)
        header.addWidget(open_browser)
        layout.addLayout(header)
        layout.addWidget(folder_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._load_report()

    def _load_report(self) -> None:
        self.tabs.clear()
        out_dir = self.path.parent
        self.tabs.addTab(self._summary_tab(out_dir), "Overview")
        for title, filename in [
            ("Tracks", "tracks.csv"),
            ("Known Objects", "known_objects.csv"),
            ("Alignment", "alignment_report.csv"),
            ("Hot Pixels", "hot_pixel_report.csv"),
            ("Photometry", "known_object_forced_photometry.csv"),
        ]:
            rows = _read_csv_file(out_dir / filename)
            if rows:
                self.tabs.addTab(_table_widget(rows), f"{title} ({len(rows)})")
        self.tabs.addTab(self._files_tab(out_dir), "Files")

    def _summary_tab(self, out_dir: Path) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        tracks = _read_csv_file(out_dir / "tracks.csv")
        known = _read_csv_file(out_dir / "known_objects.csv")
        alignment = _read_csv_file(out_dir / "alignment_report.csv")
        hotpix = _read_csv_file(out_dir / "hot_pixel_report.csv")

        metrics = [
            ("Detected tracks", str(_unique_track_count(tracks)), "#38bdf8"),
            ("Known objects", str(len(known)), "#34d399"),
            ("Aligned frames", str(len(alignment)), "#a78bfa"),
            ("Hot pixel frames", str(len(hotpix)), "#fbbf24"),
        ]

        grid = QHBoxLayout()
        for label, value, color in metrics:
            card = QVBoxLayout()
            num = QLabel(value)
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {color};")
            desc = QLabel(label)
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc.setObjectName("MutedText")
            card.addWidget(num)
            card.addWidget(desc)
            frame = QGroupBox()
            frame.setLayout(card)
            grid.addWidget(frame)
        layout.addLayout(grid)
        layout.addStretch(1)
        return widget

    def _files_tab(self, out_dir: Path) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        files = QListWidget()
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                files.addItem(str(path.relative_to(out_dir)))
        files.itemDoubleClicked.connect(lambda item: webbrowser.open((out_dir / item.text()).resolve().as_uri()))
        layout.addWidget(files, 1)
        return widget


class QaWindow(QDialog):
    def __init__(
        self,
        title: str,
        csv_path: Path,
        folder: Path,
        *,
        mask_controls: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.csv_path = csv_path
        self.folder = folder
        self.mask_controls = mask_controls
        self.image_paths: list[Path] = []
        self.image_list = QListWidget()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #0b111a;")
        self._current_pixmap = QPixmap()
        self.setWindowTitle(title)
        self.resize(980, 640)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        open_folder = QPushButton("Open Folder")
        open_folder.clicked.connect(lambda: webbrowser.open(self.folder.resolve().as_uri()))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._load)
        self.invert_mask = QCheckBox("Invert")
        self.invert_mask.toggled.connect(self._select_current_image)
        self.color_mask = QCheckBox("Color")
        self.color_mask.setChecked(True)
        self.color_mask.toggled.connect(self._select_current_image)
        controls.addWidget(QLabel(str(csv_path)))
        controls.addStretch(1)
        if mask_controls:
            controls.addWidget(self.color_mask)
            controls.addWidget(self.invert_mask)
        controls.addWidget(refresh)
        controls.addWidget(open_folder)
        layout.addLayout(controls)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._load()

    def _load(self) -> None:
        self.tabs.clear()
        self.image_paths = sorted(self.folder.glob("*.png"))
        if self.image_paths:
            self.tabs.addTab(self._image_tab(), "Masks")
        rows = _read_csv_file(self.csv_path)
        if rows:
            self.tabs.addTab(_table_widget(rows), "Summary")
        files = QListWidget()
        for path in sorted(self.folder.rglob("*")):
            if path.is_file():
                files.addItem(str(path.relative_to(self.folder)))
        files.itemDoubleClicked.connect(lambda item: webbrowser.open((self.folder / item.text()).resolve().as_uri()))
        self.tabs.addTab(files, "Files")

    def _image_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(300)
        for path in self.image_paths:
            self.image_list.addItem(path.name)
        self.image_list.currentRowChanged.connect(self._select_image)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setBackgroundRole(QPalette.ColorRole.Base)
        scroll.setWidget(self.image_label)
        self.mask_scroll = scroll
        layout.addWidget(self.image_list)
        layout.addWidget(scroll, 1)
        self.image_list.setCurrentRow(0)
        return widget

    def _select_image(self, row: int) -> None:
        if row < 0 or row >= len(self.image_paths):
            return
        self._current_pixmap = _mask_preview_pixmap(self.image_paths[row], invert=self.invert_mask.isChecked(), color=self.color_mask.isChecked())
        self._update_image_preview()

    def _select_current_image(self) -> None:
        self._select_image(self.image_list.currentRow())

    def _update_image_preview(self) -> None:
        if self._current_pixmap.isNull():
            self.image_label.setText("No preview")
            return
        viewport = self.mask_scroll.viewport().size() if hasattr(self, "mask_scroll") else self.size()
        available_w = max(64, viewport.width() - 16)
        available_h = max(64, viewport.height() - 16)
        scaled = self._current_pixmap.scaled(
            available_w,
            available_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_image_preview()


def _mask_preview_pixmap(path: Path, *, invert: bool, color: bool) -> QPixmap:
    image = PILImage.open(path).convert("RGB")
    if color:
        array = np.asarray(image, dtype=np.uint8)
        gray = array.mean(axis=2)
        colored = np.full(array.shape, 255, dtype=np.uint8)
        colored[gray < 64] = (255, 24, 72)
        colored[(gray >= 64) & (gray < 210)] = (36, 160, 255)
        image = PILImage.fromarray(colored, mode="RGB")
    if invert:
        image = ImageOps.invert(image)
    raw = image.tobytes("raw", "RGB")
    qimage = QImage(raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _read_csv_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _known_object_row(obj: KnownObject) -> dict[str, str]:
    return {
        "Frame": obj.frame.name,
        "Date": obj.date_obs,
        "Number": obj.number,
        "Name": obj.name,
        "Type": obj.object_type,
        "RA deg": _format_optional_float(obj.ra_deg, 7),
        "Dec deg": _format_optional_float(obj.dec_deg, 7),
        "X": _format_optional_float(obj.x, 2),
        "Y": _format_optional_float(obj.y, 2),
        "V mag": _format_optional_float(obj.v_mag, 2),
        "Center arcsec": _format_optional_float(obj.center_distance_arcsec, 1),
        "RA rate arcsec/hr": _format_optional_float(obj.ra_rate_arcsec_per_hour, 3),
        "Dec rate arcsec/hr": _format_optional_float(obj.dec_rate_arcsec_per_hour, 3),
    }


def _format_optional_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _table_widget(rows: list[dict[str, str]]) -> QTableWidget:
    headers = list(rows[0].keys()) if rows else []
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    for row_index, row in enumerate(rows):
        for col_index, header in enumerate(headers):
            table.setItem(row_index, col_index, QTableWidgetItem(row.get(header, "")))
    _install_table_copy_shortcut(table)
    return table
