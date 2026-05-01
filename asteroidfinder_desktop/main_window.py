from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import webbrowser

from astropy.wcs import WCS
from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QAction, QPalette, QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
from asteroidfinder.known_objects import KnownObject, query_known_objects_for_frames, write_known_objects_csv
from asteroidfinder.mpc import write_detected_track_mpc
from asteroidfinder.platesolve import solve_image
from asteroidfinder.report import generate_html_report
from asteroidfinder.tracking import Track, track_moving_objects
from asteroidfinder.workflow import run_asteroid_workflow

from .session import FITS_EXTENSIONS, FrameInfo, SessionState, filter_image_files, save_session
from .viewer import FitsViewer
from .workers import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AsteroidFinder")
        self.session = SessionState()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)
        self.thread_pool.setStackSize(32 * 1024 * 1024)
        self._workers: list[FunctionWorker] = []
        self._diagnostic_windows: list[QDialog] = []
        self._report_windows: list[QDialog] = []
        self.tracks: list[Track] = []
        self.diagnostic_paths: list[Path] = []
        self.known_objects: list[KnownObject] = []
        self.track_known_matches: dict[int, str] = {}
        self._pending_open_graph = False
        self._updating_track_table = False
        self._current_frame_index = 0
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._advance_blink)

        self.viewer = FitsViewer()
        self.frame_table = QTableWidget(0, 5)
        self.track_table = QTableWidget(0, 9)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("MutedText")
        self.plate_status = QLabel("No frame selected")
        self.plate_center = QLabel("")
        self.plate_scale = QLabel("")
        self.plate_path = QLabel("")
        self.plate_path.setWordWrap(True)

        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.output_edit = QLineEdit()
        self.index_edit = QLineEdit(self.session.settings.index_dir)
        self.scale_low = _optional_double_spin()
        self.scale_high = _optional_double_spin()
        self.solve_timeout = QSpinBox()
        self.solve_timeout.setRange(30, 1800)
        self.solve_timeout.setSingleStep(30)
        self.solve_timeout.setValue(300)
        self.solve_timeout.setToolTip("Maximum astrometry.net solve-field time per image, in seconds.")
        self.hot_sigma = _double_spin(self.session.settings.hot_sigma, 0.0, 50.0, 0.5)
        self.detect_sigma = _double_spin(self.session.settings.detect_sigma, 1.0, 20.0, 0.25)
        self.min_detections = QSpinBox()
        self.min_detections.setRange(2, 20)
        self.min_detections.setValue(self.session.settings.min_detections)
        self.observatory_code = QLineEdit(self.session.settings.observatory_code)
        self.alignment_output = QComboBox()
        self.alignment_output.addItem("Crop clean overlap", True)
        self.alignment_output.addItem("Keep reference canvas", False)
        self.alignment_output.setToolTip("Crop removes black no-data borders after alignment. Keep reference canvas is better for debugging.")
        self.track_overlay_mode = QComboBox()
        self.track_overlay_mode.addItem("Circle on current frame", "circle")
        self.track_overlay_mode.addItem("Motion path", "path")
        self.track_overlay_mode.addItem("Circle + path", "both")
        self.track_overlay_mode.setToolTip("Circle marks the selected track on the current frame. Path shows the full fitted motion trail.")
        self.track_overlay_mode.currentIndexChanged.connect(lambda _: self._draw_selected_track())
        self.invert_check = QCheckBox("Invert")
        self.blink_slider = QSlider(Qt.Orientation.Horizontal)
        self.blink_slider.setRange(1, 20)
        self.blink_slider.setValue(15)
        self.blink_slider.setToolTip("Blink speed: left is slower, right is faster")

        self._build_ui()
        self._wire_actions()

    def _build_ui(self) -> None:
        open_action = QAction("Import Images", self)
        open_action.triggered.connect(self.import_images)
        self.addAction(open_action)
        self._build_menus(open_action)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        toolbar.addAction("Save Session", self.save_session_file)

        root = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._workflow_panel())
        root.addWidget(self._center_panel())
        root.addWidget(self._analysis_panel())
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 0)
        self.setCentralWidget(root)

    def _workflow_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("SidePanel")
        layout = QVBoxLayout(panel)

        browse_output = QPushButton("Output")
        browse_output.clicked.connect(self.choose_output_folder)

        layout.addWidget(QLabel("Input Images"))
        layout.addWidget(self.input_edit)

        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_output)
        layout.addWidget(QLabel("Output Folder"))
        layout.addLayout(out_row)

        settings = QFormLayout()
        settings.addRow("Index dir", self.index_edit)
        settings.addRow("Scale low", self.scale_low)
        settings.addRow("Scale high", self.scale_high)
        settings.addRow("Solve timeout", self.solve_timeout)
        settings.addRow("Hot sigma", self.hot_sigma)
        settings.addRow("Detect sigma", self.detect_sigma)
        settings.addRow("Min detections", self.min_detections)
        settings.addRow("Observatory", self.observatory_code)
        settings.addRow("Alignment output", self.alignment_output)
        layout.addLayout(settings)

        for label, handler in [
            ("Calibrate Hot Pixels", self.run_calibration),
            ("Plate Solve", self.run_plate_solve),
            ("Align Frames", self.run_alignment),
            ("Track Moving Objects", self.run_tracking),
            ("Query Known Objects", self.query_known_objects),
            ("Open Report", self.open_report_window),
            ("Full Basic Run", self.run_full_workflow),
            ("Export MPC", self.export_mpc),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            layout.addWidget(button)

        layout.addStretch(1)
        return panel

    def _center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        controls = QHBoxLayout()
        import_button = QPushButton("Import Images")
        import_button.clicked.connect(self.import_images)
        previous = _icon_button("⏮", "Previous frame")
        previous.clicked.connect(lambda: self._step_frame(-1))
        self.play_button = _icon_button("▶", "Start blink")
        self.play_button.clicked.connect(self.toggle_blink)
        next_button = _icon_button("⏭", "Next frame")
        next_button.clicked.connect(lambda: self._step_frame(1))
        fit_button = _icon_button("⛶", "Fit image to view")
        fit_button.clicked.connect(self.viewer.fit_to_view)
        self.invert_check.toggled.connect(self.viewer.set_inverted)
        controls.addWidget(import_button)
        controls.addStretch(1)
        controls.addWidget(previous)
        controls.addWidget(self.play_button)
        controls.addWidget(next_button)
        controls.addWidget(fit_button)
        controls.addWidget(self.invert_check)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.blink_slider)
        layout.addLayout(controls)
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
        file_menu.addAction("Save Session", self.save_session_file)
        file_menu.addSeparator()
        file_menu.addAction("Quit", QApplication.instance().quit)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Fit Image To View", self.viewer.fit_to_view)
        invert_action = QAction("Invert Image", self)
        invert_action.setCheckable(True)
        invert_action.toggled.connect(self.invert_check.setChecked)
        self.invert_check.toggled.connect(invert_action.setChecked)
        view_menu.addAction(invert_action)

        run_menu = self.menuBar().addMenu("Run")
        run_menu.addAction("Calibrate Hot Pixels", self.run_calibration)
        run_menu.addAction("Plate Solve", self.run_plate_solve)
        run_menu.addAction("Align Frames", self.run_alignment)
        run_menu.addAction("Track Moving Objects", self.run_tracking)
        run_menu.addAction("Query Known Objects", self.query_known_objects)
        run_menu.addAction("Generate Movement Graphs", self.generate_movement_graphs)
        run_menu.addAction("Open Report", self.open_report_window)
        run_menu.addAction("Full Basic Run", self.run_full_workflow)
        run_menu.addSeparator()
        run_menu.addAction("Export MPC", self.export_mpc)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About AsteroidFinder", self._show_about)

    def _analysis_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("SidePanel")
        layout = QVBoxLayout(panel)

        self.frame_table.setHorizontalHeaderLabels(["Frame", "Time", "Filter", "Size", "WCS"])
        self.frame_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.frame_table.itemSelectionChanged.connect(self._selected_frame_changed)
        layout.addWidget(QLabel("Frames"))
        layout.addWidget(self.frame_table, 2)

        plate_box = QGroupBox("Plate Solve")
        plate_layout = QFormLayout(plate_box)
        plate_layout.addRow("Status", self.plate_status)
        plate_layout.addRow("Center", self.plate_center)
        plate_layout.addRow("Pixel scale", self.plate_scale)
        plate_layout.addRow("File", self.plate_path)
        layout.addWidget(plate_box)

        self.track_table.setHorizontalHeaderLabels(["Show", "ID", "Known", "Hits", "Vx", "Vy", "Sky speed", "PA", "Score"])
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.track_table.itemSelectionChanged.connect(self._selected_track_changed)
        self.track_table.itemChanged.connect(self._track_item_changed)
        layout.addWidget(QLabel("Detected Tracks"))
        layout.addWidget(self.track_overlay_mode)
        layout.addWidget(self.track_table, 1)
        graph_row = QHBoxLayout()
        graph_button = QPushButton("Open Movement Graph")
        graph_button.clicked.connect(self.open_or_generate_movement_graph)
        graph_row.addWidget(graph_button)
        layout.addLayout(graph_row)

        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log, 1)
        return panel

    def _wire_actions(self) -> None:
        self.blink_slider.valueChanged.connect(lambda _: self._blink_timer.setInterval(self._blink_interval_ms()))

    def import_images(self) -> None:
        suffixes = " ".join(f"*{suffix}" for suffix in sorted(FITS_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import FITS images",
            str(Path.home()),
            f"FITS images ({suffixes});;All files (*)",
        )
        if not files:
            return
        self.load_paths(filter_image_files(files))

    def load_paths(self, paths: list[Path]) -> None:
        if not paths:
            self._error("No supported images", "Choose FITS, FIT, FTS, or solved .new files.")
            return
        paths = sorted(paths)
        common_parent = paths[0].parent
        self.session.input_dir = str(common_parent)
        self.session.output_dir = self.session.output_dir or str(common_parent / "asteroidfinder_output")
        self.input_edit.setText(f"{len(paths)} images imported")
        self.output_edit.setText(self.session.output_dir)
        self.session.frames = [self._frame_info(path) for path in paths]
        self._populate_frames()
        self._log(f"Imported {len(paths)} images")
        self._show_frame(0, keep_view=False)

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_edit.text() or str(Path.home()))
        if folder:
            self.session.output_dir = folder
            self.output_edit.setText(folder)

    def save_session_file(self) -> None:
        if not self.session.output_dir:
            self.choose_output_folder()
        if not self.session.output_dir:
            return
        self._sync_settings()
        path = save_session(self.session, Path(self.session.output_dir) / "session.json")
        self._log(f"Saved session: {path}")

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
            hot_sigma=self.hot_sigma.value(),
        )

    def run_plate_solve(self) -> None:
        paths = self._require_paths()
        if not paths:
            return
        self._sync_settings()
        out_dir = self._output_dir() / "solved"

        def solve_all() -> list[Path]:
            solved = []
            for path in paths:
                result = solve_image(
                    path,
                    output_dir=out_dir,
                    index_dir=self.session.settings.index_dir or None,
                    scale_low=self.session.settings.scale_low,
                    scale_high=self.session.settings.scale_high,
                    timeout=self.solve_timeout.value(),
                )
                solved.append(result.solved_fits or result.path)
            return solved

        self._start_worker("plate solve", solve_all)

    def run_alignment(self) -> None:
        paths = self._require_paths(prefer_solved=True)
        if not paths:
            return
        crop_overlap = bool(self.alignment_output.currentData())
        self._start_worker("alignment", align_images, paths, output_dir=self._output_dir() / "aligned", crop_overlap=crop_overlap)

    def run_tracking(self) -> None:
        paths = self._require_paths(prefer_aligned=True, require_same_shape=True)
        if not paths:
            return
        self._start_worker("tracking", track_moving_objects, paths, sigma=self.detect_sigma.value(), min_detections=self.min_detections.value())

    def generate_movement_graphs(self) -> None:
        if not self.tracks:
            self._error("No tracks", "Run tracking before generating movement graphs.")
            return
        self._start_worker("movement graphs", plot_track_diagnostics, self.tracks, self._output_dir() / "diagnostics")

    def open_or_generate_movement_graph(self) -> None:
        if self.diagnostic_paths:
            self.open_selected_movement_graph()
            return
        self._pending_open_graph = True
        self.generate_movement_graphs()

    def query_known_objects(self) -> None:
        paths = self._require_paths(prefer_solved=True)
        if not paths:
            return

        def query() -> list[KnownObject]:
            objects = query_known_objects_for_frames(paths, location=self.observatory_code.text().strip() or "500")
            write_known_objects_csv(objects, self._output_dir() / "known_objects.csv")
            return objects

        self._start_worker("known objects", query)

    def open_report_window(self) -> None:
        self._start_worker("report", generate_html_report, self._output_dir())

    def run_full_workflow(self) -> None:
        paths = self._require_paths()
        if not paths:
            return
        self._start_worker(
            "basic workflow",
            run_asteroid_workflow,
            paths,
            output_dir=self._output_dir(),
            hot_sigma=self.hot_sigma.value(),
            sigma=self.detect_sigma.value(),
            min_detections=self.min_detections.value(),
            crop_overlap=bool(self.alignment_output.currentData()),
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

        def export() -> Path:
            from asteroidfinder.alignment import AlignedFrame

            frames = [AlignedFrame(load_image(path), load_image(path).data, None, None) for path in paths]
            return write_detected_track_mpc(
                self.tracks,
                frames,
                mpc_path,
                observatory_code=self.observatory_code.text().strip() or "500",
                csv_path=csv_path,
            )

        self._start_worker("MPC export", export)

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
        worker = FunctionWorker(name, fn, *args, **kwargs)
        worker.signals.started.connect(self._worker_started)
        worker.signals.failed.connect(self._worker_failed)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.finished.connect(lambda *_: self._forget_worker(worker))
        worker.signals.failed.connect(lambda *_: self._forget_worker(worker))
        self._workers.append(worker)
        self.thread_pool.start(worker)

    def _worker_started(self, name: str) -> None:
        self._log(f"Started {name}")
        self.progress_label.setText(f"Running {name}")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

    def _worker_failed(self, name: str, details: str) -> None:
        self._log(f"{name} failed")
        self.progress_label.setText(f"{name} failed")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._error(f"{name} failed", _short_error(details))

    def _worker_finished(self, name: str, result: object) -> None:
        self._log(f"Finished {name}")
        self.progress_label.setText(f"Finished {name}")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setVisible(True)
        if name == "tracking":
            self.tracks = list(result)  # type: ignore[arg-type]
            self._match_known_objects_to_tracks()
            self._populate_tracks()
            if self.tracks:
                self.generate_movement_graphs()
        elif name == "basic workflow":
            tracks = getattr(result, "tracks", [])
            self.tracks = list(tracks)
            self._match_known_objects_to_tracks()
            self._populate_tracks()
            if self.tracks:
                self.generate_movement_graphs()
        elif name == "alignment":
            aligned_dir = self._output_dir() / "aligned"
            mode = self.alignment_output.currentText()
            self._log(f"Aligned FITS written to {aligned_dir} ({mode})")
        elif name == "plate solve":
            self._log(f"Solved FITS written to {self._output_dir() / 'solved'}")
            solved_paths = [Path(path) for path in result] if isinstance(result, list) else []
            if solved_paths:
                self.session.frames = [self._frame_info(path) for path in solved_paths]
                self._populate_frames()
                self._show_frame(0, keep_view=False)
        elif name == "movement graphs":
            self.diagnostic_paths = [Path(path) for path in result] if isinstance(result, list) else []
            self._log(f"Movement graphs written to {self._output_dir() / 'diagnostics'}")
            if self._pending_open_graph:
                self._pending_open_graph = False
                self.open_selected_movement_graph()
        elif name == "known objects":
            self.known_objects = list(result)  # type: ignore[arg-type]
            self._match_known_objects_to_tracks()
            self._populate_tracks()
            self._log(f"Known objects found: {len(self.known_objects)}")
        elif name == "report":
            path = Path(result) if isinstance(result, (str, Path)) else self._output_dir() / "report.html"
            self._open_report_dialog(path)

    def _forget_worker(self, worker: FunctionWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

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
            show_item = QTableWidgetItem()
            show_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            show_item.setCheckState(Qt.CheckState.Checked if row == 0 else Qt.CheckState.Unchecked)
            self.track_table.setItem(row, 0, show_item)
            values = [
                str(row + 1),
                self.track_known_matches.get(row, ""),
                str(len(track.detections)),
                f"{track.velocity_x:.3f}",
                f"{track.velocity_y:.3f}",
                "" if track.angular_rate_arcsec_per_frame is None else f"{track.angular_rate_arcsec_per_frame:.3f}",
                "" if track.position_angle_deg is None else f"{track.position_angle_deg:.1f}",
                f"{track.score:.3f}",
            ]
            for column, value in enumerate(values):
                self.track_table.setItem(row, column + 1, QTableWidgetItem(value))
        self._updating_track_table = False
        self._log(f"Tracking found {len(self.tracks)} candidate tracks")
        if self.tracks:
            self.track_table.selectRow(0)
            self._draw_checked_tracks()

    def _track_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_track_table or item.column() != 0:
            return
        self._draw_checked_tracks()

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
        if not self._checked_track_indices():
            self._draw_selected_track()

    def _draw_selected_track(self) -> None:
        index = self._selected_track_index()
        if index is None:
            return
        self._draw_track_indices([index])

    def _draw_checked_tracks(self) -> None:
        indices = self._checked_track_indices()
        self.viewer.clear_overlays()
        if indices:
            self._draw_track_indices(indices, clear_first=False)

    def _draw_track_indices(self, indices: list[int], *, clear_first: bool = True) -> None:
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
                f"AF{index + 1:04d}",
                mode=str(self.track_overlay_mode.currentData()),
                current_index=current_detection_index,
                color=colors[index % len(colors)],
            )

    def _checked_track_indices(self) -> list[int]:
        indices = []
        for row in range(self.track_table.rowCount()):
            item = self.track_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def open_selected_movement_graph(self) -> None:
        index = self._selected_track_index()
        if index is None or index >= len(self.diagnostic_paths):
            self._error("No graph", "Generate movement graphs, then select a detected track.")
            return
        self._open_graph_window([self.diagnostic_paths[index]], f"Track {index + 1} Movement")

    def open_all_movement_graphs(self) -> None:
        if not self.diagnostic_paths:
            self._error("No graphs", "Generate movement graphs after tracking.")
            return
        self._open_graph_window(self.diagnostic_paths, "All Movement Graphs")

    def _open_graph_window(self, paths: list[Path], title: str) -> None:
        window = DiagnosticWindow(paths, title, parent=self)
        window.destroyed.connect(lambda *_: self._forget_diagnostic_window(window))
        self._diagnostic_windows.append(window)
        window.show()

    def _forget_diagnostic_window(self, window: QDialog) -> None:
        if window in self._diagnostic_windows:
            self._diagnostic_windows.remove(window)

    def _open_report_dialog(self, path: Path) -> None:
        window = ReportWindow(path, parent=self)
        window.destroyed.connect(lambda *_: self._forget_report_window(window))
        self._report_windows.append(window)
        window.show()

    def _forget_report_window(self, window: QDialog) -> None:
        if window in self._report_windows:
            self._report_windows.remove(window)

    def _match_known_objects_to_tracks(self, *, radius_px: float = 20.0) -> None:
        self.track_known_matches = {}
        if not self.tracks or not self.known_objects:
            return
        objects_by_frame: dict[int, list[KnownObject]] = {}
        frame_names = [Path(frame.path).name for frame in self.session.frames]
        for obj in self.known_objects:
            try:
                frame_index = frame_names.index(Path(obj.frame).name)
            except ValueError:
                frame_index = 0
            objects_by_frame.setdefault(frame_index, []).append(obj)
        for track_index, track in enumerate(self.tracks):
            best_name = ""
            best_distance = radius_px
            for det in track.detections:
                for obj in objects_by_frame.get(det.frame_index, []):
                    distance = ((det.source.x - obj.x) ** 2 + (det.source.y - obj.y) ** 2) ** 0.5
                    if distance <= best_distance:
                        best_distance = distance
                        best_name = obj.name or obj.number
            if best_name:
                self.track_known_matches[track_index] = best_name

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
        self.viewer.load_path(frame.path, keep_view=keep_view)
        self.frame_table.selectRow(self._current_frame_index)
        self._update_plate_info(Path(frame.path))
        self._draw_checked_tracks()

    def _step_frame(self, delta: int) -> None:
        if self.session.frames:
            self._show_frame(self._current_frame_index + delta)

    def _advance_blink(self) -> None:
        self._step_frame(1)

    def _require_paths(
        self,
        *,
        prefer_solved: bool = False,
        prefer_aligned: bool = False,
        require_same_shape: bool = False,
    ) -> list[Path]:
        if prefer_aligned:
            aligned = sorted((self._output_dir() / "aligned").glob("*_aligned.fits"))
            if aligned:
                return self._validate_paths(aligned, require_same_shape=require_same_shape)
        if prefer_solved:
            solved = sorted((self._output_dir() / "solved").glob("*.new"))
            if solved:
                return self._validate_paths(solved, require_same_shape=require_same_shape)
        paths = self.session.frame_paths()
        if not paths:
            self._error("No frames", "Import FITS images first.")
            return []
        return self._validate_paths(paths, require_same_shape=require_same_shape)

    def _validate_paths(self, paths: list[Path], *, require_same_shape: bool) -> list[Path]:
        if not require_same_shape:
            return paths
        groups = self._shape_groups(paths)
        if len(groups) <= 1:
            return paths
        lines = [f"{shape[1]} x {shape[0]}: {len(shape_paths)} image(s)" for shape, shape_paths in groups.items()]
        self._error(
            "Image sizes do not match",
            "This step needs all selected images to have the same dimensions.\n\n"
            + "\n".join(lines)
            + "\n\nImport one matching image set, or plate solve/calibrate matching frames separately.",
        )
        return []

    def _shape_groups(self, paths: list[Path]) -> dict[tuple[int, int], list[Path]]:
        groups: dict[tuple[int, int], list[Path]] = {}
        for path in paths:
            try:
                shape = load_image(path).data.shape
            except Exception:
                shape = (-1, -1)
            groups.setdefault(shape, []).append(path)
        return groups

    def _output_dir(self) -> Path:
        if self.output_edit.text().strip():
            self.session.output_dir = self.output_edit.text().strip()
        elif self.session.input_dir:
            self.session.output_dir = str(Path(self.session.input_dir) / "asteroidfinder_output")
            self.output_edit.setText(self.session.output_dir)
        else:
            self.session.output_dir = str(Path.cwd() / "asteroidfinder_output")
            self.output_edit.setText(self.session.output_dir)
        output = Path(self.session.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _sync_settings(self) -> None:
        self.session.settings.index_dir = self.index_edit.text().strip()
        self.session.settings.scale_low = self.scale_low.value() if self.scale_low.value() > 0 else None
        self.session.settings.scale_high = self.scale_high.value() if self.scale_high.value() > 0 else None
        self.session.settings.hot_sigma = self.hot_sigma.value()
        self.session.settings.detect_sigma = self.detect_sigma.value()
        self.session.settings.min_detections = self.min_detections.value()
        self.session.settings.observatory_code = self.observatory_code.text().strip() or "500"

    def _frame_info(self, path: Path) -> FrameInfo:
        try:
            image = load_image(path)
            header = image.header
            has_wcs = False
            if header is not None:
                try:
                    has_wcs = WCS(header).has_celestial
                except Exception:
                    has_wcs = False
            return FrameInfo(
                path=str(path),
                width=int(image.data.shape[1]),
                height=int(image.data.shape[0]),
                date_obs=None if header is None else str(header.get("DATE-OBS", "")) or None,
                filter_name=None if header is None else str(header.get("FILTER", "")) or None,
                has_wcs=has_wcs,
            )
        except Exception as exc:
            self._log(f"Could not inspect {path}: {exc}")
            return FrameInfo(path=str(path))

    def _log(self, text: str) -> None:
        self.log.append(text)

    def _error(self, title: str, detail: str) -> None:
        QMessageBox.critical(self, title, detail)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About AsteroidFinder",
            "AsteroidFinder desktop preview\n\nReal FITS loading, plate solving, alignment, tracking, and MPC export.",
        )

    def _update_plate_info(self, path: Path) -> None:
        try:
            image = load_image(path)
            header = image.header
            if header is None:
                self._set_unsolved_plate_info(path, "No FITS header")
                return
            wcs = WCS(header)
            if not wcs.has_celestial:
                self._set_unsolved_plate_info(path, "No WCS")
                return
            height, width = image.data.shape
            ra, dec = wcs.pixel_to_world_values(width / 2, height / 2)
            scale = _pixel_scale_arcsec(wcs)
            self.plate_status.setText("Solved WCS")
            self.plate_center.setText(f"RA {float(ra):.6f}, Dec {float(dec):.6f}")
            self.plate_scale.setText("unknown" if scale is None else f"{scale:.3f} arcsec/px")
            self.plate_path.setText(path.name)
        except Exception as exc:
            self._set_unsolved_plate_info(path, f"Read failed: {exc}")

    def _set_unsolved_plate_info(self, path: Path, status: str) -> None:
        self.plate_status.setText(status)
        self.plate_center.setText("")
        self.plate_scale.setText("")
        self.plate_path.setText(path.name)


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
            padding: 7px 10px;
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


def _pixel_scale_arcsec(wcs: WCS) -> float | None:
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
        return float(sum(abs(value) for value in scales) / len(scales))
    except Exception:
        return None


class DiagnosticWindow(QDialog):
    def __init__(self, paths: list[Path], title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.index = 0
        self.zoom = 100
        self.setWindowTitle(title)
        self.resize(980, 760)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._step(-1))
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self._step(1))
        self.caption = QLabel()
        self.caption.setObjectName("MutedText")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(35, 180)
        self.zoom_slider.setValue(self.zoom)
        self.zoom_slider.valueChanged.connect(self._set_zoom)
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(self.caption, 1)
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom_slider)
        layout.addLayout(controls)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, 1)
        self._render()

    def _step(self, delta: int) -> None:
        if not self.paths:
            return
        self.index = (self.index + delta) % len(self.paths)
        self._render()

    def _set_zoom(self, value: int) -> None:
        self.zoom = value
        self._render()

    def _render(self) -> None:
        if not self.paths:
            self.image_label.setText("No graphs")
            return
        path = self.paths[self.index]
        pixmap = QPixmap(str(path))
        self.caption.setText(f"{self.index + 1} / {len(self.paths)}  {path.name}")
        if pixmap.isNull():
            self.image_label.setText(f"Could not load {path.name}")
            return
        width = max(240, int(pixmap.width() * self.zoom / 100))
        self.image_label.setPixmap(pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation))


class ReportWindow(QDialog):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("AsteroidFinder Report")
        self.resize(1120, 780)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        open_browser = QPushButton("Open In Browser")
        open_browser.clicked.connect(lambda: webbrowser.open(self.path.resolve().as_uri()))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._load_report)
        controls.addWidget(QLabel(str(path)))
        controls.addStretch(1)
        controls.addWidget(refresh)
        controls.addWidget(open_browser)
        layout.addLayout(controls)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._load_report()

    def _load_report(self) -> None:
        self.tabs.clear()
        out_dir = self.path.parent
        self.tabs.addTab(self._summary_tab(out_dir), "Summary")
        for title, filename in [
            ("Tracks", "tracks.csv"),
            ("Known Objects", "known_objects.csv"),
            ("Alignment", "alignment_report.csv"),
            ("Hot Pixels", "hot_pixel_report.csv"),
            ("Forced Photometry", "known_object_forced_photometry.csv"),
        ]:
            rows = _read_csv_file(out_dir / filename)
            if rows:
                self.tabs.addTab(_table_widget(rows), title)
        self.tabs.addTab(self._files_tab(out_dir), "Files")

    def _summary_tab(self, out_dir: Path) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        metrics = [
            ("Output folder", str(out_dir)),
            ("Tracks", str(len(_read_csv_file(out_dir / "tracks.csv")))),
            ("Known objects", str(len(_read_csv_file(out_dir / "known_objects.csv")))),
            ("Alignment rows", str(len(_read_csv_file(out_dir / "alignment_report.csv")))),
            ("Movement graphs", str(len(list((out_dir / "diagnostics").glob("*.png"))) if (out_dir / "diagnostics").exists() else 0)),
        ]
        for label, value in metrics:
            row = QLabel(f"<b>{label}</b>: {value}")
            row.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(row)
        layout.addStretch(1)
        return widget

    def _files_tab(self, out_dir: Path) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        files = QListWidget()
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                files.addItem(str(path.relative_to(out_dir)))
        layout.addWidget(files, 1)
        return widget


def _read_csv_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _table_widget(rows: list[dict[str, str]]) -> QTableWidget:
    headers = list(rows[0].keys()) if rows else []
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for row_index, row in enumerate(rows):
        for col_index, header in enumerate(headers):
            table.setItem(row_index, col_index, QTableWidgetItem(row.get(header, "")))
    return table
