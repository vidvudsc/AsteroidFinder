from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from PySide6.QtWidgets import QApplication

from asteroidfinder.io import save_fits
from asteroidfinder.known_objects import KnownObject
from asteroidfinder_desktop.main_window import MainWindow, _frame_match_key, _import_status_text, _initial_progress_total, _known_objects_matching_frame, _progress_bar_text
from asteroidfinder_desktop.session import FrameInfo, SessionState, discover_fits_files, filter_image_files, load_session, save_session
from asteroidfinder_desktop.viewer import _display_luminance


def test_discover_fits_files_only_returns_supported_images(tmp_path: Path) -> None:
    save_fits(np.zeros((4, 5), dtype=np.float32), tmp_path / "b.fits")
    save_fits(np.zeros((4, 5), dtype=np.float32), tmp_path / "a.fit")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_fits_files(tmp_path)

    assert [path.name for path in paths] == ["a.fit", "b.fits"]


def test_filter_image_files_accepts_supported_image_paths(tmp_path: Path) -> None:
    raw = [str(tmp_path / "one.fit"), str(tmp_path / "two.fits"), str(tmp_path / "preview.jpg"), str(tmp_path / "notes.txt")]

    paths = filter_image_files(raw)

    assert [path.name for path in paths] == ["one.fit", "two.fits", "preview.jpg"]


def test_desktop_import_skips_empty_corrupt_fits(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    valid = tmp_path / "valid.fit"
    corrupt = tmp_path / "corrupt.fit"
    save_fits(np.zeros((4, 5), dtype=np.float32), valid)
    corrupt.write_bytes(b"")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.load_paths([corrupt, valid])

    assert [Path(frame.path).name for frame in window.session.frames] == ["valid.fit"]
    assert window.input_edit.text() == "1 images imported, 1 skipped"


def test_import_status_mentions_skipped_images() -> None:
    assert _import_status_text(4, 1) == "4 images imported, 1 skipped"
    assert _import_status_text(4, 0) == "4 images imported"


def test_session_round_trip(tmp_path: Path) -> None:
    state = SessionState(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        frames=[FrameInfo(path=str(tmp_path / "input" / "frame.fits"), width=10, height=20, has_wcs=True)],
    )

    session_path = save_session(state, tmp_path / "session.json")
    loaded = load_session(session_path)

    assert loaded.input_dir == state.input_dir
    assert loaded.output_dir == state.output_dir
    assert loaded.frames[0].name == "frame.fits"
    assert loaded.frames[0].has_wcs


def test_raw_bayer_preview_uses_same_size_luminance() -> None:
    data = np.array(
        [
            [0, 10, 20, 30],
            [40, 50, 60, 70],
            [80, 90, 100, 110],
            [120, 130, 140, 150],
        ],
        dtype=np.float32,
    )

    preview = _display_luminance(data, {"BAYERPAT": "VALID"})

    assert preview.shape == data.shape
    assert np.all(preview[:2, :2] == 25)
    assert np.all(preview[2:, 2:] == 125)


def test_mono_preview_keeps_original_pixels() -> None:
    data = np.arange(9, dtype=np.float32).reshape(3, 3)

    preview = _display_luminance(data, {"BAYERPAT": "INVALID"})

    assert np.array_equal(preview, data)


def test_known_object_overlay_matches_solved_variant_name() -> None:
    obj = _known_object(Path("frame-001-solveinput.new"))

    assert _frame_match_key(Path("frame-001.fits")) == _frame_match_key(obj.frame)
    assert _known_objects_matching_frame([obj], Path("frame-001.fits"), 0) == [obj]


def test_known_object_overlay_falls_back_to_frame_order() -> None:
    first = _known_object(Path("solved/a-solveinput.new"))
    second = _known_object(Path("solved/b-solveinput.new"))

    matches = _known_objects_matching_frame([first, second], Path("raw/not-the-same-name.fit"), 1)

    assert matches == [second]


def test_initial_progress_totals_for_slow_desktop_steps() -> None:
    paths = [Path("a.fit"), Path("b.fit"), Path("c.fit")]

    assert _initial_progress_total("plate solve", (paths,)) == 3
    assert _initial_progress_total("calibration", (paths,)) == 3
    assert _initial_progress_total("alignment", (paths,)) == 3
    assert _initial_progress_total("tracking", (paths,)) is None


def test_progress_bar_text_compacts_long_filenames() -> None:
    text = _progress_bar_text(
        "Scanning calibrated-T68-vidvuds1-ASTEROID_SEARCH_early-20260428-233730-Color-BIN1-W-120-003.fit"
    )

    assert text.endswith(" - %p%")
    assert len(text) < 72
    assert "..." in text


def _known_object(frame: Path) -> KnownObject:
    return KnownObject(
        frame=frame,
        date_obs="2026-01-01T00:00:00",
        number="",
        name="Test",
        object_type="MB",
        ra_deg=1.0,
        dec_deg=2.0,
        x=10.0,
        y=20.0,
        v_mag=None,
        center_distance_arcsec=None,
        ra_rate_arcsec_per_hour=None,
        dec_rate_arcsec_per_hour=None,
    )
