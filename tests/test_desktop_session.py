from __future__ import annotations

from pathlib import Path

import numpy as np

from asteroidfinder.io import save_fits
from asteroidfinder_desktop.session import FrameInfo, SessionState, discover_fits_files, load_session, save_session


def test_discover_fits_files_only_returns_supported_images(tmp_path: Path) -> None:
    save_fits(np.zeros((4, 5), dtype=np.float32), tmp_path / "b.fits")
    save_fits(np.zeros((4, 5), dtype=np.float32), tmp_path / "a.fit")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_fits_files(tmp_path)

    assert [path.name for path in paths] == ["a.fit", "b.fits"]


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
