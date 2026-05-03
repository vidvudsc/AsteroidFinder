from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np
from astropy.wcs import WCS

from .io import load_image


ASTROMETRY_INDEX_URL = "https://data.astrometry.net/4200/"
_SPLIT_INDEX_COUNTS = {
    "4205": 12,
    "4206": 12,
    "4207": 12,
}


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def run_doctor(
    *,
    index_dir: str | Path | None = None,
    sample_image: str | Path | None = None,
    scale_low: float | None = None,
    scale_high: float | None = None,
) -> list[DoctorCheck]:
    """Check local environment pieces needed for the asteroid pipeline."""

    checks: list[DoctorCheck] = []
    solve_field = shutil.which("solve-field")
    checks.append(
        DoctorCheck(
            "astrometry.net solve-field",
            solve_field is not None,
            solve_field or "not found on PATH",
        )
    )

    index_paths = _find_index_files(index_dir)
    index_detail = f"{len(index_paths)} index file(s)"
    if index_dir is not None:
        index_detail += f" in {Path(index_dir).expanduser()}"
    checks.append(DoctorCheck("astrometry index files", bool(index_paths), index_detail))

    if sample_image is not None:
        image = load_image(sample_image)
        checks.append(DoctorCheck("sample image load", True, f"{image.path} shape={image.data.shape}"))
        if image.header is not None:
            try:
                wcs = WCS(image.header)
                checks.append(
                    DoctorCheck(
                        "sample embedded WCS",
                        True,
                        "celestial WCS present" if wcs.has_celestial else "no celestial WCS; blind solve required",
                    )
                )
            except Exception as exc:
                checks.append(DoctorCheck("sample embedded WCS", False, f"{type(exc).__name__}: {exc}"))
        if scale_low is not None and scale_high is not None:
            width = max(image.data.shape)
            checks.append(
                DoctorCheck(
                    "scale estimate",
                    True,
                    f"{scale_low:.3f}-{scale_high:.3f} arcsec/pixel; field width about {width * np.mean([scale_low, scale_high]) / 3600:.2f} deg",
                )
            )

    return checks


def recommend_index_series(
    *,
    image_width_px: int,
    scale_low: float,
    scale_high: float,
) -> list[str]:
    """Return a conservative 4200-series recommendation for a field size."""

    field_low_deg = image_width_px * scale_low / 3600.0
    field_high_deg = image_width_px * scale_high / 3600.0
    field_mid_deg = 0.5 * (field_low_deg + field_high_deg)
    if field_mid_deg >= 1.0:
        return ["4210", "4211", "4212"]
    if field_mid_deg >= 0.4:
        return ["4206", "4207", "4208"]
    if field_mid_deg >= 0.15:
        return ["4204", "4205", "4206"]
    return ["4202", "4203", "4204"]


def install_astrometry_indexes(series: list[str], index_dir: str | Path) -> list[Path]:
    """Download 4200-series index files, including split healpix series."""

    out_dir = Path(index_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in series:
        for filename in _index_filenames(item):
            dest = out_dir / filename
            if dest.exists():
                written.append(dest)
                continue
            url = f"{ASTROMETRY_INDEX_URL}{filename}"
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)
            written.append(dest)
    return written


def _index_filenames(series: str) -> list[str]:
    count = _SPLIT_INDEX_COUNTS.get(series)
    if count is None:
        return [f"index-{series}.fits"]
    return [f"index-{series}-{index:02d}.fits" for index in range(count)]


def _find_index_files(index_dir: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if index_dir is not None:
        candidates.append(Path(index_dir).expanduser())
    candidates.extend(
        [
            Path.home() / "astrometry-indexes" / "4200",
            Path("/opt/homebrew/Cellar/astrometry-net/0.97/data"),
            Path("/usr/local/astrometry/data"),
        ]
    )
    files: list[Path] = []
    for directory in candidates:
        if directory.exists():
            files.extend(sorted(directory.glob("index-*.fits")))
            files.extend(sorted(directory.glob("index-*.fits.fz")))
    return sorted(set(files))
