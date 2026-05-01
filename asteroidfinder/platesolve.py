from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import textwrap
import subprocess
import tempfile

from astropy.io import fits
from astropy.wcs import WCS

from .io import load_image, save_fits


@dataclass(frozen=True)
class PlateSolution:
    path: Path
    wcs: WCS
    solved_fits: Path | None = None
    method: str = "embedded-wcs"


def solve_image(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    force_astrometry: bool = False,
    overwrite: bool = True,
    timeout: int = 180,
    scale_low: float | None = None,
    scale_high: float | None = None,
    scale_units: str = "arcsecperpix",
) -> PlateSolution:
    """Return a real WCS solution from embedded headers or astrometry.net."""

    image = load_image(path)
    if image.header is not None and not force_astrometry:
        wcs = WCS(image.header)
        if wcs.has_celestial:
            return PlateSolution(path=image.path, wcs=wcs, solved_fits=image.path, method="embedded-wcs")
    return _solve_with_astrometry_net(
        image.path,
        output_dir=output_dir,
        index_dir=index_dir,
        overwrite=overwrite,
        timeout=timeout,
        scale_low=scale_low,
        scale_high=scale_high,
        scale_units=scale_units,
    )


def _solve_with_astrometry_net(
    path: Path,
    *,
    output_dir: str | Path | None,
    index_dir: str | Path | None,
    overwrite: bool,
    timeout: int,
    scale_low: float | None,
    scale_high: float | None,
    scale_units: str,
) -> PlateSolution:
    solve_field = shutil.which("solve-field")
    if solve_field is None:
        raise RuntimeError(
            "No embedded WCS was found and astrometry.net 'solve-field' is not on PATH. "
            "Install astrometry.net plus matching index files for real blind plate solving."
        )

    out_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="asteroidfinder-solve-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    working_input = path
    if path.suffix.lower() not in {".fit", ".fits", ".fts"}:
        working_input = out_dir / f"{path.stem}.fits"
        save_fits(load_image(path).data, working_input)

    cmd = [
        solve_field,
        "--dir",
        str(out_dir),
        "--no-plots",
        "--fits-image",
        "--cpulimit",
        str(timeout),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if index_dir is not None:
        cmd.extend(["--index-dir", str(index_dir)])
    if scale_low is not None and scale_high is not None:
        cmd.extend(["--scale-low", str(scale_low), "--scale-high", str(scale_high), "--scale-units", scale_units])
    cmd.append(str(working_input))
    try:
        subprocess.run(cmd, check=True, timeout=timeout + 30, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        summary = textwrap.shorten(details.replace("\n", " "), width=2000, placeholder=" ...")
        raise RuntimeError(f"astrometry.net solve-field failed with exit code {exc.returncode}: {summary}") from exc

    solved_path = out_dir / f"{working_input.stem}.new"
    if not solved_path.exists():
        raise RuntimeError(f"astrometry.net completed without producing a solved FITS file: {solved_path}")
    header = fits.getheader(solved_path)
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise RuntimeError(f"astrometry.net output has no celestial WCS: {solved_path}")
    return PlateSolution(path=path, wcs=wcs, solved_fits=solved_path, method="astrometry.net")
