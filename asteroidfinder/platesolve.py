from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import textwrap
import subprocess
import tempfile

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
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
    search_radius_deg: float | None = None,
    downsample: int | None = 2,
    max_sources: int | None = 200,
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
        search_radius_deg=search_radius_deg,
        downsample=downsample,
        max_sources=max_sources,
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
    search_radius_deg: float | None,
    downsample: int | None,
    max_sources: int | None,
) -> PlateSolution:
    solve_field = shutil.which("solve-field")
    if solve_field is None:
        raise RuntimeError(
            "No embedded WCS was found and astrometry.net 'solve-field' is not on PATH. "
            "Install astrometry.net plus matching index files for real blind plate solving."
        )

    out_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="asteroidfinder-solve-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(path)
    header = image.header
    working_input = out_dir / f"{path.stem}-solveinput.fits"
    save_fits(image.data, working_input, header)
    if scale_low is None or scale_high is None:
        inferred_scale = _scale_arcsec_per_pixel(header)
        if inferred_scale is not None:
            scale_low = inferred_scale * 0.85
            scale_high = inferred_scale * 1.15
            scale_units = "arcsecperpix"
    coord_hint = _coordinate_hint(header)
    if search_radius_deg is None and coord_hint is not None:
        search_radius_deg = _search_radius_for_image(image.data.shape, scale_high)

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
    if coord_hint is not None and search_radius_deg is not None:
        cmd.extend(["--ra", f"{coord_hint[0]:.8f}", "--dec", f"{coord_hint[1]:.8f}", "--radius", f"{search_radius_deg:.3f}"])
    if downsample is not None and downsample > 1:
        cmd.extend(["--downsample", str(downsample)])
    if max_sources is not None and max_sources > 0:
        cmd.extend(["--objs", str(max_sources), "--depth", f"1-{max_sources}"])
    cmd.extend(["--crpix-center", "--no-tweak"])
    cmd.append(str(working_input))
    try:
        subprocess.run(cmd, check=True, timeout=timeout + 30, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"astrometry.net solve-field timed out after {timeout + 30} seconds for {path}. "
            f"Command used: {' '.join(cmd)}"
        ) from exc
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


def _scale_arcsec_per_pixel(header: fits.Header | None) -> float | None:
    if header is None:
        return None
    for key in ("PIXSCALE", "SCALE", "SECPIX", "iTelescopePlateScaleH", "iTelescopePlateScaleV"):
        value = _optional_float(header.get(key))
        if value is not None and value > 0:
            return value
    pixel_um = _optional_float(header.get("XPIXSZ")) or _optional_float(header.get("YPIXSZ"))
    focal_mm = _optional_float(header.get("FOCALLEN"))
    binning = _optional_float(header.get("XBINNING")) or 1.0
    if pixel_um is None or focal_mm is None or focal_mm <= 0:
        return None
    return 206.265 * pixel_um * binning / focal_mm


def _coordinate_hint(header: fits.Header | None) -> tuple[float, float] | None:
    if header is None:
        return None
    ra_value = header.get("OBJCTRA") or header.get("RA") or header.get("TELRA")
    dec_value = header.get("OBJCTDEC") or header.get("DEC") or header.get("TELDEC")
    if not ra_value or not dec_value:
        return None
    try:
        coord = SkyCoord(str(ra_value).replace(" ", ":"), str(dec_value).replace(" ", ":"), unit=(u.hourangle, u.deg))
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception:
        try:
            coord = SkyCoord(float(ra_value) * u.deg, float(dec_value) * u.deg)
            return float(coord.ra.deg), float(coord.dec.deg)
        except Exception:
            return None


def _search_radius_for_image(shape: tuple[int, ...], scale_high_arcsec_per_pixel: float | None) -> float:
    if len(shape) < 2 or scale_high_arcsec_per_pixel is None:
        return 5.0
    height, width = shape[-2], shape[-1]
    diagonal_deg = float(np.hypot(width, height) * scale_high_arcsec_per_pixel / 3600.0)
    return max(1.0, min(8.0, diagonal_deg * 1.25))


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None
