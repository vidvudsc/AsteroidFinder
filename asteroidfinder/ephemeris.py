from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import csv

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.wcs import WCS

from .io import load_image


@dataclass(frozen=True)
class MpcEphemerisPrediction:
    target: str
    frame: Path
    date_obs: str
    ra_deg: float
    dec_deg: float
    x: float
    y: float
    v_mag: float | None
    ra_rate_arcsec_per_hour: float | None
    dec_rate_arcsec_per_hour: float | None


def query_mpc_ephemeris_for_frames(
    target: str,
    paths: Sequence[str | Path],
    *,
    location: str = "500",
    observer_location: EarthLocation | None = None,
    only_inside: bool = True,
) -> list[MpcEphemerisPrediction]:
    """Query MPC target ephemerides at each frame time and project them into image pixels."""

    try:
        from astroquery.mpc import MPC
    except ImportError as exc:
        raise RuntimeError("astroquery is required for MPC ephemeris queries") from exc

    predictions: list[MpcEphemerisPrediction] = []
    location_arg: str | EarthLocation = observer_location if observer_location is not None else location
    for path in paths:
        image = load_image(path)
        if image.header is None:
            raise ValueError(f"No FITS header available for MPC ephemeris query: {path}")
        observation_time = _observation_time(image.header)
        if observation_time is None:
            raise ValueError(f"Observation time is required for MPC ephemeris query: {path}")
        wcs = WCS(image.header)
        if not wcs.has_celestial:
            raise ValueError(f"Solved celestial WCS is required for MPC ephemeris query: {path}")

        table = MPC.get_ephemeris(
            target,
            location=location_arg,
            start=observation_time,
            step=1 * u.minute,
            number=1,
            proper_motion="sky",
            proper_motion_unit="arcsec/h",
        )
        if len(table) == 0:
            continue
        row = table[0]
        ra_deg = _quantity_float(row["RA"], u.deg)
        dec_deg = _quantity_float(row["Dec"], u.deg)
        x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
        height, width = image.data.shape
        if only_inside and not (0 <= x < width and 0 <= y < height):
            continue
        predictions.append(
            MpcEphemerisPrediction(
                target=target,
                frame=Path(path),
                date_obs=observation_time.isot,
                ra_deg=ra_deg % 360.0,
                dec_deg=dec_deg,
                x=float(x),
                y=float(y),
                v_mag=_optional_quantity_float(row["V"]),
                ra_rate_arcsec_per_hour=_optional_quantity_float(row["dRA cos(Dec)"], u.arcsec / u.hour),
                dec_rate_arcsec_per_hour=_optional_quantity_float(row["dDec"], u.arcsec / u.hour),
            )
        )
    return predictions


def write_mpc_ephemeris_csv(predictions: Sequence[MpcEphemerisPrediction], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target",
                "frame",
                "date_obs",
                "ra_deg",
                "dec_deg",
                "x",
                "y",
                "v_mag",
                "ra_rate_arcsec_per_hour",
                "dec_rate_arcsec_per_hour",
            ]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.target,
                    prediction.frame,
                    prediction.date_obs,
                    f"{prediction.ra_deg:.8f}",
                    f"{prediction.dec_deg:.8f}",
                    f"{prediction.x:.3f}",
                    f"{prediction.y:.3f}",
                    _fmt(prediction.v_mag, 2),
                    _fmt(prediction.ra_rate_arcsec_per_hour, 4),
                    _fmt(prediction.dec_rate_arcsec_per_hour, 4),
                ]
            )
    return output


def _observation_time(header: object) -> Time | None:
    for key in ("DATE-OBS", "DATEOBS", "DATE"):
        value = header.get(key)
        if value:
            try:
                return Time(value)
            except Exception:
                pass
    for key in ("OBSJD", "JD", "JULDATE"):
        value = header.get(key)
        if value not in {None, ""}:
            try:
                return Time(float(value), format="jd")
            except Exception:
                pass
    for key in ("MJD-OBS", "MJD"):
        value = header.get(key)
        if value not in {None, ""}:
            try:
                return Time(float(value), format="mjd")
            except Exception:
                pass
    return None


def _quantity_float(value: object, unit: u.Unit) -> float:
    if hasattr(value, "to_value"):
        return float(value.to_value(unit))
    return float(value)


def _optional_quantity_float(value: object, unit: u.Unit | None = None) -> float | None:
    if np.ma.is_masked(value):
        return None
    try:
        raw = value.value if hasattr(value, "value") else value
        if np.ma.is_masked(raw):
            return None
        if unit is not None and hasattr(value, "to_value"):
            return float(value.to_value(unit))
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"
