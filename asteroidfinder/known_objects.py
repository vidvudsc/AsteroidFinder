from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .detection import Source
from .io import load_image
from .photometry import aperture_photometry


@dataclass(frozen=True)
class KnownObject:
    frame: Path
    date_obs: str
    number: str
    name: str
    object_type: str
    ra_deg: float
    dec_deg: float
    x: float
    y: float
    v_mag: float | None
    center_distance_arcsec: float | None
    ra_rate_arcsec_per_hour: float | None
    dec_rate_arcsec_per_hour: float | None


@dataclass(frozen=True)
class KnownObjectPhotometry:
    known_object: KnownObject
    net_flux: float
    snr: float
    instrumental_mag: float | None
    background: float


def query_known_objects_in_frame(
    path: str | Path,
    *,
    location: str = "500",
    only_inside: bool = True,
) -> list[KnownObject]:
    """Query IMCCE SkyBoT for known solar-system objects expected in a solved frame."""

    try:
        from astroquery.imcce import Skybot
    except ImportError as exc:
        raise RuntimeError("astroquery is required for known-object checks") from exc

    image = load_image(path)
    if image.header is None:
        raise ValueError(f"No FITS header available for known-object query: {path}")
    observation_time = _observation_time(image.header)
    if observation_time is None:
        raise ValueError(f"Observation time is required for known-object query: {path}")

    wcs = WCS(image.header)
    if not wcs.has_celestial:
        raise ValueError(f"Solved celestial WCS is required for known-object query: {path}")

    height, width = image.data.shape
    center, radius = _frame_center_radius(wcs, width, height)
    table = Skybot.cone_search(center, radius, observation_time, location=location)

    objects: list[KnownObject] = []
    for row in table:
        ra = _float(row["RA"])
        dec = _float(row["DEC"])
        x, y = wcs.world_to_pixel_values(ra, dec)
        if only_inside and not (0 <= x < width and 0 <= y < height):
            continue
        objects.append(
            KnownObject(
                frame=Path(path),
                date_obs=observation_time.isot,
                number=_str(row["Number"]),
                name=_str(row["Name"]),
                object_type=_str(row["Type"]),
                ra_deg=ra,
                dec_deg=dec,
                x=float(x),
                y=float(y),
                v_mag=_optional_float(row["V"]),
                center_distance_arcsec=_optional_float(row["centerdist"]),
                ra_rate_arcsec_per_hour=_optional_float(row["RA_rate"]),
                dec_rate_arcsec_per_hour=_optional_float(row["DEC_rate"]),
            )
        )
    return objects


def query_known_objects_for_frames(
    paths: Sequence[str | Path],
    *,
    location: str = "500",
    only_inside: bool = True,
) -> list[KnownObject]:
    objects: list[KnownObject] = []
    for path in paths:
        objects.extend(query_known_objects_in_frame(path, location=location, only_inside=only_inside))
    return objects


def query_known_objects_with_motion_cache(
    paths: Sequence[str | Path],
    *,
    location: str = "500",
    only_inside: bool = True,
    anchor_index: int | None = None,
) -> list[KnownObject]:
    """Query SkyBoT once, then propagate expected positions to all frames.

    SkyBoT calls are slow. For one short image sequence, a known object's
    SkyBoT RA/Dec rates are good enough to predict per-frame positions without
    another network request for every exposure.
    """

    frame_paths = [Path(path) for path in paths]
    if not frame_paths:
        return []
    anchor = min(max(anchor_index if anchor_index is not None else len(frame_paths) // 2, 0), len(frame_paths) - 1)
    anchor_objects = query_known_objects_in_frame(frame_paths[anchor], location=location, only_inside=only_inside)
    return predict_known_objects_for_frames(anchor_objects, frame_paths, only_inside=only_inside)


def predict_known_objects_for_frames(
    anchor_objects: Sequence[KnownObject],
    paths: Sequence[str | Path],
    *,
    only_inside: bool = True,
) -> list[KnownObject]:
    """Project known-object positions into each frame using WCS and rates."""

    predicted: list[KnownObject] = []
    for path in paths:
        image = load_image(path)
        if image.header is None:
            continue
        observation_time = _observation_time(image.header)
        if observation_time is None:
            continue
        wcs = WCS(image.header)
        if not wcs.has_celestial:
            continue
        height, width = image.data.shape
        for obj in anchor_objects:
            ra_deg, dec_deg = _propagate_radec(obj, observation_time)
            x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
            if only_inside and not (0 <= x < width and 0 <= y < height):
                continue
            predicted.append(
                replace(
                    obj,
                    frame=Path(path),
                    date_obs=observation_time.isot,
                    ra_deg=float(ra_deg),
                    dec_deg=float(dec_deg),
                    x=float(x),
                    y=float(y),
                )
            )
    return predicted


def write_known_objects_csv(objects: Sequence[KnownObject], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        handle.write(
            "frame,date_obs,number,name,type,ra_deg,dec_deg,x,y,v_mag,"
            "center_distance_arcsec,ra_rate_arcsec_per_hour,dec_rate_arcsec_per_hour\n"
        )
        for obj in objects:
            handle.write(
                ",".join(
                    [
                        str(obj.frame),
                        obj.date_obs,
                        obj.number,
                        obj.name,
                        obj.object_type,
                        f"{obj.ra_deg:.8f}",
                        f"{obj.dec_deg:.8f}",
                        f"{obj.x:.3f}",
                        f"{obj.y:.3f}",
                        _fmt(obj.v_mag, 2),
                        _fmt(obj.center_distance_arcsec, 3),
                        _fmt(obj.ra_rate_arcsec_per_hour, 4),
                        _fmt(obj.dec_rate_arcsec_per_hour, 4),
                    ]
                )
                + "\n"
            )
    return output


def forced_photometry_for_known_objects(
    objects: Sequence[KnownObject],
    *,
    aperture_radius: float = 4.0,
    annulus_inner: float = 7.0,
    annulus_outer: float = 12.0,
) -> list[KnownObjectPhotometry]:
    """Measure forced aperture photometry at predicted known-object pixels."""

    by_frame: dict[Path, list[KnownObject]] = {}
    for obj in objects:
        by_frame.setdefault(obj.frame, []).append(obj)

    measurements: list[KnownObjectPhotometry] = []
    for frame, frame_objects in by_frame.items():
        image = load_image(frame)
        height, width = image.data.shape
        for obj in frame_objects:
            if not (0 <= obj.x < width and 0 <= obj.y < height):
                continue
            source = Source(x=obj.x, y=obj.y, flux=0.0, a=0.0, b=0.0, theta=0.0, snr=0.0)
            phot = aperture_photometry(
                image.data,
                source,
                aperture_radius=aperture_radius,
                annulus_inner=annulus_inner,
                annulus_outer=annulus_outer,
            )
            measurements.append(
                KnownObjectPhotometry(
                    known_object=obj,
                    net_flux=phot.net_flux,
                    snr=phot.snr,
                    instrumental_mag=phot.instrumental_mag,
                    background=phot.background,
                )
            )
    return measurements


def write_known_object_photometry_csv(measurements: Sequence[KnownObjectPhotometry], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        handle.write(
            "frame,date_obs,number,name,type,x,y,ra_deg,dec_deg,expected_v_mag,"
            "net_flux,snr,instrumental_mag,background\n"
        )
        for item in measurements:
            obj = item.known_object
            handle.write(
                ",".join(
                    [
                        str(obj.frame),
                        obj.date_obs,
                        obj.number,
                        obj.name,
                        obj.object_type,
                        f"{obj.x:.3f}",
                        f"{obj.y:.3f}",
                        f"{obj.ra_deg:.8f}",
                        f"{obj.dec_deg:.8f}",
                        _fmt(obj.v_mag, 2),
                        f"{item.net_flux:.3f}",
                        f"{item.snr:.3f}",
                        _fmt(item.instrumental_mag, 5),
                        f"{item.background:.3f}",
                    ]
                )
                + "\n"
            )
    return output


def write_mpc_observations(objects: Sequence[KnownObject], path: str | Path) -> Path:
    """Write a simple MPC-style observation draft from predicted known objects.

    This is intentionally a draft/export aid, not a final MPC submission file:
    observatory code, band, magnitude calibration, and object confirmation need
    real observer-specific handling before submission.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        handle.write("# MPC-style draft generated from predicted known-object positions\n")
        handle.write("# name,date_utc,ra_hms,dec_dms,mag,observatory_code\n")
        for obj in objects:
            coord = SkyCoord(obj.ra_deg * u.deg, obj.dec_deg * u.deg, frame="icrs")
            ra = coord.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True)
            dec = coord.dec.to_string(unit=u.deg, sep=" ", precision=1, alwayssign=True, pad=True)
            handle.write(f"{obj.name},{obj.date_obs},{ra},{dec},{_fmt(obj.v_mag, 1)},XXX\n")
    return output


def _frame_center_radius(wcs: WCS, width: int, height: int) -> tuple[SkyCoord, u.Quantity]:
    center = wcs.pixel_to_world(width / 2.0, height / 2.0)
    corners = wcs.pixel_to_world([0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1])
    radius = max(center.separation(corners)).to(u.deg)
    return center, radius


def _observation_time(header: fits.Header) -> Time | None:
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


def _propagate_radec(obj: KnownObject, observation_time: Time) -> tuple[float, float]:
    try:
        anchor_time = Time(obj.date_obs)
    except Exception:
        return obj.ra_deg, obj.dec_deg
    dt_hours = float((observation_time - anchor_time).to_value(u.hour))
    dec_deg = obj.dec_deg
    ra_deg = obj.ra_deg
    if obj.dec_rate_arcsec_per_hour is not None:
        dec_deg += obj.dec_rate_arcsec_per_hour * dt_hours / 3600.0
    if obj.ra_rate_arcsec_per_hour is not None:
        cos_dec = max(1e-6, abs(np.cos(np.deg2rad(dec_deg))))
        ra_deg += obj.ra_rate_arcsec_per_hour * dt_hours / 3600.0 / cos_dec
    return ra_deg % 360.0, dec_deg


def _float(value: object) -> float:
    if hasattr(value, "to_value"):
        return float(value.to_value(u.deg))
    return float(value)


def _optional_float(value: object) -> float | None:
    if np.ma.is_masked(value):
        return None
    try:
        if hasattr(value, "value"):
            raw = value.value
        else:
            raw = value
        if np.ma.is_masked(raw):
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _str(value: object) -> str:
    if np.ma.is_masked(value):
        return ""
    return str(value).replace(",", " ")


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"
