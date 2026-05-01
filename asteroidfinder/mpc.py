from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .alignment import AlignedFrame
from .tracking import Track
from .wcs import image_wcs, pixel_to_sky


@dataclass(frozen=True)
class DetectedTrackObservation:
    track_id: int
    frame_index: int
    frame: Path
    date_obs: str
    ra_deg: float
    dec_deg: float
    x: float
    y: float
    original_x: float
    original_y: float
    magnitude: float | None
    magnitude_kind: str
    band: str
    observatory_code: str
    net_flux: float | None
    snr: float | None


def measured_observations_from_tracks(
    tracks: Sequence[Track],
    aligned_frames: Sequence[AlignedFrame],
    *,
    observatory_code: str,
    object_prefix: str = "AF",
) -> list[DetectedTrackObservation]:
    """Convert detected moving-object tracks into measured astrometric observations."""

    observations: list[DetectedTrackObservation] = []
    for track_id, track in enumerate(tracks, start=1):
        for detection in track.detections:
            frame = aligned_frames[detection.frame_index]
            header = frame.image.header
            if header is None:
                raise ValueError(f"No FITS header for frame {frame.image.path}")
            date_obs = _date_obs(header)
            x_aligned = float(detection.source.x)
            y_aligned = float(detection.source.y)
            x_original, y_original = _original_pixel(frame, x_aligned, y_aligned)
            ra, dec = detection.ra_deg, detection.dec_deg
            if ra is None or dec is None:
                ra, dec = pixel_to_sky(image_wcs(frame.image), x_original, y_original)
            if ra is None or dec is None:
                raise ValueError(f"No WCS RA/Dec for track {track_id} frame {detection.frame_index}")
            magnitude, magnitude_kind = _magnitude(header, detection)
            observations.append(
                DetectedTrackObservation(
                    track_id=track_id,
                    frame_index=detection.frame_index,
                    frame=frame.image.path,
                    date_obs=date_obs,
                    ra_deg=float(ra),
                    dec_deg=float(dec),
                    x=x_aligned,
                    y=y_aligned,
                    original_x=x_original,
                    original_y=y_original,
                    magnitude=magnitude,
                    magnitude_kind=magnitude_kind,
                    band=_band(header),
                    observatory_code=observatory_code,
                    net_flux=None if detection.photometry is None else float(detection.photometry.net_flux),
                    snr=None if detection.photometry is None else float(detection.photometry.snr),
                )
            )
    return observations


def write_detected_track_observations_csv(
    observations: Sequence[DetectedTrackObservation],
    path: str | Path,
) -> Path:
    """Write a full audit CSV for measured track observations."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "track_id",
                "frame_index",
                "frame",
                "date_obs",
                "ra_deg",
                "dec_deg",
                "aligned_x",
                "aligned_y",
                "original_x",
                "original_y",
                "magnitude",
                "magnitude_kind",
                "band",
                "observatory_code",
                "net_flux",
                "snr",
            ]
        )
        for obs in observations:
            writer.writerow(
                [
                    obs.track_id,
                    obs.frame_index,
                    obs.frame,
                    obs.date_obs,
                    f"{obs.ra_deg:.8f}",
                    f"{obs.dec_deg:.8f}",
                    f"{obs.x:.3f}",
                    f"{obs.y:.3f}",
                    f"{obs.original_x:.3f}",
                    f"{obs.original_y:.3f}",
                    _fmt(obs.magnitude, 3),
                    obs.magnitude_kind,
                    obs.band,
                    obs.observatory_code,
                    _fmt(obs.net_flux, 3),
                    _fmt(obs.snr, 3),
                ]
            )
    return output


def write_detected_track_mpc(
    tracks: Sequence[Track],
    aligned_frames: Sequence[AlignedFrame],
    path: str | Path,
    *,
    observatory_code: str,
    object_prefix: str = "AF",
    csv_path: str | Path | None = None,
) -> Path:
    """Write measured detected-track observations as an MPC-style draft.

    This uses measured track centroids and frame WCS, not SkyBoT predicted
    positions. The text format is intentionally labeled as a draft because final
    MPC submission formatting, packed designations, uncertainty handling, and
    calibrated photometry are observer/workflow specific.
    """

    observations = measured_observations_from_tracks(
        tracks,
        aligned_frames,
        observatory_code=observatory_code,
        object_prefix=object_prefix,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if csv_path is not None:
        write_detected_track_observations_csv(observations, csv_path)

    with output.open("w", newline="") as handle:
        handle.write("# MPC-style draft generated from measured detected tracks\n")
        handle.write("# source=asteroidfinder measured centroids, not SkyBoT predictions\n")
        handle.write("# object_id,date_utc,ra_hms,dec_dms,mag,band,observatory_code,track_id,frame_index\n")
        for obs in observations:
            object_id = f"{object_prefix}{obs.track_id:04d}"
            coord = SkyCoord(obs.ra_deg * u.deg, obs.dec_deg * u.deg, frame="icrs")
            ra = coord.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True)
            dec = coord.dec.to_string(unit=u.deg, sep=" ", precision=1, alwayssign=True, pad=True)
            handle.write(
                ",".join(
                    [
                        object_id,
                        obs.date_obs,
                        ra,
                        dec,
                        _fmt(obs.magnitude, 1),
                        obs.band,
                        obs.observatory_code,
                        str(obs.track_id),
                        str(obs.frame_index),
                    ]
                )
                + "\n"
            )
    return output


def _original_pixel(frame: AlignedFrame, x: float, y: float) -> tuple[float, float]:
    if frame.transform is None:
        return x, y
    original = frame.transform.inverse(np.array([[x, y]], dtype=float))[0]
    return float(original[0]), float(original[1])


def _date_obs(header: object) -> str:
    if "DATE-OBS" in header:
        return str(header["DATE-OBS"])
    if "OBSJD" in header:
        return Time(float(header["OBSJD"]), format="jd", scale="utc").isot
    if "JD" in header:
        return Time(float(header["JD"]), format="jd", scale="utc").isot
    raise ValueError("DATE-OBS, OBSJD, or JD is required for MPC export")


def _band(header: object) -> str:
    value = str(header.get("FILTER", header.get("FILTERID", "C"))).strip()
    low = value.lower()
    if "ztf_r" in low or low in {"zr", "r", "2"}:
        return "r"
    if "ztf_g" in low or low in {"zg", "g", "1"}:
        return "g"
    if "ztf_i" in low or low in {"zi", "i", "3"}:
        return "i"
    if value:
        return value[:1]
    return "C"


def _magnitude(header: object, detection: object) -> tuple[float | None, str]:
    phot = detection.photometry
    if phot is None:
        return None, ""
    if phot.net_flux > 0 and "MAGZP" in header:
        return float(header["MAGZP"]) - 2.5 * math.log10(float(phot.net_flux)), "calibrated_magzp"
    return phot.instrumental_mag, "instrumental"


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"
