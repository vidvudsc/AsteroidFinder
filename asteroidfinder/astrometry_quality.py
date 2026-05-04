from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import csv
import math

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .detection import Source, detect_sources
from .io import load_image, save_fits


@dataclass(frozen=True)
class GaiaReferenceSource:
    source_id: str
    ra_deg: float
    dec_deg: float
    g_mag: float | None = None


@dataclass(frozen=True)
class AstrometryMatch:
    source: Source
    reference: GaiaReferenceSource
    image_ra_deg: float
    image_dec_deg: float
    ra_residual_arcsec: float
    dec_residual_arcsec: float
    total_residual_arcsec: float


@dataclass(frozen=True)
class AstrometryQualityResult:
    path: Path
    detected_sources: int
    catalog_sources: int
    matched_sources: int
    median_residual_arcsec: float | None
    rms_residual_arcsec: float | None
    p95_residual_arcsec: float | None
    max_residual_arcsec: float | None
    median_ra_residual_arcsec: float | None
    median_dec_residual_arcsec: float | None
    suggested_crval1_deg: float | None
    suggested_crval2_deg: float | None
    matches: tuple[AstrometryMatch, ...]


def measure_gaia_astrometry_quality(
    path: str | Path,
    *,
    catalog_sources: Sequence[GaiaReferenceSource] | None = None,
    sigma: float = 5.0,
    max_detected_sources: int = 500,
    mag_limit: float = 20.0,
    max_catalog_sources: int = 3000,
    match_radius_arcsec: float = 3.0,
) -> AstrometryQualityResult:
    """Measure WCS quality by matching detected image stars against Gaia DR3."""

    image = load_image(path)
    if image.header is None:
        raise ValueError(f"No FITS header available for astrometry QA: {path}")
    wcs = WCS(image.header)
    if not wcs.has_celestial:
        raise ValueError(f"Solved celestial WCS is required for astrometry QA: {path}")

    detected = detect_sources(image.data, sigma=sigma, max_sources=max_detected_sources)
    references = list(catalog_sources) if catalog_sources is not None else query_gaia_sources_for_frame(
        path,
        mag_limit=mag_limit,
        max_sources=max_catalog_sources,
    )
    matches = _match_detected_to_catalog(detected, references, wcs, match_radius_arcsec=match_radius_arcsec)
    summary = _summarize_matches(Path(path), len(detected), len(references), matches, image.header)
    return summary


def run_gaia_astrometry_quality(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    sigma: float = 5.0,
    match_radius_arcsec: float = 3.0,
    mag_limit: float = 20.0,
    max_detected_sources: int = 500,
    max_catalog_sources: int = 3000,
    write_corrected: bool = True,
    progress_callback: object | None = None,
) -> list[AstrometryQualityResult]:
    """Run Gaia WCS QA for each frame and write summary/match CSV files."""

    results: list[AstrometryQualityResult] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        if callable(progress_callback):
            progress_callback(index - 1, total, f"Gaia QA {Path(path).name}")
        results.append(
            measure_gaia_astrometry_quality(
                path,
                sigma=sigma,
                match_radius_arcsec=match_radius_arcsec,
                mag_limit=mag_limit,
                max_detected_sources=max_detected_sources,
                max_catalog_sources=max_catalog_sources,
            )
        )
        if callable(progress_callback):
            progress_callback(index, total, f"Measured {Path(path).name}")
    if write_corrected:
        write_wcs_offset_corrections(results, Path(output_dir))
    write_astrometry_quality_csv(results, Path(output_dir))
    return results


def write_wcs_offset_corrections(
    results: Sequence[AstrometryQualityResult],
    output_dir: str | Path,
) -> list[Path]:
    """Write CRVAL-offset corrected FITS copies for QA results with enough matches."""

    corrected: list[Path] = []
    for result in results:
        if result.suggested_crval1_deg is None or result.suggested_crval2_deg is None:
            continue
        out_path = _corrected_wcs_path(Path(output_dir), result.path)
        corrected.append(write_wcs_offset_corrected_fits(result, out_path))
    return corrected


def query_gaia_sources_for_frame(
    path: str | Path,
    *,
    mag_limit: float = 20.0,
    max_sources: int = 3000,
) -> list[GaiaReferenceSource]:
    """Query Gaia DR3 sources for the WCS footprint of a solved FITS frame."""

    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise RuntimeError("astroquery is required for Gaia astrometry QA") from exc

    image = load_image(path)
    if image.header is None:
        raise ValueError(f"No FITS header available for Gaia query: {path}")
    wcs = WCS(image.header)
    if not wcs.has_celestial:
        raise ValueError(f"Solved celestial WCS is required for Gaia query: {path}")
    center, radius = _frame_center_radius(wcs, image.data.shape[1], image.data.shape[0])
    radius_deg = float(radius.to_value(u.deg))
    adql = f"""
        SELECT TOP {int(max_sources)}
            source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {center.ra.deg:.10f}, {center.dec.deg:.10f}, {radius_deg:.10f})
        )
        AND phot_g_mean_mag <= {float(mag_limit):.3f}
        ORDER BY phot_g_mean_mag ASC
    """
    job = Gaia.launch_job_async(adql, dump_to_file=False)
    table = job.get_results()
    return [
        GaiaReferenceSource(
            source_id=str(row["source_id"]),
            ra_deg=float(row["ra"]),
            dec_deg=float(row["dec"]),
            g_mag=None if np.ma.is_masked(row["phot_g_mean_mag"]) else float(row["phot_g_mean_mag"]),
        )
        for row in table
    ]


def write_astrometry_quality_csv(results: Sequence[AstrometryQualityResult], output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "astrometry_qa.csv"
    matches_path = out_dir / "astrometry_matches.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "detected_sources",
                "catalog_sources",
                "matched_sources",
                "median_residual_arcsec",
                "rms_residual_arcsec",
                "p95_residual_arcsec",
                "max_residual_arcsec",
                "median_ra_residual_arcsec",
                "median_dec_residual_arcsec",
                "suggested_crval1_deg",
                "suggested_crval2_deg",
                "corrected_fits",
                "quality",
            ]
        )
        for result in results:
            corrected_path = ""
            if result.suggested_crval1_deg is not None and result.suggested_crval2_deg is not None:
                corrected_path = str(_corrected_wcs_path(out_dir, result.path))
            writer.writerow(
                [
                    result.path.name,
                    result.detected_sources,
                    result.catalog_sources,
                    result.matched_sources,
                    _fmt(result.median_residual_arcsec, 4),
                    _fmt(result.rms_residual_arcsec, 4),
                    _fmt(result.p95_residual_arcsec, 4),
                    _fmt(result.max_residual_arcsec, 4),
                    _fmt(result.median_ra_residual_arcsec, 4),
                    _fmt(result.median_dec_residual_arcsec, 4),
                    _fmt(result.suggested_crval1_deg, 8),
                    _fmt(result.suggested_crval2_deg, 8),
                    corrected_path,
                    _quality_label(result),
                ]
            )
    with matches_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "source_id",
                "x",
                "y",
                "image_ra_deg",
                "image_dec_deg",
                "gaia_ra_deg",
                "gaia_dec_deg",
                "g_mag",
                "o_minus_c_ra_arcsec",
                "o_minus_c_dec_arcsec",
                "residual_arcsec",
            ]
        )
        for result in results:
            for match in result.matches:
                writer.writerow(
                    [
                        result.path.name,
                        match.reference.source_id,
                        f"{match.source.x:.3f}",
                        f"{match.source.y:.3f}",
                        f"{match.image_ra_deg:.8f}",
                        f"{match.image_dec_deg:.8f}",
                        f"{match.reference.ra_deg:.8f}",
                        f"{match.reference.dec_deg:.8f}",
                        _fmt(match.reference.g_mag, 3),
                        f"{match.ra_residual_arcsec:.4f}",
                        f"{match.dec_residual_arcsec:.4f}",
                        f"{match.total_residual_arcsec:.4f}",
                    ]
                )
    return summary_path, matches_path


def write_wcs_offset_corrected_fits(
    result: AstrometryQualityResult,
    output_path: str | Path,
) -> Path:
    """Write a FITS copy with a constant CRVAL offset from Gaia median residuals."""

    if result.suggested_crval1_deg is None or result.suggested_crval2_deg is None:
        raise ValueError("Astrometry QA has no WCS correction suggestion")
    image = load_image(result.path)
    if image.header is None:
        raise ValueError(f"No FITS header available for WCS correction: {result.path}")
    header = image.header.copy()
    header["CRVAL1"] = result.suggested_crval1_deg
    header["CRVAL2"] = result.suggested_crval2_deg
    header["AFWCSQA"] = (_fmt(result.rms_residual_arcsec, 4), "AsteroidFinder Gaia RMS residual arcsec")
    return save_fits(image.data, output_path, header)


def _corrected_wcs_path(output_dir: Path, source_path: Path) -> Path:
    return output_dir / "wcs_corrected" / f"{source_path.stem}_gaia_wcs.fits"


def _match_detected_to_catalog(
    detected: Sequence[Source],
    references: Sequence[GaiaReferenceSource],
    wcs: WCS,
    *,
    match_radius_arcsec: float,
) -> tuple[AstrometryMatch, ...]:
    if not detected or not references:
        return ()
    image_ra, image_dec = wcs.pixel_to_world_values(
        [source.x for source in detected],
        [source.y for source in detected],
    )
    image_coords = SkyCoord(np.asarray(image_ra) * u.deg, np.asarray(image_dec) * u.deg)
    reference_coords = SkyCoord(
        [ref.ra_deg for ref in references] * u.deg,
        [ref.dec_deg for ref in references] * u.deg,
    )
    nearest_index, separation, _ = image_coords.match_to_catalog_sky(reference_coords)
    candidates: list[tuple[float, int, int, AstrometryMatch]] = []
    for source_index, (catalog_index, sep) in enumerate(zip(nearest_index, separation)):
        sep_arcsec = float(sep.to_value(u.arcsec))
        if sep_arcsec > match_radius_arcsec:
            continue
        source = detected[source_index]
        reference = references[int(catalog_index)]
        ra_residual, dec_residual, total = _signed_sky_residual_arcsec(
            reference.ra_deg,
            reference.dec_deg,
            float(image_ra[source_index]),
            float(image_dec[source_index]),
        )
        candidates.append(
            (
                total,
                source_index,
                int(catalog_index),
                AstrometryMatch(
                    source=source,
                    reference=reference,
                    image_ra_deg=float(image_ra[source_index]),
                    image_dec_deg=float(image_dec[source_index]),
                    ra_residual_arcsec=ra_residual,
                    dec_residual_arcsec=dec_residual,
                    total_residual_arcsec=total,
                ),
            )
        )
    candidates.sort(key=lambda item: item[0])
    used_sources: set[int] = set()
    used_references: set[int] = set()
    matches: list[AstrometryMatch] = []
    for _, source_index, catalog_index, match in candidates:
        if source_index in used_sources or catalog_index in used_references:
            continue
        used_sources.add(source_index)
        used_references.add(catalog_index)
        matches.append(match)
    return tuple(matches)


def _summarize_matches(
    path: Path,
    detected_sources: int,
    catalog_sources: int,
    matches: Sequence[AstrometryMatch],
    header: fits.Header,
) -> AstrometryQualityResult:
    if not matches:
        return AstrometryQualityResult(
            path=path,
            detected_sources=detected_sources,
            catalog_sources=catalog_sources,
            matched_sources=0,
            median_residual_arcsec=None,
            rms_residual_arcsec=None,
            p95_residual_arcsec=None,
            max_residual_arcsec=None,
            median_ra_residual_arcsec=None,
            median_dec_residual_arcsec=None,
            suggested_crval1_deg=None,
            suggested_crval2_deg=None,
            matches=(),
        )
    total = np.array([match.total_residual_arcsec for match in matches], dtype=float)
    ra = np.array([match.ra_residual_arcsec for match in matches], dtype=float)
    dec = np.array([match.dec_residual_arcsec for match in matches], dtype=float)
    median_ra = float(np.median(ra))
    median_dec = float(np.median(dec))
    suggested_crval1, suggested_crval2 = _suggested_crval(header, median_ra, median_dec)
    return AstrometryQualityResult(
        path=path,
        detected_sources=detected_sources,
        catalog_sources=catalog_sources,
        matched_sources=len(matches),
        median_residual_arcsec=float(np.median(total)),
        rms_residual_arcsec=float(np.sqrt(np.mean(total**2))),
        p95_residual_arcsec=float(np.percentile(total, 95)),
        max_residual_arcsec=float(np.max(total)),
        median_ra_residual_arcsec=median_ra,
        median_dec_residual_arcsec=median_dec,
        suggested_crval1_deg=suggested_crval1,
        suggested_crval2_deg=suggested_crval2,
        matches=tuple(matches),
    )


def _suggested_crval(header: fits.Header, median_ra_arcsec: float, median_dec_arcsec: float) -> tuple[float | None, float | None]:
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None, None
    dec = float(header["CRVAL2"])
    cos_dec = max(1e-6, abs(math.cos(math.radians(dec))))
    crval1 = float(header["CRVAL1"]) - median_ra_arcsec / 3600.0 / cos_dec
    crval2 = float(header["CRVAL2"]) - median_dec_arcsec / 3600.0
    return crval1 % 360.0, crval2


def _signed_sky_residual_arcsec(
    catalog_ra_deg: float,
    catalog_dec_deg: float,
    image_ra_deg: float,
    image_dec_deg: float,
) -> tuple[float, float, float]:
    dra = ((image_ra_deg - catalog_ra_deg + 180.0) % 360.0) - 180.0
    dec_mid = math.radians((catalog_dec_deg + image_dec_deg) / 2.0)
    ra_arcsec = dra * math.cos(dec_mid) * 3600.0
    dec_arcsec = (image_dec_deg - catalog_dec_deg) * 3600.0
    total = math.hypot(ra_arcsec, dec_arcsec)
    return float(ra_arcsec), float(dec_arcsec), float(total)


def _frame_center_radius(wcs: WCS, width: int, height: int) -> tuple[SkyCoord, u.Quantity]:
    center = wcs.pixel_to_world(width / 2.0, height / 2.0)
    corners = wcs.pixel_to_world([0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1])
    radius = max(center.separation(corners)).to(u.deg) * 1.05
    return center, radius


def _quality_label(result: AstrometryQualityResult) -> str:
    rms = result.rms_residual_arcsec
    if result.matched_sources < 8 or rms is None:
        return "unknown"
    if rms <= 0.5:
        return "excellent"
    if rms <= 1.0:
        return "good"
    if rms <= 2.0:
        return "warning"
    return "bad"


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"
