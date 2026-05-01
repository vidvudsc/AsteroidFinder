from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Source


@dataclass(frozen=True)
class Photometry:
    x: float
    y: float
    flux: float
    background: float
    net_flux: float
    snr: float
    instrumental_mag: float | None


def aperture_photometry(
    data: np.ndarray,
    source: Source,
    *,
    aperture_radius: float = 4.0,
    annulus_inner: float = 7.0,
    annulus_outer: float = 12.0,
) -> Photometry:
    """Measure simple circular-aperture photometry with annulus background."""

    image = np.asarray(data, dtype=np.float32)
    padding = int(np.ceil(annulus_outer)) + 1
    x0 = max(0, int(np.floor(source.x)) - padding)
    x1 = min(image.shape[1], int(np.floor(source.x)) + padding + 1)
    y0 = max(0, int(np.floor(source.y)) - padding)
    y1 = min(image.shape[0], int(np.floor(source.y)) + padding + 1)
    cutout = image[y0:y1, x0:x1]
    if cutout.size == 0:
        return Photometry(source.x, source.y, 0.0, 0.0, 0.0, 0.0, None)
    yy, xx = np.indices(cutout.shape)
    radius = np.hypot((xx + x0) - source.x, (yy + y0) - source.y)
    aperture = radius <= aperture_radius
    annulus = (radius >= annulus_inner) & (radius <= annulus_outer)
    if not np.any(aperture):
        return Photometry(source.x, source.y, 0.0, 0.0, 0.0, 0.0, None)

    background = float(np.nanmedian(cutout[annulus])) if np.any(annulus) else 0.0
    flux = float(np.nansum(cutout[aperture]))
    net_flux = float(flux - background * np.count_nonzero(aperture))
    noise = _estimate_noise(cutout[annulus], background)
    snr = float(net_flux / max(noise * np.sqrt(np.count_nonzero(aperture)), 1e-6))
    mag = float(-2.5 * np.log10(net_flux)) if net_flux > 0 else None
    return Photometry(source.x, source.y, flux, background, net_flux, snr, mag)


def _estimate_noise(samples: np.ndarray, background: float) -> float:
    if samples.size == 0:
        return 1.0
    residual = samples.astype(np.float32) - background
    mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
    noise = 1.4826 * mad if mad > 0 else float(np.nanstd(residual))
    return noise if np.isfinite(noise) and noise > 0 else 1.0
