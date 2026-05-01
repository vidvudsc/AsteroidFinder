from __future__ import annotations

from astropy.wcs import WCS

from .io import AstroImage


def image_wcs(image: AstroImage) -> WCS | None:
    """Return celestial WCS from an image header if present."""

    if image.header is None:
        return None
    wcs = WCS(image.header)
    return wcs if wcs.has_celestial else None


def pixel_to_sky(wcs: WCS | None, x: float, y: float) -> tuple[float | None, float | None]:
    """Convert pixel coordinates to RA/Dec degrees when WCS is available."""

    if wcs is None:
        return None, None
    ra, dec = wcs.pixel_to_world_values(x, y)
    return float(ra), float(dec)

