from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image


@dataclass(frozen=True)
class AstroImage:
    """A loaded astronomical image with optional FITS metadata."""

    data: np.ndarray
    path: Path
    header: fits.Header | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[-2], self.data.shape[-1]


def load_image(path: str | Path, *, channel: str = "luminance") -> AstroImage:
    """Load FITS/FIT or common raster images as a finite float32 2-D image."""

    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix in {".fit", ".fits", ".fts", ".new"}:
        return _load_fits(image_path, channel=channel)
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        return _load_raster(image_path)
    raise ValueError(f"Unsupported image format: {image_path}")


def save_fits(data: np.ndarray, path: str | Path, header: fits.Header | None = None) -> Path:
    """Save image data as a FITS file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(output, np.asarray(data, dtype=np.float32), header=header, overwrite=True)
    return output


def save_jpeg(data: np.ndarray, path: str | Path, *, percentile: tuple[float, float] = (0.5, 99.5)) -> Path:
    """Save a stretched preview JPEG from numeric image data."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stretched = stretch_to_uint8(data, percentile=percentile)
    Image.fromarray(stretched, mode="L").save(output, quality=95)
    return output


def stretch_to_uint8(data: np.ndarray, *, percentile: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, percentile)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _load_fits(path: Path, *, channel: str) -> AstroImage:
    with fits.open(path, memmap=False) as hdul:
        hdu = next((item for item in hdul if item.data is not None), None)
        if hdu is None:
            raise ValueError(f"No image data found in FITS file: {path}")
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    return AstroImage(_as_2d(data, channel=channel), path=path, header=header)


def _load_raster(path: Path) -> AstroImage:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    return AstroImage(_sanitize(arr), path=path, header=None)


def _as_2d(data: np.ndarray, *, channel: str) -> np.ndarray:
    if data.ndim == 2:
        return _sanitize(data)
    if data.ndim == 3:
        # FITS color frames commonly arrive as (channels, y, x). Some raster-like
        # FITS writers use (y, x, channels). Support both without guessing WCS.
        if data.shape[0] in {3, 4}:
            cube = data
        elif data.shape[-1] in {3, 4}:
            cube = np.moveaxis(data, -1, 0)
        else:
            raise ValueError(f"Unsupported FITS cube shape: {data.shape}")
        if channel == "luminance":
            return _sanitize(np.nanmean(cube[:3], axis=0))
        if channel in {"r", "red"}:
            return _sanitize(cube[0])
        if channel in {"g", "green"}:
            return _sanitize(cube[1])
        if channel in {"b", "blue"}:
            return _sanitize(cube[2])
        raise ValueError(f"Unsupported channel: {channel}")
    raise ValueError(f"Unsupported image dimensions: {data.shape}")


def _sanitize(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
