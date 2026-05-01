from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import maximum_filter, median_filter

from .io import AstroImage, load_image, save_fits


@dataclass(frozen=True)
class CalibrationResult:
    image: AstroImage
    data: np.ndarray
    hot_pixel_mask: np.ndarray


def make_master_frame(paths: Sequence[str | Path], *, method: str = "median") -> np.ndarray:
    """Combine calibration frames into a master frame."""

    if not paths:
        raise ValueError("No calibration frames provided")
    images = [load_image(path).data for path in paths]
    _require_same_shape(images, context="master calibration frames")
    cube = np.stack(images).astype(np.float32)
    if method == "median":
        return np.nanmedian(cube, axis=0).astype(np.float32)
    if method == "mean":
        return np.nanmean(cube, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported master-frame method: {method}")


def calibrate_image(
    image: AstroImage | str | Path,
    *,
    master_bias: np.ndarray | str | Path | None = None,
    master_dark: np.ndarray | str | Path | None = None,
    master_flat: np.ndarray | str | Path | None = None,
    exposure_seconds: float | None = None,
    dark_exposure_seconds: float | None = None,
    hot_sigma: float = 8.0,
    hot_filter_size: int = 3,
    hot_pixel_mask: np.ndarray | None = None,
) -> CalibrationResult:
    """Apply bias/dark/flat correction and replace hot pixels."""

    astro = load_image(image) if isinstance(image, (str, Path)) else image
    data = astro.data.astype(np.float32, copy=True)

    bias = _optional_frame(master_bias)
    dark = _optional_frame(master_dark)
    flat = _optional_frame(master_flat)
    _check_shapes(data, bias=bias, dark=dark, flat=flat)

    if bias is not None:
        data -= bias
    if dark is not None:
        scale = 1.0
        if exposure_seconds is not None and dark_exposure_seconds not in {None, 0}:
            scale = exposure_seconds / float(dark_exposure_seconds)
        data -= dark * scale
    if flat is not None:
        normalized = flat / max(float(np.nanmedian(flat)), 1e-6)
        data /= np.where(normalized <= 1e-6, 1.0, normalized)

    if hot_pixel_mask is not None:
        _check_shapes(data, hot_pixel_mask=hot_pixel_mask)
        cleaned = apply_hot_pixel_mask(data, hot_pixel_mask, filter_size=hot_filter_size)
        mask = hot_pixel_mask.astype(bool)
    else:
        cleaned, mask = remove_hot_pixels(data, sigma=hot_sigma, filter_size=hot_filter_size)
    return CalibrationResult(image=astro, data=cleaned, hot_pixel_mask=mask)


def calibrate_images(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    **kwargs: object,
) -> list[CalibrationResult]:
    """Calibrate a sequence of images and optionally write corrected FITS files."""

    results = [calibrate_image(path, **kwargs) for path in paths]
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            save_fits(result.data, out_dir / f"{result.image.path.stem}_calibrated.fits", result.image.header)
    return results


def calibrate_images_with_persistent_hot_pixels(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    hot_sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    min_frames: int | None = None,
    **kwargs: object,
) -> list[CalibrationResult]:
    """Calibrate images using one conservative persistent hot-pixel map.

    This is safer for star fields than single-frame hot-pixel detection: a pixel
    must be both isolated and repeatedly hot at the same sensor coordinate.
    """

    grouped_paths = _group_paths_by_shape(paths)
    results = []
    for shape_paths in grouped_paths.values():
        mask = build_persistent_hot_pixel_mask(
            shape_paths,
            sigma=hot_sigma,
            neighbor_sigma=neighbor_sigma,
            min_center_neighbor_ratio=min_center_neighbor_ratio,
            min_frames=min_frames,
        )
        results.extend(calibrate_image(path, hot_pixel_mask=mask, hot_sigma=hot_sigma, **kwargs) for path in shape_paths)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            save_fits(result.data, out_dir / f"{result.image.path.stem}_calibrated.fits", result.image.header)
    return results


def build_persistent_hot_pixel_mask(
    paths: Sequence[str | Path],
    *,
    sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    min_frames: int | None = None,
    filter_size: int = 3,
) -> np.ndarray:
    """Build a fixed sensor-coordinate hot-pixel mask from a sequence."""

    if not paths:
        raise ValueError("No images provided")
    detections: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for path in paths:
        image = load_image(path).data
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(
                "Persistent hot-pixel masks require images with matching dimensions. "
                f"Expected {shape}, got {image.shape} for {path}."
            )
        detections.append(
            detect_isolated_hot_pixels(
                image,
                sigma=sigma,
                neighbor_sigma=neighbor_sigma,
                min_center_neighbor_ratio=min_center_neighbor_ratio,
                filter_size=filter_size,
            )
        )
    counts = np.sum(np.stack(detections), axis=0)
    required = min_frames if min_frames is not None else max(2, int(np.ceil(len(detections) * 0.8)))
    return counts >= required


def detect_isolated_hot_pixels(
    data: np.ndarray,
    *,
    sigma: float = 25.0,
    neighbor_sigma: float = 6.0,
    min_center_neighbor_ratio: float = 2.0,
    filter_size: int = 3,
) -> np.ndarray:
    """Detect isolated hot-pixel candidates without replacing them."""

    image = np.asarray(data, dtype=np.float32)
    local = median_filter(image, size=filter_size)
    residual = image - local
    robust_sigma = _robust_sigma(residual)
    if robust_sigma <= 0 or not np.isfinite(robust_sigma):
        return np.zeros(image.shape, dtype=bool)

    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    neighbor_max = maximum_filter(image, footprint=footprint, mode="nearest")
    neighbor_residual = neighbor_max - local
    bright_center = residual > sigma * robust_sigma
    quiet_neighbors = neighbor_residual < neighbor_sigma * robust_sigma
    isolated_contrast = image / np.maximum(neighbor_max, 1.0) >= min_center_neighbor_ratio
    return bright_center & quiet_neighbors & isolated_contrast


def apply_hot_pixel_mask(data: np.ndarray, mask: np.ndarray, *, filter_size: int = 3) -> np.ndarray:
    """Replace only the pixels in a supplied mask with local medians."""

    image = np.asarray(data, dtype=np.float32)
    local = median_filter(image, size=filter_size)
    cleaned = image.copy()
    cleaned[np.asarray(mask, dtype=bool)] = local[np.asarray(mask, dtype=bool)]
    return cleaned.astype(np.float32)


def remove_hot_pixels(
    data: np.ndarray,
    *,
    sigma: float = 8.0,
    filter_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated hot pixels with a local median estimate."""

    image = np.asarray(data, dtype=np.float32)
    local = median_filter(image, size=filter_size)
    residual = image - local
    robust_sigma = _robust_sigma(residual)
    if robust_sigma <= 0 or not np.isfinite(robust_sigma):
        return image.copy(), np.zeros(image.shape, dtype=bool)
    mask = residual > sigma * robust_sigma
    cleaned = image.copy()
    cleaned[mask] = local[mask]
    return cleaned.astype(np.float32), mask


def _robust_sigma(data: np.ndarray) -> float:
    mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
    return 1.4826 * mad if mad > 0 else float(np.nanstd(data))


def _optional_frame(frame: np.ndarray | str | Path | None) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, (str, Path)):
        return load_image(frame).data.astype(np.float32)
    return np.asarray(frame, dtype=np.float32)


def _check_shapes(data: np.ndarray, **frames: np.ndarray | None) -> None:
    for name, frame in frames.items():
        if frame is not None and frame.shape != data.shape:
            raise ValueError(f"{name} shape {frame.shape} does not match image shape {data.shape}")


def _require_same_shape(images: Sequence[np.ndarray], *, context: str) -> None:
    if not images:
        return
    shape = images[0].shape
    for image in images[1:]:
        if image.shape != shape:
            raise ValueError(f"All {context} must have the same shape. Expected {shape}, got {image.shape}.")


def _group_paths_by_shape(paths: Sequence[str | Path]) -> dict[tuple[int, int], list[str | Path]]:
    groups: dict[tuple[int, int], list[str | Path]] = {}
    for path in paths:
        shape = load_image(path).data.shape
        groups.setdefault(shape, []).append(path)
    return groups
